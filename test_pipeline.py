#!/usr/bin/env python3
"""
LocalLoom Test Pipeline

Processes a video file through the complete tutorial generation pipeline:
1. Extract frames using ffmpeg
2. Analyze frames with QwenVL vision model (batched)
3. Transcribe audio with Whisper ASR
4. Synthesize walkthrough from visual + audio
5. Generate Markdown tutorial

Usage:
    python test_pipeline.py <video_path>

Environment variables (from .env):
    QWENVL_BASE_URL - QwenVL Modal VM URL (or VITE_QWENVL_BASE_URL)
    WHISPER_ASR_BASE_URL - Whisper ASR Modal VM URL
    MODAL_API_TOKEN - Optional API token for Modal VMs
    VITE_OPENAI_API_KEY - Optional OpenAI key for synthesis

Output:
    - Tutorial saved to src/tutorial_<timestamp>.md
    - All model outputs logged to logs/ directory
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import aiohttp
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# ============================================================================
# Configuration
# ============================================================================

FRAMES_PER_BATCH = 4  # ~1085 tokens/frame, 4 fits within 8192 context
TARGET_FPS = 1.0      # Extract 1 frame per second
MAX_FRAMES = 60       # Cap at 60 frames for long videos
SIMILARITY_THRESHOLD = 0.85  # Deduplicate if >85% similar
MAX_CONCURRENT_BATCHES = 3   # Limit concurrent VLM requests to avoid timeouts
MAX_RETRIES = 2              # Retry failed batches up to 2 times

# Vision model for frame analysis
VISION_MODEL = "QuantTrio/Qwen3.5-4B-AWQ"
WHISPER_MODEL = "openai/whisper-large-v3-turbo"

# System prompts
VISION_SYSTEM_PROMPT = """You are a precise screen recording analyst for software tutorials. Your primary task is to read and transcribe ALL text visible on screen EXACTLY as written, character-by-character.

CRITICAL: When you see text on screen (variable names, URLs, file names, code, terminal commands), copy them EXACTLY. Do not paraphrase or "correct" technical terms. Examples:
- "VITE_SMOLVLM_BASE_URL" must be written exactly as "VITE_SMOLVLM_BASE_URL"
- "MODAL_API_TOKEN" must be written exactly as "MODAL_API_TOKEN"
- "openloom-whisper-asr" must be written exactly as "openloom-whisper-asr"
- ".env.example" must be written exactly as ".env.example"

Pay special attention to:
- Environment variable names (usually UPPER_CASE_WITH_UNDERSCORES)
- URLs and domain names
- File paths and filenames
- Terminal commands
- Code snippets"""

# User prompts (matching src/lib/prompts.ts)
VISION_FRAME_PROMPT = """Analyze these frames from a screen recording. For each frame, describe:
1. MOUSE POINTER: Where is the cursor? What is it hovering over or clicking?
2. UI ELEMENTS: What buttons, menus, dialogs, input fields, tabs, or panels are visible? Read any text labels EXACTLY as shown.
3. SCREEN CONTENT: What application is shown? What is the current view or page? Copy any URLs, filenames, variable names, or code EXACTLY as written.
4. STATE CHANGES: What changed between frames? Did a menu open, a button get clicked, text get typed, a page navigate?
5. HIGHLIGHTED/ACTIVE: What element has focus or is selected? Note any tooltips, dropdowns, or popups.

IMPORTANT: Copy all technical text (variable names, URLs, commands, code) character-by-character. Do not paraphrase."""

SYNTHESIS_SYSTEM_PROMPT = """You are a technical writer creating software tutorials. When the visual descriptions and audio transcript contain technical terms (variable names, URLs, commands), ALWAYS prefer the spelling from the VISUAL descriptions since those are read directly from the screen.

Common audio transcription errors to watch for and correct:
- "white" in audio usually means "VITE_" (environment variable prefix)
- "small vlm" or "small BLM" in audio usually means "SmolVLM"
- "modal" may be transcribed as "modale" or similar

Always use the EXACT spelling from visual descriptions for: variable names, URLs, file names, and commands."""

SYNTHESIS_PROMPT = """You are a technical writer creating a step-by-step tutorial from a screen recording. You are given visual descriptions of frames and an audio transcript.

Create a clear, numbered tutorial in Markdown. For each step:
- Write a short heading describing the action
- Write 1-2 sentences telling the user exactly what to do (click, type, select, etc)
- Mention the exact UI element names, menu paths, and button labels
- Note what the user should see after completing the step

Rules:
- Use the transcript to understand intent and context
- Use the visual descriptions for exact UI element names and locations
- Combine overlapping information — don't repeat the same action
- Use imperative voice: "Click the Save button" not "The user clicks Save"
- If the transcript mentions something not visible in frames, still include it
- Start with a title and one-line summary
- Output ONLY the Markdown, no preamble
"""


# ============================================================================
# Data Classes
# ============================================================================

@dataclass(frozen=True)
class TranscriptSegment:
    start: float
    end: float
    text: str
    confidence: float = 0.85


@dataclass(frozen=True)
class VisualEvent:
    start: float
    end: float
    description: str
    screenshot: str | None = None
    confidence: float = 0.75


@dataclass
class WalkthroughStep:
    index: int
    heading: str
    instruction: str
    timestamp_start: float
    timestamp_end: float
    screenshot: str | None
    evidence: str
    confidence: float


@dataclass
class Walkthrough:
    title: str
    summary: str
    steps: list[WalkthroughStep]
    warnings: list[str]


# ============================================================================
# Logging Setup
# ============================================================================

class PipelineLogger:
    """Handles all logging for the pipeline, including model output logging."""

    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Main pipeline log
        self.main_log = log_dir / f"pipeline_{self.timestamp}.log"

        # Model-specific logs
        self.smolvlm_log = log_dir / f"smolvlm_{self.timestamp}.json"
        self.whisper_log = log_dir / f"whisper_{self.timestamp}.json"
        self.synthesis_log = log_dir / f"synthesis_{self.timestamp}.json"

        # Setup main logger
        self.logger = logging.getLogger("LocalLoom")
        self.logger.setLevel(logging.DEBUG)

        # File handler
        file_handler = logging.FileHandler(self.main_log)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter("%(levelname)-8s | %(message)s")
        console_handler.setFormatter(console_formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        # Model output storage
        self.smolvlm_outputs: list[dict[str, Any]] = []
        self.whisper_outputs: list[dict[str, Any]] = []
        self.synthesis_outputs: list[dict[str, Any]] = []

    def info(self, msg: str) -> None:
        self.logger.info(msg)

    def debug(self, msg: str) -> None:
        self.logger.debug(msg)

    def warning(self, msg: str) -> None:
        self.logger.warning(msg)

    def error(self, msg: str) -> None:
        self.logger.error(msg)

    def log_smolvlm_request(self, batch_idx: int, total_batches: int,
                            frame_range: tuple[int, int], request: dict[str, Any]) -> None:
        """Log SmolVLM API request."""
        entry = {
            "type": "request",
            "batch_idx": batch_idx,
            "total_batches": total_batches,
            "frame_range": frame_range,
            "timestamp": datetime.now().isoformat(),
            "model": VISION_MODEL,
            "num_images": len([c for c in request.get("messages", [{}])[0].get("content", [])
                              if isinstance(c, dict) and c.get("type") == "image_url"]),
        }
        self.smolvlm_outputs.append(entry)
        self.debug(f"SmolVLM request batch {batch_idx + 1}/{total_batches}: frames {frame_range}")

    def log_smolvlm_response(self, batch_idx: int, response: dict[str, Any],
                             description: str) -> None:
        """Log SmolVLM API response."""
        entry = {
            "type": "response",
            "batch_idx": batch_idx,
            "timestamp": datetime.now().isoformat(),
            "raw_response": response,
            "extracted_description": description,
            "description_length": len(description),
        }
        self.smolvlm_outputs.append(entry)
        self.debug(f"SmolVLM response batch {batch_idx + 1}: {len(description)} chars")

    def log_whisper_request(self, filename: str, file_size_mb: float) -> None:
        """Log Whisper ASR request."""
        entry = {
            "type": "request",
            "timestamp": datetime.now().isoformat(),
            "model": WHISPER_MODEL,
            "filename": filename,
            "file_size_mb": round(file_size_mb, 2),
        }
        self.whisper_outputs.append(entry)
        self.debug(f"Whisper request: {filename} ({file_size_mb:.2f} MB)")

    def log_whisper_response(self, response: dict[str, Any],
                             segments: list[TranscriptSegment],
                             duration: float) -> None:
        """Log Whisper ASR response."""
        entry = {
            "type": "response",
            "timestamp": datetime.now().isoformat(),
            "raw_response": response,
            "num_segments": len(segments),
            "duration_seconds": duration,
            "segments": [
                {
                    "start": s.start,
                    "end": s.end,
                    "text": s.text,
                    "confidence": s.confidence
                }
                for s in segments
            ],
        }
        self.whisper_outputs.append(entry)
        self.debug(f"Whisper response: {len(segments)} segments, {duration:.1f}s duration")

    def log_synthesis_request(self, visual_descriptions: list[str],
                              transcript_text: str) -> None:
        """Log synthesis request."""
        entry = {
            "type": "request",
            "timestamp": datetime.now().isoformat(),
            "num_visual_descriptions": len(visual_descriptions),
            "visual_descriptions": visual_descriptions,
            "transcript_text": transcript_text,
            "transcript_length": len(transcript_text),
        }
        self.synthesis_outputs.append(entry)
        self.debug(f"Synthesis request: {len(visual_descriptions)} visual batches, "
                  f"{len(transcript_text)} chars transcript")

    def log_synthesis_response(self, markdown: str, method: str) -> None:
        """Log synthesis response."""
        entry = {
            "type": "response",
            "timestamp": datetime.now().isoformat(),
            "method": method,
            "markdown_length": len(markdown),
            "markdown": markdown,
        }
        self.synthesis_outputs.append(entry)
        self.debug(f"Synthesis response ({method}): {len(markdown)} chars")

    def save_all_logs(self) -> None:
        """Save all model outputs to JSON files."""
        with open(self.smolvlm_log, "w") as f:
            json.dump(self.smolvlm_outputs, f, indent=2, default=str)
        self.info(f"SmolVLM logs saved to: {self.smolvlm_log}")

        with open(self.whisper_log, "w") as f:
            json.dump(self.whisper_outputs, f, indent=2, default=str)
        self.info(f"Whisper logs saved to: {self.whisper_log}")

        with open(self.synthesis_log, "w") as f:
            json.dump(self.synthesis_outputs, f, indent=2, default=str)
        self.info(f"Synthesis logs saved to: {self.synthesis_log}")


# ============================================================================
# Frame Extraction
# ============================================================================

def extract_frames(
    video_path: Path,
    fps: float = TARGET_FPS,
    max_frames: int = MAX_FRAMES,
    logger: PipelineLogger | None = None,
) -> list[str]:
    """
    Extract frames from video at fixed FPS using ffmpeg.
    Returns list of base64-encoded JPEG strings.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        output_pattern = Path(tmpdir) / "frame_%04d.jpg"

        # Get video duration first
        probe_cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path)
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True)
        duration = float(result.stdout.strip()) if result.stdout.strip() else 10.0

        # Calculate expected frames and apply max limit
        expected_frames = int(duration * fps)
        actual_max = min(expected_frames, max_frames)

        if logger:
            logger.info(
                f"Extracting frames: duration={duration:.1f}s, fps={fps}, "
                f"expected={expected_frames}, max={max_frames}, extracting={actual_max}"
            )

        # Extract frames at fixed FPS
        extract_cmd = [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vf", f"fps={fps},scale=512:-1",
            "-frames:v", str(actual_max),
            "-q:v", "2",
            str(output_pattern)
        ]

        subprocess.run(extract_cmd, capture_output=True, check=True)

        # Read and encode frames
        frames: list[str] = []
        frame_files = sorted(Path(tmpdir).glob("frame_*.jpg"))

        for frame_file in frame_files[:actual_max]:
            with open(frame_file, "rb") as f:
                frame_data = f.read()
                frames.append(base64.b64encode(frame_data).decode("utf-8"))

        if logger:
            logger.info(f"Extracted {len(frames)} frames")

        return frames


# ============================================================================
# SmolVLM Analysis
# ============================================================================

async def analyze_frames_batched(
    session: aiohttp.ClientSession,
    base_url: str,
    frames: list[str],
    api_token: str | None,
    logger: PipelineLogger,
) -> list[str]:
    """
    Analyze frames in batches using SmolVLM.
    Returns one description string per batch.
    Uses semaphore to limit concurrent requests and avoid API overload.
    """
    batches: list[list[str]] = []
    for i in range(0, len(frames), FRAMES_PER_BATCH):
        batches.append(frames[i:i + FRAMES_PER_BATCH])

    logger.info(f"Processing {len(frames)} frames in {len(batches)} batches (max {MAX_CONCURRENT_BATCHES} concurrent)")

    # Use semaphore to limit concurrency
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_BATCHES)

    async def process_batch_with_semaphore(batch: list[str], idx: int) -> str:
        async with semaphore:
            return await analyze_frame_batch_with_retry(
                session, base_url, batch, idx, len(batches), len(frames), api_token, logger
            )

    tasks = [
        process_batch_with_semaphore(batch, idx)
        for idx, batch in enumerate(batches)
    ]

    return await asyncio.gather(*tasks)


async def analyze_frame_batch_with_retry(
    session: aiohttp.ClientSession,
    base_url: str,
    batch: list[str],
    batch_idx: int,
    total_batches: int,
    total_frames: int,
    api_token: str | None,
    logger: PipelineLogger,
) -> str:
    """Analyze a batch with retry logic for resilience."""
    last_error: Exception | None = None

    for attempt in range(MAX_RETRIES + 1):
        try:
            return await analyze_frame_batch(
                session, base_url, batch, batch_idx, total_batches, total_frames, api_token, logger
            )
        except (asyncio.TimeoutError, aiohttp.ClientError) as e:
            last_error = e
            error_type = type(e).__name__
            error_msg = str(e) or error_type
            if attempt < MAX_RETRIES:
                wait_time = (attempt + 1) * 5  # 5s, 10s backoff
                logger.warning(
                    f"SmolVLM batch {batch_idx + 1}/{total_batches} failed ({error_msg}), "
                    f"retry {attempt + 1}/{MAX_RETRIES} in {wait_time}s"
                )
                await asyncio.sleep(wait_time)
            else:
                logger.error(
                    f"SmolVLM batch {batch_idx + 1}/{total_batches} failed after {MAX_RETRIES + 1} attempts: {error_msg}"
                )

    # Return fallback description if all retries failed
    error_msg = str(last_error) if last_error else "Unknown error"
    return f"[Analysis failed: {error_msg}]"


async def analyze_frame_batch(
    session: aiohttp.ClientSession,
    base_url: str,
    batch: list[str],
    batch_idx: int,
    total_batches: int,
    total_frames: int,
    api_token: str | None,
    logger: PipelineLogger,
) -> str:
    """Analyze a single batch of frames with SmolVLM."""
    url = f"{base_url.rstrip('/')}/v1/chat/completions"

    # Build content with images
    content: list[dict[str, Any]] = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        }
        for b64 in batch
    ]

    # Add context about frame position
    frame_start = batch_idx * FRAMES_PER_BATCH + 1
    frame_end = frame_start + len(batch) - 1
    batch_context = (
        f"These are frames {frame_start}-{frame_end} of {total_frames} from a screen recording.\n\n"
        if total_batches > 1 else ""
    )

    content.append({"type": "text", "text": batch_context + VISION_FRAME_PROMPT})

    payload = {
        "model": VISION_MODEL,
        "messages": [
            {"role": "system", "content": VISION_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        "temperature": 1
    }

    headers = {"Content-Type": "application/json"}
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"

    logger.log_smolvlm_request(batch_idx, total_batches, (frame_start, frame_end), payload)
    logger.info(f"SmolVLM batch {batch_idx + 1}/{total_batches} → {url}")

    # Use aiohttp.ClientTimeout for proper timeout handling
    timeout = aiohttp.ClientTimeout(total=300)  # 5 minutes per request
    async with session.post(url, json=payload, headers=headers, timeout=timeout) as response:
        response.raise_for_status()
        result = await response.json()

    description = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    if not description:
        description = "No visual description returned."

    logger.log_smolvlm_response(batch_idx, result, description)
    logger.info(f"SmolVLM batch {batch_idx + 1}/{total_batches} ← {len(description)} chars")

    return description


# ============================================================================
# Deduplication
# ============================================================================

def deduplicate_descriptions(
    descriptions: list[str],
    threshold: float = SIMILARITY_THRESHOLD,
    logger: PipelineLogger | None = None,
) -> list[str]:
    """
    Deduplicate consecutive batch descriptions that are too similar.
    Uses Jaccard similarity to detect redundant frames (e.g., static screens).
    """
    if len(descriptions) <= 1:
        return descriptions

    result: list[str] = [descriptions[0]]

    for i in range(1, len(descriptions)):
        prev = descriptions[i - 1]
        curr = descriptions[i]

        similarity = calculate_similarity(prev, curr)

        if similarity < threshold:
            result.append(curr)
        elif logger:
            logger.debug(
                f"Dedup: skipping batch {i + 1} ({similarity * 100:.0f}% similar to previous)"
            )

    return result


def calculate_similarity(a: str, b: str) -> float:
    """
    Calculate text similarity between two descriptions using Jaccard similarity.
    Returns a value between 0 (completely different) and 1 (identical).
    """
    # Normalize: lowercase, remove punctuation, split into words
    def normalize(s: str) -> list[str]:
        import re
        return [w for w in re.sub(r'[^\w\s]', '', s.lower()).split() if w]

    words_a = normalize(a)
    words_b = normalize(b)

    if not words_a and not words_b:
        return 1.0
    if not words_a or not words_b:
        return 0.0

    set_a = set(words_a)
    set_b = set(words_b)
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)

    return intersection / union if union > 0 else 0.0


# ============================================================================
# Whisper Transcription
# ============================================================================

async def transcribe_with_whisper(
    session: aiohttp.ClientSession,
    base_url: str,
    video_path: Path,
    api_token: str | None,
    logger: PipelineLogger,
) -> tuple[list[TranscriptSegment], float]:
    """
    Transcribe audio from video using Whisper ASR.
    Returns list of transcript segments and duration.
    """
    url = f"{base_url.rstrip('/')}/v1/audio/transcriptions"

    # Read video file
    with open(video_path, "rb") as f:
        video_data = f.read()

    file_size_mb = len(video_data) / (1024 * 1024)
    filename = video_path.name

    logger.log_whisper_request(filename, file_size_mb)
    logger.info(f"Whisper → {url} ({file_size_mb:.2f} MB)")

    # Prepare multipart form data
    form = aiohttp.FormData()
    form.add_field("file", video_data, filename=filename, content_type="video/mp4")
    form.add_field("model", WHISPER_MODEL)
    form.add_field("response_format", "verbose_json")
    form.add_field("timestamp_granularities[]", "segment")

    headers = {}
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"

    # Audio transcription can take a while for long videos
    timeout = aiohttp.ClientTimeout(total=600)  # 10 minutes for transcription
    async with session.post(url, data=form, headers=headers, timeout=timeout) as response:
        response.raise_for_status()
        result = await response.json()

    # Parse segments
    segments: list[TranscriptSegment] = []
    raw_segments = result.get("segments", [])

    for seg in raw_segments:
        text = str(seg.get("text", "")).strip()
        if text:
            # Calculate confidence from avg_logprob if available
            avg_logprob = seg.get("avg_logprob")
            confidence = (
                min(1.0, max(0.0, 1.0 + avg_logprob / 2))
                if avg_logprob is not None
                else 0.85
            )

            segments.append(TranscriptSegment(
                start=float(seg.get("start", 0)),
                end=float(seg.get("end", 0)),
                text=text,
                confidence=confidence,
            ))

    # Get duration (ensure it's a float)
    raw_duration = result.get("duration", 0)
    duration = float(raw_duration) if raw_duration else 0.0
    if not duration and segments:
        duration = segments[-1].end

    logger.log_whisper_response(result, segments, duration)
    logger.info(f"Whisper ← {len(segments)} segments, {duration:.1f}s duration")

    if segments:
        logger.debug(f"First segment: {segments[0].start:.1f}s-{segments[0].end:.1f}s "
                    f'"{segments[0].text[:50]}..."')

    return segments, duration


# ============================================================================
# Synthesis
# ============================================================================

async def synthesize_with_openai(
    session: aiohttp.ClientSession,
    batch_descriptions: list[str],
    transcript_text: str,
    api_key: str,
    logger: PipelineLogger,
) -> str:
    """Use OpenAI to synthesize tutorial from descriptions + transcript."""
    logger.log_synthesis_request(batch_descriptions, transcript_text)

    visual_section = "\n\n".join(
        f"[Batch {i + 1}]\n{desc}" for i, desc in enumerate(batch_descriptions)
    )

    user_prompt = (
        f"{SYNTHESIS_PROMPT}\n\n"
        f"--- VISUAL DESCRIPTIONS (use these for exact spelling of technical terms) ---\n{visual_section}\n\n"
        f"--- AUDIO TRANSCRIPT (may contain phonetic errors for technical terms) ---\n{transcript_text or '(no narration detected)'}\n\n"
        f"--- OUTPUT ---\nWrite the tutorial in Markdown now:"
    )

    url = "https://api.openai.com/v1/chat/completions"
    payload = {
        "model": "gpt-5.4-mini",
        "messages": [
            {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "reasoning_effort": "medium",
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    logger.info("Calling OpenAI for synthesis...")

    async with session.post(url, json=payload, headers=headers, timeout=120) as response:
        response.raise_for_status()
        result = await response.json()

    markdown = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    logger.log_synthesis_response(markdown, "openai")

    return markdown


def fallback_markdown(descriptions: list[str], transcript: str, logger: PipelineLogger) -> str:
    """Simple fallback if OpenAI is unavailable."""
    logger.log_synthesis_request(descriptions, transcript)

    lines: list[str] = []
    first_content = transcript or (descriptions[0] if descriptions else "Tutorial")
    title_words = " ".join(first_content.split()[:8])
    lines.extend([f"# {title_words}", ""])

    if transcript:
        summary = transcript[:300] + "..." if len(transcript) > 300 else transcript
        lines.extend([f"> {summary}", ""])

    for i, desc in enumerate(descriptions):
        # Get first sentence for heading
        first_sentence = next(
            (s.strip() for s in desc.split(".") if s.strip() and len(s.strip()) > 5),
            f"Step {i + 1}"
        )
        heading = first_sentence[:80] + "..." if len(first_sentence) > 80 else first_sentence
        lines.extend([f"## Step {i + 1}: {heading}", "", desc.strip(), ""])

    if transcript:
        lines.extend(["---", "", "## Transcript", "", transcript, ""])

    markdown = "\n".join(lines)
    logger.log_synthesis_response(markdown, "fallback")

    return markdown


# ============================================================================
# Walkthrough Synthesis
# ============================================================================

def synthesize_walkthrough(
    segments: list[TranscriptSegment],
    events: list[VisualEvent],
    max_steps: int = 12,
) -> Walkthrough:
    """Synthesize walkthrough from transcript segments and visual events."""
    if not segments and not events:
        return Walkthrough(
            title="Untitled Walkthrough",
            summary="No transcript or visual events were detected.",
            steps=[],
            warnings=["The analysis did not produce usable transcript or visual events."],
        )

    # Use events as timeline, or create from segments
    timeline = events or [
        VisualEvent(
            start=seg.start,
            end=seg.end,
            description=seg.text,
            confidence=seg.confidence,
        )
        for seg in segments
    ]

    steps: list[WalkthroughStep] = []
    warnings: list[str] = []

    for event in timeline[:max_steps]:
        # Find overlapping transcript segments
        aligned = [
            seg for seg in segments
            if seg.start <= event.end and seg.end >= event.start
        ]

        narration = " ".join(seg.text.strip() for seg in aligned if seg.text.strip())
        heading = heading_from(narration or event.description)
        instruction = instruction_from(narration, event.description)

        confidence_values = [event.confidence] + [seg.confidence for seg in aligned]
        confidence = sum(confidence_values) / len(confidence_values)

        if confidence < 0.55:
            warnings.append(f"Review step {len(steps) + 1}; confidence is low.")

        steps.append(WalkthroughStep(
            index=len(steps) + 1,
            heading=heading,
            instruction=instruction,
            timestamp_start=round(event.start, 2),
            timestamp_end=round(event.end, 2),
            screenshot=event.screenshot,
            evidence=evidence_from(narration, event.description),
            confidence=round(max(0.0, min(confidence, 1.0)), 2),
        ))

    title = heading_from(segments[0].text if segments else events[0].description)
    summary = summary_from(segments, events)

    if not segments:
        warnings.append("No narration transcript was available; steps are based on visual analysis only.")
    if not events:
        warnings.append("No visual events were available; steps are based on narration only.")

    return Walkthrough(title=title, summary=summary, steps=steps, warnings=warnings)


def heading_from(text: str) -> str:
    """Extract heading from text."""
    words = [w.strip(".,;:!?") for w in text.split() if w.strip(".,;:!?")]
    if not words:
        return "Review this step"
    compact = " ".join(words[:7])
    return compact[0].upper() + compact[1:] if compact else "Review this step"


def instruction_from(narration: str, description: str) -> str:
    """Create instruction from narration and description."""
    if narration and description:
        return f"{narration.strip()} The screen shows: {description.strip()}"
    if narration:
        return narration.strip()
    return f"Follow the visible action: {description.strip()}"


def evidence_from(narration: str, description: str) -> str:
    """Create evidence string from narration and description."""
    parts = []
    if narration:
        parts.append(f"narration: {narration}")
    if description:
        parts.append(f"visual: {description}")
    return "; ".join(parts)


def summary_from(segments: list[TranscriptSegment], events: list[VisualEvent]) -> str:
    """Create summary from segments and events."""
    if segments:
        text = " ".join(seg.text for seg in segments[:3]).strip()
        return text if text else "Generated from screen recording narration and visual events."
    return events[0].description if events else "Generated from screen recording analysis."


def format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS."""
    safe = max(0, int(seconds))
    return f"{safe // 60:02d}:{safe % 60:02d}"


def render_markdown(walkthrough: Walkthrough) -> str:
    """Render walkthrough to Markdown format."""
    lines = [f"# {walkthrough.title}", "", walkthrough.summary, ""]

    for step in walkthrough.steps:
        lines.extend([f"## {step.index}. {step.heading}", ""])
        if step.screenshot:
            lines.extend([f"![{step.heading}](assets/{step.screenshot})", ""])
        lines.extend([
            step.instruction,
            "",
            f"Timestamp: {format_timestamp(step.timestamp_start)} - {format_timestamp(step.timestamp_end)}",
            f"Confidence: {round(step.confidence * 100)}%",
        ])
        if step.evidence:
            lines.append(f"Evidence: {step.evidence}")
        lines.append("")

    if walkthrough.warnings:
        lines.extend(["## Review Notes", ""])
        lines.extend(f"- {warning}" for warning in walkthrough.warnings)
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


# ============================================================================
# Main Pipeline
# ============================================================================

async def run_pipeline(video_path: Path, output_dir: Path, log_dir: Path) -> Path:
    """Run the complete tutorial generation pipeline."""
    logger = PipelineLogger(log_dir)

    logger.info("=" * 60)
    logger.info("LocalLoom Test Pipeline")
    logger.info("=" * 60)
    logger.info(f"Video: {video_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Log directory: {log_dir}")

    # Get configuration from environment
    vision_url = os.getenv("QWENVL_BASE_URL") or os.getenv("VITE_QWENVL_BASE_URL")
    whisper_url = os.getenv("WHISPER_ASR_BASE_URL") or os.getenv("VITE_WHISPER_ASR_BASE_URL")
    api_token = os.getenv("MODAL_API_TOKEN")
    openai_api_key = os.getenv("VITE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

    if not vision_url or not whisper_url:
        logger.error("Missing QWENVL_BASE_URL or WHISPER_ASR_BASE_URL environment variables")
        raise ValueError("Missing required environment variables. Check .env file.")

    logger.info(f"Vision URL: {vision_url}")
    logger.info(f"Whisper URL: {whisper_url}")
    logger.info(f"API token: {'set' if api_token else 'not set'}")
    logger.info(f"OpenAI API key: {'set' if openai_api_key else 'not set'}")

    # Step 1: Extract frames at fixed FPS
    logger.info("-" * 40)
    logger.info("Step 1: Extracting frames at fixed FPS")
    frames = extract_frames(video_path, fps=TARGET_FPS, max_frames=MAX_FRAMES, logger=logger)

    # Step 2: Run SmolVLM and Whisper in parallel
    logger.info("-" * 40)
    logger.info("Step 2: Running SmolVLM + Whisper in parallel")

    async with aiohttp.ClientSession() as session:
        # Run both in parallel
        smolvlm_task = analyze_frames_batched(
            session, vision_url, frames, openai_api_key, logger
        )
        whisper_task = transcribe_with_whisper(
            session, whisper_url, video_path, api_token, logger
        )

        raw_batch_descriptions, (segments, duration) = await asyncio.gather(
            smolvlm_task, whisper_task
        )

    # Deduplicate consecutive similar descriptions
    batch_descriptions = deduplicate_descriptions(raw_batch_descriptions, logger=logger)

    logger.info(
        f"SmolVLM: {len(raw_batch_descriptions)} batches → {len(batch_descriptions)} after dedup"
    )
    logger.info(f"Whisper: {len(segments)} segments, {duration:.1f}s duration")

    # Step 3: Synthesize tutorial
    logger.info("-" * 40)
    logger.info("Step 3: Synthesizing tutorial")

    transcript_text = " ".join(seg.text.strip() for seg in segments if seg.text.strip())

    async with aiohttp.ClientSession() as session:
        if openai_api_key:
            try:
                markdown = await synthesize_with_openai(
                    session, batch_descriptions, transcript_text, openai_api_key, logger
                )
            except Exception as e:
                logger.warning(f"OpenAI synthesis failed: {e}, using fallback")
                markdown = fallback_markdown(batch_descriptions, transcript_text, logger)
        else:
            logger.warning("No OpenAI API key, using fallback synthesis")
            markdown = fallback_markdown(batch_descriptions, transcript_text, logger)

    logger.info(f"Generated markdown: {len(markdown)} chars")

    # Step 4: Build walkthrough for backward compatibility
    logger.info("-" * 40)
    logger.info("Step 4: Building walkthrough structure")

    num_batches = len(batch_descriptions)
    events = [
        VisualEvent(
            start=idx * (duration / num_batches) if num_batches > 0 else 0,
            end=(idx + 1) * (duration / num_batches) if num_batches > 0 else 0,
            description=desc,
            confidence=0.72,
        )
        for idx, desc in enumerate(batch_descriptions)
    ]

    walkthrough = synthesize_walkthrough(segments, events, max_steps=20)
    logger.info(f"Walkthrough: {len(walkthrough.steps)} steps, {len(walkthrough.warnings)} warnings")

    # Step 5: Save outputs
    logger.info("-" * 40)
    logger.info("Step 5: Saving outputs")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_name = video_path.stem

    # Save markdown tutorial
    tutorial_path = output_dir / f"tutorial_{video_name}_{timestamp}.md"
    with open(tutorial_path, "w") as f:
        f.write(markdown)
    logger.info(f"Tutorial saved: {tutorial_path}")

    # Save walkthrough JSON
    walkthrough_path = output_dir / f"walkthrough_{video_name}_{timestamp}.json"
    walkthrough_dict = {
        "title": walkthrough.title,
        "summary": walkthrough.summary,
        "steps": [asdict(step) for step in walkthrough.steps],
        "warnings": walkthrough.warnings,
    }
    with open(walkthrough_path, "w") as f:
        json.dump(walkthrough_dict, f, indent=2)
    logger.info(f"Walkthrough saved: {walkthrough_path}")

    # Save structured markdown from walkthrough
    structured_md_path = output_dir / f"walkthrough_{video_name}_{timestamp}.md"
    structured_markdown = render_markdown(walkthrough)
    with open(structured_md_path, "w") as f:
        f.write(structured_markdown)
    logger.info(f"Structured markdown saved: {structured_md_path}")

    # Save all model logs
    logger.save_all_logs()

    logger.info("=" * 60)
    logger.info("Pipeline completed successfully!")
    logger.info("=" * 60)

    return tutorial_path


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="LocalLoom Test Pipeline - Generate tutorials from video recordings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python test_pipeline.py video.mp4
    python test_pipeline.py /path/to/recording.mp4

Environment variables (from .env):
    QWENVL_BASE_URL - QwenVL Modal VM URL (or VITE_QWENVL_BASE_URL)
    WHISPER_ASR_BASE_URL - Whisper ASR Modal VM URL
    MODAL_API_TOKEN - Optional API token
    VITE_OPENAI_API_KEY - Optional OpenAI key for synthesis

Output:
    - Tutorial: src/tutorial_<name>_<timestamp>.md
    - Walkthrough: src/walkthrough_<name>_<timestamp>.json
    - Logs: logs/pipeline_<timestamp>.log
    - Model outputs: logs/vision_<timestamp>.json, etc.
        """,
    )
    parser.add_argument(
        "video_path",
        type=Path,
        help="Path to the video file to process",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for tutorials (default: src/)",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Log directory (default: logs/)",
    )

    args = parser.parse_args()

    # Resolve paths relative to script location
    script_dir = Path(__file__).parent.resolve()

    video_path = args.video_path.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else script_dir / "src"
    log_dir = args.log_dir.resolve() if args.log_dir else script_dir / "logs"

    # Validate video exists
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Run pipeline
    try:
        result = asyncio.run(run_pipeline(video_path, output_dir, log_dir))
        print(f"\nTutorial generated: {result}")
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e) if str(e) else error_type
        print(f"\nPipeline failed: {error_type}: {error_msg}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
