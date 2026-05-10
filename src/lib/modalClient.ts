import { invoke } from "@tauri-apps/api/core";
import {
  synthesizeWalkthrough,
  type TranscriptResponse,
  type VisualResponse,
  type Walkthrough,
} from "./guide";
import {
  VISION_SYSTEM_PROMPT,
  VISION_FRAME_PROMPT,
  SYNTHESIS_SYSTEM_PROMPT,
  SYNTHESIS_PROMPT,
} from "./prompts";

export type AppConfig = {
  recordingDirectory: string;
  exportDirectory: string;
  audioDeviceIndex: string;
  screenDeviceIndex: string;
  webcamEnabled: boolean;
  webcamDeviceIndex: string;
  webcamPosition: string;
  webcamSize: number;
  videoQuality: string;
  // OpenAI integration
  openaiApiKey: string;
  vlmProvider: "modal" | "openai";
  vlmModel: string;
  asrProvider: "modal" | "openai";
  asrModel: string;
};

function env(key: string): string {
  return import.meta.env[key] ?? "";
}

export type AnalysisResult = {
  walkthrough: Walkthrough;
  markdown: string;
  assets: ExportAsset[];
};

export type ExportAsset = {
  name: string;
  mediaBase64: string;
};

// Audio chunk from Rust backend
type AudioChunk = {
  path: string;
  startTime: number;
  endTime: number;
  sizeBytes: number;
};

// Silence segment from ffmpeg
type SilenceSegment = {
  start: number;
  end: number;
};

// Constants for OpenAI limits
const OPENAI_MAX_AUDIO_SIZE = 25 * 1024 * 1024; // 25MB

export async function analyzeRecording(
  recordingPath: string,
  config: AppConfig,
): Promise<AnalysisResult> {
  const filename = recordingPath.split("/").pop() || "recording.mp4";

  // Validate OpenAI API key if using OpenAI providers
  if ((config.vlmProvider === "openai" || config.asrProvider === "openai") && !config.openaiApiKey) {
    throw new Error("OpenAI API key is required. Please set it in Settings.");
  }

  console.log("[OpenLoom] VLM Provider:", config.vlmProvider, "Model:", config.vlmModel);
  console.log("[OpenLoom] ASR Provider:", config.asrProvider, "Model:", config.asrModel);

  // Extract frames at fixed FPS (1 frame per second, max 60 frames)
  console.log("[OpenLoom] Extracting frames from:", recordingPath);
  const frames = await invoke<string[]>("extract_frames", {
    path: recordingPath,
    fps: 1.0,
    maxFrames: 60,
  });
  console.log("[OpenLoom] Extracted", frames.length, "frames at 1 fps");

  // Run frame analysis and transcription in parallel using configured providers
  const [transcript, rawBatchDescriptions] = await Promise.all([
    transcribeAudio(recordingPath, filename, config),
    analyzeFrames(frames, config),
  ]);

  // Deduplicate consecutive batches with similar descriptions
  const batchDescriptions = deduplicateDescriptions(rawBatchDescriptions);

  console.log("[OpenLoom] Transcript segments:", transcript.segments.length);
  console.log(
    "[OpenLoom] Batch descriptions:",
    rawBatchDescriptions.length,
    "→",
    batchDescriptions.length,
    "(after dedup)"
  );

  // Synthesize tutorial via OpenAI (combines descriptions + transcript)
  const transcriptText = transcript.segments
    .map((s) => s.text.trim())
    .filter(Boolean)
    .join(" ");
  const markdown = await synthesizeTutorial(batchDescriptions, transcriptText, config);
  console.log("[OpenLoom] Synthesis done, markdown length:", markdown.length);

  // Build walkthrough for backward compat (used by the UI)
  const duration = Number(transcript.duration) || 0;
  const numBatches = batchDescriptions.length;
  const visual: VisualResponse = {
    events: batchDescriptions.map((desc, idx) => {
      const batchDuration = numBatches > 0 ? duration / numBatches : 0;
      const start = idx * batchDuration;
      const end = (idx + 1) * batchDuration;
      return {
        start,
        end,
        description: desc,
        confidence: 0.72,
      };
    }),
  };
  console.log(`[OpenLoom] Visual events: ${numBatches} batches over ${duration.toFixed(1)}s`);
  const walkthrough = synthesizeWalkthrough(transcript, visual, {
    maxSteps: 20,
  });
  const assets: ExportAsset[] = [];
  return { walkthrough, markdown, assets };
}

// ============================================================================
// Provider-Aware Frame Analysis
// ============================================================================

async function analyzeFrames(
  frames: string[],
  config: AppConfig,
): Promise<string[]> {
  if (config.vlmProvider === "openai") {
    return analyzeFramesOpenAI(frames, config);
  } else {
    return analyzeFramesModal(frames, config);
  }
}

const OPENAI_IMAGES_PER_BATCH = 10; // OpenAI allows up to 10 images per request

async function analyzeFramesOpenAI(
  frames: string[],
  config: AppConfig,
): Promise<string[]> {
  const batches: string[][] = [];
  for (let i = 0; i < frames.length; i += OPENAI_IMAGES_PER_BATCH) {
    batches.push(frames.slice(i, i + OPENAI_IMAGES_PER_BATCH));
  }

  console.log(`[OpenLoom] OpenAI Vision: ${frames.length} frames in ${batches.length} batches`);

  const results = await Promise.all(
    batches.map(async (batch, idx) => {
      const frameStart = idx * OPENAI_IMAGES_PER_BATCH + 1;
      const frameEnd = frameStart + batch.length - 1;
      const frameContext = batches.length > 1
        ? `These are frames ${frameStart}-${frameEnd} of ${frames.length} from a screen recording.\n\n`
        : "";

      console.log(`[OpenLoom] OpenAI Vision batch ${idx + 1}/${batches.length} → ${config.vlmModel}`);

      const result = await invoke<string>("call_openai_vision", {
        apiKey: config.openaiApiKey,
        model: config.vlmModel,
        systemPrompt: VISION_SYSTEM_PROMPT,
        framesBase64: batch,
        userPrompt: frameContext + VISION_FRAME_PROMPT,
      });

      console.log(`[OpenLoom] OpenAI Vision batch ${idx + 1}/${batches.length} ← ${result.length} chars`);
      return result;
    })
  );

  return results;
}

async function analyzeFramesModal(
  frames: string[],
  config: AppConfig,
): Promise<string[]> {
  const smolvlmBaseUrl = trim(env("VITE_QWENVL_BASE_URL"));
  const apiToken = env("MODAL_API_TOKEN");

  if (!smolvlmBaseUrl) {
    throw new Error("Missing VITE_QWENVL_BASE_URL for Modal VLM provider");
  }

  const smolvlmUrl = `${smolvlmBaseUrl}/v1/chat/completions`;
  console.log("[OpenLoom] Modal VLM endpoint:", smolvlmUrl);

  return analyzeFramesBatched(smolvlmUrl, frames, apiToken);
}

// ============================================================================
// Provider-Aware Audio Transcription
// ============================================================================

async function transcribeAudio(
  recordingPath: string,
  filename: string,
  config: AppConfig,
): Promise<TranscriptResponse> {
  if (config.asrProvider === "openai") {
    return transcribeWithOpenAI(recordingPath, filename, config);
  } else {
    return transcribeWithModal(recordingPath, filename, config);
  }
}

async function transcribeWithOpenAI(
  recordingPath: string,
  filename: string,
  config: AppConfig,
): Promise<TranscriptResponse> {
  // Check file size first
  const fileSize = await invoke<number>("get_file_size", { path: recordingPath });
  const fileSizeMB = fileSize / (1024 * 1024);

  console.log(`[OpenLoom] OpenAI Transcription: file size ${fileSizeMB.toFixed(1)} MB`);

  if (fileSize > OPENAI_MAX_AUDIO_SIZE) {
    console.log("[OpenLoom] File too large, splitting audio at silence points");
    return transcribeWithOpenAISplit(recordingPath, config);
  }

  // Direct transcription for smaller files
  const mediaBase64 = await invoke<string>("read_file_base64", { path: recordingPath });

  console.log(`[OpenLoom] OpenAI Transcription → ${config.asrModel}`);
  const raw = await invoke<string>("transcribe_openai", {
    apiKey: config.openaiApiKey,
    audioBase64: mediaBase64,
    filename,
    model: config.asrModel,
  });

  return parseTranscriptResponse(raw);
}

async function transcribeWithOpenAISplit(
  recordingPath: string,
  config: AppConfig,
): Promise<TranscriptResponse> {
  // Step 1: Extract audio from video
  console.log("[OpenLoom] Extracting audio for splitting...");
  const audioPath = await invoke<string>("extract_audio", { videoPath: recordingPath });

  // Step 2: Detect silence points
  console.log("[OpenLoom] Detecting silence points...");
  const silences = await invoke<SilenceSegment[]>("detect_silence", {
    audioPath,
    noiseDb: -40,
    minDuration: 1.5,
  });
  console.log(`[OpenLoom] Found ${silences.length} silence segments`);

  // Step 3: Calculate split points to keep chunks under 25MB
  // Use silence midpoints as split candidates, aim for ~10 min chunks
  const splitCandidates = silences.map(s => (s.start + s.end) / 2);
  const splitTimes = selectSplitPoints(splitCandidates);

  // Step 4: Split audio at silence points
  console.log(`[OpenLoom] Splitting audio at ${splitTimes.length} points`);
  const chunks = await invoke<AudioChunk[]>("split_audio_at_times", {
    audioPath,
    splitTimes,
  });
  console.log(`[OpenLoom] Created ${chunks.length} audio chunks`);

  // Step 5: Transcribe each chunk
  const allSegments: TranscriptResponse["segments"] = [];
  let totalDuration = 0;

  for (let i = 0; i < chunks.length; i++) {
    const chunk = chunks[i];
    const chunkSizeMB = chunk.sizeBytes / (1024 * 1024);
    console.log(`[OpenLoom] Transcribing chunk ${i + 1}/${chunks.length} (${chunkSizeMB.toFixed(1)} MB)`);

    try {
      const chunkBase64 = await invoke<string>("read_file_base64", { path: chunk.path });
      const chunkFilename = chunk.path.split("/").pop() || "chunk.mp3";

      const raw = await invoke<string>("transcribe_openai", {
        apiKey: config.openaiApiKey,
        audioBase64: chunkBase64,
        filename: chunkFilename,
        model: config.asrModel,
      });

      const chunkTranscript = parseTranscriptResponse(raw);

      // Adjust timestamps by chunk offset
      for (const segment of chunkTranscript.segments) {
        allSegments.push({
          start: segment.start + chunk.startTime,
          end: segment.end + chunk.startTime,
          text: segment.text,
          confidence: segment.confidence,
        });
      }

      totalDuration = Math.max(totalDuration, chunk.endTime);
    } catch (err) {
      console.warn(`[OpenLoom] Failed to transcribe chunk ${chunk.path}:`, err);
    }
  }

  console.log(`[OpenLoom] Combined ${allSegments.length} segments, total duration: ${totalDuration.toFixed(1)}s`);

  return {
    segments: allSegments,
    duration: totalDuration,
  };
}

function selectSplitPoints(candidates: number[]): number[] {
  // Select split points that create roughly equal-sized chunks
  // Aim for ~10 minute chunks (600 seconds) to stay well under 25MB
  const TARGET_CHUNK_DURATION = 600;
  const selected: number[] = [];
  let lastSplit = 0;

  for (const t of candidates) {
    if (t - lastSplit >= TARGET_CHUNK_DURATION) {
      selected.push(t);
      lastSplit = t;
    }
  }

  return selected;
}

function parseTranscriptResponse(raw: string): TranscriptResponse {
  const body = JSON.parse(raw);
  const segments: TranscriptResponse["segments"] = [];

  if (body.segments && Array.isArray(body.segments)) {
    for (const seg of body.segments) {
      const text = String(seg.text ?? "").trim();
      if (text.length > 0) {
        segments.push({
          start: seg.start ?? 0,
          end: seg.end ?? 0,
          text,
          confidence: seg.confidence ?? 0.85,
        });
      }
    }
  } else {
    const text = String(body.text ?? "").trim();
    if (text.length > 0) {
      segments.push({ start: 0, end: 0, text, confidence: 0.85 });
    }
  }

  const rawDuration = body.duration ?? (segments.length > 0 ? segments[segments.length - 1].end : 0);
  const duration = typeof rawDuration === "number" ? rawDuration : parseFloat(rawDuration) || 0;

  console.log(`[OpenLoom] Parsed ${segments.length} segments, duration: ${duration.toFixed(1)}s`);

  return { segments, duration };
}

async function transcribeWithModal(
  recordingPath: string,
  filename: string,
  _config: AppConfig,
): Promise<TranscriptResponse> {
  const whisperAsrBaseUrl = trim(env("VITE_WHISPER_ASR_BASE_URL"));
  const apiToken = env("MODAL_API_TOKEN");

  if (!whisperAsrBaseUrl) {
    throw new Error("Missing VITE_WHISPER_ASR_BASE_URL for Modal ASR provider");
  }

  const whisperUrl = `${whisperAsrBaseUrl}/v1/audio/transcriptions`;
  console.log("[OpenLoom] Modal Whisper endpoint:", whisperUrl);

  const mediaBase64 = await invoke<string>("read_file_base64", { path: recordingPath });

  return transcribeWithWhisper(whisperUrl, mediaBase64, filename, apiToken);
}

// ============================================================================
// Tutorial Synthesis
// ============================================================================

async function synthesizeTutorial(
  batchDescriptions: string[],
  transcriptText: string,
  config: AppConfig,
): Promise<string> {
  // Use the API key from config if available, otherwise fall back to env
  const apiKey = config.openaiApiKey || env("VITE_OPENAI_API_KEY");

  if (!apiKey) {
    console.warn("No OpenAI API key available, falling back to local synthesis");
    return fallbackMarkdown(batchDescriptions, transcriptText);
  }

  const visualSection = batchDescriptions
    .map((desc, i) => `[Batch ${i + 1}]\n${desc}`)
    .join("\n\n");

  const userPrompt =
    `${SYNTHESIS_PROMPT}\n\n` +
    `--- VISUAL DESCRIPTIONS (use these for exact spelling of technical terms) ---\n${visualSection}\n\n` +
    `--- AUDIO TRANSCRIPT (may contain phonetic errors for technical terms) ---\n${transcriptText || "(no narration detected)"}\n\n` +
    `--- OUTPUT ---\nWrite the tutorial in Markdown now:`;

  try {
    console.log("[OpenLoom] Calling OpenAI for synthesis...");
    const markdown = await invoke<string>("call_openai", {
      apiKey,
      systemPrompt: SYNTHESIS_SYSTEM_PROMPT,
      prompt: userPrompt,
    });
    return markdown || fallbackMarkdown(batchDescriptions, transcriptText);
  } catch (err) {
    console.warn("OpenAI synthesis failed, falling back to local:", err);
    return fallbackMarkdown(batchDescriptions, transcriptText);
  }
}

const FRAMES_PER_BATCH = 4; // ~1085 tokens/frame, 4 fits within 8192 context
const QWENVL_MODEL = "Qwen/Qwen3.5-4B";

/**
 * Deduplicate consecutive batch descriptions that are too similar.
 * Uses simple text similarity to detect redundant frames (e.g., static screens).
 */
function deduplicateDescriptions(descriptions: string[]): string[] {
  if (descriptions.length <= 1) return descriptions;

  const result: string[] = [descriptions[0]];

  for (let i = 1; i < descriptions.length; i++) {
    const prev = descriptions[i - 1];
    const curr = descriptions[i];

    // Calculate similarity using normalized common words
    const similarity = calculateSimilarity(prev, curr);

    // If descriptions are less than 85% similar, keep the new one
    if (similarity < 0.85) {
      result.push(curr);
    } else {
      console.log(
        `[OpenLoom] Dedup: skipping batch ${i + 1} (${Math.round(similarity * 100)}% similar to previous)`
      );
    }
  }

  return result;
}

/**
 * Calculate text similarity between two descriptions.
 * Returns a value between 0 (completely different) and 1 (identical).
 */
function calculateSimilarity(a: string, b: string): number {
  // Normalize: lowercase, remove punctuation, split into words
  const normalize = (s: string) =>
    s.toLowerCase().replace(/[^\w\s]/g, "").split(/\s+/).filter(Boolean);

  const wordsA = normalize(a);
  const wordsB = normalize(b);

  if (wordsA.length === 0 && wordsB.length === 0) return 1;
  if (wordsA.length === 0 || wordsB.length === 0) return 0;

  // Count common words
  const setA = new Set(wordsA);
  const setB = new Set(wordsB);
  const intersection = [...setA].filter((w) => setB.has(w)).length;
  const union = new Set([...wordsA, ...wordsB]).size;

  // Jaccard similarity
  return intersection / union;
}

/** Send all frames in batches and return one description string per batch. */
async function analyzeFramesBatched(
  url: string,
  frames: string[],
  apiToken: string,
): Promise<string[]> {
  const batches: string[][] = [];
  for (let i = 0; i < frames.length; i += FRAMES_PER_BATCH) {
    batches.push(frames.slice(i, i + FRAMES_PER_BATCH));
  }

  return Promise.all(
    batches.map((batch, idx) =>
      analyzeFrameBatch(url, batch, idx, batches.length, frames.length, apiToken),
    ),
  );
}

async function analyzeFrameBatch(
  url: string,
  batch: string[],
  batchIdx: number,
  totalBatches: number,
  totalFrames: number,
  apiToken: string,
): Promise<string> {
  const content: Array<Record<string, unknown>> = batch.map((b64) => ({
    type: "image_url",
    image_url: { url: `data:image/jpeg;base64,${b64}` },
  }));

  const frameStart = batchIdx * FRAMES_PER_BATCH + 1;
  const frameEnd = frameStart + batch.length - 1;
  const batchContext =
    totalBatches > 1
      ? `These are frames ${frameStart}-${frameEnd} of ${totalFrames} from a screen recording.\n\n`
      : "";

  content.push({ type: "text", text: batchContext + VISION_FRAME_PROMPT });

  const payload = {
    model: QWENVL_MODEL,
    messages: [
      { role: "system", content: VISION_SYSTEM_PROMPT },
      { role: "user", content },
    ],
  };
  console.log(`[OpenLoom] SmolVLM batch ${batchIdx + 1}/${totalBatches} → ${url}`);
  const raw = await invoke<string>("http_post", {
    url,
    body: JSON.stringify(payload),
    headers: {
      "content-type": "application/json",
      ...authHeader(apiToken),
    },
  });
  console.log(`[OpenLoom] SmolVLM batch ${batchIdx + 1}/${totalBatches} ← response length: ${raw.length}`);
  const body = JSON.parse(raw);
  const result = body.choices?.[0]?.message?.content?.trim() || "No visual description returned.";
  console.log(`[OpenLoom] SmolVLM batch ${batchIdx + 1} result: ${result.slice(0, 100)}...`);
  return result;
}

/** Simple fallback if OpenAI is unavailable. */
function fallbackMarkdown(
  descriptions: string[],
  transcript: string,
): string {
  const lines: string[] = [];
  const firstContent = transcript || descriptions[0] || "Tutorial";
  const titleWords = firstContent.split(/\s+/).slice(0, 8).join(" ");
  lines.push(`# ${titleWords}`, "");

  if (transcript) {
    const summary = transcript.length > 300 ? transcript.slice(0, 300) + "..." : transcript;
    lines.push(`> ${summary}`, "");
  }

  descriptions.forEach((desc, i) => {
    const firstSentence =
      desc.split(/[.\n]/).find((s) => s.trim().length > 5)?.trim() || `Step ${i + 1}`;
    const heading =
      firstSentence.length > 80 ? firstSentence.slice(0, 80) + "..." : firstSentence;
    lines.push(`## Step ${i + 1}: ${heading}`, "", desc.trim(), "");
  });

  if (transcript) {
    lines.push("---", "", "## Transcript", "", transcript, "");
  }

  return lines.join("\n");
}

async function transcribeWithWhisper(
  url: string,
  mediaBase64: string,
  filename: string,
  apiToken: string,
): Promise<TranscriptResponse> {
  console.log(`[OpenLoom] Whisper → ${url}`);
  const raw = await invoke<string>("transcribe_audio", {
    url,
    audioBase64: mediaBase64,
    filename,
    model: "openai/whisper-large-v3-turbo",
    apiToken,
  });
  console.log(`[OpenLoom] Whisper ← response length: ${raw.length}`);
  const body = JSON.parse(raw);

  // Parse verbose_json format with segments and timestamps
  const segments: TranscriptResponse["segments"] = [];
  if (body.segments && Array.isArray(body.segments)) {
    for (const seg of body.segments) {
      const text = String(seg.text ?? "").trim();
      if (text.length > 0) {
        segments.push({
          start: seg.start ?? 0,
          end: seg.end ?? 0,
          text,
          confidence: seg.confidence ?? seg.avg_logprob ? Math.exp(seg.avg_logprob) : 0.85,
        });
      }
    }
    console.log(`[OpenLoom] Whisper parsed ${segments.length} segments with timestamps`);
  } else {
    // Fallback for non-verbose response
    const text = String(body.text ?? "").trim();
    if (text.length > 0) {
      segments.push({ start: 0, end: 0, text, confidence: 0.85 });
    }
    console.log(`[OpenLoom] Whisper fallback: single segment without timestamps`);
  }

  // Get duration from response or estimate from last segment
  const rawDuration = body.duration ?? (segments.length > 0 ? segments[segments.length - 1].end : 0);
  const duration = typeof rawDuration === "number" ? rawDuration : parseFloat(rawDuration) || 0;
  console.log(`[OpenLoom] Whisper duration: ${duration.toFixed(1)}s`);

  if (segments.length > 0) {
    console.log(`[OpenLoom] Whisper first segment: ${segments[0].start.toFixed(1)}s-${segments[0].end.toFixed(1)}s "${segments[0].text.slice(0, 50)}..."`);
  }
  return { segments, duration };
}

function trim(value: string) {
  return value.replace(/\/+$/, "");
}

function authHeader(apiToken: string): Record<string, string> {
  return apiToken ? { authorization: `Bearer ${apiToken}` } : {};
}