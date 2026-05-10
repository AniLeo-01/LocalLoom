"""
Test script to check if the SmolVLM vLLM deployment supports multi-image and text inputs.

Usage:
    # Set the base URL of your deployed SmolVLM service
    export QWENVL_BASE_URL="https://your-modal-url.modal.run"
    # Optional: set API token if configured
    export OPENLOOM_API_TOKEN="your-token"

    python services/test_smolvlm.py

    # Or test with a real video file (extracts frames via ffmpeg):
    python services/test_smolvlm.py /path/to/recording.mp4
"""

import asyncio
import base64
import os
import struct
import subprocess
import sys
import tempfile
import zlib

import aiohttp
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("QWENVL_BASE_URL", "").rstrip("/")
API_TOKEN = os.getenv("OPENLOOM_API_TOKEN", "")
MODEL = "Qwen/Qwen3.5-4B"


def headers() -> dict[str, str]:
    h = {"Content-Type": "application/json"}
    if API_TOKEN:
        h["Authorization"] = f"Bearer {API_TOKEN}"
    return h


def make_minimal_png(r: int = 255, g: int = 0, b: int = 0) -> bytes:
    """Create a minimal 1x1 PNG with the given color."""

    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        c = chunk_type + data
        return (
            struct.pack(">I", len(data))
            + c
            + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
        )

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    raw_row = bytes([0, r, g, b])
    idat = zlib.compress(raw_row)
    return sig + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b"")


def extract_frames_from_video(video_path: str, count: int = 8) -> list[str]:
    """Extract evenly-spaced JPEG frames from a video file, return as base64 strings."""
    tmp_dir = tempfile.mkdtemp(prefix="openloom_test_frames_")
    output_pattern = os.path.join(tmp_dir, "frame_%03d.jpg")

    # Get duration
    probe = subprocess.run(
        ["ffmpeg", "-i", video_path, "-f", "null", "-"],
        capture_output=True,
        timeout=30,
    )
    stderr = probe.stderr.decode(errors="replace")
    duration = 10.0
    for line in reversed(stderr.splitlines()):
        if "Duration:" in line:
            parts = line.split("Duration:")[1].split(",")[0].strip().split(":")
            if len(parts) == 3:
                duration = (
                    float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
                )
            break

    fps = count / max(duration, 0.1)
    subprocess.run(
        [
            "ffmpeg",
            "-i",
            video_path,
            "-vf",
            f"fps={fps:.4f},scale=512:-2",
            "-frames:v",
            str(count),
            "-q:v",
            "5",
            "-y",
            output_pattern,
        ],
        capture_output=True,
        timeout=30,
    )

    frames = []
    for i in range(1, count + 1):
        path = os.path.join(tmp_dir, f"frame_{i:03d}.jpg")
        if os.path.exists(path):
            with open(path, "rb") as f:
                frames.append(base64.b64encode(f.read()).decode())
            os.unlink(path)
    os.rmdir(tmp_dir)
    return frames


async def test_health(session: aiohttp.ClientSession) -> bool:
    print("─── Test: Health check ───")
    try:
        async with session.get(
            "/health", timeout=aiohttp.ClientTimeout(total=30)
        ) as resp:
            print(f"  Status: {resp.status}")
            if resp.status == 200:
                print("  ✅ Server is healthy")
                return True
            else:
                print(f"  ❌ Unhealthy: {await resp.text()}")
                return False
    except Exception as e:
        print(f"  ❌ Connection failed: {e}")
        return False


async def test_text_only(session: aiohttp.ClientSession) -> bool:
    print("\n─── Test: Text-only ───")
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Say hello in one word."}],
        "temperature": 0,
        "max_completion_tokens": 16,
    }
    try:
        async with session.post(
            "/v1/chat/completions",
            json=payload,
            headers=headers(),
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            body = await resp.json()
            if resp.status == 200:
                content = body["choices"][0]["message"]["content"]
                print(f"  Response: {content}")
                print("  ✅ Text-only works")
                return True
            else:
                print(f"  ❌ Status {resp.status}: {body}")
                return False
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


async def test_single_image(session: aiohttp.ClientSession) -> bool:
    print("\n─── Test: Single image ───")
    img_b64 = base64.b64encode(make_minimal_png()).decode()
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                    },
                    {"type": "text", "text": "What color is this image?"},
                ],
            }
        ],
        "temperature": 0,
        "max_completion_tokens": 32,
    }
    try:
        async with session.post(
            "/v1/chat/completions",
            json=payload,
            headers=headers(),
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            body = await resp.json()
            if resp.status == 200:
                content = body["choices"][0]["message"]["content"]
                print(f"  Response: {content}")
                print("  ✅ Single image works")
                return True
            else:
                error = body.get("message", body.get("detail", str(body)))
                print(f"  ❌ Status {resp.status}: {error}")
                return False
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


FRAMES_PER_BATCH = 4  # ~1085 tokens/frame, 4 fits within 8192 context


async def test_batched_frames(
    session: aiohttp.ClientSession, frame_b64s: list[str] | None = None
) -> bool:
    """Test sending frames in batches (how the app actually works)."""
    if frame_b64s is None:
        # Generate 8 synthetic test frames
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (128, 128, 0),
            (0, 128, 128), (128, 0, 128), (255, 128, 0), (0, 255, 128),
        ]
        frame_b64s = [base64.b64encode(make_minimal_png(*c)).decode() for c in colors]

    total = len(frame_b64s)
    batches = [
        frame_b64s[i : i + FRAMES_PER_BATCH]
        for i in range(0, total, FRAMES_PER_BATCH)
    ]
    print(f"\n─── Test: Batched frames ({total} frames, {len(batches)} batches of {FRAMES_PER_BATCH}) ───")

    all_ok = True
    descriptions: list[str] = []
    for batch_idx, batch in enumerate(batches):
        prompt = (
            f"These are frames {batch_idx * FRAMES_PER_BATCH + 1}-"
            f"{batch_idx * FRAMES_PER_BATCH + len(batch)} of {total} "
            f"from a screen recording. Describe what you see."
        )
        content: list[dict] = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            for b64 in batch
        ]
        content.append({"type": "text", "text": prompt})

        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "max_completion_tokens": 128,
        }
        try:
            async with session.post(
                "/v1/chat/completions",
                json=payload,
                headers=headers(),
                timeout=aiohttp.ClientTimeout(total=180),
            ) as resp:
                body = await resp.json()
                if resp.status == 200:
                    desc = body["choices"][0]["message"]["content"]
                    descriptions.append(desc)
                    print(f"  Batch {batch_idx + 1}/{len(batches)}: {desc[:80]}...")
                else:
                    error = body.get("message", body.get("detail", str(body)))
                    print(f"  ❌ Batch {batch_idx + 1} status {resp.status}: {error}")
                    all_ok = False
        except Exception as e:
            print(f"  ❌ Batch {batch_idx + 1} error: {e}")
            all_ok = False

    if all_ok:
        print(f"  ✅ All {len(batches)} batches succeeded!")
        print(f"\n  Combined description:")
        for i, desc in enumerate(descriptions):
            print(f"    [{i + 1}] {desc}")
    return all_ok


async def main():
    if not BASE_URL:
        print("Error: Set QWENVL_BASE_URL environment variable.")
        print("  export QWENVL_BASE_URL='https://your-url.modal.run'")
        sys.exit(1)

    print(f"SmolVLM endpoint: {BASE_URL}")
    print(f"Model: {MODEL}")
    print(f"Auth: {'yes' if API_TOKEN else 'no'}\n")

    # If a video file was provided, extract frames from it
    video_frames = None
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        if os.path.exists(video_path):
            print(f"Extracting frames from: {video_path}")
            video_frames = extract_frames_from_video(video_path, count=20)
            print(f"Extracted {len(video_frames)} frames\n")
        else:
            print(f"Warning: {video_path} not found, will use synthetic test images.\n")

    results = {}
    async with aiohttp.ClientSession(base_url=BASE_URL) as session:
        results["health"] = await test_health(session)
        if not results["health"]:
            print("\n⛔ Server is not healthy. Aborting.")
            sys.exit(1)

        results["text"] = await test_text_only(session)
        results["single_image"] = await test_single_image(session)
        results["batched_frames"] = await test_batched_frames(session, video_frames)

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name:14s} {status}")

    if all(results.values()):
        print("\n🎉 All tests passed! Frame-based analysis is working.")


if __name__ == "__main__":
    asyncio.run(main())
