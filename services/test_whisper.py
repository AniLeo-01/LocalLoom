"""
Test script to check if the Whisper vLLM deployment works with /v1/audio/transcriptions (multipart).

Usage:
    python services/test_whisper.py
    python services/test_whisper.py /path/to/audio.mp4
"""

import asyncio
import base64
import os
import subprocess
import sys
import tempfile

import aiohttp
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv(
    "WHISPER_ASR_BASE_URL", os.getenv("VITE_WHISPER_ASR_BASE_URL", "")
).rstrip("/")
API_TOKEN = os.getenv("OPENLOOM_API_TOKEN", os.getenv("MODAL_API_TOKEN", ""))
MODEL = "openai/whisper-large-v3-turbo"


def headers() -> dict[str, str]:
    h = {"Content-Type": "application/json"}
    if API_TOKEN:
        h["Authorization"] = f"Bearer {API_TOKEN}"
    return h


def make_test_audio() -> bytes:
    """Generate a short silent WAV file for testing."""
    import struct

    sample_rate = 16000
    duration = 1  # 1 second
    num_samples = sample_rate * duration
    # WAV header + silence
    data_size = num_samples * 2  # 16-bit samples
    wav = bytearray()
    wav += b"RIFF"
    wav += struct.pack("<I", 36 + data_size)
    wav += b"WAVE"
    wav += b"fmt "
    wav += struct.pack("<I", 16)  # chunk size
    wav += struct.pack("<H", 1)  # PCM
    wav += struct.pack("<H", 1)  # mono
    wav += struct.pack("<I", sample_rate)
    wav += struct.pack("<I", sample_rate * 2)  # byte rate
    wav += struct.pack("<H", 2)  # block align
    wav += struct.pack("<H", 16)  # bits per sample
    wav += b"data"
    wav += struct.pack("<I", data_size)
    wav += b"\x00" * data_size  # silence
    return bytes(wav)


def make_speech_audio() -> bytes | None:
    """Generate a short audio with speech via ffmpeg TTS (if available), else return None."""
    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        # Generate a sine wave tone — not speech, but at least non-silent audio
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=2",
                "-ar",
                "16000",
                "-ac",
                "1",
                tmp.name,
            ],
            capture_output=True,
            timeout=10,
        )
        if result.returncode == 0:
            with open(tmp.name, "rb") as f:
                return f.read()
    except Exception:
        pass
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass
    return None


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


async def test_models(session: aiohttp.ClientSession) -> bool:
    print("\n─── Test: List models ───")
    try:
        async with session.get(
            "/v1/models", headers=headers(), timeout=aiohttp.ClientTimeout(total=30)
        ) as resp:
            body = await resp.json()
            if resp.status == 200:
                models = [m["id"] for m in body.get("data", [])]
                print(f"  Available models: {models}")
                print("  ✅ Models endpoint works")
                return True
            else:
                print(f"  ❌ Status {resp.status}: {body}")
                return False
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


async def test_audio_transcriptions(
    session: aiohttp.ClientSession, audio_bytes: bytes, audio_format: str, label: str
) -> bool:
    print(f"\n─── Test: /v1/audio/transcriptions ({label}) ───")
    print(f"  Audio: {len(audio_bytes)} bytes, format={audio_format}")

    # Use multipart form data with verbose_json for timestamps
    data = aiohttp.FormData()
    content_type = f"audio/{audio_format}" if audio_format != "mp4" else "video/mp4"
    data.add_field(
        "file", audio_bytes, filename=f"audio.{audio_format}", content_type=content_type
    )
    data.add_field("model", MODEL)
    data.add_field("response_format", "verbose_json")
    data.add_field("timestamp_granularities[]", "segment")

    h = {}
    if API_TOKEN:
        h["Authorization"] = f"Bearer {API_TOKEN}"

    try:
        async with session.post(
            "/v1/audio/transcriptions",
            data=data,
            headers=h,
            timeout=aiohttp.ClientTimeout(total=180),
        ) as resp:
            body_text = await resp.text()
            if resp.status == 200:
                import json
                try:
                    body = json.loads(body_text)
                    if "segments" in body:
                        print(f"  Duration: {body.get('duration', 'N/A')}s")
                        print(f"  Segments: {len(body['segments'])}")
                        for seg in body["segments"][:3]:  # Show first 3 segments
                            start = seg.get("start", 0)
                            end = seg.get("end", 0)
                            text = seg.get("text", "")[:]
                            print(f"    [{start:.1f}s-{end:.1f}s] {text}...")
                        if len(body["segments"]) > 3:
                            print(f"    ... and {len(body['segments']) - 3} more segments")
                    else:
                        print(f"  Text: {body.get('text', body_text)[:200]}")
                except json.JSONDecodeError:
                    print(f"  Response: {body_text[:200]}")
                print(f"  ✅ /v1/audio/transcriptions works ({label})")
                return True
            else:
                print(f"  ❌ Status {resp.status}: {body_text[:200]}")
                return False
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


async def main():
    if not BASE_URL:
        print("Error: Set WHISPER_ASR_BASE_URL or VITE_WHISPER_ASR_BASE_URL in .env")
        sys.exit(1)

    print(f"Whisper endpoint: {BASE_URL}")
    print(f"Model: {MODEL}")
    print(f"Auth: {'yes' if API_TOKEN else 'no'}\n")

    # Load user-provided audio or generate test audio
    user_audio = None
    user_format = "mp4"
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
        if os.path.exists(audio_path):
            with open(audio_path, "rb") as f:
                user_audio = f.read()
            ext = audio_path.rsplit(".", 1)[-1].lower()
            user_format = ext if ext in ("wav", "mp3", "mp4", "webm", "m4a") else "mp4"
            print(
                f"Using audio file: {audio_path} ({len(user_audio) / 1024:.1f} KB, format={user_format})\n"
            )
        else:
            print(f"Warning: {audio_path} not found, will use generated test audio.\n")

    results: dict[str, bool] = {}
    async with aiohttp.ClientSession(base_url=BASE_URL) as session:
        results["health"] = await test_health(session)
        if not results["health"]:
            print("\n⛔ Server is not healthy. Aborting.")
            sys.exit(1)

        results["models"] = await test_models(session)

        # Test with silent WAV
        silent_wav = make_test_audio()
        results["transcriptions_silent"] = await test_audio_transcriptions(
            session, silent_wav, "wav", "silent WAV"
        )

        # Test with generated tone
        tone_audio = make_speech_audio()
        if tone_audio:
            results["transcriptions_tone"] = await test_audio_transcriptions(
                session, tone_audio, "wav", "440Hz tone"
            )

        # Test with user-provided audio
        if user_audio:
            results["transcriptions_user"] = await test_audio_transcriptions(
                session, user_audio, user_format, "user file"
            )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name:30s} {status}")

    transcription_works = results.get("transcriptions_silent", False) or results.get(
        "transcriptions_tone", False
    )

    if not transcription_works:
        print("\n💡 Transcription endpoint not working. Check:")
        print(
            "  1. Is the Whisper service deployed? Run: modal deploy services/modal_whisper_asr.py"
        )
        print("  2. Check Modal logs for startup errors")
        print("  3. Ensure --max-model-len 448 and --limit-mm-per-prompt are set")
    else:
        print("\n🎉 /v1/audio/transcriptions with multipart works!")


if __name__ == "__main__":
    asyncio.run(main())
