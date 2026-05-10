# Open-Source Loom-Style Recorder With SmolVLM2 + Whisper ASR

## Summary
Build a macOS-first **Tauri + React + TypeScript** desktop app that records screen video and microphone audio, exports the MP4 locally, then generates a Markdown tutorial using two Modal-hosted vLLM inference VMs:

- **SmolVLM2-2.2B-Instruct** for visual video understanding.
- **openai/whisper-large-v3-turbo** for narration transcription.

The generated walkthrough combines visible UI actions with spoken intent, producing a step-by-step Markdown guide with timestamps, transcript-aware instructions, and optional screenshot references.

References: [SmolVLM2 model card](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct), [Whisper large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo).

## Key Changes
- Create the desktop recorder:
  - macOS-first screen capture via `getDisplayMedia` with optional microphone capture via `getUserMedia`.
  - Local recording library with in-app video preview.
  - Tauri backend for persisting recordings, reading files as base64, and writing export folders containing `video.mp4`/`video.webm`, `guide.md`, `walkthrough.json`, and `assets/*.jpg`.
  - Settings for per-service Modal vLLM URLs (`smolvlmBaseUrl`, `whisperAsrBaseUrl`), a legacy combined URL fallback (`modalBaseUrl`), and API token.
- Create a Modal-hosted transcription VM (`services/modal_whisper_asr.py`):
  - Runs `openai/whisper-large-v3-turbo` via vLLM on an A10G GPU.
  - Exposes an OpenAI-compatible `/v1/chat/completions` endpoint.
  - Accepts base64 `input_audio` in chat messages and returns transcribed text.
  - ASR only; TTS and voice chat are out of scope.
- Create a Modal-hosted visual analysis VM (`services/modal_smolvlm.py`):
  - Runs `HuggingFaceTB/SmolVLM2-2.2B-Instruct` via vLLM on an A10G GPU.
  - Exposes an OpenAI-compatible `/v1/chat/completions` endpoint.
  - Accepts base64 video via `video_url` data URLs in chat messages and returns visual descriptions.
- Add a tutorial synthesis step (both TypeScript `src/lib/guide.ts` and Python `services/guide.py`):
  - Align transcript segments with visual events by timestamp overlap.
  - Prefer narration for user intent and SmolVLM2 for visual evidence.
  - Generate Markdown with numbered steps, screenshot references, timestamps, and confidence notes.

## Public Interfaces
- Desktop config (`AppConfig`):
  - `modalBaseUrl: string` (legacy combined fallback)
  - `smolvlmBaseUrl: string`
  - `whisperAsrBaseUrl: string`
  - `modalApiToken: string`
  - `recordingDirectory: string`
  - `exportDirectory: string`
- Modal VM endpoints (both VMs share the same OpenAI-compatible interface):
  - `GET /health` — returns vLLM server readiness.
  - `POST /v1/chat/completions` — standard OpenAI chat completions format.
    - SmolVLM VM accepts `video_url` content parts with base64 data URLs.
    - Whisper ASR VM accepts `input_audio` content parts with base64 audio data.
  - Authentication: if `OPENLOOM_API_TOKEN` is set on the VM, requests require `Authorization: Bearer <token>`.
- Guide synthesis is performed client-side (not via a Modal endpoint). The desktop app calls both VMs in parallel, then runs local alignment and Markdown rendering.
- Walkthrough JSON:
  - `title: string`
  - `summary: string`
  - `steps: Array<{ index, heading, instruction, timestampStart, timestampEnd, screenshot, evidence, confidence }>`
  - `warnings: string[]`

## Test Plan
- Unit test Markdown generation from fixed transcript + visual event fixtures (TypeScript: `src/lib/guide.test.ts`, Python: `tests/test_guide.py`).
- Unit test timestamp alignment logic, including silent sections and narration with no visual change.
- Integration test Modal `/health` and `/v1/chat/completions` using a short fixture recording (requires deployed VMs).
- Manual macOS QA:
  - grant screen and microphone permissions
  - record a short tutorial
  - preview recording in-app
  - generate transcript via Whisper ASR VM
  - generate visual events via SmolVLM VM
  - export and open Markdown with working screenshots
- Failure cases:
  - Modal VM unavailable or cold-starting
  - missing API token
  - no microphone audio (screen-only recording)
  - empty transcript
  - GPU/model load failure
  - low-confidence visual analysis

## Assumptions
- MVP target is **macOS first**.
- Inference runs in **two separate Modal-hosted vLLM VMs**, not inside the desktop app.
- `openai/whisper-large-v3-turbo` is used for ASR/transcription only (chosen over LFM2.5-Audio for better vLLM compatibility).
- `SmolVLM2-2.2B-Instruct` is used for video understanding only.
- Guide synthesis happens **client-side** in the desktop app, not on a server.
- Output is **local Markdown export** with optional screenshots and the original recording.
- Hosted Loom-style share pages, auth, teams, comments, browser extension capture, and cloud video storage are out of scope for v1.
