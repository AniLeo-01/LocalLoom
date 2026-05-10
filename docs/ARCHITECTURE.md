# Architecture

OpenLoom has three parts:

1. A macOS-first Tauri desktop app for recording and local export.
2. Two Modal-hosted vLLM VMs: Qwen VLM2 for visual analysis and Whisper for transcription.
3. A deterministic guide renderer that turns structured model output into Markdown.

## Recording Flow

The desktop app captures the display through `navigator.mediaDevices.getDisplayMedia` and attempts to add microphone audio with `getUserMedia`. Tauri persists the media file locally and later reads it as base64 for upload to Modal.

## Inference Flow

The app calls two OpenAI-compatible Modal vLLM VMs in parallel.

- Whisper ASR VM: `/v1/chat/completions` sends the recording as base64 `input_audio` to `openai/whisper-large-v3-turbo`.
- Qwen VLM VM: `/v1/chat/completions` sends the MP4 as a base64 `video_url` data URL to `HuggingFaceTB/Qwen VLM2-2.2B-Instruct`.
- Desktop: aligns transcript segments and visual events by timestamp, then renders Markdown.

If `OPENLOOM_API_TOKEN` is set in either Modal environment, POST endpoints require `Authorization: Bearer <token>`. The desktop app stores that value as `modalApiToken`.

## Export Flow

The app writes an export directory containing:

- `video.mp4` or `video.webm`
- `guide.md`
- `walkthrough.json`
- `assets/` when the analysis response includes image assets

The direct vLLM MVP does not require a separate screenshot-extraction worker. The export path still supports image assets if later analysis responses include them.
