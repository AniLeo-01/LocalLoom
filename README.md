# OpenLoom SmolVLM

OpenLoom is a macOS-first, open-source Loom-style recorder that generates tutorial walkthroughs from screen recordings.

The MVP records screen video and microphone audio locally, sends the MP4 to two Modal-hosted inference VMs, and exports:

- `video.mp4`
- `guide.md`
- `walkthrough.json`

Models:

- `HuggingFaceTB/SmolVLM2-2.2B-Instruct` for video understanding
- `openai/whisper-large-v3-turbo` for transcription

## Local Desktop Development

```bash
npm install
npm run tauri:dev
```

The app stores recordings in the configured recording directory and writes exports to the configured export directory.

## Modal Services

OpenLoom uses two OpenAI-compatible vLLM deployments on Modal:

- `services/modal_smolvlm.py`: SmolVLM2 video-analysis VM with `/health` and `/v1/chat/completions`
- `services/modal_whisper_asr.py`: Whisper ASR transcription VM with `/health` and `/v1/chat/completions`

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r services/requirements.txt
modal deploy services/modal_smolvlm.py
modal deploy services/modal_whisper_asr.py
```

Set `OPENLOOM_API_TOKEN` in either Modal environment to require the desktop app to send a matching bearer token.

Paste each deployment URL into the desktop app:

- SmolVLM VM URL: `https://your-workspace--openloom-smolvlm-serve.modal.run`
- Whisper ASR vLLM URL: `https://your-workspace--openloom-whisper-asr-serve.modal.run`

The desktop app calls both vLLM servers in parallel and synthesizes the final walkthrough locally.

## Tests

```bash
npm test
npm run test:services
```

## Scope

The MVP is local-export only. Hosted sharing links, auth, comments, team workspaces, browser extension capture, and cloud video storage are intentionally out of scope.
