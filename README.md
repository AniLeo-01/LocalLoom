# LocalLoom

LocalLoom is a macOS-first, open-source screen recorder that generates step-by-step tutorial walkthroughs from your recordings using AI.

## Features

- Screen recording with circular webcam PiP overlay
- AI-powered frame analysis (vision model)
- Audio transcription with timestamps
- Automatic tutorial generation in Markdown
- Local-first: all processing happens on your machine or your own infrastructure

## Two Ways to Use LocalLoom

### Option 1: Download DMG (Easiest)

Download the pre-built app from [Releases](https://github.com/AniLeo-01/LocalLoom/releases).

**Uses OpenAI models:**
- Vision: GPT-5.4-mini, GPT-5.2, GPT-5.4, GPT-5.5
- Transcription: GPT-4o-mini-transcribe, GPT-4o-transcribe

**Setup:**
1. Download `OpenLoom_x.x.x_aarch64.dmg`
2. Drag to Applications
3. Open LocalLoom → Settings → Enter your OpenAI API key
4. Start recording!

### Option 2: Build from Source (Customizable)

Build from source to use your own Modal-hosted models or customize the pipeline.

**Uses Modal-hosted models (via .env):**
- Vision: Qwen3.5-4B (or any vLLM-compatible model)
- Transcription: Whisper-large-v3-turbo

```bash
# Clone and install
git clone https://github.com/AniLeo-01/LocalLoom.git
cd LocalLoom
npm install

# Configure environment
cp .env.example .env
# Edit .env with your Modal URLs and API keys

# Run in development
npm run tauri:dev

# Build for production
npm run tauri:build
```

## Environment Variables (.env)

```bash
# Modal VM URLs (for build-from-source option)
VITE_QWENVL_BASE_URL=https://your-workspace--openloom-qwenvlm-serve.modal.run
VITE_WHISPER_ASR_BASE_URL=https://your-workspace--openloom-whisper-asr-serve.modal.run

# Optional: API token if OPENLOOM_API_TOKEN is set on your Modal VMs
MODAL_API_TOKEN=

# OpenAI API key (used for tutorial synthesis in both options)
VITE_OPENAI_API_KEY=sk-...
```

## Modal Services (Self-Hosted)

Deploy your own inference VMs on Modal:

```bash
# Setup Python environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r services/requirements.txt

# Deploy services
modal deploy services/modal_qwenvlm.py      # Qwen3.5-4B vision model
modal deploy services/modal_whisper_asr.py  # Whisper ASR
```

**Services:**
- `services/modal_qwenvlm.py`: Qwen3.5-4B vision model with `/v1/chat/completions`
- `services/modal_whisper_asr.py`: Whisper ASR with `/v1/audio/transcriptions`

Set `OPENLOOM_API_TOKEN` in Modal environment to require bearer token authentication.

## Output

LocalLoom exports:
- `video.mp4` - Your screen recording
- `guide.md` - AI-generated tutorial
- `walkthrough.json` - Structured walkthrough data

## Test Pipeline

Test the full pipeline from command line:

```bash
python3 test_pipeline.py path/to/video.mp4
```

This uses the Modal-hosted models configured in `.env`.

## Tests

```bash
npm test              # TypeScript tests
npm run test:services # Python service tests
```

## Tech Stack

- **Frontend:** React + TypeScript + Vite
- **Backend:** Rust + Tauri
- **Vision Models:** OpenAI GPT / Qwen3.5-4B (Modal)
- **ASR:** OpenAI Transcribe / Whisper (Modal)

## Scope

LocalLoom is local-export only. Hosted sharing, auth, comments, team workspaces, and cloud storage are intentionally out of scope.

## License

MIT
