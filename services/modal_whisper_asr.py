from __future__ import annotations

import os
import subprocess

import aiohttp
import modal

APP_NAME = "openloom-whisper-asr"
MODEL_NAME = "openai/whisper-large-v3-turbo"
MODEL_REVISION = os.getenv("OPENLOOM_WHISPER_REVISION", "main")
SERVED_MODEL_NAME = MODEL_NAME
VLLM_PORT = 8000
N_GPU = int(os.getenv("OPENLOOM_WHISPER_GPU_COUNT", "1"))
GPU_TYPE = os.getenv("OPENLOOM_WHISPER_GPU", "A10G")
FAST_BOOT = os.getenv("OPENLOOM_FAST_BOOT", "1") == "1"
MINUTES = 60

hf_cache_vol = modal.Volume.from_name(
    "openloom-huggingface-cache", create_if_missing=True
)
vllm_cache_vol = modal.Volume.from_name("openloom-vllm-cache", create_if_missing=True)

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm[audio]==0.19.0", "transformers>=4.56.0,<5")
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "VLLM_MAX_AUDIO_CLIP_FILESIZE_MB": os.getenv(
                "VLLM_MAX_AUDIO_CLIP_FILESIZE_MB", "200"
            ),
        }
    )
)

app = modal.App(APP_NAME)


@app.function(
    image=vllm_image,
    gpu=f"{GPU_TYPE}:{N_GPU}",
    scaledown_window=15 * MINUTES,
    timeout=10 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
@modal.concurrent(max_inputs=20)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
def serve() -> None:
    cmd = [
        "vllm",
        "serve",
        MODEL_NAME,
        "--served-model-name",
        SERVED_MODEL_NAME,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--uvicorn-log-level=info",
        "--dtype bfloat16",
        "--reasoning-parser qwen3",
        "--mm-encoder-tp-mode data",
        "--speculative-config '{"method": "mtp", "num_speculative_tokens": 1}'"
    ]
    if os.getenv("OPENLOOM_API_TOKEN"):
        cmd += ["--api-key", os.environ["OPENLOOM_API_TOKEN"]]
    cmd += ["--enforce-eager" if FAST_BOOT else "--no-enforce-eager"]
    cmd += ["--tensor-parallel-size", str(N_GPU)]
    print(" ".join(cmd))
    subprocess.Popen(cmd)


@app.local_entrypoint()
async def test(test_timeout: int = 10 * MINUTES) -> None:
    url = await serve.get_web_url.aio()
    async with aiohttp.ClientSession(base_url=url) as session:
        print(f"Running Whisper vLLM health check at {url}")
        async with session.get("/health", timeout=test_timeout) as response:
            response.raise_for_status()
