from __future__ import annotations

import json
import os
import subprocess
from typing import Any

import aiohttp
import modal

APP_NAME = "openloom-qwenvlm"
MODEL_NAME = "QuantTrio/Qwen3.5-4B-AWQ"
MODEL_REVISION = os.getenv("OPENLOOM_QWENVL_REVISION", "main")
SERVED_MODEL_NAME = "qwen35-4b-awq"
VLLM_PORT = 8000
N_GPU = int(os.getenv("OPENLOOM_QWENVL_GPU_COUNT", "1"))
GPU_TYPE = os.getenv("OPENLOOM_QWENVL_GPU", "A10G")
FAST_BOOT = os.getenv("OPENLOOM_FAST_BOOT", "1") == "1"
MINUTES = 60

hf_cache_vol = modal.Volume.from_name(
    "openloom-huggingface-cache", create_if_missing=True
)
vllm_cache_vol = modal.Volume.from_name("openloom-vllm-cache", create_if_missing=True)

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.19.0", "transformers>=4.56.0,<5", "decord", "pillow", "num2words")
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "OMP_NUM_THREADS": "4"})
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
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--tensor-parallel-size",
        str(N_GPU),
        "--gpu-memory-utilization",
        "0.9",
        "--max-num-seqs",
        "1",
        "--max-model-len",
        "65536",
        "--speculative-config",
        '{"method":"qwen3_next_mtp","num_speculative_tokens":2}',
        "--reasoning-parser",
        "qwen3",
        "--trust-remote-code",
        "--uvicorn-log-level=info",
    ]
    if os.getenv("OPENLOOM_API_TOKEN"):
        cmd += ["--api-key", os.environ["OPENLOOM_API_TOKEN"]]
    cmd += ["--enforce-eager" if FAST_BOOT else "--no-enforce-eager"]
    print(" ".join(cmd))
    subprocess.Popen(cmd)


@app.local_entrypoint()
async def test(test_timeout: int = 10 * MINUTES) -> None:
    import base64

    url = await serve.get_web_url.aio()
    async with aiohttp.ClientSession(base_url=url) as session:
        print(f"Running QwenVL vLLM health check at {url}")
        async with session.get("/health", timeout=test_timeout) as response:
            response.raise_for_status()

        # Test 1: text-only
        payload: dict[str, Any] = {
            "model": SERVED_MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": "Say OpenLoom QwenVL is ready in one sentence.",
                }
            ],
            "temperature": 0,
            "max_completion_tokens": 32,
        }
        headers = _headers()
        async with session.post(
            "/v1/chat/completions", json=payload, headers=headers, timeout=test_timeout
        ) as response:
            response.raise_for_status()
            result = await response.json()
            print("Text-only:", result["choices"][0]["message"]["content"])

        # Test 2: with a tiny test image (1x1 red pixel JPEG)
        red_pixel = base64.b64encode(
            bytes(
                [
                    0xFF,
                    0xD8,
                    0xFF,
                    0xE0,
                    0x00,
                    0x10,
                    0x4A,
                    0x46,
                    0x49,
                    0x46,
                    0x00,
                    0x01,
                    0x01,
                    0x00,
                    0x00,
                    0x01,
                    0x00,
                    0x01,
                    0x00,
                    0x00,
                    0xFF,
                    0xDB,
                    0x00,
                    0x43,
                    0x00,
                    0x08,
                    0x06,
                    0x06,
                    0x07,
                    0x06,
                    0x05,
                    0x08,
                    0x07,
                    0x07,
                    0x07,
                    0x09,
                    0x09,
                    0x08,
                    0x0A,
                    0x0C,
                    0x14,
                    0x0D,
                    0x0C,
                    0x0B,
                    0x0B,
                    0x0C,
                    0x19,
                    0x12,
                    0x13,
                    0x0F,
                    0x14,
                    0x1D,
                    0x1A,
                    0x1F,
                    0x1E,
                    0x1D,
                    0x1A,
                    0x1C,
                    0x1C,
                    0x20,
                    0x24,
                    0x2E,
                    0x27,
                    0x20,
                    0x22,
                    0x2C,
                    0x23,
                    0x1C,
                    0x1C,
                    0x28,
                    0x37,
                    0x29,
                    0x2C,
                    0x30,
                    0x31,
                    0x34,
                    0x34,
                    0x34,
                    0x1F,
                    0x27,
                    0x39,
                    0x3D,
                    0x38,
                    0x32,
                    0x3C,
                    0x2E,
                    0x33,
                    0x34,
                    0x32,
                    0xFF,
                    0xC0,
                    0x00,
                    0x0B,
                    0x08,
                    0x00,
                    0x01,
                    0x00,
                    0x01,
                    0x01,
                    0x01,
                    0x11,
                    0x00,
                    0xFF,
                    0xC4,
                    0x00,
                    0x1F,
                    0x00,
                    0x00,
                    0x01,
                    0x05,
                    0x01,
                    0x01,
                    0x01,
                    0x01,
                    0x01,
                    0x01,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                    0x01,
                    0x02,
                    0x03,
                    0x04,
                    0x05,
                    0x06,
                    0x07,
                    0x08,
                    0x09,
                    0x0A,
                    0x0B,
                    0xFF,
                    0xC4,
                    0x00,
                    0xB5,
                    0x10,
                    0x00,
                    0x02,
                    0x01,
                    0x03,
                    0x03,
                    0x02,
                    0x04,
                    0x03,
                    0x05,
                    0x05,
                    0x04,
                    0x04,
                    0x00,
                    0x00,
                    0x01,
                    0x7D,
                    0x01,
                    0x02,
                    0x03,
                    0x00,
                    0x04,
                    0x11,
                    0x05,
                    0x12,
                    0x21,
                    0x31,
                    0x41,
                    0x06,
                    0x13,
                    0x51,
                    0x61,
                    0x07,
                    0x22,
                    0x71,
                    0x14,
                    0x32,
                    0x81,
                    0x91,
                    0xA1,
                    0x08,
                    0x23,
                    0x42,
                    0xB1,
                    0xC1,
                    0x15,
                    0x52,
                    0xD1,
                    0xF0,
                    0x24,
                    0x33,
                    0x62,
                    0x72,
                    0x82,
                    0x09,
                    0x0A,
                    0x16,
                    0x17,
                    0x18,
                    0x19,
                    0x1A,
                    0x25,
                    0x26,
                    0x27,
                    0x28,
                    0x29,
                    0x2A,
                    0x34,
                    0x35,
                    0x36,
                    0x37,
                    0x38,
                    0x39,
                    0x3A,
                    0x43,
                    0x44,
                    0x45,
                    0x46,
                    0x47,
                    0x48,
                    0x49,
                    0x4A,
                    0x53,
                    0x54,
                    0x55,
                    0x56,
                    0x57,
                    0x58,
                    0x59,
                    0x5A,
                    0x63,
                    0x64,
                    0x65,
                    0x66,
                    0x67,
                    0x68,
                    0x69,
                    0x6A,
                    0x73,
                    0x74,
                    0x75,
                    0x76,
                    0x77,
                    0x78,
                    0x79,
                    0x7A,
                    0x83,
                    0x84,
                    0x85,
                    0x86,
                    0x87,
                    0x88,
                    0x89,
                    0xFF,
                    0xDA,
                    0x00,
                    0x08,
                    0x01,
                    0x01,
                    0x00,
                    0x00,
                    0x3F,
                    0x00,
                    0x7B,
                    0x94,
                    0x11,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                    0xFF,
                    0xD9,
                ]
            )
        ).decode()
        payload_img: dict[str, Any] = {
            "model": SERVED_MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What do you see in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{red_pixel}"},
                        },
                    ],
                }
            ],
            "temperature": 0,
            "max_completion_tokens": 32,
        }
        async with session.post(
            "/v1/chat/completions",
            json=payload_img,
            headers=headers,
            timeout=test_timeout,
        ) as response:
            response.raise_for_status()
            result = await response.json()
            print("Image test:", result["choices"][0]["message"]["content"])


def _headers() -> dict[str, str]:
    if os.getenv("OPENLOOM_API_TOKEN"):
        return {"Authorization": f"Bearer {os.environ['OPENLOOM_API_TOKEN']}"}
    return {}
