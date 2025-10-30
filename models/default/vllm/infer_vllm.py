import os
import subprocess
import time

import modal

MINUTES = 60

MODEL = "meta-llama/Llama-3.3-70B-Instruct"
GPU_TYPE = "B200"
GPU_COUNT = 1
PORT = 8000
MAX_BATCH_SIZE = 64

# Image configuration with vLLM environment variables
vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("huggingface_hub[hf_xet]", "requests")
    .uv_pip_install("hf_transfer")
    .uv_pip_install("vllm==0.10.2", extra_options="--torch-backend=cu128")
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
        }
    )
)

app = modal.App("figma-vllm-llama3.3-70b")

with vllm_image.imports():
    import httpx

def serve():
    """Launch vLLM server with configured parameters."""
    vllm_cmd = [
        "vllm",
        "serve",
        MODEL,
        "--host",
        "0.0.0.0",
        "--port",
        str(PORT),
        "--max-model-len",
        "8192",
        "--gpu-memory-utilization",
        "0.95",
        "--tensor-parallel-size",
        str(GPU_COUNT),
        "--enable-prefix-caching",  # Performance optimization
    ]

    print(f"vLLM command: {' '.join(vllm_cmd)}")

    subprocess.Popen(" ".join(vllm_cmd), shell=True)

@app.cls(
    gpu=f"{GPU_TYPE}:{GPU_COUNT}",
    image=vllm_image,
    timeout=30 * MINUTES,
    volumes={
        "/root/.cache/huggingface": modal.Volume.from_name("huggingface-cache", create_if_missing=True),
        "/root/.cache/vllm": modal.Volume.from_name("vllm-cache", create_if_missing=True),
    },
    min_containers=1,
    max_containers=1,
)
@modal.concurrent(max_inputs=MAX_BATCH_SIZE)
class Inference:
    @modal.enter()
    def enter(self):
        """Initialize vLLM server with health checking."""
        serve()

        deadline: float = time.time() + 5 * 60
        while time.time() < deadline:
            try:
                with httpx.Client(timeout=5) as client:
                    response = client.get(f"http://127.0.0.1:{PORT}/health")
                    if response.status_code == 200:
                        print("Server is healthy 🚀")
                        break
            except Exception:  # pylint: disable=broad-except
                pass
            time.sleep(5)
        else:
            raise RuntimeError("Health-check failed – server did not respond in time")

    @modal.web_server(port=PORT)
    def method(self):
        """Keep server running."""
        pass
