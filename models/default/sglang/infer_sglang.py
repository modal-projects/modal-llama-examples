import os
import subprocess
import time

import modal

MINUTES = 60

MODEL_PATH = "meta-llama/Llama-3.3-70B-Instruct"
GPU_TYPE = "B200"
GPU_COUNT = 1
PORT = 8000
MAX_BATCH_SIZE = 64


image = (
    modal.Image.from_registry("lmsysorg/sglang:v0.5.3rc1-cu128-b200")
        .env(
            {
                "HF_HUB_ENABLE_HF_TRANSFER": "1",
                "PMIX_MCA_gds": "hash",
            }
        )
)

app = modal.App("figma-sglang-llama3.3-70b")

with image.imports():
    import httpx

def serve():
    cmd = [
        "python",
        "-m",
        "sglang.launch_server",
        "--model-path",
        MODEL_PATH,
        "--host",
        "0.0.0.0",
        "--port",
        str(PORT),
        "--mem-fraction-static",
        "0.7",
        "--max-running-requests",
        str(MAX_BATCH_SIZE),
        "--tp-size",
        str(GPU_COUNT),
        "--enable-metrics",
    ]

    print(cmd)

    subprocess.Popen(" ".join(cmd), shell=True)

def is_healthy(timeout=20 * MINUTES):
    url: str = f"http://127.0.0.1:{PORT}/health"
    deadline: float = time.time() + timeout
    while time.time() < deadline:
        try:
            with httpx.Client(timeout=5) as client:
                response = client.get(url)
                if response.status_code == 200:
                    print("Server is healthy 🚀")
                    return True
        except Exception:  # pylint: disable=broad-except
            pass
        time.sleep(5)

    return False

@app.cls(
    gpu=f"{GPU_TYPE}:{GPU_COUNT}",
    image=image,
    timeout=30 * MINUTES,
    volumes={
        "/root/.cache/flashinfer": modal.Volume.from_name("flashinfer-cache", create_if_missing=True),
        "/root/.cache/huggingface": modal.Volume.from_name("huggingface-cache", create_if_missing=True),
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
    max_containers=1,
    min_containers=1,
)
class SGLang:
    @modal.enter()
    def enter(self):
        serve()

        timeout = 20 * MINUTES
        if not is_healthy(timeout=timeout):
            raise Exception(f"Container not healthy after {timeout} seconds")

    @modal.web_server(port=PORT)
    def method(self):
        pass
