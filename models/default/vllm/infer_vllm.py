import socket
import subprocess

import modal

MINUTES = 60

# MODEL = "meta-llama/Llama-3.3-70B-Instruct"
MODEL = "meta-llama/Llama-3.3-70B-Instruct"
APP_NAME = "vllm-llama3.3-70b"
GPU_TYPE = "H200"
GPU_COUNT = 2
PORT = 8000

VLLM_IMAGE: modal.Image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("huggingface_hub[hf_xet]", "hf_transfer", "httpx")
    .uv_pip_install("vllm==0.11.0", extra_options="--torch-backend=cu128")
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
        }
    )
)

# Persistent volumes for caching model weights and vLLM binaries
VLLM_CACHE_VOL: modal.Volume = modal.Volume.from_name(
    "vllm-cache", create_if_missing=True
)
HF_CACHE_VOL: modal.Volume = modal.Volume.from_name(
    "huggingface-cache", create_if_missing=True
)


def wait_ready(proc: subprocess.Popen, port: int = PORT):
    while True:
        try:
            socket.create_connection(("localhost", port), timeout=1).close()
            return
        except OSError:
            if proc.poll() is not None:
                raise RuntimeError(f"server exited with {proc.returncode}")


def serve():
    cmd = [
        "vllm",
        "serve",
        MODEL,
        "--host",
        "0.0.0.0",
        "--port",
        str(PORT),
        "--gpu-memory-utilization",
        "0.95",
        "--tensor-parallel-size",
        str(GPU_COUNT),  # 1 H200 GPU
        "--enable-prefix-caching",  # Performance optimization
        "--async-scheduling",
        # "--enable-sleep-mode",
        # "--api-key", os.environ["MODAL_API_KEY"], # Optional API key
    ]
    print(f"cmd: {cmd}")
    return subprocess.Popen(" ".join(cmd), shell=True)


app = modal.App(name=APP_NAME)


@app.cls(
    image=VLLM_IMAGE,
    gpu=f"{GPU_TYPE}:{GPU_COUNT}",  # "h100:8" for 8 H100s if needed
    volumes={
        "/root/.cache/vllm": VLLM_CACHE_VOL,
        "/root/.cache/huggingface": HF_CACHE_VOL,
    },
    region="us-west",
    experimental_options={"input_plane_region": "us-west"},
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        # modal.Secret.from_name("optional-api-key"),
    ],
    startup_timeout=30 * MINUTES,
)
@modal.concurrent(max_inputs=10, target_inputs=8)
class ModelVLLM:
    @modal.enter()
    def enter(self):
        self._proc = serve()

        wait_ready(self._proc)

    @modal.web_server(8000)
    def serve(self):
        return  # vLLM handles the route

    @modal.exit()
    def exit(self):
        self._proc.terminate()


# This part runs locally as part of `modal run` to test the model is configured correctly, behind a temporary dev endpoint.
# To run evals, use `modal deploy` to create a persistent endpoint (e.g. for LangSmith evals).
@app.local_entrypoint()
def main():
    import subprocess

    # Grab the temporary dev endpoint URL.
    url = ModelVLLM().serve.get_web_url()
    print(f"Testing model at {url}")
    subprocess.run(
        f'curl -X POST {url}/v1/chat/completions -d \'{{"messages": [{{"role": "user", "content": "Hello, how are you?"}}]}}\' -H \'Content-Type: application/json\'',
        shell=True,
        check=True,
    )
    print("Test successful")
