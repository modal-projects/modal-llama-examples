import socket
import subprocess

import modal

MINUTES = 60

# MODEL = "meta-llama/Llama-3.3-70B-Instruct"
MODEL_PATH = "meta-llama/Llama-3.3-70B-Instruct"
APP_NAME = "sglang-llama3.3-70b"
GPU_TYPE = "H200"
GPU_COUNT = 2
PORT = 8000

SGLANG_IMAGE: modal.Image = modal.Image.from_registry(
    "lmsysorg/sglang:v0.5.3rc1-cu128-b200"
).env(
    {
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "PMIX_MCA_gds": "hash",
    }
)

# Persistent volumes for caching model weights and FlashInfer artifacts
FLASHINFER_CACHE_VOL: modal.Volume = modal.Volume.from_name(
    "flashinfer-cache", create_if_missing=True
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
        "0.9",
        "--max-running-requests",
        "16",
        "--tp-size",
        str(GPU_COUNT),
        "--enable-metrics",
    ]

    print(f"cmd: {cmd}")
    return subprocess.Popen(" ".join(cmd), shell=True)


app = modal.App(APP_NAME)


@app.cls(
    image=SGLANG_IMAGE,
    gpu=f"{GPU_TYPE}:{GPU_COUNT}",
    volumes={
        "/root/.cache/flashinfer": FLASHINFER_CACHE_VOL,
        "/root/.cache/huggingface": HF_CACHE_VOL,
    },
    region="us-west",
    experimental_options={"input_plane_region": "us-west"},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    startup_timeout=30 * MINUTES,
)
@modal.concurrent(max_inputs=10, target_inputs=8)
class ModelSGLang:
    @modal.enter()
    def enter(self):
        self._proc = serve()

        wait_ready(self._proc)

    @modal.web_server(port=PORT)
    def method(self):
        return

    @modal.exit()
    def exit(self):
        self._proc.terminate()


# This part runs locally as part of `modal run` to test the model is configured correctly, behind a temporary dev endpoint.
# To run evals, use `modal deploy` to create a persistent endpoint (e.g. for LangSmith evals).
@app.local_entrypoint()
def main():
    import subprocess

    # Grab the temporary dev endpoint URL.
    url = ModelSGLang().serve.get_web_url()
    print(f"Testing model at {url}")
    subprocess.run(
        f'curl -X POST {url}/v1/chat/completions -d \'{{"messages": [{{"role": "user", "content": "Hello, how are you?"}}]}}\' -H \'Content-Type: application/json\'',
        shell=True,
        check=True,
    )
    print("Test successful")
