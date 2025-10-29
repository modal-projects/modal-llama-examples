import subprocess
import time

import modal

MINUTES = 60

MODEL = "meta-llama/Llama-3.3-70B-Instruct"
GPU_TYPE = "B200"
GPU_COUNT = 1
PORT = 8000

trtllm_image = (
    modal.Image.from_registry("nvcr.io/nvidia/tensorrt-llm/release:1.1.0rc1")
    .entrypoint([])  # Remove verbose logging by base image on entry
    .pip_install("hf_transfer")
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "PMIX_MCA_gds": "hash",
        },
    )
    .add_local_file(f"models/default/trtllm/trtllm.yaml", "/configs/llm_api_options.yaml")
)

app = modal.App("figma-llama3.3-70b-test")

with trtllm_image.imports():
    import httpx


def serve():
    cmd = [
        "trtllm-serve",
        MODEL,
        "--host",
        "0.0.0.0",
        "--port",
        str(PORT),
        "--tp_size",
        str(GPU_COUNT),
        "--backend",
        "pytorch",
        "--trust_remote_code",
        "--extra_llm_api_options",
        "/configs/llm_api_options.yaml",
    ]

    print(cmd)

    subprocess.Popen(" ".join(cmd), shell=True)


@app.cls(
    gpu=f"{GPU_TYPE}:{GPU_COUNT}",
    image=trtllm_image,
    timeout=30 * MINUTES,
    volumes={
        "/root/.cache/huggingface": modal.Volume.from_name("huggingface-cache", create_if_missing=True),
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
    min_containers=1,
)
class TRTLLM:
    @modal.enter()
    def enter(self):
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

    @modal.web_server(8000)
    def method(self):
        pass
