import os
import subprocess
import time
import urllib.request

import modal
import modal.experimental

MINUTES = 60

MODEL_PATH = "openai/gpt-oss-120b"
GPU_TYPE = os.environ.get("GPU_TYPE", "B200")
GPU_COUNT = os.environ.get("GPU_COUNT", "1")
GPU_CONFIG = f"{GPU_TYPE}:{GPU_COUNT}"
PORT = 8000


app = modal.App("gpt-oss-120b-sglang")
hf_vol = modal.Volume.from_name("big-model-hfcache", create_if_missing=True)
sglang_image = (
    modal.Image.from_registry("lmsysorg/sglang:v0.5.3rc1-cu128-b200")
        .env(
            {
                "HF_HUB_ENABLE_HF_TRANSFER": "1",
                "PMIX_MCA_gds": "hash",
            }
        )
)


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
        "0.8",
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
            with urllib.request.urlopen(url) as response:  # nosec B310
                if response.status == 200:
                    print("Server is healthy :rocket: –", url)
                    return True
        except Exception:  # pylint: disable=broad-except
            pass
        time.sleep(5)

    return False

@app.cls(
    gpu=GPU_CONFIG,
    image=sglang_image,
    timeout=30 * MINUTES,
    volumes={
        "/root/.cache/flashinfer": modal.Volume.from_name("flashinfer-cache", create_if_missing=True),
        "/root/.cache/huggingface": hf_vol,
    },
    experimental_options={"flash": "us-east"},
    min_containers=1,
)
class SGLang:
    @modal.enter()
    def enter(self):
        serve()

        timeout = 20 * MINUTES
        if not is_healthy():
            raise Exception(f"Container not healthy after {timeout} seconds")

        self.flash_handle = modal.experimental.flash_forward(PORT)

    @modal.method()
    def method(self):
        pass

    @modal.exit()
    def exit(self):
        print(f"{self.flash_handle.get_container_url()} Stopping flash handle")
        self.flash_handle.stop()
        print(
            f"{self.flash_handle.get_container_url()} Waiting 15 seconds to finish requests"
        )
        time.sleep(15)
        print(f"{self.flash_handle.get_container_url()} Closing flash handle")
        self.flash_handle.close()
