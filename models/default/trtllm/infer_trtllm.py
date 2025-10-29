import subprocess
import time

from pathlib import Path

import modal
import modal.experimental

MINUTES = 60

MODEL = "deepseek-ai/DeepSeek-V3"

TRTLLM_PROFILES = modal.Volume.from_name("trtllm-profiles", create_if_missing=True)
TRTLLM_PROFILES_PATH = Path("/trtllm/profiles")
TRTLLM_ENGINE_PATH = Path("/trtllm/engines")
TRTLLM_ENGINE_BUILDS = modal.Volume.from_name(
    "trtllm-engine-builds", create_if_missing=True
)
MODEL_VOL = modal.Volume.from_name("big-model-hfcache", create_if_missing=True)
MODEL_VOL_PATH = Path("/root/.cache/")
volumes = {
    MODEL_VOL_PATH: MODEL_VOL,
    TRTLLM_PROFILES_PATH: TRTLLM_PROFILES,
    TRTLLM_ENGINE_PATH: TRTLLM_ENGINE_BUILDS,
}

cuda_version = "12.8.1"  # should be no greater than host CUDA version
flavor = "devel"  # includes full CUDA toolkit
operating_sys = "ubuntu24.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

trtllm_image = (
    modal.Image.from_registry("nvcr.io/nvidia/tensorrt-llm/release:1.1.0rc1")
    .entrypoint([])  # Remove verbose logging by base image on entry
    .pip_install("hf_transfer")
    .env(
        {
            "HF_HUB_CACHE": "/root/.cache/hub",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HUB_VERBOSITY": "debug",
            "PMIX_MCA_gds": "hash",
            "TRTLLM_VERBOSE_ARTIFACTS": "1",
            "TRTLLM_LOG_LEVEL": "DEBUG",
            "TRTLLM_ENABLE_PDL": "1",
        },
    )
    .add_local_file(f"configs/{MODEL}/trtllm.yaml", "/configs/llm_api_options.yaml")
)

app = modal.App("deepseek-v3-nospec")

N_CONTAINERS = 1
PORT = 8000


def serve():
    cmd = [
        "trtllm-serve",
        MODEL,
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--max_seq_len",
        "48000",
        "--tp_size",
        "8",
        "--trust_remote_code",
        "--extra_llm_api_options",
        "/configs/llm_api_options.yaml",
    ]

    print(cmd)

    subprocess.Popen(" ".join(cmd), shell=True)


@app.cls(
    gpu="B200:8",
    image=trtllm_image,
    timeout=30 * MINUTES,
    volumes=volumes,
    experimental_options={"flash": "us-east"},
    min_containers=N_CONTAINERS,
)
class TRTLLM:
    @modal.enter()
    def enter(self):
        serve()

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
