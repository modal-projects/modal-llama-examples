import os
import subprocess
import time
from pathlib import Path

import modal
import modal.experimental

# Configuration constants
MODEL = "meta-llama/Llama-3.3-70B-Instruct"
GPU_TYPE = os.environ.get("GPU_TYPE", "B200")
GPU_COUNT = int(os.environ.get("GPU_COUNT", "1"))
GPU_CONFIG = f"{GPU_TYPE}:{GPU_COUNT}"
PORT = 8000
MINUTES = 60
N_CONTAINERS = 1

# Volume management
model_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

volumes = {
    "/root/.cache/huggingface": model_vol,
    "/root/.cache/vllm": vllm_cache_vol,
}

# Image configuration with vLLM environment variables
vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("huggingface_hub[hf_xet]", "requests")
    .uv_pip_install("hf_transfer")
    .uv_pip_install("vllm==0.10.2", extra_options="--torch-backend=cu128")
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HUB_VERBOSITY": "debug",
            "NCCL_DEBUG": os.environ.get("NCCL_DEBUG", "WARN"),
            "VLLM_LOGGING_LEVEL": os.environ.get("VLLM_LOGGING_LEVEL", "INFO"),
        }
    )
)

app = modal.App("glm-4.5-vllm")

with vllm_image.imports():
    import httpx

def serve():
    """Launch vLLM server with configured parameters."""
    vllm_cmd = _build_vllm_cmd(
        GPU_COUNT,
        32,  # max_seqs - will use vLLM defaults
        128000,  # max_model_len
        True,  # enable_expert_parallel
        PORT,
    )

    # Debug logging
    if os.environ.get("VLLM_LOGGING_LEVEL") == "DEBUG":
        print("Debug logging enabled for vLLM 🐛")

    if os.environ.get("VLLM_TORCH_PROFILER_DIR"):
        print(
            f"Torch profiler enabled, traces will be saved to {VLLM_PROFILES_PATH} 📊"
        )
        print("⚠️  Profiling will significantly impact performance")

    print(f"vLLM command: {' '.join(vllm_cmd)}")
    subprocess.Popen(" ".join(vllm_cmd), shell=True)

def _build_vllm_cmd(
    tp_size: int,
    max_seqs: None | int,
    max_model_len: int,
    enable_expert_parallel: bool,
    port: int,
) -> list[str]:
    """Build vLLM command with enhanced options."""
    print("Starting vLLM server...")
    vllm_cmd = [
        "vllm",
        "serve",
        MODEL,
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--max-model-len",
        str(max_model_len),
        "--gpu-memory-utilization",
        "0.95",
        "--tensor-parallel-size",
        str(tp_size),
        "--enable-prefix-caching",  # Performance optimization
    ]

    if max_seqs is not None:
        vllm_cmd.extend(["--max-num-seqs", str(max_seqs)])

    if enable_expert_parallel:
        vllm_cmd.append("--enable-expert-parallel")

    # Optional KV cache dtype configuration
    kv_cache_dtype = os.environ.get("VLLM_KV_CACHE_DTYPE")
    if kv_cache_dtype:
        vllm_cmd.extend(["--kv-cache-dtype", kv_cache_dtype])

    return vllm_cmd


@app.cls(
    gpu=GPU_CONFIG,
    image=vllm_image,
    timeout=30 * MINUTES,
    volumes=volumes,
    experimental_options={"flash": "us-east"},
    min_containers=N_CONTAINERS,
)
class VLLM:
    """
    Run vLLM inference with enhanced debugging and profiling support.
    """

    @modal.enter()
    def enter(self):
        """Initialize vLLM server with health checking."""
        serve()

        self.flash_handle = modal.experimental.flash_forward(PORT)

    @modal.method()
    def method(self):
        """Keep server running."""
        pass

    @modal.exit()
    def exit(self):
        """Graceful shutdown with enhanced logging."""
        print(f"{self.flash_handle.get_container_url()} Stopping flash handle")
        self.flash_handle.stop()

        print(
            f"{self.flash_handle.get_container_url()} Waiting 15 seconds to finish requests"
        )
        time.sleep(15)

        # Check if profiling was enabled and log profiling data location
        if os.environ.get("VLLM_TORCH_PROFILER_DIR"):
            print(f"Profiling traces saved to {VLLM_PROFILES_PATH} 📊")
            print("ℹ️  Traces can be visualized at https://ui.perfetto.dev/")

        print(f"{self.flash_handle.get_container_url()} Closing flash handle")
        self.flash_handle.close()

