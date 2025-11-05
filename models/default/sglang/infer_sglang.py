import os
import socket
import subprocess

import modal

MINUTES = 60

# MODEL = "meta-llama/Llama-3.3-70B-Instruct"
MODEL_PATH = "meta-llama/Llama-3.3-70B-Instruct"
DRAFT_MODEL_PATH = "lmsys/sglang-EAGLE3-LLaMA3.3-Instruct-70B"
APP_NAME = "sglang-llama3.3-70b"
GPU_TYPE = "H200"
GPU_COUNT = 2
PORT = 8000

SGLANG_IMAGE: modal.Image = modal.Image.from_registry(
    "lmsysorg/sglang:v0.5.4.post3"
).env(
    {
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "PMIX_MCA_gds": "hash",
        "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",  # Allow EAGLE draft model with smaller context
        "SGLANG_ENABLE_SPEC_V2": "1",
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


# Base class with configuration and logic - NOT decorated with Modal
class ModelSGLangBase:
    # Configuration parameters - override in subclasses
    speculative_algorithm: str = (
        ""  # "" for vanilla, "EAGLE" for EAGLE, "NGRAM" for ngram
    )
    speculative_draft_model: str = ""  # Path to draft model for EAGLE
    speculative_num_steps: int = 5
    speculative_eagle_topk: int = 8
    speculative_num_draft_tokens: int = 32
    # Ngram parameters
    speculative_ngram_min_match_window_size: int = 1
    speculative_ngram_max_match_window_size: int = 12
    speculative_ngram_branch_length: int = 18
    # Baseline testing flag
    speculative_baseline: bool = False  # Set to True to simulate acc_len=1

    def serve(self):
        # Set environment variable for baseline testing if enabled
        env = os.environ.copy()
        if self.speculative_baseline:
            env["SGLANG_SIMULATE_ACC_LEN"] = "1"
            print("BASELINE MODE: Setting SGLANG_SIMULATE_ACC_LEN=1")

        cmd = [
            "python",
            "-m",
            "sglang.launch_server",
            "--model-path",
            MODEL_PATH,
            "--dtype",
            "bfloat16",
            "--host",
            "0.0.0.0",
            "--port",
            str(PORT),
            "--mem-fraction-static",
            "0.7",
            "--max-running-requests",
            "8",
            "--tp-size",
            str(GPU_COUNT),
            "--enable-metrics",
        ]

        # Add speculative decoding parameters if enabled
        if self.speculative_algorithm:
            cmd.extend(
                [
                    "--speculative-algorithm",
                    self.speculative_algorithm,
                ]
            )

            if self.speculative_algorithm.startswith("EAGLE"):
                if self.speculative_draft_model:
                    cmd.extend(
                        [
                            "--speculative-draft-model-path",
                            self.speculative_draft_model,
                        ]
                    )
                cmd.extend(
                    [
                        "--speculative-num-steps",
                        str(self.speculative_num_steps),
                        "--speculative-eagle-topk",
                        str(self.speculative_eagle_topk),
                        "--speculative-num-draft-tokens",
                        str(self.speculative_num_draft_tokens),
                    ]
                )
                print(
                    f"EAGLE config: steps={self.speculative_num_steps}, topk={self.speculative_eagle_topk}, draft_tokens={self.speculative_num_draft_tokens}"
                )
            elif self.speculative_algorithm == "NGRAM":
                cmd.extend(
                    [
                        "--speculative-num-steps",
                        str(self.speculative_num_steps),
                        "--speculative-ngram-min-match-window-size",
                        str(self.speculative_ngram_min_match_window_size),
                        "--speculative-ngram-max-match-window-size",
                        str(self.speculative_ngram_max_match_window_size),
                        "--speculative-ngram-branch-length",
                        str(self.speculative_ngram_branch_length),
                    ]
                )
                print(
                    f"Ngram config: steps={self.speculative_num_steps}, min_window={self.speculative_ngram_min_match_window_size}, max_window={self.speculative_ngram_max_match_window_size}, branch_length={self.speculative_ngram_branch_length}"
                )

        print(f"cmd: {cmd}")
        return subprocess.Popen(" ".join(cmd), shell=True, env=env)

    @modal.enter()
    def enter(self):
        self._proc = self.serve()
        wait_ready(self._proc)

    @modal.web_server(port=PORT)
    def method(self):
        return

    @modal.exit()
    def exit(self):
        self._proc.terminate()


app = modal.App(APP_NAME)


@app.cls(
    image=SGLANG_IMAGE,
    gpu=f"{GPU_TYPE}:{GPU_COUNT}",
    volumes={
        # "/root/.cache/flashinfer": FLASHINFER_CACHE_VOL,
        "/root/.cache/huggingface": HF_CACHE_VOL,
    },
    region="us-east",
    experimental_options={"input_plane_region": "us-east"},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    min_containers=1,
    startup_timeout=30 * MINUTES,
)
@modal.concurrent(max_inputs=10, target_inputs=8)
class ModelSGLang(ModelSGLangBase):
    # Vanilla configuration - no speculative decoding
    speculative_algorithm = ""


# EAGLE3 variant - Conservative
@app.cls(
    image=SGLANG_IMAGE,
    gpu=f"{GPU_TYPE}:{GPU_COUNT}",
    volumes={
        # "/root/.cache/flashinfer": FLASHINFER_CACHE_VOL,
        "/root/.cache/huggingface": HF_CACHE_VOL,
    },
    region="us-east",
    experimental_options={"input_plane_region": "us-east"},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    min_containers=1,
    startup_timeout=30 * MINUTES,
)
@modal.concurrent(max_inputs=10, target_inputs=8)
class ModelSGLangEagle3Small(ModelSGLangBase):
    speculative_algorithm = "EAGLE3"
    speculative_draft_model = DRAFT_MODEL_PATH
    speculative_num_steps = 3
    speculative_eagle_topk = 4
    speculative_num_draft_tokens = 16


# EAGLE3 variant - Balanced
@app.cls(
    image=SGLANG_IMAGE,
    gpu=f"{GPU_TYPE}:{GPU_COUNT}",
    volumes={
        # "/root/.cache/flashinfer": FLASHINFER_CACHE_VOL,
        "/root/.cache/huggingface": HF_CACHE_VOL,
    },
    region="us-east",
    experimental_options={"input_plane_region": "us-east"},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    min_containers=1,
    startup_timeout=30 * MINUTES,
)
@modal.concurrent(max_inputs=10, target_inputs=8)
class ModelSGLangEagle3Medium(ModelSGLangBase):
    speculative_algorithm = "EAGLE3"
    speculative_draft_model = DRAFT_MODEL_PATH
    speculative_num_steps = 5
    speculative_eagle_topk = 8
    speculative_num_draft_tokens = 32


# EAGLE3 variant - Aggressive
@app.cls(
    image=SGLANG_IMAGE,
    gpu=f"{GPU_TYPE}:{GPU_COUNT}",
    volumes={
        # "/root/.cache/flashinfer": FLASHINFER_CACHE_VOL,
        "/root/.cache/huggingface": HF_CACHE_VOL,
    },
    region="us-east",
    experimental_options={"input_plane_region": "us-east"},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    min_containers=1,
    startup_timeout=30 * MINUTES,
)
@modal.concurrent(max_inputs=10, target_inputs=8)
class ModelSGLangEagle3Large(ModelSGLangBase):
    speculative_algorithm = "EAGLE3"
    speculative_draft_model = DRAFT_MODEL_PATH
    speculative_num_steps = 7
    speculative_eagle_topk = 8
    speculative_num_draft_tokens = 64


# Ngram variant - Small
@app.cls(
    image=SGLANG_IMAGE,
    gpu=f"{GPU_TYPE}:{GPU_COUNT}",
    volumes={
        # "/root/.cache/flashinfer": FLASHINFER_CACHE_VOL,
        "/root/.cache/huggingface": HF_CACHE_VOL,
    },
    region="us-east",
    experimental_options={"input_plane_region": "us-east"},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    min_containers=1,
    startup_timeout=30 * MINUTES,
)
@modal.concurrent(max_inputs=10, target_inputs=8)
class ModelSGLangNgramSmaller(ModelSGLangBase):
    speculative_algorithm = "NGRAM"
    speculative_num_steps = 1  # Single deep branch
    speculative_ngram_min_match_window_size = 3  # Avoid false matches
    speculative_ngram_max_match_window_size = 15  # Must be < branch_length
    speculative_ngram_branch_length = 20  # All tokens in one shot


# Ngram variant - Small
@app.cls(
    image=SGLANG_IMAGE,
    gpu=f"{GPU_TYPE}:{GPU_COUNT}",
    volumes={
        # "/root/.cache/flashinfer": FLASHINFER_CACHE_VOL,
        "/root/.cache/huggingface": HF_CACHE_VOL,
    },
    region="us-east",
    experimental_options={"input_plane_region": "us-east"},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    min_containers=1,
    startup_timeout=30 * MINUTES,
)
@modal.concurrent(max_inputs=10, target_inputs=8)
class ModelSGLangNgramSmall(ModelSGLangBase):
    speculative_algorithm = "NGRAM"
    speculative_num_steps = 2  # Two deep branches
    speculative_ngram_min_match_window_size = 2  # Avoid single-token matches
    speculative_ngram_max_match_window_size = 18  # Large lookback for JSX
    speculative_ngram_branch_length = 24  # 12 tokens per branch


# Ngram variant - Medium
@app.cls(
    image=SGLANG_IMAGE,
    gpu=f"{GPU_TYPE}:{GPU_COUNT}",
    volumes={
        # "/root/.cache/flashinfer": FLASHINFER_CACHE_VOL,
        "/root/.cache/huggingface": HF_CACHE_VOL,
    },
    region="us-east",
    experimental_options={"input_plane_region": "us-east"},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    min_containers=1,
    startup_timeout=30 * MINUTES,
)
@modal.concurrent(max_inputs=10, target_inputs=8)
class ModelSGLangNgramMedium(ModelSGLangBase):
    speculative_algorithm = "NGRAM"
    speculative_num_steps = 3  # Fewer steps for deeper speculation
    speculative_ngram_min_match_window_size = 2  # Better precision
    speculative_ngram_max_match_window_size = 20  # Larger pattern matching
    speculative_ngram_branch_length = 30  # 10 tokens per step


# Ngram variant - Large (original)
@app.cls(
    image=SGLANG_IMAGE,
    gpu=f"{GPU_TYPE}:{GPU_COUNT}",
    volumes={
        # "/root/.cache/flashinfer": FLASHINFER_CACHE_VOL,
        "/root/.cache/huggingface": HF_CACHE_VOL,
    },
    region="us-east",
    experimental_options={"input_plane_region": "us-east"},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    min_containers=1,
    startup_timeout=30 * MINUTES,
)
@modal.concurrent(max_inputs=10, target_inputs=8)
class ModelSGLangNgramLarge(ModelSGLangBase):
    speculative_algorithm = "NGRAM"
    speculative_num_steps = 4  # Balanced steps
    speculative_ngram_min_match_window_size = 3  # Higher confidence
    speculative_ngram_max_match_window_size = 28  # Large lookback with margin
    speculative_ngram_branch_length = 36  # 9 tokens per step


# Ngram variant - XLarge Window (much larger windows for JSX patterns)
@app.cls(
    image=SGLANG_IMAGE,
    gpu=f"{GPU_TYPE}:{GPU_COUNT}",
    volumes={
        # "/root/.cache/flashinfer": FLASHINFER_CACHE_VOL,
        "/root/.cache/huggingface": HF_CACHE_VOL,
    },
    region="us-east",
    experimental_options={"input_plane_region": "us-east"},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    min_containers=1,
    startup_timeout=30 * MINUTES,
)
@modal.concurrent(max_inputs=10, target_inputs=8)
class ModelSGLangNgramXLarge(ModelSGLangBase):
    speculative_algorithm = "NGRAM"
    speculative_num_steps = 3  # Fewer steps, much deeper
    speculative_ngram_min_match_window_size = 4  # High confidence only
    speculative_ngram_max_match_window_size = 35  # Very large window
    speculative_ngram_branch_length = 45  # 15 tokens per step!


# Ngram variant - JSX Optimized (tuned for JSX structure)
@app.cls(
    image=SGLANG_IMAGE,
    gpu=f"{GPU_TYPE}:{GPU_COUNT}",
    volumes={
        # "/root/.cache/flashinfer": FLASHINFER_CACHE_VOL,
        "/root/.cache/huggingface": HF_CACHE_VOL,
    },
    region="us-east",
    experimental_options={"input_plane_region": "us-east"},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    min_containers=1,
    startup_timeout=30 * MINUTES,
)
@modal.concurrent(max_inputs=10, target_inputs=8)
class ModelSGLangNgramJSX(ModelSGLangBase):
    speculative_algorithm = "NGRAM"
    speculative_num_steps = 2  # Two very deep branches
    speculative_ngram_min_match_window_size = 5  # JSX tokens are longer
    speculative_ngram_max_match_window_size = (
        35  # Capture full elements (must be < branch)
    )
    speculative_ngram_branch_length = 45  # 22.5 tokens per branch!


# Ngram variant - Aggressive (maximum parameters)
@app.cls(
    image=SGLANG_IMAGE,
    gpu=f"{GPU_TYPE}:{GPU_COUNT}",
    volumes={
        # "/root/.cache/flashinfer": FLASHINFER_CACHE_VOL,
        "/root/.cache/huggingface": HF_CACHE_VOL,
    },
    region="us-east",
    experimental_options={"input_plane_region": "us-east"},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    min_containers=1,
    startup_timeout=30 * MINUTES,
)
@modal.concurrent(max_inputs=10, target_inputs=8)
class ModelSGLangNgramAggressive(ModelSGLangBase):
    speculative_algorithm = "NGRAM"
    speculative_num_steps = 1  # Single very aggressive branch
    speculative_ngram_min_match_window_size = 4  # High confidence
    speculative_ngram_max_match_window_size = 40  # Must be < branch_length
    speculative_ngram_branch_length = 50  # Go big or go home


# Ngram variant - Medium Baseline (for comparison)
@app.cls(
    image=SGLANG_IMAGE,
    gpu=f"{GPU_TYPE}:{GPU_COUNT}",
    volumes={
        # "/root/.cache/flashinfer": FLASHINFER_CACHE_VOL,
        "/root/.cache/huggingface": HF_CACHE_VOL,
    },
    region="us-east",
    experimental_options={"input_plane_region": "us-east"},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    min_containers=1,
    startup_timeout=30 * MINUTES,
)
@modal.concurrent(max_inputs=10, target_inputs=8)
class ModelSGLangNgramMediumBaseline(ModelSGLangBase):
    speculative_algorithm = "NGRAM"
    speculative_num_steps = 3  # Same as Medium
    speculative_ngram_min_match_window_size = 2
    speculative_ngram_max_match_window_size = 20
    speculative_ngram_branch_length = 30
    speculative_baseline = True  # SIMULATE_ACC_LEN=1


# Ngram variant - Balanced Baseline (for comparison)
@app.cls(
    image=SGLANG_IMAGE,
    gpu=f"{GPU_TYPE}:{GPU_COUNT}",
    volumes={
        # "/root/.cache/flashinfer": FLASHINFER_CACHE_VOL,
        "/root/.cache/huggingface": HF_CACHE_VOL,
    },
    region="us-east",
    experimental_options={"input_plane_region": "us-east"},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    min_containers=1,
    startup_timeout=30 * MINUTES,
)
@modal.concurrent(max_inputs=10, target_inputs=8)
class ModelSGLangNgramBalancedBaseline(ModelSGLangBase):
    speculative_algorithm = "NGRAM"
    speculative_num_steps = 4  # Same as Balanced
    speculative_ngram_min_match_window_size = 3
    speculative_ngram_max_match_window_size = 25
    speculative_ngram_branch_length = 32
    speculative_baseline = True  # SIMULATE_ACC_LEN=1


# Ngram variant - Balanced Plus (moderate everything, slightly higher)
@app.cls(
    image=SGLANG_IMAGE,
    gpu=f"{GPU_TYPE}:{GPU_COUNT}",
    volumes={
        # "/root/.cache/flashinfer": FLASHINFER_CACHE_VOL,
        "/root/.cache/huggingface": HF_CACHE_VOL,
    },
    region="us-east",
    experimental_options={"input_plane_region": "us-east"},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    min_containers=1,
    startup_timeout=30 * MINUTES,
)
@modal.concurrent(max_inputs=10, target_inputs=8)
class ModelSGLangNgramBalanced(ModelSGLangBase):
    speculative_algorithm = "NGRAM"
    speculative_num_steps = 4  # Balanced step count
    speculative_ngram_min_match_window_size = 3  # Better precision
    speculative_ngram_max_match_window_size = 25  # Good window with safe margin
    speculative_ngram_branch_length = 32  # 8 tokens per step


# This part runs locally as part of `modal run` to test the model is configured correctly, behind a temporary dev endpoint.
# To run evals, use `modal deploy` to create a persistent endpoint (e.g. for LangSmith evals).
@app.local_entrypoint()
def main(variant: str = "vanilla"):
    import subprocess

    # Choose which variant to test
    if variant == "vanilla":
        model_cls = ModelSGLang
    # elif variant == "eagle3-small":
    #     model_cls = ModelSGLangEagle3Small
    # elif variant == "eagle3-medium":
    #     model_cls = ModelSGLangEagle3Medium
    # elif variant == "eagle3-large":
    #     model_cls = ModelSGLangEagle3Large
    elif variant == "ngram-small":
        model_cls = ModelSGLangNgramSmall
    elif variant == "ngram-medium":
        model_cls = ModelSGLangNgramMedium
    elif variant == "ngram-large":
        model_cls = ModelSGLangNgramLarge
    elif variant == "ngram-xlarge":
        model_cls = ModelSGLangNgramXLarge
    elif variant == "ngram-jsx":
        model_cls = ModelSGLangNgramJSX
    elif variant == "ngram-aggressive":
        model_cls = ModelSGLangNgramAggressive
    elif variant == "ngram-medium-baseline":
        model_cls = ModelSGLangNgramMediumBaseline
    elif variant == "ngram-balanced-baseline":
        model_cls = ModelSGLangNgramBalancedBaseline
    elif variant == "ngram-balanced":
        model_cls = ModelSGLangNgramBalanced
    elif variant == "ngram-medium-baseline":
        model_cls = ModelSGLangNgramMediumBaseline
    elif variant == "ngram-balanced-baseline":
        model_cls = ModelSGLangNgramBalancedBaseline
    else:
        raise ValueError(f"Unknown variant: {variant}")

    # Grab the temporary dev endpoint URL.
    url = model_cls().method.web_url
    print(f"Testing {variant} model at {url}")
    subprocess.run(
        f'curl -X POST {url}/v1/chat/completions -d \'{{"messages": [{{"role": "user", "content": "Hello, how are you?"}}]}}\' -H \'Content-Type: application/json\'',
        shell=True,
        check=True,
    )
    print("Test successful")
