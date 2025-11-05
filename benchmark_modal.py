#!/usr/bin/env python
"""
Modal-based parallel benchmark runner.
Runs benchmarks across all deployments in parallel using Modal.
"""

import modal
import json
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
import statistics

app = modal.App("benchmark-runner")

# Create volume for results
RESULTS_VOL = modal.Volume.from_name("benchmark-results", create_if_missing=True)


@app.function(
    image=modal.Image.debian_slim(python_version="3.11").pip_install("httpx", "pandas"),
    volumes={"/results": RESULTS_VOL},
    timeout=60 * 20,
    region="us-east",
)
def benchmark_single_deployment(
    deployment_class_name: str,
    engine: str,
    endpoint_url: str,
    prompts: List[Dict[str, Any]],
    max_tokens: int = 256,
) -> Dict[str, Any]:
    """
    Run benchmark for a single deployment.
    """
    import httpx
    import asyncio
    import time
    import json

    async def measure_request(prompt_data):
        """Measure a single request with streaming."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            if isinstance(prompt_data, str):
                messages = [{"role": "user", "content": prompt_data}]
            elif isinstance(prompt_data, list):
                messages = prompt_data
            else:
                messages = prompt_data.get(
                    "messages", [{"role": "user", "content": str(prompt_data)}]
                )

            request_start = time.perf_counter()
            chunk_times = []  # arrival times of streamed chunks with content
            first_token_time = None
            last_token_time = None
            output_text = []  # collect streamed content pieces
            usage = None  # will be filled if server sends usage in final chunk
            finish_reason = None
            extra_metrics = {}

            try:
                async with client.stream(
                    "POST",
                    f"{endpoint_url}/v1/chat/completions",
                    json={
                        "model": "auto",
                        "messages": messages,
                        "max_tokens": max_tokens,
                        "temperature": 0,
                        "stream": True,
                        # Ask server to include usage in the final stream chunk (OpenAI-compatible)
                        "stream_options": {"include_usage": True},
                    },
                ) as response:
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            chunk = line[6:]
                            if chunk == "[DONE]":
                                break

                            try:
                                data = json.loads(chunk)
                                # Capture any usage provided in the final chunk
                                if "usage" in data and data["usage"]:
                                    usage = data["usage"]

                                # Best-effort capture of finish_reason and other metadata
                                if "choices" in data and data["choices"]:
                                    choice0 = data["choices"][0]
                                    finish_reason = choice0.get(
                                        "finish_reason", finish_reason
                                    )
                                    content = choice0.get("delta", {}).get(
                                        "content", ""
                                    )
                                    if content:
                                        current_time = time.perf_counter()
                                        if first_token_time is None:
                                            first_token_time = current_time
                                        last_token_time = current_time
                                        chunk_times.append(current_time)
                                        output_text.append(content)

                                # Heuristically collect SGLang-specific speculative metrics if present
                                # Some SGLang builds emit fields like "metrics", "sgl_metadata", or keys containing "spec"/"ngram"
                                for k, v in list(data.items()):
                                    if k in {
                                        "id",
                                        "object",
                                        "created",
                                        "model",
                                        "choices",
                                        "usage",
                                    }:
                                        continue
                                    # Only keep reasonably small dicts/numbers/strings
                                    if isinstance(v, (dict, list, str, int, float)):
                                        extra_metrics[k] = v
                            except json.JSONDecodeError:
                                continue

                e2e_time = time.perf_counter() - request_start

                # Calculate inter-chunk latency (proxy for ITL if chunk==token)
                if len(chunk_times) > 1:
                    itl_values = [
                        chunk_times[i] - chunk_times[i - 1]
                        for i in range(1, len(chunk_times))
                    ]
                    mean_itl = statistics.mean(itl_values) * 1000  # ms
                    median_itl = statistics.median(itl_values) * 1000
                    p95_itl = (
                        statistics.quantiles(itl_values, n=20)[18] * 1000
                        if len(itl_values) > 2
                        else max(itl_values) * 1000
                    )
                    p99_itl = (
                        statistics.quantiles(itl_values, n=100)[98] * 1000
                        if len(itl_values) > 2
                        else max(itl_values) * 1000
                    )
                else:
                    mean_itl = median_itl = p95_itl = p99_itl = 0

                ttft = (
                    (first_token_time - request_start) * 1000 if first_token_time else 0
                )

                # Prefer token counts from server usage when available
                completion_tokens = None
                prompt_tokens = None
                total_tokens = None
                if isinstance(usage, dict):
                    completion_tokens = usage.get("completion_tokens")
                    prompt_tokens = usage.get("prompt_tokens")
                    total_tokens = usage.get("total_tokens")

                # Derive token counts and TPS metrics
                # e2e_tps: tokens per second over whole request
                # decode_tps: tokens per second between first and last token (when observable)
                if completion_tokens is not None:
                    e2e_tps = (completion_tokens / e2e_time) if e2e_time > 0 else 0
                else:
                    # Fallback: chunk count as a proxy (imperfect)
                    e2e_tps = (len(chunk_times) / e2e_time) if e2e_time > 0 else 0

                if (
                    first_token_time
                    and last_token_time
                    and last_token_time > first_token_time
                ):
                    decode_duration = last_token_time - first_token_time
                    if completion_tokens is not None and decode_duration > 0:
                        decode_tps = completion_tokens / decode_duration
                    else:
                        decode_tps = (
                            (len(chunk_times) / decode_duration)
                            if decode_duration > 0
                            else 0
                        )
                else:
                    decode_tps = e2e_tps

                return {
                    "success": True,
                    "ttft": ttft,
                    "e2e_latency": e2e_time,
                    "mean_itl": mean_itl,
                    "median_itl": median_itl,
                    "p95_itl": p95_itl,
                    "p99_itl": p99_itl,
                    # Prefer usage counts; fall back to chunk count
                    "completion_tokens": completion_tokens,
                    "prompt_tokens": prompt_tokens,
                    "total_tokens": total_tokens,
                    "num_chunks": len(chunk_times),
                    "e2e_tps": e2e_tps,
                    "decode_tps": decode_tps,
                    "finish_reason": finish_reason,
                    "usage": usage,
                    "extra": extra_metrics,
                    # For spot-checking content
                    "messages": messages,
                    "output_text": "".join(output_text),
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "e2e_latency": time.perf_counter() - request_start,
                }

    print(f"Benchmarking {engine}/{deployment_class_name} at {endpoint_url}")

    # Run warmup
    asyncio.run(measure_request("Hello"))

    # Run benchmarks
    results = []
    for i, prompt in enumerate(prompts):
        print(f"  Testing prompt {i + 1}/{len(prompts)}...")
        result = asyncio.run(measure_request(prompt))
        results.append(result)
        time.sleep(0.2)  # Brief pause between requests

    # Calculate aggregates
    successful = [r for r in results if r.get("success")]

    if successful:
        aggregate = {
            "deployment": deployment_class_name,
            "engine": engine,
            "endpoint": endpoint_url,
            "num_prompts": len(prompts),
            "num_successful": len(successful),
            "avg_ttft": statistics.mean([r["ttft"] for r in successful]),
            "avg_e2e": statistics.mean([r["e2e_latency"] for r in successful]),
            "avg_mean_itl": statistics.mean([r["mean_itl"] for r in successful]),
            "avg_median_itl": statistics.mean([r["median_itl"] for r in successful]),
            "avg_p95_itl": statistics.mean([r["p95_itl"] for r in successful]),
            "avg_p99_itl": statistics.mean([r["p99_itl"] for r in successful]),
            # Prefer server token counts where available
            "avg_completion_tokens": statistics.mean(
                [
                    r["completion_tokens"]
                    for r in successful
                    if r.get("completion_tokens") is not None
                ]
            )
            if any(r.get("completion_tokens") is not None for r in successful)
            else None,
            "avg_prompt_tokens": statistics.mean(
                [
                    r["prompt_tokens"]
                    for r in successful
                    if r.get("prompt_tokens") is not None
                ]
            )
            if any(r.get("prompt_tokens") is not None for r in successful)
            else None,
            "avg_total_tokens": statistics.mean(
                [
                    r["total_tokens"]
                    for r in successful
                    if r.get("total_tokens") is not None
                ]
            )
            if any(r.get("total_tokens") is not None for r in successful)
            else None,
            # Also record chunk-based proxy counts for reference
            "avg_num_chunks": statistics.mean([r["num_chunks"] for r in successful]),
            "avg_e2e_tps": statistics.mean([r["e2e_tps"] for r in successful]),
            "avg_decode_tps": statistics.mean([r["decode_tps"] for r in successful]),
            "individual_results": results,
        }
    else:
        aggregate = {
            "deployment": deployment_class_name,
            "engine": engine,
            "endpoint": endpoint_url,
            "num_prompts": len(prompts),
            "num_successful": 0,
            "error": "All requests failed",
            "individual_results": results,
        }

    # Save to volume
    timestamp = int(time.time())
    filename = f"/results/{engine}_{deployment_class_name}_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(aggregate, f, indent=2)

    return aggregate


@app.local_entrypoint()
def main(
    engine: str = "all",
    prompts_file: Optional[str] = None,
    max_tokens: int = 256,
):
    """
    Run benchmarks in parallel across all deployments.

    Args:
        engine: Which engine to test ("sglang", "trtllm", or "all")
        prompts_file: Path to prompts file
        max_tokens: Max tokens to generate
        app_names: Mapping of engine to deployed Modal app name
    """
    import pandas as pd
    from datetime import datetime

    # Load prompts
    if prompts_file and Path(prompts_file).exists():
        with open(prompts_file) as f:
            if prompts_file.endswith(".json"):
                prompts = json.load(f)
            else:
                prompts = [line.strip() for line in f if line.strip()]
    else:
        prompts = [
            "Write a Python function to calculate factorial.",
            "Explain quantum computing in simple terms.",
            "What are the key principles of functional programming?",
            "Write a haiku about artificial intelligence.",
            "Describe the process of photosynthesis.",
        ]

    print(f"Testing with {len(prompts)} prompts, max_tokens={max_tokens}")

    # Define deployments to test
    deployments = []

    if engine in ["all", "trtllm"]:
        # You'll need to add TRT-LLM URLs when they're deployed
        trtllm_urls = {
            # Add your TRT-LLM endpoint URLs here when deployed
            # "ModelTRTLLM": "https://...",
            # "ModelTRTLLMLookaheadSmall": "https://...",
            # etc.
        }
        for name, url in trtllm_urls.items():
            deployments.append(("trtllm", name, url))

    if engine in ["all", "sglang"]:
        # Map the actual deployed URLs based on your output
        sglang_urls = {
            "ModelSGLang": "https://modal-labs-jason-dev--sglang-llama3-3-70b-modelsglang-method.us-east.modal.run",
            "ModelSGLangEagle3Small": "https://modal-labs-jason-dev--sglang-llama3-3-70b-modelsglangeag-f6795d.us-east.modal.run",
            "ModelSGLangEagle3Medium": "https://modal-labs-jason-dev--sglang-llama3-3-70b-modelsglangeag-8e6780.us-east.modal.run",
            "ModelSGLangEagle3Large": "https://modal-labs-jason-dev--sglang-llama3-3-70b-modelsglangeag-36d8b3.us-east.modal.run",
            "ModelSGLangNgramSmaller": "https://modal-labs-jason-dev--sglang-llama3-3-70b-modelsglangngr-fce5d0.us-east.modal.run",
            "ModelSGLangNgramSmall": "https://modal-labs-jason-dev--sglang-llama3-3-70b-modelsglangngr-44262d.us-east.modal.run",
            "ModelSGLangNgramMedium": "https://modal-labs-jason-dev--sglang-llama3-3-70b-modelsglangngr-1f1bc4.us-east.modal.run",
            "ModelSGLangNgramLarge": "https://modal-labs-jason-dev--sglang-llama3-3-70b-modelsglangngr-913ec4.us-east.modal.run",
            "ModelSGLangNgramXLarge": "https://modal-labs-jason-dev--sglang-llama3-3-70b-modelsglangngr-094d83.us-east.modal.run",
            "ModelSGLangNgramJSX": "https://modal-labs-jason-dev--sglang-llama3-3-70b-modelsglangngr-f9ec45.us-east.modal.run",
            "ModelSGLangNgramAggressive": "https://modal-labs-jason-dev--sglang-llama3-3-70b-modelsglangngr-24f9cd.us-east.modal.run",
            "ModelSGLangNgramBalanced": "https://modal-labs-jason-dev--sglang-llama3-3-70b-modelsglangngr-b9634b.us-east.modal.run",
            "ModelSGLangNgramMediumBaseline": "https://modal-labs-jason-dev--sglang-llama3-3-70b-modelsglangngr-891ff9.us-east.modal.run",
            "ModelSGLangNgramBalancedBaseline": "https://modal-labs-jason-dev--sglang-llama3-3-70b-modelsglangngr-3ad626.us-east.modal.run",
        }
    for name, url in sglang_urls.items():
        deployments.append(("sglang", name, url))

    print(f"Will benchmark {len(deployments)} deployments in parallel")

    # Run benchmarks in parallel using Modal's map
    print("\nRunning benchmarks in parallel...")
    results = list(
        benchmark_single_deployment.map(
            [d[1] for d in deployments],  # deployment class names
            [d[0] for d in deployments],  # engines
            [d[2] for d in deployments],  # URLs
            [prompts] * len(deployments),  # Same prompts for all
            [max_tokens] * len(deployments),  # Same max_tokens for all
        )
    )

    # Display results
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)

    # Create summary table
    summary_data = []
    for r in results:
        if r.get("num_successful", 0) > 0:
            summary_data.append(
                {
                    "Engine": r["engine"],
                    "Deployment": r["deployment"]
                    .replace("Model", "")
                    .replace("TRTLLM", "")
                    .replace("SGLang", ""),
                    "Success Rate": f"{r['num_successful']}/{r['num_prompts']}",
                    "Prompt Tokens": (
                        f"{r['avg_prompt_tokens']:.1f}"
                        if r.get("avg_prompt_tokens") is not None
                        else "-"
                    ),
                    "Comp Tokens": (
                        f"{r['avg_completion_tokens']:.1f}"
                        if r.get("avg_completion_tokens") is not None
                        else "-"
                    ),
                    "Total Tokens": (
                        f"{r['avg_total_tokens']:.1f}"
                        if r.get("avg_total_tokens") is not None
                        else "-"
                    ),
                    "E2E TPS": f"{r['avg_e2e_tps']:.1f}",
                    "Decode TPS": f"{r['avg_decode_tps']:.1f}",
                    "TTFT (ms)": f"{r['avg_ttft']:.1f}",
                    "Mean ITL (ms)": f"{r['avg_mean_itl']:.1f}",
                    "Median ITL (ms)": f"{r['avg_median_itl']:.1f}",
                    "P95 ITL (ms)": f"{r['avg_p95_itl']:.1f}",
                    "P99 ITL (ms)": f"{r['avg_p99_itl']:.1f}",
                    "E2E (s)": f"{r['avg_e2e']:.2f}",
                }
            )

    if summary_data:
        df = pd.DataFrame(summary_data)
        # Sort by Decode TPS (usage-based throughput)
        if "Decode TPS" in df.columns:
            df = df.sort_values("Decode TPS", ascending=False)
        else:
            df = df.sort_values("Mean ITL (ms)")
        print("\n" + df.to_string(index=False))

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed results
        detailed_file = f"benchmark_detailed_{engine}_{timestamp}.json"
        with open(detailed_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to {detailed_file}")

        # Save summary
        summary_file = f"benchmark_summary_{engine}_{timestamp}.csv"
        df.to_csv(summary_file, index=False)
        print(f"Summary saved to {summary_file}")

        # Save chat transcripts (messages + assistant output) per deployment for spot checks
        from pathlib import Path as _Path
        import json as _json
        import re as _re

        out_dir = _Path(f"benchmark_chats_{engine}_{timestamp}")
        out_dir.mkdir(parents=True, exist_ok=True)
        for agg in results:
            dep_name = (
                agg.get("deployment", "unknown")
                .replace("Model", "")
                .replace("TRTLLM", "")
                .replace("SGLang", "")
            )
            dep_slug = (
                _re.sub(r"[^A-Za-z0-9_\-]+", "_", dep_name).strip("_") or "deployment"
            )
            outfile = out_dir / f"{dep_slug}.jsonl"
            with open(outfile, "w") as f:
                for ind in agg.get("individual_results", []):
                    msgs = ind.get("messages") or []
                    assistant_text = ind.get("output_text", "")
                    chat = list(msgs) + (
                        [{"role": "assistant", "content": assistant_text}]
                        if assistant_text is not None
                        else []
                    )
                    record = {
                        "engine": agg.get("engine"),
                        "deployment": agg.get("deployment"),
                        "endpoint": agg.get("endpoint"),
                        "usage": ind.get("usage"),
                        "finish_reason": ind.get("finish_reason"),
                        "metrics": {
                            "ttft_ms": ind.get("ttft"),
                            "e2e_s": ind.get("e2e_latency"),
                            "decode_tps": ind.get("decode_tps"),
                            "e2e_tps": ind.get("e2e_tps"),
                        },
                        "chat": chat,
                        "extra": ind.get("extra"),
                    }
                    f.write(_json.dumps(record) + "\n")
            print(f"Chats saved to {outfile}")

        # Show best performers
        print("\n" + "=" * 80)
        print("TOP PERFORMERS")
        print("=" * 80)

        print("\nLowest Mean ITL:")
        print(
            df.head(3)[["Engine", "Deployment", "Mean ITL (ms)"]].to_string(index=False)
        )

        print("\nLowest TTFT:")
        df_ttft = df.sort_values("TTFT (ms)")
        print(
            df_ttft.head(3)[["Engine", "Deployment", "TTFT (ms)"]].to_string(
                index=False
            )
        )

        print("\nHighest Decode TPS (Tokens/Second):")
        df_tps = df.sort_values("Decode TPS", ascending=False)
        print(
            df_tps.head(3)[["Engine", "Deployment", "Decode TPS"]].to_string(
                index=False
            )
        )

    else:
        print("\nNo successful benchmarks completed!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run parallel benchmarks using Modal")
    parser.add_argument("--engine", default="all", choices=["sglang", "trtllm", "all"])
    parser.add_argument("--prompts", type=str, help="Path to prompts file")
    parser.add_argument("--max-tokens", type=int, default=256)

    args = parser.parse_args()

    main(
        engine=args.engine,
        prompts_file=args.prompts,
        max_tokens=args.max_tokens,
    )
