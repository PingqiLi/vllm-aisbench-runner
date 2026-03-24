#!/usr/bin/env python3
"""
Perplexity evaluation tool using vLLM offline inference.

Computes PPL on wikitext-2-test via vLLM's prompt_logprobs, matching the
methodology in vllm/tests/models/language/generation_ppl_test/ppl_utils.py.

Deterministic: no sampling involved — uses prompt log probabilities only.

Usage:
    # Evaluate a single model
    python tools/eval_ppl.py --model-path /path/to/model

    # Evaluate quantized model (ResQ / W4A4 / W8A8)
    python tools/eval_ppl.py \
        --model-path /path/to/quantized_model \
        --quantization ascend \
        --enforce-eager

    # Compare quantized vs bf16 baseline (auto-caches bf16 result)
    python tools/eval_ppl.py \
        --model-path /path/to/quantized_model \
        --quantization ascend \
        --enforce-eager \
        --baseline-model-path /path/to/bf16_model

    # Use cached bf16 baseline (skip re-evaluation)
    python tools/eval_ppl.py \
        --model-path /path/to/quantized_model \
        --quantization ascend \
        --enforce-eager \
        --baseline-ppl 7.52

Output:
    Prints PPL to stdout and saves results to JSON in --output-dir.
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch


def load_eval_text(eval_data_path=None):
    """Load evaluation text.

    Args:
        eval_data_path: Path to local data file. Supports:
            - .parquet (HuggingFace parquet export, expects 'text' column)
            - .jsonl (one JSON object per line, expects 'text' key)
            - .txt (plain text, used as-is)
            If None, auto-downloads wikitext-2-test via HuggingFace datasets.

    Returns:
        Concatenated text string.
    """
    if eval_data_path is not None:
        print(f"Loading eval data from: {eval_data_path}")
        path = eval_data_path

        if path.endswith(".parquet"):
            import pandas as pd
            df = pd.read_parquet(path)
            return "\n\n".join(df["text"].tolist())

        elif path.endswith(".jsonl"):
            import json
            texts = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    texts.append(obj.get("text", ""))
            return "\n\n".join(texts)

        else:
            # Plain text
            with open(path, "r", encoding="utf-8") as f:
                return f.read()

    # Auto-download
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' library required. Install with: pip install datasets")
        print("Or provide --eval-data-path with a local file (.parquet / .jsonl / .txt)")
        sys.exit(1)

    print("Loading wikitext-2-raw-v1 test split...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    return "\n\n".join(ds["text"])


def compute_ppl_vllm(model_path, text, max_length=1024, tensor_parallel_size=1,
                     quantization=None, gpu_memory_utilization=0.85,
                     trust_remote_code=False):
    """Compute perplexity using vLLM offline LLM with prompt_logprobs.

    This matches the methodology in vllm's ppl_utils.py:
    - Tokenize full text, split into chunks of max_length
    - For each chunk, request prompt_logprobs (no generation, max_tokens=1)
    - PPL = exp(-sum(logprobs) / n_tokens)
    """
    from vllm import LLM, SamplingParams

    print(f"Loading model: {model_path}")
    print(f"  tensor_parallel_size={tensor_parallel_size}")
    print(f"  quantization={quantization}")
    print(f"  enforce_eager=True")
    print(f"  max_model_len={max_length}")

    llm_kwargs = dict(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_length + 1,  # +1 for the generated token (max_tokens=1)
        max_num_seqs=1,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=trust_remote_code,
        enforce_eager=True,  # always eager for PPL eval (stable on NPU, no perf need)
    )
    if quantization:
        llm_kwargs["quantization"] = quantization

    llm = LLM(**llm_kwargs)

    tokenizer = llm.get_tokenizer()
    tokens = tokenizer.encode(text)
    n_total = len(tokens)
    print(f"Total tokens: {n_total}")

    stride = max_length
    chunks = []
    for begin in range(0, n_total, stride):
        end = min(begin + max_length, n_total)
        chunks.append(tokens[begin:end])
    print(f"Split into {len(chunks)} chunks (max_length={max_length}, stride={stride})")

    # prompt_logprobs=0 returns logprob for each prompt token (no top-k alternatives)
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        logprobs=None,
        prompt_logprobs=0,
    )

    # Convert token lists to TokensPrompt dicts
    prompts = [{"prompt_token_ids": chunk} for chunk in chunks]

    print("Running inference...")
    t0 = time.time()
    outputs = llm.generate(
        prompts=prompts,
        sampling_params=sampling_params,
        use_tqdm=True,
    )
    elapsed = time.time() - t0
    print(f"Inference done in {elapsed:.1f}s")

    # Aggregate log probabilities
    nll_sum = 0.0
    n_tokens = 0
    for output in outputs:
        prompt_logprobs = output.prompt_logprobs
        # First token has no logprob (it's the BOS/first token)
        for token_data in prompt_logprobs[1:]:
            if token_data is None:
                continue
            # token_data is a dict {token_id: Logprob}
            # With prompt_logprobs=0, it contains exactly the actual token's logprob
            logprob = list(token_data.values())[0].logprob
            nll_sum -= logprob
            n_tokens += 1

    ppl = math.exp(nll_sum / n_tokens)
    print(f"PPL = {ppl:.4f} (over {n_tokens} tokens)")
    return ppl, n_tokens, elapsed


def model_cache_key(model_path):
    """Generate a short cache key from the model path."""
    # Use basename + hash of full path for uniqueness
    basename = Path(model_path).name
    path_hash = hashlib.md5(os.path.abspath(model_path).encode()).hexdigest()[:8]
    return f"{basename}_{path_hash}"


def load_cached_ppl(cache_dir, cache_key):
    """Load cached PPL result if it exists."""
    cache_file = os.path.join(cache_dir, f"ppl_cache_{cache_key}.json")
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            data = json.load(f)
        print(f"Loaded cached PPL: {data['ppl']:.4f} (from {cache_file})")
        return data
    return None


def save_cached_ppl(cache_dir, cache_key, result):
    """Save PPL result to cache."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"ppl_cache_{cache_key}.json")
    with open(cache_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Cached PPL result to {cache_file}")


def save_result(output_dir, result):
    """Save evaluation result to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(output_dir, f"ppl_result_{timestamp}.json")
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Result saved to {result_file}")
    return result_file


def print_comparison(model_ppl, baseline_ppl, model_path, baseline_path=None):
    """Print formatted PPL comparison."""
    print("\n" + "=" * 60)
    print("PPL Evaluation Result")
    print("=" * 60)
    if baseline_ppl is not None:
        degradation = model_ppl - baseline_ppl
        degradation_pct = (model_ppl / baseline_ppl - 1) * 100
        print(f"  Baseline  ({baseline_path or 'cached'}):")
        print(f"    PPL = {baseline_ppl:.4f}")
        print(f"  Evaluated ({model_path}):")
        print(f"    PPL = {model_ppl:.4f}")
        print(f"  Degradation: {degradation:+.4f} ({degradation_pct:+.2f}%)")
    else:
        print(f"  Model: {model_path}")
        print(f"  PPL = {model_ppl:.4f}")
    print("=" * 60)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate perplexity on wikitext-2-test using vLLM offline inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model configuration
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the model to evaluate")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="Tensor parallel size (default: 1)")
    parser.add_argument("--quantization", type=str, default=None,
                        help="Quantization method (e.g., 'ascend' for ResQ/W4A4/W8A8)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85,
                        help="GPU memory utilization (default: 0.85)")
    parser.add_argument("--max-length", type=int, default=1024,
                        help="Max sequence length per chunk (default: 1024)")
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="Trust remote code for model loading")

    # Dataset
    default_data = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "datasets", "wikitext2_test.parquet")
    parser.add_argument("--eval-data-path", type=str, default=default_data,
                        help="Path to local eval data file (.parquet / .jsonl / .txt). "
                             "Default: datasets/wikitext2_test.parquet")

    # Baseline comparison
    parser.add_argument("--baseline-model-path", type=str, default=None,
                        help="Path to baseline (bf16) model for comparison. "
                             "Result is cached for future runs.")
    parser.add_argument("--baseline-tensor-parallel-size", type=int, default=2,
                        help="Tensor parallel size for baseline model (default: 2). "
                             "BF16 32B models typically need TP>=2 to avoid OOM.")
    parser.add_argument("--baseline-ppl", type=float, default=None,
                        help="Pre-computed baseline PPL (skip baseline evaluation)")

    # Output
    parser.add_argument("--output-dir", type=str, default="outputs/ppl",
                        help="Directory for output results (default: outputs/ppl)")
    parser.add_argument("--cache-dir", type=str, default="outputs/ppl/cache",
                        help="Directory for caching baseline PPL results")

    return parser.parse_args()


def main():
    args = parse_args()

    # Load evaluation text
    if not os.path.exists(args.eval_data_path):
        print(f"ERROR: eval data not found: {args.eval_data_path}")
        print("The default dataset (datasets/wikitext2_test.parquet) should be in the repo.")
        sys.exit(1)
    text = load_eval_text(args.eval_data_path)

    # Evaluate baseline if requested
    baseline_ppl = args.baseline_ppl
    baseline_info = None

    if baseline_ppl is None and args.baseline_model_path:
        # Check cache first
        cache_key = model_cache_key(args.baseline_model_path)
        cached = load_cached_ppl(args.cache_dir, cache_key)

        if cached:
            baseline_ppl = cached["ppl"]
            baseline_info = cached
        else:
            print("\n" + "=" * 60)
            print("Evaluating baseline model...")
            print("=" * 60)
            baseline_tp = args.baseline_tensor_parallel_size
            baseline_ppl, baseline_tokens, baseline_time = compute_ppl_vllm(
                model_path=args.baseline_model_path,
                text=text,
                max_length=args.max_length,
                tensor_parallel_size=baseline_tp,
                gpu_memory_utilization=args.gpu_memory_utilization,
                trust_remote_code=args.trust_remote_code,
            )
            baseline_info = {
                "model_path": os.path.abspath(args.baseline_model_path),
                "ppl": baseline_ppl,
                "n_tokens": baseline_tokens,
                "inference_time_s": baseline_time,
                "max_length": args.max_length,
                "dataset": "wikitext-2-raw-v1/test",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            save_cached_ppl(args.cache_dir, cache_key, baseline_info)

    # Evaluate target model
    print("\n" + "=" * 60)
    print("Evaluating target model...")
    print("=" * 60)
    model_ppl, model_tokens, model_time = compute_ppl_vllm(
        model_path=args.model_path,
        text=text,
        max_length=args.max_length,
        tensor_parallel_size=args.tensor_parallel_size,
        quantization=args.quantization,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=args.trust_remote_code,
    )

    # Print comparison
    print_comparison(model_ppl, baseline_ppl, args.model_path, args.baseline_model_path)

    # Save result
    result = {
        "model_path": os.path.abspath(args.model_path),
        "ppl": model_ppl,
        "n_tokens": model_tokens,
        "inference_time_s": model_time,
        "max_length": args.max_length,
        "quantization": args.quantization,
        "dataset": "wikitext-2-raw-v1/test",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if baseline_ppl is not None:
        result["baseline_ppl"] = baseline_ppl
        result["baseline_model_path"] = (os.path.abspath(args.baseline_model_path)
                                         if args.baseline_model_path else None)
        result["degradation"] = model_ppl - baseline_ppl
        result["degradation_pct"] = (model_ppl / baseline_ppl - 1) * 100

    save_result(args.output_dir, result)


if __name__ == "__main__":
    main()
