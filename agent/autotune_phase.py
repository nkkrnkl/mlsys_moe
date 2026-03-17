"""
Automated tile-size grid search (no LLM needed).

Generates configurations by substituting constants into the kernel source,
packs each as a Solution, and benchmarks on Modal. Returns the best config.

Run every N iterations to extract the last few percent without LLM overhead.
"""

from __future__ import annotations

import itertools
import re
from pathlib import Path
from typing import Any

# Grid to search
AUTOTUNE_GRID = {
    "BLOCK_M": [64, 128],
    "BLOCK_N": [128, 256],
    "BLOCK_K": [128],       # Fixed at 128 for FP8 scale alignment
    "num_warps": [8, 16],
    "num_stages": [3, 4, 5],
}

# B200 shared memory limit: 228KB = 233,472 bytes
SMEM_LIMIT_BYTES = 228 * 1024

# FP8 = 1 byte, FP32 = 4 bytes
FP8_BYTES = 1
FP32_BYTES = 4


def _estimate_smem(bm: int, bn: int, bk: int, ns: int) -> int:
    """
    Estimate shared memory usage for a tiled GEMM kernel.
    Double-buffer: 2 stages × (A tile + B tile).
    """
    a_tile = bm * bk * FP8_BYTES      # input hidden tile
    b_tile = bn * bk * FP8_BYTES      # weight tile
    return ns * (a_tile + b_tile)


def _is_valid_config(bm: int, bn: int, bk: int, nw: int, ns: int) -> bool:
    """Filter configs that violate B200 constraints."""
    # B200 tensor core minimum
    if bm < 64 or bn < 64:
        return False
    # TMA pipeline
    if ns < 3:
        return False
    # Shared memory
    if _estimate_smem(bm, bn, bk, ns) > SMEM_LIMIT_BYTES:
        return False
    return True


def _heuristic_score(bm: int, bn: int, bk: int, nw: int, ns: int, is_memory_bound: bool) -> float:
    """
    Score a config without benchmarking. Higher is more likely to be good.
    Memory-bound: prefer large BLOCK_M (more tokens per weight load).
    Compute-bound: prefer large BLOCK_N (more output per fetch).
    """
    base = float(bm * bn * bk)
    stage_bonus = ns * 0.5
    if is_memory_bound:
        return base * bm / 64 + stage_bonus
    else:
        return base * bn / 128 + stage_bonus


def generate_candidate_configs(
    is_memory_bound: bool = True,
    top_n: int = 16,
) -> list[dict]:
    """
    Generate and rank all valid tile configurations.

    Returns top_n configs sorted by heuristic score (best first).
    """
    all_combos = list(itertools.product(
        AUTOTUNE_GRID["BLOCK_M"],
        AUTOTUNE_GRID["BLOCK_N"],
        AUTOTUNE_GRID["BLOCK_K"],
        AUTOTUNE_GRID["num_warps"],
        AUTOTUNE_GRID["num_stages"],
    ))

    valid = [
        (bm, bn, bk, nw, ns)
        for bm, bn, bk, nw, ns in all_combos
        if _is_valid_config(bm, bn, bk, nw, ns)
    ]

    scored = sorted(
        valid,
        key=lambda t: _heuristic_score(*t, is_memory_bound),
        reverse=True,
    )

    return [
        {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "num_warps": nw, "num_stages": ns}
        for bm, bn, bk, nw, ns in scored[:top_n]
    ]


def inject_config(kernel_code: str, config: dict) -> str:
    """
    Substitute tile-size constants into the kernel source code.

    Replaces the _gemm_configs() function with a single hardcoded config.
    """
    bm = config["BLOCK_M"]
    bn = config["BLOCK_N"]
    bk = config["BLOCK_K"]
    nw = config["num_warps"]
    ns = config["num_stages"]

    new_configs_fn = f'''def _gemm_configs():
    return [
        triton.Config(
            {{"BLOCK_M": {bm}, "BLOCK_N": {bn}, "BLOCK_K": {bk}}},
            num_stages={ns},
            num_warps={nw},
        )
    ]
'''
    # Replace the _gemm_configs function
    patched = re.sub(
        r"def _gemm_configs\(\):.*?(?=\ndef |\Z)",
        new_configs_fn,
        kernel_code,
        flags=re.DOTALL,
    )
    return patched


def run_autotune_phase(
    best_kernel_code: str,
    benchmark_fn,          # callable: (kernel_code: str) → dict with latency_ms, speedup_factor
    is_memory_bound: bool = True,
    top_n: int = 16,
    verbose: bool = True,
) -> tuple[str, dict, dict]:
    """
    Run grid search over tile sizes and return the best configuration.

    Args:
        best_kernel_code: Current best kernel source code.
        benchmark_fn: Function that takes kernel code and returns benchmark result dict.
        is_memory_bound: Used to prioritize configs (from roofline analysis).
        top_n: Number of configs to benchmark.

    Returns:
        (best_code, best_config, best_result) tuple.
    """
    configs = generate_candidate_configs(is_memory_bound=is_memory_bound, top_n=top_n)
    if verbose:
        print(f"[autotune] Testing {len(configs)} tile configurations...")

    best_code = best_kernel_code
    best_config: dict = {}
    best_result: dict = {"speedup_factor": 0.0}

    for i, config in enumerate(configs):
        if verbose:
            print(f"  [{i+1}/{len(configs)}] Config: {config}", end=" ... ")
        try:
            patched = inject_config(best_kernel_code, config)
            result = benchmark_fn(patched)
            sf = result.get("speedup_factor", 0.0)
            if verbose:
                print(f"{sf:.3f}x")
            if sf > best_result.get("speedup_factor", 0.0):
                best_result = result
                best_config = config
                best_code = patched
        except Exception as e:
            if verbose:
                print(f"FAILED: {e}")
            continue

    if verbose and best_config:
        print(f"[autotune] Best config: {best_config} → {best_result.get('speedup_factor', 0):.3f}x")

    return best_code, best_config, best_result
