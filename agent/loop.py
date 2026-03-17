"""
Core optimization iteration loop.

Orchestrates: NCU profile → Judge → Coder (×3 parallel) → Benchmark → Tree update
Includes autotune phase every N iters and crossbreed every M iters.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .autotune_phase import run_autotune_phase
from .failure_memory import FailureMemory
from .judge import run_judge
from .modal_runner import benchmark_kernel, get_first_workload_uuid, profile_kernel_ncu
from .ncu_parser import parse_ncu_output
from .parallel_explore import generate_variants
from .population import KernelPopulation
from .roofline import compute_roofline
from .tree import KernelNode, KernelTree

STATE_DIR = Path("agent_state")
HISTORY_PATH = STATE_DIR / "history.jsonl"
TREE_PATH = STATE_DIR / "tree.json"
BEST_KERNEL_PATH = STATE_DIR / "best_kernel.py"

# Estimated tokens per expert at p95 workload (used for roofline)
# This is a rough estimate; adjust based on actual trace data.
ESTIMATED_TOKENS_PER_EXPERT = 32


def _load_seed_kernel() -> str:
    """Load the current best kernel from solution/ or agent_state/."""
    if BEST_KERNEL_PATH.exists():
        return BEST_KERNEL_PATH.read_text()
    solution_path = Path("solution/triton/kernel.py")
    if solution_path.exists():
        return solution_path.read_text()
    raise FileNotFoundError("No seed kernel found. Run the baseline first.")


def _save_best_kernel(code: str) -> None:
    STATE_DIR.mkdir(exist_ok=True)
    BEST_KERNEL_PATH.write_text(code)
    # Also update solution/ for pack_solution to pick up
    (Path("solution/triton/kernel.py")).write_text(code)


def _log_iteration(record: dict) -> None:
    STATE_DIR.mkdir(exist_ok=True)
    with HISTORY_PATH.open("a") as f:
        f.write(json.dumps(record) + "\n")


def run_optimization_loop(
    max_iterations: int = 100,
    autotune_every: int = 5,
    crossbreed_every: int = 10,
    backtrack_window: int = 8,
    judge_model: str = "claude-opus-4-6",
    coder_model: str = "claude-sonnet-4-6",
    resume: bool = True,
    verbose: bool = True,
) -> None:
    """
    Main optimization loop.

    Args:
        max_iterations: Maximum number of Judge+Coder iterations.
        autotune_every: Run tile-size grid search every N iterations.
        crossbreed_every: Run population crossbreed every M iterations.
        backtrack_window: Backtrack if no improvement in last N iterations.
        judge_model: Claude model for the Judge agent.
        coder_model: Claude model for the Coder agent.
        resume: If True, load existing tree state and continue.
    """
    STATE_DIR.mkdir(exist_ok=True)

    # ---- Initialize / resume ----
    tree = KernelTree.load_or_create(TREE_PATH) if resume else KernelTree()
    population = KernelPopulation(max_size=5)
    failure_memory = FailureMemory(STATE_DIR / "failures.jsonl")
    workload_uuid = get_first_workload_uuid()

    # Seed the tree with the baseline kernel if empty
    if not tree.nodes:
        seed_code = _load_seed_kernel()
        seed_node = KernelNode(
            id="iter_000_seed",
            parent_id=None,
            kernel_code=seed_code,
            iteration=0,
            variant="seed",
        )
        if verbose:
            print("Benchmarking seed kernel on Modal B200...")
        try:
            result = benchmark_kernel(seed_code)
            seed_node.latency_ms = result.get("latency_ms")
            seed_node.speedup_factor = result.get("speedup_factor", 0.0)
            seed_node.metrics = result
            if verbose:
                print(f"  Seed: {seed_node.speedup_factor:.3f}x speedup, {seed_node.latency_ms:.3f}ms")
        except Exception as e:
            print(f"  Seed benchmark failed: {e}. Proceeding with speedup=0.")
            seed_node.speedup_factor = 0.0
        tree.add_node(seed_node)
        tree.current_node_id = seed_node.id
        population.update(seed_node)

    current_node = tree.get_current_node() or tree.get_best_node()

    # ---- Main loop ----
    for iteration in range(1, max_iterations + 1):
        t_start = time.time()
        if verbose:
            print(f"\n{'='*60}")
            print(f"Iteration {iteration}/{max_iterations}")
            print(tree.summary())

        # ---- 1. NCU profile ----
        if verbose:
            print("  [1/6] Profiling with NCU on Modal...")
        try:
            ncu_text = profile_kernel_ncu(
                current_node.kernel_code,
                workload_uuid=workload_uuid,
            )
            metrics = parse_ncu_output(ncu_text)
        except Exception as e:
            print(f"  NCU profiling failed: {e}. Using empty metrics.")
            metrics = {}
            ncu_text = ""

        # ---- 2. Roofline ----
        roofline = compute_roofline(metrics, ESTIMATED_TOKENS_PER_EXPERT)
        is_memory_bound = roofline["bound"] == "MEMORY_BOUND"
        if verbose:
            print(f"  [2/6] Roofline: {roofline['bound']} "
                  f"(AI={roofline['arithmetic_intensity']:.0f} FLOP/B, "
                  f"ridge={roofline['ridge_point']:.0f})")

        # ---- 3. Judge ----
        if verbose:
            print("  [3/6] Running Judge agent...")
        failure_ctx = failure_memory.get_context_for_prompt(n_recent=5)
        try:
            brief = run_judge(
                metrics=metrics,
                roofline=roofline,
                failure_context=failure_ctx,
                model=judge_model,
            )
            if verbose:
                print(f"         Bottleneck: {brief.get('primary_bottleneck', '')[:80]}")
        except Exception as e:
            print(f"  Judge failed: {e}. Skipping iteration.")
            continue

        # ---- 4. Generate 3 variants in parallel ----
        if verbose:
            print("  [4/6] Generating 3 kernel variants (parallel)...")
        try:
            variants = generate_variants(
                brief=brief,
                current_kernel=current_node.kernel_code,
                model=coder_model,
            )
            if verbose:
                print(f"         Generated {len(variants)} variants: {list(variants.keys())}")
        except Exception as e:
            print(f"  Coder failed: {e}. Skipping iteration.")
            continue

        # ---- 5. Benchmark all variants ----
        if verbose:
            print("  [5/6] Benchmarking variants on Modal...")
        variant_results: dict[str, dict] = {}
        variant_nodes: dict[str, KernelNode] = {}
        for v_name, v_code in variants.items():
            node_id = f"iter_{iteration:03d}_{v_name}"
            node = KernelNode(
                id=node_id,
                parent_id=current_node.id,
                kernel_code=v_code,
                iteration=iteration,
                variant=v_name,
                brief=brief,
            )
            try:
                result = benchmark_kernel(v_code)
                node.latency_ms = result.get("latency_ms")
                node.speedup_factor = result.get("speedup_factor", 0.0)
                node.metrics = result
                variant_results[v_name] = result
                variant_nodes[v_name] = node
                if verbose:
                    status = result.get("status", "?")
                    sf = result.get("speedup_factor", 0)
                    lat = result.get("latency_ms", 0)
                    print(f"         {v_name}: {sf:.3f}x @ {lat:.3f}ms [{status}]")
            except Exception as e:
                print(f"         {v_name}: FAILED ({e})")

        if not variant_results:
            print("  All variants failed. Skipping iteration.")
            continue

        # ---- 6. Select best variant ----
        best_v = max(
            variant_results,
            key=lambda k: variant_results[k].get("speedup_factor", 0),
        )
        best_result = variant_results[best_v]
        best_node = variant_nodes[best_v]
        best_sf = best_result.get("speedup_factor", 0.0)
        current_sf = current_node.speedup_factor or 0.0

        improved = best_sf > current_sf * 1.001  # > 0.1% improvement
        regressed = best_sf < current_sf * 0.98  # > 2% regression

        if verbose:
            print(f"  [6/6] Best variant: {best_v} @ {best_sf:.3f}x "
                  f"({'IMPROVED' if improved else 'REGRESSED' if regressed else 'FLAT'})")

        # Add all benchmarked nodes to tree regardless of outcome
        for node in variant_nodes.values():
            tree.add_node(node)

        if improved:
            current_node = best_node
            tree.current_node_id = best_node.id
            population.update(best_node)
            _save_best_kernel(best_node.kernel_code)
            print(f"  ✓ New best: {best_sf:.3f}x speedup ({best_node.latency_ms:.3f}ms)")
        elif regressed:
            failure_memory.log(
                iteration=iteration,
                parent_code=current_node.kernel_code,
                child_code=best_node.kernel_code,
                parent_latency_ms=current_node.latency_ms or 0.0,
                child_latency_ms=best_node.latency_ms or 0.0,
                brief_used=brief,
                metric_changes={},
            )

        # ---- Autotune phase ----
        if iteration % autotune_every == 0:
            if verbose:
                print(f"\n  [autotune] Running grid search (every {autotune_every} iters)...")
            try:
                best_code, best_cfg, at_result = run_autotune_phase(
                    best_kernel_code=current_node.kernel_code,
                    benchmark_fn=benchmark_kernel,
                    is_memory_bound=is_memory_bound,
                    top_n=12,
                    verbose=verbose,
                )
                at_sf = at_result.get("speedup_factor", 0.0)
                if at_sf > (current_node.speedup_factor or 0.0) * 1.001:
                    at_node = KernelNode(
                        id=f"iter_{iteration:03d}_autotune",
                        parent_id=current_node.id,
                        kernel_code=best_code,
                        iteration=iteration,
                        variant="autotune",
                        latency_ms=at_result.get("latency_ms"),
                        speedup_factor=at_sf,
                        metrics=at_result,
                    )
                    tree.add_node(at_node)
                    population.update(at_node)
                    current_node = at_node
                    tree.current_node_id = at_node.id
                    _save_best_kernel(best_code)
                    print(f"  [autotune] New best from grid: {at_sf:.3f}x with {best_cfg}")
            except Exception as e:
                print(f"  [autotune] Failed: {e}")

        # ---- Crossbreed phase ----
        if iteration % crossbreed_every == 0 and len(population.members) >= 2:
            if verbose:
                print(f"\n  [crossbreed] Running population crossbreed...")
                print(population.summary())
            try:
                cb_code = population.crossbreed(model=coder_model)
                if cb_code:
                    cb_result = benchmark_kernel(cb_code)
                    cb_sf = cb_result.get("speedup_factor", 0.0)
                    cb_node = KernelNode(
                        id=f"iter_{iteration:03d}_crossbreed",
                        parent_id=population.members[0].id,
                        kernel_code=cb_code,
                        iteration=iteration,
                        variant="crossbreed",
                        latency_ms=cb_result.get("latency_ms"),
                        speedup_factor=cb_sf,
                        metrics=cb_result,
                    )
                    tree.add_node(cb_node)
                    if cb_sf > (current_node.speedup_factor or 0.0) * 1.001:
                        population.update(cb_node)
                        current_node = cb_node
                        tree.current_node_id = cb_node.id
                        _save_best_kernel(cb_code)
                        print(f"  [crossbreed] New best: {cb_sf:.3f}x")
                    else:
                        print(f"  [crossbreed] No improvement ({cb_sf:.3f}x)")
            except Exception as e:
                print(f"  [crossbreed] Failed: {e}")

        # ---- Backtrack check ----
        if tree.should_backtrack(window=backtrack_window):
            alt_node = tree.backtrack()
            if alt_node and alt_node.id != current_node.id:
                print(
                    f"\n  [backtrack] Stagnated for {backtrack_window} iters. "
                    f"Branching from {alt_node.id} ({alt_node.speedup_factor:.3f}x)"
                )
                current_node = alt_node
                tree.current_node_id = alt_node.id

        # ---- Save state ----
        tree.save(TREE_PATH)

        # ---- Log iteration ----
        elapsed = time.time() - t_start
        _log_iteration({
            "iteration": iteration,
            "timestamp": time.time(),
            "elapsed_s": elapsed,
            "current_node_id": current_node.id,
            "current_speedup": current_node.speedup_factor,
            "current_latency_ms": current_node.latency_ms,
            "best_node_id": tree.best_node_id,
            "best_speedup": (tree.get_best_node() or current_node).speedup_factor,
            "bound": roofline.get("bound"),
            "dominant_stall": metrics.get("dominant_stall"),
            "variants_tried": {
                v: {"speedup_factor": r.get("speedup_factor"), "status": r.get("status")}
                for v, r in variant_results.items()
            },
        })

    best = tree.get_best_node()
    if best:
        print(f"\nOptimization complete. Best: {best.id} @ {best.speedup_factor:.3f}x ({best.latency_ms:.3f}ms)")
        print(f"Best kernel saved to: {BEST_KERNEL_PATH}")
