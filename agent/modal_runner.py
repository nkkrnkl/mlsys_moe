"""
Programmatic Modal interface for the agent optimization loop.

Packs kernel code → Solution object, then dispatches to Modal B200 for
benchmarking and NCU profiling without touching the file system solution.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from flashinfer_bench import BuildSpec, Solution
from flashinfer_bench.agents import pack_solution_from_files


def _load_config() -> dict:
    config_path = PROJECT_ROOT / "config.toml"
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def pack_kernel_code(kernel_code: str) -> Solution:
    """
    Write kernel_code to a temp directory and pack it into a Solution object.

    Returns a Solution that can be passed to Modal functions.
    """
    config = _load_config()
    sol_cfg = config["solution"]
    build_cfg = config["build"]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        kernel_file = tmp_path / "kernel.py"
        kernel_file.write_text(kernel_code)

        spec = BuildSpec(
            language="triton",
            target_hardware=["cuda"],
            entry_point=build_cfg["entry_point"],
        )

        solution_json = pack_solution_from_files(
            path=str(tmp_path),
            spec=spec,
            name=sol_cfg["name"],
            definition=sol_cfg["definition"],
            author=sol_cfg["author"],
        )

    return solution_json


def benchmark_kernel(kernel_code: str) -> dict[str, Any]:
    """
    Pack a kernel and benchmark it on Modal B200.

    Returns aggregate result dict:
        {latency_ms, speedup_factor, max_abs_error, status, workload_results}
    """
    from scripts.run_modal import run_benchmark  # type: ignore

    solution = pack_kernel_code(kernel_code)
    raw_results = run_benchmark.remote(solution)

    return _aggregate_results(raw_results)


def profile_kernel_ncu(
    kernel_code: str,
    workload_uuid: str = "",
    ncu_set: str = "detailed",
) -> str:
    """
    Pack a kernel and run Nsight Compute profiling on Modal B200.

    Returns raw NCU text output string.
    """
    from scripts.run_modal import run_ncu_on_modal  # type: ignore

    solution = pack_kernel_code(kernel_code)
    return run_ncu_on_modal.remote(solution, workload_uuid, ncu_set)


def get_first_workload_uuid() -> str:
    """
    Return the UUID of the first workload in the trace set (for NCU profiling).
    Reads from the local trace set if FIB_DATASET_PATH is set, otherwise returns "".
    """
    import os
    path = os.environ.get("FIB_DATASET_PATH", "")
    if not path:
        return ""
    try:
        from flashinfer_bench import TraceSet
        trace_set = TraceSet.from_path(path)
        workloads = trace_set.workloads.get("fused_moe", [])
        if workloads:
            return str(workloads[0].uuid)
    except Exception:
        pass
    return ""


def _aggregate_results(raw_results: dict) -> dict[str, Any]:
    """
    Aggregate per-workload benchmark results into summary statistics.
    Uses p95 latency across workloads as the primary metric.
    """
    all_latencies = []
    all_speedups = []
    all_errors = []
    statuses = []
    workload_results = {}

    for def_name, traces in raw_results.items():
        for uuid, entry in traces.items():
            statuses.append(entry.get("status", "UNKNOWN"))
            workload_results[uuid] = entry
            if entry.get("latency_ms") is not None:
                all_latencies.append(entry["latency_ms"])
            if entry.get("speedup_factor") is not None:
                all_speedups.append(entry["speedup_factor"])
            if entry.get("max_abs_error") is not None:
                all_errors.append(entry["max_abs_error"])

    # p95 latency
    if all_latencies:
        sorted_lat = sorted(all_latencies)
        p95_idx = max(0, int(0.95 * len(sorted_lat)) - 1)
        p95_latency = sorted_lat[p95_idx]
    else:
        p95_latency = None

    return {
        "latency_ms": p95_latency,
        "mean_latency_ms": sum(all_latencies) / len(all_latencies) if all_latencies else None,
        "speedup_factor": sum(all_speedups) / len(all_speedups) if all_speedups else None,
        "max_abs_error": max(all_errors) if all_errors else None,
        "status": "PASS" if all(s == "PASS" for s in statuses) else "FAIL",
        "statuses": statuses,
        "workload_results": workload_results,
        "num_workloads": len(workload_results),
    }
