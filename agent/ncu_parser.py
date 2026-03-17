"""
NCU output parser: converts raw Nsight Compute text → structured metric dict.

Parses 50+ metrics across speed-of-light, memory, compute, stalls,
shared memory, occupancy, and MoE-specific indicators.
"""

from __future__ import annotations

import re
from typing import Any


# ---------------------------------------------------------------------------
# Metric extraction patterns
# Each entry: (key, regex_pattern, unit)
# ---------------------------------------------------------------------------
_METRIC_PATTERNS: list[tuple[str, str, str]] = [
    # ---- Speed of light ----
    ("compute_throughput_pct",
     r"sm__throughput\.avg\.pct_of_peak_sustained_elapsed\s+([\d.]+)", "%"),
    ("memory_throughput_pct",
     r"gpu__dram_throughput\.avg\.pct_of_peak_sustained_elapsed\s+([\d.]+)", "%"),

    # ---- Occupancy ----
    ("achieved_occupancy_pct",
     r"sm__warps_active\.avg\.pct_of_peak_sustained_active\s+([\d.]+)", "%"),
    ("theoretical_occupancy_pct",
     r"theoretical_occupancy\s+([\d.]+)", "%"),
    ("active_warps_per_sm",
     r"sm__warps_active\.avg\.per_cycle_active\s+([\d.]+)", "warps"),

    # ---- Compute ----
    ("tensor_active_pct",
     r"sm__pipe_tensor(?:_op_hmma)?_cycles_active\.avg\.pct_of_peak_sustained_active\s+([\d.]+)", "%"),
    ("fp8_tensor_active_pct",
     r"sm__pipe_tensor_op_imma_cycles_active\.avg\.pct_of_peak_sustained_active\s+([\d.]+)", "%"),
    ("fp64_active_pct",
     r"sm__pipe_fp64_cycles_active\.avg\.pct_of_peak_sustained_active\s+([\d.]+)", "%"),
    ("warp_cycles_per_issued_instr",
     r"smsp__average_warp_latency_per_inst_issued\.ratio\s+([\d.]+)", "cycles"),

    # ---- Memory ----
    ("l1_hit_rate_pct",
     r"l1tex__t_sector_hit_rate\.pct\s+([\d.]+)", "%"),
    ("l2_hit_rate_pct",
     r"lts__t_sector_hit_rate\.pct\s+([\d.]+)", "%"),
    ("dram_read_gbps",
     r"l1tex__m_l1tex2xbar_pipe_lsu_mem_global_op_ld_bytes_not_lookup_hit_in_l1tex\.sum\.per_second\s+([\d.]+)", "GB/s"),
    ("dram_write_gbps",
     r"l1tex__m_l1tex2xbar_pipe_lsu_mem_global_op_st_bytes_not_lookup_hit_in_l1tex\.sum\.per_second\s+([\d.]+)", "GB/s"),
    ("dram_sector_reads",
     r"l1tex__m_l1tex2xbar_pipe_lsu_mem_global_op_ld\.sum\s+([\d.]+)", "sectors"),
    ("dram_sector_writes",
     r"l1tex__m_l1tex2xbar_pipe_lsu_mem_global_op_st\.sum\s+([\d.]+)", "sectors"),
    ("l2_read_sectors",
     r"lts__t_sectors_op_read\.sum\s+([\d.]+)", "sectors"),
    ("l2_write_sectors",
     r"lts__t_sectors_op_write\.sum\s+([\d.]+)", "sectors"),

    # ---- Shared memory ----
    ("shared_load_bank_conflicts",
     r"l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld\.sum\s+([\d.]+)", "conflicts"),
    ("shared_store_bank_conflicts",
     r"l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st\.sum\s+([\d.]+)", "conflicts"),
    ("shared_mem_transactions_ld",
     r"l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld\.sum\s+([\d.]+)", "txn"),
    ("shared_mem_transactions_st",
     r"l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st\.sum\s+([\d.]+)", "txn"),

    # ---- Resources (occupancy limiters) ----
    ("registers_per_thread",
     r"launch__registers_per_thread\s+([\d.]+)", "regs"),
    ("shared_mem_per_block_bytes",
     r"launch__shared_mem_per_block_static\s+([\d.]+)", "bytes"),
    ("blocks_per_sm",
     r"launch__occupancy_limit_blocks\s+([\d.]+)", "blocks"),
    ("warps_per_block",
     r"launch__warps_per_block\s+([\d.]+)", "warps"),

    # ---- Stalls ----
    ("stall_mio_throttle_pct",
     r"smsp__warp_issue_stalled_mio_throttle_per_warp_active\.pct\s+([\d.]+)", "%"),
    ("stall_long_scoreboard_pct",
     r"smsp__warp_issue_stalled_long_scoreboard_per_warp_active\.pct\s+([\d.]+)", "%"),
    ("stall_wait_pct",
     r"smsp__warp_issue_stalled_wait_per_warp_active\.pct\s+([\d.]+)", "%"),
    ("stall_math_throttle_pct",
     r"smsp__warp_issue_stalled_math_throttle_per_warp_active\.pct\s+([\d.]+)", "%"),
    ("stall_not_selected_pct",
     r"smsp__warp_issue_stalled_not_selected_per_warp_active\.pct\s+([\d.]+)", "%"),
    ("stall_short_scoreboard_pct",
     r"smsp__warp_issue_stalled_short_scoreboard_per_warp_active\.pct\s+([\d.]+)", "%"),
    ("stall_imc_miss_pct",
     r"smsp__warp_issue_stalled_imc_miss_per_warp_active\.pct\s+([\d.]+)", "%"),
    ("stall_barrier_pct",
     r"smsp__warp_issue_stalled_barrier_per_warp_active\.pct\s+([\d.]+)", "%"),
    ("stall_tex_throttle_pct",
     r"smsp__warp_issue_stalled_tex_throttle_per_warp_active\.pct\s+([\d.]+)", "%"),

    # ---- MoE-specific ----
    # Warp divergence: threads active per instruction (lower = more divergence)
    ("warp_divergence_active_pct",
     r"smsp__thread_inst_executed_per_inst_executed\.ratio\s+([\d.]+)", "ratio"),
    # mbarrier (TMA async) stalls
    ("stall_mbarrier_wait_pct",
     r"smsp__warp_issue_stalled_mbarrier_wait_per_warp_active\.pct\s+([\d.]+)", "%"),
    # Atomic operation throughput
    ("atomic_transactions",
     r"l1tex__t_set_accesses_pipe_lsu_mem_global_op_atom\.sum\s+([\d.]+)", "txn"),

    # ---- Instruction mix ----
    ("inst_executed",
     r"smsp__inst_executed\.sum\s+([\d.]+)", "insts"),
    ("inst_issued",
     r"smsp__inst_issued\.sum\s+([\d.]+)", "insts"),
    ("fp32_active_pct",
     r"smsp__sass_thread_inst_executed_op_fadd_pred_on\.sum\s+([\d.]+)", "insts"),
]

# Fallback: parse the "Speed Of Light" section table format
_SOL_PATTERN = re.compile(
    r"([\w/]+(?:\s[\w/]+)*)\s+([\d.]+)\s*(%|GB/s|Tflop/s|GHz)?",
    re.MULTILINE,
)


def parse_ncu_output(ncu_text: str) -> dict[str, Any]:
    """
    Parse NCU text output into a structured metric dict.

    Returns dict with float values for all found metrics plus
    derived values: bound, dominant_stall, occupancy_limiter.
    """
    metrics: dict[str, Any] = {}

    # Primary extraction: regex per known metric
    for key, pattern, _unit in _METRIC_PATTERNS:
        m = re.search(pattern, ncu_text, re.IGNORECASE | re.MULTILINE)
        if m:
            try:
                metrics[key] = float(m.group(1))
            except ValueError:
                pass

    # Fallback: parse throughput table rows (handles format variations)
    if "compute_throughput_pct" not in metrics or "memory_throughput_pct" not in metrics:
        for line in ncu_text.splitlines():
            line = line.strip()
            if "SM Active Cycles" in line or "Compute" in line:
                nums = re.findall(r"([\d.]+)\s*%", line)
                if nums and "compute_throughput_pct" not in metrics:
                    metrics["compute_throughput_pct"] = float(nums[-1])
            if "Memory" in line and "%" in line:
                nums = re.findall(r"([\d.]+)\s*%", line)
                if nums and "memory_throughput_pct" not in metrics:
                    metrics["memory_throughput_pct"] = float(nums[-1])

    # ---- Derived metrics ----

    # 1. Bound classification
    compute_pct = metrics.get("compute_throughput_pct", 0.0)
    memory_pct = metrics.get("memory_throughput_pct", 0.0)
    if compute_pct > 0 or memory_pct > 0:
        metrics["bound"] = "COMPUTE_BOUND" if compute_pct >= memory_pct else "MEMORY_BOUND"
        metrics["compute_pct"] = compute_pct
        metrics["memory_pct"] = memory_pct

    # 2. Dominant stall
    stall_keys = [k for k in metrics if k.startswith("stall_") and k.endswith("_pct")]
    if stall_keys:
        dominant = max(stall_keys, key=lambda k: metrics[k])
        metrics["dominant_stall"] = dominant
        metrics["dominant_stall_pct"] = metrics[dominant]

    # 3. Occupancy limiter
    regs = metrics.get("registers_per_thread", 0)
    smem = metrics.get("shared_mem_per_block_bytes", 0)
    if regs > 128:
        metrics["occupancy_limiter"] = "registers"
    elif smem > 196_608:  # > 192KB
        metrics["occupancy_limiter"] = "shared_memory"
    else:
        metrics["occupancy_limiter"] = "warps_or_blocks"

    # 4. Total DRAM bandwidth
    metrics["dram_total_gbps"] = (
        metrics.get("dram_read_gbps", 0.0) + metrics.get("dram_write_gbps", 0.0)
    )

    # 5. Bank conflict severity
    total_bc = (
        metrics.get("shared_load_bank_conflicts", 0.0)
        + metrics.get("shared_store_bank_conflicts", 0.0)
    )
    metrics["total_bank_conflicts"] = total_bc
    if total_bc > 0:
        total_txn = (
            metrics.get("shared_mem_transactions_ld", 1.0)
            + metrics.get("shared_mem_transactions_st", 1.0)
        )
        metrics["bank_conflict_rate"] = total_bc / max(total_txn, 1.0)

    # 6. Warp divergence flag
    div_ratio = metrics.get("warp_divergence_active_pct", 32.0)
    metrics["warp_divergence_pct"] = max(0.0, 100.0 * (1.0 - div_ratio / 32.0))

    return metrics


def format_metrics_for_prompt(metrics: dict[str, Any]) -> str:
    """Format parsed metrics as a readable block for LLM prompts."""
    lines = []

    def _add(label: str, key: str, fmt: str = ".1f", suffix: str = ""):
        v = metrics.get(key)
        if v is not None:
            lines.append(f"  {label}: {v:{fmt}}{suffix}")

    lines.append("=== Speed-of-Light ===")
    _add("Compute throughput", "compute_throughput_pct", suffix="%")
    _add("Memory throughput", "memory_throughput_pct", suffix="%")
    _add("Bound classification", "bound", fmt="s")

    lines.append("\n=== Occupancy ===")
    _add("Achieved occupancy", "achieved_occupancy_pct", suffix="%")
    _add("Theoretical occupancy", "theoretical_occupancy_pct", suffix="%")
    _add("Registers/thread", "registers_per_thread", fmt=".0f")
    _add("Shared mem/block", "shared_mem_per_block_bytes", fmt=".0f", suffix=" bytes")
    _add("Occupancy limiter", "occupancy_limiter", fmt="s")

    lines.append("\n=== Compute ===")
    _add("Tensor core active", "tensor_active_pct", suffix="%")
    _add("FP8 tensor active", "fp8_tensor_active_pct", suffix="%")
    _add("Warp cycles/issued instr", "warp_cycles_per_issued_instr")

    lines.append("\n=== Memory ===")
    _add("L1 hit rate", "l1_hit_rate_pct", suffix="%")
    _add("L2 hit rate", "l2_hit_rate_pct", suffix="%")
    _add("DRAM read", "dram_read_gbps", suffix=" GB/s")
    _add("DRAM write", "dram_write_gbps", suffix=" GB/s")
    _add("DRAM total", "dram_total_gbps", suffix=" GB/s")
    _add("DRAM sector reads", "dram_sector_reads", fmt=".0f")

    lines.append("\n=== Shared Memory ===")
    _add("Load bank conflicts", "shared_load_bank_conflicts", fmt=".0f")
    _add("Store bank conflicts", "shared_store_bank_conflicts", fmt=".0f")
    _add("Bank conflict rate", "bank_conflict_rate", fmt=".3f")

    lines.append("\n=== Stalls ===")
    _add("MIO throttle", "stall_mio_throttle_pct", suffix="%")
    _add("Long scoreboard", "stall_long_scoreboard_pct", suffix="%")
    _add("Wait", "stall_wait_pct", suffix="%")
    _add("Math throttle", "stall_math_throttle_pct", suffix="%")
    _add("Not selected", "stall_not_selected_pct", suffix="%")
    _add("mbarrier wait", "stall_mbarrier_wait_pct", suffix="%")
    _add("Dominant stall", "dominant_stall", fmt="s")
    _add("Dominant stall %", "dominant_stall_pct", suffix="%")

    lines.append("\n=== MoE-Specific ===")
    _add("Warp divergence", "warp_divergence_pct", suffix="%  (0=none, 100=max divergence)")
    _add("Atomic transactions", "atomic_transactions", fmt=".0f")

    return "\n".join(lines)
