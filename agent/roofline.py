"""
Roofline analysis for NVIDIA B200 with the fused MoE workload.

Computes:
  - Theoretical arithmetic intensity for given num_tokens_per_expert
  - Whether the kernel is memory-bound or compute-bound
  - Specific action items for each case
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# NVIDIA B200 hardware constants
# ---------------------------------------------------------------------------
# FP8 tensor core peak throughput (FLOPS, based on B200 spec sheet)
B200_PEAK_FP8_TFLOPS = 7457.0  # TFLOPS (FP8 sparse: ~14.9 PF; dense: ~7.5 PF)
B200_PEAK_BF16_TFLOPS = 3958.0  # TFLOPS BF16 dense
B200_PEAK_HBM_GBPS = 8_000.0   # HBM3e bandwidth GB/s
B200_PEAK_L2_GBPS = 36_864.0   # L2 bandwidth GB/s (estimated)
B200_SMEM_PER_SM_KB = 228      # Shared memory per SM (KB)
B200_REGS_PER_SM = 65536       # Registers per SM

# FP8 = 1 byte, BF16 = 2 bytes, FP32 = 4 bytes
FP8_BYTES = 1
FP32_BYTES = 4
BF16_BYTES = 2

# Kernel dimensions
HIDDEN_DIM = 7168
INTER_DIM = 2048
GATE_UP_DIM = INTER_DIM * 2  # 4096


def compute_roofline(
    metrics: dict[str, Any],
    num_tokens_per_expert: int,
) -> dict[str, Any]:
    """
    Compute roofline analysis for a single expert's FFN given token count.

    Returns a dict with bound classification, arithmetic intensity,
    ridge point, and specific action items for the Judge agent.
    """
    T = num_tokens_per_expert

    # ---- FLOPs ----
    # GEMM1: [T, 7168] × [4096, 7168]^T  →  2 * T * 4096 * 7168
    flops_gemm1 = 2 * T * GATE_UP_DIM * HIDDEN_DIM
    # GEMM2: [T, 2048] × [7168, 2048]^T  →  2 * T * 7168 * 2048
    flops_gemm2 = 2 * T * HIDDEN_DIM * INTER_DIM
    total_flops = flops_gemm1 + flops_gemm2

    # ---- Bytes read (FP8 = 1 byte per element) ----
    # Hidden states: T × 7168 bytes
    bytes_hidden = T * HIDDEN_DIM * FP8_BYTES
    # GEMM1 weights: 4096 × 7168 bytes (reused across tokens if T > 1)
    bytes_w1 = GATE_UP_DIM * HIDDEN_DIM * FP8_BYTES
    # GEMM2 weights: 7168 × 2048 bytes
    bytes_w2 = HIDDEN_DIM * INTER_DIM * FP8_BYTES
    # Scale tensors (FP32)
    bytes_scales = (
        (HIDDEN_DIM // 128) * T * FP32_BYTES       # hidden_states_scale [56, T]
        + (GATE_UP_DIM // 128) * (HIDDEN_DIM // 128) * FP32_BYTES  # w1scale [32, 56]
        + (HIDDEN_DIM // 128) * (INTER_DIM // 128) * FP32_BYTES    # w2scale [56, 16]
    )
    # Intermediate buffer: T × 4096 float32 write + T × 2048 float32 read
    bytes_intermediate = T * (GATE_UP_DIM + INTER_DIM) * FP32_BYTES
    # Output: T × 7168 BF16 (atomic add, but roughly this many bytes written)
    bytes_output = T * HIDDEN_DIM * BF16_BYTES

    total_bytes = (
        bytes_hidden + bytes_w1 + bytes_w2 + bytes_scales
        + bytes_intermediate + bytes_output
    )

    # ---- Arithmetic intensity ----
    ai = total_flops / max(total_bytes, 1)
    ridge_point = (B200_PEAK_FP8_TFLOPS * 1e12) / (B200_PEAK_HBM_GBPS * 1e9)

    # ---- Classification ----
    if ai >= ridge_point:
        bound = "COMPUTE_BOUND"
        sol_ratio = (
            metrics.get("compute_throughput_pct", 0.0) / 100.0
        )
    else:
        bound = "MEMORY_BOUND"
        sol_ratio = (
            metrics.get("memory_throughput_pct", 0.0) / 100.0
        )

    # ---- Action items ----
    actions: list[str] = []
    metrics_bound = metrics.get("bound", bound)

    if metrics_bound == "MEMORY_BOUND" or bound == "MEMORY_BOUND":
        l2_hit = metrics.get("l2_hit_rate_pct", -1)
        l1_hit = metrics.get("l1_hit_rate_pct", -1)
        bank_conflicts = metrics.get("total_bank_conflicts", 0)
        dram_gbps = metrics.get("dram_total_gbps", 0)

        if l2_hit >= 0 and l2_hit < 50:
            actions.append(
                f"L2 hit rate is {l2_hit:.0f}% (LOW). "
                "Increase BLOCK_K or BLOCK_M to improve weight reuse across K-iterations."
            )
        if l1_hit >= 0 and l1_hit < 40:
            actions.append(
                f"L1 hit rate is {l1_hit:.0f}% (LOW). "
                "Reduce shared memory footprint or increase num_stages for better pipelining."
            )
        if bank_conflicts > 1000:
            actions.append(
                f"High shared memory bank conflicts ({bank_conflicts:.0f}). "
                "Add padding (BLOCK_K+1) to shared memory arrays to eliminate conflicts."
            )
        if dram_gbps > 0.8 * B200_PEAK_HBM_GBPS:
            actions.append(
                f"DRAM bandwidth near saturation ({dram_gbps:.0f} GB/s vs {B200_PEAK_HBM_GBPS} peak). "
                "Use larger tiles to amortize weight loads over more tokens."
            )
        if T < 32:
            actions.append(
                f"Only {T} tokens per expert — kernel is heavily memory-bound. "
                "Consider persistent CTAs or token batching to reuse weights across calls."
            )

    else:  # COMPUTE_BOUND
        tensor_pct = metrics.get("tensor_active_pct", -1)
        fp8_pct = metrics.get("fp8_tensor_active_pct", -1)
        warp_div = metrics.get("warp_divergence_pct", 0)

        if tensor_pct >= 0 and tensor_pct < 80:
            actions.append(
                f"Tensor core utilization only {tensor_pct:.0f}%. "
                "Ensure BLOCK_M >= 64 and BLOCK_N >= 64 for WGMMA efficiency."
            )
        if fp8_pct >= 0 and fp8_pct < tensor_pct * 0.8:
            actions.append(
                f"FP8 tensor core usage ({fp8_pct:.0f}%) below general TC ({tensor_pct:.0f}%). "
                "Verify FP8 inputs are aligned to 128-element boundaries."
            )
        if warp_div > 20:
            actions.append(
                f"High warp divergence ({warp_div:.0f}%). "
                "Tokens routed to different experts cause thread divergence — "
                "consider sorting tokens by expert before launch."
            )

    # Always add occupancy action if low
    achieved_occ = metrics.get("achieved_occupancy_pct", -1)
    theoretical_occ = metrics.get("theoretical_occupancy_pct", -1)
    if achieved_occ >= 0 and theoretical_occ >= 0:
        occ_ratio = achieved_occ / max(theoretical_occ, 1)
        if occ_ratio < 0.7:
            limiter = metrics.get("occupancy_limiter", "unknown")
            actions.append(
                f"Achieved occupancy ({achieved_occ:.0f}%) is only {100*occ_ratio:.0f}% of theoretical "
                f"({theoretical_occ:.0f}%). Limiter: {limiter}. "
                + (
                    "Reduce register usage (e.g., fewer persistent accumulators)."
                    if limiter == "registers"
                    else "Reduce static shared memory allocation."
                    if limiter == "shared_memory"
                    else "Increase parallelism or reduce CTA launch overhead."
                )
            )

    # mbarrier stall
    mbar_pct = metrics.get("stall_mbarrier_wait_pct", 0)
    if mbar_pct > 10:
        actions.append(
            f"mbarrier wait stall is {mbar_pct:.0f}% — TMA pipeline is under-filled. "
            "Increase num_stages to 4 or 5 to overlap memory fetches with compute."
        )

    return {
        "bound": bound,
        "metrics_bound": metrics_bound,
        "arithmetic_intensity": ai,
        "ridge_point": ridge_point,
        "ai_vs_ridge_ratio": ai / ridge_point,
        "sol_ratio": sol_ratio,
        "total_flops": total_flops,
        "total_bytes": total_bytes,
        "action_items": actions,
        "num_tokens_per_expert": T,
        "peak_fp8_tflops": B200_PEAK_FP8_TFLOPS,
        "peak_hbm_gbps": B200_PEAK_HBM_GBPS,
    }


def format_roofline_for_prompt(roofline: dict[str, Any]) -> str:
    """Format roofline analysis as a block for LLM prompts."""
    lines = [
        "=== Roofline Analysis ===",
        f"  Bound: {roofline['bound']} (measured: {roofline['metrics_bound']})",
        f"  Arithmetic intensity: {roofline['arithmetic_intensity']:.1f} FLOP/byte",
        f"  Ridge point (B200 FP8): {roofline['ridge_point']:.1f} FLOP/byte",
        f"  AI / ridge: {roofline['ai_vs_ridge_ratio']:.2f}x",
        f"  SoL utilization: {roofline['sol_ratio']*100:.1f}%",
        f"  Tokens/expert (estimate): {roofline['num_tokens_per_expert']}",
        "",
        "  Action items:",
    ]
    for i, item in enumerate(roofline["action_items"], 1):
        lines.append(f"    {i}. {item}")
    if not roofline["action_items"]:
        lines.append("    (none identified — kernel is near peak efficiency)")
    return "\n".join(lines)
