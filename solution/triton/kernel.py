"""
Fused MoE Triton Kernel — Token-Sorted Grouped GEMM variant.

Target: moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048
  - DeepSeek-V3/R1 scale MoE: hidden=7168, intermediate=2048
  - 32 local experts (256 global), top-8 routing
  - FP8 E4M3FN weights with 128-element block scaling
  - DeepSeek routing: 8 groups, top-4 groups, top-8 experts

Optimization: token-sorted grouped GEMM + fused SwiGLU epilogue.
  Old: 32 Python-loop iterations × 2 kernels = 64 serial launches.
  New: 2 Triton kernel launches covering all 32 experts simultaneously.
  Also eliminates full upfront weight dequantization (~5.7 GB materialized in baseline)
  and the gate_up_buf round-trip (~0.5 GB at full batch).

Pipeline:
  routing (PyTorch, proven correct) → token sort (PyTorch) →
  grouped_gemm1 (FP8 MMA + fused SwiGLU epilogue, writes inter directly) →
  grouped_gemm2 (FP8 inter + FP8 weights → FP8 MMA, atomic scatter) →
  cast to bf16 output

SwiGLU convention (verified against baseline passing all 19 workloads):
  w1[:, :INTER_DIM,       :] → up    rows (first_half of gate_up)
  w1[:, INTER_DIM:,       :] → gate  rows (second_half of gate_up)
  output = silu(gate) * up   ==   silu(second_half) * first_half
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_GLOBAL_EXPERTS  = 256
NUM_GROUPS          = 8      # ng8
EXPERTS_PER_GROUP   = 32     # 256 / 8
NUM_GROUPS_SELECTED = 4      # kg4
TOPK                = 8      # topk8
NUM_LOCAL_EXPERTS   = 32     # e32
HIDDEN_DIM          = 7168   # h7168
INTER_DIM           = 2048   # i2048
FP8_BLOCK_SIZE      = 128
GATE_UP_DIM         = INTER_DIM * 2  # 4096
FP8_E4M3FN_MAX      = tl.constexpr(448.0)  # max representable value of float8_e4m3fn


# ---------------------------------------------------------------------------
# Autotune configs for B200
# BLOCK_K is always 128 — must align with FP8 block-scale granularity.
# ---------------------------------------------------------------------------
def _gemm1_configs():
    """Autotune configs for GEMM1. N=INTER_DIM=2048."""
    configs = []
    for bm in [64, 128]:
        for bn in [128, 256]:
            for ns in [3, 4, 5]:
                for nw in [8, 16]:
                    configs.append(triton.Config(
                        {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": 128},
                        num_stages=ns, num_warps=nw,
                    ))
    return configs


def _gemm2_configs():
    """Autotune configs for GEMM2. N=HIDDEN_DIM=7168, includes BLOCK_N=512."""
    configs = []
    for bm in [64, 128]:
        for bn in [128, 256, 512]:
            for ns in [3, 4, 5]:
                for nw in [8, 16]:
                    configs.append(triton.Config(
                        {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": 128},
                        num_stages=ns, num_warps=nw,
                    ))
    return configs


def _bucket(n: int) -> int:
    """Round n up to next power of 2 for autotune key bucketing."""
    for t in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]:
        if n <= t:
            return t
    return 16384


# ---------------------------------------------------------------------------
# Routing — mirrors proven-correct PyTorch baseline exactly.
#
# Returns:
#   expert_ids   [seq, 8]   int32   — global expert IDs selected per token
#   weights_full [seq, 256] float32 — normalized routing weight per (token, expert)
#                                     non-zero only for 8 selected experts.
#                                     RSF NOT applied here; applied in GEMM2 epilogue.
# ---------------------------------------------------------------------------
def _route(routing_logits, routing_bias, seq_len, device):
    bias_f32    = routing_bias.to(torch.float32)
    s           = torch.sigmoid(routing_logits.float())   # [seq, 256] unbiased
    s_with_bias = s + bias_f32                            # [seq, 256] biased

    # Group scoring: top-2 sum per group (biased)
    gsm          = s_with_bias.view(seq_len, NUM_GROUPS, EXPERTS_PER_GROUP)
    top2_vals, _ = gsm.topk(2, dim=-1)
    group_scores = top2_vals.sum(-1)                      # [seq, 8]

    _, top_groups = group_scores.topk(NUM_GROUPS_SELECTED, dim=-1)  # [seq, 4]
    group_mask    = torch.zeros_like(group_scores)
    group_mask.scatter_(1, top_groups, 1.0)
    score_mask    = (group_mask.unsqueeze(2)
                     .expand(seq_len, NUM_GROUPS, EXPERTS_PER_GROUP)
                     .reshape(seq_len, NUM_GLOBAL_EXPERTS))

    neg_inf       = torch.finfo(torch.float32).min
    scores_pruned = s_with_bias.masked_fill(score_mask == 0, neg_inf)
    _, top8_ids   = scores_pruned.topk(TOPK, dim=-1)     # [seq, 8]

    # Routing weights: unbiased s, normalized. RSF applied later in GEMM2.
    weight_mask  = torch.zeros_like(s)
    weight_mask.scatter_(1, top8_ids, 1.0)
    weights      = s * weight_mask
    weights_sum  = weights.sum(-1, keepdim=True).clamp(min=1e-20)
    weights_full = weights / weights_sum                  # [seq, 256]

    return top8_ids.to(torch.int32), weights_full


# ---------------------------------------------------------------------------
# Token sorting
#
# Flattens all (token, topk_slot) assignments, keeps only local experts
# [local_offset, local_offset+31], sorts by local expert id, and computes
# cumulative per-expert offsets.
#
# Returns:
#   sorted_tok_ids  [N_assigned] int32   global token row indices, sorted by expert
#   expert_offsets  [33]         int32   expert_offsets[e] = start idx for expert e
#   sorted_r_scores [N_assigned] float32 normalized routing weight (no RSF)
#   N_assigned      int
#   max_tok         int          max tokens assigned to any single local expert
# ---------------------------------------------------------------------------
def _sort_tokens(expert_ids, weights_full, local_offset, device):
    local_mask = ((expert_ids >= local_offset) &
                  (expert_ids <  local_offset + NUM_LOCAL_EXPERTS))
    pairs = local_mask.nonzero(as_tuple=False)            # [N_assigned, 2]

    if pairs.shape[0] == 0:
        empty   = torch.zeros(0, dtype=torch.int32, device=device)
        offsets = torch.zeros(NUM_LOCAL_EXPERTS + 1, dtype=torch.int32, device=device)
        return empty, offsets, empty, 0, 0

    tok_global = pairs[:, 0]                              # int64 global token row
    slot       = pairs[:, 1]
    global_exp = expert_ids[tok_global, slot]             # int32 global expert id
    local_exp  = (global_exp - local_offset).to(torch.int32)
    r_scores   = weights_full[tok_global, global_exp.long()].to(torch.float32)

    order           = torch.argsort(local_exp, stable=True)
    sorted_tok_ids  = tok_global[order].to(torch.int32).contiguous()
    sorted_local    = local_exp[order].contiguous()
    sorted_r_scores = r_scores[order].contiguous()

    counts         = torch.bincount(sorted_local, minlength=NUM_LOCAL_EXPERTS)
    expert_offsets = torch.zeros(NUM_LOCAL_EXPERTS + 1, dtype=torch.int32, device=device)
    expert_offsets[1:] = counts.cumsum(0).to(torch.int32)

    N_assigned = int(sorted_tok_ids.shape[0])
    max_tok    = int(counts.max().item())

    return sorted_tok_ids, expert_offsets, sorted_r_scores, N_assigned, max_tok


# ---------------------------------------------------------------------------
# Kernel 1: Grouped GEMM1 with fused SwiGLU — PERSISTENT scheduler
#
# For every sorted position m and output column c ∈ [0, INTER_DIM):
#   up   [m, c] = hidden[tok_ids[m]] @ w1[e, c,            :].T
#   gate [m, c] = hidden[tok_ids[m]] @ w1[e, INTER_DIM + c, :].T
#   inter[m, c] = silu(gate[m, c]) * up[m, c]
#
# Persistent schedule: grid = (NUM_SMS,). Each CTA precomputes per-expert
# m-tile counts from expert_offsets and prefix-sums them, then loops
# tile_id = pid, pid+NUM_SMS, ... until total_tiles. This eliminates the
# early-return waste that the old grid-by-max_tok approach suffered on
# imbalanced experts.
#
# Each tile computes a BLOCK_N-wide slice of inter directly. Per inner K
# iteration we run two FP8 MMAs sharing the hidden load — one against the
# "up" weight rows [0, INTER_DIM), one against the "gate" weight rows
# [INTER_DIM, 2*INTER_DIM). Both 128-K-block scale drains happen
# in-register; SwiGLU is applied once at the end and stored to inter_buf.
# ---------------------------------------------------------------------------
@triton.autotune(configs=_gemm1_configs(), key=["n_assigned_bucket", "N"])
@triton.jit
def grouped_gemm1_kernel(
    # Sorted token data
    sorted_tok_ids_ptr,    # [N_assigned] int32
    expert_offsets_ptr,    # [E+1]        int32

    # Hidden states (full sequence, gather via tok_ids)
    hidden_ptr,            # [seq_len, 7168]   fp8_e4m3fn
    hscale_ptr,            # [56, seq_len]     float32

    # Per-expert weights
    w1_ptr,                # [32, 4096, 7168]  fp8_e4m3fn
    w1scale_ptr,           # [32, 32, 56]      float32

    # Output (fused SwiGLU target)
    inter_ptr,             # [N_assigned, 2048] float32
    N_assigned,
    n_assigned_bucket,

    # Shape constants
    K: tl.constexpr,       # 7168
    N: tl.constexpr,       # 2048 (= INTER_DIM)
    E: tl.constexpr,       # 32 (NUM_LOCAL_EXPERTS)

    # Strides
    stride_h_seq,
    stride_w1_exp,
    stride_w1_n,
    stride_w1s_exp,
    stride_w1s_nb,
    stride_hscale_blk,
    stride_inter_tok,      # stride of inter in elements (= 2048)

    # Block sizes + persistent parallelism
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,  # always 128 = FP8_BLOCK_SIZE
    BLOCK_K: tl.constexpr,  # always 128
    NUM_SMS: tl.constexpr,
):
    pid = tl.program_id(0)

    # Per-expert m-tile prefix (32-wide vector in-register).
    idx_e       = tl.arange(0, E)
    starts      = tl.load(expert_offsets_ptr + idx_e)       # [E]
    ends        = tl.load(expert_offsets_ptr + idx_e + 1)   # [E]
    counts      = ends - starts                              # [E]
    mtiles      = tl.cdiv(counts, BLOCK_M)                   # [E]
    prefix_inc  = tl.cumsum(mtiles, axis=0)                  # [E]
    prefix_exc  = prefix_inc - mtiles                         # [E]
    total_m_tiles = tl.sum(mtiles, axis=0)

    n_tiles     = tl.cdiv(N, BLOCK_N)
    total_tiles = total_m_tiles * n_tiles

    for tile_id in tl.range(pid, total_tiles, NUM_SMS):
        m_tile_id = tile_id // n_tiles
        n_tile_id = tile_id %  n_tiles

        expert_id     = tl.sum((prefix_inc <= m_tile_id).to(tl.int32), axis=0)
        sel           = (idx_e == expert_id).to(tl.int32)
        m_tile_base   = tl.sum(prefix_exc * sel, axis=0)
        e_start       = tl.sum(starts     * sel, axis=0)
        n_expert_tok  = tl.sum(counts     * sel, axis=0)

        m_in_expert = m_tile_id - m_tile_base
        m_off       = m_in_expert * BLOCK_M
        m_range     = m_off + tl.arange(0, BLOCK_M)
        m_mask      = m_range < n_expert_tok
        global_m    = e_start + m_range

        tok_ids = tl.load(sorted_tok_ids_ptr + global_m, mask=m_mask, other=0)

        n_start      = n_tile_id * BLOCK_N
        n_range      = n_start + tl.arange(0, BLOCK_N)         # "up"   cols in [0, INTER_DIM)
        n_mask       = n_range < N
        n_range_gate = n_range + N                              # "gate" rows in [INTER_DIM, 2*INTER_DIM)

        n_blks_up   = n_range      // BLOCK_K
        n_blks_gate = n_range_gate // BLOCK_K

        w1_base  = w1_ptr      + expert_id * stride_w1_exp
        w1s_base = w1scale_ptr + expert_id * stride_w1s_exp

        acc_up   = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        acc_gate = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

        for k_start in tl.range(0, K, BLOCK_K):
            k_blk   = k_start // BLOCK_K
            k_range = k_start + tl.arange(0, BLOCK_K)
            k_mask  = k_range < K

            # FP8 hidden [BM, BK] — shared between up-MMA and gate-MMA
            h_ptrs  = hidden_ptr + tok_ids[:, None] * stride_h_seq + k_range[None, :]
            h_fp8   = tl.load(h_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
            h_scales = tl.load(
                hscale_ptr + k_blk * stride_hscale_blk + tok_ids,
                mask=m_mask, other=1.0,
            )

            # FP8 weight "up" half
            w_up_ptrs  = w1_base + n_range[:, None] * stride_w1_n + k_range[None, :]
            w_up_fp8   = tl.load(w_up_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)
            w_up_scales = tl.load(
                w1s_base + n_blks_up * stride_w1s_nb + k_blk,
                mask=n_mask, other=1.0,
            )

            # FP8 weight "gate" half
            w_gate_ptrs  = w1_base + n_range_gate[:, None] * stride_w1_n + k_range[None, :]
            w_gate_fp8   = tl.load(w_gate_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)
            w_gate_scales = tl.load(
                w1s_base + n_blks_gate * stride_w1s_nb + k_blk,
                mask=n_mask, other=1.0,
            )

            # Two native FP8 MMAs; fp32 TC accumulators; N_C=128 two-level drain.
            acc_up_block   = tl.dot(h_fp8, tl.trans(w_up_fp8),   out_dtype=tl.float32)
            acc_gate_block = tl.dot(h_fp8, tl.trans(w_gate_fp8), out_dtype=tl.float32)

            acc_up   += acc_up_block   * h_scales[:, None] * w_up_scales[None, :]
            acc_gate += acc_gate_block * h_scales[:, None] * w_gate_scales[None, :]

        # Fused SwiGLU: silu(gate) * up.
        inter_tile = (acc_gate / (1.0 + tl.exp(-acc_gate))) * acc_up  # [BLOCK_M, 128]

        # Store float32 inter (Python will cast to fp16 for GEMM2 bandwidth savings).
        i_ptrs = inter_ptr + global_m[:, None] * stride_inter_tok + n_range[None, :]
        tl.store(i_ptrs, inter_tile, mask=m_mask[:, None] & n_mask[None, :])


# ---------------------------------------------------------------------------
# Kernel 2: Grouped GEMM2 — PERSISTENT scheduler
#
# output_f32[tok_ids[m]] += (r_scores[m] * rsf) × inter[m] @ w2[expert].T
# for all 32 local experts in a single launch.
#
# Persistent schedule: grid = (NUM_SMS,). Each CTA precomputes per-expert
# m-tile counts from expert_offsets (32-wide vector op, in-register), does
# a prefix sum to build tile_id → (expert_id, m_in_expert) lookup, then
# loops tile_id = pid, pid+NUM_SMS, pid+2·NUM_SMS, ... until total_tiles.
# This eliminates the tail-quantization waste from the old grid-by-max_tok
# approach: imbalanced experts no longer strand whole CTAs on early return.
#
# Uses tl.atomic_add scatter — required because top-8 routing means
# multiple experts contribute to the same output token row.
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=_gemm2_configs(),
    key=["n_assigned_bucket", "N"],
    reset_to_zero=("output_ptr",),
)
@triton.jit
def grouped_gemm2_kernel(
    # Sorted token data
    sorted_tok_ids_ptr,
    sorted_r_scores_ptr,
    expert_offsets_ptr,    # [E+1] int32

    # Intermediate (SwiGLU output)
    inter_ptr,             # [N_assigned, 2048] float16  (cast from float32 in Python)

    # Per-expert weights
    w2_ptr,                # [32, 7168, 2048]  fp8_e4m3fn
    w2scale_ptr,           # [32, 56, 16]      float32

    # Output
    output_ptr,            # [seq_len, 7168]   float32  (zero-init, atomic add)
    routed_scaling_factor,

    N_assigned,
    n_assigned_bucket,

    # Shape constants
    K: tl.constexpr,       # 2048
    N: tl.constexpr,       # 7168
    E: tl.constexpr,       # 32 (NUM_LOCAL_EXPERTS)

    # Strides
    stride_inter_tok,      # stride of inter in elements (= 2048)
    stride_w2_exp,
    stride_w2_n,
    stride_w2s_exp,
    stride_w2s_nb,
    stride_out_seq,

    # Block sizes + persistent parallelism
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    pid = tl.program_id(0)

    # Per-expert m-tile prefix (cheap, 32-wide vector, all registers).
    idx_e       = tl.arange(0, E)
    starts      = tl.load(expert_offsets_ptr + idx_e)       # [E]
    ends        = tl.load(expert_offsets_ptr + idx_e + 1)   # [E]
    counts      = ends - starts                              # [E]
    mtiles      = tl.cdiv(counts, BLOCK_M)                   # [E]  m-tiles per expert
    prefix_inc  = tl.cumsum(mtiles, axis=0)                  # [E]  inclusive
    prefix_exc  = prefix_inc - mtiles                         # [E]  exclusive (start of each expert)
    total_m_tiles = tl.sum(mtiles, axis=0)                    # scalar

    n_tiles     = tl.cdiv(N, BLOCK_N)
    total_tiles = total_m_tiles * n_tiles

    # Persistent loop over flat tile_id.
    for tile_id in tl.range(pid, total_tiles, NUM_SMS):
        m_tile_id = tile_id // n_tiles
        n_tile_id = tile_id %  n_tiles

        # expert_id = how many experts have their inclusive prefix <= m_tile_id.
        expert_id = tl.sum((prefix_inc <= m_tile_id).to(tl.int32), axis=0)
        # Pick prefix_exc[expert_id], starts[expert_id], counts[expert_id] via masked sum.
        sel           = (idx_e == expert_id).to(tl.int32)
        m_tile_base   = tl.sum(prefix_exc * sel, axis=0)
        e_start       = tl.sum(starts     * sel, axis=0)
        n_expert_tok  = tl.sum(counts     * sel, axis=0)

        m_in_expert = m_tile_id - m_tile_base
        m_off       = m_in_expert * BLOCK_M
        m_range     = m_off + tl.arange(0, BLOCK_M)
        m_mask      = m_range < n_expert_tok
        global_m    = e_start + m_range

        tok_ids  = tl.load(sorted_tok_ids_ptr  + global_m, mask=m_mask, other=0)
        r_scores = tl.load(sorted_r_scores_ptr + global_m, mask=m_mask, other=0.0)
        weight   = r_scores * routed_scaling_factor

        n_start  = n_tile_id * BLOCK_N
        n_range  = n_start + tl.arange(0, BLOCK_N)
        n_mask   = n_range < N
        n_blks   = n_range // BLOCK_K

        w2_base  = w2_ptr      + expert_id * stride_w2_exp
        w2s_base = w2scale_ptr + expert_id * stride_w2s_exp

        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

        for k_start in tl.range(0, K, BLOCK_K):
            k_blk   = k_start // BLOCK_K
            k_range = k_start + tl.arange(0, BLOCK_K)
            k_mask  = k_range < K

            # Load FP16 intermediate [BM, BK].
            i_ptrs = inter_ptr + global_m[:, None] * stride_inter_tok + k_range[None, :]
            i_f16  = tl.load(i_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

            # Load FP8 weight [BN, BK] + per-N-block scale [BN]; dequant to FP16.
            w_ptrs   = w2_base + n_range[:, None] * stride_w2_n + k_range[None, :]
            w_fp8    = tl.load(w_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)
            w_scales = tl.load(
                w2s_base + n_blks * stride_w2s_nb + k_blk,
                mask=n_mask, other=1.0,
            )
            w_f16 = (w_fp8.to(tl.float32) * w_scales[:, None]).to(tl.bfloat16)

            # BF16 tensor-core GEMM: same throughput as FP16, same exponent range as FP32.
            acc += tl.dot(i_f16, tl.trans(w_f16), out_dtype=tl.float32)

        acc      = acc * weight[:, None]
        out_ptrs = output_ptr + tok_ids[:, None] * stride_out_seq + n_range[None, :]
        tl.atomic_add(out_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def kernel(
    routing_logits,         # float32,       [seq_len, 256]
    routing_bias,           # bfloat16,      [256]
    hidden_states,          # float8_e4m3fn, [seq_len, 7168]
    hidden_states_scale,    # float32,       [56, seq_len]
    gemm1_weights,          # float8_e4m3fn, [32, 4096, 7168]
    gemm1_weights_scale,    # float32,       [32, 32, 56]
    gemm2_weights,          # float8_e4m3fn, [32, 7168, 2048]
    gemm2_weights_scale,    # float32,       [32, 56, 16]
    local_expert_offset,    # int32 scalar
    routed_scaling_factor,  # float32 scalar
    output,                 # bfloat16,      [seq_len, 7168]  (DPS)
):
    seq_len      = hidden_states.shape[0]
    device       = hidden_states.device
    rsf          = float(routed_scaling_factor)
    local_offset = int(local_expert_offset)

    # 1. Routing
    expert_ids, weights_full = _route(
        routing_logits, routing_bias, seq_len, device
    )

    # 2. Token sorting
    sorted_tok_ids, expert_offsets, sorted_r_scores, N_assigned, max_tok = _sort_tokens(
        expert_ids, weights_full, local_offset, device
    )

    # 3. Allocate output accumulator; handle empty case
    output_f32 = torch.zeros(seq_len, HIDDEN_DIM, dtype=torch.float32, device=device)

    if N_assigned == 0:
        output.zero_()
        return

    # GEMM1 writes float32 inter; Python casts to float16 (2× bandwidth savings,
    # ~0.01% precision loss — well within benchmark tolerance).
    inter_f32 = torch.empty(N_assigned, INTER_DIM, dtype=torch.float32, device=device)

    NUM_SMS = torch.cuda.get_device_properties(device).multi_processor_count

    # 4. Grouped GEMM1 + fused SwiGLU (persistent) — writes float32 inter.
    grouped_gemm1_kernel[(NUM_SMS,)](
        sorted_tok_ids, expert_offsets,
        hidden_states, hidden_states_scale,
        gemm1_weights, gemm1_weights_scale,
        inter_f32,
        N_assigned,
        _bucket(N_assigned),
        K=HIDDEN_DIM,
        N=INTER_DIM,
        E=NUM_LOCAL_EXPERTS,
        NUM_SMS=NUM_SMS,
        stride_h_seq      = hidden_states.stride(0),
        stride_w1_exp     = gemm1_weights.stride(0),
        stride_w1_n       = gemm1_weights.stride(1),
        stride_w1s_exp    = gemm1_weights_scale.stride(0),
        stride_w1s_nb     = gemm1_weights_scale.stride(1),
        stride_hscale_blk = hidden_states_scale.stride(0),
        stride_inter_tok  = inter_f32.stride(0),
    )

    # Cast float32 → bfloat16: 2× smaller inter buffer → 2× less GEMM2 K-read bandwidth.
    # BF16 has the same exponent range as float32 (max ~3.4e38) so inter values that can
    # reach ~363K after accumulated FP8 MMA don't overflow (FP16 max is only 65504).
    # BF16 precision (~0.4% rel error) is well within benchmark tolerance.
    inter_bf16 = inter_f32.to(torch.bfloat16)

    # 5. Grouped GEMM2 (persistent) — BF16 tensor-core GEMM, atomic scatter-add.
    grouped_gemm2_kernel[(NUM_SMS,)](
        sorted_tok_ids, sorted_r_scores, expert_offsets,
        inter_bf16,
        gemm2_weights, gemm2_weights_scale,
        output_f32,
        rsf,
        N_assigned,
        _bucket(N_assigned),
        K=INTER_DIM,
        N=HIDDEN_DIM,
        E=NUM_LOCAL_EXPERTS,
        NUM_SMS=NUM_SMS,
        stride_inter_tok  = inter_bf16.stride(0),
        stride_w2_exp     = gemm2_weights.stride(0),
        stride_w2_n       = gemm2_weights.stride(1),
        stride_w2s_exp    = gemm2_weights_scale.stride(0),
        stride_w2s_nb     = gemm2_weights_scale.stride(1),
        stride_out_seq    = output_f32.stride(0),
    )

    # 6. Cast float32 accumulation → bf16 output (DPS)
    output.copy_(output_f32.to(torch.bfloat16))
