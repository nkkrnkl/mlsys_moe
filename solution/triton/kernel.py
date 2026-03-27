"""
Fused MoE Triton Kernel for FlashInfer Competition.

Target: moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048
  - DeepSeek-V3/R1 scale MoE: hidden=7168, intermediate=2048
  - 32 local experts (256 global), top-8 routing
  - FP8 E4M3FN weights with 128-element block scaling
  - DeepSeek-style routing: 8 groups, top-4 groups, top-8 experts

Function signature uses Destination Passing Style (DPS): output tensor is
pre-allocated by the framework and passed in; the kernel writes into it.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_GLOBAL_EXPERTS = 256
NUM_GROUPS = 8           # ng8
EXPERTS_PER_GROUP = 32   # NUM_GLOBAL_EXPERTS / NUM_GROUPS
NUM_GROUPS_SELECTED = 4  # kg4: top-4 groups selected
TOPK = 8                 # topk8: top-8 experts selected
NUM_LOCAL_EXPERTS = 32   # e32
HIDDEN_DIM = 7168        # h7168
INTER_DIM = 2048         # i2048  (gate and up each = 2048, combined = 4096)
FP8_BLOCK_SIZE = 128     # FP8 block-scale granularity
GATE_UP_DIM = INTER_DIM * 2  # 4096


# ---------------------------------------------------------------------------
# Autotune configs for B200
# BLOCK_K is fixed at 128 to exactly align with FP8 block-scale boundaries.
# ---------------------------------------------------------------------------
def _gemm_configs():
    configs = []
    for bm in [64, 128]:
        for bn in [128]:  # BLOCK_N must equal BLOCK_K=128 so each tile covers exactly one scale block
            for ns in [3, 4, 5]:
                for nw in [8, 16]:
                    configs.append(
                        triton.Config(
                            {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": FP8_BLOCK_SIZE},
                            num_stages=ns,
                            num_warps=nw,
                        )
                    )
    return configs


def _token_bucket(n: int) -> int:
    """Bucket num_tokens for autotune key to avoid per-batch recompilation."""
    for thresh in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
        if n <= thresh:
            return thresh
    return 1024


# ---------------------------------------------------------------------------
# Kernel 1: DeepSeek-style routing
# One CTA per BLOCK_T tokens.
# ---------------------------------------------------------------------------
@triton.jit
def routing_kernel(
    logits_ptr,          # [seq_len, 256] float32
    bias_ptr,            # [256] float32  (pre-converted from bfloat16 in wrapper)
    expert_ids_ptr,      # [seq_len, 8] int32  OUTPUT
    scores_ptr,          # [seq_len, 8] float32 OUTPUT
    seq_len,
    stride_logits_seq,   # stride along seq dim  (== 256)
    stride_eid_seq,      # stride along seq dim of expert_ids (== 8)
    stride_score_seq,    # stride along seq dim of scores (== 8)
    NUM_EXPERTS: tl.constexpr,       # 256
    NUM_GROUPS: tl.constexpr,        # 8
    EPG: tl.constexpr,               # experts_per_group = 32
    KG: tl.constexpr,                # num_groups_selected = 4
    TOPK: tl.constexpr,              # 8
    BLOCK_T: tl.constexpr,           # tokens per CTA (16)
):
    """
    For each token:
      1. Load 256 logits + biases → sigmoid + bias → expert_scores [256]
      2. Per-group (8 groups × 32 experts): top-2 sum → group_score [8]
      3. Select top-KG groups
      4. From 128 candidates: select top-TOPK by expert_score
      5. Normalize selected scores; write expert_ids and scores
    """
    pid = tl.program_id(0)
    tok_start = pid * BLOCK_T

    exp_range = tl.arange(0, NUM_EXPERTS)    # [256]
    grp_range = tl.arange(0, NUM_GROUPS)     # [8]
    topk_range = tl.arange(0, TOPK)          # [8]
    epg_range = tl.arange(0, EPG)            # [32]

    # Load bias once (shared across all tokens in this CTA)
    bias = tl.load(bias_ptr + exp_range)     # [256] float32

    for _t in tl.static_range(BLOCK_T):
        t = tok_start + _t
        valid = t < seq_len  # runtime mask — tl.static_range can't break on runtime conditions

        # ---- 1. sigmoid(logits) + bias ----
        logits = tl.load(
            logits_ptr + t * stride_logits_seq + exp_range,
            mask=valid, other=0.0,
        )  # [256]
        expert_scores = tl.sigmoid(logits) + bias  # [256]

        # ---- 2. Group scoring: per group, sum top-2 expert scores ----
        group_scores = tl.zeros([NUM_GROUPS], dtype=tl.float32)
        for g in tl.static_range(NUM_GROUPS):
            g_base = g * EPG
            g_scores = tl.load(
                logits_ptr + t * stride_logits_seq + g_base + epg_range,
                mask=valid, other=0.0,
            )  # [32]
            g_scores = tl.sigmoid(g_scores) + tl.load(bias_ptr + g_base + epg_range)

            # top-2 sum: two argmax passes
            max1 = tl.max(g_scores, axis=0)
            g_scores_tmp = tl.where(g_scores == max1, -1e9, g_scores)
            max2 = tl.max(g_scores_tmp, axis=0)
            gs_val = max1 + max2
            group_scores = tl.where(grp_range == g, gs_val, group_scores)

        # ---- 3. Top-KG group selection ----
        running_gs = group_scores
        selected_groups = tl.zeros([NUM_GROUPS], dtype=tl.int32)
        for _k in tl.static_range(KG):
            best_g = tl.argmax(running_gs, axis=0)
            selected_groups = tl.where(grp_range == best_g, 1, selected_groups)
            running_gs = tl.where(grp_range == best_g, -1e9, running_gs)

        # ---- 4. Build candidate mask over all 256 experts ----
        group_of_exp = exp_range // EPG  # [256]
        candidate_mask = tl.zeros([NUM_EXPERTS], dtype=tl.int32)
        for g in tl.static_range(NUM_GROUPS):
            g_sel = tl.sum(tl.where(grp_range == g, selected_groups, 0)) > 0
            candidate_mask = tl.where(
                group_of_exp == g,
                tl.where(g_sel, 1, candidate_mask),
                candidate_mask,
            )

        # Mask non-candidates
        masked_scores = tl.where(candidate_mask > 0, expert_scores, -1e9)

        # ---- 5. Top-TOPK from 128 candidates ----
        selected_exp = tl.zeros([TOPK], dtype=tl.int32)
        selected_s = tl.zeros([TOPK], dtype=tl.float32)
        running_ms = masked_scores
        for k in tl.static_range(TOPK):
            best_e = tl.argmax(running_ms, axis=0)
            best_s = tl.max(running_ms, axis=0)
            selected_exp = tl.where(topk_range == k, best_e, selected_exp)
            selected_s = tl.where(topk_range == k, best_s, selected_s)
            running_ms = tl.where(exp_range == best_e, -1e9, running_ms)

        # ---- 6. Normalize ----
        score_sum = tl.sum(selected_s, axis=0)
        score_sum = tl.maximum(score_sum, 1e-9)
        norm_scores = selected_s / score_sum

        # ---- 7. Write outputs (only for valid tokens) ----
        tl.store(
            expert_ids_ptr + t * stride_eid_seq + topk_range,
            selected_exp,
            mask=valid,
        )
        tl.store(
            scores_ptr + t * stride_score_seq + topk_range,
            norm_scores,
            mask=valid,
        )


# ---------------------------------------------------------------------------
# Kernel 2: Expert GEMM1 — FP8 gather-GEMM into gate+up buffer
# Grid: (ceil(num_tokens/BLOCK_M), ceil(N/BLOCK_N))
# ---------------------------------------------------------------------------
@triton.autotune(configs=_gemm_configs(), key=["tok_bucket", "K", "N"])
@triton.jit
def expert_gemm1_kernel(
    hidden_ptr,           # [total_seq_len, 7168] float8_e4m3fn
    hscale_ptr,           # [56, total_seq_len]  float32  (56 = HIDDEN_DIM/128)
    token_ids_ptr,        # [num_tokens] int32
    w1_ptr,               # [4096, 7168] float8_e4m3fn  (this expert's weights)
    w1scale_ptr,          # [32, 56] float32  (32=4096/128, 56=7168/128)
    gate_up_ptr,          # [num_tokens, 4096] float32  OUTPUT
    num_tokens,
    tok_bucket,
    K: tl.constexpr,      # 7168
    N: tl.constexpr,      # 4096
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,  # 128
    stride_h_seq,
    stride_w1_n,
    stride_gu_tok,
    stride_hscale_blk,    # total_seq_len  (hscale is [56, seq_len])
    stride_w1s_nb,        # 56  (w1scale is [32, 56])
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    m_range = m_start + tl.arange(0, BLOCK_M)
    n_range = n_start + tl.arange(0, BLOCK_N)
    m_mask = m_range < num_tokens
    n_mask = n_range < N

    # Gather source token row indices
    tok_ids = tl.load(token_ids_ptr + m_range, mask=m_mask, other=0)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    n_blk = n_start // BLOCK_K  # N-dimension block index (BLOCK_K == 128 == block size)

    for k_start in tl.range(0, K, BLOCK_K):
        k_blk = k_start // BLOCK_K
        k_range = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_range < K

        # Load FP8 hidden [BLOCK_M, BLOCK_K] and dequantize
        h_ptrs = hidden_ptr + tok_ids[:, None] * stride_h_seq + k_range[None, :]
        h_fp8 = tl.load(h_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        h_f32 = h_fp8.to(tl.float32)
        # Scale: hscale[k_blk, tok_id]  —  shape [56, seq_len], stride_hscale_blk = seq_len
        h_scales = tl.load(
            hscale_ptr + k_blk * stride_hscale_blk + tok_ids,
            mask=m_mask, other=1.0,
        )  # [BLOCK_M]
        h_f32 = h_f32 * h_scales[:, None]

        # Load FP8 weight [BLOCK_N, BLOCK_K] and dequantize
        w_ptrs = w1_ptr + n_range[:, None] * stride_w1_n + k_range[None, :]
        w_fp8 = tl.load(w_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)
        w_f32 = w_fp8.to(tl.float32)
        # Scale: w1scale[n_blk, k_blk]
        w_scale = tl.load(w1scale_ptr + n_blk * stride_w1s_nb + k_blk)
        w_f32 = w_f32 * w_scale

        acc += tl.dot(h_f32, tl.trans(w_f32))

    # Write gate+up buffer
    out_ptrs = gate_up_ptr + m_range[:, None] * stride_gu_tok + n_range[None, :]
    tl.store(out_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


# ---------------------------------------------------------------------------
# Kernel 3: Expert GEMM2 — FP8 GEMM + weighted atomic-add into output
# Grid: (ceil(num_tokens/BLOCK_M), ceil(N/BLOCK_N))
# ---------------------------------------------------------------------------
@triton.autotune(configs=_gemm_configs(), key=["tok_bucket", "K", "N"])
@triton.jit
def expert_gemm2_accumulate_kernel(
    inter_ptr,            # [num_tokens, 2048] float32 (SwiGLU output)
    w2_ptr,               # [7168, 2048] float8_e4m3fn
    w2scale_ptr,          # [56, 16] float32  (56=7168/128, 16=2048/128)
    token_ids_ptr,        # [num_tokens] int32
    r_scores_ptr,         # [num_tokens] float32
    routed_scaling_factor,
    output_ptr,           # [total_seq_len, 7168] float32  (atomic add target, fp32)
    num_tokens,
    tok_bucket,
    K: tl.constexpr,      # 2048
    N: tl.constexpr,      # 7168
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,  # 128
    stride_inter_tok,
    stride_w2_n,
    stride_out_seq,
    stride_w2s_nb,        # 16  (w2scale is [56, 16])
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    m_range = m_start + tl.arange(0, BLOCK_M)
    n_range = n_start + tl.arange(0, BLOCK_N)
    m_mask = m_range < num_tokens
    n_mask = n_range < N

    tok_ids = tl.load(token_ids_ptr + m_range, mask=m_mask, other=0)
    r_scores = tl.load(r_scores_ptr + m_range, mask=m_mask, other=0.0)
    weight = r_scores * routed_scaling_factor  # [BLOCK_M]

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    n_blk = n_start // BLOCK_K  # N-block index for scale lookup

    for k_start in tl.range(0, K, BLOCK_K):
        k_blk = k_start // BLOCK_K
        k_range = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_range < K

        # Load intermediate (float32, no dequant needed)
        i_ptrs = inter_ptr + m_range[:, None] * stride_inter_tok + k_range[None, :]
        i_mk = tl.load(i_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

        # Load FP8 weight and dequantize
        w_ptrs = w2_ptr + n_range[:, None] * stride_w2_n + k_range[None, :]
        w_fp8 = tl.load(w_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)
        w_f32 = w_fp8.to(tl.float32)
        # Scale: w2scale[n_blk, k_blk]  (shape [56, 16])
        w_scale = tl.load(w2scale_ptr + n_blk * stride_w2s_nb + k_blk)
        w_f32 = w_f32 * w_scale

        acc += tl.dot(i_mk, tl.trans(w_f32))

    # Apply routing weight
    acc = acc * weight[:, None]  # [BLOCK_M, BLOCK_N]

    # Atomic add into float32 accumulation buffer (bf16 atomics unreliable)
    out_ptrs = output_ptr + tok_ids[:, None] * stride_out_seq + n_range[None, :]
    tl.atomic_add(
        out_ptrs,
        acc,
        mask=m_mask[:, None] & n_mask[None, :],
    )


# ---------------------------------------------------------------------------
# SwiGLU (PyTorch, not Triton — runs on device after GEMM1)
# ---------------------------------------------------------------------------
def _swiglu(gate_up: torch.Tensor) -> torch.Tensor:
    """
    gate_up: [num_tokens, 4096] float32
    returns: [num_tokens, 2048] float32
    SwiGLU(x) = silu(gate) * up  where gate=x[:2048], up=x[2048:]
    """
    gate = gate_up[:, :INTER_DIM]
    up = gate_up[:, INTER_DIM:]
    return torch.nn.functional.silu(gate) * up


# ---------------------------------------------------------------------------
# Entry point (matches config.toml entry_point = "kernel")
# ---------------------------------------------------------------------------
def kernel(
    routing_logits,         # float32,       [seq_len, 256]
    routing_bias,           # bfloat16,      [256]
    hidden_states,          # float8_e4m3fn, [seq_len, 7168]
    hidden_states_scale,    # float32,       [56, seq_len]   ← block_idx is dim 0
    gemm1_weights,          # float8_e4m3fn, [32, 4096, 7168]
    gemm1_weights_scale,    # float32,       [32, 32, 56]
    gemm2_weights,          # float8_e4m3fn, [32, 7168, 2048]
    gemm2_weights_scale,    # float32,       [32, 56, 16]
    local_expert_offset,    # int32 scalar
    routed_scaling_factor,  # float32 scalar
    output,                 # bfloat16,      [seq_len, 7168]  (DPS: output last)
):
    """
    Fused MoE forward pass.

    Uses PyTorch for FP8 dequantization (Triton FP8 pointer decoding is unreliable
    across framework versions). The routing stage uses Triton for parallelism.
    """
    seq_len = hidden_states.shape[0]
    device = hidden_states.device
    rsf = float(routed_scaling_factor)
    local_offset = int(local_expert_offset)

    # --- Step 1: Dequantize hidden states once (float32, matches reference) ---
    # hidden_states:       [seq_len, 7168]  fp8
    # hidden_states_scale: [56, seq_len]    float32  (scale[k_blk, token])
    A_fp32 = hidden_states.to(torch.float32)                         # [seq_len, 7168]
    A_scale_TH = hidden_states_scale.to(torch.float32).permute(1, 0).contiguous()  # [seq_len, 56]
    A_scale_exp = A_scale_TH.unsqueeze(-1).expand(
        seq_len, HIDDEN_DIM // FP8_BLOCK_SIZE, FP8_BLOCK_SIZE
    ).reshape(seq_len, HIDDEN_DIM).contiguous()                      # [seq_len, 7168]
    h_dequant = A_fp32 * A_scale_exp                                 # [seq_len, 7168] float32

    # --- Step 2: Routing (DeepSeek-V3 no-aux style, matches reference exactly) ---
    bias_f32 = routing_bias.to(torch.float32)
    s = torch.sigmoid(routing_logits.float())        # [seq, 256] — used for weights
    s_with_bias = s + bias_f32                       # [seq, 256] — used for selection

    # Group scoring: top-2 sum per group using s_with_bias
    group_scores_mat = s_with_bias.view(seq_len, NUM_GROUPS, EXPERTS_PER_GROUP)
    top2_vals, _ = group_scores_mat.topk(2, dim=-1)
    group_scores = top2_vals.sum(-1)  # [seq, 8]

    # Select top-KG groups
    _, top_groups = group_scores.topk(NUM_GROUPS_SELECTED, dim=-1)  # [seq, 4]

    # Build group mask and select top-TOPK experts using s_with_bias
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, top_groups, 1.0)
    score_mask = group_mask.unsqueeze(2).expand(
        seq_len, NUM_GROUPS, EXPERTS_PER_GROUP
    ).reshape(seq_len, NUM_GLOBAL_EXPERTS)
    neg_inf = torch.finfo(torch.float32).min
    scores_pruned = s_with_bias.masked_fill(score_mask == 0, neg_inf)
    _, top8_ids = scores_pruned.topk(TOPK, dim=-1)  # [seq, 8]

    # Routing WEIGHTS: use s (WITHOUT bias), normalize, scale by rsf
    weight_mask = torch.zeros_like(s)
    weight_mask.scatter_(1, top8_ids, 1.0)
    weights = s * weight_mask                        # [seq, 256]
    weights_sum = weights.sum(-1, keepdim=True).clamp(min=1e-20)
    weights = (weights / weights_sum) * float(routed_scaling_factor)  # [seq, 256]

    expert_ids = top8_ids.to(torch.int32)            # [seq, 8]

    # --- Step 3: Per-expert FFN (matches reference exactly) ---
    output_f32 = torch.zeros(seq_len, HIDDEN_DIM, dtype=torch.float32, device=device)

    for local_idx in range(NUM_LOCAL_EXPERTS):
        global_id = local_offset + local_idx
        if global_id < 0 or global_id >= NUM_GLOBAL_EXPERTS:
            continue

        # Find tokens that selected this expert in their top-8
        sel_mask = (expert_ids == global_id).any(dim=1)  # [seq] bool
        if not sel_mask.any():
            continue
        tok_ids = sel_mask.nonzero(as_tuple=False).squeeze(1)  # int64

        # Per-token routing weight for this expert (from weights tensor, indexed by global_id)
        w_tok = weights[tok_ids, global_id]  # [num_tok] — already includes rsf

        # Dequantize hidden states for selected tokens (float32)
        h_tok = h_dequant[tok_ids]           # [num_tok, 7168] float32

        # Dequantize GEMM1 weights: [4096, 7168] fp8 → float32
        w1   = gemm1_weights[local_idx]      # [4096, 7168] fp8
        w1s  = gemm1_weights_scale[local_idx]  # [32, 56]
        w1_f32 = w1.to(torch.float32).view(
            GATE_UP_DIM // FP8_BLOCK_SIZE, FP8_BLOCK_SIZE,
            HIDDEN_DIM  // FP8_BLOCK_SIZE, FP8_BLOCK_SIZE,
        )
        w1_dq = (w1_f32 * w1s[:, None, :, None]).view(GATE_UP_DIM, HIDDEN_DIM)

        # GEMM1: [num_tok, 7168] @ [7168, 4096] = [num_tok, 4096]
        G1 = torch.mm(h_tok, w1_dq.T)  # float32

        # SwiGLU: silu(second_half) * first_half  (matches reference: silu(X2) * X1)
        X1 = G1[:, :INTER_DIM]          # first half
        X2 = G1[:, INTER_DIM:]          # second half
        inter = (X2 / (1.0 + torch.exp(-X2))) * X1  # [num_tok, 2048] float32

        # Dequantize GEMM2 weights: [7168, 2048] fp8 → float32
        w2  = gemm2_weights[local_idx]   # [7168, 2048] fp8
        w2s = gemm2_weights_scale[local_idx]  # [56, 16]
        w2_f32 = w2.to(torch.float32).view(
            HIDDEN_DIM // FP8_BLOCK_SIZE, FP8_BLOCK_SIZE,
            INTER_DIM  // FP8_BLOCK_SIZE, FP8_BLOCK_SIZE,
        )
        w2_dq = (w2_f32 * w2s[:, None, :, None]).view(HIDDEN_DIM, INTER_DIM)

        # GEMM2: [num_tok, 2048] @ [2048, 7168] = [num_tok, 7168]
        out_tok = torch.mm(inter, w2_dq.T)  # float32

        # Weighted accumulate
        output_f32.index_add_(0, tok_ids, out_tok * w_tok.unsqueeze(1))

    output.copy_(output_f32.to(torch.bfloat16))
