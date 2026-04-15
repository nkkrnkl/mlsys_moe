"""
Baseline Fused MoE Kernel for FlashInfer Competition.

Implements DeepSeek-V3 style FP8 block-scale MoE with no-aux routing.
This baseline uses PyTorch operations for correctness; will be optimized
with Triton JIT kernels in subsequent iterations.

Architecture:
  1. FP8 block-scale dequantization (block_size=128)
  2. DeepSeek-V3 no-aux routing (topk=8, n_group=8, topk_group=4)
  3. Per-expert: GEMM1 → SwiGLU → GEMM2
  4. Weighted accumulation into output
"""

import torch


def kernel(
    routing_logits: torch.Tensor,       # [seq_len, num_experts=256]    float32
    routing_bias: torch.Tensor,         # [num_experts=256]             bfloat16
    hidden_states: torch.Tensor,        # [seq_len, hidden_size=7168]   float8_e4m3fn
    hidden_states_scale: torch.Tensor,  # [56, seq_len]                 float32
    gemm1_weights: torch.Tensor,        # [32, 4096, 7168]              float8_e4m3fn
    gemm1_weights_scale: torch.Tensor,  # [32, 32, 56]                  float32
    gemm2_weights: torch.Tensor,        # [32, 7168, 2048]              float8_e4m3fn
    gemm2_weights_scale: torch.Tensor,  # [32, 56, 16]                  float32
    local_expert_offset: int,           # scalar int32
    routed_scaling_factor: float,       # scalar float32
    output: torch.Tensor,              # [seq_len, hidden_size=7168]   bfloat16 (DPS)
):
    """
    Fused MoE kernel with DPS (Destination Passing Style).
    All computation results are written into the pre-allocated `output` tensor.
    """
    # ── Constants ──
    BLOCK = 128
    H = 7168             # hidden_size
    I = 2048             # intermediate_size
    TOP_K = 8
    N_GROUP = 8
    TOPK_GROUP = 4

    E_local = gemm1_weights.shape[0]   # 32
    E_global = routing_logits.shape[1]  # 256
    T = routing_logits.shape[0]         # seq_len

    device = hidden_states.device

    # ── 1) FP8 block-scale dequantization ──
    # Hidden states: [T, H] fp8 × [H/128, T] scales → [T, H] float32
    A_fp32 = hidden_states.to(torch.float32)
    A_scale = hidden_states_scale.to(torch.float32)          # [H/128, T]
    A_scale_TH = A_scale.permute(1, 0).contiguous()          # [T, H/128]
    A_scale_expanded = (
        A_scale_TH.unsqueeze(-1)
        .repeat(1, 1, BLOCK)                                 # [T, H/128, 128]
        .reshape(T, H)                                       # [T, H]
        .contiguous()
    )
    A = A_fp32 * A_scale_expanded                            # [T, H]

    # GEMM1 weights: [E, 2I, H] fp8 × [E, 2I/128, H/128] scales → [E, 2I, H] float32
    W13_fp32 = gemm1_weights.to(torch.float32)
    S13 = gemm1_weights_scale.to(torch.float32)
    S13_expanded = torch.repeat_interleave(S13, BLOCK, dim=1)      # [E, 2I, H/128]
    S13_expanded = torch.repeat_interleave(S13_expanded, BLOCK, dim=2)  # [E, 2I, H]
    W13 = W13_fp32 * S13_expanded

    # GEMM2 weights: [E, H, I] fp8 × [E, H/128, I/128] scales → [E, H, I] float32
    W2_fp32 = gemm2_weights.to(torch.float32)
    S2 = gemm2_weights_scale.to(torch.float32)
    S2_expanded = torch.repeat_interleave(S2, BLOCK, dim=1)        # [E, H, I/128]
    S2_expanded = torch.repeat_interleave(S2_expanded, BLOCK, dim=2)  # [E, H, I]
    W2 = W2_fp32 * S2_expanded

    # ── 2) DeepSeek-V3 no-aux routing ──
    logits = routing_logits.to(torch.float32)                # [T, E_global]
    bias = routing_bias.to(torch.float32).reshape(-1)        # [E_global]

    # Sigmoid scores
    s = 1.0 / (1.0 + torch.exp(-logits))                    # [T, E_global]
    s_with_bias = s + bias                                   # [T, E_global]

    # Group experts: 256 experts → 8 groups of 32
    group_size = E_global // N_GROUP                         # 32
    s_wb_grouped = s_with_bias.view(T, N_GROUP, group_size)  # [T, 8, 32]

    # Group scores = sum of top-2 values within each group
    top2_vals, _ = torch.topk(s_wb_grouped, k=2, dim=2, largest=True, sorted=False)
    group_scores = top2_vals.sum(dim=2)                      # [T, 8]

    # Select top-4 groups
    _, group_idx = torch.topk(group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=False)
    group_mask = torch.zeros_like(group_scores)              # [T, 8]
    group_mask.scatter_(1, group_idx, 1.0)
    score_mask = (
        group_mask.unsqueeze(2)
        .expand(T, N_GROUP, group_size)
        .reshape(T, E_global)
    )                                                        # [T, E_global]

    # Global top-8 experts from kept groups
    neg_inf = torch.finfo(torch.float32).min
    scores_pruned = s_with_bias.masked_fill(score_mask == 0, neg_inf)
    _, topk_idx = torch.topk(scores_pruned, k=TOP_K, dim=1, largest=True, sorted=False)

    # Combination weights: use s (without bias), normalized, scaled
    M = torch.zeros_like(s)
    M.scatter_(1, topk_idx, 1.0)
    weights = s * M                                          # [T, E_global]
    weights_sum = weights.sum(dim=1, keepdim=True) + 1e-20
    weights = (weights / weights_sum) * routed_scaling_factor

    # ── 3) Per-expert compute + accumulation ──
    result = torch.zeros((T, H), dtype=torch.float32, device=device)
    local_start = int(local_expert_offset)

    for le in range(E_local):
        ge = local_start + le
        if ge < 0 or ge >= E_global:
            continue

        # Find tokens that selected this expert
        sel_mask = (topk_idx == ge).any(dim=1)               # [T] bool
        if not sel_mask.any():
            continue

        token_idx = torch.nonzero(sel_mask, as_tuple=False).squeeze(1)

        # Gather inputs for this expert
        A_e = A.index_select(0, token_idx)                   # [Tk, H]
        W13_e = W13[le]                                      # [2I, H]
        W2_e = W2[le]                                        # [H, I]

        # GEMM1: [Tk, H] @ [H, 2I] → [Tk, 2I]
        G1 = A_e.matmul(W13_e.t())

        # SwiGLU: silu(second_half) * first_half
        X1 = G1[:, :I]                                       # gate projection
        X2 = G1[:, I:]                                       # up projection
        silu_X2 = X2 / (1.0 + torch.exp(-X2))
        C = silu_X2 * X1                                     # [Tk, I]

        # GEMM2: [Tk, I] @ [I, H] → [Tk, H]
        O = C.matmul(W2_e.t())

        # Weighted accumulation
        w_tok = weights.index_select(0, token_idx)[:, ge]    # [Tk]
        result.index_add_(0, token_idx, O * w_tok.unsqueeze(1))

    # ── DPS: write into pre-allocated output ──
    output.copy_(result.to(torch.bfloat16))
