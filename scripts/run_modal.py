"""
FlashInfer-Bench Modal Cloud Benchmark Runner.

Automatically packs the solution from source files and runs benchmarks
on NVIDIA B200 GPUs via Modal.

Setup (one-time):
    modal setup
    modal volume create flashinfer-trace
    modal volume put flashinfer-trace /path/to/flashinfer-trace/
"""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal
from flashinfer_bench import Benchmark, BenchmarkConfig, Solution, TraceSet

app = modal.App("flashinfer-bench")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_SET_PATH = "/data"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("flashinfer-bench", "torch", "triton==3.5.1", "numpy")
)


@app.function(image=image, gpu="B200:1", timeout=3600, volumes={TRACE_SET_PATH: trace_volume})
def run_benchmark(solution: Solution, config: BenchmarkConfig = None) -> dict:
    """Run benchmark on Modal B200 and return results."""
    if config is None:
        config = BenchmarkConfig(warmup_runs=3, iterations=100, num_trials=5)

    trace_set = TraceSet.from_path(TRACE_SET_PATH)

    if solution.definition not in trace_set.definitions:
        raise ValueError(f"Definition '{solution.definition}' not found in trace set")

    definition = trace_set.definitions[solution.definition]
    workloads = trace_set.workloads.get(solution.definition, [])

    if not workloads:
        raise ValueError(f"No workloads found for definition '{solution.definition}'")

    bench_trace_set = TraceSet(
        root=trace_set.root,
        definitions={definition.name: definition},
        solutions={definition.name: [solution]},
        workloads={definition.name: workloads},
        traces={definition.name: []},
    )

    benchmark = Benchmark(bench_trace_set, config)
    result_trace_set = benchmark.run_all(dump_traces=True)

    traces = result_trace_set.traces.get(definition.name, [])
    results = {definition.name: {}}

    for trace in traces:
        if trace.evaluation:
            entry = {
                "status": trace.evaluation.status.value,
                "solution": trace.solution,
            }
            if trace.evaluation.performance:
                entry["latency_ms"] = trace.evaluation.performance.latency_ms
                entry["reference_latency_ms"] = trace.evaluation.performance.reference_latency_ms
                entry["speedup_factor"] = trace.evaluation.performance.speedup_factor
            if trace.evaluation.correctness:
                entry["max_abs_error"] = trace.evaluation.correctness.max_absolute_error
                entry["max_rel_error"] = trace.evaluation.correctness.max_relative_error
            results[definition.name][trace.workload.uuid] = entry

    return results


def print_results(results: dict):
    """Print benchmark results in a formatted way."""
    for def_name, traces in results.items():
        print(f"\n{def_name}:")
        for workload_uuid, result in traces.items():
            status = result.get("status")
            print(f"  Workload {workload_uuid[:8]}...: {status}", end="")

            if result.get("latency_ms") is not None:
                print(f" | {result['latency_ms']:.3f} ms", end="")

            if result.get("speedup_factor") is not None:
                print(f" | {result['speedup_factor']:.2f}x speedup", end="")

            if result.get("max_abs_error") is not None:
                abs_err = result["max_abs_error"]
                rel_err = result.get("max_rel_error", 0)
                print(f" | abs_err={abs_err:.2e}, rel_err={rel_err:.2e}", end="")

            print()


@app.function(image=image, gpu="B200:1", timeout=600, volumes={TRACE_SET_PATH: trace_volume})
def run_ncu_on_modal(solution: Solution, workload_uuid: str, ncu_set: str = "detailed") -> str:
    """
    Run Nsight Compute profiling on a specific workload and return raw NCU text.
    Used by the agent optimization loop to get performance metrics.
    """
    from flashinfer_bench.agents import flashinfer_bench_run_ncu

    trace_set = TraceSet.from_path(TRACE_SET_PATH)

    if solution.definition not in trace_set.definitions:
        raise ValueError(f"Definition '{solution.definition}' not found in trace set")

    workloads = trace_set.workloads.get(solution.definition, [])
    workload = next(
        (w for w in workloads if str(w.uuid) == workload_uuid),
        None,
    )
    if workload is None:
        # Fall back to first workload if UUID not found
        if not workloads:
            raise ValueError(f"No workloads for '{solution.definition}'")
        workload = workloads[0]

    return flashinfer_bench_run_ncu(
        solution=solution,
        workload=workload,
        set=ncu_set,
        page="details",
    )


@app.function(image=image, gpu="B200:1", timeout=300, volumes={TRACE_SET_PATH: trace_volume})
def debug_kernel(solution: Solution, workload_uuid: str = None) -> str:
    """Load a real workload, inspect tensor shapes/values, run kernel, compare to reference."""
    import traceback
    import torch
    import importlib, sys, tempfile, os
    from flashinfer_bench import TraceSet, Benchmark, BenchmarkConfig
    from flashinfer_bench.data import Solution as Sol

    # Load kernel from solution
    tmpdir = tempfile.mkdtemp()
    src_file = os.path.join(tmpdir, "kernel.py")
    kernel_source = next(
        f.content for f in solution.sources if f.path.endswith("kernel.py")
    )
    with open(src_file, "w") as f:
        f.write(kernel_source)
    sys.path.insert(0, tmpdir)
    mod = importlib.import_module("kernel")

    # Load a specific workload by UUID, or default to first
    trace_set = TraceSet.from_path(TRACE_SET_PATH)
    workloads = trace_set.workloads.get(solution.definition, [])
    if not workloads:
        return "No workloads found"
    if workload_uuid is not None:
        matches = [w for w in workloads if str(w.workload.uuid).startswith(workload_uuid)]
        if not matches:
            return f"No workload matching prefix '{workload_uuid}' (found {len(workloads)} total)"
        workload = matches[0]
    else:
        workload = workloads[0]

    lines = []
    try:
        actual_wl = workload.workload
        lines.append(f"uuid: {actual_wl.uuid}, axes: {actual_wl.axes}")

        # Load kernel
        tmpdir = tempfile.mkdtemp()
        kernel_source = next(f.content for f in solution.sources if f.path.endswith("kernel.py"))
        with open(os.path.join(tmpdir, "kernel.py"), "w") as f:
            f.write(kernel_source)
        sys.path.insert(0, tmpdir)
        mod = importlib.import_module("kernel")

        # ---- Load REAL workload inputs (mix of safetensors / scalar / random) ----
        from flashinfer_bench.bench.utils import gen_inputs, load_safetensors
        definition = trace_set.definitions[solution.definition]
        safe_cpu = load_safetensors(definition, actual_wl, trace_set_root=trace_set.root)
        inputs_list = gen_inputs(definition, actual_wl, device="cuda", safe_tensors=safe_cpu)
        input_names = list(definition.inputs.keys())
        lines.append(f"\n=== Input names (definition order): {input_names}")

        tensors = {}
        scalars = {}
        for name, val in zip(input_names, inputs_list):
            if isinstance(val, torch.Tensor):
                tensors[name] = val
                lines.append(f"  {name}: shape={tuple(val.shape)}, dtype={val.dtype}, "
                             f"min={float(val.float().min()):.4e}, max={float(val.float().max()):.4e}")
            else:
                scalars[name] = val
                lines.append(f"  {name}: scalar={val}")

        local_expert_offset = scalars.get("local_expert_offset", 192)
        routed_scaling_factor = scalars.get("routed_scaling_factor", 2.5)
        lines.append(f"\n  local_expert_offset={local_expert_offset}, routed_scaling_factor={routed_scaling_factor}")

        # ---- (b') Pure-PyTorch emulation of the kernel's algorithm ----
        # Uses mod._route + mod._sort_tokens (same as kernel) + manual fp32 linear-dequant GEMMs.
        # If this matches contest ref, the bug is in Triton. If not, bug is in Python glue.
        def _kernel_py_emulation():
            H_, I_, BLK = 7168, 2048, 128
            rsf_ = float(routed_scaling_factor)
            offset_ = int(local_expert_offset)
            hs_ = tensors["hidden_states"]
            hss_ = tensors["hidden_states_scale"]
            w1_ = tensors["gemm1_weights"]
            w1s_ = tensors["gemm1_weights_scale"]
            w2_ = tensors["gemm2_weights"]
            w2s_ = tensors["gemm2_weights_scale"]
            rl_ = tensors["routing_logits"]
            rb_ = tensors["routing_bias"]
            T_ = hs_.shape[0]

            expert_ids_, weights_full_ = mod._route(rl_, rb_, T_, "cuda")
            sorted_tok_ids_, offsets_, sorted_rs_, N_, max_tok_ = mod._sort_tokens(
                expert_ids_, weights_full_, offset_, "cuda"
            )
            if N_ == 0:
                return torch.zeros(T_, H_, dtype=torch.bfloat16, device="cuda")

            # GEMM1 + SwiGLU
            gate_up = torch.zeros(N_, 2 * I_, dtype=torch.float32, device="cuda")
            for le in range(32):
                es = int(offsets_[le].item())
                ee = int(offsets_[le + 1].item())
                if es == ee:
                    continue
                tids = sorted_tok_ids_[es:ee].long()
                h_sel = hs_[tids].float()                                             # [Tk, H]
                hss_sel = hss_[:, tids].T                                             # [Tk, 56]
                h_dq = (h_sel.view(-1, H_ // BLK, BLK) * hss_sel.unsqueeze(-1)).view(-1, H_)
                w1e = w1_[le].float().view(2 * I_ // BLK, BLK, H_ // BLK, BLK)
                w1se = w1s_[le]
                w1_dq = (w1e * w1se[:, None, :, None]).view(2 * I_, H_)
                gate_up[es:ee] = h_dq @ w1_dq.T

            X1 = gate_up[:, :I_]                 # first_half = up
            X2 = gate_up[:, I_:]                 # second_half = gate
            inter = (X2 / (1.0 + torch.exp(-X2))) * X1

            # GEMM2 with scatter-add
            out_f32 = torch.zeros(T_, H_, dtype=torch.float32, device="cuda")
            for le in range(32):
                es = int(offsets_[le].item())
                ee = int(offsets_[le + 1].item())
                if es == ee:
                    continue
                tids = sorted_tok_ids_[es:ee].long()
                r_sc = sorted_rs_[es:ee]
                w2e = w2_[le].float().view(H_ // BLK, BLK, I_ // BLK, BLK)
                w2se = w2s_[le]
                w2_dq = (w2e * w2se[:, None, :, None]).view(H_, I_)
                out_tok = inter[es:ee] @ w2_dq.T
                weight = r_sc * rsf_
                out_f32.index_add_(0, tids, out_tok * weight.unsqueeze(-1))
            return out_f32.to(torch.bfloat16)

        output_pyem = _kernel_py_emulation()

        # Inspect routing + sort before running the kernel (same funcs the kernel uses)
        seq_len = tensors["hidden_states"].shape[0]
        try:
            expert_ids_dbg, weights_full_dbg = mod._route(
                tensors["routing_logits"], tensors["routing_bias"], seq_len, "cuda"
            )
            sorted_tok_ids_dbg, expert_offsets_dbg, sorted_rs_dbg, N_assigned_dbg, max_tok_dbg = mod._sort_tokens(
                expert_ids_dbg, weights_full_dbg, int(local_expert_offset), "cuda"
            )
            lines.append(f"\n=== Kernel internal state ===")
            lines.append(f"  N_assigned = {N_assigned_dbg}")
            lines.append(f"  max_tok    = {max_tok_dbg}")
            counts = (expert_offsets_dbg[1:].cpu() - expert_offsets_dbg[:-1].cpu()).tolist()
            lines.append(f"  per-local-expert token counts: {counts}")
            lines.append(f"  sorted_r_scores stats: min={float(sorted_rs_dbg.min()) if N_assigned_dbg>0 else 'NA'}, max={float(sorted_rs_dbg.max()) if N_assigned_dbg>0 else 'NA'}")
            # bucket calc (same as _bucket)
            def _bkt(n):
                for t in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]:
                    if n <= t: return t
                return 16384
            lines.append(f"  autotune bucket = {_bkt(N_assigned_dbg)}")
        except Exception as e:
            lines.append(f"  (internal inspect failed: {e})")

        # ---- Isolate GEMM1: call grouped_gemm1_kernel directly and compare to PyTorch ----
        try:
            import triton
            H_, I_, BLK = 7168, 2048, 128
            GATE_UP_DIM_ = I_ * 2
            if N_assigned_dbg > 0:
                # Kernel version: call grouped_gemm1_kernel into a fresh buffer
                gate_up_kern = torch.empty(N_assigned_dbg, GATE_UP_DIM_, dtype=torch.float32, device="cuda")
                hs_t = tensors["hidden_states"]
                hss_t = tensors["hidden_states_scale"]
                w1_t = tensors["gemm1_weights"]
                w1s_t = tensors["gemm1_weights_scale"]
                mod.grouped_gemm1_kernel[
                    lambda meta: (
                        32 * triton.cdiv(max_tok_dbg, meta["BLOCK_M"]),
                        triton.cdiv(GATE_UP_DIM_, meta["BLOCK_N"]),
                    )
                ](
                    sorted_tok_ids_dbg, expert_offsets_dbg, max_tok_dbg,
                    hs_t, hss_t,
                    w1_t, w1s_t,
                    gate_up_kern,
                    N_assigned_dbg,
                    _bkt(N_assigned_dbg),
                    K=H_,
                    N=GATE_UP_DIM_,
                    stride_h_seq      = hs_t.stride(0),
                    stride_w1_exp     = w1_t.stride(0),
                    stride_w1_n       = w1_t.stride(1),
                    stride_w1s_exp    = w1s_t.stride(0),
                    stride_w1s_nb     = w1s_t.stride(1),
                    stride_hscale_blk = hss_t.stride(0),
                    stride_gu_tok     = gate_up_kern.stride(0),
                )

                # Reference version: pytorch equivalent
                gate_up_ref = torch.zeros(N_assigned_dbg, GATE_UP_DIM_, dtype=torch.float32, device="cuda")
                for le in range(32):
                    es = int(expert_offsets_dbg[le].item())
                    ee = int(expert_offsets_dbg[le + 1].item())
                    if es == ee:
                        continue
                    tids = sorted_tok_ids_dbg[es:ee].long()
                    h_sel = hs_t[tids].float()
                    hss_sel = hss_t[:, tids].T
                    h_dq = (h_sel.view(-1, H_ // BLK, BLK) * hss_sel.unsqueeze(-1)).view(-1, H_)
                    w1e = w1_t[le].float().view(GATE_UP_DIM_ // BLK, BLK, H_ // BLK, BLK)
                    w1se = w1s_t[le]
                    w1_dq = (w1e * w1se[:, None, :, None]).view(GATE_UP_DIM_, H_)
                    gate_up_ref[es:ee] = h_dq @ w1_dq.T

                # ---- Isolate SwiGLU: run Triton swiglu_kernel vs pytorch ----
                inter_kern = torch.empty(N_assigned_dbg, I_, dtype=torch.float32, device="cuda")
                SWIGLU_BLOCK = 1024
                mod.swiglu_kernel[triton.cdiv(N_assigned_dbg * I_, SWIGLU_BLOCK),](
                    gate_up_ref, inter_kern,
                    N_assigned_dbg,
                    INTER_DIM=I_,
                    BLOCK_SIZE=SWIGLU_BLOCK,
                )
                # pytorch SwiGLU: first_half=up=gate_up[:, :I], second_half=gate=gate_up[:, I:]
                X1 = gate_up_ref[:, :I_]
                X2 = gate_up_ref[:, I_:]
                inter_py = (X2 / (1.0 + torch.exp(-X2))) * X1

                diff_sw = (inter_kern - inter_py).abs()
                lines.append(f"\n=== SwiGLU isolation: Triton vs PyTorch ===")
                lines.append(f"  inter_kern:  min={float(inter_kern.min()):.4e}, max={float(inter_kern.max()):.4e}, mean_abs={float(inter_kern.abs().mean()):.4e}")
                lines.append(f"  inter_py  :  min={float(inter_py.min()):.4e}, max={float(inter_py.max()):.4e}, mean_abs={float(inter_py.abs().mean()):.4e}")
                lines.append(f"  max_abs_err : {float(diff_sw.max()):.4e}")
                lines.append(f"  mean_abs_err: {float(diff_sw.mean()):.4e}")

                # ---- Isolate GEMM2: run Triton grouped_gemm2_kernel vs pytorch ----
                output_kern_gemm2 = torch.zeros(seq_len, H_, dtype=torch.float32, device="cuda")
                mod.grouped_gemm2_kernel[
                    lambda meta: (
                        32 * triton.cdiv(max_tok_dbg, meta["BLOCK_M"]),
                        triton.cdiv(H_, meta["BLOCK_N"]),
                    )
                ](
                    sorted_tok_ids_dbg, sorted_rs_dbg, expert_offsets_dbg, max_tok_dbg,
                    inter_py,  # use correct pytorch inter so we isolate GEMM2 alone
                    tensors["gemm2_weights"], tensors["gemm2_weights_scale"],
                    output_kern_gemm2,
                    float(routed_scaling_factor),
                    N_assigned_dbg,
                    _bkt(N_assigned_dbg),
                    K=I_,
                    N=H_,
                    stride_inter_tok  = inter_py.stride(0),
                    stride_w2_exp     = tensors["gemm2_weights"].stride(0),
                    stride_w2_n       = tensors["gemm2_weights"].stride(1),
                    stride_w2s_exp    = tensors["gemm2_weights_scale"].stride(0),
                    stride_w2s_nb     = tensors["gemm2_weights_scale"].stride(1),
                    stride_out_seq    = output_kern_gemm2.stride(0),
                )
                # pytorch GEMM2
                output_py_gemm2 = torch.zeros(seq_len, H_, dtype=torch.float32, device="cuda")
                for le in range(32):
                    es = int(expert_offsets_dbg[le].item())
                    ee = int(expert_offsets_dbg[le + 1].item())
                    if es == ee:
                        continue
                    tids = sorted_tok_ids_dbg[es:ee].long()
                    r_sc = sorted_rs_dbg[es:ee]
                    w2e = tensors["gemm2_weights"][le].float().view(H_ // BLK, BLK, I_ // BLK, BLK)
                    w2se = tensors["gemm2_weights_scale"][le]
                    w2_dq = (w2e * w2se[:, None, :, None]).view(H_, I_)
                    out_tok = inter_py[es:ee] @ w2_dq.T
                    weight = r_sc * float(routed_scaling_factor)
                    output_py_gemm2.index_add_(0, tids, out_tok * weight.unsqueeze(-1))

                diff_g2 = (output_kern_gemm2 - output_py_gemm2).abs()
                lines.append(f"\n=== GEMM2 isolation: Triton (with pytorch inter) vs PyTorch ===")
                lines.append(f"  out_kern:  min={float(output_kern_gemm2.min()):.4e}, max={float(output_kern_gemm2.max()):.4e}, mean_abs={float(output_kern_gemm2.abs().mean()):.4e}")
                lines.append(f"  out_py  :  min={float(output_py_gemm2.min()):.4e}, max={float(output_py_gemm2.max()):.4e}, mean_abs={float(output_py_gemm2.abs().mean()):.4e}")
                lines.append(f"  max_abs_err : {float(diff_g2.max()):.4e}")
                lines.append(f"  mean_abs_err: {float(diff_g2.mean()):.4e}")

                diff1 = (gate_up_kern - gate_up_ref).abs()
                lines.append(f"\n=== GEMM1 isolation: kernel vs PyTorch ===")
                lines.append(f"  gate_up_kern:  min={float(gate_up_kern.min()):.4e}, max={float(gate_up_kern.max()):.4e}, mean_abs={float(gate_up_kern.abs().mean()):.4e}")
                lines.append(f"  gate_up_ref :  min={float(gate_up_ref.min()):.4e}, max={float(gate_up_ref.max()):.4e}, mean_abs={float(gate_up_ref.abs().mean()):.4e}")
                lines.append(f"  max_abs_err : {float(diff1.max()):.4e}")
                lines.append(f"  mean_abs_err: {float(diff1.mean()):.4e}")
                # Sample values at (row=0, col=0..3)
                lines.append(f"  kern[0, :4]: {gate_up_kern[0, :4].tolist()}")
                lines.append(f"  ref [0, :4]: {gate_up_ref[0, :4].tolist()}")
                # Ratio
                nonzero = gate_up_ref.abs() > 1e-3
                if nonzero.any():
                    ratio = (gate_up_kern[nonzero] / gate_up_ref[nonzero]).abs()
                    lines.append(f"  |kern/ref| ratio: min={float(ratio.min()):.4e}, median={float(ratio.median()):.4e}, max={float(ratio.max()):.4e}")
        except Exception as e:
            import traceback
            lines.append(f"\n=== GEMM1 isolation failed: {e}\n{traceback.format_exc()}")

        # Run our kernel
        H = 7168
        output_ours = torch.zeros(seq_len, H, dtype=torch.bfloat16, device="cuda")
        mod.kernel(
            tensors["routing_logits"],
            tensors["routing_bias"],
            tensors["hidden_states"],
            tensors["hidden_states_scale"],
            tensors["gemm1_weights"],
            tensors["gemm1_weights_scale"],
            tensors["gemm2_weights"],
            tensors["gemm2_weights_scale"],
            local_expert_offset=local_expert_offset,
            routed_scaling_factor=routed_scaling_factor,
            output=output_ours,
        )
        lines.append(f"\n=== Our kernel output ===")
        lines.append(f"  shape={tuple(output_ours.shape)}, dtype={output_ours.dtype}")
        lines.append(f"  min={float(output_ours.float().min()):.4e}, max={float(output_ours.float().max()):.4e}")
        lines.append(f"  mean_abs={float(output_ours.float().abs().mean()):.4e}")
        lines.append(f"  nonzero={int((output_ours != 0).sum())}/{output_ours.numel()}")

        # ---- Reference: use the CONTEST'S official reference from the definition JSON ----
        ref_src = definition.reference
        ref_ns = {}
        exec(ref_src, ref_ns)
        output_ref_bf16 = ref_ns["run"](
            tensors["routing_logits"],
            tensors["routing_bias"],
            tensors["hidden_states"],
            tensors["hidden_states_scale"],
            tensors["gemm1_weights"],
            tensors["gemm1_weights_scale"],
            tensors["gemm2_weights"],
            tensors["gemm2_weights_scale"],
            int(local_expert_offset),
            float(routed_scaling_factor),
        )  # bf16, shape [seq, H]

        # Compare pure-PyTorch emulation to contest ref
        diff_pyem_vs_ref = (output_pyem.float() - output_ref_bf16.float()).abs()
        lines.append(f"\n=== Pure-PyTorch emulation of kernel algorithm ===")
        lines.append(f"  py_em : min={float(output_pyem.float().min()):.4e}, max={float(output_pyem.float().max()):.4e}, mean_abs={float(output_pyem.float().abs().mean()):.4e}")
        lines.append(f"  vs contest ref: max_abs_err = {float(diff_pyem_vs_ref.max()):.4e}, mean_abs_err = {float(diff_pyem_vs_ref.mean()):.4e}")
        lines.append(f"  If max_abs_err \u226a kernel's error  -> bug is in Triton (FP8 dequant / dot / strides).")
        lines.append(f"  If max_abs_err \u2248 kernel's error   -> bug is in the Python glue (routing/sort/swiglu).")

        # Also compute routing topk info (for display only)
        rl = tensors["routing_logits"].float()
        rb = tensors["routing_bias"].float()
        expert_s = torch.sigmoid(rl) + rb
        group_s = expert_s.view(seq_len, 8, 32)
        top2, _ = group_s.topk(2, dim=-1)
        group_scores = top2.sum(-1)
        _, top4g = group_scores.topk(4, dim=-1)
        candidate_mask = torch.zeros(seq_len, 256, device="cuda", dtype=torch.bool)
        for b in range(seq_len):
            for g in top4g[b].tolist():
                candidate_mask[b, g*32:(g+1)*32] = True
        masked_s = torch.where(candidate_mask, expert_s, torch.full_like(expert_s, -1e9))
        _, top8i = masked_s.topk(8, dim=-1)
        lines.append(f"\n=== Reference routing (first 3 tokens) ===")
        for t in range(min(3, seq_len)):
            lines.append(f"  token {t}: experts={top8i[t].tolist()}")
        diff = (output_ours.float() - output_ref_bf16.float()).abs()
        lines.append(f"\n=== Our kernel vs PyTorch reference ===")
        lines.append(f"  ref   : min={float(output_ref_bf16.float().min()):.4e}, max={float(output_ref_bf16.float().max()):.4e}, mean_abs={float(output_ref_bf16.float().abs().mean()):.4e}")
        lines.append(f"  ours  : min={float(output_ours.float().min()):.4e}, max={float(output_ours.float().max()):.4e}, mean_abs={float(output_ours.float().abs().mean()):.4e}")
        lines.append(f"  max_abs_err : {float(diff.max()):.4e}")
        lines.append(f"  mean_abs_err: {float(diff.mean()):.4e}")
        lines.append(f"  pct_within_1: {float((diff <= 1.0).float().mean()):.4f}")

        # Error localization
        row_max = diff.max(dim=1).values  # per-token
        col_max = diff.max(dim=0).values  # per-hidden-dim
        lines.append(f"\n=== Error localization ===")
        lines.append(f"  per-token (row) max_err: min={float(row_max.min()):.4e}, max={float(row_max.max()):.4e}, mean={float(row_max.mean()):.4e}")
        lines.append(f"  per-hidden-dim (col) max_err: min={float(col_max.min()):.4e}, max={float(col_max.max()):.4e}, mean={float(col_max.mean()):.4e}")
        # How concentrated is the error?
        row_bad = int((row_max > 1.0).sum())
        col_bad = int((col_max > 1.0).sum())
        lines.append(f"  tokens with max_err>1: {row_bad}/{seq_len}")
        lines.append(f"  hidden-dims with max_err>1: {col_bad}/{H}")
        # Identify a specific worst token for inspection
        worst_tok = int(row_max.argmax())
        worst_col = int(col_max.argmax())
        lines.append(f"  worst token idx: {worst_tok} (max_err={float(row_max[worst_tok]):.4e})")
        lines.append(f"    ours[{worst_tok}] sample: {output_ours[worst_tok, :8].float().tolist()}")
        lines.append(f"    ref [{worst_tok}] sample: {output_ref_bf16[worst_tok, :8].float().tolist()}")
        lines.append(f"  worst hidden-dim idx: {worst_col} (max_err={float(col_max[worst_col]):.4e})")

        # Check if output has NaN/Inf
        lines.append(f"\n=== NaN/Inf check ===")
        lines.append(f"  ours has NaN: {bool(torch.isnan(output_ours).any())}, Inf: {bool(torch.isinf(output_ours).any())}")
        lines.append(f"  ref  has NaN: {bool(torch.isnan(output_ref_bf16).any())}, Inf: {bool(torch.isinf(output_ref_bf16).any())}")

        # Expert load distribution (how many tokens route to each local expert)
        loc_off = int(local_expert_offset)
        local_mask = (top8i >= loc_off) & (top8i < loc_off + 32)
        lines.append(f"\n=== Expert load (local experts {loc_off}..{loc_off+31}) ===")
        lines.append(f"  total local routes: {int(local_mask.sum())}/{seq_len*8}")
        lines.append(f"  local_expert_offset={loc_off}, routed_scaling_factor={float(routed_scaling_factor)}")

    except Exception:
        lines.append("\n=== EXCEPTION ===")
        lines.append(traceback.format_exc())

    return "\n".join(lines)


@app.local_entrypoint()
def main(debug: bool = False, uuid: str = ""):
    """Pack solution and run benchmark on Modal. Use --debug to test kernel directly.

    --uuid <prefix>: when combined with --debug, selects a specific workload by UUID prefix.
    """
    from scripts.pack_solution import pack_solution

    print("Packing solution from source files...")
    solution_path = pack_solution()

    print("\nLoading solution...")
    solution = Solution.model_validate_json(solution_path.read_text())
    print(f"Loaded: {solution.name} ({solution.definition})")

    if debug:
        print(f"\nRunning debug kernel call on Modal B200 (uuid={uuid or '<first>'})...")
        result = debug_kernel.remote(solution, workload_uuid=(uuid or None))
        print(result)
        return

    print("\nRunning benchmark on Modal B200...")
    results = run_benchmark.remote(solution)

    if not results:
        print("No results returned!")
        return

    print_results(results)
