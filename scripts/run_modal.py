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
    .pip_install("flashinfer-bench", "torch", "triton", "numpy")
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
def debug_kernel(solution: Solution) -> str:
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

    # Load one real workload from the dataset
    trace_set = TraceSet.from_path(TRACE_SET_PATH)
    workloads = trace_set.workloads.get(solution.definition, [])
    if not workloads:
        return "No workloads found"
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

        # ---- Load REAL workload inputs ----
        wl_inputs = actual_wl.inputs  # dict of tensor name -> tensor
        lines.append(f"\n=== Real workload input keys: {list(wl_inputs.keys())}")

        # Load each tensor to CUDA
        tensors = {}
        for k, v in wl_inputs.items():
            t = v.to("cuda") if hasattr(v, "to") else v
            tensors[k] = t
            lines.append(f"  {k}: shape={tuple(t.shape)}, dtype={t.dtype}, "
                         f"min={float(t.float().min()):.4e}, max={float(t.float().max()):.4e}")

        # Get const axes
        local_expert_offset = actual_wl.axes.get("local_expert_offset", 192)
        routed_scaling_factor = actual_wl.axes.get("routed_scaling_factor", 2.5)
        lines.append(f"\n  local_expert_offset={local_expert_offset}, routed_scaling_factor={routed_scaling_factor}")

        # Run our kernel
        seq_len = tensors["hidden_states"].shape[0]
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

        # ---- Reference: pure PyTorch with block-scale dequant ----
        import torch.nn.functional as F
        rl = tensors["routing_logits"].float()
        rb = tensors["routing_bias"].float()
        hs = tensors["hidden_states"]
        hss = tensors["hidden_states_scale"]  # [56, seq_len]
        w1 = tensors["gemm1_weights"]
        w1s = tensors["gemm1_weights_scale"]  # [32, 32, 56]
        w2 = tensors["gemm2_weights"]
        w2s = tensors["gemm2_weights_scale"]  # [32, 56, 16]

        BLOCK = 128
        H, I = 7168, 2048

        # Dequantize hidden
        h_f32 = hs.float().view(seq_len, H // BLOCK, BLOCK)
        hss_T = hss.T.contiguous()  # [seq_len, 56]
        h_dq = (h_f32 * hss_T.unsqueeze(-1)).view(seq_len, H)

        # Routing: sigmoid + bias, group top-2 sum, top-4 groups, top-8 experts, normalize
        expert_s = torch.sigmoid(rl) + rb  # [seq, 256]
        group_s = expert_s.view(seq_len, 8, 32)
        top2, _ = group_s.topk(2, dim=-1)
        group_scores = top2.sum(-1)  # [seq, 8]
        _, top4g = group_scores.topk(4, dim=-1)  # [seq, 4]
        candidate_mask = torch.zeros(seq_len, 256, device="cuda", dtype=torch.bool)
        for b in range(seq_len):
            for g in top4g[b].tolist():
                candidate_mask[b, g*32:(g+1)*32] = True
        masked_s = torch.where(candidate_mask, expert_s, torch.full_like(expert_s, -1e9))
        top8s, top8i = masked_s.topk(8, dim=-1)  # [seq, 8]
        # Normalize
        norm_s = top8s / top8s.sum(-1, keepdim=True).clamp(min=1e-9)

        lines.append(f"\n=== Reference routing (first 3 tokens) ===")
        for t in range(min(3, seq_len)):
            lines.append(f"  token {t}: experts={top8i[t].tolist()}, scores={[f'{s:.3f}' for s in norm_s[t].tolist()]}")

        output_ref = torch.zeros(seq_len, H, dtype=torch.float32, device="cuda")
        local_offset = int(local_expert_offset)
        rsf = float(routed_scaling_factor)
        for local_idx in range(32):
            gid = local_offset + local_idx
            pairs = (top8i == gid).nonzero(as_tuple=False)
            if pairs.shape[0] == 0:
                continue
            tok_ids, slot_ids = pairs[:, 0], pairs[:, 1]
            r_sc = norm_s[tok_ids, slot_ids] * rsf
            # Dequant w1
            w1e = w1[local_idx].float().view(32, BLOCK, 56, BLOCK)
            w1s_e = w1s[local_idx]  # [32, 56]
            w1_dq = (w1e * w1s_e[:, None, :, None]).view(I*2, H).to(torch.bfloat16)
            h_tok = h_dq[tok_ids].to(torch.bfloat16)
            gu = torch.mm(h_tok, w1_dq.T).float()
            inter = F.silu(gu[:, :I]) * gu[:, I:]
            # Dequant w2
            w2e = w2[local_idx].float().view(56, BLOCK, 16, BLOCK)
            w2s_e = w2s[local_idx]  # [56, 16]
            w2_dq = (w2e * w2s_e[:, None, :, None]).view(H, I).to(torch.bfloat16)
            out_tok = torch.mm(inter.to(torch.bfloat16), w2_dq.T).float()
            output_ref.index_add_(0, tok_ids, out_tok * r_sc.unsqueeze(-1))

        output_ref_bf16 = output_ref.to(torch.bfloat16)
        diff = (output_ours.float() - output_ref_bf16.float()).abs()
        lines.append(f"\n=== Our kernel vs PyTorch reference ===")
        lines.append(f"  ref   : min={float(output_ref_bf16.float().min()):.4e}, max={float(output_ref_bf16.float().max()):.4e}, mean_abs={float(output_ref_bf16.float().abs().mean()):.4e}")
        lines.append(f"  ours  : min={float(output_ours.float().min()):.4e}, max={float(output_ours.float().max()):.4e}, mean_abs={float(output_ours.float().abs().mean()):.4e}")
        lines.append(f"  max_abs_err : {float(diff.max()):.4e}")
        lines.append(f"  mean_abs_err: {float(diff.mean()):.4e}")
        lines.append(f"  pct_within_1: {float((diff <= 1.0).float().mean()):.4f}")

    except Exception:
        lines.append("\n=== EXCEPTION ===")
        lines.append(traceback.format_exc())

    return "\n".join(lines)


@app.local_entrypoint()
def main(debug: bool = False):
    """Pack solution and run benchmark on Modal. Use --debug to test kernel directly."""
    from scripts.pack_solution import pack_solution

    print("Packing solution from source files...")
    solution_path = pack_solution()

    print("\nLoading solution...")
    solution = Solution.model_validate_json(solution_path.read_text())
    print(f"Loaded: {solution.name} ({solution.definition})")

    if debug:
        print("\nRunning debug kernel call on Modal B200...")
        result = debug_kernel.remote(solution)
        print(result)
        return

    print("\nRunning benchmark on Modal B200...")
    results = run_benchmark.remote(solution)

    if not results:
        print("No results returned!")
        return

    print_results(results)
