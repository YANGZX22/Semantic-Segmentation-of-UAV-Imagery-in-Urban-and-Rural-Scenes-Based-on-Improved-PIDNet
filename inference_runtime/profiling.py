"""Port of the paper's CUDA-event measurement, with explicit cost scopes."""
import collections
import gc
import os
from pathlib import Path
import statistics
import subprocess
import time
import torch
from torch.nn import functional as F
from .common import (build_model, configuration, environment, load_weights,
                     strip_training_heads)
from .data import split_root


def head_cost(model):
    heads = {}
    for name in ("final_layer", "seghead_p", "seghead_d"):
        head, macs, modules = getattr(model, name), 0, []
        for key, module in head.named_modules():
            if isinstance(module, torch.nn.ConvTranspose2d):
                if module.stride != (8,8) or module.kernel_size != (16,16):
                    raise ValueError("Unexpected DCU structure")
                count = module.weight.numel() * 64 * 64
            elif isinstance(module, torch.nn.Conv2d):
                size = 512 if key.startswith("upconv.") else 64
                count = module.weight.numel() * size * size
            else:
                continue
            macs += count
            modules.append(dict(module=key, type=type(module).__name__, MACs=count))
        heads[name] = dict(parameters=sum(p.numel() for p in head.parameters()),
                           conv_MACs=macs, modules=modules)
    total = sum(h["conv_MACs"] for h in heads.values())
    return dict(parameters=sum(h["parameters"] for h in heads.values()), conv_MACs=total,
                paper_GFLOPs=total/1e9, heads=heads,
                scope="All three prediction heads, including auxiliary and boundary heads; CA is outside these heads.",
                convention="Paper head-cost table: convolution only, one multiply-add = one operation. BN/bias included in parameters.")


class Predict(torch.nn.Module):
    def __init__(self, model, name):
        super().__init__()
        self.model, self.name = model, name

    def forward(self, x):
        # Keep the historical timed path free of diagnostics and CPU transfers.
        y = self.model(x)
        if isinstance(y, (list, tuple)):
            y = y[0]
        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=True)
        return y


def complexity(predict, x):
    from torch.utils.flop_counter import FlopCounterMode, flop_registry
    from torch.utils._python_dispatch import TorchDispatchMode
    class OpAudit(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.counts = collections.Counter()

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            self.counts[str(func.overloadpacket)] += 1
            return func(*args, **(kwargs or {}))
    op = OpAudit()
    with torch.inference_mode(), FlopCounterMode(display=False) as counter, op:
        y = predict(x)
    if y.shape[-2:] != (512,512) or not torch.isfinite(y).all():
        raise ValueError("Invalid profiled prediction")
    count = counter.get_total_flops()
    if count <= 0:
        raise ValueError("No FLOPs were counted")
    return dict(counted_FLOPs=count, GFLOPs=count/1e9,
                FLOPs_by_operator={str(k): int(v) for k,v in counter.flop_counts["Global"].items()},
                uncounted_operators={k: v for k,v in op.counts.items() if k not in {str(o) for o in flop_registry}},
                output_shape=list(y.shape),
                convention="Whole inference network: PyTorch FlopCounterMode, multiply-add=2; supported conv/matmul operators. Other executed operators listed separately.")


def gpu_snapshot():
    fields = "uuid,name,driver_version,memory.total,memory.used,utilization.gpu,pstate,clocks.current.sm,clocks.current.memory,temperature.gpu,power.draw"
    # CUDA_VISIBLE_DEVICES selects one physical GPU. NVML accepts its UUID.
    props = torch.cuda.get_device_properties(0)
    uuid = str(props.uuid)
    rows = subprocess.check_output(["nvidia-smi", "--id=" + uuid, "--query-gpu=" + fields,
                                    "--format=csv,noheader,nounits"], text=True).strip()
    return dict(zip(fields.split(","), [s.strip() for s in rows.split(",")]))


def check_exclusive(uuid):
    output = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                                      "--format=csv,noheader,nounits"], text=True)
    others = []
    for line in output.splitlines():
        fields = [v.strip() for v in line.split(",")]
        if len(fields) == 2 and fields[0] == uuid and int(fields[1]) != os.getpid():
            others.append(int(fields[1]))
    if others:
        raise RuntimeError(f"GPU has other compute processes {others}; no process was stopped")


def profile(args):
    if not torch.cuda.is_available():
        raise RuntimeError("Paper FPS/latency measurement requires an NVIDIA CUDA GPU")
    cfg = configuration(args.dataset, args.model, args.fold)
    torch.manual_seed(304)
    if args.dataset == "caihua":
        torch.set_num_threads(2)
    # UDD retains the original script's host thread default unless explicitly
    # overridden by the caller; this effective value is saved in every report.
    if args.cpu_threads:
        torch.set_num_threads(args.cpu_threads)
    before = gpu_snapshot()
    check_exclusive(before["uuid"])
    if args.paper_initialization:
        from experiment_runtime.support import build, seed
        seed(304)
        model, initial = build(cfg)
        weight = dict(mode="paper_initialization", seed=304, pretrained=initial["pretrained"])
    else:
        model = build_model(args.dataset, args.model, cfg)
        weight = load_weights(model, args.checkpoint, args.dataset, args.model, cfg,
                              split_root(args.data_root, args.dataset, args.fold) / "manifest.json")
        weight["mode"] = "trained_checkpoint"
    total = sum(p.numel() for p in model.parameters())
    heads = head_cost(model) if args.model.startswith("pidnet_") else None
    removed = strip_training_heads(model, args.model)
    resident = sum(p.numel() for p in model.parameters())
    predict = Predict(model, args.model).eval().cuda()
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    x = torch.randn(1, 3, 512, 512, device="cuda", dtype=torch.float32)
    shapes, hooks = {}, []
    def tap(name):
        return lambda module, inputs, output: shapes.update({name: list(output.shape)})
    hooks.append(model.register_forward_hook(tap("native_output")))
    classifier = model.final_layer.conv2 if args.model.startswith("pidnet_") else (
        model.last_layer if args.model == "hrnet" else None)
    if classifier is not None:
        hooks.append(classifier.register_forward_hook(tap("classifier")))
    whole = complexity(predict, x)
    for hook in hooks:
        hook.remove()
    if classifier is not None and shapes["classifier"] != [1,cfg.DATASET.NUM_CLASSES,64,64]:
        raise ValueError("The formal classifier must operate on the 1/8 grid")
    if whole["output_shape"] != [1,cfg.DATASET.NUM_CLASSES,512,512]:
        raise ValueError("Wrong output channel count")
    warmup, rounds_count, iterations = (100,5,200) if args.dataset == "caihua" else (100,3,100)
    with torch.inference_mode():
        for _ in range(warmup):
            y = predict(x)
        torch.cuda.synchronize()
        del y
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()
        rounds = []
        for index in range(rounds_count):
            check_exclusive(before["uuid"])
            hw_before = gpu_snapshot()
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
            ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
            torch.cuda.synchronize()
            start_wall = time.perf_counter()
            for a, b in zip(starts, ends):
                a.record()
                y = predict(x)
                b.record()
                del y
            torch.cuda.synchronize()
            wall = time.perf_counter() - start_wall
            samples = [float(a.elapsed_time(b)) for a,b in zip(starts,ends)]
            if not all(v > 0 for v in samples):
                raise ValueError("Invalid CUDA timing")
            rounds.append(dict(round=index+1, latencies_ms=samples,
                               mean_ms=statistics.mean(samples), wall_seconds=wall,
                               hardware_before=hw_before, hardware_after=gpu_snapshot()))
            print(f"{args.model}: timing round {index+1}/{rounds_count}", flush=True)
        peak, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
    samples = [v for r in rounds for v in r["latencies_ms"]]
    after = gpu_snapshot()
    check_exclusive(before["uuid"])
    if after["uuid"] != before["uuid"]:
        raise RuntimeError("GPU changed during profiling")
    return dict(dataset=args.dataset, model=args.model, fold=args.fold, weight=weight,
                environment=environment(), hardware_before=before, hardware_after=after,
                thread_environment={k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
                cpu=next((s.split(":",1)[1].strip() for s in Path("/proc/cpuinfo").read_text().splitlines() if s.startswith("model name")), ""),
                cpu_affinity=sorted(os.sched_getaffinity(0)),
                total_training_parameters=total, inference_resident_parameters=resident,
                removed_training_only_modules=removed, paper_head_cost=heads,
                whole_network_cost=whole, observed_shapes=shapes,
                FPS=1000/statistics.mean(samples), latency_mean_ms=statistics.mean(samples),
                latency_sample_sd_ms=statistics.stdev(samples), rounds=rounds,
                baseline_allocated_bytes=baseline, peak_allocated_bytes=peak, peak_reserved_bytes=reserved,
                peak_increment_above_model_input_bytes=peak-baseline,
                protocol=dict(input=[1,3,512,512], dtype="FP32", TF32=False, AMP=False, compile=False,
                              eval=True, inference_mode=True, cudnn_benchmark=True, cudnn_deterministic=False,
                              warmup=warmup, timed_rounds=rounds_count, iterations_per_round=iterations,
                              timing="CUDA events, main logits restored to 512x512; no IO, preprocessing, CPU transfer, or argmax",
                              memory="PyTorch CUDA allocator, model+input+forward; excludes driver context",
                              seed=304, input_content="synthetic normal random tensor", data_samples_used=0,
                              paper_weight_state="Caihua: trained seed304 best checkpoints, five folds. UDD: seed304 pretrained backbone plus initialized heads.",
                              custom_cpu_threads=args.cpu_threads))


def mean_profiles(records):
    keys = ("FPS", "latency_mean_ms", "total_training_parameters", "inference_resident_parameters",
            "peak_allocated_bytes", "peak_reserved_bytes")
    out = {k: statistics.mean(r[k] for r in records) for k in keys}
    out["whole_network_GFLOPs"] = statistics.mean(r["whole_network_cost"]["GFLOPs"] for r in records)
    if all(r["paper_head_cost"] is not None for r in records):
        out["paper_head_GFLOPs"] = statistics.mean(r["paper_head_cost"]["paper_GFLOPs"] for r in records)
        out["paper_head_parameters"] = statistics.mean(r["paper_head_cost"]["parameters"] for r in records)
    out["aggregation"] = "Arithmetic means over five separately measured folds; mean FPS is not the reciprocal of mean latency."
    return out
