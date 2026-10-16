"""Argument parsing without torch imports, including isolated five-fold runs."""
import argparse
import datetime
import os
from pathlib import Path
import subprocess
import sys
from .common import ROOT, MODELS, read_json, write_json


def main(kind):
    parser = argparse.ArgumentParser(description=("Evaluate trained weights using the paper metrics" if kind == "evaluation"
                                                 else "Profile the paper inference path on one CUDA GPU"))
    parser.add_argument("--dataset", required=True, choices=("caihua", "udd"))
    parser.add_argument("--model", required=True, choices=MODELS)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", help="Trained weight path; use {fold} with --fold all")
    if kind == "profiling":
        group.add_argument("--paper-initialization", action="store_true",
                           help="UDD original profiler: pretrained backbone plus seed304 initialized heads")
        parser.add_argument("--cpu-threads", type=int, help="Override host threads; recorded as a protocol override")
    parser.add_argument("--fold", choices=("1","2","3","4","5","all"), help="Caihua only; defaults to 1")
    parser.add_argument("--data-root", type=Path, default=ROOT / "datasets", help="Root containing caihua/, udd/, sources/")
    parser.add_argument("--output", type=Path, required=True, help="New report directory; must not exist")
    parser.add_argument("--gpu", default="0", help="Single physical CUDA device index or UUID")
    if kind == "evaluation":
        parser.add_argument("--split", choices=("val", "test"), default="val")
        parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
        parser.add_argument("--batch-size", type=int, help="Defaults to the model's evaluation configuration")
        parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    if args.dataset == "udd" and args.fold is not None:
        parser.error("UDD has one fixed split; --fold applies only to Caihua")
    args.fold = (args.fold or "1") if args.dataset == "caihua" else None
    if kind == "profiling" and args.paper_initialization and args.dataset != "udd":
        parser.error("Caihua profiling requires trained checkpoints")
    for key in ("batch_size", "cpu_threads"):
        if getattr(args, key, None) is not None and getattr(args, key) < 1:
            parser.error("--" + key.replace("_", "-") + " must be positive")
    if getattr(args, "workers", 0) < 0 or "," in args.gpu:
        parser.error("Use nonnegative workers and a single GPU")
    args.output = args.output.expanduser().resolve()
    args.data_root = args.data_root.expanduser().resolve()
    for folder in ("code", "configs", "datasets", "pretrained", "environment", "inference_runtime"):
        if args.output.is_relative_to(ROOT / folder):
            parser.error("Output must be outside source/data directories")
    if args.output.exists():
        parser.error("Output directory already exists; select a new directory")
    folds = range(1,6) if args.fold == "all" else [args.fold]
    paths = []
    if args.checkpoint:
        if args.fold == "all" and "{fold}" not in args.checkpoint:
            parser.error("Five-fold evaluation needs a checkpoint pattern containing {fold}")
        for fold in folds:
            path = Path(args.checkpoint.replace("{fold}", str(fold))).expanduser().resolve()
            if not path.is_file():
                parser.error("Missing checkpoint: " + str(path))
            paths.append(path)
        if len(set(paths)) != len(paths):
            parser.error("Each fold must use a different checkpoint file")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    sys.dont_write_bytecode = True
    args.output.mkdir(parents=True)
    if args.fold == "all":
        records = []
        for fold, path in zip(range(1,6), paths):
            cmd = [sys.executable, str(ROOT / f"run_{kind}.py"), "--dataset", "caihua", "--model", args.model,
                   "--fold", str(fold), "--checkpoint", str(path), "--data-root", str(args.data_root),
                   "--gpu", args.gpu, "--output", str(args.output / f"rs{fold}")]
            if kind == "evaluation":
                cmd += ["--split", args.split, "--device", args.device, "--workers", str(args.workers)]
                if args.batch_size:
                    cmd += ["--batch-size", str(args.batch_size)]
            elif args.cpu_threads:
                cmd += ["--cpu-threads", str(args.cpu_threads)]
            subprocess.run(cmd, check=True, env=os.environ | {"PYTHONDONTWRITEBYTECODE": "1"})
            records.append(read_json(args.output / f"rs{fold}/report.json"))
        if kind == "evaluation":
            from .metrics import mean_metrics
            mean = mean_metrics([r["metrics"] for r in records])
        else:
            from .profiling import mean_profiles
            devices = {r["hardware_before"]["uuid"] for r in records}
            if len(devices) != 1:
                raise RuntimeError("Five-fold timing must use the same GPU")
            mean = mean_profiles(records)
        report = dict(dataset="caihua", model=args.model, folds=[1,2,3,4,5], mean=mean,
                      per_fold_reports=[f"rs{n}/report.json" for n in range(1,6)],
                      split=getattr(args, "split", None))
    else:
        args.fold = int(args.fold) if args.fold is not None else None
        if paths:
            args.checkpoint = paths[0]
        if kind == "evaluation":
            from .evaluation import evaluate
            report = evaluate(args)
        else:
            from .profiling import profile
            report = profile(args)
    report.update(kind=kind, utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    write_json(args.output / "report.json", report)
    value = report.get("mean", report.get("metrics"))
    if value is not None:
        import json
        print(json.dumps({k:v for k,v in value.items() if k in ("class_IoU", "mIoU", "OA", "FPS", "latency_mean_ms")}, indent=2))
    else:
        print(f"FPS {report['FPS']:.4f}, latency {report['latency_mean_ms']:.4f} ms")
    print("Report: " + str(args.output / "report.json"))
