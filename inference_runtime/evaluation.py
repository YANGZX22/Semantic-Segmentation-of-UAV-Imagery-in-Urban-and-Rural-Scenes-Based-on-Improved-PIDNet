"""Single-scale evaluation of an explicitly selected trained checkpoint."""
import numpy as np
import torch
from .common import build_model, configuration, environment, load_weights, logits
from .data import FrozenEvaluationSet
from .metrics import from_confusion


def evaluate(args):
    cfg = configuration(args.dataset, args.model, args.fold)
    dataset = FrozenEvaluationSet(args.data_root, args.dataset, args.fold, args.split, args.model)
    torch.manual_seed(304)
    model = build_model(args.dataset, args.model, cfg)
    weight = load_weights(model, args.checkpoint, args.dataset, args.model, cfg, dataset.manifest_path)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu for CPU evaluation")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    model.eval().to(device)
    batch = args.batch_size or cfg.TEST.BATCH_SIZE_PER_GPU
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=False,
                                        num_workers=args.workers, pin_memory=device.type == "cuda")
    n = cfg.DATASET.NUM_CLASSES
    cm = np.zeros((n, n), dtype=np.int64)
    with torch.inference_mode():
        for index, (x, truth) in enumerate(loader):
            y = logits(model, x.to(device), args.model, cfg.MODEL.ALIGN_CORNERS)
            if y.shape[1] != n or not torch.isfinite(y).all():
                raise ValueError("Invalid semantic logits")
            pred = y.argmax(1).cpu().numpy()
            truth = truth.numpy().astype(np.int64)
            mask = truth != 255
            cm += np.bincount((truth[mask] * n + pred[mask]).ravel(), minlength=n*n).reshape(n, n)
            if index == 0 or (index + 1) % 25 == 0 or index + 1 == len(loader):
                print(f"{args.dataset} {args.model} {args.split}: {min((index+1)*batch,len(dataset))}/{len(dataset)}", flush=True)
    support = dataset.manifest["class_support"][args.split]
    if isinstance(support, dict):
        support = support["class_pixel_counts"]
    if cm.sum(1).tolist() != support:
        raise ValueError("Evaluated pixel support differs from the frozen partition")
    return dict(dataset=args.dataset, model=args.model, fold=args.fold, split=args.split,
                checkpoint=weight, data=dataset.record, metrics=from_confusion(cm, args.dataset),
                environment=environment(), protocol=dict(input=[3,512,512], batch_size=batch,
                precision="FP32", TF32=False, flip=False, multiscale=False,
                main_head_index=1 if args.model.startswith("pidnet_") else 0,
                output_interpolation="bilinear", align_corners=bool(cfg.MODEL.ALIGN_CORNERS),
                selection="Evaluate the supplied checkpoint; no best-epoch reselection under the seven-class metric.",
                paper_partition="val; --split test reports the held-out partition separately"))
