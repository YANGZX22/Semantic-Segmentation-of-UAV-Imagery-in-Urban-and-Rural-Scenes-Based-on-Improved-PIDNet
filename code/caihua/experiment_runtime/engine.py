"""Single-GPU comparison runner. Importing this module never starts a run."""
import argparse
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import platform
import random
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.nn import functional as F

from .registry import build_model
from .weights import initialize, read_checkpoint, load_trained_state
from .objectives import TrainingObjective, main_logits
from .metrics import confusion_matrix, metrics_from_confusion
from .provenance import make_run_record, write_json, require_same_run, validate_for_evaluation


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(_worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def configure_device(cfg, seed):
    if seed < 0 or seed >= 2 ** 32:
        raise ValueError("Seed must be an integer in [0, 2**32)")
    if len(cfg.GPUS) != 1:
        raise ValueError("comparison_v1 uses one GPU to fix effective batch size and BatchNorm behavior")
    gpu = cfg.GPUS[0]
    if not torch.cuda.is_available() or gpu < 0 or gpu >= torch.cuda.device_count():
        raise RuntimeError("The configured CUDA device is unavailable")
    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=cfg.EXPERIMENT.DETERMINISM == "warn_only")
    seed_all(seed)
    return torch.device("cuda", gpu)


def dataset_for(cfg, role):
    import datasets
    classes = {"caihua": "Caihua", "udd": "UDD"}
    name = cfg.DATASET.DATASET.lower()
    if name not in classes or not hasattr(datasets, classes[name]):
        raise ValueError("Dataset does not belong to this project: " + name)
    training = role == "train"
    size = cfg.TRAIN.IMAGE_SIZE if training else cfg.TEST.IMAGE_SIZE
    return getattr(datasets, classes[name])(
        root=cfg.DATASET.ROOT, list_path=role + ".lst", num_classes=cfg.DATASET.NUM_CLASSES,
        multi_scale=cfg.TRAIN.MULTI_SCALE if training else False,
        flip=cfg.TRAIN.FLIP if training else False, ignore_label=cfg.TRAIN.IGNORE_LABEL,
        base_size=cfg.TRAIN.BASE_SIZE if training else cfg.TEST.BASE_SIZE,
        crop_size=(size[1], size[0]), scale_factor=cfg.TRAIN.SCALE_FACTOR)


def make_loader(dataset, cfg, training, generator):
    return torch.utils.data.DataLoader(
        dataset, batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU if training else cfg.TEST.BATCH_SIZE_PER_GPU,
        shuffle=bool(training and cfg.TRAIN.SHUFFLE), num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY, drop_last=training, worker_init_fn=seed_worker,
        generator=generator, persistent_workers=False)


def optimizer_for(model, cfg):
    name = cfg.TRAIN.OPTIMIZER.lower()
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=cfg.TRAIN.LR, momentum=cfg.TRAIN.MOMENTUM,
                               weight_decay=cfg.TRAIN.WD, nesterov=cfg.TRAIN.NESTEROV)
    if name == "adamw":
        decay, no_decay = [], []
        for key, parameter in model.named_parameters():
            if parameter.requires_grad:
                (no_decay if parameter.ndim <= 1 or key.endswith(".bias") else decay).append(parameter)
        return torch.optim.AdamW([{"params": decay, "weight_decay": cfg.TRAIN.WD},
                                  {"params": no_decay, "weight_decay": 0.0}], lr=cfg.TRAIN.LR)
    raise ValueError("Unsupported optimizer: " + name)


def train_epoch(objective, loader, optimizer, cfg, device, epoch, total_steps):
    objective.train()
    sums = {"loss": 0.0, "semantic": 0.0, "boundary": 0.0, "boundary_guided": 0.0}
    count = 0
    for index, batch in enumerate(loader):
        images, labels, edges = batch[:3]
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, dtype=torch.long, non_blocking=True)
        edges = edges.to(device, dtype=torch.float32, non_blocking=True)
        step = epoch * len(loader) + index
        lr = cfg.TRAIN.LR * (1.0 - step / total_steps) ** 0.9
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        loss, parts = objective(images, labels, edges)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Non-finite loss at epoch %d, step %d" % (epoch + 1, index))
        loss.backward()
        optimizer.step()
        n = images.shape[0]
        count += n
        sums["loss"] += float(loss.detach()) * n
        for key, value in parts.items():
            sums[key] += float(value.detach()) * n
        if index % cfg.PRINT_FREQ == 0:
            logging.info("epoch=%d step=%d/%d lr=%.8f loss=%.6f", epoch + 1, index, len(loader), lr, loss.item())
    return {key: value / count for key, value in sums.items()}


@torch.no_grad()
def evaluate(model, spec, loader, cfg, device):
    model.eval()
    matrix = np.zeros((cfg.DATASET.NUM_CLASSES, cfg.DATASET.NUM_CLASSES), dtype=np.int64)
    ce_sum, pixels = 0.0, 0
    for batch in loader:
        images, labels = batch[:2]
        images, labels = images.to(device), labels.to(device, dtype=torch.long)
        logits = main_logits(model, images, labels, spec, cfg)
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("Non-finite evaluation logits")
        matrix += confusion_matrix(labels.cpu().numpy(), logits.argmax(1).cpu().numpy(),
                                   cfg.DATASET.NUM_CLASSES, cfg.TRAIN.IGNORE_LABEL)
        valid = int((labels != cfg.TRAIN.IGNORE_LABEL).sum())
        ce_sum += float(F.cross_entropy(logits, labels, ignore_index=cfg.TRAIN.IGNORE_LABEL, reduction="sum"))
        pixels += valid
    result = metrics_from_confusion(matrix)
    result["main_cross_entropy"] = ce_sum / pixels if pixels else None
    return result


def save_checkpoint(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def parse_config(project, evaluation=False):
    from configs import config, update_config
    parser = argparse.ArgumentParser(description="Versioned PIDNet model comparison")
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--seed", type=int, default=304)
    if evaluation:
        parser.add_argument("--weight", required=True)
        parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("opts", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()
    candidate = Path(args.cfg).expanduser()
    if not candidate.is_file():
        candidate = Path(project) / candidate
    args.cfg = str(candidate.resolve())
    update_config(config, args)
    return args, config


def train_main(project):
    from data_integrity_bridge import training_preflight, write_run_stamp
    args, cfg = parse_config(project)
    data_stamp = training_preflight(cfg)
    device = configure_device(cfg, args.seed)
    model, spec = build_model(cfg)
    initialization = initialize(model, spec, cfg)
    train_data, val_data = dataset_for(cfg, "train"), dataset_for(cfg, "val")
    train_generator, val_generator = torch.Generator(), torch.Generator()
    train_loader = make_loader(train_data, cfg, True, train_generator)
    val_loader = make_loader(val_data, cfg, False, val_generator)
    if len(train_loader) == 0:
        raise ValueError("Training set is smaller than one full batch")
    environment = {"python": platform.python_version(), "torch": str(torch.__version__),
                   "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(device),
                   "cudnn_version": torch.backends.cudnn.version(),
                   "numpy": str(np.__version__), "platform": platform.platform(),
                   "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                   "dependencies": {name: importlib.metadata.version(name) for name in ("yacs", "PyYAML", "Pillow")}}
    import cv2
    environment["opencv"] = str(cv2.__version__)
    record = make_run_record(cfg, spec, project, data_stamp, initialization, args.seed, len(train_loader), environment)
    record["parameters"] = {"total": sum(p.numel() for p in model.parameters()),
                            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    contract = record["experiment_protocol"]
    output = (Path(cfg.OUTPUT_DIR) / data_stamp["dataset"] / data_stamp["split_id"] / spec.name
              / ("seed_%d" % args.seed))
    if output.exists() and any(output.iterdir()) and not cfg.TRAIN.RESUME:
        raise FileExistsError("Run directory already exists; use explicit resume or a new experiment: " + str(output))
    if cfg.TRAIN.RESUME and not (output / "checkpoint.pth.tar").is_file():
        raise FileNotFoundError("Explicit resume requested but the exact run checkpoint is missing")
    output.mkdir(parents=True, exist_ok=True)
    write_run_stamp(output, data_stamp)
    if (output / "run.json").exists():
        old_record = json.loads((output / "run.json").read_text(encoding="utf-8"))
        if old_record["experiment_protocol"] != contract or old_record["environment"] != environment:
            raise ValueError("Resume requires the original experiment and software/hardware environment")
    else:
        write_json(output / "run.json", record)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(output / "train.log", encoding="utf-8"), logging.StreamHandler()])
    logging.captureWarnings(True)
    logging.info("Run directory: %s", output)
    logging.info("Model identity: %s; initialization coverage: %s", spec.record(), initialization.get("loaded_parameter_fraction"))
    model = model.to(device)
    objective = TrainingObjective(model, spec, cfg, train_data.class_weights).to(device)
    optimizer = optimizer_for(model, cfg)
    last_epoch, best_miou, best_epoch = 0, -1.0, None
    if cfg.TRAIN.RESUME:
        checkpoint = read_checkpoint(output / "checkpoint.pth.tar")
        require_same_run(checkpoint, record)
        load_trained_state(model, checkpoint)
        optimizer.load_state_dict(checkpoint["optimizer"])
        last_epoch, best_miou, best_epoch = checkpoint["epoch"], checkpoint["best_mIoU"], checkpoint["best_epoch"]
        best_record = json.loads((output / "best_validation.json").read_text(encoding="utf-8"))
        if (best_record["best_epoch"] != best_epoch or best_record["best_mIoU"] != best_miou
                or best_record["experiment_protocol"] != contract
                or not (output / "best.pt").is_file()):
            raise ValueError("Best checkpoint/result publication is incomplete or inconsistent; resume stopped")
    total_steps = len(train_loader) * cfg.TRAIN.END_EPOCH
    for epoch in range(last_epoch, cfg.TRAIN.END_EPOCH):
        # The data stream does not depend on how many RNG draws model construction used.
        epoch_seed = (args.seed + 1000003 * epoch) % (2 ** 32)
        seed_all(epoch_seed)
        train_generator.manual_seed(epoch_seed)
        train_metrics = train_epoch(objective, train_loader, optimizer, cfg, device, epoch, total_steps)
        val_generator.manual_seed((args.seed + 2147483647) % (2 ** 32))
        validation = evaluate(model, spec, val_loader, cfg, device)
        if validation["mIoU"] is None:
            raise ValueError("Validation split has no evaluable pixels")
        improved = validation["mIoU"] > best_miou
        if improved:
            best_miou, best_epoch = validation["mIoU"], epoch + 1
        result = {"epoch": epoch + 1, "role": "val", "train": train_metrics, "validation": validation,
                  "best_mIoU": best_miou, "best_epoch": best_epoch, "experiment_protocol": contract}
        write_json(output / "epochs" / ("%04d.json" % (epoch + 1)), result)
        checkpoint = {"epoch": epoch + 1, "best_mIoU": best_miou, "best_epoch": best_epoch,
                      "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "data_protocol": data_stamp, "experiment_protocol": contract,
                      "validation": validation, "run_record": record}
        if improved:
            save_checkpoint(output / "best.pt", checkpoint)
            write_json(output / "best_validation.json", result)
        # Save AFTER updating the best result; never reuse stale validation scores.
        save_checkpoint(output / "checkpoint.pth.tar", checkpoint)
        logging.info("epoch=%d val_mIoU=%.6f best_mIoU=%.6f best_epoch=%d", epoch + 1, validation["mIoU"], best_miou, best_epoch)
    if last_epoch < cfg.TRAIN.END_EPOCH:
        save_checkpoint(output / "final_state.pt", checkpoint)
    write_json(output / "completion.json", {"status": "complete", "epochs": cfg.TRAIN.END_EPOCH,
                                           "best_epoch": best_epoch, "best_mIoU": best_miou,
                                           "selection_role": "val", "test_evaluated": False,
                                           "experiment_protocol": contract})


def evaluate_main(project):
    from data_integrity_bridge import training_preflight, validate_evaluation
    args, cfg = parse_config(project, evaluation=True)
    data_stamp = training_preflight(cfg)
    device = configure_device(cfg, args.seed)
    model, spec = build_model(cfg)
    weight = Path(args.weight).expanduser().resolve()
    validate_evaluation(weight, cfg.DATASET.ROOT)
    checkpoint = read_checkpoint(weight)
    validate_for_evaluation(checkpoint, cfg, spec, project, data_stamp)
    load_trained_state(model, checkpoint)
    model.to(device)
    dataset = dataset_for(cfg, args.split)
    generator = torch.Generator().manual_seed(args.seed)
    loader = make_loader(dataset, cfg, False, generator)
    result = {"role": args.split, "epoch": checkpoint["epoch"], "weight": str(weight),
              "experiment_protocol": checkpoint["experiment_protocol"],
              "metrics": evaluate(model, spec, loader, cfg, device)}
    result_path = weight.parent / ("evaluation_%s_%s.json" % (args.split, weight.stem))
    write_json(result_path, result)
    print(json.dumps({"result": str(result_path), "role": args.split, "mIoU": result["metrics"]["mIoU"],
                      "OA": result["metrics"]["OA"]}, ensure_ascii=False))
