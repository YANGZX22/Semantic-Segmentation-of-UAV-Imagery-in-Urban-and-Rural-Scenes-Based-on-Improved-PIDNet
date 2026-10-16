"""Bind results to model identity, data, and comparison controls."""
import json
from pathlib import Path
from . import PROTOCOL


def plain(value):
    if hasattr(value, "items"):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(plain(value), indent=2, sort_keys=True,
                                     ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def make_run_record(cfg, spec, project, data_stamp, initialization, seed, steps, environment):
    training = plain(cfg.TRAIN)
    training.pop("RESUME", None)
    evaluation = plain(cfg.TEST)
    evaluation.pop("MODEL_FILE", None)
    inference = {"model": spec.record(), "num_classes": cfg.DATASET.NUM_CLASSES,
                 "ignore_label": cfg.TRAIN.IGNORE_LABEL, "align_corners": cfg.MODEL.ALIGN_CORNERS,
                 "metric_policy": "miou_union_present_v1", "evaluation": evaluation}
    controls = {
        "group": cfg.EXPERIMENT.GROUP, "seed": seed, "data": data_stamp,
        "training": training, "loss": plain(cfg.LOSS), "evaluation": evaluation,
        "align_corners": cfg.MODEL.ALIGN_CORNERS, "workers": cfg.WORKERS,
        "gpu_count": len(cfg.GPUS), "precision": "float32",
        "initialization": {key: initialization.get(key) for key in ("policy", "load_policy")},
        "steps_per_epoch": steps, "total_optimizer_steps": steps * cfg.TRAIN.END_EPOCH,
        "validation_frequency_epochs": 1, "selection_metric": "val.mIoU",
        "metric_policy": "miou_union_present_v1", "scheduler": "poly_power_0.9_before_step",
        "data_order": "independent_epoch_seed_v1", "determinism": cfg.EXPERIMENT.DETERMINISM,
        "cudnn": plain(cfg.CUDNN), "tf32": False,
    }
    contract = {"protocol": PROTOCOL, "model": spec.record(), "data": data_stamp,
                "inference": inference, "comparison_controls": controls}
    return {"experiment_protocol": contract, "comparison_controls": controls,
            "initialization": initialization,
            "resolved_config": plain(cfg), "environment": environment,
            "parameter_note": "Initialization coverage is reported, not assumed equal across changed architectures."}


def require_same_run(checkpoint, record):
    if checkpoint.get("experiment_protocol") != record["experiment_protocol"]:
        raise ValueError("Checkpoint belongs to a different model, seed, recipe, or split")
    if checkpoint.get("data_protocol") != record["experiment_protocol"]["data"]:
        raise ValueError("Checkpoint data provenance differs")


def validate_for_evaluation(checkpoint, cfg, spec, project, data_stamp):
    contract = checkpoint.get("experiment_protocol", {})
    if contract.get("protocol") != PROTOCOL or contract.get("model") != spec.record():
        raise ValueError("Checkpoint has a different or missing model-comparison identity")
    if contract.get("data") != data_stamp or checkpoint.get("data_protocol") != data_stamp:
        raise ValueError("Checkpoint was trained with another data protocol")
    inference = contract["inference"]
    evaluation = plain(cfg.TEST)
    evaluation.pop("MODEL_FILE", None)
    if (inference["num_classes"] != cfg.DATASET.NUM_CLASSES
            or inference["ignore_label"] != cfg.TRAIN.IGNORE_LABEL
            or inference["align_corners"] != cfg.MODEL.ALIGN_CORNERS
            or inference["evaluation"] != evaluation):
        raise ValueError("Evaluation settings differ from the frozen run contract")
