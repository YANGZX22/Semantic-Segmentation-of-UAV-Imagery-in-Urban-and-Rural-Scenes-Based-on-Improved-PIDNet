"""Relocatable paths and one explicit comparison protocol."""
from pathlib import Path
import math
from . import PROTOCOL
from .registry import validate_contract

CURRENT = Path(__file__).resolve().parent.parent
REPOSITORY = CURRENT.parents[1]


def add_defaults(cfg):
    from yacs.config import CfgNode as CN
    cfg.EXPERIMENT = CN()
    cfg.EXPERIMENT.PROTOCOL = PROTOCOL
    cfg.EXPERIMENT.GROUP = ""
    cfg.EXPERIMENT.DETERMINISM = "warn_only"
    cfg.MODEL.INITIALIZATION = "external"
    cfg.LOSS.BOUNDARY_WEIGHT = 20.0
    cfg.LOSS.BOUNDARY_THRESHOLD = 0.8
    cfg.LOSS.AUX_USE_OHEM = False


def resolve_path(value, project):
    if not value:
        return ""
    value = str(value).replace("${REPOSITORY}", str(REPOSITORY)).replace("${CURRENT}", str(CURRENT)).replace("${PROJECT}", str(project))
    if value.startswith(("http://", "https://")):
        raise ValueError("Use an explicit local pretrained file; runtime downloads are disabled")
    path = Path(value).expanduser()
    return str((path if path.is_absolute() else project / path).resolve())


def finalize_config(cfg, project):
    project = Path(project).resolve()
    for section, field in ((cfg.DATASET, "ROOT"), (cfg.MODEL, "PRETRAINED"),
                           (cfg.TEST, "MODEL_FILE"), (cfg, "OUTPUT_DIR"), (cfg, "LOG_DIR")):
        setattr(section, field, resolve_path(getattr(section, field), project))
    if cfg.EXPERIMENT.PROTOCOL != PROTOCOL or not cfg.EXPERIMENT.GROUP:
        raise ValueError("A named, versioned comparison group is required")
    if cfg.EXPERIMENT.DETERMINISM not in ("strict", "warn_only"):
        raise ValueError("DETERMINISM must be strict or warn_only")
    if cfg.MODEL.INITIALIZATION not in ("external", "random"):
        raise ValueError("MODEL.INITIALIZATION must be external or random")
    if (cfg.MODEL.INITIALIZATION == "external") != bool(cfg.MODEL.PRETRAINED):
        raise ValueError("Initialization policy and PRETRAINED path disagree")
    if cfg.TRAIN.BEGIN_EPOCH != 0 or cfg.TRAIN.EXTRA_EPOCH != 0:
        raise ValueError("Use a zero-based training budget; resume uses the saved epoch")
    if cfg.TRAIN.END_EPOCH < 1 or cfg.TRAIN.BATCH_SIZE_PER_GPU < 1:
        raise ValueError("Epoch and batch budgets must be positive")
    if cfg.TEST.BATCH_SIZE_PER_GPU < 1 or cfg.WORKERS < 0 or cfg.PRINT_FREQ < 1:
        raise ValueError("Invalid evaluation batch size, worker count, or logging interval")
    if not math.isfinite(cfg.TRAIN.LR) or cfg.TRAIN.LR <= 0 or not math.isfinite(cfg.TRAIN.WD) or cfg.TRAIN.WD < 0:
        raise ValueError("Invalid learning rate or weight decay")
    if cfg.LOSS.OHEMKEEP < 1 or not 0 < cfg.LOSS.OHEMTHRES <= 1:
        raise ValueError("Invalid OHEM settings")
    if cfg.TEST.FLIP_TEST or cfg.TEST.MULTI_SCALE:
        raise ValueError("comparison_v1 fixes single-scale, unflipped evaluation")
    if cfg.CUDNN.BENCHMARK or not cfg.CUDNN.DETERMINISTIC:
        raise ValueError("comparison_v1 requires deterministic settings")
    if cfg.TRAIN.OPTIMIZER.lower() not in ("sgd", "adamw"):
        raise ValueError("Supported explicit optimizers: sgd, adamw")
    if cfg.LOSS.SB_WEIGHTS < 0 or cfg.LOSS.BOUNDARY_WEIGHT < 0:
        raise ValueError("Loss weights must be nonnegative")
    if not 0 < cfg.LOSS.BOUNDARY_THRESHOLD < 1:
        raise ValueError("Boundary threshold must be between zero and one")
    validate_contract(cfg)
