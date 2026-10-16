"""Explicit semantic/auxiliary/boundary supervision shared by both datasets."""
import torch
from torch import nn
from torch.nn import functional as F
from .registry import semantic_outputs


def resize_logits(value, size, align_corners):
    if value.shape[-2:] != tuple(size):
        value = F.interpolate(value, size=size, mode="bilinear", align_corners=align_corners)
    return value


def semantic_loss(logits, labels, ignore, weights=None, ohem=False, threshold=0.9, min_kept=65536):
    valid = labels != ignore
    if not bool(valid.any()):
        return logits.sum() * 0.0
    if not ohem:
        return F.cross_entropy(logits, labels, weight=weights, ignore_index=ignore)
    losses = F.cross_entropy(logits, labels, weight=weights, ignore_index=ignore, reduction="none")[valid]
    with torch.no_grad():
        safe_labels = labels.masked_fill(~valid, 0)
        confidence = logits.softmax(dim=1).gather(1, safe_labels.unsqueeze(1)).squeeze(1)[valid]
        sorted_confidence = confidence.sort().values
        keep = min(max(int(min_kept), 1), confidence.numel())
        cutoff = max(float(threshold), float(sorted_confidence[keep - 1]))
        selected = confidence <= cutoff
    return losses[selected].mean()


def boundary_loss(logits, targets, valid, coefficient):
    values = logits[:, 0][valid]
    targets = targets.to(values.dtype)[valid]
    if values.numel() == 0:
        return logits.sum() * 0.0
    positive = targets > 0.5
    npos, nneg = positive.sum(), (~positive).sum()
    if npos == 0 or nneg == 0:
        weights = torch.ones_like(values)
    else:
        weights = torch.where(positive, nneg / values.numel(), npos / values.numel())
    return coefficient * F.binary_cross_entropy_with_logits(values, targets, weight=weights)


class TrainingObjective(nn.Module):
    def __init__(self, model, spec, cfg, class_weights):
        super().__init__()
        self.model, self.spec, self.cfg = model, spec, cfg
        self.register_buffer("class_weights", class_weights if cfg.LOSS.CLASS_BALANCE else None)

    def forward(self, images, labels, edges):
        cfg, spec = self.cfg, self.spec
        semantic, main, boundary = semantic_outputs(self.model(images), spec, training=True)
        if any(value.shape[1] != cfg.DATASET.NUM_CLASSES for value in semantic):
            raise ValueError("Semantic head class count differs from the dataset")
        semantic = [resize_logits(value, labels.shape[-2:], cfg.MODEL.ALIGN_CORNERS) for value in semantic]
        pieces = [semantic_loss(value, labels, cfg.TRAIN.IGNORE_LABEL, self.class_weights,
                                ohem=bool(cfg.LOSS.USE_OHEM and (index == main or cfg.LOSS.AUX_USE_OHEM)),
                                threshold=cfg.LOSS.OHEMTHRES, min_kept=cfg.LOSS.OHEMKEEP)
                  for index, value in enumerate(semantic)]
        semantic_total = sum(weight * loss for weight, loss in zip(cfg.LOSS.BALANCE_WEIGHTS, pieces))
        boundary_total = semantic_total.new_zeros(())
        guided_total = semantic_total.new_zeros(())
        if boundary is not None:
            boundary = resize_logits(boundary, labels.shape[-2:], cfg.MODEL.ALIGN_CORNERS)
            valid = labels != cfg.TRAIN.IGNORE_LABEL
            boundary_total = boundary_loss(boundary, edges, valid, cfg.LOSS.BOUNDARY_WEIGHT)
            if cfg.LOSS.SB_WEIGHTS:
                selected = (boundary[:, 0].detach().sigmoid() > cfg.LOSS.BOUNDARY_THRESHOLD) & valid
                boundary_labels = labels.masked_fill(~selected, cfg.TRAIN.IGNORE_LABEL)
                guided_total = cfg.LOSS.SB_WEIGHTS * semantic_loss(
                    semantic[main], boundary_labels, cfg.TRAIN.IGNORE_LABEL, self.class_weights,
                    ohem=cfg.LOSS.USE_OHEM, threshold=cfg.LOSS.OHEMTHRES, min_kept=cfg.LOSS.OHEMKEEP)
        total = semantic_total + boundary_total + guided_total
        return total, {"semantic": semantic_total, "boundary": boundary_total, "boundary_guided": guided_total}


def main_logits(model, images, labels, spec, cfg):
    semantic, main, _ = semantic_outputs(model(images), spec, training=False)
    logits = semantic[main]
    if logits.shape[1] != cfg.DATASET.NUM_CLASSES:
        raise ValueError("Evaluation output has the wrong number of classes")
    return resize_logits(logits, labels.shape[-2:], cfg.MODEL.ALIGN_CORNERS)
