"""Metrics from a whole-split confusion matrix, with explicit absent classes."""
import numpy as np

METRIC_POLICY = "miou_union_present_v1"


def confusion_matrix(target, prediction, classes, ignore=255):
    target = np.asarray(target, dtype=np.int64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.int64).reshape(-1)
    if target.shape != prediction.shape:
        raise ValueError("Prediction and target shapes differ")
    valid = target != ignore
    target, prediction = target[valid], prediction[valid]
    if np.any((target < 0) | (target >= classes)) or np.any((prediction < 0) | (prediction >= classes)):
        raise ValueError("Unregistered class ID in target or prediction")
    return np.bincount(target * classes + prediction,
                       minlength=classes * classes).reshape(classes, classes)


def metrics_from_confusion(matrix):
    matrix = np.asarray(matrix, dtype=np.int64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or np.any(matrix < 0):
        raise ValueError("Expected a square nonnegative confusion matrix")
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    correct = np.diag(matrix)
    union = support + predicted - correct
    present = union > 0
    iou = np.divide(correct, union, out=np.full(len(union), np.nan), where=present)
    accuracy = np.divide(correct, support, out=np.full(len(union), np.nan), where=support > 0)
    return {
        "metric_policy": METRIC_POLICY,
        "mIoU": float(iou[present].mean()) if present.any() else None,
        "OA": float(correct.sum() / matrix.sum()) if matrix.sum() else None,
        "mean_accuracy": float(accuracy[support > 0].mean()) if (support > 0).any() else None,
        "class_IoU": [float(v) if np.isfinite(v) else None for v in iou],
        "class_support": support.tolist(), "class_predictions": predicted.tolist(),
        "class_union": union.tolist(), "included_classes": np.flatnonzero(present).tolist(),
        "absent_from_truth": np.flatnonzero(support == 0).tolist(),
        "excluded_empty_classes": np.flatnonzero(~present).tolist(),
        "valid_pixels": int(matrix.sum()), "confusion_matrix": matrix.tolist(),
    }
