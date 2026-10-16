"""Paper metrics from the complete confusion matrix, without deleting classes."""
import statistics

CLASS_NAMES = {
    "caihua": ["Background", "Farmland", "Bare soil", "Road", "Woodland", "Building", "Water", "Other"],
    "udd": ["Building", "Vegetation", "Road", "Vehicle", "Background"],
}
DISPLAY_ORDER = {"caihua": [0, 1, 2, 3, 5, 4, 6], "udd": [4, 0, 2, 3, 1]}


def from_confusion(confusion, dataset):
    import numpy as np
    cm = np.asarray(confusion)
    n = len(CLASS_NAMES[dataset])
    if cm.shape != (n, n) or not np.issubdtype(cm.dtype, np.integer) or (cm < 0).any():
        raise ValueError("Expected a nonnegative integer confusion matrix")
    support, predicted, tp = cm.sum(1), cm.sum(0), cm.diagonal()
    union = support + predicted - tp
    if not cm.sum():
        raise ValueError("No valid labelled pixels")
    iou = [float(tp[i] / union[i] * 100) if union[i] else None for i in range(n)]
    ids = DISPLAY_ORDER[dataset]
    # Never silently change the paper's fixed class set to union-present classes.
    miou = statistics.mean(iou[i] for i in ids) if all(iou[i] is not None for i in ids) else None
    return dict(units="percent", mIoU=miou, OA=float(tp.sum() / cm.sum() * 100),
                class_IoU={CLASS_NAMES[dataset][i]: iou[i] for i in ids},
                class_IoU_model_order=iou, class_names_model_order=CLASS_NAMES[dataset],
                included_mIoU_class_ids=ids, confusion_matrix=cm.tolist(),
                ground_truth_pixels=support.tolist(), valid_pixels=int(cm.sum()),
                metric_policy="Fixed seven-class mIoU from full 8x8 matrix; OA includes all eight classes."
                if dataset == "caihua" else "Fixed five-class mIoU and all-valid-pixel OA.",
                undefined_policy="An absent class is null; a fixed-class mean containing null is null.")


def mean_metrics(records):
    def mean(values):
        return statistics.mean(values) if all(v is not None for v in values) else None
    return dict(units="percent", mIoU=mean([r["mIoU"] for r in records]),
                OA=mean([r["OA"] for r in records]),
                class_IoU={k: mean([r["class_IoU"][k] for r in records])
                           for k in records[0]["class_IoU"]},
                aggregation="Unweighted arithmetic mean of five fold metrics; confusion matrices are not pooled.")
