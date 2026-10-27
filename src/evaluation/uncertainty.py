"""Detection evaluation and bootstrap uncertainty helpers.

The matching procedure mirrors Ultralytics' greedy per-image matching at ten
IoU thresholds (0.50:0.05:0.95). Predictions must be supplied in confidence
order; callers can use :func:`build_image_row` which sorts them. Bootstrap
resampling is done over images (or source clusters) and reuses the cached
per-image true-positive rows, so no model inference is repeated for each
resample.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)


def _as_boxes(values) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return np.zeros((0, 4), dtype=np.float64)
    return array.reshape(-1, 4)


def box_iou_xyxy(boxes_a, boxes_b) -> np.ndarray:
    """Pairwise IoU for two arrays of xyxy boxes."""
    a = _as_boxes(boxes_a)
    b = _as_boxes(boxes_b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    left = np.maximum(a[:, None, 0], b[None, :, 0])
    top = np.maximum(a[:, None, 1], b[None, :, 1])
    right = np.minimum(a[:, None, 2], b[None, :, 2])
    bottom = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(right - left, 0, None) * np.clip(bottom - top, 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def match_predictions_sorted(
    true_classes,
    true_boxes,
    pred_classes,
    pred_boxes,
    thresholds: np.ndarray = IOU_THRESHOLDS,
) -> np.ndarray:
    """Return a (N, T) boolean TP matrix for confidence-sorted predictions.

    This is the same greedy per-threshold matching used by Ultralytics: for each
    detection in descending confidence order, select the best unclaimed target
    of the same class at each IoU threshold independently.
    """
    true_classes = np.asarray(true_classes, dtype=np.int64).reshape(-1)
    pred_classes = np.asarray(pred_classes, dtype=np.int64).reshape(-1)
    true_boxes = _as_boxes(true_boxes)
    pred_boxes = _as_boxes(pred_boxes)
    n_true = len(true_classes)
    n_pred = len(pred_classes)
    tp = np.zeros((n_pred, len(thresholds)), dtype=bool)
    if n_true == 0 or n_pred == 0:
        return tp

    iou = box_iou_xyxy(pred_boxes, true_boxes)
    iou = iou * (true_classes[None, :] == pred_classes[:, None])

    matched = np.zeros((n_true, len(thresholds)), dtype=bool)
    candidate_rows = np.flatnonzero((iou >= thresholds.min()).any(axis=1))
    for row in candidate_rows:
        available = np.where(matched, 0.0, iou[row, :, None])
        best_target = available.argmax(axis=0)
        best_iou = available[best_target, np.arange(len(thresholds))]
        tp[row] = best_iou >= thresholds
        matched[best_target, np.arange(len(thresholds))] |= tp[row]
    return tp


def build_image_row(
    true_classes,
    true_boxes,
    pred_classes,
    pred_confidences,
    pred_boxes,
) -> dict:
    """Build one cached evaluation row with predictions sorted by confidence."""
    true_classes = np.asarray(true_classes, dtype=np.int64).reshape(-1)
    pred_classes = np.asarray(pred_classes, dtype=np.int64).reshape(-1)
    pred_confidences = np.asarray(pred_confidences, dtype=np.float64).reshape(-1)
    true_boxes = _as_boxes(true_boxes)
    pred_boxes = _as_boxes(pred_boxes)
    if len(pred_confidences) != len(pred_classes) or len(pred_classes) != len(pred_boxes):
        raise ValueError("Prediction arrays have inconsistent lengths")
    if len(true_classes) != len(true_boxes):
        raise ValueError("Ground-truth arrays have inconsistent lengths")

    order = np.argsort(-pred_confidences, kind="stable")
    pred_confidences = pred_confidences[order]
    pred_classes = pred_classes[order]
    pred_boxes = pred_boxes[order]
    tp = match_predictions_sorted(true_classes, true_boxes, pred_classes, pred_boxes)
    return {
        "tp": tp,
        "conf": pred_confidences,
        "pred_cls": pred_classes,
        "target_cls": true_classes,
    }


def metrics_from_rows(rows: Sequence[dict], indices: Iterable[int], names=None) -> dict:
    """Compute AP metrics for the selected image rows.

    Returns overall precision/recall at the max-F1 operating point, mAP@50,
    mAP@50-95, and per-class results for classes that have ground truth.
    """
    from ultralytics.utils.metrics import ap_per_class

    selected = [rows[int(index)] for index in indices]
    if not selected:
        raise ValueError("No image rows selected")
    tp = np.concatenate([row["tp"] for row in selected], axis=0)
    conf = np.concatenate([row["conf"] for row in selected], axis=0)
    pred_cls = np.concatenate([row["pred_cls"] for row in selected], axis=0)
    target_cls = np.concatenate([row["target_cls"] for row in selected], axis=0)

    if len(target_cls) == 0:
        raise ValueError("Selected rows contain no ground-truth instances")

    if len(conf) == 0:
        unique_classes = np.unique(target_cls).astype(int)
        ap = np.zeros((len(unique_classes), len(IOU_THRESHOLDS)), dtype=np.float64)
        precision = np.zeros(len(unique_classes), dtype=np.float64)
        recall = np.zeros(len(unique_classes), dtype=np.float64)
    else:
        _, _, precision, recall, _, ap, unique_classes, *_ = ap_per_class(
            tp,
            conf,
            pred_cls,
            target_cls,
            plot=False,
        )

    overall = {
        "precision": float(np.mean(precision)) if len(precision) else 0.0,
        "recall": float(np.mean(recall)) if len(recall) else 0.0,
        "mAP50": float(np.mean(ap[:, 0])) if len(ap) else 0.0,
        "mAP50_95": float(np.mean(ap)) if len(ap) else 0.0,
    }

    counts = np.bincount(target_cls.astype(int), minlength=(len(names) if names else 0))
    per_class = {}
    for row_index, class_id in enumerate(unique_classes):
        class_id = int(class_id)
        label = names[class_id] if names is not None and class_id < len(names) else str(class_id)
        per_class[label] = {
            "class_id": class_id,
            "instances": int(counts[class_id]) if class_id < len(counts) else None,
            "precision": float(precision[row_index]),
            "recall": float(recall[row_index]),
            "ap50": float(ap[row_index, 0]),
            "ap50_95": float(np.mean(ap[row_index])),
        }
    return {"overall": overall, "per_class": per_class}


def image_bootstrap_indices(n_images: int, resamples: int, seed: int) -> np.ndarray:
    """Return an (resamples, n_images) array of image indices."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_images, size=(resamples, n_images), dtype=np.int64)


def cluster_bootstrap_indices(
    clusters: Sequence[Sequence[int]],
    resamples: int,
    seed: int,
) -> list[np.ndarray]:
    """Cluster bootstrap: resample clusters with replacement, keep all images."""
    if not clusters:
        raise ValueError("No clusters supplied")
    rng = np.random.default_rng(seed)
    cluster_array = [np.asarray(cluster, dtype=np.int64) for cluster in clusters]
    n_clusters = len(cluster_array)
    output = []
    for _ in range(resamples):
        chosen = rng.integers(0, n_clusters, size=n_clusters, dtype=np.int64)
        output.append(np.concatenate([cluster_array[index] for index in chosen]))
    return output


def summarize_bootstrap(values, point_estimate=None, alpha: float = 0.05) -> dict:
    """Summarize a bootstrap distribution with a percentile confidence interval."""
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {
            "point": None if point_estimate is None else float(point_estimate),
            "bootstrap_mean": None,
            "bootstrap_median": None,
            "ci_low": None,
            "ci_high": None,
            "std": None,
            "resamples": int(array.size),
            "finite_resamples": 0,
        }
    low, high = np.percentile(finite, [100 * alpha / 2.0, 100 * (1.0 - alpha / 2.0)])
    return {
        "point": None if point_estimate is None else float(point_estimate),
        "bootstrap_mean": float(np.mean(finite)),
        "bootstrap_median": float(np.median(finite)),
        "ci_low": float(low),
        "ci_high": float(high),
        "std": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
        "resamples": int(array.size),
        "finite_resamples": int(finite.size),
    }


def paired_delta_summary(deltas, point_delta: float, alpha: float = 0.05) -> dict:
    """Summarize a paired bootstrap distribution and approximate p-values."""
    array = np.asarray(deltas, dtype=np.float64)
    finite = array[np.isfinite(array)]
    summary = summarize_bootstrap(array, point_delta, alpha=alpha)
    if finite.size == 0:
        summary.update(
            {
                "fraction_positive": None,
                "p_two_sided": None,
                "p_one_sided_improvement": None,
            }
        )
        return summary

    n_le = int(np.sum(finite <= 0.0))
    n_ge = int(np.sum(finite >= 0.0))
    denominator = finite.size + 1
    p_two = min(1.0, 2.0 * min((n_le + 1) / denominator, (n_ge + 1) / denominator))
    summary.update(
        {
            "fraction_positive": float(np.mean(finite > 0.0)),
            "p_two_sided": float(p_two),
            "p_one_sided_improvement": float((n_ge + 1) / denominator),
        }
    )
    return summary


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, preserving input order."""
    values = [float(value) for value in p_values]
    order = sorted(range(len(values)), key=lambda index: values[index])
    adjusted = [1.0] * len(values)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = (len(values) - rank) * values[index]
        running = max(running, candidate)
        adjusted[index] = min(1.0, running)
    return adjusted


def source_clusters(image_names: Sequence[str], stem_function) -> list[list[int]]:
    """Group image indices by source filename stem."""
    groups: dict[str, list[int]] = {}
    for index, name in enumerate(image_names):
        groups.setdefault(stem_function(name), []).append(index)
    return [groups[key] for key in sorted(groups)]
