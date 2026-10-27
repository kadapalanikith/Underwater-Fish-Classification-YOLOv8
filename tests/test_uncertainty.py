"""Focused tests for the bootstrap/AP evaluation helpers.

These tests use synthetic boxes and do not run model inference.
"""

import numpy as np
import pytest

from src.evaluation.uncertainty import (
    IOU_THRESHOLDS,
    box_iou_xyxy,
    build_image_row,
    cluster_bootstrap_indices,
    holm_adjust,
    image_bootstrap_indices,
    match_predictions_sorted,
    metrics_from_rows,
    paired_delta_summary,
    source_clusters,
    summarize_bootstrap,
)


def test_box_iou_identity_and_disjoint():
    boxes = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=float)
    iou = box_iou_xyxy(boxes, boxes)
    assert np.allclose(np.diag(iou), 1.0)
    assert iou[0, 1] == 0.0


def test_match_same_class_only():
    gt = [[0, 0, 10, 10]]
    pred = [[0, 0, 10, 10]]
    tp = match_predictions_sorted([1], gt, [0], pred)
    assert not tp.any(), "Different classes must not match"


def test_duplicate_predictions_only_one_true_positive():
    gt = [[0, 0, 10, 10]]
    pred = [[0, 0, 10, 10], [0, 0, 10, 10]]
    tp = match_predictions_sorted([0], gt, [0, 0], pred)
    assert tp[:, 0].tolist() == [True, False]
    assert tp.shape == (2, len(IOU_THRESHOLDS))


def test_iou_threshold_boundary():
    gt = [[0, 0, 10, 10]]
    pred = [[0, 0, 10, 6]]
    tp = match_predictions_sorted([0], gt, [0], pred)
    assert tp[0, 0], "IoU 0.6 should match at threshold 0.50"
    assert tp[0, 2], "IoU 0.6 should match at threshold 0.60"
    assert not tp[0, 3], "IoU 0.6 should not match at threshold 0.65"


def test_build_image_row_sorts_by_confidence():
    row = build_image_row(
        true_classes=[0],
        true_boxes=[[0, 0, 10, 10]],
        pred_classes=[0, 0],
        pred_confidences=[0.2, 0.9],
        pred_boxes=[[0, 0, 10, 10], [100, 100, 110, 110]],
    )
    assert row["conf"].tolist() == [0.9, 0.2]
    assert row["pred_cls"].tolist() == [0, 0]
    assert row["tp"][0, 0] is np.False_
    assert row["tp"][1, 0] is np.True_


def test_metrics_from_rows_perfect_detection():
    row = build_image_row(
        true_classes=[0, 1],
        true_boxes=[[0, 0, 10, 10], [20, 20, 30, 30]],
        pred_classes=[0, 1],
        pred_confidences=[0.9, 0.8],
        pred_boxes=[[0, 0, 10, 10], [20, 20, 30, 30]],
    )
    metrics = metrics_from_rows([row], [0], names=["A", "B"])
    assert metrics["overall"]["mAP50"] == pytest.approx(0.995)
    assert metrics["overall"]["mAP50_95"] == pytest.approx(0.995)
    assert metrics["per_class"]["A"]["instances"] == 1
    assert metrics["per_class"]["B"]["instances"] == 1


def test_metrics_from_rows_missed_class_has_zero_ap():
    row = build_image_row(
        true_classes=[0, 1],
        true_boxes=[[0, 0, 10, 10], [20, 20, 30, 30]],
        pred_classes=[0],
        pred_confidences=[0.9],
        pred_boxes=[[0, 0, 10, 10]],
    )
    metrics = metrics_from_rows([row], [0], names=["A", "B"])
    assert metrics["per_class"]["A"]["ap50"] == pytest.approx(0.995)
    assert metrics["per_class"]["B"]["ap50"] == 0.0
    assert metrics["overall"]["mAP50"] < 1.0


def test_metrics_from_rows_no_predictions_is_zero():
    row = build_image_row(
        true_classes=[0],
        true_boxes=[[0, 0, 10, 10]],
        pred_classes=[],
        pred_confidences=[],
        pred_boxes=[],
    )
    metrics = metrics_from_rows([row], [0], names=["A"])
    assert metrics["overall"]["mAP50"] == 0.0
    assert metrics["overall"]["precision"] == 0.0


def test_image_bootstrap_is_deterministic():
    first = image_bootstrap_indices(10, 5, 123)
    second = image_bootstrap_indices(10, 5, 123)
    assert np.array_equal(first, second)
    assert first.shape == (5, 10)
    assert first.min() >= 0 and first.max() < 10


def test_cluster_bootstrap_keeps_cluster_images_together():
    clusters = [[0, 1], [2], [3, 4]]
    samples = cluster_bootstrap_indices(clusters, 4, 7)
    assert len(samples) == 4
    for sample in samples:
        for image_index in (0, 1):
            assert (image_index in sample) == (0 in sample) or (1 in sample)
    # Each selected cluster contributes all of its images.
    allowed = {0, 1, 2, 3, 4}
    assert set(np.concatenate(samples).tolist()).issubset(allowed)


def test_summarize_bootstrap_percentiles():
    summary = summarize_bootstrap(np.arange(101, dtype=float), point_estimate=50.0)
    assert summary["point"] == 50.0
    assert summary["ci_low"] == pytest.approx(2.5)
    assert summary["ci_high"] == pytest.approx(97.5)
    assert summary["finite_resamples"] == 101


def test_paired_delta_summary_signals_positive_shift():
    deltas = np.linspace(0.01, 0.05, 200)
    summary = paired_delta_summary(deltas, point_delta=0.03)
    assert summary["fraction_positive"] == 1.0
    assert summary["ci_low"] > 0
    assert summary["p_two_sided"] < 0.05


def test_holm_adjust_preserves_order_and_monotonicity():
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    assert adjusted[0] <= adjusted[2] <= adjusted[1]
    assert all(0.0 <= value <= 1.0 for value in adjusted)


def test_source_clusters_group_roboflow_stems():
    from scratch.eval_common import source_stem

    names = [
        "AngelFish_image_1_jpg.rf.aaa.jpg",
        "AngelFish_image_1_jpg.rf.bbb.jpg",
        "ClownFish_image_2_jpg.rf.ccc.jpg",
    ]
    clusters = source_clusters(names, source_stem)
    assert sorted(len(cluster) for cluster in clusters) == [1, 2]
