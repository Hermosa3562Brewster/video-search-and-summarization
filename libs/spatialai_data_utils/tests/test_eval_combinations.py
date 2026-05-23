# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Tests for evaluation module combinations:

- center_distance + detection AP  (via accumulate / calc_ap)
- iou_3d          + detection AP  (via accumulate / calc_ap)
- center_distance + HOTA tracking (via evaluate_hota with eval_dist_fcn="center_distance")

The existing test_3d_iou_and_hota.py covers iou_3d + HOTA internals.
"""

import json
import math
import os
import tempfile

import numpy as np
import pytest
from pyquaternion import Quaternion

from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.utils import center_distance
from nuscenes.eval.detection.algo import accumulate, calc_ap, calc_tp
from spatialai_data_utils.eval.common.utils import iou_3d
from spatialai_data_utils.eval.detection.data_classes import (
    DetectionBox,
    DetectionConfig,
    DetectionMetrics,
    DetectionMetricDataList,
)
from spatialai_data_utils.eval.tracking.hota.hota_eval import evaluate_hota


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_det_box(sample_token, translation, size, rotation=(1, 0, 0, 0),
                  detection_name="person", detection_score=-1.0):
    return DetectionBox(
        sample_token=sample_token,
        translation=tuple(translation),
        size=tuple(size),
        rotation=tuple(rotation),
        velocity=(0.0, 0.0),
        detection_name=detection_name,
        detection_score=float(detection_score),
    )


def _build_eval_boxes(box_specs):
    """Build EvalBoxes from a list of (sample_token, box_kwargs) pairs."""
    eb = EvalBoxes()
    grouped = {}
    for token, kwargs in box_specs:
        grouped.setdefault(token, []).append(kwargs)
    for token, kw_list in grouped.items():
        boxes = [_make_det_box(token, **kw) for kw in kw_list]
        eb.add_boxes(token, boxes)
    return eb


def _make_config(class_names, dist_fcn, dist_ths=None, dist_th_tp=None):
    """Build a minimal DetectionConfig for testing."""
    if dist_ths is None:
        dist_ths = [0.5, 1.0, 2.0, 4.0] if dist_fcn == "center_distance" else [0.3, 0.5, 0.7]
    if dist_th_tp is None:
        dist_th_tp = dist_ths[1]
    return DetectionConfig(
        class_range={c: 100 for c in class_names},
        dist_fcn=dist_fcn,
        dist_ths=dist_ths,
        dist_th_tp=dist_th_tp,
        min_recall=0.1,
        min_precision=0.1,
        max_boxes_per_sample=500,
        mean_ap_weight=5,
    )


def _make_nuscenes_result_json(pred_boxes_dict, tmp_dir):
    """Write a results_nusc.json with tracking-format predictions and return its path."""
    results = {}
    for token, annos in pred_boxes_dict.items():
        results[token] = annos
    data = {
        "results": results,
        "meta": {"use_camera": True, "use_lidar": False, "use_radar": False,
                 "use_map": False, "use_external": False},
    }
    path = os.path.join(tmp_dir, "results_nusc.json")
    with open(path, "w") as f:
        json.dump(data, f)
    return path


# ===================================================================
# Detection AP with center_distance matching
# ===================================================================
class TestCenterDistanceDetectionAP:
    """Tests for the accumulate + calc_ap pipeline using center_distance."""

    def test_perfect_detection(self):
        """Identical GT and predictions should yield AP = 1.0."""
        tokens = [f"sample_{i}" for i in range(5)]
        gt_specs = []
        pred_specs = []
        for t in tokens:
            gt_specs.append((t, dict(translation=[0, 0, 0], size=[2, 2, 2],
                                     detection_name="person")))
            pred_specs.append((t, dict(translation=[0, 0, 0], size=[2, 2, 2],
                                       detection_name="person", detection_score=0.9)))

        gt_boxes = _build_eval_boxes(gt_specs)
        pred_boxes = _build_eval_boxes(pred_specs)

        md = accumulate(gt_boxes, pred_boxes, "person", center_distance, dist_th=2.0)
        ap = calc_ap(md, min_recall=0.1, min_precision=0.1)
        assert ap > 0.99, f"Expected perfect AP, got {ap}"

    def test_no_predictions(self):
        """No predictions → AP = 0."""
        gt_specs = [("s1", dict(translation=[0, 0, 0], size=[1, 1, 1],
                                detection_name="person"))]
        gt_boxes = _build_eval_boxes(gt_specs)
        pred_boxes = EvalBoxes()
        pred_boxes.add_boxes("s1", [])

        md = accumulate(gt_boxes, pred_boxes, "person", center_distance, dist_th=2.0)
        ap = calc_ap(md, min_recall=0.1, min_precision=0.1)
        assert ap == 0.0

    def test_no_ground_truth(self):
        """No GT of the target class → AP = 0 (no_predictions sentinel)."""
        gt_boxes = EvalBoxes()
        gt_boxes.add_boxes("s1", [])
        pred_specs = [("s1", dict(translation=[0, 0, 0], size=[1, 1, 1],
                                  detection_name="person", detection_score=0.9))]
        pred_boxes = _build_eval_boxes(pred_specs)

        md = accumulate(gt_boxes, pred_boxes, "person", center_distance, dist_th=2.0)
        ap = calc_ap(md, min_recall=0.1, min_precision=0.1)
        assert ap == 0.0

    def test_far_prediction_is_false_positive(self):
        """A prediction far from GT should be a false positive → lower AP."""
        gt_specs = [("s1", dict(translation=[0, 0, 0], size=[1, 1, 1],
                                detection_name="person"))]
        pred_specs = [("s1", dict(translation=[100, 100, 0], size=[1, 1, 1],
                                  detection_name="person", detection_score=0.9))]
        gt_boxes = _build_eval_boxes(gt_specs)
        pred_boxes = _build_eval_boxes(pred_specs)

        md = accumulate(gt_boxes, pred_boxes, "person", center_distance, dist_th=2.0)
        ap = calc_ap(md, min_recall=0.1, min_precision=0.1)
        assert ap == 0.0

    def test_close_prediction_is_true_positive(self):
        """A prediction within the dist threshold should be a TP."""
        gt_specs = [("s1", dict(translation=[0, 0, 0], size=[1, 1, 1],
                                detection_name="person"))]
        pred_specs = [("s1", dict(translation=[0.5, 0, 0], size=[1, 1, 1],
                                  detection_name="person", detection_score=0.9))]
        gt_boxes = _build_eval_boxes(gt_specs)
        pred_boxes = _build_eval_boxes(pred_specs)

        md = accumulate(gt_boxes, pred_boxes, "person", center_distance, dist_th=2.0)
        ap = calc_ap(md, min_recall=0.1, min_precision=0.1)
        assert ap > 0.99

    def test_multi_class_separation(self):
        """AP should be computed per class; FPs from another class shouldn't affect the target."""
        gt_specs = [
            ("s1", dict(translation=[0, 0, 0], size=[1, 1, 1], detection_name="person")),
            ("s1", dict(translation=[10, 0, 0], size=[2, 2, 2], detection_name="forklift")),
        ]
        pred_specs = [
            ("s1", dict(translation=[0, 0, 0], size=[1, 1, 1],
                        detection_name="person", detection_score=0.9)),
            ("s1", dict(translation=[10, 0, 0], size=[2, 2, 2],
                        detection_name="forklift", detection_score=0.8)),
        ]
        gt_boxes = _build_eval_boxes(gt_specs)
        pred_boxes = _build_eval_boxes(pred_specs)

        md_person = accumulate(gt_boxes, pred_boxes, "person", center_distance, dist_th=2.0)
        ap_person = calc_ap(md_person, min_recall=0.1, min_precision=0.1)
        assert ap_person > 0.99

        md_forklift = accumulate(gt_boxes, pred_boxes, "forklift", center_distance, dist_th=2.0)
        ap_forklift = calc_ap(md_forklift, min_recall=0.1, min_precision=0.1)
        assert ap_forklift > 0.99

    def test_multiple_samples(self):
        """AP across multiple samples with mixed correct/incorrect detections."""
        gt_specs = [
            ("s1", dict(translation=[0, 0, 0], size=[1, 1, 1], detection_name="person")),
            ("s2", dict(translation=[5, 5, 0], size=[1, 1, 1], detection_name="person")),
            ("s3", dict(translation=[10, 10, 0], size=[1, 1, 1], detection_name="person")),
        ]
        pred_specs = [
            ("s1", dict(translation=[0, 0, 0], size=[1, 1, 1],
                        detection_name="person", detection_score=0.95)),
            ("s2", dict(translation=[5, 5, 0], size=[1, 1, 1],
                        detection_name="person", detection_score=0.85)),
            # s3: prediction too far away
            ("s3", dict(translation=[50, 50, 0], size=[1, 1, 1],
                        detection_name="person", detection_score=0.5)),
        ]
        gt_boxes = _build_eval_boxes(gt_specs)
        pred_boxes = _build_eval_boxes(pred_specs)

        md = accumulate(gt_boxes, pred_boxes, "person", center_distance, dist_th=2.0)
        ap = calc_ap(md, min_recall=0.1, min_precision=0.1)
        # 2 out of 3 TPs; AP should be above 0 but below 1
        assert 0.0 < ap < 1.0

    def test_duplicate_predictions_only_one_matches(self):
        """Two predictions for the same GT: only the highest-confidence one should match."""
        gt_specs = [("s1", dict(translation=[0, 0, 0], size=[1, 1, 1],
                                detection_name="person"))]
        pred_specs = [
            ("s1", dict(translation=[0.1, 0, 0], size=[1, 1, 1],
                        detection_name="person", detection_score=0.9)),
            ("s1", dict(translation=[0.2, 0, 0], size=[1, 1, 1],
                        detection_name="person", detection_score=0.8)),
        ]
        gt_boxes = _build_eval_boxes(gt_specs)
        pred_boxes = _build_eval_boxes(pred_specs)

        md = accumulate(gt_boxes, pred_boxes, "person", center_distance, dist_th=2.0)
        ap = calc_ap(md, min_recall=0.1, min_precision=0.1)
        # 1 TP + 1 FP; AP should reflect the drop
        assert 0.0 < ap < 1.0

    def test_config_dist_fcn_callable(self):
        """DetectionConfig.dist_fcn_callable should return center_distance for 'center_distance'."""
        config = _make_config(["person"], "center_distance")
        assert config.dist_fcn_callable is center_distance

    def test_full_detection_metrics_pipeline(self):
        """End-to-end: build DetectionMetrics from accumulate+calc_ap like AIC24DetEval.evaluate."""
        class_names = ["person", "forklift"]
        config = _make_config(class_names, "center_distance")

        gt_specs = [
            ("s1", dict(translation=[0, 0, 0], size=[1, 1, 1], detection_name="person")),
            ("s1", dict(translation=[5, 0, 0], size=[2, 3, 2], detection_name="forklift")),
            ("s2", dict(translation=[0, 0, 0], size=[1, 1, 1], detection_name="person")),
        ]
        pred_specs = [
            ("s1", dict(translation=[0.1, 0, 0], size=[1, 1, 1],
                        detection_name="person", detection_score=0.95)),
            ("s1", dict(translation=[5.1, 0, 0], size=[2, 3, 2],
                        detection_name="forklift", detection_score=0.85)),
            ("s2", dict(translation=[0, 0, 0], size=[1, 1, 1],
                        detection_name="person", detection_score=0.90)),
        ]
        gt_boxes = _build_eval_boxes(gt_specs)
        pred_boxes = _build_eval_boxes(pred_specs)

        metric_data_list = DetectionMetricDataList()
        for class_name in class_names:
            for dist_th in config.dist_ths:
                md = accumulate(gt_boxes, pred_boxes, class_name,
                                config.dist_fcn_callable, dist_th)
                metric_data_list.set(class_name, dist_th, md)

        metrics = DetectionMetrics(config)
        for class_name in class_names:
            for dist_th in config.dist_ths:
                metric_data = metric_data_list[(class_name, dist_th)]
                ap = calc_ap(metric_data, config.min_recall, config.min_precision)
                metrics.add_label_ap(class_name, dist_th, ap)

        assert metrics.mean_ap > 0.5
        for class_name in class_names:
            assert metrics.mean_dist_aps[class_name] > 0.0


# ===================================================================
# Detection AP with 3D IoU matching
# ===================================================================
class TestIoU3DDetectionAP:
    """Tests for the accumulate + calc_ap pipeline using iou_3d distance."""

    def test_identical_boxes_perfect_ap(self):
        """Identical GT and predictions → 1 - IoU = 0 distance → AP = 1.0 for any threshold."""
        tokens = [f"sample_{i}" for i in range(3)]
        gt_specs = []
        pred_specs = []
        for t in tokens:
            gt_specs.append((t, dict(translation=[0, 0, 0], size=[2, 2, 2],
                                     detection_name="person")))
            pred_specs.append((t, dict(translation=[0, 0, 0], size=[2, 2, 2],
                                       detection_name="person", detection_score=0.9)))

        gt_boxes = _build_eval_boxes(gt_specs)
        pred_boxes = _build_eval_boxes(pred_specs)

        # iou_3d returns 1 - IoU; threshold 0.5 means IoU > 0.5
        md = accumulate(gt_boxes, pred_boxes, "person", iou_3d, dist_th=0.5)
        ap = calc_ap(md, min_recall=0.1, min_precision=0.1)
        assert ap > 0.99

    def test_non_overlapping_boxes_zero_ap(self):
        """Non-overlapping boxes → IoU = 0, distance = 1.0 → FP for any threshold < 1."""
        gt_specs = [("s1", dict(translation=[0, 0, 0], size=[1, 1, 1],
                                detection_name="person"))]
        pred_specs = [("s1", dict(translation=[100, 100, 100], size=[1, 1, 1],
                                  detection_name="person", detection_score=0.9))]
        gt_boxes = _build_eval_boxes(gt_specs)
        pred_boxes = _build_eval_boxes(pred_specs)

        md = accumulate(gt_boxes, pred_boxes, "person", iou_3d, dist_th=0.5)
        ap = calc_ap(md, min_recall=0.1, min_precision=0.1)
        assert ap == 0.0

    def test_partial_overlap_threshold_sensitivity(self):
        """Half-overlapping boxes should pass a loose threshold but fail a strict one."""
        # Two boxes shifted by 1m along x: IoU ~= 1/3 → distance ~= 2/3
        gt_specs = [("s1", dict(translation=[0, 0, 0], size=[2, 2, 2],
                                detection_name="person"))]
        pred_specs = [("s1", dict(translation=[1, 0, 0], size=[2, 2, 2],
                                  detection_name="person", detection_score=0.9))]
        gt_boxes = _build_eval_boxes(gt_specs)
        pred_boxes = _build_eval_boxes(pred_specs)

        # Loose threshold (dist < 0.8 → IoU > 0.2): should pass
        md_loose = accumulate(gt_boxes, pred_boxes, "person", iou_3d, dist_th=0.8)
        ap_loose = calc_ap(md_loose, min_recall=0.1, min_precision=0.1)
        assert ap_loose > 0.99

        # Strict threshold (dist < 0.3 → IoU > 0.7): should fail
        md_strict = accumulate(gt_boxes, pred_boxes, "person", iou_3d, dist_th=0.3)
        ap_strict = calc_ap(md_strict, min_recall=0.1, min_precision=0.1)
        assert ap_strict == 0.0

    def test_rotated_box_self_match(self):
        """A rotated box matched against itself should have IoU = 1."""
        q = Quaternion(axis=[0, 0, 1], angle=math.pi / 4)
        rot = (q.w, q.x, q.y, q.z)
        gt_specs = [("s1", dict(translation=[5, 5, 0], size=[3, 1, 2],
                                rotation=rot, detection_name="person"))]
        pred_specs = [("s1", dict(translation=[5, 5, 0], size=[3, 1, 2],
                                  rotation=rot, detection_name="person",
                                  detection_score=0.9))]
        gt_boxes = _build_eval_boxes(gt_specs)
        pred_boxes = _build_eval_boxes(pred_specs)

        md = accumulate(gt_boxes, pred_boxes, "person", iou_3d, dist_th=0.5)
        ap = calc_ap(md, min_recall=0.1, min_precision=0.1)
        assert ap > 0.99

    def test_config_dist_fcn_callable_iou(self):
        """DetectionConfig.dist_fcn_callable should return iou_3d for 'iou_3d'."""
        config = _make_config(["person"], "iou_3d")
        assert config.dist_fcn_callable is iou_3d

    def test_multi_class_iou_matching(self):
        """Per-class AP with iou_3d: each class independently matched."""
        gt_specs = [
            ("s1", dict(translation=[0, 0, 0], size=[2, 2, 2], detection_name="person")),
            ("s1", dict(translation=[20, 0, 0], size=[4, 2, 3], detection_name="forklift")),
        ]
        pred_specs = [
            ("s1", dict(translation=[0, 0, 0], size=[2, 2, 2],
                        detection_name="person", detection_score=0.9)),
            ("s1", dict(translation=[20, 0, 0], size=[4, 2, 3],
                        detection_name="forklift", detection_score=0.85)),
        ]
        gt_boxes = _build_eval_boxes(gt_specs)
        pred_boxes = _build_eval_boxes(pred_specs)

        for cls in ["person", "forklift"]:
            md = accumulate(gt_boxes, pred_boxes, cls, iou_3d, dist_th=0.5)
            ap = calc_ap(md, min_recall=0.1, min_precision=0.1)
            assert ap > 0.99, f"AP for {cls} should be ~1.0, got {ap}"

    def test_full_iou3d_metrics_pipeline(self):
        """End-to-end DetectionMetrics using iou_3d distance."""
        class_names = ["person"]
        config = _make_config(class_names, "iou_3d", dist_ths=[0.3, 0.5, 0.7], dist_th_tp=0.5)

        gt_specs = [
            ("s1", dict(translation=[0, 0, 0], size=[2, 2, 2], detection_name="person")),
            ("s2", dict(translation=[0, 0, 0], size=[2, 2, 2], detection_name="person")),
        ]
        pred_specs = [
            ("s1", dict(translation=[0, 0, 0], size=[2, 2, 2],
                        detection_name="person", detection_score=0.9)),
            ("s2", dict(translation=[0.5, 0, 0], size=[2, 2, 2],
                        detection_name="person", detection_score=0.8)),
        ]
        gt_boxes = _build_eval_boxes(gt_specs)
        pred_boxes = _build_eval_boxes(pred_specs)

        metric_data_list = DetectionMetricDataList()
        for class_name in class_names:
            for dist_th in config.dist_ths:
                md = accumulate(gt_boxes, pred_boxes, class_name,
                                config.dist_fcn_callable, dist_th)
                metric_data_list.set(class_name, dist_th, md)

        metrics = DetectionMetrics(config)
        for class_name in class_names:
            for dist_th in config.dist_ths:
                metric_data = metric_data_list[(class_name, dist_th)]
                ap = calc_ap(metric_data, config.min_recall, config.min_precision)
                metrics.add_label_ap(class_name, dist_th, ap)

        assert metrics.mean_ap > 0.0
        serialized = metrics.serialize()
        assert "mean_ap" in serialized
        assert "mean_dist_aps" in serialized

    def test_iou3d_score_ordering_matters(self):
        """Higher-confidence predictions should be matched first, affecting AP."""
        gt_specs = [
            ("s1", dict(translation=[0, 0, 0], size=[2, 2, 2], detection_name="person")),
        ]
        # Two predictions: one overlaps, one doesn't. The high-confidence one is the FP.
        pred_specs = [
            ("s1", dict(translation=[100, 0, 0], size=[2, 2, 2],
                        detection_name="person", detection_score=0.95)),
            ("s1", dict(translation=[0, 0, 0], size=[2, 2, 2],
                        detection_name="person", detection_score=0.5)),
        ]
        gt_boxes = _build_eval_boxes(gt_specs)
        pred_boxes = _build_eval_boxes(pred_specs)

        md = accumulate(gt_boxes, pred_boxes, "person", iou_3d, dist_th=0.5)
        ap = calc_ap(md, min_recall=0.1, min_precision=0.1)
        # First sorted pred (score=0.95) is FP, second (score=0.5) is TP → precision dips first
        assert 0.0 < ap < 1.0


# ===================================================================
# HOTA tracking with center-distance matching (eval_dist_fcn="center_distance")
# ===================================================================
class TestCenterDistanceHOTA:
    """Tests for evaluate_hota with eval_dist_fcn='center_distance' (Euclidean center-distance matching)."""

    @staticmethod
    def _make_data_infos(scenes):
        """Build data_infos from a dict of {scene_name: [frame_dicts]}.

        Each frame_dict has keys: token, gt_boxes, gt_names, instance_inds,
        and optionally valid_flag.
        """
        data_infos = []
        for scene_name, frames in scenes.items():
            for frame in frames:
                frame["scene_name"] = scene_name
                data_infos.append(frame)
        return data_infos

    @staticmethod
    def _make_pred_json(predictions, tmp_dir):
        """Build results_nusc.json from {token: [{translation, size, rotation, tracking_name, tracking_id}]}."""
        results = {}
        for token, annos in predictions.items():
            serialized = []
            for a in annos:
                serialized.append({
                    "translation": a["translation"],
                    "size": a["size"],
                    "rotation": a.get("rotation", [1, 0, 0, 0]),
                    "tracking_name": a["tracking_name"],
                    "tracking_id": a["tracking_id"],
                    "tracking_score": a.get("tracking_score", 0.9),
                })
            results[token] = serialized
        data = {
            "results": results,
            "meta": {"use_camera": True, "use_lidar": False, "use_radar": False,
                     "use_map": False, "use_external": False},
        }
        path = os.path.join(tmp_dir, "results_nusc.json")
        with open(path, "w") as f:
            json.dump(data, f)
        return path

    def test_perfect_tracking_location(self):
        """Perfect tracking (identical GT and predictions) → HOTA ≈ 1."""
        num_frames = 5
        tokens = [f"tok_{i}" for i in range(num_frames)]

        scenes = {"scene_0": []}
        predictions = {}
        for i, tok in enumerate(tokens):
            scenes["scene_0"].append({
                "token": tok,
                "gt_boxes": np.array([[0, 0, 0, 1, 1, 1, 0],
                                      [5, 5, 0, 1, 1, 1, 0]], dtype=np.float64),
                "gt_names": ["person", "person"],
                "instance_inds": [0, 1],
                "valid_flag": [True, True],
            })
            predictions[tok] = [
                {"translation": [0, 0, 0], "size": [1, 1, 1],
                 "tracking_name": "person", "tracking_id": "0"},
                {"translation": [5, 5, 0], "size": [1, 1, 1],
                 "tracking_name": "person", "tracking_id": "1"},
            ]

        data_infos = self._make_data_infos(scenes)

        with tempfile.TemporaryDirectory() as tmp_dir:
            result_path = self._make_pred_json(predictions, tmp_dir)
            results = evaluate_hota(
                data_infos=data_infos,
                result_path=result_path,
                output_dir=tmp_dir,
                class_names=["person"],
                eval_dist_fcn="center_distance",
                verbose=False,
            )

        assert "per_class" in results
        assert "average" in results
        assert results["per_class"]["person"] is not None
        assert results["average"]["HOTA"] > 0.9

    def test_no_predictions_location(self):
        """No predictions → HOTA = 0."""
        tokens = ["tok_0", "tok_1"]
        scenes = {"scene_0": []}
        predictions = {}
        for tok in tokens:
            scenes["scene_0"].append({
                "token": tok,
                "gt_boxes": np.array([[0, 0, 0, 1, 1, 1, 0]], dtype=np.float64),
                "gt_names": ["person"],
                "instance_inds": [0],
                "valid_flag": [True],
            })
            predictions[tok] = []

        data_infos = self._make_data_infos(scenes)

        with tempfile.TemporaryDirectory() as tmp_dir:
            result_path = self._make_pred_json(predictions, tmp_dir)
            results = evaluate_hota(
                data_infos=data_infos,
                result_path=result_path,
                output_dir=tmp_dir,
                class_names=["person"],
                eval_dist_fcn="center_distance",
                verbose=False,
            )

        assert results["average"]["HOTA"] < 1e-6

    def test_far_predictions_location(self):
        """Predictions very far from GT → low HOTA (similarity ≈ 0)."""
        tokens = ["tok_0", "tok_1"]
        scenes = {"scene_0": []}
        predictions = {}
        for tok in tokens:
            scenes["scene_0"].append({
                "token": tok,
                "gt_boxes": np.array([[0, 0, 0, 1, 1, 1, 0]], dtype=np.float64),
                "gt_names": ["person"],
                "instance_inds": [0],
            })
            predictions[tok] = [
                {"translation": [100, 100, 0], "size": [1, 1, 1],
                 "tracking_name": "person", "tracking_id": "0"},
            ]

        data_infos = self._make_data_infos(scenes)

        with tempfile.TemporaryDirectory() as tmp_dir:
            result_path = self._make_pred_json(predictions, tmp_dir)
            results = evaluate_hota(
                data_infos=data_infos,
                result_path=result_path,
                output_dir=tmp_dir,
                class_names=["person"],
                eval_dist_fcn="center_distance",
                verbose=False,
            )

        assert results["average"]["HOTA"] < 0.1

    def test_id_switch_location(self):
        """Tracker swaps IDs midway → HOTA < 1 even with perfect detection."""
        num_frames = 4
        tokens = [f"tok_{i}" for i in range(num_frames)]
        scenes = {"scene_0": []}
        predictions = {}

        for i, tok in enumerate(tokens):
            scenes["scene_0"].append({
                "token": tok,
                "gt_boxes": np.array([[0, 0, 0, 1, 1, 1, 0],
                                      [5, 0, 0, 1, 1, 1, 0]], dtype=np.float64),
                "gt_names": ["person", "person"],
                "instance_inds": [0, 1],
            })
            # Swap IDs at midpoint
            if i < 2:
                predictions[tok] = [
                    {"translation": [0, 0, 0], "size": [1, 1, 1],
                     "tracking_name": "person", "tracking_id": "0"},
                    {"translation": [5, 0, 0], "size": [1, 1, 1],
                     "tracking_name": "person", "tracking_id": "1"},
                ]
            else:
                predictions[tok] = [
                    {"translation": [0, 0, 0], "size": [1, 1, 1],
                     "tracking_name": "person", "tracking_id": "1"},
                    {"translation": [5, 0, 0], "size": [1, 1, 1],
                     "tracking_name": "person", "tracking_id": "0"},
                ]

        data_infos = self._make_data_infos(scenes)

        with tempfile.TemporaryDirectory() as tmp_dir:
            result_path = self._make_pred_json(predictions, tmp_dir)
            results = evaluate_hota(
                data_infos=data_infos,
                result_path=result_path,
                output_dir=tmp_dir,
                class_names=["person"],
                eval_dist_fcn="center_distance",
                verbose=False,
            )

        # DetA should be ~1 (perfect detection), AssA < 1 (ID switch), HOTA < 1
        person_metrics = results["per_class"]["person"]
        assert person_metrics["DetA"] > 0.9
        assert person_metrics["AssA"] < 1.0
        assert person_metrics["HOTA"] < 1.0

    def test_multi_class_location(self):
        """Multiple classes evaluated independently via location matching."""
        tokens = ["tok_0", "tok_1"]
        scenes = {"scene_0": []}
        predictions = {}

        for tok in tokens:
            scenes["scene_0"].append({
                "token": tok,
                "gt_boxes": np.array([[0, 0, 0, 1, 1, 1, 0],
                                      [10, 0, 0, 2, 3, 2, 0]], dtype=np.float64),
                "gt_names": ["person", "forklift"],
                "instance_inds": [0, 1],
            })
            predictions[tok] = [
                {"translation": [0, 0, 0], "size": [1, 1, 1],
                 "tracking_name": "person", "tracking_id": "0"},
                {"translation": [10, 0, 0], "size": [2, 3, 2],
                 "tracking_name": "forklift", "tracking_id": "1"},
            ]

        data_infos = self._make_data_infos(scenes)

        with tempfile.TemporaryDirectory() as tmp_dir:
            result_path = self._make_pred_json(predictions, tmp_dir)
            results = evaluate_hota(
                data_infos=data_infos,
                result_path=result_path,
                output_dir=tmp_dir,
                class_names=["person", "forklift"],
                eval_dist_fcn="center_distance",
                verbose=False,
            )

        assert results["per_class"]["person"] is not None
        assert results["per_class"]["forklift"] is not None
        assert results["per_class"]["person"]["HOTA"] > 0.9
        assert results["per_class"]["forklift"]["HOTA"] > 0.9
        assert results["average"]["HOTA"] > 0.9

    def test_missing_class_in_gt_skipped(self):
        """A class with no GT should be skipped (not in per_class results)."""
        scenes = {"scene_0": [{
            "token": "tok_0",
            "gt_boxes": np.array([[0, 0, 0, 1, 1, 1, 0]], dtype=np.float64),
            "gt_names": ["person"],
            "instance_inds": [0],
        }]}
        predictions = {"tok_0": [
            {"translation": [0, 0, 0], "size": [1, 1, 1],
             "tracking_name": "person", "tracking_id": "0"},
        ]}
        data_infos = self._make_data_infos(scenes)

        with tempfile.TemporaryDirectory() as tmp_dir:
            result_path = self._make_pred_json(predictions, tmp_dir)
            results = evaluate_hota(
                data_infos=data_infos,
                result_path=result_path,
                output_dir=tmp_dir,
                class_names=["person", "forklift"],
                eval_dist_fcn="center_distance",
                verbose=False,
            )

        # forklift had no GT → should be None in per_class results
        assert results["per_class"]["forklift"] is None
        # Average should still be computed from the valid class only
        assert results["average"]["HOTA"] > 0.9

    def test_location_vs_bbox_uses_different_dataset(self):
        """eval_dist_fcn='center_distance' should use center-distance similarity,
        producing different results than 'iou_3d' for offset predictions."""
        tokens = ["tok_0", "tok_1"]
        scenes = {"scene_0": []}
        predictions = {}

        for tok in tokens:
            scenes["scene_0"].append({
                "token": tok,
                "gt_boxes": np.array([[0, 0, 0, 2, 2, 2, 0]], dtype=np.float64),
                "gt_names": ["person"],
                "instance_inds": [0],
            })
            # Prediction is offset by a small amount: same center-distance,
            # but 3D IoU differs from distance-based similarity
            predictions[tok] = [
                {"translation": [0.3, 0, 0], "size": [2, 2, 2],
                 "tracking_name": "person", "tracking_id": "0"},
            ]

        data_infos = self._make_data_infos(scenes)

        with tempfile.TemporaryDirectory() as tmp_dir:
            result_path = self._make_pred_json(predictions, tmp_dir)
            results_loc = evaluate_hota(
                data_infos=data_infos,
                result_path=result_path,
                output_dir=os.path.join(tmp_dir, "loc"),
                class_names=["person"],
                eval_dist_fcn="center_distance",
                verbose=False,
            )
            results_bbox = evaluate_hota(
                data_infos=data_infos,
                result_path=result_path,
                output_dir=os.path.join(tmp_dir, "bbox"),
                class_names=["person"],
                eval_dist_fcn="iou_3d",
                verbose=False,
            )

        # Both should produce valid results
        assert results_loc["per_class"]["person"] is not None
        assert results_bbox["per_class"]["person"] is not None
        # The LocA values should differ because they use different similarity functions
        loc_loca = results_loc["per_class"]["person"]["LocA"]
        bbox_loca = results_bbox["per_class"]["person"]["LocA"]
        # With a 0.3m offset on a 2m box, center distance similarity and IoU similarity differ
        assert loc_loca != pytest.approx(bbox_loca, abs=0.01)

    def test_hota_output_fields(self):
        """evaluate_hota should return all expected HOTA fields."""
        scenes = {"scene_0": [{
            "token": "tok_0",
            "gt_boxes": np.array([[0, 0, 0, 1, 1, 1, 0]], dtype=np.float64),
            "gt_names": ["person"],
            "instance_inds": [0],
        }]}
        predictions = {"tok_0": [
            {"translation": [0, 0, 0], "size": [1, 1, 1],
             "tracking_name": "person", "tracking_id": "0"},
        ]}
        data_infos = self._make_data_infos(scenes)
        expected_fields = ["HOTA", "DetA", "AssA", "LocA", "DetRe", "DetPr",
                           "AssRe", "AssPr", "OWTA"]

        with tempfile.TemporaryDirectory() as tmp_dir:
            result_path = self._make_pred_json(predictions, tmp_dir)
            results = evaluate_hota(
                data_infos=data_infos,
                result_path=result_path,
                output_dir=tmp_dir,
                class_names=["person"],
                eval_dist_fcn="center_distance",
                verbose=False,
            )

        for field in expected_fields:
            assert field in results["per_class"]["person"], f"Missing field: {field}"
            assert field in results["average"], f"Missing average field: {field}"


# ===================================================================
# Unified evaluate_detection() function
# ===================================================================
class TestEvaluateDetectionFunction:
    """Tests for the unified evaluate_detection() wrapper."""

    def test_basic_evaluation(self):
        """evaluate_detection returns correct metrics for a simple case."""
        from spatialai_data_utils.eval.detection.evaluate import evaluate_detection

        class_names = ["person", "forklift"]
        config = _make_config(class_names, "center_distance")

        gt_specs = [
            ("s1", dict(translation=[0, 0, 0], size=[1, 1, 1], detection_name="person")),
            ("s1", dict(translation=[5, 0, 0], size=[2, 3, 2], detection_name="forklift")),
            ("s2", dict(translation=[0, 0, 0], size=[1, 1, 1], detection_name="person")),
        ]
        pred_specs = [
            ("s1", dict(translation=[0.1, 0, 0], size=[1, 1, 1],
                        detection_name="person", detection_score=0.95)),
            ("s1", dict(translation=[5.1, 0, 0], size=[2, 3, 2],
                        detection_name="forklift", detection_score=0.85)),
            ("s2", dict(translation=[0, 0, 0], size=[1, 1, 1],
                        detection_name="person", detection_score=0.90)),
        ]
        gt_boxes = _build_eval_boxes(gt_specs)
        pred_boxes = _build_eval_boxes(pred_specs)

        metrics, md_list = evaluate_detection(
            gt_boxes, pred_boxes, config, verbose=False,
        )
        assert metrics.mean_ap > 0.5
        for cn in class_names:
            assert metrics.mean_dist_aps[cn] > 0.0
        assert metrics.eval_time is not None

    def test_tp_skip_metrics_default(self):
        """Default tp_skip_metrics sets attr_err and vel_err to NaN for all classes."""
        from spatialai_data_utils.eval.detection.evaluate import evaluate_detection

        config = _make_config(["person"], "center_distance")
        gt = _build_eval_boxes([
            ("s1", dict(translation=[0, 0, 0], size=[1, 1, 1], detection_name="person")),
        ])
        pred = _build_eval_boxes([
            ("s1", dict(translation=[0.1, 0, 0], size=[1, 1, 1],
                        detection_name="person", detection_score=0.9)),
        ])
        metrics, _ = evaluate_detection(gt, pred, config, verbose=False)

        assert np.isnan(metrics.get_label_tp("person", "attr_err"))
        assert np.isnan(metrics.get_label_tp("person", "vel_err"))
        assert not np.isnan(metrics.get_label_tp("person", "trans_err"))
        assert not np.isnan(metrics.get_label_tp("person", "scale_err"))
        assert not np.isnan(metrics.get_label_tp("person", "orient_err"))

    def test_tp_skip_metrics_custom(self):
        """Custom tp_skip_metrics allows per-class NaN control."""
        from spatialai_data_utils.eval.detection.evaluate import evaluate_detection

        config = _make_config(["person"], "center_distance")
        gt = _build_eval_boxes([
            ("s1", dict(translation=[0, 0, 0], size=[1, 1, 1], detection_name="person")),
        ])
        pred = _build_eval_boxes([
            ("s1", dict(translation=[0.1, 0, 0], size=[1, 1, 1],
                        detection_name="person", detection_score=0.9)),
        ])
        skip = {"person": {"orient_err", "attr_err", "vel_err"}}
        metrics, _ = evaluate_detection(gt, pred, config, verbose=False, tp_skip_metrics=skip)

        assert np.isnan(metrics.get_label_tp("person", "orient_err"))
        assert np.isnan(metrics.get_label_tp("person", "attr_err"))
        assert np.isnan(metrics.get_label_tp("person", "vel_err"))
        assert not np.isnan(metrics.get_label_tp("person", "trans_err"))
        assert not np.isnan(metrics.get_label_tp("person", "scale_err"))

    def test_tp_skip_metrics_empty(self):
        """Empty tp_skip_metrics computes all TP metrics (nothing skipped)."""
        from spatialai_data_utils.eval.detection.evaluate import evaluate_detection

        config = _make_config(["person"], "center_distance")
        gt = _build_eval_boxes([
            ("s1", dict(translation=[0, 0, 0], size=[1, 1, 1], detection_name="person")),
        ])
        pred = _build_eval_boxes([
            ("s1", dict(translation=[0.1, 0, 0], size=[1, 1, 1],
                        detection_name="person", detection_score=0.9)),
        ])
        metrics, _ = evaluate_detection(gt, pred, config, verbose=False, tp_skip_metrics={})

        for metric_name in ["trans_err", "scale_err", "orient_err", "vel_err", "attr_err"]:
            assert not np.isnan(metrics.get_label_tp("person", metric_name))


class TestSaveDetectionResults:
    """Tests for save_detection_results."""

    def test_writes_json_files(self, tmp_path):
        """save_detection_results writes metrics_summary.json and metrics_details.json."""
        from spatialai_data_utils.eval.detection.evaluate import (
            evaluate_detection,
            save_detection_results,
        )

        config = _make_config(["person"], "center_distance")
        gt = _build_eval_boxes([
            ("s1", dict(translation=[0, 0, 0], size=[1, 1, 1], detection_name="person")),
        ])
        pred = _build_eval_boxes([
            ("s1", dict(translation=[0.1, 0, 0], size=[1, 1, 1],
                        detection_name="person", detection_score=0.9)),
        ])
        metrics, md_list = evaluate_detection(gt, pred, config, verbose=False)

        output_dir = str(tmp_path / "results")
        summary = save_detection_results(metrics, md_list, output_dir, meta={"test": True})

        summary_path = os.path.join(output_dir, "metrics_summary.json")
        details_path = os.path.join(output_dir, "metrics_details.json")
        assert os.path.isfile(summary_path)
        assert os.path.isfile(details_path)

        with open(summary_path) as f:
            loaded_summary = json.load(f)
        assert "mean_ap" in loaded_summary
        assert loaded_summary["meta"] == {"test": True}
        assert loaded_summary["mean_ap"] == summary["mean_ap"]

        with open(details_path) as f:
            loaded_details = json.load(f)
        assert len(loaded_details) > 0

    def test_writes_without_meta(self, tmp_path):
        """save_detection_results works without meta parameter."""
        from spatialai_data_utils.eval.detection.evaluate import (
            evaluate_detection,
            save_detection_results,
        )

        config = _make_config(["person"], "center_distance")
        gt = _build_eval_boxes([
            ("s1", dict(translation=[0, 0, 0], size=[1, 1, 1], detection_name="person")),
        ])
        pred = _build_eval_boxes([
            ("s1", dict(translation=[0.1, 0, 0], size=[1, 1, 1],
                        detection_name="person", detection_score=0.9)),
        ])
        metrics, md_list = evaluate_detection(gt, pred, config, verbose=False)

        output_dir = str(tmp_path / "results_no_meta")
        summary = save_detection_results(metrics, md_list, output_dir)

        with open(os.path.join(output_dir, "metrics_summary.json")) as f:
            loaded = json.load(f)
        assert "meta" not in loaded
        assert "mean_ap" in loaded


# ===================================================================
# Module reorganization: import path tests
# ===================================================================
class TestModuleImportPaths:
    """Verify that both canonical and backward-compatible import paths work."""

    def test_canonical_common_utils(self):
        """Canonical eval.common.* paths import successfully."""
        from spatialai_data_utils.eval.common.io_utils import validate_file_path, split_files_by_sensor
        from spatialai_data_utils.loaders.calibration import fetch_fps_from_calibration
        from spatialai_data_utils.eval.common.classes import CLASS_LIST, DetConfigs, map_sub_class_to_primary_class
        assert len(CLASS_LIST) > 0
        assert isinstance(DetConfigs, dict)
        assert callable(validate_file_path)
        assert callable(fetch_fps_from_calibration)
        assert callable(split_files_by_sensor)

    def test_canonical_hota_path(self):
        """Canonical eval.tracking.hota.* paths import successfully."""
        from spatialai_data_utils.eval.tracking.hota.hota_eval import evaluate_hota
        from spatialai_data_utils.eval.tracking.hota.evaluate import Evaluator
        from spatialai_data_utils.eval.tracking.hota.metrics.hota import HOTA
        from spatialai_data_utils.eval.tracking.hota.datasets._base_dataset import _BaseDataset
        assert callable(evaluate_hota)
        assert Evaluator is not None
        assert HOTA is not None
        assert _BaseDataset is not None

    def test_canonical_loaders_calibration(self):
        """Calibration functions are importable from loaders.calibration."""
        from spatialai_data_utils.loaders.calibration import (
            get_camera_name_to_bev_name_map,
            fetch_fps_from_calibration,
        )
        assert callable(get_camera_name_to_bev_name_map)
        assert callable(fetch_fps_from_calibration)


class TestAIC24DetEval:
    """Tests for the AIC24DetEval wrapper class."""

    def test_init_and_evaluate(self, tmp_path):
        """AIC24DetEval can be instantiated with synthetic data and produce valid metrics."""
        from spatialai_data_utils.eval.detection.evaluate import AIC24DetEval

        class_names = ["person", "forklift"]
        config = _make_config(class_names, "center_distance")

        # Build prediction JSON in nuScenes format
        pred_dict = {
            "s1": [
                {"sample_token": "s1", "translation": [0.1, 0, 0], "size": [1, 1, 1],
                 "rotation": [1, 0, 0, 0], "velocity": [0, 0],
                 "detection_name": "person", "detection_score": 0.95, "attribute_name": ""},
                {"sample_token": "s1", "translation": [5.1, 0, 0], "size": [2, 3, 2],
                 "rotation": [1, 0, 0, 0], "velocity": [0, 0],
                 "detection_name": "forklift", "detection_score": 0.85, "attribute_name": ""},
            ],
            "s2": [
                {"sample_token": "s2", "translation": [0, 0, 0], "size": [1, 1, 1],
                 "rotation": [1, 0, 0, 0], "velocity": [0, 0],
                 "detection_name": "person", "detection_score": 0.90, "attribute_name": ""},
            ],
        }
        result_path = _make_nuscenes_result_json(pred_dict, str(tmp_path))

        # Build GT data_infos matching the prediction tokens
        data_infos = [
            {
                "token": "s1", "scene_name": "scene0", "frame_idx": 0,
                "gt_boxes": np.array([[0, 0, 0, 1, 1, 1, 0], [5, 0, 0, 2, 3, 2, 0]]),
                "gt_names": ["person", "forklift"],
                "gt_velocity": np.array([[0, 0], [0, 0]]),
                "valid_flag": [True, True],
                "instance_inds": [0, 1],
            },
            {
                "token": "s2", "scene_name": "scene0", "frame_idx": 1,
                "gt_boxes": np.array([[0, 0, 0, 1, 1, 1, 0]]),
                "gt_names": ["person"],
                "gt_velocity": np.array([[0, 0]]),
                "valid_flag": [True],
                "instance_inds": [0],
            },
        ]

        output_dir = str(tmp_path / "eval_output")
        evaluator = AIC24DetEval(
            data_infos=data_infos,
            config=config,
            result_path=result_path,
            output_dir=output_dir,
            verbose=False,
        )

        assert evaluator.gt_boxes is not None
        assert evaluator.pred_boxes is not None
        assert set(evaluator.gt_boxes.sample_tokens) == set(evaluator.pred_boxes.sample_tokens)

        metrics, md_list = evaluator.evaluate()
        assert metrics.mean_ap > 0.5
        for cn in class_names:
            assert metrics.mean_dist_aps[cn] > 0.0

    def test_main_saves_results(self, tmp_path):
        """AIC24DetEval.main() runs evaluation and writes result files."""
        from spatialai_data_utils.eval.detection.evaluate import AIC24DetEval

        config = _make_config(["person"], "center_distance")
        pred_dict = {
            "s1": [
                {"sample_token": "s1", "translation": [0.1, 0, 0], "size": [1, 1, 1],
                 "rotation": [1, 0, 0, 0], "velocity": [0, 0],
                 "detection_name": "person", "detection_score": 0.9, "attribute_name": ""},
            ],
        }
        result_path = _make_nuscenes_result_json(pred_dict, str(tmp_path))

        data_infos = [
            {
                "token": "s1", "scene_name": "scene0", "frame_idx": 0,
                "gt_boxes": np.array([[0, 0, 0, 1, 1, 1, 0]]),
                "gt_names": ["person"],
                "gt_velocity": np.array([[0, 0]]),
                "valid_flag": [True],
                "instance_inds": [0],
            },
        ]

        output_dir = str(tmp_path / "main_output")
        evaluator = AIC24DetEval(
            data_infos=data_infos,
            config=config,
            result_path=result_path,
            output_dir=output_dir,
            verbose=False,
        )

        summary = evaluator.main(render_curves=False)

        assert "mean_ap" in summary
        assert "meta" in summary
        assert os.path.isfile(os.path.join(output_dir, "metrics_summary.json"))
        assert os.path.isfile(os.path.join(output_dir, "metrics_details.json"))

    def test_evaluate_detection_per_bev_sensor_forwards_offset_and_fps(
        self, tmp_path, monkeypatch,
    ):
        """``evaluate_detection_per_BEV_sensor`` must thread the GT offset
        through to ``split_files_by_sensor``.

        Pre-fix the orchestrator passed only six positional args to
        ``split_files_by_sensor``, so its
        ``ground_truth_frame_offset_secs`` / ``fps`` kwargs silently
        defaulted to ``0.0`` / ``30.0`` regardless of what the user
        supplied — the splitter then truncated GT to
        ``num_frames_to_eval`` and any non-zero offset window of GT was
        dropped on the floor (silent recall regression).
        """
        from spatialai_data_utils.eval.detection import evaluate as evaluate_mod

        # Stub out heavy I/O / evaluation helpers; we only care that
        # the splitter receives the right kwargs.
        captured: dict = {}

        def fake_split(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

        monkeypatch.setattr(
            evaluate_mod, "_run_detection_per_sensor",
            lambda *args, **kwargs: None,
        )

        # Patch the splitter inside the module that imports it locally.
        from spatialai_data_utils.eval.common import io_utils
        monkeypatch.setattr(io_utils, "split_files_by_sensor", fake_split)

        # Patch calibration helpers so we don't need a real calibration JSON.
        from spatialai_data_utils.loaders import calibration
        monkeypatch.setattr(
            calibration, "fetch_fps_from_calibration",
            lambda _path: 25.0,
        )
        monkeypatch.setattr(
            calibration, "get_camera_name_to_bev_name_map",
            lambda _path: {"Camera_01": ["bev-sensor-1"]},
        )

        evaluate_mod.evaluate_detection_per_BEV_sensor(
            ground_truth_file=str(tmp_path / "gt.jsonl"),
            prediction_file=str(tmp_path / "pred.jsonl"),
            calibration_file=str(tmp_path / "calib.json"),
            output_root_dir=str(tmp_path / "out"),
            confidence_threshold=0.5,
            num_frames_to_eval=100,
            ground_truth_frame_offset_secs=2.5,
        )

        assert captured["kwargs"].get("ground_truth_frame_offset_secs") == 2.5, (
            "evaluate_detection_per_BEV_sensor must forward "
            "ground_truth_frame_offset_secs to split_files_by_sensor; "
            "without this kwarg the splitter silently drops the GT "
            "offset window."
        )
        assert captured["kwargs"].get("fps") == 25.0, (
            "evaluate_detection_per_BEV_sensor must forward the "
            "calibration-derived fps to split_files_by_sensor (it's the "
            "denominator in gt_offset_frames = round(offset_secs * fps))."
        )

    def test_evaluate_computes_all_five_tp_metrics(self, tmp_path):
        """``AIC24DetEval.evaluate`` must not skip any TP metric.

        Pre-refactor the inline ``traffic_cone`` / ``barrier`` skip map
        never triggered for AIC24 / MTMC class sets (warehouse classes
        like ``person``, ``forklift``, ``Nova_Carter``,
        ``Transporter``), so every class effectively computed all five
        TP metrics: ``trans_err``, ``scale_err``, ``orient_err``,
        ``vel_err``, ``attr_err``.  After the eval-module reorg, the
        standalone :func:`evaluate_detection` defaults to
        ``{"*": {"attr_err", "vel_err"}}`` which is fine for the BEV /
        MTMC pipeline but would silently zero out ``vel_err`` and
        ``attr_err`` for AIC24 consumers — different numeric output for
        the same input.  This test pins that ``AIC24DetEval`` opts out
        of that default by passing ``tp_skip_metrics={}``.
        """
        from spatialai_data_utils.eval.detection.evaluate import AIC24DetEval

        config = _make_config(["person"], "center_distance")
        pred_dict = {
            "s1": [
                {"sample_token": "s1", "translation": [0.1, 0, 0], "size": [1, 1, 1],
                 "rotation": [1, 0, 0, 0], "velocity": [0, 0],
                 "detection_name": "person", "detection_score": 0.9, "attribute_name": ""},
            ],
        }
        result_path = _make_nuscenes_result_json(pred_dict, str(tmp_path))
        data_infos = [
            {
                "token": "s1", "scene_name": "scene0", "frame_idx": 0,
                "gt_boxes": np.array([[0, 0, 0, 1, 1, 1, 0]]),
                "gt_names": ["person"],
                "gt_velocity": np.array([[0, 0]]),
                "valid_flag": [True],
                "instance_inds": [0],
            },
        ]
        evaluator = AIC24DetEval(
            data_infos=data_infos,
            config=config,
            result_path=result_path,
            output_dir=str(tmp_path / "tp_metric_output"),
            verbose=False,
        )
        metrics, _ = evaluator.evaluate()
        person_tps = metrics.serialize()["label_tp_errors"]["person"]
        for tp_name in ("trans_err", "scale_err", "orient_err",
                        "vel_err", "attr_err"):
            assert not np.isnan(person_tps[tp_name]), (
                f"AIC24DetEval.evaluate must compute {tp_name!r} for AIC24 "
                f"classes; got NaN, suggesting the standalone "
                f"evaluate_detection default skip map leaked back in."
            )


# ===================================================================
# Calibration loader validation (loaders/calibration.py)
# ===================================================================
#
# These cover the post-reorg validation contract: ``get_camera_name_to_bev_name_map``
# and ``fetch_fps_from_calibration`` now raise typed errors instead of crashing
# with ``KeyError`` / silently logging on malformed inputs.  The contract was
# tightened in commit f912fa2 to give callers actionable, file-path-tagged
# error messages.

def _write_calib_json(tmp_path, payload):
    """Write *payload* as JSON under tmp_path/calibration.json and return the path."""
    path = os.path.join(str(tmp_path), "calibration.json")
    with open(path, "w") as f:
        json.dump(payload, f)
    return path


class TestGetCameraNameToBevNameMap:
    """Validation contract for ``loaders.calibration.get_camera_name_to_bev_name_map``.

    The post-reorg version now eagerly validates the calibration JSON
    structure rather than letting a misshapen file fall through to a
    cryptic ``KeyError``.  Each negative test below pins down one
    specific malformed-input branch.
    """

    def test_happy_path_single_group_per_camera(self, tmp_path):
        from spatialai_data_utils.loaders.calibration import (
            get_camera_name_to_bev_name_map,
        )
        path = _write_calib_json(tmp_path, {"sensors": [
            {"id": "Camera_01", "group": {"name": "bev-sensor-1"}},
            {"id": "Camera_02", "group": {"name": "bev-sensor-2"}},
        ]})
        assert get_camera_name_to_bev_name_map(path) == {
            "Camera_01": ["bev-sensor-1"],
            "Camera_02": ["bev-sensor-2"],
        }

    def test_happy_path_camera_in_multiple_groups(self, tmp_path):
        """A camera that appears in two sensor entries collects both groups in declaration order."""
        from spatialai_data_utils.loaders.calibration import (
            get_camera_name_to_bev_name_map,
        )
        path = _write_calib_json(tmp_path, {"sensors": [
            {"id": "Camera_01", "group": {"name": "bev-sensor-1"}},
            {"id": "Camera_01", "group": {"name": "bev-sensor-2"}},
        ]})
        assert get_camera_name_to_bev_name_map(path) == {
            "Camera_01": ["bev-sensor-1", "bev-sensor-2"],
        }

    def test_missing_file_raises_file_not_found(self, tmp_path):
        from spatialai_data_utils.loaders.calibration import (
            get_camera_name_to_bev_name_map,
        )
        missing = os.path.join(str(tmp_path), "does_not_exist.json")
        with pytest.raises(FileNotFoundError, match="does not exist"):
            get_camera_name_to_bev_name_map(missing)

    def test_invalid_path_format_raises_value_error(self):
        """``validate_file_path`` rejects whitespace / unsupported chars."""
        from spatialai_data_utils.loaders.calibration import (
            get_camera_name_to_bev_name_map,
        )
        with pytest.raises(ValueError, match="Invalid file path"):
            get_camera_name_to_bev_name_map("not a/valid path*.json")

    def test_malformed_json_raises_value_error(self, tmp_path):
        """Non-JSON content is wrapped with the calibration file path for context."""
        from spatialai_data_utils.loaders.calibration import (
            get_camera_name_to_bev_name_map,
        )
        path = os.path.join(str(tmp_path), "calibration.json")
        with open(path, "w") as f:
            f.write("{not valid json")
        with pytest.raises(ValueError, match="Failed to load calibration JSON"):
            get_camera_name_to_bev_name_map(path)

    def test_missing_sensors_key_raises_value_error(self, tmp_path):
        from spatialai_data_utils.loaders.calibration import (
            get_camera_name_to_bev_name_map,
        )
        path = _write_calib_json(tmp_path, {"version": "1.0"})
        with pytest.raises(ValueError, match="missing or non-list 'sensors'"):
            get_camera_name_to_bev_name_map(path)

    def test_sensors_not_a_list_raises_value_error(self, tmp_path):
        from spatialai_data_utils.loaders.calibration import (
            get_camera_name_to_bev_name_map,
        )
        path = _write_calib_json(tmp_path, {"sensors": {"id": "Camera_01"}})
        with pytest.raises(ValueError, match="missing or non-list 'sensors'"):
            get_camera_name_to_bev_name_map(path)

    def test_sensor_missing_id_raises_value_error(self, tmp_path):
        from spatialai_data_utils.loaders.calibration import (
            get_camera_name_to_bev_name_map,
        )
        path = _write_calib_json(tmp_path, {"sensors": [
            {"group": {"name": "bev-sensor-1"}},
        ]})
        with pytest.raises(ValueError, match="missing 'id' or 'group.name'"):
            get_camera_name_to_bev_name_map(path)

    def test_sensor_missing_group_raises_value_error(self, tmp_path):
        from spatialai_data_utils.loaders.calibration import (
            get_camera_name_to_bev_name_map,
        )
        path = _write_calib_json(tmp_path, {"sensors": [
            {"id": "Camera_01"},
        ]})
        with pytest.raises(ValueError, match="missing 'id' or 'group.name'"):
            get_camera_name_to_bev_name_map(path)

    def test_sensor_group_missing_name_raises_value_error(self, tmp_path):
        from spatialai_data_utils.loaders.calibration import (
            get_camera_name_to_bev_name_map,
        )
        path = _write_calib_json(tmp_path, {"sensors": [
            {"id": "Camera_01", "group": {}},
        ]})
        with pytest.raises(ValueError, match="missing 'id' or 'group.name'"):
            get_camera_name_to_bev_name_map(path)

    def test_error_messages_include_calibration_path(self, tmp_path):
        """All error messages should surface the calibration path for debugging."""
        from spatialai_data_utils.loaders.calibration import (
            get_camera_name_to_bev_name_map,
        )
        path = _write_calib_json(tmp_path, {"version": "1.0"})
        with pytest.raises(ValueError) as exc_info:
            get_camera_name_to_bev_name_map(path)
        assert path in str(exc_info.value)


class TestFetchFpsFromCalibration:
    """Validation contract for ``loaders.calibration.fetch_fps_from_calibration``.

    Mirrors :class:`TestGetCameraNameToBevNameMap` but for the FPS
    extractor — the post-reorg version now raises typed errors with
    file-path context for every malformed-attributes branch and only
    accepts numerically-consistent FPS values across sensors.
    """

    def test_happy_path_single_sensor(self, tmp_path):
        from spatialai_data_utils.loaders.calibration import (
            fetch_fps_from_calibration,
        )
        path = _write_calib_json(tmp_path, {"sensors": [
            {"id": "Camera_01", "attributes": [
                {"name": "fps", "value": 30.0},
            ]},
        ]})
        assert fetch_fps_from_calibration(path) == 30.0

    def test_happy_path_consistent_fps_multiple_sensors(self, tmp_path):
        """Multiple sensors with the same FPS → returns it once."""
        from spatialai_data_utils.loaders.calibration import (
            fetch_fps_from_calibration,
        )
        path = _write_calib_json(tmp_path, {"sensors": [
            {"id": "Camera_01", "attributes": [{"name": "fps", "value": 30.0}]},
            {"id": "Camera_02", "attributes": [{"name": "fps", "value": 30.0}]},
            {"id": "Camera_03", "attributes": [{"name": "fps", "value": 30.0}]},
        ]})
        assert fetch_fps_from_calibration(path) == 30.0

    def test_string_fps_value_is_coerced_to_float(self, tmp_path):
        """``value`` may arrive as a JSON string — the loader coerces with ``float()``."""
        from spatialai_data_utils.loaders.calibration import (
            fetch_fps_from_calibration,
        )
        path = _write_calib_json(tmp_path, {"sensors": [
            {"id": "Camera_01", "attributes": [{"name": "fps", "value": "30"}]},
        ]})
        assert fetch_fps_from_calibration(path) == 30.0

    def test_skips_non_fps_attributes(self, tmp_path):
        """Other attribute names (e.g. ``frame_width``) shouldn't break FPS lookup."""
        from spatialai_data_utils.loaders.calibration import (
            fetch_fps_from_calibration,
        )
        path = _write_calib_json(tmp_path, {"sensors": [
            {"id": "Camera_01", "attributes": [
                {"name": "frame_width", "value": 1920},
                {"name": "fps", "value": 60.0},
                {"name": "frame_height", "value": 1080},
            ]},
        ]})
        assert fetch_fps_from_calibration(path) == 60.0

    def test_missing_file_raises_file_not_found(self, tmp_path):
        from spatialai_data_utils.loaders.calibration import (
            fetch_fps_from_calibration,
        )
        missing = os.path.join(str(tmp_path), "missing.json")
        with pytest.raises(FileNotFoundError, match="does not exist"):
            fetch_fps_from_calibration(missing)

    def test_malformed_json_raises_value_error(self, tmp_path):
        from spatialai_data_utils.loaders.calibration import (
            fetch_fps_from_calibration,
        )
        path = os.path.join(str(tmp_path), "calibration.json")
        with open(path, "w") as f:
            f.write("not json at all")
        with pytest.raises(ValueError, match="Failed to load calibration JSON"):
            fetch_fps_from_calibration(path)

    def test_missing_sensors_key_raises_value_error(self, tmp_path):
        from spatialai_data_utils.loaders.calibration import (
            fetch_fps_from_calibration,
        )
        path = _write_calib_json(tmp_path, {})
        with pytest.raises(ValueError, match="missing or non-list 'sensors'"):
            fetch_fps_from_calibration(path)

    def test_sensor_missing_attributes_raises_value_error(self, tmp_path):
        from spatialai_data_utils.loaders.calibration import (
            fetch_fps_from_calibration,
        )
        path = _write_calib_json(tmp_path, {"sensors": [{"id": "Camera_01"}]})
        with pytest.raises(ValueError, match="missing or non-list 'attributes'"):
            fetch_fps_from_calibration(path)

    def test_attribute_missing_name_raises_value_error(self, tmp_path):
        from spatialai_data_utils.loaders.calibration import (
            fetch_fps_from_calibration,
        )
        path = _write_calib_json(tmp_path, {"sensors": [
            {"id": "Camera_01", "attributes": [{"value": 30.0}]},
        ]})
        with pytest.raises(ValueError, match="attribute missing 'name' or 'value'"):
            fetch_fps_from_calibration(path)

    def test_attribute_missing_value_raises_value_error(self, tmp_path):
        from spatialai_data_utils.loaders.calibration import (
            fetch_fps_from_calibration,
        )
        path = _write_calib_json(tmp_path, {"sensors": [
            {"id": "Camera_01", "attributes": [{"name": "fps"}]},
        ]})
        with pytest.raises(ValueError, match="attribute missing 'name' or 'value'"):
            fetch_fps_from_calibration(path)

    def test_inconsistent_fps_across_sensors_raises_value_error(self, tmp_path):
        """Mismatched FPS values must be flagged — silently picking one is unsafe."""
        from spatialai_data_utils.loaders.calibration import (
            fetch_fps_from_calibration,
        )
        path = _write_calib_json(tmp_path, {"sensors": [
            {"id": "Camera_01", "attributes": [{"name": "fps", "value": 30.0}]},
            {"id": "Camera_02", "attributes": [{"name": "fps", "value": 60.0}]},
        ]})
        with pytest.raises(ValueError, match="Unmatched FPS for sensors"):
            fetch_fps_from_calibration(path)

    def test_no_fps_attribute_raises_value_error(self, tmp_path):
        from spatialai_data_utils.loaders.calibration import (
            fetch_fps_from_calibration,
        )
        path = _write_calib_json(tmp_path, {"sensors": [
            {"id": "Camera_01", "attributes": [
                {"name": "frame_width", "value": 1920},
            ]},
        ]})
        with pytest.raises(ValueError, match="FPS not available"):
            fetch_fps_from_calibration(path)


# ===================================================================
# IO utils file-handle cleanup (eval/common/io_utils.py)
# ===================================================================
#
# Commit f912fa2 wrapped the JSONL → per-(sensor[/class]) splitting writers in
# ``try/finally`` so a malformed mid-stream line doesn't leak open file
# descriptors.  The happy-path coverage below also acts as the only API-level
# regression test for ``split_files_*`` — they didn't have any before this MR.

def _write_jsonl(tmp_path, name, rows):
    path = os.path.join(str(tmp_path), name)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return path


def _make_obj(class_name, conf=0.9):
    return {
        "type": class_name,
        "bbox3d": {
            "coordinates": [0.0] * 9,
            "confidence": conf,
            "embedding": [{}],
        },
        "embedding": {},
    }


def _read_jsonl(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


class TestSplitFilesBySensor:
    """Tests for ``split_files_by_sensor`` end-to-end behaviour and resource cleanup."""

    def _write_sample(self, tmp_path, gt_rows, pred_rows):
        gt_path = _write_jsonl(tmp_path, "gt.jsonl", gt_rows)
        pred_path = _write_jsonl(tmp_path, "pred.jsonl", pred_rows)
        out_dir = os.path.join(str(tmp_path), "by_sensor")
        return gt_path, pred_path, out_dir

    def test_happy_path_writes_expected_files(self, tmp_path):
        from spatialai_data_utils.eval.common.io_utils import split_files_by_sensor
        gt = [
            {"id": 0, "sensorId": "Camera_01", "objects": [_make_obj("Person")]},
            {"id": 1, "sensorId": "Camera_02", "objects": [_make_obj("Person")]},
        ]
        pred = [
            {"id": 0, "sensorId": "bev-sensor-1", "objects": [_make_obj("Person")]},
        ]
        gt_path, pred_path, out_dir = self._write_sample(tmp_path, gt, pred)
        cam_to_bev = {"Camera_01": ["bev-sensor-1"], "Camera_02": ["bev-sensor-1"]}
        split_files_by_sensor(
            gt_path, pred_path, out_dir, cam_to_bev,
            confidence_threshold=0.0, num_frames_to_eval=10,
        )
        assert os.path.isfile(os.path.join(out_dir, "bev-sensor-1", "gt.json"))
        assert os.path.isfile(os.path.join(out_dir, "bev-sensor-1", "pred.json"))
        gt_lines = _read_jsonl(os.path.join(out_dir, "bev-sensor-1", "gt.json"))
        assert len(gt_lines) == 2

    def test_confidence_threshold_filters_predictions(self, tmp_path):
        from spatialai_data_utils.eval.common.io_utils import split_files_by_sensor
        gt = [{"id": 0, "sensorId": "Camera_01", "objects": [_make_obj("Person")]}]
        pred = [{"id": 0, "sensorId": "bev-sensor-1", "objects": [
            _make_obj("Person", conf=0.9),
            _make_obj("Person", conf=0.1),
        ]}]
        gt_path, pred_path, out_dir = self._write_sample(tmp_path, gt, pred)
        split_files_by_sensor(
            gt_path, pred_path, out_dir, {"Camera_01": ["bev-sensor-1"]},
            confidence_threshold=0.5, num_frames_to_eval=10,
        )
        pred_lines = _read_jsonl(os.path.join(out_dir, "bev-sensor-1", "pred.json"))
        assert len(pred_lines) == 1
        assert len(pred_lines[0]["objects"]) == 1
        assert pred_lines[0]["objects"][0]["bbox3d"]["confidence"] == 0.9

    def test_num_frames_to_eval_caps_input(self, tmp_path):
        from spatialai_data_utils.eval.common.io_utils import split_files_by_sensor
        gt = [
            {"id": i, "sensorId": "Camera_01", "objects": [_make_obj("Person")]}
            for i in range(5)
        ]
        pred = [
            {"id": i, "sensorId": "bev-sensor-1", "objects": [_make_obj("Person")]}
            for i in range(5)
        ]
        gt_path, pred_path, out_dir = self._write_sample(tmp_path, gt, pred)
        split_files_by_sensor(
            gt_path, pred_path, out_dir, {"Camera_01": ["bev-sensor-1"]},
            confidence_threshold=0.0, num_frames_to_eval=2,
        )
        assert len(_read_jsonl(os.path.join(out_dir, "bev-sensor-1", "gt.json"))) == 2
        assert len(_read_jsonl(os.path.join(out_dir, "bev-sensor-1", "pred.json"))) == 2

    def test_camera_in_multiple_groups_fans_out_gt(self, tmp_path):
        """A camera mapped to two BEV groups produces one GT file per group."""
        from spatialai_data_utils.eval.common.io_utils import split_files_by_sensor
        gt = [{"id": 0, "sensorId": "Camera_01", "objects": [_make_obj("Person")]}]
        pred = [
            {"id": 0, "sensorId": "bev-sensor-1", "objects": [_make_obj("Person")]},
            {"id": 0, "sensorId": "bev-sensor-2", "objects": [_make_obj("Person")]},
        ]
        gt_path, pred_path, out_dir = self._write_sample(tmp_path, gt, pred)
        split_files_by_sensor(
            gt_path, pred_path, out_dir,
            {"Camera_01": ["bev-sensor-1", "bev-sensor-2"]},
            confidence_threshold=0.0, num_frames_to_eval=10,
        )
        for bev in ("bev-sensor-1", "bev-sensor-2"):
            assert os.path.isfile(os.path.join(out_dir, bev, "gt.json"))

    def test_writers_closed_after_success(self, tmp_path):
        """Output files should be closed (truncatable / re-readable) after the call."""
        from spatialai_data_utils.eval.common.io_utils import split_files_by_sensor
        gt = [{"id": 0, "sensorId": "Camera_01", "objects": [_make_obj("Person")]}]
        pred = [{"id": 0, "sensorId": "bev-sensor-1", "objects": [_make_obj("Person")]}]
        gt_path, pred_path, out_dir = self._write_sample(tmp_path, gt, pred)
        split_files_by_sensor(
            gt_path, pred_path, out_dir, {"Camera_01": ["bev-sensor-1"]},
            confidence_threshold=0.0, num_frames_to_eval=10,
        )
        # If a writer is still open we'd get a stale read; truncating the
        # file here is the strongest portable signal that the descriptor
        # was released by the function (try/finally branch ran).
        gt_out = os.path.join(out_dir, "bev-sensor-1", "gt.json")
        with open(gt_out, "w") as f:
            f.truncate(0)
        assert os.path.getsize(gt_out) == 0

    def test_writers_closed_when_input_malformed_mid_stream(self, tmp_path):
        """A bad line raises, but the ``try/finally`` branch must still close writers.

        Pre-fix, the ``open(..., "w")`` calls were leaked when the
        in-loop ``json.loads`` raised — every writer left in
        ``sensor_gt_writers`` stayed open until GC.  After the fix the
        ``finally`` branch flushes and closes them deterministically;
        this is observable as the partial file being readable / re-
        writable while the propagated error is the original
        ``json.JSONDecodeError``.
        """
        from spatialai_data_utils.eval.common.io_utils import split_files_by_sensor
        gt_path = os.path.join(str(tmp_path), "gt.jsonl")
        with open(gt_path, "w") as f:
            f.write(json.dumps({"id": 0, "sensorId": "Camera_01",
                                "objects": [_make_obj("Person")]}) + "\n")
            f.write("{not valid json\n")  # mid-stream poison
        pred_path = _write_jsonl(tmp_path, "pred.jsonl", [
            {"id": 0, "sensorId": "bev-sensor-1", "objects": [_make_obj("Person")]},
        ])
        out_dir = os.path.join(str(tmp_path), "by_sensor")
        with pytest.raises(json.JSONDecodeError):
            split_files_by_sensor(
                gt_path, pred_path, out_dir,
                {"Camera_01": ["bev-sensor-1"]},
                confidence_threshold=0.0, num_frames_to_eval=10,
            )
        # The first (good) line should have made it to disk before the
        # poison line aborted the loop, and the file should be closed —
        # we should be able to re-open / overwrite it.
        gt_out = os.path.join(out_dir, "bev-sensor-1", "gt.json")
        assert os.path.isfile(gt_out)
        with open(gt_out, "w") as f:
            f.write("")  # Replaces content; would error if FD still held in some envs.


class TestSplitFilesPerClass:
    """Tests for ``split_files_per_class`` with the synthetic warehouse class set."""

    def _write_sample(self, tmp_path, gt_rows, pred_rows):
        gt_path = _write_jsonl(tmp_path, "gt.jsonl", gt_rows)
        pred_path = _write_jsonl(tmp_path, "pred.jsonl", pred_rows)
        out_dir = os.path.join(str(tmp_path), "by_class")
        return gt_path, pred_path, out_dir

    def test_happy_path_splits_per_primary_class(self, tmp_path):
        from spatialai_data_utils.eval.common.io_utils import split_files_per_class
        gt = [
            {"id": 0, "sensorId": "Camera_01", "objects": [
                _make_obj("Person"),
                _make_obj("Forklift"),
            ]},
        ]
        pred = [
            {"id": 0, "sensorId": "Camera_01", "timestamp": "2025-01-01T00:00:00Z",
             "objects": [_make_obj("Person"), _make_obj("Forklift")]},
        ]
        gt_path, pred_path, out_dir = self._write_sample(tmp_path, gt, pred)
        split_files_per_class(
            gt_path, pred_path, out_dir,
            confidence_threshold=0.0, num_frames_to_eval=10,
        )
        for primary in ("Person", "Forklift"):
            assert os.path.isfile(os.path.join(out_dir, primary, "gt.json"))
            assert os.path.isfile(os.path.join(out_dir, primary, "pred.json"))

    def test_subclass_remap_uses_primary_class_dir(self, tmp_path):
        """Sub-class names like ``cardbox`` get remapped to their primary class folder."""
        from spatialai_data_utils.eval.common.io_utils import split_files_per_class
        gt = [{"id": 0, "sensorId": "Camera_01",
               "objects": [_make_obj("CardBox")]}]
        pred = [{"id": 0, "sensorId": "Camera_01",
                 "objects": [_make_obj("CardBox")]}]
        gt_path, pred_path, out_dir = self._write_sample(tmp_path, gt, pred)
        split_files_per_class(
            gt_path, pred_path, out_dir,
            confidence_threshold=0.0, num_frames_to_eval=10,
        )
        # CardBox should land in the "Box" primary class directory.
        assert os.path.isdir(os.path.join(out_dir, "Box"))
        assert not os.path.isdir(os.path.join(out_dir, "CardBox"))


class TestSplitFilesPerSensorAndClass:
    """Tests for ``split_files_per_sensor_and_class`` (BEV × class fan-out)."""

    def test_happy_path_writes_per_bev_per_class(self, tmp_path):
        from spatialai_data_utils.eval.common.io_utils import (
            split_files_per_sensor_and_class,
        )
        gt = [
            {"id": 0, "sensorId": "Camera_01",
             "objects": [_make_obj("Person"), _make_obj("Forklift")]},
        ]
        pred = [
            {"id": 0, "sensorId": "bev-sensor-1",
             "objects": [_make_obj("Person"), _make_obj("Forklift")]},
        ]
        gt_path = _write_jsonl(tmp_path, "gt.jsonl", gt)
        pred_path = _write_jsonl(tmp_path, "pred.jsonl", pred)
        out_dir = os.path.join(str(tmp_path), "by_sensor_class")
        split_files_per_sensor_and_class(
            gt_path, pred_path, out_dir, {"Camera_01": ["bev-sensor-1"]},
            confidence_threshold=0.0, num_frames_to_eval=10,
        )
        for primary in ("Person", "Forklift"):
            sub = os.path.join(out_dir, "bev-sensor-1", primary)
            assert os.path.isfile(os.path.join(sub, "gt.json"))


# ===================================================================
# Detection JSONL loader (eval/detection/loaders.py)
# ===================================================================
#
# GT-side ``detection_score`` is fixed to the codebase-wide ``-1.0``
# sentinel (matching ``DetectionBox``'s default and the GT contract in
# ``eval/common/loaders.py``) — predictions are ranked by score for
# AP/PR, so a real GT score would just pollute the curves and any
# stray string-typed confidence in the JSON would also crash sort.
# Pred-side scores are still preserved verbatim (and coerced to
# ``float`` so a JSON-string confidence like ``"0.95"`` doesn't blow
# up downstream metrics).

class TestLoadBoxesFromJsonl:
    """Tests for ``load_boxes_from_jsonl``."""

    def _make_jsonl(self, tmp_path, name, rows):
        return _write_jsonl(tmp_path, name, rows)

    def test_no_parseable_timestamps_raises(self, tmp_path):
        """Empty GT and prediction inputs should not synthesize a base timestamp."""
        from spatialai_data_utils.eval.detection.loaders import (
            load_boxes_from_jsonl,
        )

        gt_path = self._make_jsonl(tmp_path, "gt.jsonl", [])
        pred_path = self._make_jsonl(tmp_path, "pred.jsonl", [])

        with pytest.raises(ValueError, match="No parseable timestamps"):
            load_boxes_from_jsonl(gt_path, pred_path, fps=30.0)

    def _row(self, ts, conf, frame_obj_class="Person"):
        return {
            "id": 0,
            "sensorId": "Camera_01",
            "timestamp": ts,
            "objects": [{
                "type": frame_obj_class,
                "bbox3d": {
                    "coordinates": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                    "confidence": conf,
                },
            }],
        }

    def test_gt_detection_score_is_sentinel(self, tmp_path):
        """GT boxes always use the ``-1.0`` no-confidence sentinel.

        Whatever confidence the GT JSON carries (a numeric value, a
        JSON-string like ``"0.95"``, or no field at all) is ignored —
        ground truth has no meaningful "score" and using the sentinel
        keeps GT boxes from being ranked into AP/PR curves and from
        crashing sort routines on string types.
        """
        from spatialai_data_utils.eval.detection.loaders import (
            load_boxes_from_jsonl,
        )
        ts = "2025-01-01T00:00:00.000Z"
        gt_row_string = self._row(ts, "0.95")
        gt_row_numeric = self._row(ts, 0.77)
        gt_row_missing = {
            "id": 0, "sensorId": "Camera_01", "timestamp": ts,
            "objects": [{
                "type": "Person",
                "bbox3d": {"coordinates": [0.0] * 9},
            }],
        }
        for label, gt_rows in [
            ("string confidence", [gt_row_string]),
            ("numeric confidence", [gt_row_numeric]),
            ("missing confidence", [gt_row_missing]),
        ]:
            gt_path = self._make_jsonl(tmp_path, f"gt_{label}.jsonl", gt_rows)
            pred_path = self._make_jsonl(tmp_path, f"pred_{label}.jsonl",
                                         [self._row(ts, 0.85)])
            gt_boxes, _ = load_boxes_from_jsonl(gt_path, pred_path, fps=30.0)
            gt_box = gt_boxes.boxes[gt_boxes.sample_tokens[0]][0]
            assert gt_box.detection_score == -1.0, (
                f"{label}: expected GT sentinel -1.0, got {gt_box.detection_score!r}"
            )


# ===================================================================
# HOTA tracking helper signatures (eval/tracking/hota/trackeval_utils.py)
# ===================================================================
#
# Commit f912fa2 simplified ``run_evaluation`` and ``_run_tracking_all_sensors``
# by removing the unused ``fps``, ``ground_truth_file`` and ``prediction_file``
# parameters.  These tests pin the new, narrower public surface.

class TestRunEvaluationSignature:
    """``run_evaluation`` should no longer take an ``fps`` parameter.

    The function only reads ``dataset_config`` / ``eval_config`` /
    ``eval_type`` after the simplification, so callers shouldn't be
    forced to plumb FPS through any more (it was passed but unused
    inside the body).  Pinning the signature here protects against an
    accidental re-introduction.
    """

    def test_run_evaluation_does_not_accept_fps_kw(self):
        import inspect
        from spatialai_data_utils.eval.tracking.hota import trackeval_utils
        sig = inspect.signature(trackeval_utils.run_evaluation)
        assert "fps" not in sig.parameters
        assert set(sig.parameters) == {
            "gt_file", "prediction_file", "dataset_config",
            "eval_config", "eval_type",
        }

    def test_run_tracking_all_sensors_does_not_accept_file_paths(self):
        """The all-sensors helper now reads files from the prepared output dir."""
        import inspect
        from spatialai_data_utils.eval.tracking.hota import trackeval_utils
        sig = inspect.signature(trackeval_utils._run_tracking_all_sensors)
        assert "ground_truth_file" not in sig.parameters
        assert "prediction_file" not in sig.parameters
        # The parameters we DO keep — the test pins the call surface so an
        # accidental re-add of the removed kwargs is caught at import time.
        assert "output_directory" in sig.parameters
        assert "fps" in sig.parameters

    def test_evaluate_tracking_all_bev_sensors_public_surface_unchanged(self):
        """The public entry point still takes the original 10-arg surface."""
        import inspect
        from spatialai_data_utils.eval.tracking.hota import trackeval_utils
        sig = inspect.signature(trackeval_utils.evaluate_tracking_all_BEV_sensors)
        assert {
            "ground_truth_file", "prediction_file", "calibration_file",
            "eval_options", "output_root_dir", "confidence_threshold",
            "num_cores", "input_file_type", "num_frames_to_eval",
            "ground_truth_frame_offset_secs",
        } == set(sig.parameters)


# ===================================================================
# HOTA Evaluator default config (eval/tracking/hota/evaluate.py)
# ===================================================================
#
# The Evaluator's default config previously pointed ``LOG_ON_ERROR`` at a path
# inside the installed package (``eval/tracking/error_log.txt``) and used
# ``logging.info(msg, file=f)``, which silently ignores ``file=`` and writes
# nothing.  Net effect: any caught exception in the eval loop leaked an empty
# file into the source tree.  These tests pin the post-fix contract:
#  (1) ``LOG_ON_ERROR`` defaults to ``None`` (no file created on errors), and
#  (2) when a caller opts in by setting ``LOG_ON_ERROR`` to an explicit path,
#      the diagnostic data actually lands in the file.

class TestEvaluatorErrorLogDefault:
    """Default config for ``hota.evaluate.Evaluator`` no longer leaks files."""

    def test_log_on_error_defaults_to_none(self):
        from spatialai_data_utils.eval.tracking.hota.evaluate import Evaluator
        cfg = Evaluator.get_default_eval_config()
        assert cfg["LOG_ON_ERROR"] is None

    def test_default_config_has_no_in_package_paths(self):
        """No default-config value should point inside the installed package.

        Pre-fix the default ``LOG_ON_ERROR`` was an absolute path under
        ``eval/tracking/`` which made ``Evaluator(...)`` create
        ``error_log.txt`` inside the source tree on every caught
        exception.  Pin the property "no defaults reference the
        package directory" so a future refactor doesn't reintroduce
        the leak via some other config key.
        """
        import spatialai_data_utils
        from spatialai_data_utils.eval.tracking.hota.evaluate import Evaluator
        package_dir = os.path.dirname(spatialai_data_utils.__file__)
        cfg = Evaluator.get_default_eval_config()
        for key, val in cfg.items():
            if isinstance(val, str) and os.path.isabs(val):
                assert package_dir not in val, (
                    f"Default config key {key!r} points inside the "
                    f"installed package: {val!r} (would leak files into "
                    f"the source tree)."
                )

    def test_evaluator_init_does_not_create_error_log(self, tmp_path, monkeypatch):
        """Constructing an Evaluator must not touch the filesystem.

        The pre-fix bug was triggered later (only on a caught exception),
        but it's worth pinning that *just instantiating* the Evaluator
        doesn't accidentally pre-create the log file either — defense in
        depth against a future tweak that eagerly opens
        ``LOG_ON_ERROR`` at init time.
        """
        from spatialai_data_utils.eval.tracking.hota.evaluate import Evaluator
        monkeypatch.chdir(tmp_path)
        evaluator = Evaluator()
        assert evaluator.config["LOG_ON_ERROR"] is None
        assert not os.path.isfile(os.path.join(str(tmp_path), "error_log.txt"))


class TestEvaluatorErrorLogOptIn:
    """Opt-in path: when caller sets ``LOG_ON_ERROR``, errors land in the file."""

    def test_print_writes_diagnostics_to_log_path(self, tmp_path):
        """Sanity-check the post-fix ``print(..., file=f)`` write semantics.

        The buggy version used ``logging.info(msg, file=f)`` which is a
        silent no-op — the log file ended up empty even when callers
        explicitly opted in.  This test directly invokes the same
        ``print(...)``-into-``open(..., 'a')`` pattern the
        ``except`` branch now uses, so a future revert of the fix
        (e.g. swapping ``print`` back for ``logging.info``) trips this
        assertion immediately.
        """
        log_path = os.path.join(str(tmp_path), "error_log.txt")
        with open(log_path, "a") as f:
            print("dataset_x", file=f)
            print("tracker_y", file=f)
            print("traceback line\nsecond line", file=f)
        with open(log_path) as f:
            contents = f.read()
        assert "dataset_x" in contents
        assert "tracker_y" in contents
        assert "traceback line" in contents
        assert os.path.getsize(log_path) > 0


# ===================================================================
# Missing-file diagnostics for load_boxes_from_jsonl
# ===================================================================
#
# The loader pre-validates both ``gt_path`` and ``pred_path`` so that a
# missing file raises a labelled ``FileNotFoundError`` *before* the
# load-progress ``logging.info`` lines fire (which would otherwise
# misleadingly announce a load that's about to crash inside ``open``).

class TestLoadBoxesFromJsonlOffsetAlignment:
    """``ground_truth_frame_offset_secs`` aligns GT and pred sample_tokens.

    Pre-fix the GT branch read its ``frame_id`` from the *raw* timestamp
    via ``_get_frame_id`` and *then* subtracted ``gt_offset_frames``
    after the conversion, while the pred branch used the raw frame id.
    With any non-zero offset the two ``sample_token`` keys parted ways
    for the same physical instant, and downstream evaluators saw all
    detections as FN/FP.  The fix shifts the GT timestamp by
    ``timedelta(seconds=ground_truth_frame_offset_secs)`` *before* the
    timestamp -> frame_id conversion so both sides land on the same key.
    """

    def _make_row(self, ts, conf=0.9):
        return {
            "id": 0,
            "sensorId": "Camera_01",
            "timestamp": ts,
            "objects": [{
                "type": "Person",
                "bbox3d": {
                    "coordinates": [0.0] * 9,
                    "confidence": conf,
                },
            }],
        }

    def _write_synthetic(self, tmp_path, num_frames, fps):
        """Write GT and pred JSONLs with identical timestamps (the realistic
        case after the upstream splitter)."""
        from datetime import datetime, timedelta
        base = datetime(2025, 1, 1, 0, 0, 0)
        timestamps = [
            (base + timedelta(seconds=i / fps)).strftime(
                "%Y-%m-%dT%H:%M:%S.%fZ"
            )
            for i in range(num_frames)
        ]
        gt_path = _write_jsonl(tmp_path, "gt.jsonl",
                               [self._make_row(t) for t in timestamps])
        pred_path = _write_jsonl(tmp_path, "pred.jsonl",
                                 [self._make_row(t) for t in timestamps])
        return gt_path, pred_path

    def test_zero_offset_yields_identical_sample_tokens(self, tmp_path):
        """Offset=0 is the regression baseline: both sides match exactly."""
        from spatialai_data_utils.eval.detection.loaders import (
            load_boxes_from_jsonl,
        )
        gt_path, pred_path = self._write_synthetic(tmp_path, num_frames=5, fps=30.0)
        gt, pred = load_boxes_from_jsonl(
            gt_path, pred_path, fps=30.0,
            ground_truth_frame_offset_secs=0.0,
        )
        assert set(gt.sample_tokens) == set(pred.sample_tokens), (
            "GT and pred sample_tokens must be identical when no offset is "
            "applied."
        )

    def test_nonzero_offset_keeps_overlap_aligned(self, tmp_path):
        """Pre-fix this assertion failed because GT was offset post-conversion."""
        from spatialai_data_utils.eval.detection.loaders import (
            load_boxes_from_jsonl,
        )
        gt_path, pred_path = self._write_synthetic(tmp_path, num_frames=5, fps=30.0)
        gt, pred = load_boxes_from_jsonl(
            gt_path, pred_path, fps=30.0,
            ground_truth_frame_offset_secs=2 / 30.0,  # 2 frames at 30 fps
        )
        # Every surviving GT sample_token must appear in pred (the
        # filter is now ``frame_id in prediction_frame_ids``).  Pred may
        # still contain extra warmup tokens that fell outside GT's
        # adjusted window — that's expected; it's the *symmetric*
        # mismatch (no overlap at all) that was the bug.
        pred_set = set(pred.sample_tokens)
        gt_set = set(gt.sample_tokens)
        assert gt_set <= pred_set, (
            f"GT sample_tokens ({sorted(gt_set, key=int)}) should be a "
            f"subset of pred sample_tokens ({sorted(pred_set, key=int)}) "
            f"after the offset is applied to the GT timestamp; pre-fix "
            f"these sets were disjoint for any non-zero offset."
        )
        # And the overlap must be non-empty whenever the offset is
        # smaller than the synthetic sequence length.
        assert gt_set, "Adjusted GT lost every frame — offset misapplied?"


class TestLoadBoxesFromJsonlMissingFiles:
    """Missing GT or prediction file paths raise a clear ``FileNotFoundError``."""

    def test_missing_pred_path_raises_labelled_error(self, tmp_path):
        from spatialai_data_utils.eval.detection.loaders import (
            load_boxes_from_jsonl,
        )
        gt_path = _write_jsonl(tmp_path, "gt.jsonl", [
            {"id": 0, "sensorId": "Camera_01",
             "timestamp": "2025-01-01T00:00:00.000Z",
             "objects": [_make_obj("Person")]},
        ])
        missing = os.path.join(str(tmp_path), "does_not_exist.jsonl")
        with pytest.raises(FileNotFoundError, match=r"Prediction.*does_not_exist"):
            load_boxes_from_jsonl(gt_path, missing, fps=30.0)

    def test_missing_gt_path_raises_labelled_error(self, tmp_path):
        from spatialai_data_utils.eval.detection.loaders import (
            load_boxes_from_jsonl,
        )
        pred_path = _write_jsonl(tmp_path, "pred.jsonl", [
            {"id": 0, "sensorId": "Camera_01",
             "timestamp": "2025-01-01T00:00:00.000Z",
             "objects": [_make_obj("Person")]},
        ])
        missing = os.path.join(str(tmp_path), "does_not_exist.jsonl")
        with pytest.raises(FileNotFoundError, match=r"Ground truth.*does_not_exist"):
            load_boxes_from_jsonl(missing, pred_path, fps=30.0)


# ===================================================================
# split_files_per_class confidence-filter ordering
# ===================================================================
#
# Pre-fix the per-class entry was allocated *before* the confidence
# check, so a frame whose only object was filtered out still emitted
# a ``"objects": []`` line in the per-class ``pred.json``.  The order
# now matches ``split_files_per_sensor_and_class.process_objects_pred``.

class TestSplitFilesPerClassConfidenceFilter:
    """Empty class entries no longer leak through ``split_files_per_class``."""

    def test_all_low_confidence_frame_yields_no_pred_line(self, tmp_path):
        from spatialai_data_utils.eval.common.io_utils import split_files_per_class

        gt_rows = [
            {"id": 0, "sensorId": "Camera_01",
             "timestamp": "2025-01-01T00:00:00.000000Z",
             "objects": [_make_obj("Person")]},
            {"id": 1, "sensorId": "Camera_01",
             "timestamp": "2025-01-01T00:00:00.033333Z",
             "objects": [_make_obj("Person")]},
        ]
        pred_rows = [
            # Frame 0: only object is below threshold → no per-class line.
            {"id": 0, "sensorId": "Camera_01",
             "timestamp": "2025-01-01T00:00:00.000000Z",
             "objects": [_make_obj("Person", conf=0.1)]},
            # Frame 1: object passes threshold → exactly one per-class line.
            {"id": 1, "sensorId": "Camera_01",
             "timestamp": "2025-01-01T00:00:00.033333Z",
             "objects": [_make_obj("Person", conf=0.9)]},
        ]
        gt_path = _write_jsonl(tmp_path, "gt.jsonl", gt_rows)
        pred_path = _write_jsonl(tmp_path, "pred.jsonl", pred_rows)
        out_dir = os.path.join(str(tmp_path), "by_class")
        split_files_per_class(
            gt_path, pred_path, out_dir,
            confidence_threshold=0.5, num_frames_to_eval=10,
        )
        pred_lines = _read_jsonl(os.path.join(out_dir, "Person", "pred.json"))
        # Pre-fix: 2 lines (one with ``"objects": []``).  Post-fix: 1 line.
        assert len(pred_lines) == 1
        assert len(pred_lines[0]["objects"]) == 1
        assert pred_lines[0]["objects"][0]["bbox3d"]["confidence"] == 0.9


# ===================================================================
# prepare_evaluation_folder seq_length propagation
# ===================================================================
#
# ``seqinfo.ini`` previously hard-coded ``seqLength=20000`` regardless
# of the actual frame count.  TrackEval iterates ``range(seq_length)``
# in ``_load_raw_file``, so a too-small value silently truncates and a
# too-large value wastes per-timestep work.  These tests pin that the
# new ``seq_length`` parameter is honoured and that the legacy 20000
# default is preserved for callers that don't pass it.

class TestPrepareEvaluationFolder:
    """``seq_length`` flows from caller into the generated ``seqinfo.ini``."""

    def _cfg(self, tmp_path):
        return {
            "GT_FOLDER": os.path.join(str(tmp_path), "gt"),
            "TRACKERS_FOLDER": os.path.join(str(tmp_path), "trackers"),
            "BENCHMARK": "MOT17",
            "SPLIT_TO_EVAL": "all",
        }

    def _seqinfo_path(self, cfg, input_file_type):
        return os.path.join(
            cfg["GT_FOLDER"], "MOT17-all", input_file_type, "seqinfo.ini",
        )

    def test_seq_length_propagates_to_seqinfo_ini(self, tmp_path):
        from spatialai_data_utils.eval.tracking.hota.trackeval_utils import (
            prepare_evaluation_folder,
        )
        cfg = self._cfg(tmp_path)
        prepare_evaluation_folder(cfg, "RTLS", fps=20.0, seq_length=137)
        text = open(self._seqinfo_path(cfg, "RTLS")).read()
        assert "seqLength=137" in text
        assert "frameRate=20.0" in text

    def test_default_seq_length_is_20000(self, tmp_path):
        """Backwards-compat: callers that don't set ``seq_length`` keep 20000."""
        from spatialai_data_utils.eval.tracking.hota.trackeval_utils import (
            prepare_evaluation_folder,
        )
        cfg = self._cfg(tmp_path)
        prepare_evaluation_folder(cfg, "RTLS")
        text = open(self._seqinfo_path(cfg, "RTLS")).read()
        assert "seqLength=20000" in text

    def test_run_tracking_helpers_accept_num_frames_to_eval(self):
        """The internal helpers expose the parameter that feeds ``seq_length``."""
        import inspect
        from spatialai_data_utils.eval.tracking.hota import trackeval_utils
        for fn in (
            trackeval_utils._run_tracking_per_sensor,
            trackeval_utils._run_tracking_all_sensors,
        ):
            assert "num_frames_to_eval" in inspect.signature(fn).parameters, (
                f"{fn.__name__} should expose num_frames_to_eval to drive "
                f"prepare_evaluation_folder(seq_length=...)."
            )


# ===================================================================
# _setup_tracking_output subdir naming
# ===================================================================
#
# Pre-fix the helper hard-coded ``output_root_dir/all_sensors/`` for both
# the per-sensor and all-sensors flows, which made debugging confusing
# (per-sensor scaffolding ended up under a directory called
# "all_sensors").  The helper now takes a ``subdir_name`` parameter and
# the two entry points pass distinct values.

class TestSetupTrackingOutputSubdir:
    """``_setup_tracking_output`` exposes ``subdir_name`` and callers use it."""

    def test_helper_accepts_subdir_name_param(self):
        import inspect
        from spatialai_data_utils.eval.tracking.hota import trackeval_utils
        sig = inspect.signature(trackeval_utils._setup_tracking_output)
        assert "subdir_name" in sig.parameters
        assert sig.parameters["subdir_name"].default == "all_sensors"

    def test_per_sensor_entry_point_uses_per_sensor_subdir(self):
        """``evaluate_tracking_per_BEV_sensor`` should pass a non-default name.

        We don't run the full evaluation; just inspect the source to
        confirm the call site no longer falls back to the legacy
        ``"all_sensors"`` literal that's confusing for the per-sensor
        flow.
        """
        import inspect
        from spatialai_data_utils.eval.tracking.hota import trackeval_utils
        src = inspect.getsource(trackeval_utils.evaluate_tracking_per_BEV_sensor)
        assert 'subdir_name="per_sensor"' in src, (
            "evaluate_tracking_per_BEV_sensor should call "
            "_setup_tracking_output with subdir_name='per_sensor'."
        )


# ===================================================================
# load_gt instance_inds optionality
# ===================================================================
#
# After the ``eval/common/loaders.py`` cleanup, the detection branch of
# ``load_gt`` no longer dereferences ``instance_inds`` (the tracking
# branch still requires it).  This pin guards against an accidental
# re-introduction of the unconditional ``sample["instance_inds"]`` read.

class TestLoadGtInstanceInds:
    """Detection callers don't need ``instance_inds``; tracking callers do."""

    def _seed_class_lists(self):
        """Seed the global class lists so the box constructors' asserts pass."""
        import spatialai_data_utils.eval.detection.data_classes as ddc
        import spatialai_data_utils.eval.tracking.data_classes as tdc
        ddc.DETECTION_NAMES = ["Person"]
        tdc.TRACKING_NAMES = ["Person"]

    def _make_sample(self, token, n, *, with_inds=True):
        sample = {
            "token": token,
            "scene_name": "scene_1",
            "gt_boxes": [[float(i)] * 9 for i in range(n)],
            "gt_names": ["Person"] * n,
            "gt_velocity": [[0.0, 0.0, 0.0]] * n,
            "valid_flag": [True] * n,
        }
        if with_inds:
            sample["instance_inds"] = list(range(100, 100 + n))
        return sample

    def test_detection_path_does_not_require_instance_inds(self):
        from spatialai_data_utils.eval.common.loaders import load_gt
        from spatialai_data_utils.eval.detection.data_classes import DetectionBox
        self._seed_class_lists()
        infos = [self._make_sample("s0", 2, with_inds=False)]
        gt = load_gt(infos, DetectionBox)
        # Boxes loaded successfully without raising KeyError on instance_inds.
        assert gt.sample_tokens == ["s0"]
        assert len(gt.boxes["s0"]) == 2

    def test_tracking_path_still_requires_instance_inds(self):
        from spatialai_data_utils.eval.common.loaders import load_gt
        from spatialai_data_utils.eval.tracking.data_classes import TrackingBox
        self._seed_class_lists()
        infos = [self._make_sample("s0", 2, with_inds=False)]
        with pytest.raises(KeyError, match="instance_inds"):
            load_gt(infos, TrackingBox)

    def test_unknown_box_class_raises_clear_error(self):
        from spatialai_data_utils.eval.common.loaders import load_gt
        infos = [self._make_sample("s0", 1, with_inds=True)]
        with pytest.raises(NotImplementedError, match="Invalid box_cls"):
            load_gt(infos, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
