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
Tests for :mod:`spatialai_data_utils.eval.tracking.aicity25_track1_eval`.

Focuses on the pure I/O-shaped layers — the text-row → MOT-row
converter, the spec-validating per-(scene, class) splitter, the
GT-count-weighted aggregator, and the JSON persistence helper — that
can be exercised without standing up TrackEval itself.  The full HOTA
orchestrator (``run_aicity25_track1_evaluation``) is covered by the
existing end-to-end smoke run under ``tools/evaluation/`` rather than
here, to keep this test file fast.

The AICity'25 spec constants (``CLASS_ID_TO_NAME`` / ``NUM_FIELDS``)
themselves live in :mod:`spatialai_data_utils.datasets.aicity25.spec`
and are covered by ``tests/test_datasets.py``.
"""

import json
import os.path as osp

import pytest

from spatialai_data_utils.eval.tracking.aicity25_track1_eval import (
    HOTA_FIELDS,
    _aicity_line_to_mot,
    _weighted_average,
    save_aicity25_track1_results,
    split_aicity25_per_scene_per_class,
)


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


class TestHotaFields:
    """``HOTA_FIELDS`` is the leaderboard-reported metric quartet."""

    def test_exact_four_fields(self):
        assert HOTA_FIELDS == ["HOTA", "DetA", "AssA", "LocA"]


# ---------------------------------------------------------------------------
# _aicity_line_to_mot — pure row-by-row converter
# ---------------------------------------------------------------------------


class TestAICityLineToMot:
    """``_aicity_line_to_mot`` converts AICity'25 rows to TrackEval MOT rows."""

    def test_translates_field_order_and_one_indexes_frame(self):
        """0-indexed frame -> 1-indexed; pitch/roll inserted as zero."""
        parts = "17 4 11 0 -2.6 -7.6 0.83 0.6 0.33 1.65 0.07".split(" ")
        out = _aicity_line_to_mot(parts)
        # frame_id 0 -> 1, object_id 11, confidence 1
        assert out.startswith("1 11 1 ")
        # pitch and roll columns must be exactly 0
        tokens = out.strip().split(" ")
        assert tokens[9] == "0.00000"   # pitch
        assert tokens[10] == "0.00000"  # roll
        # final yaw column carries the input yaw
        assert tokens[11] == "0.07000"

    def test_preserves_dimensions_in_w_l_h_order(self):
        """Width/length/height must stay in their AICity'25 column order."""
        parts = "17 0 1 5 1.0 2.0 3.0 4.5 5.5 6.5 1.57".split(" ")
        tokens = _aicity_line_to_mot(parts).strip().split(" ")
        # tokens: frame obj 1 x y z w l h pitch roll yaw
        assert tokens[3:6] == ["1.00000", "2.00000", "3.00000"]  # x y z
        assert tokens[6:9] == ["4.50000", "5.50000", "6.50000"]  # w l h


# ---------------------------------------------------------------------------
# split_aicity25_per_scene_per_class — streaming spec-validating splitter
# ---------------------------------------------------------------------------


SCENE_MAP = {"17": "Warehouse_017", "18": "Warehouse_018"}


def _write_text(path, lines):
    with open(path, "w") as fp:
        fp.write("\n".join(lines))


class TestSplitAicity25PerScenePerClass:
    """Splitter routes rows to ``<scene>/<class>/`` and counts them."""

    def test_routes_rows_into_scene_class_files(self, tmp_path):
        gt_path = tmp_path / "gt.txt"
        out_root = tmp_path / "split"
        out_root.mkdir()
        _write_text(gt_path, [
            "17 0 1 0 0 0 0 1 1 1 0",   # Warehouse_017 / Person
            "17 0 2 1 0 0 0 1 1 1 0",   # Warehouse_017 / Person
            "17 2 5 0 0 0 0 1 1 1 0",   # Warehouse_017 / NovaCarter
            "18 1 7 0 0 0 0 1 1 1 0",   # Warehouse_018 / Forklift
        ])
        counts = split_aicity25_per_scene_per_class(
            str(gt_path), str(out_root), "gt.txt",
            SCENE_MAP, num_frames_to_eval=9000, is_pred=False,
        )
        assert counts == {
            "Warehouse_017": {"Person": 2, "NovaCarter": 1},
            "Warehouse_018": {"Forklift": 1},
        }
        assert (out_root / "Warehouse_017" / "Person" / "gt.txt").exists()
        assert (out_root / "Warehouse_017" / "NovaCarter" / "gt.txt").exists()
        assert (out_root / "Warehouse_018" / "Forklift" / "gt.txt").exists()

    def test_truncates_by_frame_id_0indexed_exclusive_upper(self, tmp_path):
        """Lines with frame_id >= num_frames_to_eval must be dropped."""
        gt_path = tmp_path / "gt.txt"
        out_root = tmp_path / "split"
        out_root.mkdir()
        _write_text(gt_path, [
            "17 0 1 0 0 0 0 1 1 1 0",    # kept (frame 0)
            "17 0 1 4 0 0 0 1 1 1 0",    # kept (frame 4)
            "17 0 1 5 0 0 0 1 1 1 0",    # dropped (frame 5 == limit)
            "17 0 1 99 0 0 0 1 1 1 0",   # dropped (well past limit)
        ])
        counts = split_aicity25_per_scene_per_class(
            str(gt_path), str(out_root), "gt.txt",
            SCENE_MAP, num_frames_to_eval=5, is_pred=False,
        )
        assert counts == {"Warehouse_017": {"Person": 2}}

    def test_drops_unknown_scene_silently_for_gt(self, tmp_path):
        """Unknown scene IDs in GT are dropped without raising."""
        gt_path = tmp_path / "gt.txt"
        out_root = tmp_path / "split"
        out_root.mkdir()
        _write_text(gt_path, [
            "17 0 1 0 0 0 0 1 1 1 0",  # mapped
            "99 0 1 0 0 0 0 1 1 1 0",  # unmapped — dropped, no error
        ])
        counts = split_aicity25_per_scene_per_class(
            str(gt_path), str(out_root), "gt.txt",
            SCENE_MAP, num_frames_to_eval=9000, is_pred=False,
        )
        assert counts == {"Warehouse_017": {"Person": 1}}

    def test_raises_on_unknown_scene_for_predictions(self, tmp_path):
        """Submissions must declare their scenes — unknown ID is a hard error."""
        pred_path = tmp_path / "pred.txt"
        out_root = tmp_path / "split"
        out_root.mkdir()
        _write_text(pred_path, ["99 0 1 0 0 0 0 1 1 1 0"])
        with pytest.raises(ValueError, match="scene id"):
            split_aicity25_per_scene_per_class(
                str(pred_path), str(out_root), "pred.txt",
                SCENE_MAP, num_frames_to_eval=9000, is_pred=True,
            )

    def test_raises_on_out_of_spec_class_id_for_predictions(self, tmp_path):
        """Submissions with class_id >= 6 are rejected per the spec."""
        pred_path = tmp_path / "pred.txt"
        out_root = tmp_path / "split"
        out_root.mkdir()
        _write_text(pred_path, ["17 6 1 0 0 0 0 1 1 1 0"])
        with pytest.raises(ValueError, match="class id 6"):
            split_aicity25_per_scene_per_class(
                str(pred_path), str(out_root), "pred.txt",
                SCENE_MAP, num_frames_to_eval=9000, is_pred=True,
            )

    def test_raises_on_wrong_field_count_for_predictions(self, tmp_path):
        """Submissions with the wrong number of columns are rejected."""
        pred_path = tmp_path / "pred.txt"
        out_root = tmp_path / "split"
        out_root.mkdir()
        _write_text(pred_path, ["17 0 1 0 0 0 0 1 1 1"])  # 10 fields, not 11
        with pytest.raises(ValueError, match="11"):
            split_aicity25_per_scene_per_class(
                str(pred_path), str(out_root), "pred.txt",
                SCENE_MAP, num_frames_to_eval=9000, is_pred=True,
            )

    def test_raises_on_non_numeric_class_or_frame_id_for_predictions(self, tmp_path):
        """Submissions with non-numeric class/frame ids are rejected."""
        pred_path = tmp_path / "pred.txt"
        out_root = tmp_path / "split"
        out_root.mkdir()
        _write_text(pred_path, ["17 abc 1 0 0 0 0 1 1 1 0"])
        with pytest.raises(ValueError, match="Non-numeric"):
            split_aicity25_per_scene_per_class(
                str(pred_path), str(out_root), "pred.txt",
                SCENE_MAP, num_frames_to_eval=9000, is_pred=True,
            )

    def test_warns_and_skips_on_non_numeric_class_or_frame_id_for_gt(
        self, tmp_path, caplog,
    ):
        """Ground truth with a malformed numeric field warns + skips, doesn't crash.

        Mirrors the field-count handling above: GT may carry stray /
        malformed rows from upstream tools, so the splitter is
        forgiving for GT while strict for submissions.
        """
        import logging
        gt_path = tmp_path / "gt.txt"
        out_root = tmp_path / "split"
        out_root.mkdir()
        _write_text(gt_path, [
            "17 0 1 0 0 0 0 1 1 1 0",       # valid
            "17 abc 2 0 0 0 0 1 1 1 0",     # malformed class_id -> skip
            "17 0 3 NaN 0 0 0 1 1 1 0",     # malformed frame_id -> skip
            "17 0 4 1 0 0 0 1 1 1 0",       # valid
        ])
        with caplog.at_level(logging.WARNING):
            counts = split_aicity25_per_scene_per_class(
                str(gt_path), str(out_root), "gt.txt",
                SCENE_MAP, num_frames_to_eval=9000, is_pred=False,
            )
        assert counts == {"Warehouse_017": {"Person": 2}}
        warning_msgs = [r.message for r in caplog.records]
        assert any("non-numeric" in m.lower() for m in warning_msgs)

    def test_skips_blank_lines(self, tmp_path):
        """Blank / whitespace-only lines do not produce a warning or row."""
        gt_path = tmp_path / "gt.txt"
        out_root = tmp_path / "split"
        out_root.mkdir()
        _write_text(gt_path, [
            "17 0 1 0 0 0 0 1 1 1 0",
            "",
            "   ",
            "17 0 1 1 0 0 0 1 1 1 0",
        ])
        counts = split_aicity25_per_scene_per_class(
            str(gt_path), str(out_root), "gt.txt",
            SCENE_MAP, num_frames_to_eval=9000, is_pred=False,
        )
        assert counts == {"Warehouse_017": {"Person": 2}}


# ---------------------------------------------------------------------------
# _weighted_average — GT-count-weighted aggregator
# ---------------------------------------------------------------------------


class TestWeightedAverage:
    """``_weighted_average`` drops missing keys instead of zeroing them."""

    def test_intersects_keys_and_normalizes(self):
        """Final = sum(w_i * v_i) / sum(w_i) over keys present in both."""
        weights = {"a": 100, "b": 200, "c": 700}
        values = {"a": 0.5, "b": 0.25, "c": 0.1}
        # 100*0.5 + 200*0.25 + 700*0.1 = 170; / 1000 = 0.17
        assert _weighted_average(weights, values) == pytest.approx(0.17)

    def test_missing_value_keys_excluded_not_zeroed(self):
        """A scene missing from *values* must not pull the mean toward zero."""
        weights = {"a": 100, "b": 200}
        values = {"a": 0.5}  # 'b' has weight but no value
        # Only key 'a' participates: 100*0.5 / 100 = 0.5.
        assert _weighted_average(weights, values) == pytest.approx(0.5)

    def test_empty_intersection_returns_zero(self):
        assert _weighted_average({"a": 1}, {"b": 0.5}) == 0.0


# ---------------------------------------------------------------------------
# save_aicity25_track1_results — JSON persistence in 0–100 scale
# ---------------------------------------------------------------------------


class TestSaveAicity25Track1Results:
    """``save_aicity25_track1_results`` writes the 0–100-scaled JSON."""

    def _make_results(self):
        return {
            "eval_type": "bbox",
            "num_frames_to_eval": 9000,
            "scene_id_to_name": {"17": "Warehouse_017"},
            "per_scene_object_counts": {"Warehouse_017": 200},
            "per_scene_per_class": {
                "Warehouse_017": {
                    "Person": {
                        "HOTA": 0.7423, "DetA": 0.7781,
                        "AssA": 0.7082, "LocA": 0.8377,
                    },
                    "Forklift": None,
                },
            },
            "per_scene": {
                "Warehouse_017": {
                    "HOTA": 0.5686, "DetA": 0.6179,
                    "AssA": 0.5333, "LocA": 0.6169,
                },
            },
            "final": {
                "HOTA": 0.6119, "DetA": 0.6311,
                "AssA": 0.5966, "LocA": 0.6965,
            },
        }

    def test_writes_json_in_0_to_100_scale(self, tmp_path):
        path = save_aicity25_track1_results(self._make_results(), str(tmp_path))
        assert osp.exists(path)
        with open(path, "r") as fp:
            on_disk = json.load(fp)
        # Headline aggregate is multiplied by 100.
        assert on_disk["final"]["HOTA"] == pytest.approx(61.19)
        # Per-(scene, class) too.
        assert on_disk["per_scene_per_class"]["Warehouse_017"]["Person"]["HOTA"] \
            == pytest.approx(74.23)
        # Failed (None) classes round-trip as JSON null.
        assert on_disk["per_scene_per_class"]["Warehouse_017"]["Forklift"] is None

    def test_does_not_mutate_input_results(self, tmp_path):
        """Saving must not change the in-memory results dict."""
        results = self._make_results()
        snapshot = json.dumps(results, sort_keys=True)
        save_aicity25_track1_results(results, str(tmp_path))
        assert json.dumps(results, sort_keys=True) == snapshot

    def test_creates_missing_output_dir(self, tmp_path):
        target = tmp_path / "nested" / "deeper"
        path = save_aicity25_track1_results(self._make_results(), str(target))
        assert osp.exists(path)
        assert path.startswith(str(target))
