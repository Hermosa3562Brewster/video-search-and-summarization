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

"""Tests for the calibration data layer.

Covers two adjacent areas:

* :func:`spatialai_data_utils.loaders.calibration.load_calib_json`
  — generalised in the 2026-Q2 ``datasets/`` reorganisation to absorb
  the previously-separate ``load_calibration_data`` helper. Pins the
  new contract: ``str`` / :class:`pathlib.Path` input; three input
  shapes (scene directory, ``calibration.json`` file, arbitrary JSON
  file); optional ``validate=True`` flag that runs schema validation
  post-load and logs (but does not raise) on failure.
* :func:`spatialai_data_utils.loaders.calibration.validate_calibration_data`
  — direct schema validation (happy + raise paths).
* The non-loader calibration helpers in
  :mod:`spatialai_data_utils.core.cameras.utils`:
  :func:`extract_camera_matrices` (handles processed and raw sensor
  formats; gracefully rejects malformed matrices) and
  :func:`save_calibration_data` (round-trip JSON write).
"""

import json
import logging

import numpy as np
import pytest
from jsonschema.exceptions import ValidationError

from spatialai_data_utils.core.cameras.utils import (
    extract_camera_matrices,
    save_calibration_data,
)
from spatialai_data_utils.loaders.calibration import (
    load_calib_json,
    validate_calibration_data,
)


def _minimal_calib_dict() -> dict:
    """Build a minimal NVSchema-shaped calibration dict.

    Includes the four top-level fields required by the calibration JSON
    schema (``version``, ``osmURL``, ``calibrationType``, ``sensors``)
    and two synthetic camera sensors with intrinsic / extrinsic
    matrices and the ``frameWidth`` / ``frameHeight`` attributes that
    the schema requires for sensors of ``type: "camera"``.
    """
    sensors = []
    for cam_id in ["Camera_01", "Camera_02"]:
        sensors.append({
            "id": cam_id,
            "type": "camera",
            "intrinsicMatrix": [
                [500.0, 0.0, 320.0],
                [0.0, 500.0, 240.0],
                [0.0, 0.0, 1.0],
            ],
            "extrinsicMatrix": np.eye(4)[:3].tolist(),
            "attributes": [
                {"name": "frameWidth", "value": "640"},
                {"name": "frameHeight", "value": "480"},
            ],
        })
    return {
        "version": "1.0",
        "osmURL": "",
        "calibrationType": "cartesian",
        "sensors": sensors,
    }


def _write_calib_json(path) -> dict:
    """Write a minimal NVSchema calibration.json under *path* and return it."""
    data = _minimal_calib_dict()
    with open(path, "w") as f:
        json.dump(data, f)
    return data


# ---------------------------------------------------------------------------
# Input shapes — directory / calibration.json file / arbitrary JSON file
# ---------------------------------------------------------------------------


class TestInputShapes:
    """Three accepted input shapes all yield equivalent parsed JSON."""

    def test_directory_input_string(self, tmp_path):
        """Passing a scene directory (str) reads its ``calibration.json``."""
        _write_calib_json(tmp_path / "calibration.json")
        result = load_calib_json(str(tmp_path), load_original=True)
        assert "sensors" in result
        assert {s["id"] for s in result["sensors"]} == {"Camera_01", "Camera_02"}

    def test_directory_input_path_object(self, tmp_path):
        """Path objects are accepted on equal terms with strings."""
        _write_calib_json(tmp_path / "calibration.json")
        result = load_calib_json(tmp_path, load_original=True)
        assert "sensors" in result

    def test_calibration_json_file_input(self, tmp_path):
        """Passing the ``calibration.json`` file directly works too."""
        calib_path = tmp_path / "calibration.json"
        _write_calib_json(calib_path)
        result = load_calib_json(calib_path, load_original=True)
        assert "sensors" in result

    def test_arbitrary_json_file_with_load_original_true(self, tmp_path):
        """Any JSON file is accepted when ``load_original=True``."""
        path = tmp_path / "custom_calib.json"
        with open(path, "w") as f:
            json.dump({"arbitrary": "structure", "nested": {"x": 1}}, f)
        result = load_calib_json(path, load_original=True)
        assert result == {"arbitrary": "structure", "nested": {"x": 1}}


# ---------------------------------------------------------------------------
# Return shapes — load_original toggles raw vs id-keyed
# ---------------------------------------------------------------------------


class TestReturnShape:
    """``load_original`` flips between raw JSON dict and id-keyed flat dict."""

    def test_load_original_true_returns_raw_json_dict(self, tmp_path):
        """``load_original=True`` preserves the top-level ``"sensors"`` array."""
        _write_calib_json(tmp_path / "calibration.json")
        result = load_calib_json(tmp_path, load_original=True)
        assert isinstance(result, dict)
        assert "sensors" in result
        assert isinstance(result["sensors"], list)
        assert len(result["sensors"]) == 2

    def test_load_original_false_returns_id_keyed_flat_dict(self, tmp_path):
        """Default mode keys raw sensor dicts by their ``id``."""
        _write_calib_json(tmp_path / "calibration.json")
        result = load_calib_json(tmp_path)  # load_original=False
        assert set(result.keys()) == {"Camera_01", "Camera_02"}
        # Each value is the raw sensor dict (with intrinsicMatrix, extrinsicMatrix, etc.).
        assert "intrinsicMatrix" in result["Camera_01"]
        assert "extrinsicMatrix" in result["Camera_01"]


# ---------------------------------------------------------------------------
# Schema validation — opt-in via validate=True
# ---------------------------------------------------------------------------


class TestValidate:
    """``validate=True`` runs schema validation; failures are logged, not raised."""

    def test_default_does_not_validate(self, tmp_path):
        """``validate=False`` (default) accepts a malformed calibration silently."""
        path = tmp_path / "calibration.json"
        # Missing the required "sensors" array — schema-invalid but
        # load_original=True returns the raw dict regardless.
        with open(path, "w") as f:
            json.dump({"version": "1.0"}, f)
        result = load_calib_json(path, load_original=True)
        assert result == {"version": "1.0"}

    def test_validate_true_warns_on_invalid_and_returns_data(self, tmp_path, caplog):
        """Schema failures are logged at WARNING but do *not* raise."""
        path = tmp_path / "calibration.json"
        with open(path, "w") as f:
            json.dump({"version": "1.0"}, f)  # missing required "sensors"

        with caplog.at_level(logging.WARNING, logger="spatialai_data_utils.loaders.calibration"):
            result = load_calib_json(
                path, load_original=True, validate=True,
            )

        assert result == {"version": "1.0"}
        assert any(
            "Calibration data validation failed" in record.message
            for record in caplog.records
        ), "Expected a WARNING log line on schema validation failure."


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestErrors:
    """Misuse paths."""

    def test_arbitrary_json_with_load_original_false_raises_value_error(self, tmp_path):
        """Arbitrary JSON files require ``load_original=True``.

        The default ``load_original=False`` mode keys the result by sensor
        id, which presupposes the NVSchema ``"sensors"`` array layout. An
        arbitrary JSON file has no such guarantee, so we refuse early
        with a clear message instead of failing later on a missing
        ``"sensors"`` key.
        """
        path = tmp_path / "custom.json"
        with open(path, "w") as f:
            json.dump({"foo": "bar"}, f)
        with pytest.raises(ValueError, match="load_original=False"):
            load_calib_json(path)

    def test_missing_directory_raises_file_not_found(self, tmp_path):
        """A nonexistent directory bubbles up the underlying ``FileNotFoundError``."""
        missing = tmp_path / "no_such_scene"
        with pytest.raises(FileNotFoundError):
            load_calib_json(missing, load_original=True)

    def test_missing_calibration_json_raises_file_not_found(self, tmp_path):
        """A directory without ``calibration.json`` raises ``FileNotFoundError``."""
        empty_dir = tmp_path / "scene_with_no_calib"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            load_calib_json(empty_dir, load_original=True)


# ---------------------------------------------------------------------------
# Direct tests for validate_calibration_data
# ---------------------------------------------------------------------------


class TestValidateCalibrationData:
    """``validate_calibration_data`` raises on schema violations.

    The raise behaviour is what :func:`load_calib_json`
    catches when called with ``validate=True`` (it logs and continues).
    Calling :func:`validate_calibration_data` directly preserves the
    raise so callers that *want* a hard failure can have one.
    """

    def test_missing_sensors_raises(self):
        """Missing the required ``"sensors"`` array raises ``ValidationError``."""
        with pytest.raises(ValidationError):
            validate_calibration_data({"version": "1.0"})


# ---------------------------------------------------------------------------
# extract_camera_matrices (core/cameras/utils.py)
# ---------------------------------------------------------------------------


class TestExtractCameraMatrices:
    """Pull intrinsic/extrinsic numpy matrices out of a sensor calibration dict."""

    def _processed_sensor(self):
        """Sensor in the *processed* format (``intrinsic_matrix`` /
        ``w2c_matrix`` keys, post-loader curation)."""
        return {
            "intrinsic_matrix": [
                [500.0, 0.0, 320.0],
                [0.0, 500.0, 240.0],
                [0.0, 0.0, 1.0],
            ],
            "w2c_matrix": np.eye(4)[:3].tolist(),
        }

    def _raw_sensor(self):
        """Sensor in the *raw* NVSchema format (``intrinsicMatrix`` /
        ``extrinsicMatrix`` keys, pre-loader curation)."""
        return {
            "id": "Camera_01",
            "intrinsicMatrix": [
                [500.0, 0.0, 320.0],
                [0.0, 500.0, 240.0],
                [0.0, 0.0, 1.0],
            ],
            "extrinsicMatrix": np.eye(4).tolist(),
        }

    def test_processed_format_returns_arrays(self):
        intrinsic, extrinsic = extract_camera_matrices(self._processed_sensor())
        assert isinstance(intrinsic, np.ndarray)
        assert isinstance(extrinsic, np.ndarray)
        assert intrinsic.shape == (3, 3)
        assert extrinsic.shape == (3, 4)

    def test_raw_nvschema_format_returns_arrays(self):
        """Raw sensors with ``intrinsicMatrix`` / ``extrinsicMatrix`` also work."""
        intrinsic, extrinsic = extract_camera_matrices(self._raw_sensor())
        assert intrinsic.shape == (3, 3)
        assert extrinsic.shape == (4, 4)

    def test_legacy_keys_via_get_calib_field_fallback(self):
        """Legacy spaced keys (``"intrinsic matrix"`` / ``"projection matrix w2c"``).

        Pins the ``get_calib_field`` legacy-key fallback in the
        processed-format branch.
        """
        sensor = {
            "intrinsic matrix": [
                [500.0, 0.0, 320.0],
                [0.0, 500.0, 240.0],
                [0.0, 0.0, 1.0],
            ],
            "projection matrix w2c": np.eye(4)[:3].tolist(),
        }
        intrinsic, extrinsic = extract_camera_matrices(sensor)
        assert intrinsic.shape == (3, 3)
        assert extrinsic.shape == (3, 4)

    def test_invalid_intrinsic_shape_returns_none_pair(self):
        """A 2×2 intrinsic is rejected by the shape validator."""
        sensor = self._processed_sensor()
        sensor["intrinsic_matrix"] = [[1.0, 0.0], [0.0, 1.0]]
        intrinsic, extrinsic = extract_camera_matrices(sensor)
        assert intrinsic is None
        assert extrinsic is None

    def test_nan_intrinsic_returns_none_pair(self):
        """``NaN`` values in the intrinsic matrix are rejected."""
        sensor = self._processed_sensor()
        sensor["intrinsic_matrix"][0][0] = float("nan")
        intrinsic, extrinsic = extract_camera_matrices(sensor)
        assert intrinsic is None
        assert extrinsic is None

    def test_missing_calibration_returns_none_pair(self):
        """A sensor with no recognised matrix fields returns ``(None, None)``."""
        intrinsic, extrinsic = extract_camera_matrices({"id": "Camera_01"})
        assert intrinsic is None
        assert extrinsic is None


# ---------------------------------------------------------------------------
# save_calibration_data (core/cameras/utils.py)
# ---------------------------------------------------------------------------


class TestSaveCalibrationData:
    """``save_calibration_data`` writes a JSON file (round-trippable)."""

    def test_round_trip(self, tmp_path):
        """Writing and re-reading should preserve the dict bit-for-bit."""
        data = _minimal_calib_dict()
        out_path = tmp_path / "saved_calibration.json"
        save_calibration_data(data, str(out_path))
        assert out_path.exists()
        with open(out_path, "r") as f:
            reloaded = json.load(f)
        assert reloaded == data

    def test_round_trip_via_load_calib_json(self, tmp_path):
        """End-to-end: save then re-load via the canonical loader."""
        data = _minimal_calib_dict()
        out_path = tmp_path / "calibration.json"
        save_calibration_data(data, str(out_path))
        reloaded = load_calib_json(
            tmp_path, load_original=True, validate=True,
        )
        assert reloaded == data

    def test_creates_missing_parent_directories(self, tmp_path):
        """Parent dirs are auto-created when they don't exist yet."""
        data = _minimal_calib_dict()
        nested = tmp_path / "deeply" / "nested" / "out.json"
        save_calibration_data(data, str(nested))
        assert nested.exists()
