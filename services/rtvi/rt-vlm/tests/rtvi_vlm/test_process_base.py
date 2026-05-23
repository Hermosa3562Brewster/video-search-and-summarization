################################################################################
#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
#  All rights reserved.
#  SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
#  NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
#  property and proprietary rights in and to this material, related
#  documentation and any modifications thereto. Any use, reproduction,
#  disclosure or distribution of this software and related documentation
#  without an express license agreement from NVIDIA CORPORATION or
#  its affiliates is strictly prohibited.
################################################################################

from vlm_pipeline import process_base as process_base_module
from vlm_pipeline.process_base import ProcessBase


class _RecordingQueue:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)


class _NoBatchProcess(ProcessBase):
    def __init__(self):
        pass

    def _supports_batching(self):
        return False


def test_handle_result_reports_frame_transfer_failure(monkeypatch):
    proc = _NoBatchProcess()
    proc._output_queue = _RecordingQueue()
    proc._final_output_queue = _RecordingQueue()
    monkeypatch.setattr(process_base_module, "_safe_cuda_empty_cache", lambda **kwargs: None)

    def fail_frame_transfer(value):
        raise RuntimeError("CUDA illegal memory access")

    monkeypatch.setattr(process_base_module, "_move_cuda_frames_to_cpu", fail_frame_transfer)

    chunk = object()
    proc._handle_result(
        {
            "chunk": chunk,
            "chunk_id": 7,
            "frames": object(),
            "error": None,
        },
        chunk=chunk,
        chunk_id=7,
    )

    assert proc._output_queue.items == []
    assert len(proc._final_output_queue.items) == 1
    error_item = proc._final_output_queue.items[0]
    assert error_item["chunk"] is chunk
    assert error_item["chunk_id"] == 7
    assert error_item["error_status_code"] == 500
    assert "CUDA illegal memory access" in error_item["error"]
    assert "frames" not in error_item


def test_handle_result_moves_error_frames_before_final_queue(monkeypatch):
    proc = _NoBatchProcess()
    proc._output_queue = _RecordingQueue()
    proc._final_output_queue = _RecordingQueue()
    monkeypatch.setattr(process_base_module, "_safe_cuda_empty_cache", lambda **kwargs: None)

    calls = []

    def record_frame_transfer(value):
        calls.append(value)
        return "cpu-frames"

    monkeypatch.setattr(process_base_module, "_move_cuda_frames_to_cpu", record_frame_transfer)

    chunk = object()
    frames = object()
    proc._handle_result(
        {
            "chunk": chunk,
            "chunk_id": 8,
            "frames": frames,
            "error": "Decode error",
        },
        chunk=chunk,
        chunk_id=8,
    )

    assert calls == [frames]
    assert proc._output_queue.items == []
    assert len(proc._final_output_queue.items) == 1
    error_item = proc._final_output_queue.items[0]
    assert error_item["chunk"] is chunk
    assert error_item["chunk_id"] == 8
    assert error_item["error"] == "Decode error"
    assert error_item["frames"] == "cpu-frames"
