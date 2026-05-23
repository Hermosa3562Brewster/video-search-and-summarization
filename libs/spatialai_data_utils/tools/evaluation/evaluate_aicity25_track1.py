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
CLI wrapper around the AICity'25 Track 1 evaluation API.

All of the actual evaluation logic lives in
:mod:`spatialai_data_utils.eval.tracking.aicity25_track1_eval`; this
script is a thin argparse + logging shell that wires the CLI flags
into ``run_aicity25_track1_evaluation`` / ``print_aicity25_track1_summary``
/ ``save_aicity25_track1_results`` so the same pipeline is callable
from a shell and from Python notebooks / CI scripts.

Example::

    python tools/evaluation/evaluate_aicity25_track1.py \\
        --ground_truth_file  data/aicity25/ground_truth/ground_truth.txt \\
        --input_file         data/aicity25/v0.6.0/aicity25_submissions_all/R101_iter_4684_conf05/track1_fixed.txt \\
        --output_dir         /tmp/aicity25_eval \\
        --quiet

When ``--scene_id_2_scene_name_file`` is omitted the tool falls back
to the packaged AICity'25 Track 1 mapping bundled at
:func:`spatialai_data_utils.datasets.aicity25.get_default_scene_id_to_name_path`
(currently scene IDs ``17``–``20`` →
``Warehouse_017``–``Warehouse_020``).  Pass ``--scene_id_2_scene_name_file``
explicitly if you are evaluating a custom scene set.
"""

import argparse
import logging
import time

from spatialai_data_utils.datasets.aicity25 import (
    get_default_scene_id_to_name_path,
    load_default_scene_id_to_name,
)
from spatialai_data_utils.eval.common.io_utils import (
    ValidateFile,
    load_json_from_file,
    validate_file_path,
)
from spatialai_data_utils.eval.tracking.aicity25_track1_eval import (
    print_aicity25_track1_summary,
    run_aicity25_track1_evaluation,
    save_aicity25_track1_results,
)


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse the CLI arguments for the AICity'25 Track 1 evaluator."""
    parser = argparse.ArgumentParser(
        description=(
            "AICity'25 Challenge - Track 1 (Multi-Camera 3D People "
            "Tracking) evaluation. Computes per-scene HOTA / DetA / "
            "AssA / LocA on the official Track 1 text format and "
            "reports the GT-object-count-weighted mean used by the "
            "competition validation server."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--ground_truth_file",
        type=validate_file_path, action=ValidateFile, required=True,
        help="Path to the AICity'25 Track 1 ground-truth text file.",
    )
    parser.add_argument(
        "--input_file",
        type=validate_file_path, action=ValidateFile, required=True,
        help="Path to the AICity'25 Track 1 prediction text file "
             "(a single submission's track1.txt or track1_fixed.txt).",
    )
    parser.add_argument(
        "--scene_id_2_scene_name_file",
        type=validate_file_path, action=ValidateFile, default=None,
        help="Optional JSON mapping {scene_id_str: scene_name} for the "
             "scenes to evaluate. Scenes outside this mapping are "
             "dropped from GT and rejected from predictions. When "
             "omitted, falls back to the packaged AICity'25 Track 1 "
             "mapping (scene IDs 17-20 -> Warehouse_017-Warehouse_020) "
             "exposed by "
             "spatialai_data_utils.datasets.aicity25.get_default_scene_id_to_name_path.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Optional output directory. When set, the per-(scene, "
             "class) split files, TrackEval scratch artefacts, and "
             "the final aicity25_track1_hota_summary.json are written "
             "here. When omitted, a tempdir is used and removed at "
             "the end of the run (no JSON summary is persisted).",
    )
    parser.add_argument(
        "--num_cores", type=int, default=1,
        help="Number of cores forwarded to TrackEval. Has negligible "
             "effect here because we run TrackEval once per "
             "(scene, class) pair with a single sequence. Default: 1.",
    )
    parser.add_argument(
        "--num_frames_to_eval", type=int, default=9000,
        help="Frame-count truncation per scene (0-indexed exclusive "
             "upper bound). Matches the official validation server "
             "default. Default: 9000.",
    )
    parser.add_argument(
        "--eval_type", choices=["bbox", "location"], default="bbox",
        help="HOTA matching function: 'bbox' (3D IoU, the official "
             "AICity Track 1 metric) or 'location' (centre distance, "
             "useful for ablation only). Default: bbox.",
    )
    parser.add_argument(
        "--fps", type=float, default=30.0,
        help="FPS written into TrackEval's per-sequence seqinfo.ini "
             "(cosmetic for single-sequence runs). Default: 30.0.",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress TrackEval's per-class INFO logs (keeps the "
             "summary table readable).",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%y/%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    args = parse_args()

    if args.scene_id_2_scene_name_file is not None:
        scene_id_to_name = load_json_from_file(args.scene_id_2_scene_name_file)
        if not isinstance(scene_id_to_name, dict):
            raise ValueError(
                f"scene-id mapping file {args.scene_id_2_scene_name_file} "
                f"must contain a JSON object, got "
                f"{type(scene_id_to_name).__name__}."
            )
        scene_id_to_name = {str(k): str(v) for k, v in scene_id_to_name.items()}
    else:
        scene_id_to_name = load_default_scene_id_to_name()
        logger.info(
            "Using packaged AICity'25 Track 1 scene-id mapping from %s: %s",
            get_default_scene_id_to_name_path(),
            scene_id_to_name,
        )

    start_time = time.time()
    results = run_aicity25_track1_evaluation(
        ground_truth_file=args.ground_truth_file,
        prediction_file=args.input_file,
        scene_id_to_name=scene_id_to_name,
        output_dir=args.output_dir,
        num_cores=args.num_cores,
        num_frames_to_eval=args.num_frames_to_eval,
        eval_type=args.eval_type,
        fps=args.fps,
        quiet=args.quiet,
    )

    print_aicity25_track1_summary(results)
    if args.output_dir is not None:
        save_aicity25_track1_results(results, args.output_dir)

    final = results["final"]
    logger.info(
        "Final weighted: HOTA=%.4f DetA=%.4f AssA=%.4f LocA=%.4f "
        "(in 0-100 scale)",
        final["HOTA"] * 100, final["DetA"] * 100,
        final["AssA"] * 100, final["LocA"] * 100,
    )
    logger.info("Total time: %.1f seconds", time.time() - start_time)


if __name__ == "__main__":
    main()
