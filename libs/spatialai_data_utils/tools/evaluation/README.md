# Evaluation Tools

CLI scripts that compute downstream evaluation metrics from already-produced
tracking outputs. Each script is a thin wrapper around the metric
implementations under `spatialai_data_utils.eval.*` — these tools do I/O,
argument parsing, and a bit of orchestration so the same evaluation logic
can be re-run from the command line without standing up a separate eval repo.

## Tools Overview

| Tool                                | Purpose                                                                                                                                                  |
|-------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| **`evaluate_aicity25_track1.py`**   | Reproduce the official AICity'25 Challenge Track 1 (Multi-Camera 3D People Tracking) HOTA evaluation on the challenge's space-separated text format.    |

---

## evaluate_aicity25_track1.py

### Overview

Reproduces the per-scene + per-class HOTA evaluation protocol used by the
official AICity'25 Track 1 validation server on top of
`spatialai_data_utils`' bundled TrackEval library. The numbers this CLI
prints should match the leaderboard numbers for the same
`(ground_truth, submission)` pair to within float noise.

For every `(scene, class)` pair that has both ground-truth and prediction
rows it runs TrackEval's HOTA metric on `MTMCChallenge3DBBox` (3D-IoU
matching — the official Track 1 metric). It then:

1. Averages per-class HOTA / DetA / AssA / LocA into a per-scene number
   (unweighted mean across classes that produced a metric).
2. Weights those per-scene numbers by the count of GT object-frame rows
   that survived the `--num_frames_to_eval` truncation, exactly the way
   the validation server does.

This script is a thin argparse + logging wrapper around the library API
in
[`spatialai_data_utils.eval.tracking.aicity25_track1_eval`](../../spatialai_data_utils/eval/tracking/aicity25_track1_eval.py)
— the same pipeline is callable from notebooks / CI without going
through the CLI. See [Library API](#library-api) below.

### Input format

Both the ground-truth file and the prediction file use the official
AICity'25 Track 1 text format — one space-separated row per object-frame:

```
<scene_id> <class_id> <object_id> <frame_id> <x> <y> <z> <w> <l> <h> <yaw>
```

with `frame_id` 0-indexed and `yaw` in radians. The class id table is
verbatim from the [AICity'25 Track 1 spec](https://www.aicitychallenge.org/2025-track1/):

| ID | Class          |
|----|----------------|
| 0  | Person         |
| 1  | Forklift       |
| 2  | NovaCarter     |
| 3  | Transporter    |
| 4  | FourierGR1T2   |
| 5  | AgilityDigit   |

AICity'25 Track 1 has exactly these six classes — predictions with any
`class_id >= 6` are rejected as out-of-spec rather than silently
accepted.

The scene-id mapping (`--scene_id_2_scene_name_file`) is a JSON object
keyed by the string form of the integer scene id, mapping to a
human-readable scene name used as a directory and table label:

```json
{
  "17": "Warehouse_017",
  "18": "Warehouse_018",
  "19": "Warehouse_019",
  "20": "Warehouse_020"
}
```

When `--scene_id_2_scene_name_file` is omitted the tool falls back to
the packaged AICity'25 Track 1 mapping at
`spatialai_data_utils/datasets/aicity25/scenes/scene_id_to_name.json`,
also accessible programmatically via
`spatialai_data_utils.datasets.aicity25.load_default_scene_id_to_name()`
and `get_default_scene_id_to_name_path()`. Pass
`--scene_id_2_scene_name_file` explicitly only when evaluating a
custom scene set.

### Quick Start

The script lives in the SDU repo and imports from the
`spatialai_data_utils` package, so run it from the repo root (or with
the package installed):

```bash
cd /path/to/spatialai_data_utils

python tools/evaluation/evaluate_aicity25_track1.py \
    --ground_truth_file  data/aicity25/ground_truth/ground_truth.txt \
    --input_file         data/aicity25/v0.6.0/aicity25_submissions_all/R101_iter_4684_conf05/track1_fixed.txt \
    --output_dir         /tmp/aicity25_eval \
    --num_frames_to_eval 9000 \
    --quiet
```

`--scene_id_2_scene_name_file` defaults to the packaged
`Warehouse_017`–`Warehouse_020` mapping, so the typical AICity'25
Track 1 invocation needs only the GT, the submission, and an output
dir.

End-to-end runtime on the four-warehouse AICity'25 GT
(~880k rows) is roughly 5 minutes on a single core, dominated by
TrackEval's HOTA computation on the largest classes (Person).

### Arguments

| Argument                       | Required | Default   | Description |
|--------------------------------|----------|-----------|-------------|
| `--ground_truth_file`          | Yes      | —         | Path to the AICity'25 Track 1 ground-truth text file. |
| `--input_file`                 | Yes      | —         | Path to the AICity'25 Track 1 prediction text file (a single submission's `track1.txt` or `track1_fixed.txt`). |
| `--scene_id_2_scene_name_file` | No       | (packaged) | JSON `{scene_id_str: scene_name}` mapping. When omitted, the tool uses the packaged AICity'25 Track 1 mapping (`Warehouse_017`–`Warehouse_020`); pass this explicitly only for a custom scene set. |
| `--output_dir`                 | No       | (tempdir) | Where the split files, TrackEval scratch artefacts, and the final `aicity25_track1_hota_summary.json` are written. Omit to use a tempdir and discard intermediates at exit. |
| `--num_cores`                  | No       | `1`       | Forwarded to TrackEval's `NUM_PARALLEL_CORES`. Has near-zero impact here because we run TrackEval once per `(scene, class)` pair with a single sequence. |
| `--num_frames_to_eval`         | No       | `9000`    | Frame-count truncation per scene (0-indexed exclusive upper bound). Matches the official validation server default. |
| `--eval_type`                  | No       | `bbox`    | HOTA matching function: `bbox` (3D-IoU — the official Track 1 metric) or `location` (centre distance — useful for ablation only). |
| `--fps`                        | No       | `30.0`    | FPS written into TrackEval's per-sequence `seqinfo.ini` (cosmetic for single-sequence runs). |
| `--quiet`                      | No       | (off)     | Suppress TrackEval's per-(scene, class) `INFO` records and per-class metric tables, keeping the final summary table readable. |

### Output

Two pieces of output:

1. A formatted summary printed to stdout (via Python `logging`), with:
   - A per-(scene, class) HOTA / DetA / AssA / LocA table.
   - A per-scene table that also shows the GT row count used as the
     aggregation weight.
   - A single `WEIGHTED FINAL` row that mirrors the leaderboard number.
2. When `--output_dir` is provided, a JSON file at
   `<output_dir>/aicity25_track1_hota_summary.json` with every metric in
   the 0–100 scale, suitable for ingestion by dashboards or CI:

   ```json
   {
     "eval_type": "bbox",
     "num_frames_to_eval": 9000,
     "scene_id_to_name": { "17": "Warehouse_017", ... },
     "per_scene_object_counts": { "Warehouse_017": 179981, ... },
     "per_scene_per_class": {
       "Warehouse_017": {
         "Person":       { "HOTA": 74.23, "DetA": 77.81, "AssA": 70.82, "LocA": 83.77 },
         "Transporter":  { "HOTA": 15.06, "DetA": 24.47, "AssA":  9.28, "LocA": 22.37 },
         ...
       },
       ...
     },
     "per_scene": {
       "Warehouse_017": { "HOTA": 56.86, "DetA": 61.79, "AssA": 53.33, "LocA": 61.69 },
       ...
     },
     "final": { "HOTA": 61.19, "DetA": 63.11, "AssA": 59.66, "LocA": 69.65 }
   }
   ```

   In addition, the `<output_dir>/split/<scene>/<class>/{gt,pred}.txt`
   intermediates are kept so you can re-run TrackEval manually on any
   single `(scene, class)` pair if you need to debug or experiment.

### Library API

Import the same functions the CLI uses from
`spatialai_data_utils.eval.tracking.aicity25_track1_eval`:

| Symbol                                | Purpose |
|---------------------------------------|---------|
| `HOTA_FIELDS`                         | `["HOTA", "DetA", "AssA", "LocA"]` — the metric quartet reported by the leaderboard. |
| `split_aicity25_per_scene_per_class(...)` | Stream-split an AICity'25 Track 1 text file into `<scene>/<class>/<basename>` MOT-format files; returns `{scene: {class: row_count}}`. Useful on its own for per-class analyses / visualizers / custom evaluators. |
| `run_aicity25_track1_evaluation(...)` | End-to-end orchestrator: splits, runs HOTA per (scene, class), aggregates, returns a results dict. |
| `print_aicity25_track1_summary(results)` | Log the per-(scene, class) + per-scene summary table to the module's logger. |
| `save_aicity25_track1_results(results, output_dir)` | Persist a results dict to `<output_dir>/aicity25_track1_hota_summary.json` in the 0–100 scale used by the official leaderboard. Returns the JSON path. |

Spec constants — the class-id → name table and the text-format
field count — live in `spatialai_data_utils.datasets.aicity25.spec`
so the eval module, the submission converters under
`tools/aicity25/`, and any future AICity'25 consumer can share a
single source of truth:

```python
from spatialai_data_utils.datasets.aicity25.spec import (
    CLASS_ID_TO_NAME,   # {0: "Person", 1: "Forklift", ...}
    NUM_FIELDS,         # 11
)
```

Minimal usage from Python:

```python
from spatialai_data_utils.datasets.aicity25 import load_default_scene_id_to_name
from spatialai_data_utils.eval.tracking.aicity25_track1_eval import (
    run_aicity25_track1_evaluation,
    print_aicity25_track1_summary,
    save_aicity25_track1_results,
)

results = run_aicity25_track1_evaluation(
    ground_truth_file="data/aicity25/ground_truth/ground_truth.txt",
    prediction_file="data/aicity25/.../track1_fixed.txt",
    scene_id_to_name=load_default_scene_id_to_name(),
    output_dir="/tmp/aicity25_eval",   # or None for a tempdir
)
print_aicity25_track1_summary(results)
save_aicity25_track1_results(results, "/tmp/aicity25_eval")

print(results["final"]["HOTA"])    # 0.611948 (in [0, 1] scale)
```

The orchestrator returns metrics as ratios in `[0, 1]` (matching
TrackEval's native scale).  Both `print_aicity25_track1_summary` and
`save_aicity25_track1_results` multiply by 100 for display /
persistence, matching the convention used by the official leaderboard.

### Notes & Gotchas

- The class names emitted here (e.g. `NovaCarter`, `FourierGR1T2`) match
  the [AICity'25 Track 1 spec](https://www.aicitychallenge.org/2025-track1/)
  verbatim — *not* the SDU evaluation pipeline's preferred
  `Nova_Carter` / `Fourier_GR1_T2_Humanoid` spelling — because this tool's
  job is to reproduce the official challenge metric, not to align with
  the SDU internal taxonomy. The class names are display labels only;
  TrackEval keys results by sequence + the literal `"class"` token, not
  by class name, so the spelling does not affect the computed metric.
- The prediction file format does **not** carry a confidence column, so
  there is no `--confidence_threshold` option here — filter your
  submission upstream (e.g. via the
  `aicity25-submission` skill's
  `tools/aicity25/convert_sparse4d_to_aicity25.py --conf_thresh`).
- `num_frames_to_eval` truncates by **frame ID**, not by **row count**,
  so it is safe to leave at its `9000` default even when one scene has
  many fewer rows.
- The auto-derived scene name (`scene_<id>`) is purely for display; the
  HOTA numbers under `scene_<id>` are identical to those under the
  named-scene mapping for the same underlying data.
