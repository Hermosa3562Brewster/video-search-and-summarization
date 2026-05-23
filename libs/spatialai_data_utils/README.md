# `spatialai_data_utils`: Utilities for SpatialAI Datasets

A Python utility package for working with SpatialAI / MTMC multi-camera
datasets: NVSchema and ground-truth loaders, calibration handling,
camera grouping, pure-numpy 3D-to-2D geometry, multi-camera 3D bounding
box visualization, evaluation (detection / tracking / HOTA), and
result-format converters.

## Package Installation

Create conda env for the package.
```bash
conda create -n spatialai_data_utils python=3.13.13 -y
conda activate spatialai_data_utils
```

### Option A: Install from source (recommended for development)

Install torch (pick one variant), then the package and pytorch3d:

**CPU-only torch:**
```bash
pip install torch>=2.10.0 --index-url https://download.pytorch.org/whl/cpu
pip install 'pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git@33824be' --no-build-isolation
pip install --no-cache-dir -e ./release
```

**GPU torch (CUDA):**
```bash
pip install torch>=2.10.0
pip install 'pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git@33824be' --no-build-isolation
pip install --no-cache-dir -e ./release
```

Alternatively, using pipenv:
```bash
pip install pipenv
pipenv install
# Then install torch and pytorch3d manually (see Pipfile for details)
```

### Option B: Install from wheel

`torch` and `pytorch3d` are **not** included in the wheel to avoid pulling in the 7 GB CUDA variant by default. Install them before the wheel:

**Fresh environment (CPU-only torch):**
```bash
pip install torch>=2.10.0 --index-url https://download.pytorch.org/whl/cpu
pip install 'pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git@33824be' --no-build-isolation
pip install spatialai_data_utils-*.whl
```

**Existing environment with GPU torch (e.g. sparse4d):**
```bash
pip install 'pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git@33824be' --no-build-isolation
pip install spatialai_data_utils-*.whl
```

> **Note:** `pytorch3d` must be built from source and requires `torch` to be installed first.
> SDU works with any torch variant (CPU or GPU). The library only uses torch for tensor
> operations and `torch.utils.data.Dataset`, which do not require CUDA.
> `fvcore` and `iopath` (pytorch3d build dependencies) are bundled in the wheel.

### Removing the environment

```bash
conda deactivate
conda remove -n spatialai_data_utils --all
```

## Tools

CLI tools live under `tools/`; each subdirectory ships its own README
with full usage, arguments, and examples.

| Directory | Purpose |
|-----------|---------|
| [`tools/camera_grouping/`](tools/camera_grouping/README.md) | Camera grouping, clustering, and BEV group-origin / dimensions calculation for multi-camera tracking systems. |
| [`tools/visualization/`](tools/visualization/README.md) | Visualization CLIs including 3D bbox rendering (`draw_3dbbox.py`, `draw_3dbbox_batch.py`) and dual-view camera placement from calibration (`draw_camera_placement.py`: 3D frustums + BEV coverage, sequence PNGs). |
| [`tools/projection/`](tools/projection/README.md) | Project NVSchema 3D bounding boxes to 2D image-space corners for a target camera (`project_bbox3d_to_2d.py`). Pure-numpy, no `mmdet3d` dependency. |
| [`tools/video_utils/`](tools/video_utils/README.md) | Video ↔ per-frame-image conversion CLIs: single-video decode (`video2frame.py`) and encode (`frame2video.py`), plus multi-camera scene-wide parallel decode (`video2frame_scene.py`) and stacked-grid encode (`frame2video_scene.py`). |

The library entry points the CLIs wrap are importable from their
defining sub-modules (the package's top-level ``__init__`` stays bare
so callers that only need, say, ``loaders.calibration`` don't pay
for pulling ``cv2`` / ``tqdm`` transitively via
``visualization.render``):

```python
from spatialai_data_utils.visualization.render import (
    visualize_nvschema,
    visualize_3dbbox,
    draw_bev_objects_bbox_in_image,
)
from spatialai_data_utils.core.geometry.projection import (
    project_bev_objects_bbox_in_image,
    project_boxes_3d_to_2d,
)
from spatialai_data_utils.core.boxes.box_3d import (
    box3d_to_corners,
    check_nvschema_coords_len,
)
from spatialai_data_utils.loaders.calibration import (
    load_calib_into_dict,            # flat {cam: calib}
    load_calib_into_dict_with_group_memberships, # flat + {group_name: [cams]} for BEV fan-out
    load_calib_into_dict_from_pkl,
)
from spatialai_data_utils.loaders.nvschema import load_nvschema
from spatialai_data_utils.datasets.frame_paths import (
    resolve_frame_path,                   # single-camera image-path resolver
    get_frame_paths_of_multi_cameras,     # scene-wide image-path lookup
)
```

Each function is documented in its module docstring.

