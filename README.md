# Nabla

Nabla is a unified pipeline for extracting expressive 3D human pose and shape — [SMPL-X](https://smpl-x.is.tue.mpg.de/) parameters and meshes — from videos. It wraps several state-of-the-art human mesh recovery models behind a single, consistent interface so you can download videos, run inference with any supported backend, post-process the results, and evaluate them with the same set of scripts.

It is built for batch processing on a Slurm/conda HPC environment (developed on the Lanta cluster) and is used to extract motion from sign-language videos for downstream text-to-sign model training.

## Supported models

Each model is vendored as a git submodule under [`utils/extraction/`](utils/extraction/) and driven through this repo's scripts and patches.

| Backend | Target | Source |
| --- | --- | --- |
| `SMPLer-X` | Whole-body (SMPL-X) | [SMPLCap/SMPLer-X](https://github.com/SMPLCap/SMPLer-X) |
| `SMPLest-X` | Whole-body (SMPL-X) | [SMPLCap/SMPLest-X](https://github.com/SMPLCap/SMPLest-X) |
| `OSX` | Whole-body (SMPL-X) | [IDEA-Research/OSX](https://github.com/IDEA-Research/OSX) |
| `WiLoR` | Hands (MANO) | [rolpotamias/WiLoR](https://github.com/rolpotamias/WiLoR) |

## Repository layout

```text
.
├── main/
│   └── prepare.py              # Download + unzip the video dataset
├── configs/
│   └── extraction/prepare.py   # Download destination / filename config
├── scripts/
│   ├── prepare/
│   │   ├── environment/        # Per-backend conda env setup (main.sh dispatcher)
│   │   ├── download-videos.sh  # Wrapper around main/prepare.py
│   │   ├── video-to-image.sh   # FFmpeg video → frame sequence
│   │   ├── smooth_smplerx.py   # One-Euro smoothing of SMPLer-X output
│   │   └── restructure_osx.sh  # Reshape OSX output into SMPLer-X layout
│   ├── inference/
│   │   └── main.sh             # Unified inference dispatcher
│   └── evaluation/
│       └── evaluate_mesh.sh    # MPVPE mesh evaluation wrapper
├── utils/
│   ├── extraction/             # Model submodules (SMPLer-X, SMPLest-X, OSX, WiLoR)
│   ├── evaluation/
│   │   └── mesh_evaluation.py  # MPVPE between GT and predicted OBJ meshes
│   └── parse-pkl.py            # Inspect exported .pkl parameter files
└── patches/                    # Drop-in fixes applied to submodules during setup
```

## Prerequisites

- Conda / Miniconda (each backend runs in its own environment)
- NVIDIA GPU with CUDA 11.3+ (PyTorch built against cu113 for most backends)
- `ffmpeg` / `ffprobe` for frame extraction
- Slurm (`SMPLer-X` inference is submitted via `srun`)

## Setup

Clone with submodules:

```bash
git clone --recurse-submodules <repo-url> Nabla
cd Nabla
# or, if already cloned:
git submodule update --init --recursive

pip install -r requirements.txt
```

### Prepare a backend environment

Each backend has its own conda environment, pretrained weights, and SMPL-X / MANO model files. Set them all up through the dispatcher:

```bash
bash scripts/prepare/environment/main.sh <smplerx|osx|wilor>
```

This creates the conda env, installs dependencies, downloads pretrained models and human-model files (from HuggingFace / Google Drive), and applies the relevant [patches](patches/).

> [!IMPORTANT]
> Each inference algorithm is a separate submodule with its own environment, weights, and model files. Cloning the submodules and installing `requirements.txt` is **not** enough — you must run this setup step individually for **every** backend you intend to run inference with. A backend cannot be used until its submodule has been set up this way.

## Data preparation

### Download videos

```bash
bash scripts/prepare/download-videos.sh <GOOGLE_DRIVE_URL>
```

Downloads and unzips the dataset into `data/` (see [`configs/extraction/prepare.py`](configs/extraction/prepare.py)).

### Extract frames

The models run on frame sequences. Convert each video into a numbered image sequence:

```bash
bash scripts/prepare/video-to-image.sh -d data/videos -t mp4 -o jpg [-O <out_dir>] [-f <fps>]
```

Frames are written to one subdirectory per video (`000001.jpg`, `000002.jpg`, …). Omit `-f` to use the source video's frame rate.

## Inference

Run any backend over a directory of per-video subfolders through the unified dispatcher:

```bash
bash scripts/inference/main.sh <ALGORITHM> <VIDEO_DIR> <FORMAT> [--skip a,b] [--only c,d]
```

- `ALGORITHM` — one of `SMPLer-X`, `SMPLest-X`, `OSX`, `WiLoR`
- `VIDEO_DIR` — directory containing one subdirectory per video
- `FORMAT` — video extension (e.g. `mp4`)
- `--skip` — comma-separated video names to exclude
- `--only` — comma-separated video names to process exclusively (mutually exclusive with `--skip`)

Examples:

```bash
bash scripts/inference/main.sh SMPLer-X data/videos mp4
bash scripts/inference/main.sh WiLoR  data/videos mp4 --skip video_01,video_03
bash scripts/inference/main.sh OSX    data/videos mp4 --only video_02,video_04
```

## Post-processing

**Smooth SMPLer-X output** — apply a One-Euro filter to reduce per-frame jitter:

```bash
python scripts/prepare/smooth_smplerx.py [--results-dir DIR] [--suffix SUFFIX] [--freq HZ]
```

**Restructure OSX output** — reshape OSX's flat output into the SMPLer-X-style `mesh/ pkl/ render/ kpts/` layout:

```bash
bash scripts/prepare/restructure_osx.sh <results_dir>            # single video
bash scripts/prepare/restructure_osx.sh <results_dir> --batch    # all subdirs
```

## Evaluation

Compute Mean Per-Vertex Position Error (MPVPE) between ground-truth and predicted OBJ mesh sequences:

```bash
bash scripts/evaluation/evaluate_mesh.sh <GT_DIR> <PRED_DIR> <OUT_DIR>
```

Results are written as a CSV in `OUT_DIR`. See [`utils/evaluation/mesh_evaluation.py`](utils/evaluation/mesh_evaluation.py) for the metric implementation.

## Output

After inference each video produces a result folder containing:

- `mesh/` — per-frame SMPL-X / MANO meshes (`.obj`)
- `pkl/` — per-frame model parameters (`.pkl`)
- `render/` — overlay renders
- `kpts/` — keypoint visualizations

Inspect an exported parameter file with [`utils/parse-pkl.py`](utils/parse-pkl.py).

## Patches

The submodules are vendored upstream; small fixes (inference loops, filename handling, PyTorch/`torchgeometry` compatibility) live in [`patches/`](patches/) and are applied automatically by the environment setup scripts. Keeping fixes as patches rather than committing into the submodules keeps the upstream history clean and easy to update.
