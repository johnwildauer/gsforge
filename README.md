# gsforge

**CLI-first 3D Gaussian Splatting for virtual production workflows.**

gsforge is an open-source, pipeline-oriented tool that takes you from raw VP footage to a trained 3DGS `.ply` model in a single command. It is designed for technical artists and pipeline TDs who want reproducible, scriptable, and fast reconstructions — not a GUI.

---

## Why gsforge?

| Problem                                 | gsforge solution                                                                  |
| --------------------------------------- | --------------------------------------------------------------------------------- |
| COLMAP on raw 24fps footage takes hours | Smart frame downsampling: extract only 5 fps (configurable), capped at 400 frames |
| Classic COLMAP incremental SfM is slow  | Default to **GLOMAP** (global SfM via COLMAP 4.x) — 5–20× faster                  |
| Projects are hard to share / move       | Everything lives in a self-contained `MyScene.gsproject/` folder                  |
| Integrating with other tools is painful | `gsforge export-colmap` produces a standard COLMAP folder any tool can open       |
| Training scripts are hard to configure  | One command: `gsforge train`                                                      |
| Can't see training progress             | Preview renders saved every N iterations to `renders/`                            |

---

## Quick install

### 1. Create a conda environment

If you don't already have `conda` installed, grab either:

- **[Anaconda](https://www.anaconda.com/download)** — full distribution with many pre-installed packages, or
- **[Miniconda](https://docs.conda.io/en/latest/miniconda.html)** — minimal installer (recommended if you just need `conda`)

Once `conda` is available on your PATH:

```bash
conda create -n gsforge python=3.10 -y
conda activate gsforge
```

### 2. Install PyTorch with the correct CUDA version

gsforge requires **PyTorch 2.4** with CUDA support. Install it **before** installing gsforge so pip resolves the correct CUDA-enabled wheels:

```bash
# CUDA 12.4 (required for the gsplat precompiled wheels)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 3. Install the NVIDIA CUDA Toolkit

gsplat is a **CUDA-accelerated** library. Without the CUDA Toolkit installed on your system, gsplat will print:

```
gsplat: No CUDA toolkit found. gsplat will be disabled.
```

and fall back to a very slow CPU-only mode (hours per training run instead of minutes).

**Windows (VP workstations)**

1. Download **CUDA Toolkit 12.4** from:
   https://developer.nvidia.com/cuda-12-4-0-download-archive
   _(Choose: Windows → x86_64 → 11 or 12 → Local)_
2. Run the installer. Select **Custom install** and ensure **CUDA Toolkit** is checked.
3. Reboot after installation.

**Linux**

```bash
# Ubuntu 22.04 example (CUDA 12.4)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update
sudo apt install cuda-toolkit-12-4
```

**Verify CUDA is installed**

```bash
# Should show your GPU and driver version
nvidia-smi

# Should print: CUDA available: True
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

> **CUDA version matching:** The CUDA Toolkit version must match the PyTorch CUDA build you installed in step 2. The gsplat precompiled wheels require **CUDA 12.4** (`cu124`) and **PyTorch 2.4** (`pt24`) — install CUDA Toolkit 12.4 to match.

### 4. Install gsplat

gsplat provides the CUDA-accelerated 3DGS rasteriser used for training:

```bash
pip install gsplat --index-url https://docs.gsplat.studio/whl/pt24cu124
```

If you see build errors during `pip install gsplat`, it usually means the CUDA Toolkit is missing or the version does not match PyTorch. See the [gsplat installation guide](https://docs.gsplat.studio/main/installation/installation.html) for troubleshooting.

### 5. Install FFmpeg

FFmpeg must be on your system PATH.

- **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add `bin/` to PATH, or use `winget install ffmpeg`.
- **Linux**: `sudo apt install ffmpeg`
- **macOS**: `brew install ffmpeg`

### 6. Install COLMAP 4.x

GLOMAP (the default SfM method) requires **COLMAP 4.0 or later**.

- Download from [github.com/colmap/colmap/releases](https://github.com/colmap/colmap/releases)
- Add the binary to PATH, **or** place it at `./bin/colmap` (or `./bin/colmap.exe` on Windows) for a project-local install.

### 7. Install gsforge

```bash
git clone https://github.com/your-org/gsforge.git
cd gsforge
pip install -e ".[dev]"
```

Verify the install:

```bash
gsforge --help
```

---

## Full example workflow — Ignatius test video

The `tests/test_data/ignatius.mp4` file is included for testing. Here is the complete workflow:

### Option A — One command (recommended)

```bash
# Create a project first
gsforge init-project --name Ignatius

# Run everything: ingest → SfM → train
gsforge run-all --video tests/test_data/ignatius.mp4 --project Ignatius.gsproject
```

### Option B — Step by step

```bash
# 1. Create the project directory
gsforge init-project --name Ignatius

# 2. Extract frames from the video
#    Default: 5 fps, max 400 frames, full resolution
gsforge ingest --video tests/test_data/ignatius.mp4 --project Ignatius.gsproject

# 3. Run GLOMAP (global SfM) — fast and robust
gsforge sfm --project Ignatius.gsproject

# 4. Train the 3DGS model
gsforge train --project Ignatius.gsproject --iterations 15000

# 5. Check project status at any time
gsforge info --project Ignatius.gsproject
```

The final model will be at `Ignatius.gsproject/models/final_scene.ply`.

---

## Training in depth

### How it works

`gsforge train` runs the full 3D Gaussian Splatting training pipeline:

1. **Loads the COLMAP reconstruction** from `sfm/sparse/0/` — cameras, image poses, and the sparse 3D point cloud.
2. **Initialises Gaussians** at the sparse point positions, seeding colour from COLMAP's per-point RGB values.
3. **Optimises** using gsplat's CUDA rasteriser with an Adam optimiser and per-parameter learning rates.
4. **Densifies and prunes** Gaussians adaptively (following the Inria 3DGS paper):
   - _Clone_ under-reconstructed fine details (high gradient, small scale)
   - _Split_ over-reconstructed coarse areas (high gradient, large scale)
   - _Prune_ nearly-transparent Gaussians (opacity < 0.005)
5. **Saves checkpoints** to `models/checkpoints/` every `--preview-every` iterations.
6. **Saves preview renders** to `renders/` every `--preview-every` iterations so you can monitor quality without waiting for training to finish.
7. **Saves the final model** as `models/final_scene.ply`.

### Training options

```bash
gsforge train [--project DIR] [--backend gsplat] [--iterations N] [--preview-every N]
```

| Option            | Default  | Description                                                                                                                         |
| ----------------- | -------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `--iterations`    | `15000`  | Total training iterations. 15k is a good balance of quality vs. time (~10–20 min on a modern GPU). Raise to 30k for higher quality. |
| `--preview-every` | `500`    | Save a checkpoint and preview render every N iterations. Lower values give more feedback but slightly slower training.              |
| `--backend`       | `gsplat` | Training backend. Currently only `gsplat` is supported. Brush and Inria backends are planned.                                       |

### GPU requirements

| VRAM     | Recommendation                                                            |
| -------- | ------------------------------------------------------------------------- |
| ≥ 16 GB  | Ideal. Can train at full resolution with 30k+ iterations.                 |
| 8–16 GB  | Good. Default settings (15k iterations) work well.                        |
| 4–8 GB   | Reduce `--iterations` to 7500 and consider `--downscale 2` during ingest. |
| CPU only | Supported but very slow (hours). Not recommended for production.          |

gsforge automatically detects and uses the best available GPU. It prints the device name and VRAM at the start of training.

### Monitoring training progress

Preview renders are saved to `renders/` every `--preview-every` iterations:

```
Ignatius.gsproject/
└── renders/
    ├── preview_000500.png
    ├── preview_001000.png
    ├── preview_001500.png
    └── ...
```

Open these in any image viewer to check reconstruction quality early. If the previews look wrong at iteration 1000–2000, it's worth aborting and checking your SfM reconstruction rather than waiting for the full 15k iterations.

### Checkpoints

Checkpoints are saved to `models/checkpoints/` at the same interval as previews:

```
Ignatius.gsproject/
└── models/
    ├── checkpoints/
    │   ├── ckpt_000500.pth
    │   ├── ckpt_001000.pth
    │   └── ...
    └── final_scene.ply
```

Each `.pth` file contains all Gaussian parameters (means, scales, rotations, opacities, spherical harmonics) and the current loss value.

### Output format

The final `models/final_scene.ply` uses the standard Inria 3DGS binary PLY format. It is compatible with:

- **[SuperSplat](https://supersplat.dev)** — web-based 3DGS viewer (drag and drop)
- **[KIRI Engine](https://www.kiriengine.com)** viewer
- **[Luma AI](https://lumalabs.ai)** viewer
- **nerfstudio** — `ns-viewer --load-config ...`
- Any custom gsplat/3DGS renderer

### CUDA out of memory

If training fails with a CUDA OOM error, gsforge prints specific suggestions:

```
CUDA ran out of memory during training.
  Suggestions:
    - Reduce --iterations (e.g. 7500 instead of 15000)
    - Use a GPU with more VRAM (>=8 GB recommended)
    - Close other GPU-intensive applications
```

You can also reduce the input resolution during ingest:

```bash
gsforge ingest --video footage.mp4 --downscale 2  # half resolution = ~4x less VRAM
```

---

## Why 5 fps and 400 frames max?

These defaults are tuned for virtual production footage:

**5 fps (default `--target-fps`)**

A typical VP shoot records at 24–60 fps. Consecutive frames at 24 fps are separated by only 42 ms — the camera barely moves. COLMAP/GLOMAP only needs 60–80% overlap between adjacent frames to reconstruct reliably. At 5 fps you get one frame every 200 ms, which gives excellent overlap for any camera speed used in VP (slow dolly, crane, handheld walk). Extracting every frame would give 5–12× more images with almost no additional reconstruction quality, but 5–12× longer feature extraction time.

**400 frames max (default `--max-frames`)**

Even at 5 fps, a 90-second clip produces 450 frames. GLOMAP's global bundle adjustment memory usage grows quadratically with image count. 300–400 images is the sweet spot: fast enough to run on a workstation in 5–15 minutes, dense enough for a high-quality reconstruction of any VP scene. For very large or complex scenes (e.g. full LED volume walk-through), raise this to 600–800.

**Adjusting for your footage**

```bash
# Slow camera, short clip — can use more frames
gsforge ingest --video footage.mp4 --target-fps 8 --max-frames 600

# Fast camera, long clip — use fewer, more spread-out frames
gsforge ingest --video footage.mp4 --target-fps 3 --max-frames 250

# Downscale to half resolution for faster SfM and less VRAM during training
gsforge ingest --video footage.mp4 --downscale 2
```

---

## Using gsforge as a GLOMAP-only tool + export to LichtFeld Studio

gsforge is designed to be modular. You can use it purely as a fast GLOMAP runner and then hand off the reconstruction to any other tool:

### Step 1 — Run GLOMAP

```bash
gsforge init-project --name MyScene
gsforge ingest --video footage.mp4 --project MyScene.gsproject
gsforge sfm --project MyScene.gsproject --method glomap
```

### Step 2 — Export a clean COLMAP folder

```bash
gsforge export-colmap --project MyScene.gsproject --output /mnt/exports/MyScene_colmap
```

This creates:

```
MyScene_colmap/
├── images/       ← all extracted frames
└── sparse/
    └── 0/        ← cameras.bin, images.bin, points3D.bin
```

### Step 3 — Open in LichtFeld Studio

In LichtFeld Studio: **File → Import COLMAP Project** → select `MyScene_colmap/`.

The same export folder works with:

- **nerfstudio**: `ns-train gaussian-splatting --data MyScene_colmap`
- **3DGS original**: `python train.py -s MyScene_colmap`
- **COLMAP GUI**: File → Import Model → `MyScene_colmap/sparse/0`

### Importing an existing reconstruction

If you already have a COLMAP reconstruction from another tool:

```bash
gsforge init-project --name MyScene
# Copy your frames into MyScene.gsproject/preprocess/ manually, then:
gsforge import-colmap --source /path/to/existing/sparse/0 --project MyScene.gsproject
gsforge train --project MyScene.gsproject
```

---

## Project folder structure

```
MyScene.gsproject/
├── project.json          ← pipeline metadata (version, status, counts, paths)
├── source/               ← original video file (copied on ingest)
├── preprocess/           ← extracted frames: frame_000001.png …
├── sfm/
│   ├── database.db       ← COLMAP feature database
│   └── sparse/
│       └── 0/            ← cameras.bin, images.bin, points3D.bin
├── models/
│   ├── checkpoints/      ← ckpt_000500.pth, ckpt_001000.pth, …
│   └── final_scene.ply   ← final trained 3DGS model
├── renders/              ← preview_000500.png, preview_001000.png, …
└── logs/                 ← per-step log files
```

The `project.json` file tracks the full pipeline state:

```json
{
	"version": "1.0",
	"name": "MyScene",
	"sfm_method": "glomap",
	"sfm_status": "completed",
	"camera_count": 312,
	"training_status": "completed",
	"final_ply": "models/final_scene.ply",
	"last_iteration": 15000
}
```

---

## All CLI commands

```
gsforge init-project   --name NAME [--project DIR]
gsforge ingest         --video PATH [--project DIR] [--target-fps N] [--max-frames N] [--downscale N]
gsforge sfm            [--project DIR] [--method glomap|colmap]
gsforge import-colmap  --source PATH [--project DIR]
gsforge export-colmap  [--project DIR] [--output DIR]
gsforge train          [--project DIR] [--backend gsplat] [--iterations N] [--preview-every N]
gsforge info           [--project DIR]
gsforge run-all        --video PATH [--project DIR] [--target-fps N] [--max-frames N] [--downscale N] [--method STR] [--iterations N]
```

For any command: `gsforge COMMAND --help`

---

## Architecture overview

```
gsforge/
├── cli.py        ← Typer CLI entrypoint — thin wrappers, no business logic
├── project.py    ← GSProject: smart folder + project.json management
├── ingest.py     ← FFmpeg frame extraction
├── sfm.py        ← COLMAP/GLOMAP runner + import/export helpers
├── train.py      ← 3DGS training: BaseTrainer ABC + GsplatTrainer
└── utils.py      ← Rich logging, tqdm progress, path helpers, constants
```

**`train.py` internals:**

| Component              | Purpose                                                                                     |
| ---------------------- | ------------------------------------------------------------------------------------------- |
| `BaseTrainer` (ABC)    | Common interface for all backends. Subclass to add Brush, Inria, nerfstudio.                |
| `GsplatTrainer`        | Full gsplat training pipeline.                                                              |
| `load_colmap_data()`   | Reads `cameras.bin/txt`, `images.bin/txt`, `points3D.bin/txt`. Binary-first, text fallback. |
| `_train_with_gsplat()` | Core loop: rasterise → L1 loss → backprop → densify/prune → checkpoint.                     |
| `_densify_and_prune()` | Inria adaptive density control: clone, split, prune.                                        |
| `_save_ply()`          | Writes standard Inria 3DGS binary PLY (compatible with all viewers).                        |
| `run_training()`       | Public CLI entry point. Updates `project.json` on success or failure.                       |

---

## Roadmap

- [x] CLI foundation (init, ingest, sfm, export, info)
- [x] `train.py` — gsplat training with adaptive densification
- [x] Preview renders during training
- [x] Checkpoint saving
- [x] Standard `.ply` export (Inria format)
- [ ] Brush backend support
- [ ] Inria 3DGS backend support
- [ ] Frame quality filtering (blur detection, exposure check)
- [ ] Multi-camera rig support
- [ ] DearPyGui desktop GUI
- [ ] Gradio web UI

---

## License

MIT — see [LICENSE](LICENSE).
