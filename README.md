# Pantheon (Effort) Reproduction Package

> 🎉 **This paper has been accepted to ACSAC 2026** (Annual Computer Security Applications Conference).

Official code for the paper:

**No One Model to Rule Them All: A Quality-Adaptive Ensemble Defense against Quality-Shift Adversaries in Deepfake Detection**

Eunbi Hwang, Hyunseok Lee, Taekyoung Kwon\*
Graduate School of Information, Yonsei University, Seoul, South Korea

\* Corresponding author

A standalone working directory that reproduces the Table 5 results of the original Pantheon (Effort) and runs the QCN → CR-FIQA(L) → Effort → Pantheon inference pipeline on new face images.

The original `/home/compu/eunbi/iqa/0.module/analysis/qcn_crfiqa`, QCN, CR-FIQA, and Effort code were left unmodified. The required materials were **copied, and the runnable code and environment were organized entirely within this folder**. Data and weights are real file copies, not symbolic links.

## Quick Start

The runtime environment folders are not uploaded to GitHub. On Linux x86_64 with Python 3.11–3.13 installed, running `bash setup.sh` creates an isolated environment and installs and verifies the required packages. If an NVIDIA GPU is present, it selects a PyTorch build matching the supported runtime. Copies of the data and weights are also prepared on this server.

```bash
cd Pantheon

# Automatic environment setup: installs both the CPU reproduction and GPU inference environments when an NVIDIA GPU is detected
bash setup.sh

# CPU demo: train the gate on FF++ and reproduce the Celeb-DF-v1 AUC
bash scripts/demo.sh

# Full Table 5 reproduction: 3 FF++ compression levels + 10 cross-datasets
bash scripts/reproduce.sh

# Calling the Python entry point directly
environment/cpu/bin/python main.py --all-models

# 8 images: real model inference from quality measurement to Pantheon prediction
bash scripts/demo_images.sh

# Verify that original files keep their size and modification time, and that results match
environment/cpu/bin/python scripts/verify.py --check-originals
```

Full reproduction reads stored scores for millions of frames, so it takes longer than the demo. The default is 4 CPU threads, adjustable via the `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and `MKL_NUM_THREADS` environment variables.

## Layout

```text
pantheon/
  core.py          Active numerical functions from the original; duplicate definitions and commented-out experiments removed
  pipeline.py      Gate training, full benchmark evaluation, tables, figures, and reference comparison
  infer.py         QCN, CR-FIQA, Effort, and Pantheon inference on images
  quality.py       Quality normalization based on the original FF++ statistics
  prepare.py       Quality score normalization and merging with model prediction CSVs
  score.py         Apply Pantheon to existing/new merged CSVs
data/
  merged/          Copies of the 11 merged CSVs used for evaluation
  standardized/    11 standardized quality CSVs and normalization_params.json
  quality_raw/     Copies of the CR-FIQA+QCN quality CSVs before standardization
weights/
  qcn_spaq.pth     QCN weights trained on SPAQ
  crfiqa_l.pth     CR-FIQA(L), iresnet100
  effort/          c23, q1–q4, and raw, c40, all weights for comparison
vendor/            Required QCN and CR-FIQA model implementations and original project descriptions
demo/
  input.csv        Paths, labels, and video IDs for 8 real face crops
  images/          Copied image files, preserving the original video/frame structure
reference/         Original result CSVs and figures, CLIP config, copy sources and SHA-256
archive/           Original analysis, preprocessing, and inference code before modification, for reference
environment/
  cpu/             Separately installed CPU reproduction environment
  gpu/             Isolated GPU inference environment prepared by setup
  gpu.legacy-*/    Archived copy of the old copied environment (excluded from Git)
  *.freeze.txt     Record of installed package versions
outputs/           New run results and verification records
```

`archive/` is kept to preserve the originals. The execution entry points are `scripts/` and `pantheon/`. Absolute paths, intermediate experiment comments, unused forced UI termination, and forced GPU index assignment left in the original were removed from the new execution path.

## Reproduction Reference and Results

The reference file is `reference/ALL_results_frame_video_auc_with_deltas.csv`. The `video_auc` for `model=ens_adapt` is the Pantheon value, in %.

| Evaluation dataset | Pantheon video AUC (%) |
|---|---:|
| FF++ raw | 100.00 |
| FF++ c23 | 100.00 |
| FF++ c40 | 99.79 |
| Celeb-DF-v1 | 98.26 |
| Celeb-DF-v2 | 98.15 |
| Celeb-DF-v3 / CDF++ | 93.65 |
| DFDC | 91.32 |
| DFo | 99.54 |
| Wilddeepfake | 89.82 |
| FFIW | 90.87 |
| KoDF | 93.38 |
| DF40 | 82.63 |
| DF-eval | 68.50 |
| Mean over 10 cross-datasets | **90.61** |

The mean of the three FF++ values is 99.93%, and the unrounded cross-dataset mean is 90.61374343684233%. Other methods from Table 5 (DML, etc.) are not included in this package.

Main files produced by the reproduction code:

- `outputs/reproduction/results.csv`: frame/video AUC for Pantheon and the Effort comparison models
- `outputs/reproduction/pantheon_table5.csv`: the 13 Pantheon evaluation values
- `outputs/reproduction/summary.json`: cross-dataset mean
- `outputs/reproduction/table5.png`: regenerated results table
- `outputs/reproduction/reference_comparison.csv`: differences from the original results
- `outputs/reproduction/calibration/pantheon.joblib`: trained gate and RBF centers

During evaluation, frame scores are averaged per video, and videos with mixed ground-truth labels are excluded. The FFIW real/fake video split, the Wilddeepfake collision-avoidance rule, and the DF40 rule excluding non-video generators are the same as in the original. The run fails if the comparison against the reference exceeds a tolerance of `1e-8` percentage points. Since the demo does not include the full data, the demo's mean is the mean over the demo datasets only.

## Pantheon Configuration

- RBF centers for Q1–Q4 computed from the FF++ standardized quality CSV, sigma=1.0
- Gate trained on all FF++ real frames and c40 fake frames
- Gate training uses high-confidence frames whose c23 score lies outside `(0.05, 0.95)`
- Gate inputs: `conf, q_std, q_margin, q_range`
- HistGradientBoosting: depth=4, learning_rate=0.05, max_iter=400, seed=0
- RBF soft score used in the low-confidence region
- Elsewhere, the global/soft mixing ratio is adjusted according to the gate's error probability
- lambda range 0.8→0.2, error-probability thresholds 0.05 and 0.40

The normalization parameters are preserved in `data/standardized/normalization_params.json`. These same parameters are applied to new inputs.

## Running on New Images

The input is **already-extracted face crop images**. Face detection and alignment from the source video are not included. The bundled demo also consists of real face crops copied from the original evaluation.

The input CSV requires a header, and `path` is either relative to the CSV file's location or absolute.

```csv
path,label,video_id
images/video_a/000.png,0,video_a
images/video_b/000.png,1,video_b
```

`label` is 0=real, 1=fake. It can be omitted for inference only; input labels are not used when computing inference scores. If `video_id` is provided, per-video mean scores are also saved.

```bash
bash scripts/demo_images.sh --input /absolute/path/input.csv \
  --output outputs/my_images --device cuda:0

# To run all 8 Effort models, including raw/c40/all for comparison
bash scripts/demo_images.sh --all-models

# The same inference code can also run on CPU. It is slow because the models are large.
bash scripts/demo_images.sh --device cpu
```

The outputs are `quality_raw.csv`, `merged_predictions.csv`, and `frame_predictions.csv`, where `ens_adapt` is the final fake score. The default decision threshold is 0.5. If video IDs are provided, `video_predictions.csv` is also generated.

QCN uses the SPAQ checkpoint, RGB/Resize(384)/ImageNet normalization, averaging of features from the original and horizontally flipped image, and score-pivot interpolation. CR-FIQA uses BGR, 112×112 input, as in the original. Effort applies RGB, bicubic 224×224, and CLIP normalization. The SVD main/residual weights in the Effort checkpoint are combined to form the same dense operations, so no separate CLIP model download is required. All checkpoints are verified with strict loading.

**Table reproduction from stored CSVs and re-inference from images are distinct.** Image re-inference may differ slightly from the original stored scores depending on CUDA operations, batch size, and mixed precision. By default, QCN on GPU uses float16 autocast as in the original, which can be disabled with `--no-amp`. The demo's original scores are preserved in `reference/demo_stored_scores.csv`.

## Merging from Quality and Prediction CSVs

Your own quality CSV uses the columns `path,cr_fiqa,qcn`, and the prediction CSV uses `path,label,c23,q1,q2,q3,q4`. The column names `rel_path` and `image_path` are also accepted in the quality CSV. The raw/c40/all prediction columns are optional. Paths must be identical keys in both files; the `_raw/_c23/_c40` suffix on the first FF++ dataset directory name is normalized as in the original. Duplicate or missing keys and NaNs are raised as errors rather than silently skipped.

```bash
environment/cpu/bin/python -m pantheon.prepare \
  --quality outputs/image_demo/quality_raw.csv \
  --predictions outputs/image_demo/merged_predictions.csv \
  --output outputs/prepared/merged_demo.csv

environment/cpu/bin/python -m pantheon.score \
  --input outputs/prepared/merged_demo.csv \
  --output outputs/prepared/pantheon_predictions.csv
```

The predictions corresponding to the per-model `frame_predictions.csv` from the original inference are already included as per-model columns in `data/merged/`. This folder does not replicate other experiments or the full original datasets. It contains all CSVs needed for Table 5 reproduction, the weights and model code needed for inference, and the demo images.

## Automatic Environment Setup

There is no need to distribute the environment directory or copy a Conda environment. Linux x86_64, Python **3.11–3.13**, and `venv` are required, and setup automatically finds a compatible installed Python. It does not modify the system Python or the existing QCN/CR-FIQA projects, and installs only into an isolated environment inside the project.

```bash
# Automatic selection: CPU + GPU if a GPU is available, otherwise the CPU table-reproduction environment
bash setup.sh

# Install only the lightweight environment needed to reproduce the table from stored CSVs
bash setup.sh cpu

# Install the NVIDIA GPU image-inference environment and the CPU table-reproduction environment
bash setup.sh gpu

# Install CPU PyTorch to run the image models without an NVIDIA GPU
bash setup.sh inference-cpu

# Check installation status only. No package changes or downloads
bash setup.sh --check
```

Automatic setup performs the following steps:

1. Find a compatible Python and create `environment/cpu`, and `environment/gpu` if needed
2. Install pinned versions of NumPy/Pandas/scikit-learn, etc. from `requirements.txt`
3. Select the cu130/cu128/cu126 PyTorch wheel based on the supported CUDA version reported by `nvidia-smi`
4. Install the packages required by QCN, CR-FIQA, and Effort
5. Verify the installation with `pip check`, imports of the actual model modules, and small CPU/GPU operations
6. Write an installation success marker and record `environment/*.freeze.txt`

If a previously copied Conda environment exists, it is archived as `environment/gpu.legacy-*` and a new, conflict-free venv is created. The archive is excluded from GitHub uploads.

If the environment exists and the requirements are unchanged, downloads are skipped and only verification runs. If installation fails partway, no success marker is left, so rerunning the same command resumes the installation. `--force` reruns the package installation step.

The GPU environment's directory is named `gpu`, but in `inference-cpu` mode it contains CPU PyTorch. No separate environment activation is needed. The demo scripts also run setup first if the environment does not exist.

```bash
# Specify Python directly
PYTHON=python3.12 bash setup.sh

# Specify the CUDA backend directly
bash setup.sh gpu --torch-backend cu128

# Change the environment location
bash setup.sh --env-root /path/to/pantheon-envs
PANTHEON_ENV_ROOT=/path/to/pantheon-envs bash scripts/demo_images.sh
```

GPU selection is based on the maximum CUDA version supported as reported by the driver. If actual CUDA operations fail, setup also fails and prints the cause. The host NVIDIA driver must be installed in the operating system. The CUDA Toolkit and Conda do not need to be installed separately. For environments with older drivers, update the driver or use `inference-cpu` mode.

Package definitions are split by role:

- `requirements.txt`: pinned versions for CPU table reproduction
- `requirements-gpu.txt`: QCN, CR-FIQA, and Effort inference packages
- `requirements-torch-cu130.txt`, `requirements-torch-cu128.txt`, `requirements-torch-cu126.txt`: PyTorch 2.9.1 / torchvision 0.24.1 per CUDA version
- `requirements-torch-cpu.txt`: CPU PyTorch / torchvision

PyTorch wheels are installed first from the official index, and the remaining packages from PyPI, following the [official PyTorch installation combinations](https://pytorch.org/get-started/previous-versions/#v291). Run `setup.sh` rather than installing `requirements-gpu.txt` on its own.

### GitHub Distribution, Data, and Weights

`.gitignore` excludes `environment/cpu/`, `environment/gpu/`, `data/`, `weights/`, and `outputs/`. Only the code, setup scripts, requirements, README, demo images, and original result tables are uploaded to GitHub. Local files are not deleted.

setup **installs the runtime environment automatically**. The evaluation CSVs and trained weights are not Python packages and must be distributed separately. Since no public download location has been decided yet, nothing is downloaded from arbitrary URLs. Place the distributed files in the following structure:

```text
data/merged/merged_{dataset}.csv        Full table reproduction: the 11 files in DATASETS
data/standardized/FaceForensics++_standardized.csv
data/standardized/normalization_params.json  Image quality normalization
weights/qcn_spaq.pth
weights/crfiqa_l.pth
weights/effort/c23.pth
weights/effort/q1.pth ~ q4.pth
weights/effort/raw.pth, c40.pth, all.pth  Additionally required with --all-models
```

The CPU demo requires the FF++ and Celeb-DF-v1 merged CSVs and the FF++ standardized CSV. The image demo also needs the two FF++ CSVs for the initial gate training. If you already have a trained `pantheon.joblib`, pass it with `--calibration /path/to/pantheon.joblib` to run inference without the FF++ CSVs. The quality normalization JSON and model weights are still required in that case. Missing default files are listed during setup verification, separately from whether environment installation completed.

## Copy Sources and Integrity

`reference/copy_manifest.json` records the original paths, copy destinations, original sizes and modification times, SHA-256 of the copies, and whether each was modified. Demo images are linked to the original evaluation frames via `source_key` in `demo/input.csv`. The original Pantheon analysis code, quality inference code, and Effort detector are preserved in `archive/`.

```bash
environment/cpu/bin/python scripts/verify.py --hashes --check-originals
```

This hashes all weights, so it may take a while. Omit `--check-originals` to verify without access to the originals. The environment directories are not covered by the file hash manifest; installed package records are preserved separately.

## Verification Completed in This Folder

- Full evaluation run in the isolated CPU environment: all 104 video AUCs match the original CSV, with a maximum difference of about `1.42e-14` percentage points
- Maximum difference of 0 across the 13 Pantheon values; cross-dataset mean 90.61374343684233%
- QCN, CR-FIQA, all 8 Effort checkpoints, and Pantheon run successfully on 8 real images in the copied GPU environment
- Round trip of quality CSV normalization/merging → separate Pantheon inference produces matching results
- `pip check` passes in both CPU and GPU environments; unused conflicting packages removed from the GPU copy
- Original files that were copied confirmed to retain their size and modification time; SHA-256 of copies verified

Verification details are saved in `outputs/validation.json`, `outputs/verification.json`, `outputs/integrity_verification.log`, and each run's logs.

Additional verification of automatic setup: the CPU demo, CUDA operations, and inference of QCN, CR-FIQA, and 8 Effort models on 8 images were confirmed starting from an empty environment. Pantheon scores in the new PyTorch 2.9.1 environment matched those of the previous environment. Results are recorded in `outputs/setup_validation.json`.

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{hwang2026pantheon,
  title     = {No One Model to Rule Them All: A Quality-Adaptive Ensemble Defense against Quality-Shift Adversaries in Deepfake Detection},
  author    = {Hwang, Eunbi and Lee, Hyunseok and Kwon, Taekyoung},
  booktitle = {Proceedings of the Annual Computer Security Applications Conference (ACSAC)},
  year      = {2026}
}
```
