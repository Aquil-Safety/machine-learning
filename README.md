# Aquili Safety Audio ML

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Data Layout](#data-layout)
- [Supported Audio Formats](#supported-audio-formats)
- [Training](#training)
- [Output Artifacts](#output-artifacts)
- [Plot Training Curves](#plot-training-curves)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [Changelog](#changelog)

## Overview
This repository trains a compact TensorFlow/Keras CNN to classify short audio clips as:
- `gunshot` (label `1`)
- `not_gunshot` (label `0`)

The training pipeline:
- Loads audio clips and converts to mono at a fixed sample rate
- Builds normalized Mel spectrogram features
- Trains a small 2D CNN
- Exports evaluation reports and optional TFLite models

## Features
- Binary audio classification for gunshot detection
- Mel spectrogram feature extraction with normalization
- Built-in lightweight augmentation (gain + noise)
- Class weighting for imbalance handling
- Validation and optional test-set reporting
- SavedModel export and optional TFLite float/int8 export

## Repository Structure
```text
.
├── train_aquil_audio_tf.py   # Training and export pipeline
├── plot.py                   # Minimal plotting helper for train_log.csv
├── requirements.txt
├── CHANGELOG.md
├── version.txt
├── data/                     # Local dataset (ignored by git)
│   ├── train/
│   ├── val/
│   └── test/                 # Optional
└── artifacts/                # Training outputs (ignored by git)
```

## Requirements
- Python 3.10+ recommended
- Packages in `requirements.txt`:
  - `tensorflow`
  - `librosa`
  - `soundfile`
  - `numpy`
  - `scikit-learn`

Optional for plotting:
- `pandas`
- `matplotlib`

## Quick Start
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

If you use `plot.py`, also install:
```bash
pip install pandas matplotlib
```

## Data Layout
You must provide separate training and validation splits.

```text
data/
  train/
    gunshot/
      *.wav|*.mp3|*.flac
    not_gunshot/
      *.wav|*.mp3|*.flac
  val/
    gunshot/
      *.wav|*.mp3|*.flac
    not_gunshot/
      *.wav|*.mp3|*.flac
  test/                        # optional
    gunshot/
    not_gunshot/
```

Notes:
- `val` is not the same as `train`; keep files separate.
- Keep both classes present in each split.
- Very small datasets will overfit and produce unstable metrics.

## Supported Audio Formats
`train_aquil_audio_tf.py` currently supports:
- `.wav`
- `.wave`
- `.mp3`
- `.flac`

## Training
Basic run:
```bash
python3 train_aquil_audio_tf.py --data_dir ./data --out_dir ./artifacts
```

Useful flags:
```bash
python3 train_aquil_audio_tf.py \
  --data_dir ./data \
  --out_dir ./artifacts \
  --epochs 40 \
  --batch_size 64 \
  --lr 1e-3 \
  --seed 42 \
  --cache_features \
  --export_tflite
```

Flags:
- `--cache_features` enables cached Mel `.npy` files
- `--no_cache_features` disables cache
- `--export_tflite` enables TFLite export
- `--no_export_tflite` disables TFLite export

## Output Artifacts
After training, outputs are written to `artifacts/`:
- `config.json`
- `train_log.csv`
- `checkpoints/best.weights.h5`
- `model.h5`
- `saved_model/` (exported model directory)
- `val_eval_threshold_0.5.json`
- `val_eval_threshold_0.95.json`
- `test_eval_threshold_0.95.json` (if test split exists)
- `deployment_hints.json`
- `model_float.tflite` and `model_int8.tflite` (when enabled)

## Plot Training Curves
`plot.py` reads `artifacts/train_log.csv` and plots training vs validation loss:
```bash
python3 plot.py
```

Interpretation shortcut:
- Train loss down + val loss flat/up: likely overfitting
- Both losses high/flat: likely underfitting

## Troubleshooting
- `No training/validation data found`:
  - Ensure `data/train/...` and `data/val/...` both exist and contain files.
- Keras save path error:
  - The script uses `model.export()` for SavedModel and `model.save(...h5)` for H5.
- Metrics look random:
  - Increase dataset size and class diversity before tuning model architecture.

## Contributing
1. Create a branch for your change.
2. Keep changes focused and documented.
3. Update `README.md` for user-facing behavior changes.
4. Update `CHANGELOG.md` using Keep a Changelog categories:
   - `Added`
   - `Changed`
   - `Deprecated`
   - `Removed`
   - `Fixed`
   - `Security`

## Changelog
This project follows Keep a Changelog:
- Changelog file: `CHANGELOG.md`
- Format: <https://keepachangelog.com/en/1.1.0/>
- Versioning: Semantic Versioning (<https://semver.org/>)
