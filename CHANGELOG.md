# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Comprehensive project documentation in `README.md`, including setup, data layout, training usage, outputs, troubleshooting, and contribution guidelines.
- `prepare_urbansound8k_binary.py` utility to build no-leakage UrbanSound8K binary splits using folds `1-8` (train), `9` (val), and `10` (test), with configurable negative sampling ratio.
- `edge_infer.py` for local sensor-style streaming simulation with rolling windows, thresholding, N-of-M voting, cooldown, and optional JSON event output.

### Changed

- README now includes a Table of Contents directly under the project title.
- Training defaults now prioritize generalization: stronger augmentation (time shift + SpecAugment), higher regularization, smaller CNN capacity, and early stopping/checkpointing based on `val_loss`.

## [0.0.0] - 2026-02-17

### Added

- Initial project scaffold for Aquil Safety audio training pipeline.
