"""Training pipeline for Aquil Safety audio classification."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator, List, Tuple

import numpy as np

# TensorFlow can be heavy to import; do it after numpy
import tensorflow as tf

# Audio
import librosa
# Metrics
from sklearn.metrics import classification_report, confusion_matrix


# ----------------------------
# Config
# ----------------------------

@dataclass(frozen=True)
class TrainConfig:
    """Immutable training configuration for model, data, and export settings."""
    # Data
    data_dir: str
    out_dir: str
    seed: int = 42

    # Audio / features
    sample_rate: int = 16000
    clip_seconds: float = 1.0
    n_fft: int = 1024
    hop_length: int = 512
    n_mels: int = 40
    fmin: int = 20
    fmax: int = 8000

    # Training
    batch_size: int = 64
    epochs: int = 40
    lr: float = 1e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.0

    # Regularization
    dropout: float = 0.25

    # Performance
    num_workers: int = 4
    cache_features: bool = True   # caches mel arrays as .npy next to source audio
    prefetch: int = 2

    # Thresholding / class imbalance
    class_weight: bool = True  # auto compute class weights from train set

    # TFLite export
    export_tflite: bool = True
    export_int8: bool = True
    rep_data_samples: int = 200  # number of samples for representative dataset


# ----------------------------
# Utilities
# ----------------------------

def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


AUDIO_EXTENSIONS = {".wav", ".wave", ".mp3", ".flac"}


def list_audio_files(class_dir: Path) -> List[Path]:
    if not class_dir.exists():
        return []
    return sorted(
        [p for p in class_dir.rglob("*") if p.suffix.lower() in AUDIO_EXTENSIONS]
    )


def human_time() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def load_audio_mono(path: str, sr: int) -> np.ndarray:
    """
    Loads audio file as mono float32 in range [-1, 1] at target sample rate.
    Uses librosa for resampling and mono conversion.
    """
    y, _sr = librosa.load(path, sr=sr, mono=True)
    if y.dtype != np.float32:
        y = y.astype(np.float32)
    return y


def fix_length(y: np.ndarray, target_len: int) -> np.ndarray:
    """Pad or trim to target length."""
    if len(y) < target_len:
        pad = target_len - len(y)
        y = np.pad(y, (0, pad), mode="constant")
    elif len(y) > target_len:
        y = y[:target_len]
    return y


def random_gain(y: np.ndarray, min_db: float = -6.0, max_db: float = 6.0) -> np.ndarray:
    """Random gain augmentation in dB."""
    db = np.random.uniform(min_db, max_db)
    gain = 10 ** (db / 20)
    y = y * gain
    return np.clip(y, -1.0, 1.0)


def add_noise(y: np.ndarray, noise_level: float = 0.005) -> np.ndarray:
    """Add small white noise."""
    if noise_level <= 0:
        return y
    n = np.random.randn(len(y)).astype(np.float32) * noise_level
    y2 = y + n
    return np.clip(y2, -1.0, 1.0)


def mel_spectrogram_db(
    y: np.ndarray,
    sr: int,
    n_fft: int,
    hop_length: int,
    n_mels: int,
    fmin: int,
    fmax: int,
) -> np.ndarray:
    """
    Returns Mel spectrogram in dB, float32.
    Shape: (n_mels, time_steps)
    """
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db.astype(np.float32)


def normalize_mel(mel_db: np.ndarray) -> np.ndarray:
    """
    Normalize spectrogram to approx [0, 1] range using min/max clipping.
    dB values typically in [-80, 0].
    """
    mel_db = np.clip(mel_db, -80.0, 0.0)
    mel_norm = (mel_db + 80.0) / 80.0
    return mel_norm.astype(np.float32)


def cached_feature_path(audio_path: Path, cfg: TrainConfig) -> Path:
    # Cache in a hidden file next to source audio
    # Name includes key feature settings in case you change them later.
    key = (
        f"sr{cfg.sample_rate}_sec{cfg.clip_seconds}_m{cfg.n_mels}_"
        f"nfft{cfg.n_fft}_hop{cfg.hop_length}_f{cfg.fmin}-{cfg.fmax}"
    )
    return audio_path.with_suffix(f".{key}.mel.npy")


def compute_or_load_mel(audio_path: Path, cfg: TrainConfig, augment: bool) -> np.ndarray:
    cache_path = cached_feature_path(audio_path, cfg)
    if cfg.cache_features and cache_path.exists() and not augment:
        mel = np.load(cache_path)
        return mel.astype(np.float32)

    target_len = int(cfg.sample_rate * cfg.clip_seconds)
    y = load_audio_mono(str(audio_path), cfg.sample_rate)
    y = fix_length(y, target_len)

    # Augment only for training
    if augment:
        y = random_gain(y)
        # light noise helps robustness; tune later
        y = add_noise(y, noise_level=0.003)

    mel_db = mel_spectrogram_db(
        y=y,
        sr=cfg.sample_rate,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        fmin=cfg.fmin,
        fmax=cfg.fmax,
    )
    mel = normalize_mel(mel_db)

    # Cache only non-augmented versions
    if cfg.cache_features and not augment:
        np.save(cache_path, mel)

    return mel.astype(np.float32)


# ----------------------------
# Dataset building
# ----------------------------

def build_file_list(split_dir: Path) -> Tuple[List[Path], List[int]]:
    """
    Returns list of source audio paths and labels for a given split directory.
    Labels:
      gunshot      -> 1
      not_gunshot  -> 0
    """
    gun = list_audio_files(split_dir / "gunshot")
    neg = list_audio_files(split_dir / "not_gunshot")

    features = gun + neg
    y = [1] * len(gun) + [0] * len(neg)
    return features, y


def compute_class_weights(y: List[int]) -> dict:
    # classic inverse frequency weights
    y_arr = np.array(y)
    pos = (y_arr == 1).sum()
    neg = (y_arr == 0).sum()
    total = len(y_arr)
    # avoid div by zero
    pos_w = total / (2.0 * max(pos, 1))
    neg_w = total / (2.0 * max(neg, 1))
    return {0: float(neg_w), 1: float(pos_w)}


def generator_features(
    paths: List[Path],
    labels: List[int],
    cfg: TrainConfig,
    augment: bool,
) -> Iterator[Tuple[np.ndarray, np.int32]]:
    """
    Yields (feature, label).
    feature is (n_mels, time_steps, 1) float32
    """
    idxs = list(range(len(paths)))
    if augment:
        random.shuffle(idxs)

    for i in idxs:
        mel = compute_or_load_mel(paths[i], cfg, augment=augment)  # (mels, t)
        mel = np.expand_dims(mel, axis=-1)  # (mels, t, 1)
        yield mel.astype(np.float32), np.int32(labels[i])


def make_tf_dataset(
    paths: List[Path],
    labels: List[int],
    cfg: TrainConfig,
    augment: bool,
    shuffle: bool,
) -> tf.data.Dataset:
    # Peek a sample to determine static shapes
    sample_mel = compute_or_load_mel(paths[0], cfg, augment=False)
    time_steps = sample_mel.shape[1]

    output_signature = (
        tf.TensorSpec(shape=(cfg.n_mels, time_steps, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )

    ds = tf.data.Dataset.from_generator(
        lambda: generator_features(paths, labels, cfg, augment=augment),
        output_signature=output_signature,
    )

    if shuffle:
        ds = ds.shuffle(
            buffer_size=min(2000, len(paths)),
            seed=cfg.seed,
            reshuffle_each_iteration=True,
        )

    ds = ds.batch(cfg.batch_size, drop_remainder=False)

    # Basic map to cast label to float for BCE
    ds = ds.map(
        lambda x, y: (x, tf.cast(y, tf.float32)),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    ds = ds.prefetch(cfg.prefetch)
    return ds


# ----------------------------
# Model
# ----------------------------

def build_small_cnn(input_shape: Tuple[int, int, int], cfg: TrainConfig) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape)

    x = inputs

    # Block 1
    x = tf.keras.layers.Conv2D(16, (3, 3), padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D((2, 2))(x)
    x = tf.keras.layers.Dropout(cfg.dropout)(x)

    # Block 2
    x = tf.keras.layers.Conv2D(32, (3, 3), padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D((2, 2))(x)
    x = tf.keras.layers.Dropout(cfg.dropout)(x)

    # Block 3 (small)
    x = tf.keras.layers.Conv2D(48, (3, 3), padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(cfg.dropout)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs, name="aquil_mel_cnn")
    return model


def make_optimizer(cfg: TrainConfig) -> tf.keras.optimizers.Optimizer:
    # AdamW if available; fallback to Adam
    try:
        return tf.keras.optimizers.AdamW(learning_rate=cfg.lr, weight_decay=cfg.weight_decay)
    except Exception:
        return tf.keras.optimizers.Adam(learning_rate=cfg.lr)


# ----------------------------
# Evaluation helpers
# ----------------------------

def evaluate_on_dataset(model: tf.keras.Model, ds: tf.data.Dataset, threshold: float = 0.5) -> dict:
    y_true = []
    y_prob = []
    for xb, yb in ds:
        prob = model.predict(xb, verbose=0).reshape(-1)
        y_prob.extend(prob.tolist())
        y_true.extend(yb.numpy().astype(int).tolist())

    y_pred = [1 if p >= threshold else 0 for p in y_prob]

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()
    return {
        "threshold": threshold,
        "classification_report": report,
        "confusion_matrix": cm,
    }


# ----------------------------
# TFLite export
# ----------------------------

def export_tflite_float(model: tf.keras.Model, out_path: Path) -> None:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite = converter.convert()
    out_path.write_bytes(tflite)


def export_tflite_int8(
    model: tf.keras.Model,
    cfg: TrainConfig,
    rep_ds: tf.data.Dataset,
    out_path: Path,
) -> None:
    """
    Full integer quantization (int8) for MCU-friendly deployment.
    NOTE: Some ops may still require float if unsupported. Keep the CNN small.
    """
    def representative_dataset():
        # yield a few batches of inputs only
        taken = 0
        for xb, _ in rep_ds:
            # xb: (B, mels, t, 1)
            for i in range(xb.shape[0]):
                yield [tf.cast(xb[i : i + 1], tf.float32)]
                taken += 1
                if taken >= cfg.rep_data_samples:
                    return

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset

    # Request full int8
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite = converter.convert()
    out_path.write_bytes(tflite)


# ----------------------------
# Main
# ----------------------------

def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_dir", type=str, required=True, help="Path to data/ directory"
    )
    p.add_argument(
        "--out_dir", type=str, default="./artifacts", help="Where to save models and logs"
    )

    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--cache_features", action="store_true", help="Cache mel features to .npy files")
    p.add_argument("--no_cache_features", action="store_true", help="Disable caching mel features")

    p.add_argument("--export_tflite", action="store_true", help="Export TFLite models")
    p.add_argument("--no_export_tflite", action="store_true", help="Disable TFLite export")

    args = p.parse_args()

    cfg = TrainConfig(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        cache_features=bool(args.cache_features),
        export_tflite=bool(args.export_tflite),
    )

    # allow explicit negations
    if args.no_cache_features:
        cfg = TrainConfig(**{**asdict(cfg), "cache_features": False})
    if args.no_export_tflite:
        cfg = TrainConfig(**{**asdict(cfg), "export_tflite": False})

    return cfg


def main() -> None:
    cfg = parse_args()
    set_seeds(cfg.seed)

    data_dir = Path(cfg.data_dir)
    out_dir = Path(cfg.out_dir)
    ensure_dir(out_dir)

    # Log config
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    print(f"[{human_time()}] Config saved to {out_dir / 'config.json'}")

    # Load splits
    train_paths, train_labels = build_file_list(data_dir / "train")
    val_paths, val_labels = build_file_list(data_dir / "val")

    if len(train_paths) == 0 or len(val_paths) == 0:
        raise RuntimeError(
            "No training/validation data found. Check directory structure:\n"
            "data/train/gunshot/*.(wav|mp3|flac), data/train/not_gunshot/*.(wav|mp3|flac), etc."
        )

    print(
        f"[{human_time()}] Train files: {len(train_paths)} "
        f"(pos={sum(train_labels)}, neg={len(train_labels)-sum(train_labels)})"
    )
    print(
        f"[{human_time()}] Val files:   {len(val_paths)} "
        f"(pos={sum(val_labels)}, neg={len(val_labels)-sum(val_labels)})"
    )

    # Build datasets
    train_ds = make_tf_dataset(train_paths, train_labels, cfg, augment=True, shuffle=True)
    val_ds = make_tf_dataset(val_paths, val_labels, cfg, augment=False, shuffle=False)

    # Determine input shape by peeking one batch
    xb, _ = next(iter(train_ds))
    input_shape = tuple(xb.shape[1:])  # (mels, t, 1)
    print(f"[{human_time()}] Input shape: {input_shape}")

    # Model
    model = build_small_cnn(input_shape, cfg)
    model.summary()

    # Loss & metrics
    loss_fn = tf.keras.losses.BinaryCrossentropy(label_smoothing=cfg.label_smoothing)
    metrics = [
        tf.keras.metrics.BinaryAccuracy(name="acc", threshold=0.5),
        tf.keras.metrics.Precision(name="precision", thresholds=0.5),
        tf.keras.metrics.Recall(name="recall", thresholds=0.5),
        tf.keras.metrics.AUC(name="auc"),
    ]

    model.compile(
        optimizer=make_optimizer(cfg),
        loss=loss_fn,
        metrics=metrics,
    )

    # Class weights (helps if you have fewer gunshots than negatives)
    cw = None
    if cfg.class_weight:
        cw = compute_class_weights(train_labels)
        print(f"[{human_time()}] Using class weights: {cw}")

    # Callbacks
    ckpt_path = out_dir / "checkpoints" / "best.weights.h5"
    ensure_dir(ckpt_path.parent)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_path),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc",
            mode="max",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(out_dir / "train_log.csv")),
    ]

    # Train
    print(f"[{human_time()}] Starting training...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.epochs,
        class_weight=cw,
        callbacks=callbacks,
        verbose=1,
    )

    # Save final model artifacts
    savedmodel_dir = out_dir / "saved_model"
    keras_h5 = out_dir / "model.h5"
    model.export(savedmodel_dir)
    model.save(keras_h5)
    print(f"[{human_time()}] SavedModel -> {savedmodel_dir}")
    print(f"[{human_time()}] Keras H5  -> {keras_h5}")

    # Evaluate and write reports
    eval_05 = evaluate_on_dataset(model, val_ds, threshold=0.5)
    (out_dir / "val_eval_threshold_0.5.json").write_text(json.dumps(eval_05, indent=2))
    print(f"[{human_time()}] Wrote val eval report @ threshold 0.5")

    # If you care about false positives, you will likely raise threshold (e.g., 0.95)
    eval_095 = evaluate_on_dataset(model, val_ds, threshold=0.95)
    (out_dir / "val_eval_threshold_0.95.json").write_text(json.dumps(eval_095, indent=2))
    print(f"[{human_time()}] Wrote val eval report @ threshold 0.95")

    # Optional: test set eval
    test_dir = data_dir / "test"
    if test_dir.exists():
        test_paths, test_labels = build_file_list(test_dir)
        if len(test_paths) > 0:
            test_ds = make_tf_dataset(
                test_paths, test_labels, cfg, augment=False, shuffle=False
            )
            test_eval_095 = evaluate_on_dataset(model, test_ds, threshold=0.95)
            (out_dir / "test_eval_threshold_0.95.json").write_text(
                json.dumps(test_eval_095, indent=2)
            )
            print(f"[{human_time()}] Wrote test eval report @ threshold 0.95")

    # TFLite export
    if cfg.export_tflite:
        tflite_float_path = out_dir / "model_float.tflite"
        export_tflite_float(model, tflite_float_path)
        print(f"[{human_time()}] Exported TFLite float -> {tflite_float_path}")

        if cfg.export_int8:
            # Build a representative dataset from TRAIN but WITHOUT augmentation
            rep_ds = make_tf_dataset(train_paths, train_labels, cfg, augment=False, shuffle=True)
            tflite_int8_path = out_dir / "model_int8.tflite"
            try:
                export_tflite_int8(model, cfg, rep_ds, tflite_int8_path)
                print(f"[{human_time()}] Exported TFLite int8 -> {tflite_int8_path}")
            except Exception as e:
                print(f"[{human_time()}] WARNING: int8 export failed: {e}")
                print("This can happen if some ops can't be quantized with your model structure.")

    # Save a minimal “deployment hints” file
    deploy_hints = {
        "recommended_threshold": 0.95,
        "input_shape": list(input_shape),
        "feature": {
            "type": "mel_spectrogram_db_normalized",
            "sample_rate": cfg.sample_rate,
            "clip_seconds": cfg.clip_seconds,
            "n_fft": cfg.n_fft,
            "hop_length": cfg.hop_length,
            "n_mels": cfg.n_mels,
            "fmin": cfg.fmin,
            "fmax": cfg.fmax,
            "db_clip_min": -80.0,
            "db_clip_max": 0.0,
            "norm": "(db+80)/80",
        },
    }
    (out_dir / "deployment_hints.json").write_text(json.dumps(deploy_hints, indent=2))
    print(f"[{human_time()}] Wrote deployment_hints.json")

    print(f"[{human_time()}] Done.")


if __name__ == "__main__":
    # Optional: reduce TF logging noise
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()
