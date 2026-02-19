"""Edge-style streaming inference simulator for Aquili audio models."""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import librosa
import numpy as np
import tensorflow as tf


def fix_length(y: np.ndarray, target_len: int) -> np.ndarray:
    """Pad or trim to target length."""
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode="constant")
    elif len(y) > target_len:
        y = y[:target_len]
    return y


def mel_spectrogram_db(
    y: np.ndarray,
    sr: int,
    n_fft: int,
    hop_length: int,
    n_mels: int,
    fmin: int,
    fmax: int,
) -> np.ndarray:
    """Build mel spectrogram in dB."""
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
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)


def normalize_mel(mel_db: np.ndarray) -> np.ndarray:
    """Match training normalization: clip to [-80, 0] then scale to [0, 1]."""
    mel_db = np.clip(mel_db, -80.0, 0.0)
    return ((mel_db + 80.0) / 80.0).astype(np.float32)


def make_feature(
    y_window: np.ndarray,
    sample_rate: int,
    clip_seconds: float,
    n_fft: int,
    hop_length: int,
    n_mels: int,
    fmin: int,
    fmax: int,
) -> np.ndarray:
    """Create model input feature for one rolling window."""
    target_len = int(sample_rate * clip_seconds)
    y_window = fix_length(y_window.astype(np.float32), target_len)
    mel_db = mel_spectrogram_db(
        y=y_window,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
    )
    mel = normalize_mel(mel_db)
    return np.expand_dims(np.expand_dims(mel, axis=0), axis=-1).astype(np.float32)


def load_keras_predictor(model_path: Path) -> Callable[[np.ndarray], float]:
    """Load keras model and return probability predictor."""
    model = tf.keras.models.load_model(model_path)

    def predict_fn(x: np.ndarray) -> float:
        out = model.predict(x, verbose=0).reshape(-1)[0]
        return float(out)

    return predict_fn


def load_tflite_predictor(model_path: Path) -> Callable[[np.ndarray], float]:
    """Load TFLite model and return probability predictor."""
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    in_index = input_details["index"]
    out_index = output_details["index"]
    in_dtype = input_details["dtype"]
    out_dtype = output_details["dtype"]
    in_scale, in_zero = input_details["quantization"]
    out_scale, out_zero = output_details["quantization"]

    def quantize_input(x: np.ndarray) -> np.ndarray:
        if in_dtype == np.float32:
            return x.astype(np.float32)
        if in_scale == 0:
            raise ValueError("Invalid TFLite input quantization scale=0.")
        q = np.round(x / in_scale + in_zero)
        if in_dtype == np.int8:
            q = np.clip(q, -128, 127)
        elif in_dtype == np.uint8:
            q = np.clip(q, 0, 255)
        return q.astype(in_dtype)

    def dequantize_output(y: np.ndarray) -> np.ndarray:
        if out_dtype == np.float32:
            return y.astype(np.float32)
        if out_scale == 0:
            raise ValueError("Invalid TFLite output quantization scale=0.")
        return (y.astype(np.float32) - out_zero) * out_scale

    def predict_fn(x: np.ndarray) -> float:
        interpreter.set_tensor(in_index, quantize_input(x))
        interpreter.invoke()
        out = interpreter.get_tensor(out_index)
        prob = dequantize_output(out).reshape(-1)[0]
        return float(prob)

    return predict_fn


def load_predictor(model_path: Path) -> Callable[[np.ndarray], float]:
    """Load Keras or TFLite predictor based on extension."""
    suffix = model_path.suffix.lower()
    if suffix in {".h5", ".keras"}:
        return load_keras_predictor(model_path)
    if suffix == ".tflite":
        return load_tflite_predictor(model_path)
    raise ValueError(
        f"Unsupported model extension: {suffix}. "
        "Use .h5, .keras, or .tflite."
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(
        description="Simulate edge streaming inference from an audio file."
    )
    parser.add_argument("--model_path", type=str, required=True, help="Path to .h5/.keras/.tflite model.")
    parser.add_argument("--input_audio", type=str, required=True, help="Path to input audio file.")
    parser.add_argument("--threshold", type=float, default=0.95, help="Probability threshold.")
    parser.add_argument("--hop_ms", type=int, default=100, help="Streaming hop size in milliseconds.")
    parser.add_argument("--clip_seconds", type=float, default=1.0, help="Rolling model window size.")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Resample rate.")
    parser.add_argument("--n_fft", type=int, default=1024, help="Mel FFT window.")
    parser.add_argument("--hop_length", type=int, default=512, help="Mel hop length.")
    parser.add_argument("--n_mels", type=int, default=40, help="Mel bins.")
    parser.add_argument("--fmin", type=int, default=20, help="Mel min frequency.")
    parser.add_argument("--fmax", type=int, default=8000, help="Mel max frequency.")
    parser.add_argument("--vote_window", type=int, default=5, help="N-of-M vote window M.")
    parser.add_argument("--vote_hits", type=int, default=3, help="N-of-M vote hits N.")
    parser.add_argument("--cooldown_sec", type=float, default=2.0, help="Minimum seconds between alerts.")
    parser.add_argument(
        "--events_json",
        type=str,
        default="",
        help="Optional path to save detected alert events as JSON.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate argument consistency."""
    if not (0.0 <= args.threshold <= 1.0):
        raise ValueError("--threshold must be between 0 and 1.")
    if args.hop_ms <= 0:
        raise ValueError("--hop_ms must be > 0.")
    if args.clip_seconds <= 0:
        raise ValueError("--clip_seconds must be > 0.")
    if args.vote_window <= 0 or args.vote_hits <= 0:
        raise ValueError("--vote_window and --vote_hits must be > 0.")
    if args.vote_hits > args.vote_window:
        raise ValueError("--vote_hits cannot be larger than --vote_window.")


def main() -> None:
    """Run streaming inference simulation and print detections."""
    args = parse_args()
    validate_args(args)

    model_path = Path(args.model_path)
    audio_path = Path(args.input_audio)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    predictor = load_predictor(model_path)
    y, _ = librosa.load(str(audio_path), sr=args.sample_rate, mono=True)
    y = y.astype(np.float32)

    win_samples = int(args.sample_rate * args.clip_seconds)
    hop_samples = max(1, int(args.sample_rate * args.hop_ms / 1000))

    buffer = np.zeros(win_samples, dtype=np.float32)
    seen = 0
    vote_history: deque[int] = deque(maxlen=args.vote_window)
    last_alert_time = -1e9
    events: List[Dict[str, float]] = []

    total_steps = 0
    raw_hits = 0

    for start in range(0, len(y), hop_samples):
        chunk = y[start : start + hop_samples]
        if len(chunk) < hop_samples:
            chunk = np.pad(chunk, (0, hop_samples - len(chunk)), mode="constant")

        if hop_samples >= win_samples:
            buffer = chunk[-win_samples:].astype(np.float32)
        else:
            buffer = np.roll(buffer, -hop_samples)
            buffer[-hop_samples:] = chunk
        seen += len(chunk)
        total_steps += 1

        if seen < win_samples:
            continue

        x = make_feature(
            y_window=buffer,
            sample_rate=args.sample_rate,
            clip_seconds=args.clip_seconds,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            n_mels=args.n_mels,
            fmin=args.fmin,
            fmax=args.fmax,
        )
        prob = predictor(x)
        is_hit = int(prob >= args.threshold)
        raw_hits += is_hit
        vote_history.append(is_hit)

        t_sec = start / args.sample_rate
        votes = int(sum(vote_history))
        in_cooldown = (t_sec - last_alert_time) < args.cooldown_sec

        if len(vote_history) == args.vote_window and votes >= args.vote_hits and not in_cooldown:
            last_alert_time = t_sec
            event = {"time_sec": round(t_sec, 3), "prob": round(prob, 4), "votes": votes}
            events.append(event)
            print(
                f"ALERT time={event['time_sec']:.3f}s "
                f"prob={event['prob']:.4f} votes={votes}/{args.vote_window}"
            )

    print("\nSummary")
    print(f"  file: {audio_path}")
    print(f"  model: {model_path}")
    print(f"  steps: {total_steps}")
    print(f"  raw threshold hits: {raw_hits}")
    print(f"  alerts (vote+cooldown): {len(events)}")
    print(
        f"  config: threshold={args.threshold}, "
        f"vote={args.vote_hits}/{args.vote_window}, cooldown={args.cooldown_sec}s"
    )

    if args.events_json:
        out_path = Path(args.events_json)
        out_path.write_text(json.dumps(events, indent=2), encoding="utf-8")
        print(f"  wrote events: {out_path}")


if __name__ == "__main__":
    main()
