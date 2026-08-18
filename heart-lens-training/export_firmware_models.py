#!/usr/bin/env python3
"""Export the trained robust classifier/denoiser as int8 TFLite for the firmware.

Loads robust_classifier.keras + robust_denoiser.keras (from auto_train.py /
evaluate_noise_robustness.py), quantizes both with a representative set from
the same patient-level split, and writes robust_classifier_int8.tflite +
robust_denoiser_int8.tflite for convert_tflite_to_headers.sh.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import tensorflow as tf

from data_loader import WINDOW_SAMPLES
from data_loader import load_record_segments, split_by_patient

MODELS_DIR = Path(__file__).parent / "models"


def quantize(model, X_val):
    def rep_data():
        for _ in range(200):
            idx = np.random.randint(0, len(X_val))
            yield [X_val[idx:idx + 1].astype(np.float32)]
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_data
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30,
                    help="unused; kept for notebook symmetry")
    args = ap.parse_args()

    _, (X_val, _), _ = split_by_patient(load_record_segments())
    X_val = X_val.reshape(-1, WINDOW_SAMPLES, 1).astype(np.float32)

    for name in ["robust_classifier", "robust_denoiser"]:
        keras_path = MODELS_DIR / f"{name}.keras"
        if not keras_path.exists():
            print(f"SKIP: {keras_path} not found")
            continue
        model = tf.keras.models.load_model(keras_path)
        tflite_model = quantize(model, X_val)
        out = MODELS_DIR / f"{name}_int8.tflite"
        out.write_bytes(tflite_model)
        print(f"  {keras_path} -> {out} ({len(tflite_model)/1024:.1f} KB)")


if __name__ == "__main__":
    main()
