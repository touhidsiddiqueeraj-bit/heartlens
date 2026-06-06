#!/usr/bin/env python3
"""Post-training int8 quantization for trained Keras models.
Usage: python quantize.py <keras_model_path> [output_path]
"""

import os
import sys
import numpy as np
import tensorflow as tf


def quantize_tflite(model_path, output_path=None, rep_data_size=200, input_shape=(360, 1)):
    """Convert Keras model to int8 quantized TFLite."""
    if output_path is None:
        output_path = os.path.splitext(model_path)[0] + "_int8.tflite"

    model = tf.keras.models.load_model(model_path)

    # Representative dataset using realistic ECG-like signals
    # (uniform in [-1, 1] approximates the centered/normalized ECG distribution)
    rng = np.random.default_rng(42)
    def rep_data():
        for _ in range(rep_data_size):
            # Uniform noise in [-1, 1] — crude approximation of centered ECG
            data = rng.uniform(-1.0, 1.0, (1, *input_shape)).astype(np.float32)
            yield [data]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_data
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()

    with open(output_path, "wb") as f:
        f.write(tflite_model)

    size_kb = len(tflite_model) / 1024
    print(f"Quantized: {model_path} → {output_path} ({size_kb:.1f} KB)")

    # Quick validation
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    in_d = interpreter.get_input_details()[0]
    out_d = interpreter.get_output_details()[0]
    print(f"  Input:  {in_d['shape']}  int8  scale={in_d['quantization'][0]:.6f}")
    print(f"  Output: {out_d['shape']}  int8  scale={out_d['quantization'][0]:.6f}")

    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python quantize.py <keras_model_path> [output_path]")
        sys.exit(1)

    model_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    quantize_tflite(model_path, output_path)
