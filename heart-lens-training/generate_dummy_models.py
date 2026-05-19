#!/usr/bin/env python3
"""Generate dummy TFLite models with correct shapes for firmware bringup.
Models have random weights — replace with trained versions later."""

import os
import numpy as np
import tensorflow as tf

OUT_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(OUT_DIR, exist_ok=True)

SEQUENCE_LEN = 360
NUM_CLASSES = 6

def build_and_quantize(inputs, outputs, name):
    model = tf.keras.Model(inputs, outputs, name=name)

    def rep_data():
        for _ in range(100):
            yield [np.random.randn(1, SEQUENCE_LEN, 1).astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_data
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()

    path = os.path.join(OUT_DIR, f"{name}.tflite")
    with open(path, "wb") as f:
        f.write(tflite_model)
    print(f"  [{name}] {len(tflite_model)/1024:.1f} KB → {path}")
    return path

print("=== Dummy Denoiser Model (Conv1D autoencoder) ===")
d_in = tf.keras.layers.Input((SEQUENCE_LEN, 1), name="denoiser_input")
x = tf.keras.layers.Conv1D(8, 15, padding="same", activation="relu", name="enc_conv")(d_in)
x = tf.keras.layers.Conv1D(1, 15, padding="same", name="dec_conv")(x)
build_and_quantize(d_in, x, "denoiser_dummy")

print("\n=== Dummy Classifier Model (1D-CNN) ===")
c_in = tf.keras.layers.Input((SEQUENCE_LEN, 1), name="classifier_input")
x = tf.keras.layers.Conv1D(8, 5, activation="relu", name="conv1")(c_in)
x = tf.keras.layers.MaxPool1D(2, name="pool1")(x)
x = tf.keras.layers.Conv1D(16, 5, activation="relu", name="conv2")(x)
x = tf.keras.layers.MaxPool1D(2, name="pool2")(x)
x = tf.keras.layers.Conv1D(32, 5, activation="relu", name="conv3")(x)
x = tf.keras.layers.GlobalAveragePooling1D(name="gap")(x)
x = tf.keras.layers.Dense(16, activation="relu", name="dense")(x)
c_out = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="classifier_output")(x)
build_and_quantize(c_in, c_out, "classifier_dummy")

print("\n=== Done ===")
