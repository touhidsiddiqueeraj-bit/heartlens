#!/usr/bin/env python3
"""Train LSTM denoiser on MIT-BIH + synthetic noise.
Produces int8 quantized TFLite model.
"""

import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from data_loader import load_all_segments, WINDOW_SAMPLES
from noise_pipeline import add_all_noise, augment_dataset

OUT_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(OUT_DIR, exist_ok=True)


def build_denoiser(input_shape=(WINDOW_SAMPLES, 1)):
    """Build LSTM encoder-decoder denoiser."""
    inputs = tf.keras.layers.Input(shape=input_shape, name="denoiser_input")
    x = tf.keras.layers.LSTM(64, return_sequences=True, name="lstm_enc_1")(inputs)
    x = tf.keras.layers.LSTM(64, return_sequences=True, name="lstm_enc_2")(x)
    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(1, name="output_dense"), name="denoiser_output"
    )(x)
    model = tf.keras.Model(inputs, x, name="lstm_denoiser")
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def main():
    print("=== Loading MIT-BIH Data ===")
    by_class = load_all_segments(max_per_class=3000)

    # Denoiser uses all segments regardless of class
    all_segs = []
    for cls, segs in by_class.items():
        all_segs.extend(segs)
    all_segs = np.array(all_segs).reshape(-1, WINDOW_SAMPLES, 1)
    print(f"Total segments: {all_segs.shape}")

    # Split
    X_train, X_tmp = train_test_split(all_segs, test_size=0.3, random_state=42)
    X_val, X_test = train_test_split(X_tmp, test_size=0.5, random_state=42)
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Augment with noise
    print("\n=== Generating Noisy Training Data ===")
    noise_levels = (0, 5, 10, 15, 20, 30, 40)
    X_clean_aug = []
    X_noisy_aug = []
    for ecg in X_train:
        for snr in noise_levels:
            noisy = add_all_noise(ecg.flatten(), snr_db=snr)
            X_clean_aug.append(ecg.flatten())
            X_noisy_aug.append(noisy)
    X_clean_aug = np.array(X_clean_aug).reshape(-1, WINDOW_SAMPLES, 1)
    X_noisy_aug = np.array(X_noisy_aug).reshape(-1, WINDOW_SAMPLES, 1)
    print(f"Augmented: {X_noisy_aug.shape}")

    # Val noise
    X_val_noisy = np.array([add_all_noise(ecg.flatten(), snr_db=15) for ecg in X_val])
    X_val_noisy = X_val_noisy.reshape(-1, WINDOW_SAMPLES, 1)

    # Build and train
    print("\n=== Building Denoiser Model ===")
    model = build_denoiser()
    model.summary()

    print("\n=== Training ===")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(OUT_DIR, "denoiser_best.keras"), save_best_only=True
        ),
    ]

    history = model.fit(
        X_noisy_aug, X_clean_aug,
        validation_data=(X_val_noisy, X_val),
        epochs=50, batch_size=64,
        callbacks=callbacks, verbose=1
    )

    # Evaluate
    print("\n=== Evaluation ===")
    X_test_noisy = np.array([add_all_noise(ecg.flatten(), snr_db=15) for ecg in X_test])
    X_test_noisy = X_test_noisy.reshape(-1, WINDOW_SAMPLES, 1)
    loss, mae = model.evaluate(X_test_noisy, X_test, verbose=0)
    print(f"Test MSE: {loss:.6f}, MAE: {mae:.6f}")

    # Convert to TFLite
    print("\n=== Converting to TFLite (int8) ===")

    def rep_data():
        for ecg in X_val[:100]:
            noisy = add_all_noise(ecg.flatten(), snr_db=15)
            yield [noisy.reshape(1, WINDOW_SAMPLES, 1).astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_data
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()

    out_path = os.path.join(OUT_DIR, "denoiser_int8.tflite")
    with open(out_path, "wb") as f:
        f.write(tflite_model)
    print(f"Saved: {out_path} ({len(tflite_model) / 1024:.1f} KB)")

    print("\nDone.")


if __name__ == "__main__":
    main()
