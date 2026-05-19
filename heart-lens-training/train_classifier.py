#!/usr/bin/env python3
"""Train 1D-CNN classifier on denoised MIT-BIH segments.
Produces int8 quantized TFLite model.
"""

import os
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

from data_loader import load_all_segments, split_dataset, WINDOW_SAMPLES, NUM_CLASSES

OUT_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(OUT_DIR, exist_ok=True)

CLASS_NAMES = ["Normal", "AFib", "PVC", "Tachycardia", "Bradycardia", "ST Abnormality"]


def build_classifier(input_shape=(WINDOW_SAMPLES, 1), num_classes=NUM_CLASSES):
    """Build 1D-CNN classifier."""
    inputs = tf.keras.layers.Input(shape=input_shape, name="classifier_input")
    x = tf.keras.layers.Conv1D(32, 5, padding="same", activation="relu", name="conv1")(inputs)
    x = tf.keras.layers.BatchNormalization(name="bn1")(x)
    x = tf.keras.layers.MaxPool1D(2, name="pool1")(x)

    x = tf.keras.layers.Conv1D(64, 5, padding="same", activation="relu", name="conv2")(x)
    x = tf.keras.layers.BatchNormalization(name="bn2")(x)
    x = tf.keras.layers.MaxPool1D(2, name="pool2")(x)

    x = tf.keras.layers.Conv1D(128, 5, padding="same", activation="relu", name="conv3")(x)
    x = tf.keras.layers.BatchNormalization(name="bn3")(x)
    x = tf.keras.layers.GlobalAveragePooling1D(name="gap")(x)

    x = tf.keras.layers.Dense(64, activation="relu", name="dense")(x)
    x = tf.keras.layers.Dropout(0.5, name="dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="classifier_output")(x)

    model = tf.keras.Model(inputs, outputs, name="cnn_classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def main():
    print("=== Loading MIT-BIH Data ===")
    by_class = load_all_segments(max_per_class=5000)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_dataset(by_class)

    print(f"\nTrain: {X_train.shape}, y: {np.bincount(y_train)}")
    print(f"Val:   {X_val.shape}, y: {np.bincount(y_val)}")
    print(f"Test:  {X_test.shape}, y: {np.bincount(y_test)}")

    # Class weights for imbalance
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, weights))
    print(f"Class weights: {class_weight_dict}")

    # Build and train
    print("\n=== Building Classifier ===")
    model = build_classifier()
    model.summary()

    print("\n=== Training ===")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(OUT_DIR, "classifier_best.keras"), save_best_only=True
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100, batch_size=64,
        class_weight=class_weight_dict,
        callbacks=callbacks, verbose=1
    )

    # Evaluate
    print("\n=== Evaluation (float32) ===")
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Per-class F1
    from sklearn.metrics import f1_score
    f1 = f1_score(y_test, y_pred, average=None)
    print(f"\nPer-class F1: {dict(zip(CLASS_NAMES, [f'{s:.4f}' for s in f1]))}")
    print(f"Macro F1: {np.mean(f1):.4f}")

    # Convert to TFLite
    print("\n=== Converting to TFLite (int8) ===")
    def rep_data():
        for i in range(200):
            idx = np.random.randint(0, len(X_val))
            yield [X_val[idx:idx + 1].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_data
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()

    out_path = os.path.join(OUT_DIR, "classifier_int8.tflite")
    with open(out_path, "wb") as f:
        f.write(tflite_model)
    print(f"Saved: {out_path} ({len(tflite_model) / 1024:.1f} KB)")

    # Validate quantized accuracy
    print("\n=== Quantized Model Validation ===")
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    scale, zero = input_details[0]["quantization"]
    y_pred_q = []
    for i in range(len(X_test)):
        x_q = (X_test[i] / scale + zero).astype(np.int8)
        interpreter.set_tensor(input_details[0]["index"], x_q.reshape(1, WINDOW_SAMPLES, 1))
        interpreter.invoke()
        out = interpreter.get_tensor(output_details[0]["index"])
        y_pred_q.append(np.argmax(out[0]))

    f1_q = f1_score(y_test, y_pred_q, average="macro")
    print(f"Quantized macro F1: {f1_q:.4f}")
    print(f"Accuracy delta: {np.mean(f1) - f1_q:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
