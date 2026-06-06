#!/usr/bin/env python3
"""Train 1D-CNN classifier on denoised MIT-BIH segments.
Produces int8 quantized TFLite model with patient-level cross-validation.
"""

import os
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from data_loader import WINDOW_SAMPLES, NUM_CLASSES, load_record, SYM_TO_CLASS

OUT_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(OUT_DIR, exist_ok=True)

CLASS_NAMES = ["Normal", "AFib", "PVC", "Tachycardia", "Bradycardia", "ST Abnormality"]


def normalize_for_firmware(segment):
    """Firmware-compatible normalization.

    Centers raw ADC/mV signal at 1650 (midpoint of 0–3300 mV range),
    right-shifts by 4 (divide by 16), then clips to [-128, 127].
    This mirrors the int8 quantization the firmware applies at inference.

    Parameters
    ----------
    segment : np.ndarray
        Raw ECG segment in mV range [0, 3300].

    Returns
    -------
    np.ndarray (int8)
        Firmware-normalized segment.
    """
    centered = segment.astype(np.float32) - 1650.0
    shifted = centered / 16.0
    clipped = np.clip(np.round(shifted), -128, 127).astype(np.int8)
    return clipped


def build_classifier(input_shape=(WINDOW_SAMPLES, 1), num_classes=NUM_CLASSES):
    """3 Conv1D blocks (32 / 64 / 128), kernel 5, BN, MaxPool → GAP → Dense(64) → Dropout(0.5) → Dense(6, Softmax)."""
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
        metrics=["accuracy"],
    )
    return model


def load_data_with_record_tracking(data_dir="./mitdb", max_per_class=None):
    """Load all MIT-BIH segments while preserving the originating record ID.

    Each returned entry is ``(segment_array, class_id, record_id)`` so that
    patient-level cross-validation can keep every record's segments together.
    """
    from data_loader import download_mitdb
    import os, shutil

    if not os.path.exists(data_dir) or not any(f.endswith(".dat") for f in os.listdir(data_dir)):
        if os.path.exists(data_dir):
            shutil.rmtree(data_dir)
        download_mitdb(data_dir)

    records = sorted(set(
        f.replace(".dat", "") for f in os.listdir(data_dir) if f.endswith(".dat")
    ))

    by_class = {c: [] for c in range(NUM_CLASSES)}
    for rec_name in records:
        try:
            signal, ann = load_record(data_dir, rec_name)
        except Exception as e:
            print(f"  {rec_name}: SKIP ({e})")
            continue
        r_peaks = ann.sample
        symbols = ann.symbol
        count = 0
        for peak, sym in zip(r_peaks, symbols):
            cls = SYM_TO_CLASS.get(sym, None)
            if cls is None:
                continue
            start = peak - WINDOW_SAMPLES // 2
            end = start + WINDOW_SAMPLES
            if start < 0 or end >= len(signal):
                continue
            seg = signal[start:end].copy()
            seg_max = max(abs(seg.min()), abs(seg.max()))
            if seg_max > 0:
                seg = seg / seg_max
            by_class[cls].append((seg, rec_name))
            count += 1
        print(f"  {rec_name}: {count} segments")

    for c in range(NUM_CLASSES):
        if max_per_class and len(by_class[c]) > max_per_class:
            by_class[c] = by_class[c][:max_per_class]

    return by_class


def record_level_split(by_class, train_ratio=0.7, val_ratio=0.15, seed=42):
    """Split data at the record (patient) level using GroupKFold semantics.

    Ensures no ECG recording contributes segments to more than one split,
    preventing optimistic bias from intra-patient correlation.

    Parameters
    ----------
    by_class : dict
        ``{class_id: [(segment, record_id), ...]}`` from
        :func:`load_data_with_record_tracking`.

    Returns
    -------
    (X_train, y_train), (X_val, y_val), (X_test, y_test)
        Each X is shape ``(n_samples, WINDOW_SAMPLES)`` and each y is
        shape ``(n_samples,)``.
    """
    np.random.seed(seed)

    all_data = []  # (segment, label, record)
    for cls, items in by_class.items():
        for seg, rec in items:
            all_data.append((seg, cls, rec))

    if not all_data:
        empty = (np.array([]), np.array([]))
        return empty, empty, empty

    X_all = np.array([d[0] for d in all_data])
    y_all = np.array([d[1] for d in all_data])
    groups_all = np.array([d[2] for d in all_data])

    unique_records = np.unique(groups_all)
    np.random.shuffle(unique_records)

    n = len(unique_records)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_recs = set(unique_records[:n_train])
    val_recs = set(unique_records[n_train:n_train + n_val])
    test_recs = set(unique_records[n_train + n_val:])

    train_idx = np.where(np.isin(groups_all, list(train_recs)))[0]
    val_idx = np.where(np.isin(groups_all, list(val_recs)))[0]
    test_idx = np.where(np.isin(groups_all, list(test_recs)))[0]

    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_val, y_val = X_all[val_idx], y_all[val_idx]
    X_test, y_test = X_all[test_idx], y_all[test_idx]

    # Leakage check — warn if any record ID appears in multiple splits
    train_unique = set(groups_all[train_idx].tolist())
    val_unique = set(groups_all[val_idx].tolist())
    test_unique = set(groups_all[test_idx].tolist())

    leaks = []
    tv = train_unique & val_unique
    tt = train_unique & test_unique
    vt = val_unique & test_unique
    if tv:
        leaks.append(f"train/val: {tv}")
    if tt:
        leaks.append(f"train/test: {tt}")
    if vt:
        leaks.append(f"val/test: {vt}")
    if leaks:
        print("WARNING: Patient leakage detected — same record(s) in multiple splits:")
        for l in leaks:
            print(f"         {l}")
    else:
        print("Patient-level split OK — no leakage across train/val/test.")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def main():
    print("=== Loading MIT-BIH Data (record-aware) ===")
    by_class = load_data_with_record_tracking(max_per_class=5000)

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = record_level_split(by_class)

    # Reshape for Conv1D: (samples, time, channels)
    X_train = X_train.reshape(-1, WINDOW_SAMPLES, 1).astype(np.float32)
    X_val = X_val.reshape(-1, WINDOW_SAMPLES, 1).astype(np.float32)
    X_test = X_test.reshape(-1, WINDOW_SAMPLES, 1).astype(np.float32)

    print(f"\nTrain: {X_train.shape}, y: {np.bincount(y_train)}")
    print(f"Val:   {X_val.shape}, y: {np.bincount(y_val)}")
    print(f"Test:  {X_test.shape}, y: {np.bincount(y_test)}")

    # Class weights for imbalance
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, weights))
    print(f"Class weights: {class_weight_dict}")
    # Alternative: imblearn.over_sampling.SMOTE can be applied before training
    # to synthetically oversample minority classes instead of using class_weight.

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
        callbacks=callbacks, verbose=1,
    )

    # Evaluate float32
    print("\n=== Evaluation (float32) ===")
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    f1_float = f1_score(y_test, y_pred, average=None)
    print(f"\nPer-class F1 (float32):")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name}: {f1_float[i]:.4f}")
    print(f"Macro F1 (float32): {np.mean(f1_float):.4f}")

    # Convert to TFLite int8
    print("\n=== Converting to TFLite (int8) ===")
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
    tflite_model = converter.convert()

    out_path = os.path.join(OUT_DIR, "classifier_int8.tflite")
    with open(out_path, "wb") as f:
        f.write(tflite_model)
    print(f"Saved: {out_path} ({len(tflite_model) / 1024:.1f} KB)")

    # Validate quantized model
    print("\n=== Quantized Model Validation ===")
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    in_scale, in_zero = input_details[0]["quantization"]
    out_scale, out_zero = output_details[0]["quantization"]
    print(f"Input quantization:  scale={in_scale}, zero_point={in_zero}")
    print(f"Output quantization: scale={out_scale}, zero_point={out_zero}")

    y_pred_q = []
    for i in range(len(X_test)):
        x_float = X_test[i].astype(np.float32)
        x_q = np.round(x_float / in_scale + in_zero).clip(-128, 127).astype(np.int8)
        interpreter.set_tensor(input_details[0]["index"], x_q.reshape(1, WINDOW_SAMPLES, 1))
        interpreter.invoke()
        raw_out = interpreter.get_tensor(output_details[0]["index"])[0]
        # Dequantize: value = (int8_val - zero_point) * scale
        float_out = (raw_out.astype(np.float32) - out_zero) * out_scale
        y_pred_q.append(np.argmax(float_out))

    y_pred_q = np.array(y_pred_q)

    f1_quant = f1_score(y_test, y_pred_q, average=None)
    print(f"\nPer-class F1 (quantized int8):")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name}: {f1_quant[i]:.4f}")
    print(f"Macro F1 (quantized): {np.mean(f1_quant):.4f}")

    print("\nPer-class F1 delta (float32 → quantized):")
    for i, name in enumerate(CLASS_NAMES):
        delta = f1_float[i] - f1_quant[i]
        sign = "+" if delta >= 0 else ""
        print(f"  {name}: {f1_float[i]:.4f} → {f1_quant[i]:.4f} ({sign}{delta:.4f})")

    delta_macro = np.mean(f1_float) - np.mean(f1_quant)
    print(f"Macro F1 delta: {delta_macro:.4f}")

    print("\nClassification report (quantized):")
    print(classification_report(y_test, y_pred_q, target_names=CLASS_NAMES, digits=4))

    print("\nDone.")


if __name__ == "__main__":
    main()
