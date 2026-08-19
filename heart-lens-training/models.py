#!/usr/bin/env python3
"""Shared model builders for the HeartLens model-comparison study.

Four architectures on the same data pipeline (review #18 / Exp 5):
- cnn  : 3x Conv1D blocks (baseline, also used by train_classifier.py)
- lstm : Conv1D front-end + LSTM
- gru  : Conv1D front-end + GRU
- tcn  : 2 dilated causal Conv1D blocks with residual connections

All share the same head (GAP -> Dense(64) -> Dropout -> Dense(num_classes))
so differences are attributable to the backbone, not the head.
"""

import tensorflow as tf

from data_loader import WINDOW_SAMPLES, NUM_CLASSES


def _head(x, num_classes):
    x = tf.keras.layers.Dense(64, activation="relu", name="dense")(x)
    x = tf.keras.layers.Dropout(0.5, name="dropout")(x)
    return tf.keras.layers.Dense(num_classes, activation="softmax",
                                 name="classifier_output")(x)


def _conv_block(x, filters, kernel=5, name=""):
    x = tf.keras.layers.Conv1D(filters, kernel, padding="same",
                               activation="relu", name=f"{name}_conv")(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name}_bn")(x)
    return tf.keras.layers.MaxPool1D(2, name=f"{name}_pool")(x)


def build_classifier(model_type="cnn", input_shape=(WINDOW_SAMPLES, 1),
                     num_classes=NUM_CLASSES):
    """Build a classifier of the requested architecture."""
    inputs = tf.keras.layers.Input(shape=input_shape, name="classifier_input")

    if model_type == "cnn":
        x = _conv_block(inputs, 32, name="conv1")
        x = _conv_block(x, 64, name="conv2")
        x = _conv_block(x, 128, name="conv3")
        x = tf.keras.layers.GlobalAveragePooling1D(name="gap")(x)

    elif model_type == "lstm":
        x = _conv_block(inputs, 32, name="conv1")
        # RNN(LSTMCell) instead of LSTM(64): never fuses into the cuDNN
        # CudnnRNNV3 kernel, so the TFLite int8 converter accepts it
        # (emits UnidirectionalSequenceLSTM, supported by TFLM).
        x = tf.keras.layers.RNN(tf.keras.layers.LSTMCell(64), name="lstm")(x)

    elif model_type == "gru":
        x = _conv_block(inputs, 32, name="conv1")
        # Same reasoning as LSTM: RNN(GRUCell) avoids cuDNN fusion.
        x = tf.keras.layers.RNN(tf.keras.layers.GRUCell(64), name="gru")(x)

    elif model_type == "tcn":
        # Two dilated causal blocks with residual connections
        x = inputs
        for i, (dil, filt) in enumerate([(1, 32), (2, 64)]):
            skip = x
            x = tf.keras.layers.Conv1D(filt, 5, padding="causal",
                                       dilation_rate=dil,
                                       activation="relu",
                                       name=f"tcn{i}_conv1")(x)
            x = tf.keras.layers.BatchNormalization(name=f"tcn{i}_bn1")(x)
            x = tf.keras.layers.Conv1D(filt, 5, padding="causal",
                                       dilation_rate=dil,
                                       activation="relu",
                                       name=f"tcn{i}_conv2")(x)
            x = tf.keras.layers.BatchNormalization(name=f"tcn{i}_bn2")(x)
            if skip.shape[-1] != filt:
                skip = tf.keras.layers.Conv1D(filt, 1, name=f"tcn{i}_skip")(skip)
            x = tf.keras.layers.Add(name=f"tcn{i}_add")([x, skip])
            x = tf.keras.layers.MaxPool1D(2, name=f"tcn{i}_pool")(x)
        x = tf.keras.layers.GlobalAveragePooling1D(name="gap")(x)

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    outputs = _head(x, num_classes)
    model = tf.keras.Model(inputs, outputs, name=f"{model_type}_classifier")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model
