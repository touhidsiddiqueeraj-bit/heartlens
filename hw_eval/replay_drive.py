#!/usr/bin/env python3
"""Experiment 6 — Hardware-domain robustness via DAC->ADC loopback replay.

Drives the ESP32-S3 in REPLAY_MODE over serial:

  MIT-BIH test segment -> [10x repeated 3600 samples] -> serial ->
  ESP32 DAC (GPIO17) -> jumper wire -> ESP32 ADC (GPIO4) -> denoiser ->
  classifier -> "HLR_RES <class> <conf>" -> parsed back on PC

Each 10-s replay runs in ~1 s of wall time (REPLAY_RATE_MULT=10).
The digital-leg F1 comes from the same segments scored by the int8
model on the PC; the hardware-leg F1 from the board's own inference.

Usage:
    # Step 1 (on PC): run the digital leg + drive the board
    python3 hw_eval/replay_drive.py --port /dev/ttyACM0 \
        --model heart-lens-training/models/robust_classifier_int8.tflite

    # Step 2 (offline): compute F1 and Delta-F1 table
    python3 hw_eval/compute_delta.py --digital hw_eval/captures/digital.json \
        --hardware hw_eval/captures/hardware.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "heart-lens-training"))
from data_loader import WINDOW_SAMPLES, load_record_segments, split_by_patient  # noqa: E402

CLASS_NAMES = ["Normal", "APB", "PVC"]


def load_test_segments(data_dir="./mitdb", max_per_class=200, seed=42):
    """Load a balanced patient-level test subset for replay."""
    record_segs = load_record_segments(data_dir)
    (_, _), (_, _), (X_te, y_te) = split_by_patient(record_segs)
    rng = np.random.default_rng(seed)
    per_class = {}
    for cls in range(3):
        idx = np.where(y_te == cls)[0]
        if len(idx) > max_per_class:
            idx = rng.choice(idx, max_per_class, replace=False)
        per_class[cls] = idx
    keep = np.concatenate(list(per_class.values()))
    return X_te[keep].reshape(-1, WINDOW_SAMPLES), y_te[keep]


def digital_f1(model_path, X_te):
    """Score the same segments with the int8 TFLite model on the PC."""
    import tensorflow as tf
    interp = tf.lite.Interpreter(model_path=model_path)
    interp.allocate_tensors()
    in_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]
    in_s, in_z = in_d["quantization"]
    out_s, out_z = out_d["quantization"]
    preds = []
    for seg in X_te:
        x = np.round(seg / in_s + in_z).clip(-128, 127).astype(np.int8)
        interp.set_tensor(in_d["index"], x.reshape(1, WINDOW_SAMPLES, 1))
        interp.invoke()
        raw = interp.get_tensor(out_d["index"])[0]
        probs = (raw.astype(np.float32) - out_z) * out_s
        preds.append(int(np.argmax(probs)))
    return np.array(preds)


def to_dac_bytes(seg):
    """Normalize [-1,1] -> 8-bit DAC values (0..255), repeated 10x."""
    center = np.mean(seg)
    dev = np.max(np.abs(seg - center))
    norm = (seg - center) / dev if dev > 1e-12 else seg - center
    norm = np.clip(norm, -1.0, 1.0)
    vals = ((norm + 1.0) / 2.0 * 255).astype(np.uint8)
    return np.tile(vals, 10).tobytes()  # 3600 bytes = 10 s @ 360 Hz


def open_serial(port, baud=115200):
    import serial
    return serial.Serial(port, baud, timeout=10)


def replay_segments(ser, X_te, y_te):
    """Stream segments, parse HLR_RES lines, return hardware predictions."""
    preds, skipped = [], []
    for i, (seg, y) in enumerate(zip(X_te, y_te)):
        payload = to_dac_bytes(seg)
        ser.write(b"HLR" + len(payload).to_bytes(2, "little") + payload)
        ser.flush()
        # Read until result line (timeout guards against stuck frames)
        deadline = time.time() + 15
        got = None
        while time.time() < deadline:
            line = ser.readline().decode(errors="ignore").strip()
            if line.startswith("HLR_RES"):
                got = line
                break
            if line.startswith("HLR_ERR") or line.startswith("HLR_UNCLR"):
                got = line
                break
            if not line:
                continue
        if got and got.startswith("HLR_RES"):
            parts = got.split()
            preds.append((int(parts[1]), float(parts[2])))
        else:
            skipped.append(i)
        if (i + 1) % 50 == 0:
            print(f"  replayed {i + 1}/{len(X_te)} segments")
    return preds, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True, help="serial port of the board")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--model", required=True,
                    help="path to robust_classifier_int8.tflite")
    ap.add_argument("--data-dir", default="./mitdb")
    ap.add_argument("--max-per-class", type=int, default=200)
    ap.add_argument("--out", default="hw_eval/captures",
                    help="output dir for digital.json / hardware.json")
    ap.add_argument("--skip-hardware", action="store_true",
                    help="only compute the digital leg")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    X_te, y_te = load_test_segments(args.data_dir, args.max_per_class)
    print(f"Test segments: {len(y_te)} "
          f"({ {c: int(np.sum(y_te == c)) for c in range(3)} })")

    # Digital leg: int8 TFLite on PC
    pred_digital = digital_f1(args.model, X_te)
    with open(os.path.join(args.out, "digital.json"), "w") as f:
        json.dump({"segments": [int(p) for p in pred_digital],
                   "labels": [int(l) for l in y_te]}, f)
    print("Digital leg done -> digital.json")

    # Hardware leg: DAC->ADC loopback replay on the board
    if args.skip_hardware:
        return
    ser = open_serial(args.port, args.baud)
    time.sleep(2)
    ser.reset_input_buffer()
    print("Streaming segments to board (REPLAY_MODE)...")
    preds, skipped = replay_segments(ser, X_te, y_te)
    ser.close()

    hw = {"labels": [int(l) for l in y_te],
          "preds": [p for p, _ in preds],
          "confs": [c for _, c in preds],
          "skipped": skipped}
    with open(os.path.join(args.out, "hardware.json"), "w") as f:
        json.dump(hw, f, indent=2)
    print(f"Hardware leg done ({len(preds)} results, {len(skipped)} skipped)")
    print("Next: python3 hw_eval/compute_delta.py --digital "
          "hw_eval/captures/digital.json --hardware hw_eval/captures/hardware.json")


if __name__ == "__main__":
    main()
