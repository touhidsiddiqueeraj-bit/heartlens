#!/usr/bin/env python3
"""MIT-BIH Atrial Fibrillation Database (afdb) loader — rhythm-level AF.

AF annotations in afdb are RHYTHM-level (AFIB, AFL, N, ...), not beat-level.
This addresses the methodological issue that beat-level windows cannot
represent atrial fibrillation (review #2).

Windows: 10 seconds at 360 Hz (resampled from native 250 Hz).
Label: AF if >= 80% of the window is annotated AFIB/AFL,
       Normal if >= 80% is annotated N.
       Windows that are neither are discarded.

Usage:
    from afdb_loader import load_afdb_rhythm
    X, y = load_afdb_rhythm("./afdb", window_sec=10)   # y in {0: Normal, 1: AF}
"""

import os
import numpy as np
from scipy import signal as sg

try:
    import wfdb
except ImportError:
    raise ImportError("Install wfdb: pip install wfdb")

SAMPLE_RATE = 250            # afdb native rate
TARGET_RATE = 360            # match mitdb / firmware

AFDB_RECORDS = [
    "04015", "04043", "04048", "04126", "04746", "04908", "04936", "05091",
    "05121", "05261", "06426", "06453", "06995", "07162", "07859", "07910",
    "08215", "08219", "08378", "08405", "08434", "08455", "08479", "09034",
    "09268",
]

# afdb rhythm annotation symbols — only true AF rhythms are class 1
AF_SYMBOLS = {"AFIB", "AFL"}
NORMAL_SYMBOLS = {"N"}


def download_afdb(target_dir="./afdb"):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
    recs = list(AFDB_RECORDS)
    print(f"Downloading {len(recs)} afdb recordings to {target_dir}...")
    wfdb.dl_database('afdb', target_dir, records=recs)
    print("Download complete.")
    return target_dir


def _record_timeline(ann):
    """Build (start_sample, end_sample, label) intervals from annotations."""
    intervals = []
    samples = list(ann.sample)
    symbols = [s.upper() for s in ann.symbol]
    n = len(samples)
    for i in range(n):
        end = samples[i + 1] if i + 1 < n else samples[i] + 10 * SAMPLE_RATE
        if symbols[i] in AF_SYMBOLS:
            label = 1
        elif symbols[i] in NORMAL_SYMBOLS:
            label = 0
        else:
            continue
        intervals.append((samples[i], end, label))
    return intervals


def load_afdb_rhythm(data_dir="./afdb", window_sec=10, stride_frac=0.25,
                     purity=0.8, max_per_class=None):
    """Return (X, y) rhythm windows: X (n, window_sec*360, 1), y in {0,1}."""
    if not os.path.exists(data_dir) or not any(
            f.endswith('.dat') for f in os.listdir(data_dir)):
        if os.path.exists(data_dir):
            import shutil
            shutil.rmtree(data_dir)
        download_afdb(data_dir)

    window_samples = int(window_sec * TARGET_RATE)
    stride = max(1, int(window_samples * stride_frac))

    X, y = [], []
    records = sorted(set(f.replace('.dat', '') for f in os.listdir(data_dir)
                         if f.endswith('.dat')))
    for rec in records:
        try:
            record = wfdb.rdrecord(os.path.join(data_dir, rec),
                                   sampto=60 * SAMPLE_RATE * 10)
            ann = wfdb.rdann(os.path.join(data_dir, rec), 'atr',
                             sampto=60 * SAMPLE_RATE * 10)
        except Exception as e:
            print(f"  {rec}: SKIP ({e})")
            continue
        sig = record.p_signal[:, 0]
        # Resample 250 -> 360 Hz
        sig = sg.resample_poly(sig, TARGET_RATE, SAMPLE_RATE)
        timeline = _record_timeline(ann)
        # Map timeline to resampled-domain boundaries
        boundaries = []
        for s, e, lab in timeline:
            boundaries.append((int(s * TARGET_RATE / SAMPLE_RATE),
                               int(e * TARGET_RATE / SAMPLE_RATE), lab))

        n_windows = 0
        for start in range(0, len(sig) - window_samples + 1, stride):
            end = start + window_samples
            af_frac, n_frac = 0.0, 0.0
            for bs, be, lab in boundaries:
                if be <= start or bs >= end:
                    continue
                ov = min(be, end) - max(bs, start)
                if ov <= 0:
                    continue
                if lab == 1:
                    af_frac += ov
                else:
                    n_frac += ov
            total = window_samples
            af_frac /= total
            n_frac /= total
            if af_frac >= purity:
                label = 1
            elif n_frac >= purity:
                label = 0
            else:
                continue
            seg = sig[start:end].copy()
            center = np.mean(seg)
            dev = np.max(np.abs(seg - center))
            if dev > 1e-12:
                seg = (seg - center) / dev
            X.append(seg.reshape(-1, 1))
            y.append(label)
            n_windows += 1
        print(f"  {rec}: {n_windows} rhythm windows")

    X = np.array(X)
    y = np.array(y)
    if max_per_class:
        keep = np.zeros(len(y), dtype=bool)
        for cls in np.unique(y):
            idx = np.where(y == cls)[0]
            if len(idx) > max_per_class:
                idx = np.random.choice(idx, max_per_class, replace=False)
            keep[idx] = True
        X, y = X[keep], y[keep]
    print(f"\nTotal rhythm windows: {len(y)}  "
          f"(Normal={np.sum(y == 0)}, AF={np.sum(y == 1)})")
    return X, y


if __name__ == "__main__":
    X, y = load_afdb_rhythm(max_per_class=3000)
    print(f"Shapes: X={X.shape}, y={y.shape}")
