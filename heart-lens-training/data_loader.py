#!/usr/bin/env python3
"""MIT-BIH Arrhythmia Database loader.
Downloads from PhysioNet if not present locally.
Expected structure:
  /path/to/mitdb/
    100.dat, 100.hea, 100.atr  (and 47 other recordings)
"""

import os
import numpy as np

try:
    import wfdb
except ImportError:
    raise ImportError("Install wfdb: pip install wfdb")

SAMPLE_RATE = 360
WINDOW_SEC = 10
WINDOW_SAMPLES = SAMPLE_RATE * WINDOW_SEC
NUM_CLASSES = 6

# MIT-BIH annotation symbols → our class mapping
SYM_TO_CLASS = {
    'N': 0,  # Normal
    'L': 0,  # Left bundle branch block → Normal (for denoiser)
    'R': 0,  # Right bundle branch block → Normal
    'A': 1,  # Atrial premature → proxy for AFib
    'V': 2,  # Premature ventricular contraction
    '!': 3,  # Ventricular flutter → proxy for Tachy
    'F': 3,  # Fusion beat → proxy for Tachy
    '/': 4,  # Paced → proxy for Brady
    'f': 4,  # Fusion of paced → proxy for Brady
    'E': 5,  # Atrial escape → proxy for ST abn
    'J': 5,  # Nodal escape → proxy for ST abn
    'a': 5,  # Aberrated atrial premature → ST abn
    'S': 5,  # Supraventricular premature → ST abn
    'j': 5,  # Nodal premature → ST abn
}

def download_mitdb(target_dir="./mitdb"):
    """Download MIT-BIH Arrhythmia Database."""
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)

    records = [str(r) for r in (
        list(range(100, 110)) + list(range(111, 120)) +
        list(range(121, 125)) + [200, 201, 202, 203, 205, 207, 208, 209,
                                 210, 212, 213, 214, 215, 217, 219, 220,
                                 221, 222, 223, 228, 230, 231, 232, 233, 234]
    )]

    print(f"Downloading {len(records)} MIT-BIH recordings to {target_dir}...")
    wfdb.dl_database('mitdb', target_dir, records=records)
    print("Download complete.")
    return target_dir


def load_record(path, record_name):
    """Load a single MIT-BIH recording and return signals + annotations."""
    record = wfdb.rdrecord(os.path.join(path, record_name), sampto=60 * SAMPLE_RATE * 10)  # up to 10 min
    ann = wfdb.rdann(os.path.join(path, record_name), 'atr', sampto=60 * SAMPLE_RATE * 10)
    signal = record.p_signal[:, 0]  # Use MLII lead
    return signal, ann


def extract_segments(signal, ann, window_samples=WINDOW_SAMPLES):
    """Extract labeled segments from a recording.
    Returns list of (segment_array, class_id) tuples.
    """
    segments = []
    r_peaks = ann.sample
    symbols = ann.symbol

    for i, (peak, sym) in enumerate(zip(r_peaks, symbols)):
        cls = SYM_TO_CLASS.get(sym, None)
        if cls is None:
            continue

        start = peak - window_samples // 2
        end = start + window_samples

        if start < 0 or end >= len(signal):
            continue

        seg = signal[start:end].copy()

        # Normalize to [-1, 1] per segment
        seg_max = max(abs(seg.min()), abs(seg.max()))
        if seg_max > 0:
            seg = seg / seg_max

        segments.append((seg, cls))

    return segments


def load_all_segments(data_dir="./mitdb", max_per_class=None):
    """Load all segments from all recordings."""
    if not os.path.exists(data_dir) or not any(f.endswith('.dat') for f in os.listdir(data_dir)):
        if os.path.exists(data_dir):
            import shutil
            shutil.rmtree(data_dir)
        print(f"Directory {data_dir} not found. Downloading...")
        download_mitdb(data_dir)

    records = sorted(set(
        f.replace('.dat', '') for f in os.listdir(data_dir) if f.endswith('.dat')
    ))

    all_segments = []
    for rec_name in records:
        try:
            signal, ann = load_record(data_dir, rec_name)
            segs = extract_segments(signal, ann)
            all_segments.extend(segs)
            print(f"  {rec_name}: {len(segs)} segments")
        except Exception as e:
            print(f"  {rec_name}: SKIP ({e})")
            continue

    # Group by class
    by_class = {c: [] for c in range(NUM_CLASSES)}
    for seg, cls in all_segments:
        by_class[cls].append(seg)

    print(f"\nTotal segments: {len(all_segments)}")
    for c in range(NUM_CLASSES):
        print(f"  Class {c}: {len(by_class[c])} segments")
        if max_per_class and len(by_class[c]) > max_per_class:
            by_class[c] = by_class[c][:max_per_class]

    return by_class


def split_dataset(by_class, train_ratio=0.7, val_ratio=0.15):
    """Split by-class segments into train/val/test."""
    from sklearn.model_selection import train_test_split

    X_train, X_val, X_test = [], [], []
    y_train, y_val, y_test = [], [], []

    for cls, segs in by_class.items():
        if len(segs) == 0:
            continue
        X = np.array(segs)
        y = np.full(len(segs), cls)

        X_t, X_tmp, y_t, y_tmp = train_test_split(
            X, y, test_size=(1 - train_ratio), random_state=42, stratify=y
        )
        X_v, X_te, y_v, y_te = train_test_split(
            X_tmp, y_tmp, test_size=val_ratio / (val_ratio + 0.15), random_state=42
        )

        X_train.append(X_t)
        X_val.append(X_v)
        X_test.append(X_te)
        y_train.append(y_t)
        y_val.append(y_v)
        y_test.append(y_te)

    X_train = np.concatenate(X_train)
    X_val = np.concatenate(X_val)
    X_test = np.concatenate(X_test)
    y_train = np.concatenate(y_train)
    y_val = np.concatenate(y_val)
    y_test = np.concatenate(y_test)

    # Reshape for Conv1D: (samples, time, channels)
    X_train = X_train.reshape(-1, WINDOW_SAMPLES, 1)
    X_val = X_val.reshape(-1, WINDOW_SAMPLES, 1)
    X_test = X_test.reshape(-1, WINDOW_SAMPLES, 1)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


if __name__ == "__main__":
    by_class = load_all_segments(max_per_class=2000)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_dataset(by_class)
    print(f"\nTrain: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"y distribution: train={np.bincount(y_train)}, val={np.bincount(y_val)}, test={np.bincount(y_test)}")
