#!/usr/bin/env python3
"""MIT-BIH Arrhythmia Database loader with patient-level splitting and normalization tracking.

Expected structure:
  /path/to/mitdb/
    100.dat, 100.hea, 100.atr  (and 47 other recordings)
"""

import os
import numpy as np
from collections import namedtuple
from typing import List, Tuple, Dict

try:
    import wfdb
except ImportError:
    raise ImportError("Install wfdb: pip install wfdb")

SAMPLE_RATE = 360
WINDOW_SEC = 10
WINDOW_SAMPLES = SAMPLE_RATE * WINDOW_SEC
NUM_CLASSES = 6

# Tracking normalization parameters per segment for firmware deployment.
# offset = segment mean, scale = max(abs(segment - offset))
NormalizationParams = namedtuple('NormalizationParams', ['scale', 'offset'])

# Canonical MIT-BIH Arrhythmia Database record numbers
RECORDS_LIST = [
    100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
    111, 112, 113, 114, 115, 116, 117, 118, 119,
    121, 122, 123, 124,
    200, 201, 202, 203, 205, 207, 208, 209,
    210, 212, 213, 214, 215, 217, 219, 220,
    221, 222, 223, 228, 230, 231, 232, 233, 234
]

# Each MIT-BIH recording maps to a patient/record identifier.
# By default, each record number identifies a unique patient recording.
PATIENT_RECORD = {str(r): r for r in RECORDS_LIST}

# Annotation symbol -> class mapping.
# Only cleanly identifiable classes are mapped. Pathological/proxy labels removed.
SYM_TO_CLASS = {
    'N': 0,  # Normal beat
    # WARNING: 'A' (atrial premature beat) is used as a PROXY for AFib (class 1).
    # Atrial premature beats are NOT the same as atrial fibrillation.
    # This mapping is a stand-in only. Replace with properly annotated AFib
    # data (e.g. from MIT-BIH AF Database or CUDB) for production use.
    'A': 1,  # Atrial premature beat -> proxy for AFib
    'V': 2,  # Premature ventricular contraction
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
    record = wfdb.rdrecord(os.path.join(path, record_name), sampto=60 * SAMPLE_RATE * 10)
    ann = wfdb.rdann(os.path.join(path, record_name), 'atr', sampto=60 * SAMPLE_RATE * 10)
    signal = record.p_signal[:, 0]  # Use MLII lead
    return signal, ann


def extract_segments(signal, ann, window_samples=WINDOW_SAMPLES):
    """Extract labeled segments from a recording.

    Returns list of (segment, class_id, NormalizationParams) tuples.
    Normalization: offset = mean(segment), scale = max(abs(segment - offset)).
    """
    segments: List[Tuple[np.ndarray, int, NormalizationParams]] = []
    r_peaks = ann.sample
    symbols = ann.symbol

    for peak, sym in zip(r_peaks, symbols):
        cls = SYM_TO_CLASS.get(sym, None)
        if cls is None:
            continue

        start = peak - window_samples // 2
        end = start + window_samples

        if start < 0 or end >= len(signal):
            continue

        seg = signal[start:end].copy()

        offset = float(np.mean(seg))
        centered = seg - offset
        scale = float(np.max(np.abs(centered)))
        if scale > 1e-12:
            seg = centered / scale
        else:
            scale = 1.0

        segments.append((seg, cls, NormalizationParams(scale, offset)))

    return segments


def load_record_segments(data_dir="./mitdb", max_per_class=None):
    """Load all segments grouped by recording name.

    Returns dict: record_name -> list of (segment, class_id, NormalizationParams).
    """
    if not os.path.exists(data_dir) or not any(f.endswith('.dat') for f in os.listdir(data_dir)):
        if os.path.exists(data_dir):
            import shutil
            shutil.rmtree(data_dir)
        print(f"Directory {data_dir} not found. Downloading...")
        download_mitdb(data_dir)

    records = sorted(set(
        f.replace('.dat', '') for f in os.listdir(data_dir) if f.endswith('.dat')
    ))

    record_segments: Dict[str, List[Tuple[np.ndarray, int, NormalizationParams]]] = {}
    class_counts = {c: 0 for c in range(NUM_CLASSES)}

    for rec_name in records:
        try:
            signal, ann = load_record(data_dir, rec_name)
            segs = extract_segments(signal, ann)
            if segs:
                record_segments[rec_name] = segs
                for _, cls, _ in segs:
                    class_counts[cls] += 1
            print(f"  {rec_name}: {len(segs)} segments")
        except Exception as e:
            print(f"  {rec_name}: SKIP ({e})")
            continue

    total = sum(len(v) for v in record_segments.values())
    print(f"\nTotal segments: {total}")
    for c in range(NUM_CLASSES):
        print(f"  Class {c}: {class_counts.get(c, 0)} segments")

    # Apply max_per_class by collecting all, truncating per class, regrouping by record
    if max_per_class is not None:
        all_items: List[Tuple[str, np.ndarray, int, NormalizationParams]] = []
        for rec_name, segs in record_segments.items():
            for seg, cls, norm in segs:
                all_items.append((rec_name, seg, cls, norm))

        per_class: Dict[int, List[Tuple[str, np.ndarray, NormalizationParams]]] = {c: [] for c in range(NUM_CLASSES)}
        for rec_name, seg, cls, norm in all_items:
            if len(per_class[cls]) < max_per_class:
                per_class[cls].append((rec_name, seg, norm))

        record_segments.clear()
        for cls, items in per_class.items():
            for rec_name, seg, norm in items:
                record_segments.setdefault(rec_name, []).append((seg, cls, norm))

    return record_segments


def load_all_segments(data_dir="./mitdb", max_per_class=None):
    """Load all segments from all recordings.

    DEPRECATED: Use load_record_segments() instead. This function drops
    record-grouping and normalization-param information. Prefer the newer
    API for patient-level splits and firmware deployment workflows.

    Returns dict of class_id -> list of segment arrays.
    """
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

    by_class = {c: [] for c in range(NUM_CLASSES)}
    for seg, cls, _ in all_segments:
        by_class[cls].append(seg)

    print(f"\nTotal segments: {len(all_segments)}")
    for c in range(NUM_CLASSES):
        print(f"  Class {c}: {len(by_class[c])} segments")
        if max_per_class and len(by_class[c]) > max_per_class:
            by_class[c] = by_class[c][:max_per_class]

    return by_class


def split_dataset(by_class, train_ratio=0.7, val_ratio=0.15):
    """Split by-class segments into train/val/test at the SEGMENT level.

    DEPRECATED: Use split_by_patient() for record-level splits that prevent
    patient data leakage between train and test sets.

    Returns (X_train, y_train), (X_val, y_val), (X_test, y_test).
    """
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

    X_train = X_train.reshape(-1, WINDOW_SAMPLES, 1)
    X_val = X_val.reshape(-1, WINDOW_SAMPLES, 1)
    X_test = X_test.reshape(-1, WINDOW_SAMPLES, 1)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def split_by_patient(record_segments, record_names=None, train_ratio=0.7, val_ratio=0.15, random_state=42):
    """Split recordings into train/val/test at the RECORD level.

    Prevents patient data leakage: all segments from one recording go to
    the same split.

    Args:
        record_segments: dict of record_name -> list of (segment, class_id, ...)
                         as returned by load_record_segments().
        record_names: optional list of record names to use (default: all keys).
        train_ratio: proportion of records for training.
        val_ratio: proportion of records for validation.
        random_state: RNG seed for reproducible splits.

    Returns (X_train, y_train), (X_val, y_val), (X_test, y_test)
    where each X has shape (samples, WINDOW_SAMPLES, 1).
    """
    if record_names is None:
        record_names = list(record_segments.keys())

    rng = np.random.default_rng(random_state)
    shuffled = list(record_names)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))

    train_recs = shuffled[:n_train]
    val_recs = shuffled[n_train:n_train + n_val]
    test_recs = shuffled[n_train + n_val:]

    def collect(recs):
        X, y = [], []
        for rec in recs:
            for seg, cls, _ in record_segments.get(rec, []):
                X.append(seg)
                y.append(cls)
        return np.array(X), np.array(y)

    X_train, y_train = collect(train_recs)
    X_val, y_val = collect(val_recs)
    X_test, y_test = collect(test_recs)

    X_train = X_train.reshape(-1, WINDOW_SAMPLES, 1)
    X_val = X_val.reshape(-1, WINDOW_SAMPLES, 1)
    X_test = X_test.reshape(-1, WINDOW_SAMPLES, 1)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


if __name__ == "__main__":
    record_segs = load_record_segments(max_per_class=2000)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_by_patient(record_segs)
    print(f"\nTrain: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"y distribution: train={np.bincount(y_train)}, val={np.bincount(y_val)}, test={np.bincount(y_test)}")
