# HeartLens AI — Training Data Summary

## MIT-BIH Arrhythmia Database

**Source**: PhysioNet (https://physionet.org/content/mitdb/)
**Recordings**: 48 half-hour ECG recordings (2 leads each, 360 Hz)

## Class Distribution (v1.1 — cleaned labels)

After removing pathological proxy mappings (see Clinical Label Audit below),
only 3 cleanly identifiable classes remain:

| Class | Name | Source Annotation | Segments |
|-------|------|-------------------|----------|
| 0 | Normal Sinus Rhythm | N (normal beat) | ~29,851 |
| 1 | Atrial Fibrillation (proxy) | A (atrial premature beat) | ~705 |
| 2 | Premature Ventricular Contraction | V (PVC) | ~2,227 |
| 3 | Tachycardia | *no training data* | — |
| 4 | Bradycardia | *no training data* | — |
| 5 | ST Abnormality | *no training data* | — |

**Classes 3–5 have no training data** after removing dangerous proxy mappings.
These classifier outputs are inactive until properly annotated data is sourced.

## Clinical Label Audit

The following proxy mappings were present in v1.0 and **removed in v1.1**:

| Removed Mapping | Original Class | Reason for Removal |
|-----------------|----------------|--------------------|
| LBBB (L) → Normal | 0 | Pathological conduction abnormality |
| RBBB (R) → Normal | 0 | Pathological conduction abnormality |
| Ventricular flutter (!) → Tachy | 3 | Pre-arrest rhythm, distinct from SVT |
| Fusion beat (F) → Tachy | 3 | Non-specific |
| Paced (/) → Brady | 4 | Pacing is treatment, not rhythm diagnosis |
| Fusion paced (f) → Brady | 4 | Same |
| Atrial escape (E) → ST Abn | 5 | Different electrophysiological mechanism |
| Nodal escape (J) → ST Abn | 5 | Same |
| Aberrated APB (a) → ST Abn | 5 | Same |
| Supraventricular premature (S) → ST Abn | 5 | Same |
| Nodal premature (j) → ST Abn | 5 | Same |

## Annotation → Class Mapping (v1.1)

| MIT-BIH Symbol | Beat Description | Mapped Class | Clinical Validity |
|----------------|-----------------|--------------|-------------------|
| N | Normal beat | 0 — Normal | ✓ Correct |
| A | Atrial premature beat | 1 — AFib (proxy) | ⚠ Proxy — replace with AFDB |
| V | Premature ventricular contraction | 2 — PVC | ✓ Correct |

## Train / Validation / Test Split

Using **patient-level (record-level) splitting** to prevent data leakage:
- 70% of records → training
- 15% of records → validation
- 15% of records → test

## Normalization

Each segment is normalized to [-1, 1] using centered scaling:
1. Subtract segment mean
2. Divide by max absolute deviation

This is reproduced identically in firmware at inference time
(per-buffer centering + scaling before TFLite input quantization).
