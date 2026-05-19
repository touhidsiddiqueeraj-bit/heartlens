# HeartLens AI — Training Data Summary

## MIT-BIH Arrhythmia Database

**Source**: PhysioNet (https://physionet.org/content/mitdb/)
**Recordings**: 48 half-hour ECG recordings (2 leads each, 360 Hz)
**Total segments extracted**: 35,864 (10-second windows centered on R-peaks)

## Class Distribution

| Class | Name | Segments |
|-------|------|----------|
| 0 | Normal Sinus Rhythm | 29,851 |
| 1 | Atrial Fibrillation (AFib) | 705 |
| 2 | Premature Ventricular Contraction (PVC) | 2,227 |
| 3 | Tachycardia | 455 |
| 4 | Bradycardia | 2,556 |
| 5 | ST Abnormality | 70 |
| **Total** | | **35,864** |

## Train / Validation / Test Split (capped at 5,000 per class)

| Set | Normal | AFib | PVC | Tachy | Brady | ST Abn | Total |
|-----|--------|------|-----|-------|-------|--------|-------|
| Train | 1,399 | 493 | 1,399 | 318 | 1,399 | 48 | 5,056 |
| Validation | 300 | 106 | 300 | 68 | 300 | 11 | 1,085 |
| Test | 301 | 106 | 301 | 69 | 301 | 11 | 1,089 |

## Annotation → Class Mapping

| MIT-BIH Symbol | Beat Description | Mapped Class |
|----------------|-----------------|--------------|
| N | Normal beat | 0 — Normal |
| L | Left bundle branch block | 0 — Normal |
| R | Right bundle branch block | 0 — Normal |
| A | Atrial premature beat | 1 — AFib |
| V | Premature ventricular contraction | 2 — PVC |
| ! | Ventricular flutter | 3 — Tachycardia |
| F | Fusion beat | 3 — Tachycardia |
| / | Paced beat | 4 — Bradycardia |
| f | Fusion of paced beat | 4 — Bradycardia |
| E | Atrial escape beat | 5 — ST Abnormality |
| J | Nodal escape beat | 5 — ST Abnormality |
| S | Supraventricular premature beat | 5 — ST Abnormality |
