#!/usr/bin/env python3
"""SQI gate ablation (M11): measure clean false-reject, corrupted reject, downstream ΔF1.

Heuristic gate from HeartLens_Firmware/src/ecg_processor.cpp:signalQualityOk
  - flat: range <8
  - saturated: >5% samples within range/20 of extremes
  - hf-noise: diffEnergy/sigEnergy > threshold^2 (threshold=0.35)

We sweep threshold and measure on clean test set vs corrupted (SNR 0 mixed) vs downstream.
Writes results/sqi_ablation.json + sqi_ablation.png
"""
import sys, pathlib, json, numpy as np
sys.path.insert(0, str(pathlib.Path("/home/touhid/heartlens/heart-lens-training")))
from data_loader import WINDOW_SAMPLES
from train_classifier import load_data_with_record_tracking, record_level_split
from noise_pipeline import add_all_noise
import tensorflow as tf
from sklearn.metrics import f1_score

TR = pathlib.Path("/home/touhid/heartlens/heart-lens-training")
OUT = TR/"results"

def signal_quality_ok(samples, threshold=0.35):
    if len(samples)<32: return False
    minV=np.min(samples); maxV=np.max(samples)
    mean=np.mean(samples)
    rng=int(maxV-minV)
    if rng<8: return False
    edge=rng//20+1
    saturated=np.sum((samples<=minV+edge)|(samples>=maxV-edge))
    if saturated/len(samples)>0.05: return False
    sigEnergy=np.sum((samples-mean)**2)
    diffEnergy=np.sum(np.diff(samples)**2)
    noiseRatio= diffEnergy/sigEnergy if sigEnergy>0 else 1.0
    if noiseRatio > threshold*threshold: return False
    return True

def main():
    by_class = load_data_with_record_tracking(str(TR/"mitdb"), max_per_class=3000)
    (X_tr, y_tr), _, (X_te, y_te) = record_level_split(by_class, train_ratio=0.7, val_ratio=0.15, seed=42)
    X_te = X_te.reshape(-1, WINDOW_SAMPLES, 1)
    y_te = np.array(y_te)
    print(f"Test {X_te.shape} {np.bincount(y_te)}")
    # scale to int16-like for SQI (mimic ADC counts): normalize to [-1,1] -> *1000
    X_int = (X_te.squeeze()*1000).astype(np.int16)

    # Load classifier
    clf_path = TR/"models"/"robust_classifier.keras"
    if clf_path.exists():
        clf = tf.keras.models.load_model(clf_path)
        print(f"Loaded {clf_path}")
    else:
        print("No robust_classifier, training quick")
        from models import build_classifier
        X_tr = X_tr.reshape(-1, WINDOW_SAMPLES, 1).astype(np.float32)
        clf = build_classifier()
        clf.fit(X_tr, y_tr, epochs=10, batch_size=64, verbose=0)

    thresholds = [0.2,0.25,0.3,0.35,0.4,0.5,0.7,1.0]
    results=[]
    # Precompute corrupted version (SNR 0 mixed)
    X_corr = np.stack([add_all_noise(x.flatten(), snr_db=0) for x in X_te]).reshape(-1, WINDOW_SAMPLES, 1)
    X_corr_int = (X_corr.squeeze()*1000).astype(np.int16)

    # Baseline F1 without gate
    pred_all = np.argmax(clf.predict(X_te.astype(np.float32), verbose=0), axis=1)
    macro_all = float(np.mean(f1_score(y_te, pred_all, average=None, zero_division=0)))

    for thr in thresholds:
        # clean reject rate
        clean_ok = np.array([signal_quality_ok(x, threshold=thr) for x in X_int])
        clean_reject = 1 - clean_ok.mean()
        corr_ok = np.array([signal_quality_ok(x, threshold=thr) for x in X_corr_int])
        corr_reject = 1 - corr_ok.mean()  # we want high reject for corrupted
        # downstream F1 on kept windows only
        keep_idx = np.where(clean_ok)[0]
        if len(keep_idx)>0:
            pred_keep = pred_all[keep_idx]
            y_keep = y_te[keep_idx]
            macro_keep = float(np.mean(f1_score(y_keep, pred_keep, average=None, zero_division=0)))
        else:
            macro_keep = None
        # Also corrupted downstream: if we kept corrupted, F1 would be low; gate should reject them so not counted
        pred_corr = np.argmax(clf.predict(X_corr.astype(np.float32), verbose=0), axis=1)
        corr_keep_idx = np.where(corr_ok)[0]
        if len(corr_keep_idx)>0:
            macro_corr_keep = float(np.mean(f1_score(y_te[corr_keep_idx], pred_corr[corr_keep_idx], average=None, zero_division=0)))
        else:
            macro_corr_keep = None

        results.append({"threshold": thr, "clean_reject": float(clean_reject), "corr_reject": float(corr_reject),
                        "macro_clean_all": macro_all, "macro_clean_kept": macro_keep, "macro_corr_kept": macro_corr_keep,
                        "kept_clean": int(clean_ok.sum()), "kept_corr": int(corr_ok.sum())})
        print(f"thr={thr:.2f} clean_reject={clean_reject:.3f} corr_reject={corr_reject:.3f} macro_keep={macro_keep} corr_keep={macro_corr_keep}")

    with open(OUT/"sqi_ablation.json","w") as f: json.dump(results,f,indent=2)
    print(f"Saved {OUT/'sqi_ablation.json'}")

    # Plot
    import matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,4))
    thrs=[r["threshold"] for r in results]
    ax1.plot(thrs, [r["clean_reject"] for r in results], 'o-', label='clean false-reject', color='#d95f02')
    ax1.plot(thrs, [r["corr_reject"] for r in results], 's-', label='corrupted reject', color='#1b9e77')
    ax1.set_xlabel("threshold"); ax1.set_ylabel("reject rate"); ax1.set_ylim(0,1); ax1.legend(); ax1.grid(alpha=0.3); ax1.set_title("SQI reject rates")
    ax2.plot(thrs, [r["macro_clean_kept"] if r["macro_clean_kept"] is not None else 0 for r in results], 'o-', label='clean kept F1', color='#7570b3')
    ax2.plot(thrs, [r["macro_corr_kept"] if r["macro_corr_kept"] is not None else 0 for r in results], 's-', label='corr kept F1', color='#e7298a')
    ax2.axhline(macro_all, color='k', linestyle='--', label='no gate')
    ax2.set_xlabel("threshold"); ax2.set_ylabel("macro F1"); ax2.set_ylim(0,1); ax2.legend(); ax2.grid(alpha=0.3); ax2.set_title("Downstream F1")
    plt.suptitle("SQI gate ablation (current thr=0.35)")
    plt.tight_layout()
    fig.savefig(OUT/"sqi_ablation.png", dpi=150)
    print(f"Saved {OUT/'sqi_ablation.png'}")

if __name__=="__main__":
    main()
