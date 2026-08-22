#!/usr/bin/env python3
"""Per-artifact noise robustness (M10): separate curves for BW, motion, PLI, EMG, mixed.

Uses same 7 SNR levels (0,5,10,15,20,30,40) and 3 front-ends (Raw, Butterworth, AE)
but loops per noise type from noise_pipeline.

Writes:
  results/per_artifact_noise.json
  results/per_artifact_noise.png (5 panels)
  results/noise_cost_table.json (AE size/latency vs Butterworth gain)

Fast: inference only, ~2 min.
"""
import sys, pathlib, json, numpy as np
sys.path.insert(0, str(pathlib.Path("/home/touhid/heartlens/heart-lens-training")))
import tensorflow as tf
from scipy import signal as sg
from sklearn.metrics import f1_score

TR = pathlib.Path("/home/touhid/heartlens/heart-lens-training")
OUT = TR/"results"
from data_loader import WINDOW_SAMPLES
from train_classifier import load_data_with_record_tracking, record_level_split
from noise_pipeline import add_baseline_wander, add_motion_artifact, add_pli, add_emg_noise

NOISE_FNS = {
    "baseline_wander": lambda ecg, scale: add_baseline_wander(ecg, amplitude=0.3*scale),
    "motion": lambda ecg, scale: add_motion_artifact(ecg, amplitude=0.5*scale),
    "pli": lambda ecg, scale: add_pli(ecg, amplitude=0.15*scale),
    "emg": lambda ecg, scale: add_emg_noise(ecg, amplitude=0.3*scale),
}
# mixed = all 4 scaled to target SNR (same as add_all_noise)
from noise_pipeline import add_all_noise

NOISE_LEVELS = (0,5,10,15,20,30,40)

def bandpass_filter(X):
    sos = sg.butter(4, [0.5,45], btype="band", fs=360, output="sos")
    out = np.stack([sg.sosfilt(sos, x[:,0]) for x in X])
    return out.reshape(-1, WINDOW_SAMPLES, 1)

def macro_f1(y_true, y_pred):
    return float(np.mean(f1_score(y_true, y_pred, average=None, zero_division=0)))

def add_noise_typed(ecg, snr_db, fn):
    # scale to target SNR like add_all_noise does
    sig_rms = np.sqrt(np.mean(ecg**2))
    noise_scale = sig_rms / (10**(snr_db/20))
    ecg_n = fn(ecg, noise_scale) if fn else ecg
    residual = ecg_n - ecg
    rms = np.sqrt(np.mean(residual**2))
    if rms>0:
        ecg_n = ecg + residual * (noise_scale / rms)
    return ecg_n

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-class", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=10)
    args = ap.parse_args()

    by_class = load_data_with_record_tracking(str(TR/"mitdb"), max_per_class=args.max_per_class)
    (X_tr, y_tr), _, (X_te, y_te) = record_level_split(by_class, train_ratio=0.7, val_ratio=0.15, seed=42)
    X_tr = X_tr.reshape(-1, WINDOW_SAMPLES, 1).astype(np.float32)
    X_te = X_te.reshape(-1, WINDOW_SAMPLES, 1).astype(np.float32)
    print(f"Train {X_tr.shape} Test {X_te.shape} test_counts {np.bincount(y_te)}")

    # Load or quick-train classifier
    clf_path = TR/"models"/"robust_classifier.keras"
    if clf_path.exists():
        clf = tf.keras.models.load_model(clf_path)
        print(f"Loaded {clf_path}")
    else:
        from models import build_classifier
        clf = build_classifier()
        clf.fit(X_tr, y_tr, epochs=args.epochs, batch_size=64, verbose=0)
    # Denoiser for AE front-end
    den_path = TR/"models"/"robust_denoiser.keras"
    if den_path.exists():
        den = tf.keras.models.load_model(den_path)
        print(f"Loaded {den_path}")
    else:
        den = None
        print("No denoiser, AE front-end will be skipped")

    results = {}
    for noise_name, fn in list(NOISE_FNS.items()) + [("mixed", None)]:
        results[noise_name] = {"raw": {}, "filter": {}, "autoencoder": {}}
        for snr in NOISE_LEVELS:
            if noise_name=="mixed":
                X_noisy = np.stack([add_all_noise(x.flatten(), snr_db=snr) for x in X_te]).reshape(-1, WINDOW_SAMPLES, 1)
            else:
                X_noisy = np.stack([add_noise_typed(x.flatten(), snr_db=snr, fn=fn) for x in X_te]).reshape(-1, WINDOW_SAMPLES, 1)
            pred_raw = np.argmax(clf.predict(X_noisy, verbose=0), axis=1)
            results[noise_name]["raw"][str(snr)] = macro_f1(y_te, pred_raw)
            pred_filt = np.argmax(clf.predict(bandpass_filter(X_noisy), verbose=0), axis=1)
            results[noise_name]["filter"][str(snr)] = macro_f1(y_te, pred_filt)
            if den is not None:
                X_deno = den.predict(X_noisy, verbose=0)
                pred_ae = np.argmax(clf.predict(X_deno, verbose=0), axis=1)
                results[noise_name]["autoencoder"][str(snr)] = macro_f1(y_te, pred_ae)
            else:
                results[noise_name]["autoencoder"][str(snr)] = None
            print(f"{noise_name:15s} SNR={snr:2d} raw={results[noise_name]['raw'][str(snr)]:.3f} filt={results[noise_name]['filter'][str(snr)]:.3f} ae={results[noise_name]['autoencoder'][str(snr)]}")

    with open(OUT/"per_artifact_noise.json","w") as f: json.dump(results, f, indent=2)
    print(f"Saved {OUT/'per_artifact_noise.json'}")

    # Cost table: AE adds 19KB + 0.59s per window (from paper), filter costs ~0
    cost = {"autoencoder": {"size_kb": 19, "latency_ms_per_window": 590, "note": "from paper 0.59s denoiser"},
            "butterworth": {"size_kb": 0, "latency_ms_per_window": 5, "note": "IIR 4th order ~5ms"},
            "raw": {"size_kb": 0, "latency_ms_per_window": 0}}
    # Find best front-end per artifact (avg across SNR)
    for art in results:
        avgs = {k: np.mean([v for v in results[art][k].values() if v is not None]) for k in ["raw","filter","autoencoder"]}
        best = max(avgs, key=avgs.get)
        cost[art] = {"best": best, "avgs": avgs}
        print(f"{art:15s} best={best} avgs {avgs}")
    with open(OUT/"noise_cost_table.json","w") as f: json.dump({"per_artifact": results, "cost": cost}, f, indent=2)

    # Plot 5 panels — IEEE single-column 3.4in @300dpi, compact but readable
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 5.5})
    fig, axes = plt.subplots(2,3, figsize=(3.4, 2.8), dpi=300)
    axes = axes.flatten()
    for idx, (art, vals) in enumerate(results.items()):
        ax = axes[idx]
        for k, label, color in [("raw","Raw","#d95f02"), ("filter","Butterworth","#1b9e77"), ("autoencoder","AE","#7570b3")]:
            ys = [vals[k][str(s)] for s in NOISE_LEVELS]
            ax.plot(NOISE_LEVELS, ys, marker="o", markersize=3.5, linewidth=1.2, label=label, color=color)
        # pretty title: baseline_wander -> Baseline Wander
        pretty = art.replace("_"," ").title()
        ax.set_title(pretty, fontsize=9, pad=6, weight="bold")
        ax.set_xlabel("SNR (dB)", fontsize=8); ax.set_ylabel("Macro F1", fontsize=8)
        ax.set_ylim(0.15,1.02); ax.set_xlim(-1,41)
        ax.set_xticks([0,10,20,30,40]); ax.grid(alpha=0.3, linewidth=0.5)
        ax.legend(fontsize=6.5, frameon=True, loc="lower right", handletextpad=0.4)
    if len(results)<6:
        axes[-1].axis("off")
    fig.suptitle("Per-Artifact Robustness: Raw vs. Butterworth vs. Autoencoder (7 SNR Levels)", fontsize=11, y=1.02)
    plt.tight_layout(pad=1.2)
    fig.savefig(OUT/"per_artifact_noise.png", dpi=300)
    print(f"Saved {OUT/'per_artifact_noise.png'}")
    plt.close()

if __name__=="__main__":
    main()
