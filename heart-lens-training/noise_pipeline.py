#!/usr/bin/env python3
"""Noise injection pipeline for ECG augmentation.
Generates realistic noise types: motion artifact, baseline wander,
power-line interference, and muscle (EMG) noise.
"""

import numpy as np
from scipy import signal as sg

SAMPLE_RATE = 360


def add_baseline_wander(ecg, amplitude=0.15, freq_range=(0.1, 0.5)):
    """Low-frequency sinusoidal baseline drift (breathing, movement)."""
    n = len(ecg)
    t = np.arange(n) / SAMPLE_RATE
    freq = np.random.uniform(*freq_range)
    phase = np.random.uniform(0, 2 * np.pi)
    baseline = amplitude * np.sin(2 * np.pi * freq * t + phase)
    return ecg + baseline


def add_motion_artifact(ecg, amplitude=0.3):
    """Random step/impulse noise simulating sudden electrode movement."""
    noisy = ecg.copy()
    n = len(ecg)
    num_events = np.random.randint(1, 5)
    for _ in range(num_events):
        pos = np.random.randint(0, n - 50)
        duration = np.random.randint(10, 50)
        direction = np.random.choice([-1, 1])
        magnitude = amplitude * np.random.uniform(0.5, 1.0) * direction
        noisy[pos:pos + duration] += magnitude
        # Exponential decay
        decay = np.exp(-np.arange(duration) / (duration / 3))
        noisy[pos:pos + duration] = ecg[pos:pos + duration] + magnitude * decay
    return noisy


def add_pli(ecg, amplitude=0.1, freq=50):
    """Power-line interference at 50 or 60 Hz."""
    n = len(ecg)
    t = np.arange(n) / SAMPLE_RATE
    phase = np.random.uniform(0, 2 * np.pi)
    pli = amplitude * np.sin(2 * np.pi * freq * t + phase)
    return ecg + pli


def add_emg_noise(ecg, amplitude=0.2):
    """Muscle noise — band-passed white noise in 20-100 Hz range."""
    n = len(ecg)
    white = np.random.randn(n)
    sos = sg.butter(4, [20, 100], btype='band', fs=SAMPLE_RATE, output='sos')
    emg = sg.sosfilt(sos, white)
    emg = emg / np.std(emg) * amplitude
    return ecg + emg


def add_all_noise(ecg, snr_db=20):
    """Add all noise types at specified SNR.
    SNR = 20 * log10(signal_rms / noise_rms)
    """
    signal_rms = np.sqrt(np.mean(ecg ** 2))
    noise_scale = signal_rms / (10 ** (snr_db / 20))

    ecg_n = ecg.copy()
    ecg_n = add_baseline_wander(ecg_n, amplitude=0.3 * noise_scale)
    ecg_n = add_motion_artifact(ecg_n, amplitude=0.5 * noise_scale)
    ecg_n = add_pli(ecg_n, amplitude=0.15 * noise_scale)
    ecg_n = add_emg_noise(ecg_n, amplitude=0.3 * noise_scale)

    # Adjust to target SNR
    residual = ecg_n - ecg
    residual_rms = np.sqrt(np.mean(residual ** 2))
    if residual_rms > 0:
        scale = noise_scale / residual_rms
        ecg_n = ecg + residual * scale

    return ecg_n


def augment_dataset(X_clean, noise_levels=(0, 5, 10, 15, 20, 30, 40)):
    """Generate augmented dataset with multiple noise levels."""
    X_noisy = []
    for ecg in X_clean:
        ecg = ecg.flatten()
        for snr in noise_levels:
            noisy = add_all_noise(ecg, snr_db=snr)
            X_noisy.append(noisy.reshape(-1, 1))

    X_clean_aug = np.tile(X_clean, (len(noise_levels), 1, 1))
    return X_clean_aug, np.array(X_noisy)
