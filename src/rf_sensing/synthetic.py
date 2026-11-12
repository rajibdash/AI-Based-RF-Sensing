"""Synthetic complex-IQ examples for a binary occupancy baseline."""

from __future__ import annotations

import numpy as np


def make_dataset(
    count: int,
    samples_per_window: int = 512,
    seed: int = 7,
    snr_db_range: tuple[float, float] = (-12.0, 12.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Return IQ windows shaped (count, 2, samples) and binary labels.

    Label 1 means a randomized QPSK-like emitter is present; label 0 is noise only.
    Complex noise has unit average power. This simple generator is not a channel
    or RF-hardware simulator and must not be used as evidence of field accuracy.
    """
    if count < 1 or samples_per_window < 2:
        raise ValueError("count must be positive and each window needs at least 2 samples")
    if snr_db_range[0] > snr_db_range[1]:
        raise ValueError("snr_db_range lower bound must not exceed upper bound")

    rng = np.random.default_rng(seed)
    iq = np.empty((count, 2, samples_per_window), dtype=np.float32)
    labels = rng.integers(0, 2, size=count, dtype=np.int64)
    time = np.arange(samples_per_window, dtype=np.float64)

    for index, occupied in enumerate(labels):
        noise = (rng.normal(size=samples_per_window) + 1j * rng.normal(size=samples_per_window)) / np.sqrt(2.0)
        signal = np.zeros(samples_per_window, dtype=np.complex128)
        if occupied:
            symbols = rng.integers(0, 4, size=samples_per_window)
            signal = np.exp(1j * (np.pi / 2.0) * symbols)
            signal *= (rng.normal() + 1j * rng.normal()) / np.sqrt(2.0)
            offset = rng.uniform(-0.08, 0.08)
            phase = rng.uniform(-np.pi, np.pi)
            signal *= np.exp(1j * (2.0 * np.pi * offset * time / samples_per_window + phase))
            snr_db = rng.uniform(*snr_db_range)
            signal *= 10.0 ** (snr_db / 20.0)
        received = signal + noise
        iq[index, 0] = received.real
        iq[index, 1] = received.imag

    return iq, labels
