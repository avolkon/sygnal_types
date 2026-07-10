"""Signal feature extraction per Description.txt (PSD short/long, 40% decay)."""

from __future__ import annotations

import numpy as np

BASELINE_BINS = 50
DECAY_FRACTION = 0.40
PSD_OFFSET_BINS = 3
PSD_SHORT_LEN = 30


def _extract_single(
    waveform: np.ndarray,
    baseline_bins: int = BASELINE_BINS,
    decay_fraction: float = DECAY_FRACTION,
    psd_offset: int = PSD_OFFSET_BINS,
    psd_short_len: int = PSD_SHORT_LEN,
) -> tuple[float, float, float, float, float, int, int]:
    """Return peak, charge, psd, decay_time, snr, i_start, i_end for one waveform."""
    n = len(waveform)
    imax = int(np.argmax(waveform))
    peak = float(waveform[imax])
    i_start = imax

    thr_decay = peak * (1.0 - decay_fraction)
    i_end = n - 1
    for j, val in enumerate(waveform[imax:]):
        if val <= thr_decay:
            i_end = imax + j
            break

    signal = waveform[i_start : i_end + 1]
    charge = float(signal.sum())
    s0 = min(len(signal) - 1, psd_offset)
    s1 = min(len(signal), s0 + psd_short_len)
    short = float(signal[s0:s1].sum())
    psd = short / (charge + 1e-9)
    decay_time = float(i_end - imax)

    baseline = float(waveform[:baseline_bins].mean())
    snr = peak / (baseline + 1e-9)
    return peak, charge, psd, decay_time, snr, i_start, i_end


def extract_description_features(
    x: np.ndarray,
    baseline_bins: int = BASELINE_BINS,
    decay_fraction: float = DECAY_FRACTION,
    psd_offset: int = PSD_OFFSET_BINS,
    psd_short_len: int = PSD_SHORT_LEN,
) -> np.ndarray:
    """
    Per-row features after Description-style pulse extraction.

    Columns: peak, charge, psd, decay_time, snr, tail_ratio, amp_area.
    """
    n = len(x)
    out = np.zeros((n, 7), dtype=np.float64)
    for i in range(n):
        peak, charge, psd, decay_time, snr, i_start, i_end = _extract_single(
            x[i],
            baseline_bins=baseline_bins,
            decay_fraction=decay_fraction,
            psd_offset=psd_offset,
            psd_short_len=psd_short_len,
        )
        tail = float(x[i, i_start : i_end + 1].sum()) / (charge + 1e-9)
        out[i] = [peak, charge, psd, decay_time, snr, tail, peak * charge]
    return out
