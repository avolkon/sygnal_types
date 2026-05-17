"""Signal window extraction per Description.txt (3σ noise, 40% decay)."""

from __future__ import annotations

import numpy as np

BASELINE_BINS = 20
NOISE_SIGMA = 3.0
DECAY_FRACTION = 0.40
PSD_OFFSET_BINS = 3
PSD_SHORT_LEN = 30


def _extract_single(
    waveform: np.ndarray,
    baseline_bins: int = BASELINE_BINS,
    noise_sigma: float = NOISE_SIGMA,
    decay_fraction: float = DECAY_FRACTION,
    psd_offset: int = PSD_OFFSET_BINS,
    psd_short_len: int = PSD_SHORT_LEN,
) -> tuple[float, float, float, float, float, int, int]:
    """Return peak, charge, psd, decay_time, snr, i_start, i_end for one waveform."""
    n = len(waveform)
    base = waveform[:baseline_bins]
    mu = float(base.mean())
    sigma = float(base.std())
    if sigma < 1e-9:
        sigma = 1e-9
    stop_level = mu + noise_sigma * sigma

    imax = int(np.argmax(waveform))
    peak = float(waveform[imax])

    # end: after maximum, first point at or below noise band
    i_end = n - 1
    for i in range(imax, n):
        if waveform[i] <= stop_level:
            i_end = i
            break

    i_start = imax  # signal anchor at peak (Description allows post-max start for PSD)
    signal = waveform[i_start : i_end + 1]
    if signal.size == 0:
        signal = waveform[imax : imax + 1]
        i_end = imax

    charge = float(signal.sum())
    s0 = min(len(signal) - 1, psd_offset)
    s1 = min(len(signal), s0 + psd_short_len)
    short = float(signal[s0:s1].sum())
    psd = short / (charge + 1e-9)

    thr_decay = peak * (1.0 - decay_fraction)
    decay_time = float(len(signal))
    for j, val in enumerate(waveform[imax : i_end + 1]):
        if val <= thr_decay:
            decay_time = float(j)
            break

    snr = peak / (mu + 1e-9)
    return peak, charge, psd, decay_time, snr, i_start, i_end


def extract_description_features(
    x: np.ndarray,
    baseline_bins: int = BASELINE_BINS,
    noise_sigma: float = NOISE_SIGMA,
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
            noise_sigma=noise_sigma,
            decay_fraction=decay_fraction,
            psd_offset=psd_offset,
            psd_short_len=psd_short_len,
        )
        tail = float(x[i, i_start : i_end + 1].sum()) / (charge + 1e-9)
        out[i] = [peak, charge, psd, decay_time, snr, tail, peak * charge]
    return out
