"""Physical waveform prep: polarity, baseline, 3σ ROI, PSD, decay (Description).

Three contracts must stay separate:
- ROI / long gate: [i_peak .. i(μ + N_SIGMA·σ)] on x0
- decay 40%: time to (1 - DECAY_FRAC) * peak_above on x0 / x_norm
- PSD short gate: (offset, short_len) inside ROI; default (long-short)/long
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Polarity = Literal["negative", "positive"]

BASELINE_BINS = 50
N_SIGMA = 3.0
DECAY_FRAC = 0.40
PSD_OFFSET = 3
PSD_SHORT = 30
EPS = 1e-9
VALLEY_RATIO_MAX = 0.7

# Legacy aliases used by older callers
DECAY_FRACTION = DECAY_FRAC
PSD_OFFSET_BINS = PSD_OFFSET
PSD_SHORT_LEN = PSD_SHORT


def apply_polarity(x: np.ndarray, polarity: Polarity) -> np.ndarray:
    """Return positively oriented waveforms (pulse goes up)."""
    if polarity == "positive":
        return np.asarray(x, dtype=np.float64)
    if polarity == "negative":
        # Reflect about per-row pedestal estimate from first bins (raw scale).
        ped = np.asarray(x, dtype=np.float64)[:, :BASELINE_BINS].mean(axis=1, keepdims=True)
        return ped - np.asarray(x, dtype=np.float64)
    raise ValueError(f"Unknown POLARITY={polarity!r}; expected 'negative' or 'positive'")


def baseline_noise(x_pos: np.ndarray, baseline_bins: int = BASELINE_BINS) -> tuple[np.ndarray, np.ndarray]:
    """Pedestal mean and noise σ on first baseline_bins of positively oriented waves."""
    if x_pos.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape {x_pos.shape}")
    if baseline_bins < 2 or baseline_bins > x_pos.shape[1]:
        raise ValueError(f"Invalid baseline_bins={baseline_bins}")
    window = x_pos[:, :baseline_bins]
    baseline = window.mean(axis=1)
    noise_std = window.std(axis=1)
    return baseline, noise_std


def to_x0(x_pos: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """Subtract pedestal → working signal x0."""
    return x_pos - baseline[:, None]


@dataclass
class PrepArrays:
    """Batch prep outputs (one row per waveform)."""

    x_pos: np.ndarray
    x0: np.ndarray
    baseline: np.ndarray
    noise_std: np.ndarray
    peak_above: np.ndarray
    i_peak: np.ndarray
    i_end_roi: np.ndarray
    roi_length: np.ndarray
    no_3sigma_hit: np.ndarray
    charge_roi: np.ndarray
    short: np.ndarray
    psd: np.ndarray
    decay_time: np.ndarray
    snr: np.ndarray
    tail_ratio: np.ndarray
    x_norm_peak: np.ndarray  # peak of x_norm (==1 if peak_above>0); kept for QC
    is_saturated: np.ndarray


def _roi_end_3sigma(wave: np.ndarray, i_peak: int, noise_std: float, n_sigma: float) -> tuple[int, bool]:
    """First index after peak where wave <= n_sigma * noise_std; else last index."""
    n = len(wave)
    thr = n_sigma * float(noise_std)
    for j in range(i_peak, n):
        if wave[j] <= thr:
            return j, False
    return n - 1, True


def extract_prep_features(
    x_raw: np.ndarray,
    *,
    polarity: Polarity = "negative",
    baseline_bins: int = BASELINE_BINS,
    n_sigma: float = N_SIGMA,
    decay_frac: float = DECAY_FRAC,
    psd_offset: int = PSD_OFFSET,
    psd_short: int = PSD_SHORT,
    eps: float = EPS,
) -> PrepArrays:
    """Full prep batch: polarity → baseline → x0 → 3σ ROI → PSD/decay."""
    if x_raw.ndim != 2:
        raise ValueError(f"Expected (n, t) matrix, got {x_raw.shape}")
    if not np.isfinite(x_raw).all():
        raise ValueError("Non-finite values in raw waveforms")

    x_pos = apply_polarity(x_raw, polarity)
    baseline, noise_std = baseline_noise(x_pos, baseline_bins=baseline_bins)
    x0 = to_x0(x_pos, baseline)

    n, t = x0.shape
    i_peak = np.argmax(x0, axis=1).astype(np.int64)
    peak_above = x0[np.arange(n), i_peak]

    i_end = np.empty(n, dtype=np.int64)
    no_hit = np.empty(n, dtype=bool)
    charge = np.empty(n, dtype=np.float64)
    short = np.empty(n, dtype=np.float64)
    decay = np.empty(n, dtype=np.float64)
    tail = np.empty(n, dtype=np.float64)

    thr_frac = 1.0 - decay_frac
    for i in range(n):
        p = int(i_peak[i])
        end, hit = _roi_end_3sigma(x0[i], p, float(noise_std[i]), n_sigma)
        i_end[i] = end
        no_hit[i] = hit
        roi = x0[i, p : end + 1]
        charge[i] = float(roi.sum())
        s0 = min(len(roi) - 1, max(0, psd_offset)) if len(roi) else 0
        s1 = min(len(roi), s0 + psd_short)
        short[i] = float(roi[s0:s1].sum()) if len(roi) else 0.0

        # decay on x0: level = thr_frac * peak_above (60% of amplitude when frac=0.4)
        thr = thr_frac * float(peak_above[i])
        below = np.where(x0[i, p:] <= thr)[0]
        decay[i] = float(below[0]) if len(below) else float(t - p)

        # tail_ratio: charge in second half of ROI / charge_roi
        if len(roi) >= 2:
            mid = len(roi) // 2
            tail[i] = float(roi[mid:].sum()) / (charge[i] + eps)
        else:
            tail[i] = 0.0

    # PSD Description default: (long - short) / long
    psd = (charge - short) / (charge + eps)
    snr = peak_above / (noise_std + eps)
    roi_length = (i_end - i_peak + 1).astype(np.float64)

    # saturation heuristic: raw ADC max near dataset ceiling
    raw_peak = np.max(x_raw, axis=1)
    global_hi = float(np.percentile(raw_peak, 99))
    is_sat = raw_peak >= (global_hi - 5.0)

    x_norm_peak = peak_above / (peak_above + eps)  # ~1; diagnostic

    # Early finite check (D11)
    for name, arr in (
        ("x0", x0),
        ("baseline", baseline),
        ("noise_std", noise_std),
        ("peak_above", peak_above),
        ("charge_roi", charge),
        ("psd", psd),
        ("decay_time", decay),
    ):
        if not np.isfinite(arr).all():
            raise ValueError(f"Non-finite values in {name} after prep")

    return PrepArrays(
        x_pos=x_pos,
        x0=x0,
        baseline=baseline,
        noise_std=noise_std,
        peak_above=peak_above,
        i_peak=i_peak,
        i_end_roi=i_end,
        roi_length=roi_length,
        no_3sigma_hit=no_hit,
        charge_roi=charge,
        short=short,
        psd=psd,
        decay_time=decay,
        snr=snr,
        tail_ratio=tail,
        x_norm_peak=x_norm_peak,
        is_saturated=is_sat,
    )


def compute_psd_batch(
    x0: np.ndarray,
    i_peak: np.ndarray,
    i_end: np.ndarray,
    noise_std: np.ndarray,  # noqa: ARG001 — kept for API symmetry
    *,
    psd_offset: int,
    psd_short: int,
    eps: float = EPS,
) -> np.ndarray:
    """Recompute PSD for a parameter grid point."""
    n = len(x0)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        p = int(i_peak[i])
        end = int(i_end[i])
        roi = x0[i, p : end + 1]
        charge = float(roi.sum()) if len(roi) else 0.0
        s0 = min(len(roi) - 1, max(0, psd_offset)) if len(roi) else 0
        s1 = min(len(roi), s0 + psd_short)
        short = float(roi[s0:s1].sum()) if len(roi) else 0.0
        out[i] = (charge - short) / (charge + eps)
    return out


def valley_ratio(
    values: np.ndarray,
    *,
    n_bins: int = 80,
    eps: float = EPS,
) -> tuple[float, dict]:
    """Bimodality gate metric on 1D feature histogram.

    valley_ratio = density(valley) / mean(density(mode1), density(mode2))
    Returns (ratio, info). If <2 modes → ratio = +inf (gate FAIL).
    """
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size < 100:
        return float("inf"), {"n_modes": 0, "reason": "too_few_samples"}

    hist, edges = np.histogram(v, bins=n_bins, density=True)
    # smooth lightly
    kernel = np.array([0.25, 0.5, 0.25])
    smooth = np.convolve(hist, kernel, mode="same")
    # local maxima
    peaks = []
    for i in range(1, len(smooth) - 1):
        if smooth[i] >= smooth[i - 1] and smooth[i] >= smooth[i + 1] and smooth[i] > 0:
            peaks.append(i)
    if len(peaks) < 2:
        return float("inf"), {"n_modes": len(peaks), "reason": "unimodal"}

    # two dominant peaks
    peak_heights = sorted(((smooth[i], i) for i in peaks), reverse=True)
    i1, i2 = sorted([peak_heights[0][1], peak_heights[1][1]])
    valley_idx = i1 + int(np.argmin(smooth[i1 : i2 + 1]))
    d1, d2 = float(smooth[i1]), float(smooth[i2])
    dv = float(smooth[valley_idx])
    ratio = dv / (0.5 * (d1 + d2) + eps)
    centers = 0.5 * (edges[:-1] + edges[1:])
    info = {
        "n_modes": len(peaks),
        "mode1": float(centers[i1]),
        "mode2": float(centers[i2]),
        "valley": float(centers[valley_idx]),
        "density_mode1": d1,
        "density_mode2": d2,
        "density_valley": dv,
    }
    return float(ratio), info


def calibrate_psd_windows(
    x0: np.ndarray,
    i_peak: np.ndarray,
    i_end: np.ndarray,
    noise_std: np.ndarray,
    *,
    offsets: list[int],
    shorts: list[int],
    max_grid_points: int = 64,
    subsample: int = 5000,
    random_state: int = 42,
    eps: float = EPS,
) -> tuple[int, int, float, dict]:
    """Grid-search (offset, short) minimizing valley_ratio on subsample."""
    pairs = [(o, s) for o in offsets for s in shorts]
    if len(pairs) > max_grid_points:
        raise ValueError(f"Grid has {len(pairs)} points > MAX_GRID_POINTS={max_grid_points}")

    n = len(x0)
    rng = np.random.default_rng(random_state)
    m = min(subsample, n)
    idx = rng.choice(n, size=m, replace=False)

    best = (PSD_OFFSET, PSD_SHORT, float("inf"), {})
    for off, sh in pairs:
        psd = compute_psd_batch(
            x0[idx], i_peak[idx], i_end[idx], noise_std[idx], psd_offset=off, psd_short=sh, eps=eps
        )
        ratio, info = valley_ratio(psd, eps=eps)
        if ratio < best[2]:
            best = (off, sh, ratio, info)
    return best[0], best[1], best[2], best[3]


def build_qc_flags(prep: PrepArrays, *, psd_lo: float | None = None, psd_hi: float | None = None) -> dict[str, np.ndarray]:
    """Physical QC flags; candidate_class2 = OR of flags (no top-q%)."""
    n = len(prep.peak_above)
    # multi-peak: count local maxima on x0 ROI above 0.3*peak
    multi = np.zeros(n, dtype=bool)
    for i in range(n):
        p, e = int(prep.i_peak[i]), int(prep.i_end_roi[i])
        seg = prep.x0[i, p : e + 1]
        if len(seg) < 5:
            continue
        thr = 0.3 * float(prep.peak_above[i])
        peaks = 0
        for j in range(1, len(seg) - 1):
            if seg[j] >= seg[j - 1] and seg[j] >= seg[j + 1] and seg[j] > thr:
                peaks += 1
        multi[i] = peaks >= 2

    decay_hang = prep.decay_time >= (prep.x0.shape[1] - prep.i_peak - 1)
    roi_short = prep.roi_length <= 5
    roi_long = prep.roi_length >= 0.9 * prep.x0.shape[1]
    # baseline drift proxy: std of first bins already in noise; use |baseline - median|
    med_b = float(np.median(prep.baseline))
    mad = float(np.median(np.abs(prep.baseline - med_b))) + EPS
    baseline_drift = np.abs(prep.baseline - med_b) > (5.0 * mad)

    if psd_lo is None:
        psd_lo = float(np.quantile(prep.psd, 0.01))
    if psd_hi is None:
        psd_hi = float(np.quantile(prep.psd, 0.99))
    psd_out = (prep.psd < psd_lo) | (prep.psd > psd_hi)

    nonfinite = ~(
        np.isfinite(prep.psd)
        & np.isfinite(prep.decay_time)
        & np.isfinite(prep.charge_roi)
        & np.isfinite(prep.snr)
    )

    candidate = multi | decay_hang | roi_short | roi_long | baseline_drift | psd_out | nonfinite | prep.no_3sigma_hit

    return {
        "qc_multi_peak": multi,
        "qc_decay_hang": decay_hang,
        "qc_roi_short": roi_short,
        "qc_roi_long": roi_long,
        "qc_baseline_drift": baseline_drift,
        "qc_psd_outlier": psd_out,
        "qc_nonfinite": nonfinite,
        "qc_no_3sigma_hit": prep.no_3sigma_hit,
        "qc_saturated": prep.is_saturated,
        "candidate_class2": candidate,
    }


def extract_description_features(
    x: np.ndarray,
    baseline_bins: int = BASELINE_BINS,
    decay_fraction: float = DECAY_FRAC,
    psd_offset: int = PSD_OFFSET,
    psd_short_len: int = PSD_SHORT,
    polarity: Polarity = "negative",
) -> np.ndarray:
    """Backward-compatible matrix: peak, charge, psd, decay_time, snr, tail_ratio, amp_area."""
    prep = extract_prep_features(
        x,
        polarity=polarity,
        baseline_bins=baseline_bins,
        decay_frac=decay_fraction,
        psd_offset=psd_offset,
        psd_short=psd_short_len,
    )
    peak = prep.peak_above
    charge = prep.charge_roi
    out = np.column_stack(
        [
            peak,
            charge,
            prep.psd,
            prep.decay_time,
            prep.snr,
            prep.tail_ratio,
            peak * charge,
        ]
    )
    return out
