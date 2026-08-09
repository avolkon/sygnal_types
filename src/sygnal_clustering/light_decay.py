"""LIGHT prep: front-align, honest decay fit, PSD from birth of the flash.

Canon (crystal):
  Type = decay character after front alignment.
  Reject (class 2) = only when decay is unmeasurable.
  Energy rules reject width, not the wave's name.

Does not replace FE-champion prep in signal_extraction.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sygnal_clustering.signal_extraction import (
    BASELINE_BINS,
    DECAY_FRAC,
    EPS,
    N_SIGMA,
    PSD_OFFSET,
    PSD_SHORT,
    Polarity,
    extract_prep_features,
)

FRONT_ALPHA = 0.2
MIN_TAIL_POINTS = 8
TAU_R2_MIN = 0.5


@dataclass
class LightArrays:
    """Per-event LIGHT features (aligned to PrepArrays rows)."""

    t0: np.ndarray
    alpha: float
    i_peak: np.ndarray
    peak_above: np.ndarray
    i_end_roi: np.ndarray
    qc_no_front: np.ndarray
    psd_argmax: np.ndarray
    psd_front: np.ndarray
    tau_eff: np.ndarray
    fit_r2: np.ndarray
    qc_bad_decay_fit: np.ndarray
    rho: np.ndarray
    charge_roi: np.ndarray
    snr: np.ndarray
    short_front: np.ndarray
    short_argmax: np.ndarray


def find_front_t0(
    wave: np.ndarray,
    i_peak: int,
    peak_above: float,
    *,
    alpha: float = FRONT_ALPHA,
    noise_std: float = 0.0,
    k_sigma: float = 3.0,
) -> tuple[int, bool]:
    """First sample on the rising edge where wave >= alpha * peak.

    Returns (t0, ok). ok=False → qc_no_front (no front / peak drowned in noise).
    """
    if not np.isfinite(peak_above) or peak_above <= 0:
        return int(i_peak), False
    if peak_above < k_sigma * float(noise_std):
        return int(i_peak), False
    thr = float(alpha) * float(peak_above)
    # search only up to peak (inclusive): birth before or at crest
    seg = wave[: int(i_peak) + 1]
    hits = np.where(seg >= thr)[0]
    if len(hits) == 0:
        return int(i_peak), False
    return int(hits[0]), True


def _psd_from_anchor(
    wave: np.ndarray,
    anchor: int,
    i_end: int,
    *,
    psd_offset: int,
    psd_short: int,
    eps: float = EPS,
) -> tuple[float, float, float]:
    """(psd, charge_long, short) with windows counted from anchor, long to i_end."""
    a = max(0, int(anchor))
    end = max(a, int(i_end))
    roi = wave[a : end + 1]
    if len(roi) == 0:
        return 0.0, 0.0, 0.0
    charge = float(roi.sum())
    s0 = min(len(roi) - 1, max(0, psd_offset))
    s1 = min(len(roi), s0 + psd_short)
    short = float(roi[s0:s1].sum())
    psd = (charge - short) / (charge + eps)
    return psd, charge, short


def fit_tau_eff(
    wave: np.ndarray,
    i_peak: int,
    peak_above: float,
    i_end: int,
    *,
    decay_frac: float = DECAY_FRAC,
    min_points: int = MIN_TAIL_POINTS,
    eps: float = EPS,
) -> tuple[float, float, bool]:
    """Single-exp τ_eff on the tail after amplitude has fallen by decay_frac.

    Fit log(I) = c - t/τ on samples from first crossing of (1-frac)*peak
    to ROI end, requiring I > 0. Returns (tau_eff, r2, ok).
    """
    if not np.isfinite(peak_above) or peak_above <= 0:
        return float("nan"), 0.0, False
    thr = (1.0 - float(decay_frac)) * float(peak_above)
    p = int(i_peak)
    end = int(i_end)
    if end <= p:
        return float("nan"), 0.0, False
    below = np.where(wave[p : end + 1] <= thr)[0]
    if len(below) == 0:
        return float("nan"), 0.0, False
    t_start = p + int(below[0])
    t = np.arange(t_start, end + 1, dtype=np.float64)
    y = wave[t_start : end + 1].astype(np.float64)
    mask = np.isfinite(y) & (y > eps)
    if int(mask.sum()) < min_points:
        return float("nan"), 0.0, False
    t = t[mask]
    y = y[mask]
    log_y = np.log(y)
    # polyfit degree 1: log_y = b0 + b1 * t  → τ = -1/b1
    b1, b0 = np.polyfit(t - t[0], log_y, 1)
    if not np.isfinite(b1) or b1 >= -eps:
        return float("nan"), 0.0, False
    tau = float(-1.0 / b1)
    pred = b0 + b1 * (t - t[0])
    ss_res = float(np.sum((log_y - pred) ** 2))
    ss_tot = float(np.sum((log_y - log_y.mean()) ** 2))
    r2 = 1.0 - ss_res / (ss_tot + eps) if ss_tot > eps else 0.0
    ok = bool(np.isfinite(tau) and tau > 0 and r2 >= TAU_R2_MIN)
    return tau, float(r2), ok


def fit_rho_two_exp(
    wave: np.ndarray,
    i_peak: int,
    peak_above: float,
    i_end: int,
    *,
    decay_frac: float = DECAY_FRAC,
    min_points: int = MIN_TAIL_POINTS,
    eps: float = EPS,
) -> tuple[float, bool]:
    """LIGHT-B: ρ = A_s τ_s / (A_f τ_f + A_s τ_s). Fallback nan if unstable.

    Practical estimator: split tail at median time; local τ on each half,
    amplitudes from intercept; enforce τ_f < τ_s.
    """
    if not np.isfinite(peak_above) or peak_above <= 0:
        return float("nan"), False
    thr = (1.0 - float(decay_frac)) * float(peak_above)
    p = int(i_peak)
    end = int(i_end)
    below = np.where(wave[p : end + 1] <= thr)[0]
    if len(below) == 0:
        return float("nan"), False
    t_start = p + int(below[0])
    t = np.arange(t_start, end + 1, dtype=np.float64)
    y = wave[t_start : end + 1].astype(np.float64)
    mask = np.isfinite(y) & (y > eps)
    if int(mask.sum()) < 2 * min_points:
        return float("nan"), False
    t = t[mask] - t[mask][0]
    y = y[mask]
    mid = len(t) // 2
    if mid < min_points or (len(t) - mid) < min_points:
        return float("nan"), False

    def _half(th: np.ndarray, yh: np.ndarray) -> tuple[float, float]:
        b1, b0 = np.polyfit(th, np.log(yh), 1)
        if b1 >= -eps:
            return float("nan"), float("nan")
        tau = -1.0 / b1
        amp = float(np.exp(b0))
        return amp, tau

    af, tf = _half(t[:mid], y[:mid])
    as_, ts = _half(t[mid:], y[mid:])
    if not np.isfinite([af, tf, as_, ts]).all():
        return float("nan"), False
    if tf <= 0 or ts <= 0 or af <= 0 or as_ <= 0:
        return float("nan"), False
    # enforce fast/slow ordering by swapping if needed
    if tf > ts:
        af, as_ = as_, af
        tf, ts = ts, tf
    denom = af * tf + as_ * ts + eps
    rho = (as_ * ts) / denom
    ok = bool(0.0 <= rho <= 1.0)
    return float(rho), ok


def extract_light_features(
    x_raw: np.ndarray,
    *,
    polarity: Polarity = "negative",
    baseline_bins: int = BASELINE_BINS,
    n_sigma: float = N_SIGMA,
    decay_frac: float = DECAY_FRAC,
    psd_offset: int = PSD_OFFSET,
    psd_short: int = PSD_SHORT,
    alpha: float = FRONT_ALPHA,
    eps: float = EPS,
) -> LightArrays:
    """LIGHT batch: front t0, psd_front vs psd_argmax, τ_eff / ρ + fit QC."""
    prep = extract_prep_features(
        x_raw,
        polarity=polarity,
        baseline_bins=baseline_bins,
        n_sigma=n_sigma,
        decay_frac=decay_frac,
        psd_offset=psd_offset,
        psd_short=psd_short,
        eps=eps,
    )
    n, _ = prep.x0.shape
    t0 = np.empty(n, dtype=np.int64)
    qc_no_front = np.zeros(n, dtype=bool)
    psd_front = np.empty(n, dtype=np.float64)
    short_front = np.empty(n, dtype=np.float64)
    short_argmax = np.empty(n, dtype=np.float64)
    psd_argmax = np.empty(n, dtype=np.float64)
    tau_eff = np.empty(n, dtype=np.float64)
    fit_r2 = np.empty(n, dtype=np.float64)
    qc_bad = np.zeros(n, dtype=bool)
    rho = np.full(n, np.nan, dtype=np.float64)

    for i in range(n):
        wave = prep.x0[i]
        p = int(prep.i_peak[i])
        end = int(prep.i_end_roi[i])
        pk = float(prep.peak_above[i])
        t0_i, ok_f = find_front_t0(
            wave, p, pk, alpha=alpha, noise_std=float(prep.noise_std[i])
        )
        t0[i] = t0_i
        qc_no_front[i] = not ok_f

        psd_a, _, sh_a = _psd_from_anchor(
            wave, p, end, psd_offset=psd_offset, psd_short=psd_short, eps=eps
        )
        psd_f, _, sh_f = _psd_from_anchor(
            wave, t0_i, end, psd_offset=psd_offset, psd_short=psd_short, eps=eps
        )
        psd_argmax[i] = psd_a
        short_argmax[i] = sh_a
        psd_front[i] = psd_f
        short_front[i] = sh_f

        tau, r2, ok_tau = fit_tau_eff(
            wave, p, pk, end, decay_frac=decay_frac, eps=eps
        )
        tau_eff[i] = tau
        fit_r2[i] = r2
        rho_i, ok_rho = fit_rho_two_exp(
            wave, p, pk, end, decay_frac=decay_frac, eps=eps
        )
        rho[i] = rho_i
        qc_bad[i] = (not ok_tau) or qc_no_front[i]

    # sanity: psd_argmax must match champion prep PSD (same windows from peak)
    if not np.allclose(psd_argmax, prep.psd, rtol=0, atol=1e-9):
        max_err = float(np.max(np.abs(psd_argmax - prep.psd)))
        raise ValueError(f"psd_argmax diverges from prep.psd (max|Δ|={max_err})")

    return LightArrays(
        t0=t0,
        alpha=float(alpha),
        i_peak=prep.i_peak,
        peak_above=prep.peak_above,
        i_end_roi=prep.i_end_roi,
        qc_no_front=qc_no_front,
        psd_argmax=psd_argmax,
        psd_front=psd_front,
        tau_eff=tau_eff,
        fit_r2=fit_r2,
        qc_bad_decay_fit=qc_bad,
        rho=rho,
        charge_roi=prep.charge_roi,
        snr=prep.snr,
        short_front=short_front,
        short_argmax=short_argmax,
    )


def labels_from_axis(
    values: np.ndarray,
    *,
    delta: float = 0.003,
    out_lo: float = 0.015,
    out_hi: float = 0.985,
    eps: float = EPS,
    force_class2: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """M0/M1-style: valley+δ split on 1D axis; thin quantile tails → 2."""
    from sygnal_clustering.signal_extraction import valley_ratio

    v = np.asarray(values, dtype=np.float64)
    vr, info = valley_ratio(v, eps=eps)
    if not np.isfinite(vr) or "valley" not in info:
        raise ValueError(f"valley gate failed: {info}")
    thr = float(info["valley"]) + float(delta)
    lab = np.where(v < thr, 0, 1).astype(np.int64)
    # polarity: class 0 = lower mean on the axis (fast / smaller tail)
    if v[lab == 0].mean() > v[lab == 1].mean():
        lab = 1 - lab
    finite = np.isfinite(v)
    qlo, qhi = np.quantile(v[finite], [out_lo, out_hi])
    lab = lab.copy()
    lab[(v < qlo) | (v > qhi)] = 2
    if force_class2 is not None:
        lab[np.asarray(force_class2, dtype=bool)] = 2
    meta = {
        "valley_ratio": float(vr),
        "valley": float(info["valley"]),
        "thr": thr,
        "qlo": float(qlo),
        "qhi": float(qhi),
        "mode1": info.get("mode1"),
        "mode2": info.get("mode2"),
    }
    return lab, meta


def gmm_log_threshold(
    values: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    random_state: int = 42,
    n_init: int = 20,
) -> tuple[float, dict]:
    """Equal-responsibility threshold of GMM-2 on log(values).

    Raw histogram valley on τ is poisoned by a heavy high-τ bump; the mixture
    valley on log τ recovers the fast/slow modes (~5 and ~15 bins).
    """
    from sklearn.mixture import GaussianMixture

    v = np.asarray(values, dtype=np.float64)
    if mask is None:
        mask = np.isfinite(v) & (v > 0)
    else:
        mask = np.asarray(mask, dtype=bool) & np.isfinite(v) & (v > 0)
    x = np.log(v[mask]).reshape(-1, 1)
    if len(x) < 100:
        raise ValueError("too few samples for GMM-2 on log τ")
    gmm = GaussianMixture(n_components=2, random_state=random_state, n_init=n_init)
    gmm.fit(x)
    means = np.sort(gmm.means_.ravel())
    grid = np.linspace(float(x.min()), float(x.max()), 5000).reshape(-1, 1)
    diff = gmm.predict_proba(grid)[:, 0] - gmm.predict_proba(grid)[:, 1]
    flips = np.where(np.diff(np.sign(diff)) != 0)[0]
    if len(flips):
        thr_log = float(grid[flips[len(flips) // 2], 0])
    else:
        thr_log = float(0.5 * (means[0] + means[1]))
    thr = float(np.exp(thr_log))
    meta = {
        "split": "gmm2_log",
        "thr": thr,
        "means_log": [float(means[0]), float(means[1])],
        "means_lin": [float(np.exp(means[0])), float(np.exp(means[1]))],
        "weights": [float(w) for w in gmm.weights_],
        "n_fit": int(mask.sum()),
    }
    return thr, meta


def labels_from_tau_eff(
    tau_eff: np.ndarray,
    *,
    honesty_mask: np.ndarray,
    out_lo: float = 0.015,
    out_hi: float = 0.985,
    random_state: int = 42,
) -> tuple[np.ndarray, dict]:
    """LIGHT-2 / M1: τ_eff split + honesty→2 + thin form tails→2.

    Polarity (LB-compatible on this Run): larger τ → class 0.
    Empirically τ anti-correlates with champion PSD, where class 0 = lower PSD.
    """
    tau = np.asarray(tau_eff, dtype=np.float64)
    honest = np.asarray(honesty_mask, dtype=bool) & np.isfinite(tau) & (tau > 0)
    thr, gmeta = gmm_log_threshold(tau, mask=honest, random_state=random_state)

    lab = np.full(len(tau), 2, dtype=np.int64)
    # large τ = slow component → class 0 (matches M0 IDs on this dataset)
    lab[honest] = np.where(tau[honest] > thr, 0, 1).astype(np.int64)

    qlo, qhi = np.quantile(tau[honest], [out_lo, out_hi])
    lab[honest & ((tau < qlo) | (tau > qhi))] = 2
    # hour of honesty: bad fit / no front / non-finite τ
    lab[~honest] = 2

    # safety: if polarity flipped vs intended (class0 should have larger mean τ)
    m0, m1 = lab == 0, lab == 1
    if m0.any() and m1.any() and tau[m0].mean() < tau[m1].mean():
        swap = lab.copy()
        swap[m0] = 1
        swap[m1] = 0
        lab = swap

    meta = {
        **gmeta,
        "qlo": float(qlo),
        "qhi": float(qhi),
        "n_honesty_class2": int((~honest).sum()),
        "n_tail_class2": int((lab == 2).sum() - (~honest).sum()),
        "polarity": "large_tau_is_0",
        "tau_mean_0": float(tau[lab == 0].mean()) if (lab == 0).any() else None,
        "tau_mean_1": float(tau[lab == 1].mean()) if (lab == 1).any() else None,
    }
    return lab, meta
