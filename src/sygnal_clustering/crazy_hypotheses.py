"""Crazy residual hypotheses on top of asymmetric lo-only PSD champ (0.89109).

Each builder returns labels for ONE absurd rule. Base 0/1+lo-class2 is frozen
unless the hypothesis explicitly rebuilds the full labeling (A3 block valley).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sygnal_clustering.signal_extraction import (
    EPS,
    BASELINE_BINS,
    extract_prep_features,
    valley_ratio,
)

OFFSET, SHORT = 4, 42
DELTA = 0.003
Q_LO = 0.07
CHAMP_SCORE = 0.89109


@dataclass
class CrazyContext:
    """Shared features for all crazy pipelines."""

    x0: np.ndarray
    psd: np.ndarray
    decay: np.ndarray
    peak: np.ndarray
    charge: np.ndarray
    snr: np.ndarray
    i_peak: np.ndarray
    i_end: np.ndarray
    noise_std: np.ndarray
    baseline: np.ndarray
    thr: float
    qlo: float
    base: np.ndarray  # P14_2b_qlo_0070-equivalent


def build_base_lo_only(psd: np.ndarray, *, q_lo: float = Q_LO, delta: float = DELTA) -> tuple[np.ndarray, dict]:
    """Frozen contract: valley+δ for 0/1; class2 = psd < q_lo only."""
    vr, info = valley_ratio(psd, eps=EPS)
    thr = float(info["valley"]) + float(delta)
    lab = np.where(psd < thr, 0, 1).astype(np.int64)
    if psd[lab == 0].mean() > psd[lab == 1].mean():
        lab = 1 - lab
    qlo = float(np.quantile(psd[np.isfinite(psd)], q_lo))
    lab = lab.copy()
    lab[psd < qlo] = 2
    meta = {
        "valley_ratio": float(vr),
        "valley": float(info["valley"]),
        "thr": thr,
        "q_lo": q_lo,
        "qlo_value": qlo,
        "champion_score_ref": CHAMP_SCORE,
    }
    return lab, meta


def make_context(x_raw: np.ndarray) -> CrazyContext:
    prep = extract_prep_features(
        x_raw, polarity="negative", psd_offset=OFFSET, psd_short=SHORT
    )
    base, meta = build_base_lo_only(prep.psd)
    return CrazyContext(
        x0=prep.x0,
        psd=prep.psd,
        decay=prep.decay_time,
        peak=prep.peak_above,
        charge=prep.charge_roi,
        snr=prep.snr,
        i_peak=prep.i_peak,
        i_end=prep.i_end_roi,
        noise_std=prep.noise_std,
        baseline=prep.baseline,
        thr=meta["thr"],
        qlo=meta["qlo_value"],
        base=base,
    )


def _pca_sign_labels(x0: np.ndarray, i_peak: np.ndarray, i_end: np.ndarray, n_comp_bins: int = 32) -> np.ndarray:
    """Cheap 1D PCA-on-tail proxy: project normalized ROI tail onto first PC → valley split."""
    n = len(x0)
    mat = np.zeros((n, n_comp_bins), dtype=np.float64)
    for i in range(n):
        p, e = int(i_peak[i]), int(i_end[i])
        seg = x0[i, p : e + 1]
        if len(seg) < 2:
            continue
        # resample to fixed length
        idx = np.linspace(0, len(seg) - 1, n_comp_bins)
        y = np.interp(idx, np.arange(len(seg)), seg)
        pk = float(np.max(y)) + EPS
        mat[i] = y / pk
    # PCA via SVD
    mat = mat - mat.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(mat, full_matrices=False)
    pc1 = mat @ vt[0]
    vr, info = valley_ratio(pc1, eps=EPS)
    if not np.isfinite(vr) or "valley" not in info:
        # fallback median
        thr = float(np.median(pc1))
    else:
        thr = float(info["valley"])
    lab = np.where(pc1 < thr, 0, 1).astype(np.int64)
    if pc1[lab == 0].mean() > pc1[lab == 1].mean():
        lab = 1 - lab
    return lab


def _decay_labels(decay: np.ndarray) -> np.ndarray:
    """Valley split on discrete/continuous decay_time."""
    vr, info = valley_ratio(decay.astype(np.float64), eps=EPS)
    if not np.isfinite(vr) or "valley" not in info:
        thr = float(np.median(decay))
    else:
        thr = float(info["valley"])
    lab = np.where(decay < thr, 0, 1).astype(np.int64)
    # align: larger decay → slow → should match high-PSD class 1? empirically decay vs types weak;
    # polarity: class0 = lower mean decay
    if decay[lab == 0].mean() > decay[lab == 1].mean():
        lab = 1 - lab
    return lab


def _multi_peak_mask(x0: np.ndarray, i_peak: np.ndarray, i_end: np.ndarray, peak: np.ndarray) -> np.ndarray:
    n = len(x0)
    multi = np.zeros(n, dtype=bool)
    for i in range(n):
        p, e = int(i_peak[i]), int(i_end[i])
        seg = x0[i, p : e + 1]
        if len(seg) < 5:
            continue
        thr = 0.3 * float(peak[i])
        peaks = 0
        for j in range(1, len(seg) - 1):
            if seg[j] >= seg[j - 1] and seg[j] >= seg[j + 1] and seg[j] > thr:
                peaks += 1
        multi[i] = peaks >= 2
    return multi


# ----- hypothesis builders -----


def hyp_A0_disagree_to2(ctx: CrazyContext) -> tuple[np.ndarray, dict]:
    """A0: if PSD-label and decay-label disagree → 2; else PSD label; keep lo-reject."""
    psd_lab = np.where(ctx.psd < ctx.thr, 0, 1).astype(np.int64)
    if ctx.psd[psd_lab == 0].mean() > ctx.psd[psd_lab == 1].mean():
        psd_lab = 1 - psd_lab
    dec_lab = _decay_labels(ctx.decay)
    # align decay polarity to PSD on majority
    agree = (dec_lab == psd_lab).mean()
    if agree < 0.5:
        dec_lab = 1 - dec_lab
        agree = (dec_lab == psd_lab).mean()
    lab = psd_lab.copy()
    disagree = dec_lab != psd_lab
    lab[disagree] = 2
    lab[ctx.psd < ctx.qlo] = 2  # keep asymmetric lo
    return lab, {
        "rule": "psd_vs_decay_disagree->2 + lo_q07",
        "n_disagree": int(disagree.sum()),
        "agree_frac_before": float(agree),
    }


def hyp_A0b_pca_disagree_to2(ctx: CrazyContext) -> tuple[np.ndarray, dict]:
    """A0 variant: PSD vs PCA-tail disagree → 2."""
    psd_lab = np.where(ctx.psd < ctx.thr, 0, 1).astype(np.int64)
    if ctx.psd[psd_lab == 0].mean() > ctx.psd[psd_lab == 1].mean():
        psd_lab = 1 - psd_lab
    pca_lab = _pca_sign_labels(ctx.x0, ctx.i_peak, ctx.i_end)
    if (pca_lab == psd_lab).mean() < 0.5:
        pca_lab = 1 - pca_lab
    lab = psd_lab.copy()
    disagree = pca_lab != psd_lab
    lab[disagree] = 2
    lab[ctx.psd < ctx.qlo] = 2
    return lab, {"rule": "psd_vs_pca_disagree->2 + lo_q07", "n_disagree": int(disagree.sum())}


def hyp_A1_isthmus_flip(ctx: CrazyContext, *, half: float = 0.015) -> tuple[np.ndarray, dict]:
    """A1: flip 0↔1 only inside soft band; far+class2 frozen from base."""
    lab = ctx.base.copy()
    soft = (lab < 2) & (np.abs(ctx.psd - ctx.thr) < half)
    lab[soft] = 1 - lab[soft]
    return lab, {"rule": f"flip 0<->1 if |psd-thr|<{half}", "n_flipped": int(soft.sum()), "half": half}


def hyp_A2_multipeak_to2(ctx: CrazyContext) -> tuple[np.ndarray, dict]:
    """A2: multi-peak morphology → class 2 (process, not particle)."""
    lab = ctx.base.copy()
    multi = _multi_peak_mask(ctx.x0, ctx.i_peak, ctx.i_end, ctx.peak)
    lab[multi] = 2
    return lab, {"rule": "multi_peak->2 on base", "n_multi": int(multi.sum())}


def hyp_A2b_nonproportional_to2(ctx: CrazyContext, *, z_abs: float = 2.5) -> tuple[np.ndarray, dict]:
    """A2b: reject = violation of amp ∝ charge (residual of log-log fit)."""
    lab = ctx.base.copy()
    m = (lab < 2) & (ctx.peak > 0) & (ctx.charge > 0)
    x = np.log(ctx.peak[m] + EPS)
    y = np.log(ctx.charge[m] + EPS)
    # y ≈ a + b x
    b, a = np.polyfit(x, y, 1)
    pred = a + b * np.log(ctx.peak + EPS)
    resid = np.log(ctx.charge + EPS) - pred
    # z-score on 0/1 population
    mu, sd = float(resid[m].mean()), float(resid[m].std() + EPS)
    bad = m & (np.abs(resid - mu) > z_abs * sd)
    # also flag current class2 stays
    lab2 = lab.copy()
    lab2[bad] = 2
    return lab2, {
        "rule": f"|logCharge - (a+b logPeak)| > {z_abs} sigma -> 2",
        "n_nonprop": int(bad.sum()),
        "fit_a": float(a),
        "fit_b": float(b),
    }


def hyp_A3_index_blocks(ctx: CrazyContext, *, n_blocks: int = 8) -> tuple[np.ndarray, dict]:
    """A3: per index-block own valley+lo-reject (non-i.i.d. Run)."""
    n = len(ctx.psd)
    lab = np.full(n, 2, dtype=np.int64)
    edges = np.linspace(0, n, n_blocks + 1, dtype=int)
    block_meta = []
    for b in range(n_blocks):
        sl = slice(int(edges[b]), int(edges[b + 1]))
        psd_b = ctx.psd[sl]
        if len(psd_b) < 200:
            lab[sl] = ctx.base[sl]
            block_meta.append({"block": b, "fallback": "base", "n": int(len(psd_b))})
            continue
        lab_b, meta_b = build_base_lo_only(psd_b, q_lo=Q_LO, delta=DELTA)
        lab[sl] = lab_b
        block_meta.append({"block": b, "thr": meta_b["thr"], "qlo": meta_b["qlo_value"], "n": int(len(psd_b))})
    return lab, {"rule": f"per-index {n_blocks} blocks valley+lo7%", "blocks": block_meta}


def hyp_A3b_rolling_baseline(
    x_raw: np.ndarray, *, window: int = 500, baseline_bins: int = BASELINE_BINS
) -> tuple[np.ndarray, dict]:
    """A3b: drifting pedestal — local baseline from neighboring events' first bins."""
    from sygnal_clustering.signal_extraction import apply_polarity

    x_pos = apply_polarity(x_raw, "negative")
    n, t = x_pos.shape
    # per-event crude pedestal from first bins
    ped = x_pos[:, :baseline_bins].mean(axis=1)
    # rolling mean of pedestals
    ker = np.ones(window) / window
    # pad reflect
    pad = window // 2
    ped_pad = np.pad(ped, (pad, pad), mode="edge")
    ped_roll = np.convolve(ped_pad, ker, mode="valid")
    if len(ped_roll) > n:
        ped_roll = ped_roll[:n]
    elif len(ped_roll) < n:
        ped_roll = np.pad(ped_roll, (0, n - len(ped_roll)), mode="edge")
    x0 = x_pos - ped_roll[:, None]
    # recompute PSD-like features on drifted x0
    i_peak = np.argmax(x0, axis=1)
    peak = x0[np.arange(n), i_peak]
    noise = x_pos[:, :baseline_bins].std(axis=1)
    psd = np.empty(n, dtype=np.float64)
    for i in range(n):
        p = int(i_peak[i])
        thr_n = 3.0 * float(noise[i])
        end = n_end = t - 1
        for j in range(p, t):
            if x0[i, j] <= thr_n:
                end = j
                break
        roi = x0[i, p : end + 1]
        charge = float(roi.sum()) if len(roi) else 0.0
        s0 = min(max(len(roi) - 1, 0), OFFSET) if len(roi) else 0
        s1 = min(len(roi), s0 + SHORT)
        short = float(roi[s0:s1].sum()) if len(roi) else 0.0
        psd[i] = (charge - short) / (charge + EPS)
    lab, meta = build_base_lo_only(psd)
    meta.update({"rule": f"rolling pedestal window={window} then lo-only PSD", "roll_window": window})
    return lab, meta


def hyp_A4b_decay_extremes_to2(ctx: CrazyContext) -> tuple[np.ndarray, dict]:
    """A4b: discrete decay hang / min extremes → 2; else base."""
    lab = ctx.base.copy()
    tmax = float(ctx.x0.shape[1] - 1)
    hang = ctx.decay >= (ctx.x0.shape[1] - ctx.i_peak - 1)
    dmin = ctx.decay <= 1.0
    # also top discrete bin that is rare
    rare_hi = ctx.decay >= np.quantile(ctx.decay, 0.99)
    mask = hang | dmin | rare_hi
    lab[mask] = 2
    return lab, {
        "rule": "decay hang|<=1|q99 -> 2 on base",
        "n_hang": int(hang.sum()),
        "n_dmin": int(dmin.sum()),
        "n_rare_hi": int(rare_hi.sum()),
        "n_mask": int(mask.sum()),
    }


def hyp_A6c_energy_quartile_mirror(ctx: CrazyContext, *, quartile: int = 0) -> tuple[np.ndarray, dict]:
    """A6c: flip 0↔1 inside one peak_above quartile; rest = base."""
    lab = ctx.base.copy()
    # quartiles on all events
    qs = np.quantile(ctx.peak, [0.0, 0.25, 0.5, 0.75, 1.0])
    lo, hi = qs[quartile], qs[quartile + 1]
    # last bin inclusive
    if quartile == 3:
        m = (ctx.peak >= lo) & (ctx.peak <= hi) & (lab < 2)
    else:
        m = (ctx.peak >= lo) & (ctx.peak < hi) & (lab < 2)
    lab[m] = 1 - lab[m]
    return lab, {
        "rule": f"mirror 0<->1 in peak quartile {quartile}",
        "n_flipped": int(m.sum()),
        "peak_lo": float(lo),
        "peak_hi": float(hi),
    }


def hyp_A6b_seed_from_text(ctx: CrazyContext, *, seed: int = 27052019, frac: float = 0.10) -> tuple[np.ndarray, dict]:
    """A6b: force random frac → 2 with seed from Description bibliography date 27.05.2019."""
    lab = ctx.base.copy()
    rng = np.random.default_rng(seed)
    # only puncture current 0/1
    idx01 = np.where(lab < 2)[0]
    k = int(round(frac * len(lab)))
    k = min(k, len(idx01))
    pick = rng.choice(idx01, size=k, replace=False)
    lab[pick] = 2
    return lab, {"rule": f"rng(seed={seed}) punch {frac:.0%} of run -> 2", "n_punch": int(k), "seed": seed}


def hyp_A5b_balance_ritual(ctx: CrazyContext) -> tuple[np.ndarray, dict]:
    """A5b: ritual fractions ~45/45/10 by expanding lo-reject until f2≈10%."""
    psd_lab = np.where(ctx.psd < ctx.thr, 0, 1).astype(np.int64)
    if ctx.psd[psd_lab == 0].mean() > ctx.psd[psd_lab == 1].mean():
        psd_lab = 1 - psd_lab
    target_f2 = 0.10
    q = float(np.quantile(ctx.psd, target_f2))
    lab = psd_lab.copy()
    lab[ctx.psd < q] = 2
    return lab, {"rule": "ritual f2~10% lo-only", "qlo_value": q, "target_f2": target_f2}


PIPELINE_REGISTRY = {
    "A0_disagree_decay": hyp_A0_disagree_to2,
    "A0b_disagree_pca": hyp_A0b_pca_disagree_to2,
    "A1_isthmus_flip": hyp_A1_isthmus_flip,
    "A2_multipeak": hyp_A2_multipeak_to2,
    "A2b_nonproportional": hyp_A2b_nonproportional_to2,
    "A3_index_blocks": hyp_A3_index_blocks,
    "A4b_decay_extremes": hyp_A4b_decay_extremes_to2,
    "A5b_balance_ritual": hyp_A5b_balance_ritual,
    "A6b_seed_text": hyp_A6b_seed_from_text,
    "A6c_q0_mirror": lambda ctx: hyp_A6c_energy_quartile_mirror(ctx, quartile=0),
    "A6c_q1_mirror": lambda ctx: hyp_A6c_energy_quartile_mirror(ctx, quartile=1),
    "A6c_q2_mirror": lambda ctx: hyp_A6c_energy_quartile_mirror(ctx, quartile=2),
    "A6c_q3_mirror": lambda ctx: hyp_A6c_energy_quartile_mirror(ctx, quartile=3),
}
