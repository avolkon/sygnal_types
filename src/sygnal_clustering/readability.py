"""H_READ: type = PSD; class2 = unreadable scintillation tail (statistics).

Consolidated from verified LB laws only (see 0808_Вычитка_консилиум_unreadable.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sygnal_clustering.signal_extraction import (
    EPS,
    extract_prep_features,
    valley_ratio,
)

OFFSET, SHORT = 4, 42
DELTA = 0.003
CHAMP_SCORE = 0.89109
Q_LO_CHAMP = 0.07


@dataclass
class ReadFeatures:
    psd: np.ndarray
    long: np.ndarray
    short: np.ndarray
    z_tail: np.ndarray
    snr: np.ndarray
    peak: np.ndarray
    thr: float


def extract_read_features(x_raw: np.ndarray) -> ReadFeatures:
    prep = extract_prep_features(
        x_raw, polarity="negative", psd_offset=OFFSET, psd_short=SHORT
    )
    long = prep.charge_roi
    short = prep.short
    # Poisson-scale tail contrast (real counting statistics prior)
    z_tail = (long - short) / np.sqrt(np.maximum(long, 0.0) + EPS)
    vr, info = valley_ratio(prep.psd, eps=EPS)
    thr = float(info["valley"]) + DELTA
    return ReadFeatures(
        psd=prep.psd,
        long=long,
        short=short,
        z_tail=z_tail.astype(np.float64),
        snr=prep.snr,
        peak=prep.peak_above,
        thr=thr,
    )


def labels_type_psd(psd: np.ndarray, thr: float) -> np.ndarray:
    lab = np.where(psd < thr, 0, 1).astype(np.int64)
    if psd[lab == 0].mean() > psd[lab == 1].mean():
        lab = 1 - lab
    return lab


def labels_h_read(
    feat: ReadFeatures,
    *,
    q_lo: float = Q_LO_CHAMP,
    z_cut: float | None = None,
    s_cut: float | None = None,
    q_lo_floor: float | None = None,
) -> tuple[np.ndarray, dict]:
    """H_READ labeling.

    - 0/1 from PSD thr (locked family)
    - never reject high-PSD side of thr as a tail policy
    - class2 inside low-PSD door: weak z_tail and/or weak snr
    - optional hard floor q_lo_floor always -> 2
    """
    psd, thr = feat.psd, feat.thr
    lab01 = labels_type_psd(psd, thr)
    qlo = float(np.quantile(psd[np.isfinite(psd)], q_lo))
    door = psd < qlo

    weak = np.zeros(len(psd), dtype=bool)
    if z_cut is not None:
        weak |= feat.z_tail < float(z_cut)
    if s_cut is not None:
        weak |= feat.snr < float(s_cut)
    if z_cut is None and s_cut is None:
        # pure lo-quantile baseline (champ family)
        weak[:] = True

    lab = lab01.copy()
    lab[door & weak] = 2
    if q_lo_floor is not None:
        qfloor = float(np.quantile(psd[np.isfinite(psd)], q_lo_floor))
        lab[psd < qfloor] = 2

    # safety: never mark as 2 solely for being above type thr with high psd
    # (hi-tail sacred) — already satisfied because door uses low quantile only

    meta = {
        "hypothesis": "H_READ",
        "thr": thr,
        "q_lo": q_lo,
        "qlo_value": qlo,
        "z_cut": z_cut,
        "s_cut": s_cut,
        "q_lo_floor": q_lo_floor,
        "n_door": int(door.sum()),
        "n_class2": int((lab == 2).sum()),
        "champion_score_ref": CHAMP_SCORE,
    }
    return lab, meta


def champ_lo_only(feat: ReadFeatures, *, q_lo: float = Q_LO_CHAMP) -> np.ndarray:
    lab, _ = labels_h_read(feat, q_lo=q_lo, z_cut=None, s_cut=None)
    return lab
