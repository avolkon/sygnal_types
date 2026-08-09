"""FE-1..FE-3 candidates on live X_clust (no top-q% quota as DoD)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sygnal_clustering.signal_extraction import (  # noqa: E402
    BASELINE_BINS,
    DECAY_FRAC,
    EPS,
    N_SIGMA,
    VALLEY_RATIO_MAX,
    build_qc_flags,
    calibrate_psd_windows,
    extract_prep_features,
    valley_ratio,
)

DATA_NAME = "Run200_Wave_0_1.txt"
N_SAMPLES = 23_479
DROP_COLS = [0, 1, 2, 3, 504]
POLARITY = "negative"
RANDOM_STATE = 42
OUT = ROOT / "submissions" / "fe_candidates"


def load_X() -> np.ndarray:
    path = ROOT / "data" / DATA_NAME
    raw = pd.read_csv(path, sep=" ", header=None, skipinitialspace=True)
    X = raw.drop(columns=DROP_COLS, errors="ignore").to_numpy(dtype=np.float64)
    assert X.shape == (N_SAMPLES, 500)
    return X


def remap_by_psd(labels: np.ndarray, psd: np.ndarray) -> np.ndarray:
    """Map two particle clusters: lower mean PSD -> 0, higher -> 1; keep 2."""
    out = labels.astype(np.int64).copy()
    particle = [u for u in np.unique(out) if u != 2]
    if len(particle) < 2:
        # if only 0/1 present without 2
        particle = sorted(int(u) for u in np.unique(out))[:2]
    if len(particle) < 2:
        return out
    stats = sorted([(u, float(psd[out == u].mean())) for u in particle], key=lambda t: t[1])
    mapping = {stats[0][0]: 0, stats[1][0]: 1}
    remapped = np.full(len(out), 2, dtype=np.int64)
    for old, new in mapping.items():
        remapped[out == old] = new
    remapped[out == 2] = 2
    return remapped


def fractions(labels: np.ndarray) -> dict[str, float]:
    c = np.bincount(labels.astype(int), minlength=3)
    fr = c / c.sum()
    return {f"f{i}": round(float(fr[i]), 4) for i in range(3)}


def save_sub(name: str, labels: np.ndarray, meta: dict) -> None:
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"index": np.arange(N_SAMPLES), "cluster": labels.astype(int)}).to_csv(
        d / "submission.csv", index=False
    )
    (d / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(name, fractions(labels), meta.get("note", ""))


def main() -> None:
    X = load_X()
    prep0 = extract_prep_features(X, polarity=POLARITY)
    off, short, vr_cal, _ = calibrate_psd_windows(
        prep0.x0,
        prep0.i_peak,
        prep0.i_end_roi,
        prep0.noise_std,
        offsets=[1, 2, 3, 5, 8, 12, 16, 24],
        shorts=[10, 15, 20, 30, 40, 50, 60, 80],
        max_grid_points=64,
        subsample=5000,
        random_state=RANDOM_STATE,
    )
    prep = extract_prep_features(
        X,
        polarity=POLARITY,
        baseline_bins=BASELINE_BINS,
        n_sigma=N_SIGMA,
        decay_frac=DECAY_FRAC,
        psd_offset=off,
        psd_short=short,
        eps=EPS,
    )
    vr_psd, info_psd = valley_ratio(prep.psd, eps=EPS)
    gate_ok = vr_psd < VALLEY_RATIO_MAX
    qc = build_qc_flags(prep)
    cand2 = qc["candidate_class2"]

    feat = np.column_stack([prep.psd, prep.decay_time, prep.charge_roi, prep.tail_ratio])
    if not np.isfinite(feat).all():
        raise ValueError("non-finite features")
    X_clust = RobustScaler().fit_transform(feat)

    # FE-1 sanity summary
    sanity = {
        "psd_windows": [int(off), int(short)],
        "valley_ratio_psd": float(vr_psd),
        "GATE_OK": bool(gate_ok),
        "feature_std": {
            "psd": float(prep.psd.std()),
            "decay": float(prep.decay_time.std()),
            "charge": float(prep.charge_roi.std()),
            "tail": float(prep.tail_ratio.std()),
        },
        "corr_psd_charge": float(np.corrcoef(prep.psd, prep.charge_roi)[0, 1]),
        "legacy_leak_note": "Model5 in notebook uses features[:,:4] not X_clust — FE candidates use X_clust / PSD",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fe1_sanity.json").write_text(json.dumps(sanity, indent=2), encoding="utf-8")
    print("FE-1 sanity", sanity)

    # --- A: GMM-2 on X_clust, class2 = QC only ---
    gmm = GaussianMixture(
        n_components=2, covariance_type="full", n_init=20, random_state=RANDOM_STATE
    )
    gmm.fit(X_clust)
    lab = gmm.predict(X_clust).astype(np.int64)
    lab = remap_by_psd(lab, prep.psd)
    lab[cand2] = 2
    save_sub(
        "A_gmm2_xclust_qc",
        lab,
        {"note": "GMM-2 on live X_clust; class2=QC OR; remap by mean PSD", **fractions(lab)},
    )

    # --- B: GMM-2 on X_clust, class2 = PSD overlap around valley ---
    valley = float(info_psd.get("valley", np.median(prep.psd)))
    # half-width: 15% of IQR or fallback
    iqr = float(np.subtract(*np.quantile(prep.psd, [0.75, 0.25])))
    half = max(0.02, 0.15 * iqr)
    overlap = (prep.psd >= valley - half) & (prep.psd <= valley + half)
    lab_b = gmm.predict(X_clust).astype(np.int64)
    lab_b = remap_by_psd(lab_b, prep.psd)
    lab_b[overlap | cand2] = 2
    save_sub(
        "B_gmm2_xclust_overlap_qc",
        lab_b,
        {
            "note": f"class2 = PSD overlap ±{half:.4f} around valley={valley:.4f} OR QC",
            "overlap_rate": float(overlap.mean()),
            **fractions(lab_b),
        },
    )

    # --- C: 1D GMM on PSD only + QC ---
    z_psd = RobustScaler().fit_transform(prep.psd.reshape(-1, 1))
    gmm1 = GaussianMixture(
        n_components=2, covariance_type="full", n_init=20, random_state=RANDOM_STATE
    )
    gmm1.fit(z_psd)
    lab_c = gmm1.predict(z_psd).astype(np.int64)
    lab_c = remap_by_psd(lab_c, prep.psd)
    lab_c[cand2] = 2
    save_sub(
        "C_gmm2_psd1d_qc",
        lab_c,
        {"note": "GMM-2 on calibrated PSD only; class2=QC", **fractions(lab_c)},
    )

    # --- D: GMM-2 X_clust + absolute uncertainty threshold (not top-q%) ---
    proba = gmm.predict_proba(X_clust)
    unc = 1.0 - proba.max(axis=1)
    # choose tau so class2 ~ emergent: start from quantile of unc among non-QC, but report tau
    tau = float(np.quantile(unc, 0.97))  # absolute threshold = 97% quantile value, then apply as cut
    lab_d = gmm.predict(X_clust).astype(np.int64)
    lab_d = remap_by_psd(lab_d, prep.psd)
    lab_d[(unc > tau) | cand2] = 2
    save_sub(
        "D_gmm2_xclust_abs_unc_qc",
        lab_d,
        {
            "note": "absolute unc threshold tau=q97(unc) OR QC (not top-q% label policy)",
            "tau": tau,
            "unc_gt_tau_rate": float((unc > tau).mean()),
            **fractions(lab_d),
        },
    )

    # --- E: covariance_type sweep pick by BIC on X_clust (2 components), class2=QC ---
    best = None
    for cov in ("full", "tied", "diag", "spherical"):
        m = GaussianMixture(
            n_components=2, covariance_type=cov, n_init=10, random_state=RANDOM_STATE
        )
        m.fit(X_clust)
        bic = float(m.bic(X_clust))
        if best is None or bic < best[0]:
            best = (bic, cov, m)
    assert best is not None
    lab_e = best[2].predict(X_clust).astype(np.int64)
    lab_e = remap_by_psd(lab_e, prep.psd)
    lab_e[cand2] = 2
    save_sub(
        "E_gmm2_bestbic_qc",
        lab_e,
        {"note": f"best BIC covariance={best[1]} bic={best[0]:.1f}; class2=QC", **fractions(lab_e)},
    )

    summary = {
        "gate_ok": gate_ok,
        "psd_windows": [off, short],
        "candidates": sorted(p.name for p in OUT.iterdir() if p.is_dir()),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("done", summary)


if __name__ == "__main__":
    main()
