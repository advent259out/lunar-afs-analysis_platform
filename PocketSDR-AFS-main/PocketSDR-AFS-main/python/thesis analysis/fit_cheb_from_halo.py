#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Chebyshev segmented fit to HALO truth with 1 Hz FITTING (segment remains 3600 s)

- Segments are fixed-length windows (default 3600 s).
- Within each segment, we FIT the Chebyshev polynomials using a sub-sampled
  time grid (default every 1.0 s), but we EVALUATE the fitted polynomials on
  the full-resolution time grid of that segment to compute residuals.
- Degree is reduced adaptively if the fit sub-grid has too few samples.

Data:
  E:/Project/HALO/output_HALO/halo_prn1.csv ... halo_prn8.csv
CSV columns:
  prn,t_sec,x_km,y_km,z_km,vx_kms,vy_kms,vz_kms
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import chebyshev

# ------------------------------ PATHS/DEFAULTS ------------------------------
HALO_DIR = Path(r"E:\Project\HALO")
DEFAULT_PRNS = [8]
DEFAULT_ORDERS = [4]
DEFAULT_SEG_LEN = 3600.0  # s
DEFAULT_OVERLAP = 0.0     # fraction
DEFAULT_FIT_STEP = 1.0    # s (fit only; evaluation stays full-res)

# ------------------------------ IO -----------------------------------------
def load_halo_state(prn: int) -> Tuple[np.ndarray, np.ndarray]:
    """Load time (s) and ECEF position (m) for a given PRN."""
    path = HALO_DIR / f"halo_prn{prn}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    # Columns: prn, t_sec, x_km, y_km, z_km, vx_kms, vy_kms, vz_kms
    times = data[:, 1]
    pos = data[:, 2:5] * 1000.0  # km -> m
    return times, pos

# ------------------------------ SEGMENTING ----------------------------------
def make_segments(times: np.ndarray, seg_len: float, overlap: float) -> List[np.ndarray]:
    """Build fixed-length segments with optional fractional overlap."""
    assert 0.0 <= overlap < 0.9, "overlap in [0,0.9)"
    if seg_len <= 0:
        return [np.ones_like(times, dtype=bool)]

    t0 = float(times[0]); tN = float(times[-1])
    step = seg_len * (1.0 - overlap)
    if step <= 0:
        step = seg_len

    segs = []
    start = t0
    eps = 1e-9
    while start <= tN + eps:
        end = start + seg_len
        mask = (times >= start - eps) & (times <= end + eps)
        if np.count_nonzero(mask) >= 1:
            segs.append(mask)
        start += step

    if not segs:
        segs = [np.ones_like(times, dtype=bool)]
    return segs

# ------------------------------ FIT/EVAL HELPERS ----------------------------
def choose_fit_indices(t_seg: np.ndarray, fit_step: float) -> np.ndarray:
    """
    Select indices within a segment to approximate an every-`fit_step` seconds grid.
    We do this by thinning using stride derived from median dt.
    """
    if fit_step <= 0:
        return np.arange(len(t_seg))
    if len(t_seg) <= 2:
        return np.arange(len(t_seg))

    dt_med = float(np.median(np.diff(t_seg)))
    if dt_med <= 0:
        return np.arange(len(t_seg))
    stride = max(1, int(round(fit_step / dt_med)))
    return np.arange(0, len(t_seg), stride)

def cheb_fit_segment_fit_subgrid_eval_full(
    t_seg_all: np.ndarray,
    y_seg_all: np.ndarray,
    order_req: int,
    fit_step: float
) -> np.ndarray:
    """
    Fit Chebyshev on a sub-grid (every `fit_step` seconds) in this segment,
    using the segment's [t_min, t_max] mapping to [-1,1], then evaluate on ALL
    times in the segment.
    """
    t_min = float(t_seg_all.min())
    t_max = float(t_seg_all.max())
    if t_max <= t_min:
        return np.full_like(y_seg_all, y_seg_all[0])

    # Build sub-grid indices
    idx_fit_local = choose_fit_indices(t_seg_all, fit_step)
    t_fit = t_seg_all[idx_fit_local]
    y_fit = y_seg_all[idx_fit_local]

    # Adaptive degree
    n_fit = len(t_fit)
    deg = int(min(max(order_req, 0), n_fit - 1)) if n_fit >= 2 else 0

    # Map both fit/eval times to [-1,1] with SAME (t_min,t_max)
    tau_fit = 2.0 * (t_fit - t_min) / (t_max - t_min) - 1.0
    tau_all = 2.0 * (t_seg_all - t_min) / (t_max - t_min) - 1.0

    if deg <= 0:
        # constant model
        y0 = float(np.mean(y_fit)) if n_fit > 0 else float(y_seg_all[0])
        return np.full_like(y_seg_all, y0)

    coeffs = chebyshev.chebfit(tau_fit, y_fit, deg=deg)
    y_eval = chebyshev.chebval(tau_all, coeffs)
    return y_eval

def cheb_fit_eval_global(times_all: np.ndarray, values_all: np.ndarray, order: int) -> np.ndarray:
    """(Optional) Global fit without sub-sampling (for comparison)."""
    t_min = float(times_all.min()); t_max = float(times_all.max())
    if t_max <= t_min:
        return np.copy(values_all)
    tau = 2.0 * (times_all - t_min) / (t_max - t_min) - 1.0
    coeffs = chebyshev.chebfit(tau, values_all, deg=order)
    return chebyshev.chebval(tau, coeffs)

# ------------------------------ PLOTTING -------------------------------------
def plot_errors(time_rel: np.ndarray,
                errors_per_prn: Dict[int, np.ndarray],
                title: str,
                outfile: Path):
    plt.figure(figsize=(10, 5))
    for prn, errs in errors_per_prn.items():
        plt.plot(time_rel, errs, label=f"PRN {prn}", linewidth=0.9)
    rms = np.sqrt(np.nanmean([errs ** 2 for errs in errors_per_prn.values()], axis=0))
    plt.plot(time_rel, rms, color="k", linewidth=2.0, label="RMS")
    plt.xlabel("Time since start (s)")
    plt.ylabel("Position error magnitude (m)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    print(f"Saved: {outfile}")

# ------------------------------ CLI -----------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Segmented Chebyshev with 1 Hz fitting (segment=3600 s).")
    p.add_argument("--orders", type=int, nargs="+", default=DEFAULT_ORDERS,
                   help="Chebyshev orders to evaluate (default: 5 7 9).")
    p.add_argument("--prn", type=int, nargs="+", default=list(DEFAULT_PRNS),
                   help="PRN numbers to include (default: 2..8).")
    p.add_argument("--seg-len", type=float, default=DEFAULT_SEG_LEN,
                   help="Segment length in seconds (default: 3600).")
    p.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP,
                   help="Fractional overlap [0..0.9), default 0.")
    p.add_argument("--fit-step", type=float, default=DEFAULT_FIT_STEP,
                   help="Fit sub-sampling step in seconds (default: 1.0s).")
    p.add_argument("--global-fit", action="store_true",
                   help="Also run a global fit (no sub-sampling) for comparison.")
    return p.parse_args()

# ------------------------------ MAIN ----------------------------------------
def main():
    args = parse_args()
    orders: List[int] = args.orders
    prns: List[int] = sorted(set(args.prn))

    # Load truth for each PRN
    time_ref = None
    pos_truth: Dict[int, np.ndarray] = {}
    for prn in prns:
        times, pos = load_halo_state(prn)
        if time_ref is None:
            time_ref = times
        else:
            if not np.allclose(times, time_ref):
                raise ValueError(f"Time grid mismatch for PRN {prn}")
        pos_truth[prn] = pos

    assert time_ref is not None
    t0 = float(time_ref[0])
    time_rel = time_ref - t0

    # Optional: global fit (for comparison; no sub-sampling)
    if args.global_fit:
        for order in orders:
            errors_per_prn: Dict[int, np.ndarray] = {}
            for prn in prns:
                truth = pos_truth[prn]
                fit_pos = np.zeros_like(truth)
                for axis in range(3):
                    fit_pos[:, axis] = cheb_fit_eval_global(time_ref, truth[:, axis], order)
                diff = np.linalg.norm(fit_pos - truth, axis=1)
                errors_per_prn[prn] = diff
            out_path = HALO_DIR / f"cheb_fit_order{order}_per_prn_GLOBAL.png"
            plot_errors(time_rel, errors_per_prn,
                        f"GLOBAL Chebyshev order {order} vs HALO truth",
                        out_path)

    # Segments (3600s each by default)
    segments = make_segments(time_ref, seg_len=args.seg_len, overlap=args.overlap)

    # Segmented fit using 1 Hz (or user-set) sub-grid for FITTING
    for order in orders:
        errors_per_prn: Dict[int, np.ndarray] = {prn: np.full_like(time_ref, np.nan, dtype=float) for prn in prns}

        for prn in prns:
            truth = pos_truth[prn]
            fit_pos_all = np.full_like(truth, np.nan, dtype=float)

            for mask in segments:
                idx = np.where(mask)[0]
                if idx.size == 0:
                    continue
                t_seg_all = time_ref[idx]

                for axis in range(3):
                    y_seg_all = truth[idx, axis]
                    y_eval = cheb_fit_segment_fit_subgrid_eval_full(
                        t_seg_all, y_seg_all, order_req=order, fit_step=args.fit_step
                    )
                    fit_pos_all[idx, axis] = y_eval

            valid = np.all(np.isfinite(fit_pos_all), axis=1)
            errs = np.full_like(time_ref, np.nan, dtype=float)
            errs[valid] = np.linalg.norm(fit_pos_all[valid] - truth[valid], axis=1)
            errors_per_prn[prn] = errs

        tag = f"SEGMENTED_{int(args.seg_len)}s_fitstep{int(round(args.fit_step))}s_overlap{int(args.overlap*100)}"
        out_path = HALO_DIR / f"cheb_fit_order{order}_per_prn_{tag}.png"
        plot_errors(time_rel, errors_per_prn,
                    f"SEGMENTED Chebyshev order {order} (seg={int(args.seg_len)}s, fit-step={args.fit_step:.1f}s)",
                    out_path)

if __name__ == "__main__":
    main()
