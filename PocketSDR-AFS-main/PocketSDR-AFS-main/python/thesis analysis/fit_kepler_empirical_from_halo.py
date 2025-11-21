#!/usr/bin/env python3
"""
Fit Kepler + empirical acceleration model to HALO truth trajectories.

For each PRN (1..8) the script:
  - reads E:\\Project\\HALO\\output\\halo_prnX.csv
  - converts Cartesian states to osculating Keplerian elements at the first epoch
  - computes observed accelerations, removes two-body lunar gravity
  - fits empirical accelerations in RSW frame with basis {1, cos(k n t), sin(k n t), k=1..N}
  - writes results to halo_kepler_empirical.csv and prints summary statistics

Harmonic order N is controlled by HARMONIC_ORDER at the top of the file.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MU_MOON = 4.9028e12  # m^3/s^2 (GM of the Moon)
HALO_DIR = Path(r"E:\Project\HALO\output")
PRNS = range(1, 9)
OUT_CSV = HALO_DIR / "halo_kepler_empirical.csv"

# 支持任意阶 harmonic：修改这里即可
# 例如：
#   HARMONIC_ORDER = 1  -> 1 阶：1, cos(nt), sin(nt)
#   HARMONIC_ORDER = 3  -> 3 阶：1, cos(nt), sin(nt), cos(2nt), sin(2nt), cos(3nt), sin(3nt)
HARMONIC_ORDER = 3  # 你可以改成任意正整数


# ---------------------------------------------------------------------------
# Data class to hold results
# ---------------------------------------------------------------------------

@dataclass
class KeplerEmpiricalParams:
    prn: int
    epoch: float
    a: float
    e: float
    inc: float
    raan: float
    argp: float
    mean_anom: float
    mean_motion: float
    harmonic_order: int
    coeff_r: List[float]  # length = 1 + 2 * harmonic_order
    coeff_s: List[float]
    coeff_w: List[float]
    rms_residual: float


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_halo_state(prn: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load time, position, velocity from halo_prnX.csv.

    Assumes CSV columns:
        [*, t, x, y, z, vx, vy, vz]
    with x,y,z in km and vx,vy,vz in km/s.
    """
    path = HALO_DIR / f"halo_prn{prn}.csv"
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    times = data[:, 1]
    pos = data[:, 2:5] * 1000.0  # km -> m
    vel = data[:, 5:8] * 1000.0  # km/s -> m/s
    return times, pos, vel


# ---------------------------------------------------------------------------
# Orbital element conversion
# ---------------------------------------------------------------------------

def rv_to_kepler(r: np.ndarray, v: np.ndarray, mu: float) -> tuple[float, float, float, float, float, float]:
    """
    Convert Cartesian state (r,v) to classical Keplerian elements:
        a, e, inc, raan, argp, mean anomaly (M)
    Angles are in radians.
    """
    h = np.cross(r, v)
    h_norm = np.linalg.norm(h)
    r_norm = np.linalg.norm(r)
    v_norm = np.linalg.norm(v)

    n_vec = np.cross([0.0, 0.0, 1.0], h)
    n_norm = np.linalg.norm(n_vec)

    e_vec = (np.cross(v, h) / mu) - (r / r_norm)
    e = np.linalg.norm(e_vec)

    energy = v_norm ** 2 / 2.0 - mu / r_norm
    a = -mu / (2.0 * energy)

    inc = np.arccos(h[2] / h_norm)

    # RAAN
    if n_norm > 1e-12:
        raan = np.arctan2(n_vec[1], n_vec[0])
    else:
        raan = 0.0

    # Argument of perigee
    if e > 1e-12 and n_norm > 1e-12:
        argp = np.arccos(np.dot(n_vec, e_vec) / (n_norm * e))
        if e_vec[2] < 0:
            argp = 2.0 * np.pi - argp
    else:
        argp = 0.0

    # True anomaly
    if e > 1e-12:
        nu = np.arccos(np.dot(e_vec, r) / (e * r_norm))
        if np.dot(r, v) < 0:
            nu = 2.0 * np.pi - nu
    else:
        # circular orbit: fall back to node-based definition
        cp = np.cross(n_vec, r)
        if np.linalg.norm(cp) > 0:
            nu = np.arccos(np.dot(n_vec, r) / (n_norm * r_norm))
            if r[2] < 0:
                nu = 2.0 * np.pi - nu
        else:
            nu = 0.0

    # Mean anomaly for elliptic case
    if e < 1.0:
        E = 2.0 * np.arctan2(np.tan(nu / 2.0), np.sqrt((1 + e) / (1 - e)))
        M = E - e * np.sin(E)
    else:
        # Hyperbolic or parabolic: not expected for HALO
        M = nu

    return (
        np.mod(a, np.inf),
        e,
        np.mod(inc, 2.0 * np.pi),
        np.mod(raan, 2.0 * np.pi),
        np.mod(argp, 2.0 * np.pi),
        np.mod(M, 2.0 * np.pi),
    )


# ---------------------------------------------------------------------------
# Residual accelerations in RSW frame
# ---------------------------------------------------------------------------

def residual_accelerations(times: np.ndarray,
                           pos: np.ndarray,
                           vel: np.ndarray) -> np.ndarray:
    """
    Compute residual accelerations a_res = a_obs - a_two_body
    and project them into the RSW frame.

    Returns:
        residuals_rsw: shape (N, 3) with columns [a_R, a_S, a_W] in m/s^2
    """
    # Numerical derivative of velocity -> observed acceleration (inertial)
    acc_obs = np.gradient(vel, times, axis=0)

    # Two-body lunar gravity
    r_norm = np.linalg.norm(pos, axis=1)
    grav = -MU_MOON * pos / (r_norm[:, None] ** 3)

    # Residual acceleration (all perturbations)
    a_res = acc_obs - grav

    # Build RSW triad
    r_hat = pos / r_norm[:, None]
    h_vec = np.cross(pos, vel)
    h_hat = h_vec / np.linalg.norm(h_vec, axis=1)[:, None]
    s_hat = np.cross(h_hat, r_hat)

    # Project residuals into R,S,W
    a_r = np.einsum("ij,ij->i", a_res, r_hat)
    a_s = np.einsum("ij,ij->i", a_res, s_hat)
    a_w = np.einsum("ij,ij->i", a_res, h_hat)

    return np.column_stack((a_r, a_s, a_w))


# ---------------------------------------------------------------------------
# Empirical acceleration fitting with arbitrary harmonic order
# ---------------------------------------------------------------------------

def build_harmonic_basis(dt: np.ndarray,
                         mean_motion: float,
                         n_harmonics: int) -> np.ndarray:
    """
    Construct basis matrix for arbitrary harmonic order.

    Columns:
        [1,
         cos(1*n*t), sin(1*n*t),
         cos(2*n*t), sin(2*n*t),
         ...
         cos(n*n*t), sin(n*n*t)]
    """
    cols = [np.ones_like(dt)]
    for k in range(1, n_harmonics + 1):
        w = k * mean_motion
        cols.append(np.cos(w * dt))
        cols.append(np.sin(w * dt))
    basis = np.column_stack(cols)
    return basis


def fit_empirical_components(times: np.ndarray,
                             residuals: np.ndarray,
                             mean_motion: float,
                             n_harmonics: int) -> tuple[np.ndarray, float]:
    """
    Fit empirical acceleration components in RSW frame using
    harmonics up to order n_harmonics.

    residuals: shape (N,3) [R,S,W]
    Returns:
        coeffs: shape (3, M) where M = 1 + 2*n_harmonics
        rms:    scalar RMS of 3D residual after fitting
    """
    dt = times - times[0]
    basis = build_harmonic_basis(dt, mean_motion, n_harmonics)

    num_axes = residuals.shape[1]  # should be 3
    num_coeffs = basis.shape[1]
    coeffs = np.zeros((num_axes, num_coeffs))
    fitted = np.zeros_like(residuals)

    for axis in range(num_axes):
        y = residuals[:, axis]
        c, *_ = np.linalg.lstsq(basis, y, rcond=None)
        coeffs[axis, :] = c
        fitted[:, axis] = basis @ c

    residual_after = residuals - fitted
    rms = np.sqrt(np.mean(np.sum(residual_after ** 2, axis=1)))
    return coeffs, rms


# ---------------------------------------------------------------------------
# CSV header helper
# ---------------------------------------------------------------------------

def coeff_header(prefix: str, n_harmonics: int) -> List[str]:
    """
    Generate CSV header names for a given axis prefix (R/S/W)
    and harmonic order n.

    Example (n=2, prefix='R'):
        ['R0', 'R_cos1', 'R_sin1', 'R_cos2', 'R_sin2']
    """
    names = [f"{prefix}0"]
    for k in range(1, n_harmonics + 1):
        names.append(f"{prefix}_cos{k}")
        names.append(f"{prefix}_sin{k}")
    return names


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def main():
    results: Dict[int, KeplerEmpiricalParams] = {}

    for prn in PRNS:
        times, pos, vel = load_halo_state(prn)
        epoch = times[0]

        # Osculating Kepler elements at first epoch
        a, e, inc, raan, argp, M0 = rv_to_kepler(pos[0], vel[0], MU_MOON)
        mean_motion = np.sqrt(MU_MOON / a ** 3)

        # Residual accelerations in RSW
        res_acc_rsw = residual_accelerations(times, pos, vel)

        # Fit empirical accelerations with arbitrary harmonic order
        coeffs, rms = fit_empirical_components(
            times,
            res_acc_rsw,
            mean_motion,
            HARMONIC_ORDER,
        )

        params = KeplerEmpiricalParams(
            prn=prn,
            epoch=epoch,
            a=a,
            e=e,
            inc=inc,
            raan=raan,
            argp=argp,
            mean_anom=M0,
            mean_motion=mean_motion,
            harmonic_order=HARMONIC_ORDER,
            coeff_r=coeffs[0, :].tolist(),
            coeff_s=coeffs[1, :].tolist(),
            coeff_w=coeffs[2, :].tolist(),
            rms_residual=rms,
        )
        results[prn] = params

        print(
            f"PRN {prn}: a={a/1000:.1f} km e={e:.6f} "
            f"inc={np.degrees(inc):.3f}° "
            f"n={mean_motion:.6e} rad/s "
            f"rms residual={rms:.3e} m/s² "
            f"(harmonics={HARMONIC_ORDER})"
        )

    # ------------------- write CSV --------------------
    num_coeffs = 1 + 2 * HARMONIC_ORDER

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Build header dynamically
        header = [
            "prn",
            "epoch_s",
            "a_m",
            "e",
            "inc_rad",
            "raan_rad",
            "argp_rad",
            "mean_anom_rad",
            "mean_motion_rad_s",
            "harmonic_order",
        ]

        header += coeff_header("R", HARMONIC_ORDER)
        header += coeff_header("S", HARMONIC_ORDER)
        header += coeff_header("W", HARMONIC_ORDER)
        header.append("rms_residual_m_s2")

        writer.writerow(header)

        # Write per-PRN rows
        for prn in PRNS:
            p = results[prn]
            row = [
                p.prn,
                p.epoch,
                p.a,
                p.e,
                p.inc,
                p.raan,
                p.argp,
                p.mean_anom,
                p.mean_motion,
                p.harmonic_order,
            ]
            row += p.coeff_r
            row += p.coeff_s
            row += p.coeff_w
            row.append(p.rms_residual)
            writer.writerow(row)

    print(f"Saved parameters to {OUT_CSV}")


if __name__ == "__main__":
    main()
