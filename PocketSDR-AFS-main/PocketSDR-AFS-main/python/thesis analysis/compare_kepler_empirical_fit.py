#!/usr/bin/env python3
"""
Reconstruct HALO trajectories using Kepler + empirical parameters and compare
against truth tracks.

Input files:
  - halo_kepler_empirical.csv (output from fit_kepler_empirical_from_halo.py)
  - E:/Project/HALO/output/halo_prn1.csv ... halo_prn8.csv

The script integrates the satellite motion using gravitational acceleration
and empirical accelerations in RSW frame, then compares reconstructed
positions to truth and plots RMS error vs time.

Supports arbitrary harmonic order N (1,2,3,...) as encoded in the CSV:
    harmonic_order,
    R0, R_cos1, R_sin1, ..., R_cosN, R_sinN,
    S0, S_cos1, S_sin1, ..., S_cosN, S_sinN,
    W0, W_cos1, W_sin1, ..., W_cosN, W_sinN
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

MU_MOON = 4.9028e12  # m^3/s^2
HALO_DIR = Path(r"E:\Project\HALO\output")
PARAM_FILE = HALO_DIR / "halo_kepler_empirical.csv"
PRNS = range(2, 9)
OUTPUT_PNG = HALO_DIR / "kepler_empirical_fit_error.png"


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class KeplerEmpParams:
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
    coeff_r: List[float]  # length = 1 + 2*N
    coeff_s: List[float]
    coeff_w: List[float]


# ---------------------------------------------------------------------------
# Helpers to build coefficient names
# ---------------------------------------------------------------------------

def coeff_header(prefix: str, n_harmonics: int) -> List[str]:
    """
    Generate CSV column names for a given axis prefix (R/S/W)
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
# Load parameters from CSV (arbitrary harmonic order)
# ---------------------------------------------------------------------------

def load_params() -> Dict[int, KeplerEmpParams]:
    import csv

    params: Dict[int, KeplerEmpParams] = {}
    with PARAM_FILE.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prn = int(row["prn"])
            n_harm = int(row["harmonic_order"])

            # 动态构建 R/S/W 系数列表
            coeff_r_names = coeff_header("R", n_harm)
            coeff_s_names = coeff_header("S", n_harm)
            coeff_w_names = coeff_header("W", n_harm)

            coeff_r = [float(row[name]) for name in coeff_r_names]
            coeff_s = [float(row[name]) for name in coeff_s_names]
            coeff_w = [float(row[name]) for name in coeff_w_names]

            params[prn] = KeplerEmpParams(
                prn=prn,
                epoch=float(row["epoch_s"]),
                a=float(row["a_m"]),
                e=float(row["e"]),
                inc=float(row["inc_rad"]),
                raan=float(row["raan_rad"]),
                argp=float(row["argp_rad"]),
                mean_anom=float(row["mean_anom_rad"]),
                mean_motion=float(row["mean_motion_rad_s"]),
                harmonic_order=n_harm,
                coeff_r=coeff_r,
                coeff_s=coeff_s,
                coeff_w=coeff_w,
            )
    return params


# ---------------------------------------------------------------------------
# Load HALO truth
# ---------------------------------------------------------------------------

def load_halo_truth(prn: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = HALO_DIR / f"halo_prn{prn}.csv"
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    times = data[:, 1]
    pos = data[:, 2:5] * 1000.0
    vel = data[:, 5:8] * 1000.0
    return times, pos, vel


# ---------------------------------------------------------------------------
# Kepler propagation (osculating elements -> r,v)
# ---------------------------------------------------------------------------

def kepler_to_rv(params: KeplerEmpParams, t_abs: float) -> Tuple[np.ndarray, np.ndarray]:
    dt = t_abs - params.epoch
    M = params.mean_anom + params.mean_motion * dt
    M = np.mod(M, 2.0 * np.pi)
    E = solve_kepler_equation(M, params.e)
    nu = 2.0 * np.arctan2(
        np.sqrt(1 + params.e) * np.sin(E / 2.0),
        np.sqrt(1 - params.e) * np.cos(E / 2.0),
    )
    nu = np.mod(nu, 2.0 * np.pi)

    r_pqw = np.array([
        params.a * (np.cos(E) - params.e),
        params.a * np.sqrt(1 - params.e ** 2) * np.sin(E),
        0.0,
    ])
    v_pqw = np.array([
        -np.sqrt(MU_MOON / params.a) * np.sin(E) / (1 - params.e * np.cos(E)),
        np.sqrt(MU_MOON / params.a) * np.sqrt(1 - params.e ** 2) * np.cos(E) / (1 - params.e * np.cos(E)),
        0.0,
    ])

    # Rotation matrix from PQW to ECI
    cos_raan = np.cos(params.raan)
    sin_raan = np.sin(params.raan)
    cos_inc = np.cos(params.inc)
    sin_inc = np.sin(params.inc)
    cos_argp = np.cos(params.argp)
    sin_argp = np.sin(params.argp)

    R = np.array([
        [cos_raan * cos_argp - sin_raan * sin_argp * cos_inc,
         -cos_raan * sin_argp - sin_raan * cos_argp * cos_inc,
         sin_raan * sin_inc],
        [sin_raan * cos_argp + cos_raan * sin_argp * cos_inc,
         -sin_raan * sin_argp + cos_raan * cos_argp * cos_inc,
         -cos_raan * sin_inc],
        [sin_argp * sin_inc,
         cos_argp * sin_inc,
         cos_inc],
    ])

    r_eci = R @ r_pqw
    v_eci = R @ v_pqw
    return r_eci, v_eci


def solve_kepler_equation(M: float, e: float, tol: float = 1e-12, max_iter: int = 50) -> float:
    E = M if e < 0.8 else np.pi
    for _ in range(max_iter):
        f = E - e * np.sin(E) - M
        fp = 1.0 - e * np.cos(E)
        dE = -f / fp
        E += dE
        if abs(dE) < tol:
            break
    return E


# ---------------------------------------------------------------------------
# Empirical acceleration with arbitrary harmonic order
# ---------------------------------------------------------------------------

def eval_empirical_series(coeffs: List[float],
                          mean_motion: float,
                          dt: float) -> float:
    """
    Evaluate scalar series:
        a(t) = c0 + sum_{k=1..N} [c_cos_k cos(k n dt) + c_sin_k sin(k n dt)]
    where coeffs = [c0, c_cos1, c_sin1, c_cos2, c_sin2, ..., c_cosN, c_sinN]
    """
    c0 = coeffs[0]
    val = c0
    N = (len(coeffs) - 1) // 2
    for k in range(1, N + 1):
        c_cos = coeffs[2 * k - 1]
        c_sin = coeffs[2 * k]
        w = k * mean_motion
        val += c_cos * np.cos(w * dt) + c_sin * np.sin(w * dt)
    return val


def empirical_acceleration(params: KeplerEmpParams,
                           state_r: np.ndarray,
                           state_v: np.ndarray,
                           t_abs: float) -> np.ndarray:
    """
    Compute empirical acceleration vector in inertial frame:

    1. Compute a_R, a_S, a_W from harmonic series in RSW.
    2. Build RSW triad from (r,v).
    3. Transform back to inertial: a = a_R R_hat + a_S S_hat + a_W W_hat.
    """
    dt = t_abs - params.epoch

    acc_r = eval_empirical_series(params.coeff_r, params.mean_motion, dt)
    acc_s = eval_empirical_series(params.coeff_s, params.mean_motion, dt)
    acc_w = eval_empirical_series(params.coeff_w, params.mean_motion, dt)

    r_hat = state_r / np.linalg.norm(state_r)
    h_vec = np.cross(state_r, state_v)
    h_hat = h_vec / np.linalg.norm(h_vec)
    s_hat = np.cross(h_hat, r_hat)

    return acc_r * r_hat + acc_s * s_hat + acc_w * h_hat


# ---------------------------------------------------------------------------
# Dynamics and integration
# ---------------------------------------------------------------------------

def rhs_acceleration(params: KeplerEmpParams,
                     r: np.ndarray,
                     v: np.ndarray,
                     t_abs: float) -> np.ndarray:
    grav = -MU_MOON * r / (np.linalg.norm(r) ** 3)
    emp = empirical_acceleration(params, r, v, t_abs)
    return grav + emp


def rk4_step(params: KeplerEmpParams,
             r: np.ndarray,
             v: np.ndarray,
             t: float,
             dt: float) -> Tuple[np.ndarray, np.ndarray]:
    def acc(r_, v_, t_):
        return rhs_acceleration(params, r_, v_, t_)

    k1_r = v
    k1_v = acc(r, v, t)

    k2_r = v + 0.5 * dt * k1_v
    k2_v = acc(r + 0.5 * dt * k1_r, v + 0.5 * dt * k1_v, t + 0.5 * dt)

    k3_r = v + 0.5 * dt * k2_v
    k3_v = acc(r + 0.5 * dt * k2_r, v + 0.5 * dt * k2_v, t + 0.5 * dt)

    k4_r = v + dt * k3_v
    k4_v = acc(r + dt * k3_r, v + dt * k3_v, t + dt)

    r_next = r + (dt / 6.0) * (k1_r + 2 * k2_r + 2 * k3_r + k4_r)
    v_next = v + (dt / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
    return r_next, v_next


def propagate(params: KeplerEmpParams,
              times: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    r0, v0 = kepler_to_rv(params, times[0])
    positions = np.zeros((len(times), 3))
    velocities = np.zeros((len(times), 3))
    positions[0] = r0
    velocities[0] = v0

    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        r_prev = positions[i - 1]
        v_prev = velocities[i - 1]
        r_next, v_next = rk4_step(params, r_prev, v_prev, times[i - 1], dt)
        positions[i] = r_next
        velocities[i] = v_next

    return positions, velocities


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    params_dict = load_params()
    rms_errors_per_prn: Dict[int, np.ndarray] = {}
    time_grid = None

    for prn in PRNS:
        times, pos_truth, _ = load_halo_truth(prn)
        params = params_dict[prn]

        if time_grid is None:
            time_grid = times
        else:
            if not np.allclose(times, time_grid):
                raise ValueError("Time grid mismatch between PRNs")

        pos_fit, _ = propagate(params, times)
        diff = pos_fit - pos_truth
        err = np.linalg.norm(diff, axis=1)
        rms_errors_per_prn[prn] = err

        print(
            f"PRN {prn}: mean error {np.mean(err):.3f} m, "
            f"max {np.max(err):.3f} m, "
            f"harmonics={params.harmonic_order}"
        )

    assert time_grid is not None
    # RMS across PRNs at each epoch
    rms_total = np.sqrt(np.mean([errs ** 2 for errs in rms_errors_per_prn.values()], axis=0))

    plt.figure(figsize=(10, 5))
    for prn, errs in rms_errors_per_prn.items():
        plt.plot(time_grid - time_grid[0], errs, alpha=0.4, label=f"PRN {prn}")
    plt.plot(time_grid - time_grid[0], rms_total, linewidth=2, label="RMS (all PRNs)")
    plt.xlabel("Time since start (s)")
    plt.ylabel("Position error magnitude (m)")
    plt.title("Kepler + Empirical fit vs HALO truth (arbitrary harmonic order)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=150)
    print(f"Saved plot to {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
