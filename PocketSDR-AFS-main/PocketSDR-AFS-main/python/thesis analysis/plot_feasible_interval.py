#!/usr/bin/env python3
"""
Estimate the maximum span (in orbit periods) over which different broadcast
orbit models meet a position-error tolerance as a function of true anomaly.

Models:
  • Chebyshev-like Cartesian polynomials of degree n (fitted independently to x/y/z).
  • Kepler + empirical accelerations expressed in the RSW frame with Fourier
    harmonics up to order n_kep (the "Kepler + RSW empirical accel" model used by afs_sim).

Inputs:
  - Truth orbit sampled at 1 Hz: E:/Project/HALO/halo_prn.csv

Outputs:
  - HALO/max_feasible_interval.png : plot with two panels comparable to the reference figure.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MU_MOON = 4.9028e12  # m^3/s^2
HALO_TRUTH = Path(r"E:\Project\HALO\halo_prn.csv")
OUTPUT_FIG = Path("HALO") / "max_feasible_interval.png"

POLY_POS_TOL = 13.34  # metres
KEPLER_POS_TOL = 13.34  # metres
SEARCH_STEP = 60.0  # seconds
ANOMALY_BINS = np.arange(0.0, 360.0, 10.0)


def wrap_angle_deg(angle: np.ndarray) -> np.ndarray:
    return np.mod(angle, 360.0)


def angle_diff_deg(a: np.ndarray, b: float) -> np.ndarray:
    return (a - b + 180.0) % 360.0 - 180.0


def load_truth(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(
        path,
        comment="#",
        names=["prn", "t_sec", "x_km", "y_km", "z_km", "vx_kms", "vy_kms", "vz_kms"],
    )
    times = df["t_sec"].to_numpy(dtype=float)
    pos = df[["x_km", "y_km", "z_km"]].to_numpy(dtype=float) * 1000.0
    vel = df[["vx_kms", "vy_kms", "vz_kms"]].to_numpy(dtype=float) * 1000.0
    return times, pos, vel


def semimajor_axis(r: np.ndarray, v: np.ndarray) -> float:
    r_norm = np.linalg.norm(r)
    energy = 0.5 * np.dot(v, v) - MU_MOON / r_norm
    return -MU_MOON / (2.0 * energy)


def true_anomaly(r: np.ndarray, v: np.ndarray) -> float:
    h_vec = np.cross(r, v)
    n_vec = np.cross(np.array([0.0, 0.0, 1.0]), h_vec)
    r_norm = np.linalg.norm(r)
    e_vec = (np.cross(v, h_vec) / MU_MOON) - (r / r_norm)
    e = np.linalg.norm(e_vec)
    n_norm = np.linalg.norm(n_vec)
    if e < 1e-12:
        if n_norm > 1e-12:
            nu = math.acos(np.clip(np.dot(n_vec, r) / (n_norm * r_norm), -1.0, 1.0))
            if r[2] < 0:
                nu = 2.0 * math.pi - nu
        else:
            nu = 0.0
    else:
        cos_nu = np.dot(e_vec, r) / (e * r_norm)
        cos_nu = np.clip(cos_nu, -1.0, 1.0)
        nu = math.acos(cos_nu)
        if np.dot(r, v) < 0:
            nu = 2.0 * math.pi - nu
    return nu


def compute_true_anomalies(pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
    return np.degrees(np.array([true_anomaly(r, v) for r, v in zip(pos, vel)]))


def basis_matrix(dt: np.ndarray, mean_motion: float, order: int) -> np.ndarray:
    cols = 1 + 2 * order
    B = np.empty((dt.size, cols))
    B[:, 0] = 1.0
    col = 1
    for k in range(1, order + 1):
        arg = mean_motion * k * dt
        B[:, col] = np.cos(arg)
        B[:, col + 1] = np.sin(arg)
        col += 2
    return B


def rsw_components(pos: np.ndarray, vel: np.ndarray, acc: np.ndarray) -> np.ndarray:
    r_norm = np.linalg.norm(pos, axis=1)
    r_hat = pos / r_norm[:, None]
    h_vec = np.cross(pos, vel)
    h_norm = np.linalg.norm(h_vec, axis=1)
    w_hat = h_vec / h_norm[:, None]
    s_hat = np.cross(w_hat, r_hat)
    a_r = np.einsum("ij,ij->i", acc, r_hat)
    a_s = np.einsum("ij,ij->i", acc, s_hat)
    a_w = np.einsum("ij,ij->i", acc, w_hat)
    return np.column_stack((a_r, a_s, a_w))


def empirical_acceleration(r_vec: np.ndarray, v_vec: np.ndarray, coeffs: Dict[str, np.ndarray],
                           mean_motion: float, tau: float, order: int) -> np.ndarray:
    basis = np.empty(1 + 2 * order)
    basis[0] = 1.0
    idx = 1
    for k in range(1, order + 1):
        arg = mean_motion * k * tau
        basis[idx] = math.cos(arg)
        basis[idx + 1] = math.sin(arg)
        idx += 2
    r_hat = r_vec / np.linalg.norm(r_vec)
    h_vec = np.cross(r_vec, v_vec)
    w_hat = h_vec / np.linalg.norm(h_vec)
    s_hat = np.cross(w_hat, r_hat)
    acc_r = float(np.dot(coeffs["R"], basis))
    acc_s = float(np.dot(coeffs["S"], basis))
    acc_w = float(np.dot(coeffs["W"], basis))
    return acc_r * r_hat + acc_s * s_hat + acc_w * w_hat


def rk4_step(r: np.ndarray, v: np.ndarray, tau: float, dt: float, coeffs: Dict[str, np.ndarray],
             mean_motion: float, order: int) -> Tuple[np.ndarray, np.ndarray]:
    def acceleration(r_state: np.ndarray, v_state: np.ndarray, tau_rel: float) -> np.ndarray:
        grav = -MU_MOON * r_state / np.linalg.norm(r_state) ** 3
        emp = empirical_acceleration(r_state, v_state, coeffs, mean_motion, tau_rel, order)
        return grav + emp

    k1_r = v
    k1_v = acceleration(r, v, tau)
    k2_r = v + 0.5 * dt * k1_v
    k2_v = acceleration(r + 0.5 * dt * k1_r, v + 0.5 * dt * k1_v, tau + 0.5 * dt)
    k3_r = v + 0.5 * dt * k2_v
    k3_v = acceleration(r + 0.5 * dt * k2_r, v + 0.5 * dt * k2_v, tau + 0.5 * dt)
    k4_r = v + dt * k3_v
    k4_v = acceleration(r + dt * k3_r, v + dt * k3_v, tau + dt)
    r_next = r + (dt / 6.0) * (k1_r + 2.0 * k2_r + 2.0 * k3_r + k4_r)
    v_next = v + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
    return r_next, v_next


def evaluate_kepler_window(times: np.ndarray, pos: np.ndarray, vel: np.ndarray,
                           start_idx: int, end_idx: int, order: int) -> float:
    t_win = times[start_idx:end_idx + 1]
    pos_win = pos[start_idx:end_idx + 1]
    vel_win = vel[start_idx:end_idx + 1]
    if len(t_win) < 1 + 2 * order + 2:
        return float("inf")
    dt_rel = t_win - t_win[0]
    a0 = semimajor_axis(pos_win[0], vel_win[0])
    mean_motion = math.sqrt(MU_MOON / a0 ** 3)
    basis = basis_matrix(dt_rel, mean_motion, order)
    acc_obs = np.gradient(vel_win, t_win, axis=0)
    r_norm = np.linalg.norm(pos_win, axis=1)
    grav = -MU_MOON * pos_win / (r_norm[:, None] ** 3)
    residual = acc_obs - grav
    residual_rsw = rsw_components(pos_win, vel_win, residual)
    coeffs = {
        "R": np.linalg.lstsq(basis, residual_rsw[:, 0], rcond=None)[0],
        "S": np.linalg.lstsq(basis, residual_rsw[:, 1], rcond=None)[0],
        "W": np.linalg.lstsq(basis, residual_rsw[:, 2], rcond=None)[0],
    }
    r_pred = np.zeros_like(pos_win)
    v_pred = np.zeros_like(vel_win)
    r_pred[0] = pos_win[0]
    v_pred[0] = vel_win[0]
    for i in range(1, len(t_win)):
        dt = t_win[i] - t_win[i - 1]
        tau_prev = dt_rel[i - 1]
        r_next, v_next = rk4_step(r_pred[i - 1], v_pred[i - 1], tau_prev, dt,
                                  coeffs, mean_motion, order)
        r_pred[i] = r_next
        v_pred[i] = v_next
    err = np.linalg.norm(r_pred - pos_win, axis=1)
    return float(np.max(err))


def evaluate_poly_window(times: np.ndarray, pos: np.ndarray,
                         start_idx: int, end_idx: int, order: int) -> float:
    t_win = times[start_idx:end_idx + 1]
    pos_win = pos[start_idx:end_idx + 1]
    if len(t_win) < order + 2:
        return float("inf")
    t_rel = t_win - t_win[len(t_win) // 2]
    fitted = np.zeros_like(pos_win)
    for axis in range(3):
        coeffs = np.polyfit(t_rel, pos_win[:, axis], order)
        fitted[:, axis] = np.polyval(coeffs, t_rel)
    err = np.linalg.norm(fitted - pos_win, axis=1)
    return float(np.max(err))


def forward_search_duration(times: np.ndarray, pos: np.ndarray, vel: np.ndarray,
                            start_idx: int, order: int, tol: float, evaluator) -> float:
    """
    从 start_idx 对应时刻作为“初始化点”，仅向未来扩张窗口，找到
    满足 max position error <= tol 的最大持续时间（秒）。
    evaluator(times, pos, vel, start_idx, end_idx, order) 返回该窗口的最大误差。
    """
    # 最大可扩张到序列末尾
    if start_idx >= len(times) - 2:
        return 0.0

    # 以 SEARCH_STEP 为步长的候选持续时间
    max_forward = times[-1] - times[start_idx]
    if max_forward <= 0:
        return 0.0
    durations = np.arange(SEARCH_STEP, max_forward + SEARCH_STEP, SEARCH_STEP)

    lo, hi = 0, len(durations) - 1
    best = 0.0
    while lo <= hi:
        mid = (lo + hi) // 2
        end_time = times[start_idx] + durations[mid]
        end_idx = np.searchsorted(times, end_time, side="right") - 1
        if end_idx <= start_idx:
            break

        err = evaluator(times, pos, vel, start_idx, end_idx, order)
        if err <= tol:
            best = durations[mid]
            lo = mid + 1
        else:
            hi = mid - 1
    return best



def nearest_index(anomalies_deg: np.ndarray, target: float) -> int:
    diff = np.abs(angle_diff_deg(anomalies_deg, target))
    return int(np.argmin(diff))


def compute_curves(times: np.ndarray, pos: np.ndarray, vel: np.ndarray,
                   anomalies_deg: np.ndarray, period: float) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    poly_orders = [6, 14]
    kep_orders = [1, 3]
    poly_res = {order: [] for order in poly_orders}
    kep_res = {order: [] for order in kep_orders}
    for theta in ANOMALY_BINS:
        idx = nearest_index(anomalies_deg, theta)
        for order in poly_orders:
            duration = forward_search_duration(
                times, pos, vel, idx, order, POLY_POS_TOL,
                evaluator=lambda t, p, v, s, e, o=order: evaluate_poly_window(t, p, s, e, o),
            )
            poly_res[order].append(duration / period)
        for order in kep_orders:
            duration = forward_search_duration(
                times, pos, vel, idx, order, KEPLER_POS_TOL,
                evaluator=lambda t, p, v, s, e, o=order: evaluate_kepler_window(t, p, v, s, e, o),
            )
            kep_res[order].append(duration / period)
    for order in poly_orders:
        poly_res[order] = np.array(poly_res[order])
    for order in kep_orders:
        kep_res[order] = np.array(kep_res[order])
    return poly_res, kep_res


def plot_results(poly: Dict[int, np.ndarray], kep: Dict[int, np.ndarray]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    width = 4.0
    axes[0].bar(ANOMALY_BINS - width / 2.0, poly[6], width=width, color="#c44e52", label="n = 6")
    axes[0].bar(ANOMALY_BINS + width / 2.0, poly[14], width=width, color="#4c72b0", label="n = 14")
    axes[0].set_title("Polynomial Model")
    axes[0].set_xlabel("True Anomaly (°)")
    axes[0].set_ylabel("Max. Feasible Interval (orbit periods)")
    axes[0].set_xticks(np.arange(0, 361, 50))
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend()

    kep_orders = sorted(kep.keys())
    kep_colors = ["#55a868", "#dd8452", "#4c72b0", "#c44e52"]
    if len(kep_orders) == 1:
        axes[1].bar(ANOMALY_BINS, kep[kep_orders[0]], width=width,
                    color=kep_colors[0],
                    label=f"{kep_orders[0]}{'st' if kep_orders[0]==1 else 'th'} harmonic")
    else:
        bar_width = width / max(1, len(kep_orders))
        spacing = bar_width * 0.6
        offsets = (np.arange(len(kep_orders)) - (len(kep_orders) - 1) / 2.0) * (bar_width + spacing)
        for idx, order in enumerate(kep_orders):
            axes[1].bar(ANOMALY_BINS + offsets[idx], kep[order], width=bar_width,
                        color=kep_colors[idx % len(kep_colors)],
                        label=f"{order}{'st' if order==1 else 'nd' if order==2 else 'rd' if order==3 else 'th'} harmonic")
    axes[1].set_title("Keplerian Model")
    axes[1].set_xlabel("True Anomaly (°)")
    axes[1].set_xticks(np.arange(0, 361, 50))
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend()

    fig.tight_layout()
    OUTPUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FIG, dpi=300)
    plt.close(fig)
    print(f"Saved {OUTPUT_FIG}")


def main() -> None:
    times, pos, vel = load_truth(HALO_TRUTH)
    anomalies_deg = wrap_angle_deg(compute_true_anomalies(pos, vel))
    a0 = semimajor_axis(pos[0], vel[0])
    period = 2.0 * math.pi * math.sqrt(a0 ** 3 / MU_MOON)
    poly_curves, kep_curves = compute_curves(times, pos, vel, anomalies_deg, period)
    plot_results(poly_curves, kep_curves)


if __name__ == "__main__":
    main()
