#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Lunar AFS UERE with visibility mask (with clock random-walk sample):
- Link budget (C/N0)
- Receiver DLL tracking noise (piecewise jitter formula)
- Multipath model
- Clock covariance + sampled clock state with hourly reset  <-- NEW: adds drift in actual error
- Robust Chebyshev orbit-fit residuals (from halo_prn truth) per-hour
- Visibility mask: remove epochs below elevation mask (set NaN)
- UERE sandpile (variance stacking; zeroed when invisible)
- CN0 and Elevation panel
- CSV + PNG outputs
- EXTRA PLOTS: Chebyshev vs Truth residuals (XYZ, ||Δr||, LOS) + per-segment LOS RMS
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from numpy.polynomial.chebyshev import chebfit, chebval

# --------------------------- CONFIG ---------------------------------

CSV_IN   = r"E:/Project/HALO/halo_prn.csv"
PRN_USE  = 2

# Link budget
f_MHz     = 2492.028
EIRP_dBW  = 10
G_rx_dBi  = 4.0
L_misc_dB = 2.0
T_sys_K   = 150.0
k_dB      = -228.6

# AFS code rate (pilot) -> chip duration
Rc   = 1.023e6           # [chips/s]
Tc   = 1.0 / Rc          # [s/chip]

# Coherent integration time
Tcoh = 0.01              # [s]

# DLL parameters
Bdll  = 0.25             # [Hz]
Bbeta = 8e6              # [Hz]
d     = 0.25             # [chips]
eps   = 0.0

# Multipath sigma model: σ_MPTH = a + b * exp(c * elev_deg)
a_mp, b_mp, c_mp = 0.1633, 1.1846, -0.0511

# --- Clock model params ---
c_light    = 299792458.0
sigma_b0_m = 0
sigma_f0_s = 0
sigma_a0_s = 0

# Process noise scales (set explicitly here)
sigma_wfm  = 5e-18
sigma_rwfm = 1e-20
sigma_wb   = 1e-20

# Chebyshev fit (per-hour)
CHEB_SEG_LEN_S = 3600
CHEB_DEG       = 6

# Visibility mask (degrees)
EL_MASK_DEG = 5.0
ELEV_MIN, ELEV_MAX = 0.0, 90.0

# Outputs
CSV_OUT  = "E:/Project/thesis c/uere_demo_prn02_visible_only_dll025.csv"
FIG1_OUT = "E:/Project/thesis c/uere_uere_cn0_elev_visible_dll025.png"
FIG2_OUT = "E:/Project/thesis c/uere_sandpile_visible_dll025.png"
FIG3_OUT = "E:/Project/thesis c/cheb_truth_residuals_timeseries.png"
FIG4_OUT = "E:/Project/thesis c/cheb_truth_residuals_segment_rms.png"

# ------------------------ HELPER FUNCTIONS --------------------------

def cheb_fit_eval(t_seg: np.ndarray, y_seg: np.ndarray, deg_target: int) -> np.ndarray:
    """
    Simple Chebyshev fit per segment (no robust fallbacks).
    Fits y(t) with degree=deg_target on the segment [t_min, t_max] and evaluates at t_seg.
    """
    t_seg = np.asarray(t_seg, dtype=float)
    y_seg = np.asarray(y_seg, dtype=float)
    t_min = float(np.min(t_seg))
    t_max = float(np.max(t_seg))
    # map to [-1, 1]
    xi = 2.0 * (t_seg - t_min) / max(t_max - t_min, np.finfo(float).eps) - 1.0
    coeff = chebfit(xi, y_seg, deg_target)
    return chebval(xi, coeff)

def dll_sigma_m(CN0_lin, Tcoh, Bdll, Bbeta, Tc, d, eps=0.0):
    """
    DLL code-tracking jitter (piecewise), returned in METERS (m).
    CN0_lin must be linear (not dB-Hz).
    """
    CN0 = np.maximum(CN0_lin, 1e-30)
    Tc   = np.maximum(Tc, 1e-9)
    Bdl = np.maximum(Bdll, 1e-9)
    Bbt = np.maximum(Bbeta, 1e-9)

    d_low  = 1.0/(Tc*Bbt)
    d_high = np.pi/(Tc*Bbt)

    case1 = (Bdl/(2*CN0*Tc))*d*(1.0 + 2.0/(Tcoh*CN0*(2.0 - eps)))                       # d >= pi/(T Bbeta)
    case2 = (Bdl/(2*CN0*Tc*Bbt))*(1.0 + 1.0/(Tcoh*CN0))                                 # d <= 1/(T Bbeta)
    case3 = (Bdl/(2*CN0))*( (1.0/(Tc*Bbt) + (Bbt*Tc)/(np.pi-1.0)*(d - 1.0/(Tc*Bbt))**2) ) \
            *(1.0 + 2.0/(Tcoh*CN0*(2.0 - eps)))                                         # middle

    var_chip = np.where(d >= d_high, case1, np.where(d <= d_low, case2, case3))

    # Convert from chips to meters: 1 chip -> c*Tc meters.
    K = c_light * Tc
    sigma_m = K * np.sqrt(var_chip)
    return sigma_m

# --------------------------- LOAD DATA -------------------------------

df = pd.read_csv(CSV_IN)
df = df[df["# prn"] == PRN_USE].reset_index(drop=True)

t = df["t_sec"].values.astype(float)
dt = np.diff(t, prepend=t[0])
if dt.size > 1: dt[0] = dt[1]
elif dt.size == 1: dt[0] = 1.0

# ------------------------- GEOMETRY ----------------------------------

R_moon_km = 1737.4
r_user = np.array([0.0, 0.0, -R_moon_km])  # south pole user

r_sat_km = df[["x_km","y_km","z_km"]].values
los_vec_km = r_sat_km - r_user
rng_km = np.linalg.norm(los_vec_km, axis=1)
los_hat = (los_vec_km.T / rng_km).T

# elevation ≈ arcsin(los·up)
up_hat = r_user / np.linalg.norm(r_user)
cos_z = np.clip(np.sum(los_hat * up_hat, axis=1), -1.0, 1.0)
elev_rad = np.arcsin(cos_z)
elev_deg = np.degrees(elev_rad)

# ------------------------- VISIBILITY MASK ---------------------------

vis = elev_deg >= EL_MASK_DEG

# ------------------------- LINK BUDGET -------------------------------

FSPL_dB = 32.44 + 20*np.log10(rng_km) + 20*np.log10(f_MHz)
C_dBW   = EIRP_dBW + G_rx_dBi - FSPL_dB - L_misc_dB
print(C_dBW)
N0_dBW_Hz = k_dB + 10*np.log10(T_sys_K)
CN0_dBHz = C_dBW - N0_dBW_Hz
CN0_lin  = 10**(CN0_dBHz/10.0)

# ------------------ RECEIVER DLL TRACKING NOISE ----------------------

sigma_rec_m_full = dll_sigma_m(CN0_lin, Tcoh, Bdll, Bbeta, Tc, d, eps)
sigma_rec_m = np.where(vis, sigma_rec_m_full, np.nan)

# --------------------------- MULTIPATH -------------------------------

elev_deg_clip = np.clip(elev_deg, ELEV_MIN, ELEV_MAX)
sigma_mp_m_full = a_mp + b_mp * np.exp(c_mp * elev_deg_clip)
sigma_mp_m = np.where(vis, sigma_mp_m_full, np.nan)

# ------------------------- CLOCK MODEL -------------------------------

sigma_b0_s = sigma_b0_m / c_light
P0 = np.diag([sigma_b0_s**2, sigma_f0_s**2, sigma_a0_s**2])

# Example intensities (you can replace by sigma_wfm^2, sigma_rwfm^2 if desired)
# h0 = 9e-24     # WFM
# h_2 = 2.7e-32  # RWFM

h0=1.6e-21
h_2=7.5e-28


P = P0.copy()
x = [0,0,0]  # [b (s), b_dot (s/s), a (s/s^2)]
e_clk_m_full = np.zeros_like(t, float)
sigma_clk_m_full = np.zeros_like(t, float)

t_start = t[0]
rng = np.random.default_rng(2025)

for k in range(len(t)):
    if k > 0:
        dtk = dt[k]

        # hourly reset
        if abs((t[k] - t_start) % 3600.0) < 0.5:
            P = P0.copy()
            x[:] = [0,0,0]

        F = np.array([[1.0, dtk, 0.5*dtk*dtk],
                      [0.0, 1.0, dtk],
                      [0.0, 0.0, 1.0]])

        tau = dtk
        Sk = np.array([
            [h0 * tau + h_2 * (tau ** 3) / 3.0,  h_2 * (tau ** 2) / 2.0, 0.0],
            [h_2 * (tau ** 2) / 2.0,             h_2 * tau,            0.0],
            [0.0,                                0.0,                  0.0],
        ])

        P = F @ P @ F.T + Sk
        w = rng.multivariate_normal(np.zeros(3), Sk)
        x = F @ x + w
        print(x)

    e_clk_m_full[k]    = c_light * (x[0]-sigma_b0_s)
    sigma_clk_m_full[k]= c_light * np.sqrt(max(P[0,0], 0.0))

e_clk_m     = np.where(vis, e_clk_m_full,    np.nan)
sigma_clk_m = np.where(vis, sigma_clk_m_full, np.nan)

# ------------- CHEBYSHEV ORBIT-FIT RESIDUALS (VISIBLE-ONLY) ---------

seg_idx = np.floor((t - t_start)/CHEB_SEG_LEN_S).astype(int)

# 我们同时输出：每个时刻的 XYZ 分量残差（m）、LOS 残差（m）、以及每段的 LOS RMS
eph_resid_los_m = np.full_like(t, np.nan, dtype=float)  # LOS 残差（m）
sigma_ephem_m   = np.full_like(t, np.nan, dtype=float)  # 段内 LOS RMS 复制到该段各点
res_xyz_m       = np.full((len(t), 3), np.nan, dtype=float)  # 分量残差（m）
seg_mid_hr      = []   # 段中心时刻（小时）
seg_rms_m       = []   # 段 LOS 残差 RMS（m）

unique_segs = np.unique(seg_idx)
for s in unique_segs:
    mask_seg = (seg_idx == s)
    if not np.any(mask_seg):
        continue

    t_seg = t[mask_seg]
    r_seg_km = r_sat_km[mask_seg, :]

    # 简单 Chebyshev 拟合
    x_hat = cheb_fit_eval(t_seg, r_seg_km[:,0], CHEB_DEG)
    y_hat = cheb_fit_eval(t_seg, r_seg_km[:,1], CHEB_DEG)
    z_hat = cheb_fit_eval(t_seg, r_seg_km[:,2], CHEB_DEG)
    r_hat_km = np.column_stack([x_hat, y_hat, z_hat])

    # 分量残差（km->m）
    dR_km = r_seg_km - r_hat_km
    dR_m  = dR_km * 1000.0
    res_xyz_m[mask_seg, :] = dR_m

    # LOS 残差（km->m）
    los_seg = los_hat[mask_seg, :]
    e_los_km = np.sum(los_seg * dR_km, axis=1)
    e_los_m  = e_los_km * 1000.0
    eph_resid_los_m[mask_seg] = e_los_m

    # 段内 RMS（用于可视化）
    seg_rms = np.sqrt(np.mean(e_los_m**2)) if e_los_m.size > 0 else np.nan
    sigma_ephem_m[mask_seg] = seg_rms

    seg_mid = ((t_seg.min() + t_seg.max())/2.0 - t_start)/3600.0
    seg_mid_hr.append(seg_mid)
    seg_rms_m.append(seg_rms)

# --------------------- UERE + SANDPILE COMPONENTS --------------------

sig_clk = sigma_clk_m
sig_rec = sigma_rec_m
sig_mp  = sigma_mp_m
sig_eph = sigma_ephem_m

var_clk = np.where(vis, sig_clk**2, 0.0)
var_rec = np.where(vis, sig_rec**2, 0.0)
var_mp  = np.where(vis, sig_mp**2,  0.0)
var_eph = np.where(vis, sig_eph**2, 0.0)

var_total     = var_clk + var_rec + var_mp + var_eph
sigma_uere_m  = np.where(vis, np.sqrt(var_total), np.nan)

# measurement-like sample (includes clock sample)
rng2 = np.random.default_rng(10)
e_rec = rng2.normal(0.0, np.nan_to_num(sig_rec, 0.0))
e_mp  = rng2.normal(0.0, np.nan_to_num(sig_mp,  0.0))
e_eph = np.nan_to_num(eph_resid_los_m, 0.0)
e_total_m = np.where(vis, e_clk_m , np.nan)

t_hr = (t - t_start)/3600.0
CN0_plot = np.where(vis, CN0_dBHz, np.nan)
elev_plot = np.where(vis, elev_deg, np.nan)

# --------------------------- PLOTS -----------------------------------

# 图1：UERE ±3σ 带 + CN0/仰角
fig = plt.figure(figsize=(11,8))
gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1.2], hspace=0.25)

ax_top = fig.add_subplot(gs[0, 0])
ax_top.fill_between(t_hr, -3*np.nan_to_num(sigma_uere_m), 3*np.nan_to_num(sigma_uere_m),
                    where=np.isfinite(sigma_uere_m), alpha=0.25, label=r'$\pm 3\sigma_{\mathrm{UERE}}$')
ax_top.plot(t_hr, e_total_m, lw=0.8, label='Actual clock error (sim)')
ax_top.set_xlabel('Time (hours)')
ax_top.set_ylabel('Pseudorange error (m)')
ax_top.set_title(f'24h UERE — hourly reset, Chebyshev fit residuals (visible only), Bdll={Bdll} Hz')
ax_top.grid(True, alpha=0.3)
ax_top.legend(loc='upper left')

ax_c = fig.add_subplot(gs[1, 0], sharex=ax_top)
ln1 = ax_c.plot(t_hr, CN0_plot, linewidth=1.0, label='C/N0 (dB-Hz)')
ax_c.set_xlabel('Time (hours)')
ax_c.set_ylabel('C/N0 (dB-Hz)')
ax_c.grid(True, alpha=0.3)

ax_e = ax_c.twinx()
ln2 = ax_e.plot(t_hr, elev_plot, linestyle='--', linewidth=1.0, label='Elevation (deg)')
ax_e.set_ylabel('Elevation (deg)')
ax_e.set_ylim(ELEV_MIN, ELEV_MAX)

lines = ln1 + ln2
labels = [l.get_label() for l in lines]
ax_c.legend(lines, labels, loc='upper left')

fig.tight_layout()
fig.savefig(FIG1_OUT, dpi=160)

# 图2：Sandpile
fig2, ax1 = plt.subplots(figsize=(11,5))
layers = [var_clk, var_rec, var_mp, var_eph]
labels = ['Clock variance', 'Receiver DLL variance', 'Multipath variance', 'Ephemeris (Cheb) variance']
ax1.stackplot(t_hr, layers, labels=labels, alpha=0.85)
ax1.set_xlabel('Time (hours)')
ax1.set_ylabel('UERE variance (m$^2$)')
ax1.set_title('UERE “Sandpile”')

def var_to_3sigma(x): return 3.0*np.sqrt(x)
def sigma3_to_var(y): return (y/3.0)**2
ax2 = ax1.secondary_yaxis('right', functions=(var_to_3sigma, sigma3_to_var))
ax2.set_ylabel('Equivalent ±3σ (m)')

ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper left')
fig2.tight_layout()
fig2.savefig(FIG2_OUT, dpi=160)

# 图3：Chebyshev vs Truth 残差（X/Y/Z、||Δr||、LOS）
res_norm_m = np.linalg.norm(res_xyz_m, axis=1)
figr, axs = plt.subplots(4, 1, figsize=(12, 9), sharex=True)

axs[0].plot(t_hr, res_xyz_m[:,0], linewidth=0.9)
axs[0].set_ylabel('ΔX (m)')
axs[0].grid(True, alpha=0.3)

axs[1].plot(t_hr, res_xyz_m[:,1], linewidth=0.9)
axs[1].set_ylabel('ΔY (m)')
axs[1].grid(True, alpha=0.3)

axs[2].plot(t_hr, res_xyz_m[:,2], linewidth=0.9)
axs[2].set_ylabel('ΔZ (m)')
axs[2].grid(True, alpha=0.3)

axs[3].plot(t_hr, res_norm_m, linewidth=1.0, label='||Δr|| (m)')
axs[3].plot(t_hr, eph_resid_los_m,  linewidth=1.0, linestyle='--', label='LOS residual (m)')
axs[3].set_xlabel('Time (hours)')
axs[3].set_ylabel('m')
axs[3].grid(True, alpha=0.3)
axs[3].legend(loc='upper right')

figr.suptitle('Chebyshev Fit Residuals vs Truth (visible only)')
figr.tight_layout()
figr.savefig(FIG3_OUT, dpi=160)

# 图4：每小时段 LOS 残差 RMS
figr2, axr2 = plt.subplots(figsize=(12, 4))
axr2.stem(seg_mid_hr, seg_rms_m, basefmt=" ")
axr2.set_xlabel('Time (hours, segment mid)')
axr2.set_ylabel('LOS RMS (m)')
axr2.set_title('Per-Segment (Hourly) LOS Residual RMS')
axr2.grid(True, alpha=0.3)
figr2.tight_layout()
figr2.savefig(FIG4_OUT, dpi=160)

# ====================== SUMMARY PARAMETERS ===========================

def dll_sigma_from_cn0_dbhz(cn0_dbhz, Tcoh, Bdll, Bbeta, Tc, d, eps=0.0):
    cn0_lin = 10**(cn0_dbhz/10.0)
    return float(dll_sigma_m(np.array([cn0_lin]), Tcoh, Bdll, Bbeta, Tc, d, eps)[0])

sigma_rx_curr_m      = float(np.nanmedian(sigma_rec_m))
sigma_rx_curr_m_p68  = float(np.nanpercentile(sigma_rec_m, 68))
sigma_rx_ref_m_37db  = dll_sigma_from_cn0_dbhz(37.0, Tcoh, Bdll, Bbeta, Tc, d, eps)

sigma_sat_pos_m = 10.0

mp_min_m = float(np.nanmin(sigma_mp_m))
mp_max_m = float(np.nanmax(sigma_mp_m))
mp_med_m = float(np.nanmedian(sigma_mp_m))

clk_sigma_s_vis = np.where(np.isfinite(sigma_clk_m), sigma_clk_m/c_light, np.nan)
clk_sigma_ns_med = float(np.nanmedian(clk_sigma_s_vis) * 1e9)
clk_sigma_ns_p25 = float(np.nanpercentile(clk_sigma_s_vis, 25) * 1e9)
clk_sigma_ns_p75 = float(np.nanpercentile(clk_sigma_s_vis, 75) * 1e9)

cn0_med = float(np.nanmedian(CN0_plot))
cn0_min = float(np.nanmin(CN0_plot))
cn0_max = float(np.nanmax(CN0_plot))

summary_lines = []
summary_lines.append("==== Lunar AFS UERE — Parameter Summary ====")
summary_lines.append(f"CN0 (dB-Hz):   median={cn0_med:.1f}, min={cn0_min:.1f}, max={cn0_max:.1f}")
summary_lines.append("")
summary_lines.append("Receiver Error ~ N(0, σ_RX)")
summary_lines.append(f"  σ_RX (current link): median={sigma_rx_curr_m:.3f} m, 68%={sigma_rx_curr_m_p68:.3f} m")
summary_lines.append(f"  σ_RX @ 37 dB-Hz (reference): {sigma_rx_ref_m_37db:.3f} m  # 文献基准 ~0.935 m")
summary_lines.append("")
summary_lines.append("Satellite i position error ~ N(0, σ_Sat-Pos)")
summary_lines.append(f"  σ_Sat-Pos,DSN = {sigma_sat_pos_m:.1f} m  (assumed)")
summary_lines.append("")
summary_lines.append("Multipath error ~ N(0, σ_MPTH(θ))")
summary_lines.append(f"  σ_MPTH (visible): min={mp_min_m:.2f} m, median={mp_med_m:.2f} m, max={mp_max_m:.2f} m")
summary_lines.append("")
summary_lines.append("Clock (Cesium) error ~ N(0, σ_CLK(t))")
summary_lines.append(f"  σ_CLK (typical, 1σ): {clk_sigma_ns_med:.1f} ns  (IQR: {clk_sigma_ns_p25:.1f}–{clk_sigma_ns_p75:.1f} ns)")
summary_text = "\n".join(summary_lines)

print("\n" + summary_text + "\n")

with open("uere_parameters_summary.txt", "w", encoding="utf-8") as f:
    f.write(summary_text)

param_row = pd.DataFrame([{
    "cn0_dbhz_median": cn0_med, "cn0_dbhz_min": cn0_min, "cn0_dbhz_max": cn0_max,
    "sigma_rx_curr_m_median": sigma_rx_curr_m, "sigma_rx_curr_m_p68": sigma_rx_curr_m_p68,
    "sigma_rx_ref_m_37db": sigma_rx_ref_m_37db,
    "sigma_sat_pos_m": sigma_sat_pos_m,
    "sigma_mp_min_m": mp_min_m, "sigma_mp_med_m": mp_med_m, "sigma_mp_max_m": mp_max_m,
    "sigma_clk_ns_med": clk_sigma_ns_med, "sigma_clk_ns_p25": clk_sigma_ns_p25, "sigma_clk_ns_p75": clk_sigma_ns_p75
}])
param_row.to_csv("uere_parameters_summary.csv", index=False)

# ---------------------------- CSV -----------------------------------

out_df = pd.DataFrame({
    "t_sec": t,
    "time_hr": t_hr,
    "visible": vis.astype(int),
    "elev_deg": elev_deg,
    "segment_idx": seg_idx,
    "cn0_dbhz": np.where(vis, CN0_dBHz, np.nan),
    "sigma_clk_m": sigma_clk_m,
    "sigma_rec_dll_m": sigma_rec_m,
    "sigma_mp_m":  sigma_mp_m,
    "sigma_ephem_m": sigma_ephem_m,
    "eph_resid_los_m": eph_resid_los_m,
    "e_clk_sample_m": e_clk_m,
    "sigma_uere_m": sigma_uere_m,
    "var_clk_m2": var_clk,
    "var_rec_dll_m2": var_rec,
    "var_mp_m2":  var_mp,
    "var_ephem_m2": var_eph,
    "var_total_m2": var_total,
    "actual_error_m": e_total_m,
    "res_dx_m": res_xyz_m[:,0],
    "res_dy_m": res_xyz_m[:,1],
    "res_dz_m": res_xyz_m[:,2],
    "res_norm_m": np.linalg.norm(res_xyz_m, axis=1)
})
out_df.to_csv(CSV_OUT, index=False)

print("Saved:", CSV_OUT)
print("Saved:", FIG1_OUT)
print("Saved:", FIG2_OUT)
print("Saved:", FIG3_OUT)
print("Saved:", FIG4_OUT)
