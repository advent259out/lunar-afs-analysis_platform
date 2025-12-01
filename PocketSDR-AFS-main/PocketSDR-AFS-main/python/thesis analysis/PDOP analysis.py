import pylupnt as pnt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm  # progress bar
import time as pytime

# Record start time
script_start_time = pytime.time()

# ==============================================================================
# Step 1: Set up simulation time and receiver
# ==============================================================================
print("Step 1: Setting up simulation time and receiver...")
t0 = pnt.gregorian2time(1984, 5, 30, 16, 44, 48.0)

# Total duration (here: 1.6 years)
dt_total_2yr = 1.6 * pnt.DAYS_YEAR * pnt.SECS_DAY
dt_step_2yr = 15 * pnt.SECS_MINUTE   # coverage analysis step
dt_prop_2yr = 1 * pnt.SECS_MINUTE    # propagation integrator step

tspan_2yr = np.arange(0, dt_total_2yr + dt_step_2yr, dt_step_2yr)
tfs_2yr = t0 + tspan_2yr
n_steps = len(tfs_2yr)

min_elevation = 10 * pnt.RAD
r_south_pole_me = pnt.lat_lon_alt2cart(np.array([-90 * pnt.RAD, 0, 0]), pnt.R_MOON)

sim_years = dt_total_2yr / (pnt.DAYS_YEAR * pnt.SECS_DAY)

print(f"  Start epoch: {pnt.time2gregorian_string(t0)} TAI")
print(f"  Simulation duration: {sim_years:.2f} years, step: {dt_step_2yr / 60.0:.1f} minutes")

# ==============================================================================
# Step 2: Define 8-satellite constellation
# ==============================================================================
print("\nStep 2: Configuring 8-satellite constellation...")
n_sat = 8
coes0_op_list = [
    [6540.0, 0.6, 56.3 * pnt.RAD,   0.0 * pnt.RAD, 90.0 * pnt.RAD,   0.0 * pnt.RAD],   # PRN-01
    [6540.0, 0.6, 56.3 * pnt.RAD,   0.0 * pnt.RAD, 90.0 * pnt.RAD,  90.0 * pnt.RAD],   # PRN-02
    [6540.0, 0.6, 56.3 * pnt.RAD,   0.0 * pnt.RAD, 90.0 * pnt.RAD, 180.0 * pnt.RAD],   # PRN-03
    [6540.0, 0.6, 56.3 * pnt.RAD,   0.0 * pnt.RAD, 90.0 * pnt.RAD, -90.0 * pnt.RAD],   # PRN-04
    [6540.0, 0.6, 56.3 * pnt.RAD, 180.0 * pnt.RAD, 90.0 * pnt.RAD,  45.0 * pnt.RAD],   # PRN-05
    [6540.0, 0.6, 56.3 * pnt.RAD, 180.0 * pnt.RAD, 90.0 * pnt.RAD, 135.0 * pnt.RAD],   # PRN-06
    [6540.0, 0.6, 56.3 * pnt.RAD, 180.0 * pnt.RAD, 90.0 * pnt.RAD, -135.0 * pnt.RAD],  # PRN-07
    [6540.0, 0.6, 56.3 * pnt.RAD, 180.0 * pnt.RAD, 90.0 * pnt.RAD, -45.0 * pnt.RAD],   # PRN-08
]
coes0_op = np.array(coes0_op_list)

rvs0_ci = np.zeros((n_sat, 6))
for i in range(n_sat):
    rv0_op_i = pnt.classical2cart(coes0_op[i], pnt.GM_MOON)
    rvs0_ci[i] = pnt.convert_frame(t0, rv0_op_i, pnt.MOON_OP, pnt.MOON_CI)

# ==============================================================================
# Step 3: Define high-fidelity dynamics model
# ==============================================================================
print("\nStep 3: Configuring high-fidelity dynamics model...")
dyn_nbody = pnt.NBodyDynamics(pnt.IntegratorType.RK4)
dyn_nbody.add_body(pnt.Body.Moon(7, 1))
dyn_nbody.add_body(pnt.Body.Earth())
dyn_nbody.add_body(pnt.Body.Sun())
dyn_nbody.set_frame(pnt.MOON_CI)
dyn_nbody.set_time_step(dt_prop_2yr)

# ==============================================================================
# Step 4: Propagate all 8 satellite orbits (high computational cost)
# ==============================================================================
print("\nStep 4: Starting 8-satellite orbit propagation (WARNING: very time-consuming!)")
rvs_ci = np.zeros((n_sat, n_steps, 6))
for i in range(n_sat):
    print(f"\n--- Propagating PRN-{i + 1:02d} ({i + 1} of {n_sat}) ---")
    rvs_ci[i] = dyn_nbody.propagate(rvs0_ci[i], t0, tfs_2yr, progress=True)
print("\n--- Orbit propagation complete ---")

# ==============================================================================
# Step 5: Coverage analysis and PDOP computation
# ==============================================================================
print("\nStep 5: Performing coverage analysis and computing PDOP...")
pdop_history = np.full(n_steps, np.inf)  # PDOP history (default: infinity)
sats_in_view_history = np.zeros(n_steps, dtype=int)

print("  (Converting coordinates to MOON_ME frame...)")
rs_me = np.zeros((n_sat, n_steps, 3))
for i in tqdm(range(n_sat), desc="Frame conversion", unit="sat"):
    rs_me[i] = pnt.convert_frame(
        tfs_2yr, rvs_ci[i], pnt.MOON_CI, pnt.MOON_ME, rotate_only=True
    )[..., :3]

print("  (Computing visibility and PDOP...)")
for i in tqdm(range(n_steps), desc="PDOP computation", unit="step"):
    t = tfs_2yr[i]
    r_sats_me_t = rs_me[:, i, :]  # satellite positions in ME frame at this epoch

    # Compute azimuth, elevation, and range
    az_el_range = pnt.cart2az_el_range(r_sats_me_t, r_south_pole_me)
    elevations = az_el_range[:, 1]
    azimuths = az_el_range[:, 0]

    # Visible satellites
    visible_mask = elevations >= min_elevation
    n_visible = np.sum(visible_mask)
    sats_in_view_history[i] = n_visible

    # --- PDOP computation (position-only, requires at least 3 satellites) ---
    if n_visible < 3:
        pdop_history[i] = np.inf
    else:
        el_vis = elevations[visible_mask]
        az_vis = azimuths[visible_mask]

        # Geometry matrix H (n_visible x 3) for ENU position solution
        H = np.zeros((n_visible, 3))
        H[:, 0] = np.cos(el_vis) * np.sin(az_vis)  # East
        H[:, 1] = np.cos(el_vis) * np.cos(az_vis)  # North
        H[:, 2] = np.sin(el_vis)                   # Up

        try:
            Q = np.linalg.inv(H.T @ H)  # covariance matrix
            pdop_history[i] = np.sqrt(Q[0, 0] + Q[1, 1] + Q[2, 2])  # PDOP
        except np.linalg.LinAlgError:
            pdop_history[i] = np.inf  # singular matrix

print("PDOP computation complete.")

# ==============================================================================
# Step 6: Print statistics
# ==============================================================================
print("\nStep 6: Computing coverage statistics...")
four_fold_coverage = np.sum(sats_in_view_history >= 4) / n_steps * 100
pdop_good_percent = np.sum(pdop_history <= 6) / n_steps * 100  # e.g., PDOP <= 6

print("\n\n--- Constellation performance at Lunar South Pole (full simulation) ---")
print(f"  >= 4 visible satellites (quadruple coverage): {four_fold_coverage:>7.3f} %")
print(f"  PDOP <= 6.0:                                  {pdop_good_percent:>7.3f} %")
print(f"  Average number of visible satellites:          {np.mean(sats_in_view_history):.2f}")

# ==============================================================================
# Step 7: Plot PDOP evolution (English titles)
# ==============================================================================
print("\nStep 7: Generating PDOP time-series plot...")
fig, ax = plt.subplots(figsize=(14, 7))

time_days = tspan_2yr / pnt.SECS_DAY
ax.plot(time_days, pdop_history, lw=0.5, label='PDOP (Position-only, min 3 sats)')

ax.set_title("8-Sat Constellation PDOP at Lunar South Pole", fontsize=16)
ax.set_xlabel(f"Days since {pnt.time2gregorian_string(t0)} TAI", fontsize=12)
ax.set_ylabel("PDOP (Position Dilution of Precision)", fontsize=12)

# Limit y-axis to a reasonable range for visualization
ax.set_ylim(0, 20)
ax.grid(True)
ax.legend()

plt.tight_layout()
plt.savefig("pdop_analysis_8sat_south_pole.png")
print("\nPlot saved as 'pdop_analysis_8sat_south_pole.png'")

script_end_time = pytime.time()
total_time = script_end_time - script_start_time
print("\n--- Simulation complete ---")
print(f"Total runtime: {total_time / 60.0:.2f} minutes.")

plt.show()
