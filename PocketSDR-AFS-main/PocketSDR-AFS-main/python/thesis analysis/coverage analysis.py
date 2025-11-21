import pylupnt as pnt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm  # Progress bar
import time as pytime

# Record start time
script_start_time = pytime.time()

# ==============================================================================
# Step 1: Simulation Time and Receiver Setup
# ==============================================================================
print("Step 1: Setting up simulation time and receiver...")

# --- Define start time ---
t0 = pnt.gregorian2time(1984, 5, 30, 16, 44, 48.0)

# --- Define 2-year simulation duration and steps ---
dt_total_2yr = 2 * pnt.DAYS_YEAR * pnt.SECS_DAY  # [s] total duration

# Coverage analysis step: 15 minutes
dt_step_2yr = 15 * pnt.SECS_MINUTE

# Orbit propagation integration step: 1 minute
dt_prop_2yr = 1 * pnt.SECS_MINUTE

tspan_2yr = np.arange(0, dt_total_2yr + dt_step_2yr, dt_step_2yr)
tfs_2yr = t0 + tspan_2yr
n_steps = len(tfs_2yr)

print(f"  Start epoch: {pnt.time2gregorian_string(t0)} TAI")
print(f"  Simulation duration: 2 years")
print(f"  Analysis step: {dt_step_2yr / 60.0} minutes (Total {n_steps} steps)")

# --- Define receiver (Lunar South Pole) ---
min_elevation = 10 * pnt.RAD
r_south_pole_me = pnt.lat_lon_alt2cart(
    np.array([-90 * pnt.RAD, 0, 0]), pnt.R_MOON
)

print(f"  Receiver location: Lunar South Pole")
print(f"  Minimum elevation mask: {min_elevation * pnt.DEG:.1f} deg")

# ==============================================================================
# Step 2: Define 8-Satellite Constellation
# ==============================================================================
print("\nStep 2: Configuring 8-satellite constellation...")

n_sat = 8

# [SMA, ECC, INC, LAN, AOP, MA]
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

# Convert orbital elements (OP frame) to Cartesian state vectors (CI frame)
rvs0_ci = np.zeros((n_sat, 6))
for i in range(n_sat):
    rv0_op_i = pnt.classical2cart(coes0_op[i], pnt.GM_MOON)
    rvs0_ci[i] = pnt.convert_frame(t0, rv0_op_i, pnt.MOON_OP, pnt.MOON_CI)

# ==============================================================================
# Step 3: High-Fidelity Dynamics Model
# ==============================================================================
print("\nStep 3: Configuring high-fidelity dynamics model (Moon + Earth + Sun)...")

dyn_nbody = pnt.NBodyDynamics(pnt.IntegratorType.RK4)

dyn_nbody.add_body(pnt.Body.Moon(7, 1))  # Moon gravity: 7x1
dyn_nbody.add_body(pnt.Body.Earth())    # Earth gravity
dyn_nbody.add_body(pnt.Body.Sun())      # Sun gravity

dyn_nbody.set_frame(pnt.MOON_CI)
dyn_nbody.set_time_step(dt_prop_2yr)

# ==============================================================================
# Step 4: Orbit Propagation
# ==============================================================================
print("\nStep 4: Starting orbit propagation (WARNING: very time-consuming!)")

rvs_ci = np.zeros((n_sat, n_steps, 6))

for i in range(n_sat):
    print(f"\n--- Propagating PRN-{i + 1:02d} ({i+1}/{n_sat}) ---")
    rvs_ci[i] = dyn_nbody.propagate(rvs0_ci[i], t0, tfs_2yr, progress=True)

print("\n--- Orbit propagation complete ---")

# ==============================================================================
# Step 5: Coverage Analysis
# ==============================================================================
print("\nStep 5: Performing coverage analysis at Lunar South Pole...")

visibility = np.zeros((n_sat, n_steps), dtype=bool)

print("  Converting satellite states to MOON_ME frame...")
rs_me = np.zeros((n_sat, n_steps, 3))

for i in tqdm(range(n_sat), desc="Frame conversion", unit="sat"):
    rs_me[i] = pnt.convert_frame(
        tfs_2yr, rvs_ci[i], pnt.MOON_CI, pnt.MOON_ME, rotate_only=True
    )[..., :3]

print("  Computing satellite visibility...")

for i in tqdm(range(n_sat), desc="Visibility computation", unit="sat"):
    elevation_i = pnt.cart2az_el_range(rs_me[i], r_south_pole_me)[:, 1]
    visibility[i] = elevation_i >= min_elevation

# ==============================================================================
# Step 6: Coverage Statistics
# ==============================================================================
print("\nStep 6: Computing coverage statistics...")

mean_pass_duration = np.zeros(n_sat)
mean_gap_duration = np.zeros(n_sat)
percent_coverage = np.zeros(n_sat)

for i in range(n_sat):
    vis_sat = visibility[i]

    edges = np.diff(np.concatenate(([0], vis_sat, [0])).astype(int))
    n_passes = np.sum(edges > 0)

    percent_coverage[i] = np.sum(vis_sat) / n_steps * 100

    if n_passes > 0:
        mean_pass_duration[i] = (
            percent_coverage[i] / 100 * dt_total_2yr / n_passes / pnt.SECS_HOUR
        )

    if n_passes > 1:
        mean_gap_duration[i] = (
            (1 - percent_coverage[i] / 100) * dt_total_2yr /
            (n_passes - 1) / pnt.SECS_HOUR
        )

# --- Constellation-level coverage ---
sats_in_view = np.sum(visibility, axis=0)

one_fold_coverage = np.sum(sats_in_view >= 1) / n_steps * 100
two_fold_coverage = np.sum(sats_in_view >= 2) / n_steps * 100
four_fold_coverage = np.sum(sats_in_view >= 4) / n_steps * 100
six_fold_coverage = np.sum(sats_in_view >= 6) / n_steps * 100

# --- Print results ---
print("\n=== Lunar South Pole Coverage Statistics (2 Years) ===")
print(f"Elevation mask: {min_elevation * pnt.DEG:.1f} deg")

print("\n--- Individual Satellite Coverage ---")
print(" PRN | Coverage (%) | Avg Pass Duration (hr) | Avg Gap Duration (hr) ")
print("-----|--------------|------------------------|----------------------")

for i in range(n_sat):
    print(
        f" {i + 1:02d}  | {percent_coverage[i]:>12.2f} | "
        f"{mean_pass_duration[i]:>22.3f} | {mean_gap_duration[i]:>20.3f}"
    )

print("\n--- Constellation Coverage ---")
print(f"  >= 1 visible satellite : {one_fold_coverage:>7.3f} %")
print(f"  >= 2 visible satellites: {two_fold_coverage:>7.3f} %")
print(f"  >= 4 visible satellites: {four_fold_coverage:>7.3f} %")
print(f"  >= 6 visible satellites: {six_fold_coverage:>7.3f} %")

# ==============================================================================
# Step 7: Plot Visibility Count
# ==============================================================================
print("\nStep 7: Generating visibility plot...")

fig, ax = plt.subplots(figsize=(14, 7))

time_days = tspan_2yr / pnt.SECS_DAY
ax.plot(time_days, sats_in_view, lw=0.5)

ax.set_title("8-Satellite Constellation Visibility at Lunar South Pole (2 Years)", fontsize=16)
ax.set_xlabel(f"Days since {pnt.time2gregorian_string(t0)} TAI", fontsize=12)
ax.set_ylabel("Number of Visible Satellites", fontsize=12)

ax.set_yticks(np.arange(0, n_sat + 2, 1))
ax.set_xlim(time_days[0], time_days[-1])
ax.set_ylim(bottom=-0.5, top=n_sat + 0.5)

ax.grid(True, which="major", linestyle="-", linewidth=0.5)
ax.grid(True, which="minor", linestyle=":", linewidth=0.5)

plt.tight_layout()
plt.savefig("coverage_analysis_8sat_south_pole.png")
print("\nPlot saved as 'coverage_analysis_8sat_south_pole.png'")

script_end_time = pytime.time()
total_time = script_end_time - script_start_time

print("\n--- Simulation Complete ---")
print(f"Total runtime: {total_time / 60.0:.2f} minutes")

plt.show()
