#!/usr/bin/env python3
"""
Compute and plot 1-hour PDOP from PocketSDR log + HALO orbit CSVs.

Inputs:
  - pocket_trk log containing $POS/$OBS lines (default: 3000noise.txt)
  - HALO/output directory with halo_prn{1..8}.csv truth orbits

The script:
  1. Reads the first $POS entry to determine the receiver LLH (Moon-fixed).
  2. Parses $OBS epochs within the first hour, recording visible satellites.
  3. Interpolates satellite ECEF positions from HALO CSVs at the corresponding TOW.
  4. Builds the geometry matrix and evaluates PDOP for each epoch.
  5. Plots PDOP vs relative time and saves a PNG + CSV summary.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

R_MOON = 1_737_400.0  # metres
C_LIGHT = 299_792_458.0


def llh_to_ecef(lat_deg: float, lon_deg: float, h_m: float) -> np.ndarray:
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    r = R_MOON + h_m
    cos_lat = math.cos(lat)
    x = r * cos_lat * math.cos(lon)
    y = r * cos_lat * math.sin(lon)
    z = r * math.sin(lat)
    return np.array([x, y, z], dtype=float)


def ecef_to_llh(ecef: np.ndarray) -> Tuple[float, float, float]:
    x, y, z = ecef
    r = np.linalg.norm(ecef)
    lat = math.asin(z / r)
    lon = math.atan2(y, x)
    h = r - R_MOON
    return lat, lon, h


def ecef_to_enu(user_llh: Tuple[float, float, float], vec: np.ndarray) -> np.ndarray:
    lat, lon, _ = user_llh
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)
    t = np.array([[-sin_lon, cos_lon, 0.0],
                  [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
                  [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat]])
    return t @ vec


def load_halo_orbits(halo_dir: Path) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    data = {}
    for csv in sorted(halo_dir.glob("halo_prn*.csv")):
        try:
            prn = int("".join(filter(str.isdigit, csv.stem)))
        except ValueError:
            continue
        arr = np.genfromtxt(csv, delimiter=",", skip_header=1)
        if arr.ndim != 2 or arr.shape[1] < 5:
            continue
        times = arr[:, 1]
        pos = arr[:, 2:5] * 1000.0  # km -> m
        data[prn] = (times, pos)
    if not data:
        raise RuntimeError(f"No halo_prn*.csv files found in {halo_dir}")
    return data


def interp_sat_pos(prn: int, tow: float, orbits: Dict[int, Tuple[np.ndarray, np.ndarray]]) -> np.ndarray | None:
    if prn not in orbits:
        return None
    times, pos = orbits[prn]
    if tow < times[0] or tow > times[-1]:
        return None
    x = np.interp(tow, times, pos[:, 0])
    y = np.interp(tow, times, pos[:, 1])
    z = np.interp(tow, times, pos[:, 2])
    return np.array([x, y, z], dtype=float)


def parse_log(log_path: Path, duration_s: float = 3600.0):
    epochs: Dict[float, Dict[str, List[str]]] = {}
    first_time = None
    user_ecef = None
    epoch_list: List[float] = []
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("$POS") and user_ecef is None:
                parts = line.strip().split(",")
                if len(parts) >= 11:
                    lat = float(parts[8])
                    lon = float(parts[9])
                    hgt = float(parts[10])
                    user_ecef = llh_to_ecef(lat, lon, hgt)
                continue
            if not line.startswith("$OBS"):
                continue
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            time = float(parts[1])
            tow = float(parts[3])
            sat = parts[4]
            if first_time is None:
                first_time = time
            rel = time - first_time
            if rel < 0:
                rel = 0.0
            if rel > duration_s:
                break
            prn_str = "".join(filter(str.isdigit, sat))
            if not prn_str:
                continue
            prn = int(prn_str)
            if time not in epochs:
                epochs[time] = {"tow": tow, "prns": []}
                epoch_list.append(time)
            epochs[time]["prns"].append(prn)
    if first_time is None:
        raise RuntimeError("No $OBS lines found in log.")
    if user_ecef is None:
        raise RuntimeError("No $POS entry found to define receiver location.")
    return user_ecef, first_time, epoch_list, epochs


def compute_pdop(orbits: Dict[int, Tuple[np.ndarray, np.ndarray]],
                 user_ecef: np.ndarray,
                 epoch_list: List[float],
                 epochs: Dict[float, Dict[str, List[int]]]) -> Tuple[np.ndarray, np.ndarray, List[Tuple[float, float, List[Tuple[int, np.ndarray]]]]]:
    pdops = []
    rel_times = []
    sat_info_per_epoch: List[Tuple[float, float, List[Tuple[int, np.ndarray]]]] = []
    start_time = epoch_list[0] if epoch_list else 0.0
    for time in epoch_list:
        tow = epochs[time]["tow"]
        prns = epochs[time]["prns"]
        rows = []
        sat_list: List[Tuple[int, np.ndarray]] = []
        for prn in prns:
            sat_pos = interp_sat_pos(prn, tow, orbits)
            if sat_pos is None:
                continue
            sat_list.append((prn, sat_pos))
            rho = sat_pos - user_ecef
            dist = np.linalg.norm(rho)
            if dist <= 0:
                continue
            u = rho / dist
            rows.append(np.array([-u[0], -u[1], -u[2], 1.0]))
        if len(rows) < 4:
            pdops.append(np.nan)
            rel_times.append(time - start_time)
            sat_info_per_epoch.append((time, tow, sat_list))
            continue
        H = np.vstack(rows)
        try:
            Q = np.linalg.inv(H.T @ H)
            pdop = math.sqrt(Q[0, 0] + Q[1, 1] + Q[2, 2])
        except np.linalg.LinAlgError:
            pdop = np.nan
        pdops.append(pdop)
        rel_times.append(time - start_time)
        sat_info_per_epoch.append((time, tow, sat_list))
    return np.array(rel_times), np.array(pdops), sat_info_per_epoch


def plot_pdop(times: np.ndarray, pdops: np.ndarray, out_path: Path) -> None:
    plt.figure(figsize=(10, 4))
    plt.plot(times / 60.0, pdops, marker='.', lw=1.0, ms=3)
    plt.xlabel("Time since start (minutes)")
    plt.ylabel("PDOP")
    plt.title("PDOP over first hour")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_skyplot(sat_list: List[Tuple[int, np.ndarray]], user_ecef: np.ndarray,
                 title: str, out_path: Path) -> None:
    if not sat_list:
        return
    user_llh = ecef_to_llh(user_ecef)
    az_list = []
    el_list = []
    labels = []
    for prn, sat_pos in sat_list:
        vec = sat_pos - user_ecef
        enu = ecef_to_enu(user_llh, vec)
        e, n, u = enu
        horiz = math.hypot(e, n)
        el = math.degrees(math.atan2(u, horiz))
        az = math.degrees(math.atan2(e, n))
        if az < 0:
            az += 360.0
        az_list.append(math.radians(az))
        el_list.append(90.0 - el)
        labels.append(str(prn))
    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_rlim(0, 90)
    ax.set_rlabel_position(135)
    ticks = [0, 30, 60, 90]
    ax.set_rticks([90 - t for t in ticks])
    ax.set_yticklabels([f"{t}°" for t in ticks])
    ax.grid(True, alpha=0.3)
    ax.scatter(az_list, el_list, c='tab:blue')
    for az, r, lbl in zip(az_list, el_list, labels):
        ax.text(az, r, lbl, fontsize=9, ha='center', va='center')
    ax.set_title(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Plot 1-hour PDOP from PocketSDR log and HALO orbits.")
    ap.add_argument("--log", type=Path,
                    default=Path(r"E:\Project\PocketSDR-AFS-main\PocketSDR-AFS-main\app\pocket_trk\3000noise.txt"),
                    help="PocketSDR log file containing $OBS and $POS entries.")
    ap.add_argument("--halo-dir", type=Path,
                    default=Path(r"E:\Project\HALO\output"),
                    help="Directory with halo_prn*.csv files.")
    ap.add_argument("--duration", type=float, default=3600.0,
                    help="Duration in seconds from first epoch to include (default 3600).")
    ap.add_argument("--out", type=Path, default=Path("HALO/pdop_first_hour.png"),
                    help="Output PDOP plot path.")
    ap.add_argument("--csv", type=Path, default=Path("HALO/pdop_first_hour.csv"),
                    help="Optional CSV output with time,PDOP.")
    ap.add_argument("--skyplot", type=Path, default=Path("HALO/pdop_max_skyplot.png"),
                    help="Skyplot image path at max PDOP (default HALO/pdop_max_skyplot.png).")
    args = ap.parse_args()

    orbits = load_halo_orbits(args.halo_dir)
    user_ecef, _, epoch_list, epochs = parse_log(args.log, duration_s=args.duration)
    times, pdops, sat_info = compute_pdop(orbits, user_ecef, epoch_list, epochs)
    plot_pdop(times, pdops, args.out)

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        data = np.column_stack((times, pdops))
        np.savetxt(args.csv, data, delimiter=",", header="time_s,pdop", comments="")
        print(f"Saved PDOP samples to {args.csv}")
    if np.any(np.isfinite(pdops)) and args.skyplot:
        idx = int(np.nanargmax(pdops))
        epoch_time, tow, sat_list = sat_info[idx]
        if sat_list and len(sat_list) >= 1:
            title = f"Skyplot at t={epoch_time:.1f}s (PDOP={pdops[idx]:.2f})"
            plot_skyplot(sat_list, user_ecef, title, args.skyplot)
            print(f"Saved skyplot to {args.skyplot}")
    print(f"Saved PDOP plot to {args.out}")


if __name__ == "__main__":
    main()
