#!/usr/bin/env python3
"""
Scatter plot of North/East errors referenced to the lunar south pole from a pocket_trk log.

Example:
    python plot_ne_south_pole.py ../app/pocket_trk/log_cheb9.txt -o cheb9_ne.png
"""
import argparse
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

R_MOON = 1_737_400.0  # meters


def parse_pos_from_log(log_path):
    """Return arrays of latitude and longitude (deg) from $POS lines in log."""
    t, lat, lon = [], [], []
    with open(log_path, 'r', errors='ignore') as f:
        for line in f:
            if not line.startswith('$POS,'):
                continue
            parts = line.strip().split(',')
            if len(parts) < 11:
                continue
            try:
                t.append(float(parts[1]))
                lat.append(float(parts[8]))
                lon.append(float(parts[9]))
            except ValueError:
                continue
    if not t:
        raise ValueError(f'No $POS lines found in {log_path}')
    return np.array(t), np.array(lat), np.array(lon)


def ll_to_ne(lat_deg, lon_deg, lat_ref=-90.0, lon_ref=0.0, h_ref=0.0):
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    lat0 = math.radians(lat_ref)
    lon0 = math.radians(lon_ref)
    radius = R_MOON + h_ref
    east = (lon - lon0) * math.cos(lat0) * radius
    north = (lat - lat0) * radius
    return east, north


def main():
    ap = argparse.ArgumentParser(description='Plot NE scatter referenced to lunar south pole')
    ap.add_argument('log', help='pocket_trk log containing $POS lines')
    ap.add_argument('-o', '--output', default='south_pole_ne.png', help='output image file')
    ap.add_argument('--circle', choices=['r95', 'max', 'std'], default='r95',
                    help='circle radius metric')
    ap.add_argument('--ref-lat', type=float, default=-89.66, help='reference latitude (deg)')
    ap.add_argument('--ref-lon', type=float, default=129.20, help='reference longitude (deg)')
    args = ap.parse_args()

    _, lat, lon = parse_pos_from_log(args.log)
    east, north = ll_to_ne(lat, lon, lat_ref=args.ref_lat, lon_ref=args.ref_lon)

    errors = np.hypot(east, north)
    if args.circle == 'r95':
        radius = float(np.percentile(errors, 95))
    elif args.circle == 'max':
        radius = float(np.max(errors))
    else:  # std ~ 2-sigma circle
        radius = 2.0 * math.sqrt(np.var(east) + np.var(north))

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(east, north, s=18, color='#1f77b4')
    circle = plt.Circle((0, 0), radius, edgecolor='#ff6f00', facecolor='none', lw=1.5)
    ax.add_patch(circle)
    span = max(np.max(np.abs(east)), np.max(np.abs(north)), radius) * 1.1
    ax.set_xlim(-span, span)
    ax.set_ylim(-span, span)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, lw=0.3)
    ax.set_xlabel('Eastward [m]')
    ax.set_ylabel('Northward [m]')
    ax.set_title('NE Errors vs Lunar South Pole')
    ax.text(0.02, 0.98,
            f'N={len(east)}\nmedian={np.median(errors):.2f} m\n95%={np.percentile(errors,95):.2f} m\ncircle={radius:.2f} m',
            transform=ax.transAxes, va='top', ha='left', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='0.75', alpha=0.85))
    fig.tight_layout()
    out_path = Path(args.output)
    fig.savefig(out_path, dpi=200)
    print(f'Saved {out_path}')


if __name__ == '__main__':
    main()
