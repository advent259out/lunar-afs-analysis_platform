#!/usr/bin/env python3
import argparse, re
import numpy as np
import matplotlib.pyplot as plt

# Moon radius (m) consistent with receiver code
R_MOON = 1737_400.0

POS_RE = re.compile(r"^\$POS,([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^,]+)")

def parse_pos_log(path, afs_only=False, tmin=None, tmax=None):
    t, lat, lon, h = [], [], [], []
    with open(path, 'r', errors='ignore') as f:
        for line in f:
            if not line.startswith('$POS,'): continue
            parts = line.strip().split(',')
            # $POS has at least: $POS,time,YYYY,MM,DD,hh,mm,sec,lat,lon,h,...
            if len(parts) < 11: continue
            try:
                # $POS,time,YYYY,MM,DD,hh,mm,sec,lat_deg,lon_deg,h_m,...
                ti = float(parts[1])
                la = float(parts[8])
                lo = float(parts[9])
                hh = float(parts[10])
                # optional AFS-only filter: second last field equals 5 in AFS solver
                if afs_only:
                    try:
                        solq = int(float(parts[-2]))
                        if solq != 5:
                            continue
                    except Exception:
                        continue
                if tmin is not None and ti < tmin: continue
                if tmax is not None and ti > tmax: continue
                t.append(ti); lat.append(la); lon.append(lo); h.append(hh)
            except ValueError:
                continue
    return np.array(t), np.array(lat), np.array(lon), np.array(h)

def llh_to_enu(lat_deg, lon_deg, h_m, lat0_deg, lon0_deg, h0_m):
    lat  = np.deg2rad(lat_deg)
    lon  = np.deg2rad(lon_deg)
    lat0 = np.deg2rad(lat0_deg)
    lon0 = np.deg2rad(lon0_deg)
    R    = R_MOON + h0_m
    dE = (lon - lon0) * np.cos(lat0) * R
    dN = (lat - lat0) * R
    dU = (h_m - h0_m)
    return dE, dN, dU

def main():
    ap = argparse.ArgumentParser(description='Plot NE track and Up time series from $POS log')
    ap.add_argument('log', help='receiver log file (from pocket_trk -log)')
    ap.add_argument('--afs-only', action='store_true', help='use only AFS solver $POS (penultimate field==5)')
    ap.add_argument('--tmin', type=float, default=None, help='min time (s) to include')
    ap.add_argument('--tmax', type=float, default=None, help='max time (s) to include')
    ap.add_argument('-ref', choices=['median','first'], default='median', help='reference position (median or first)')
    ap.add_argument('-o', default='pos_track.png', help='output image file')
    args = ap.parse_args()

    t, lat, lon, h = parse_pos_log(args.log, afs_only=args.afs_only, tmin=args.tmin, tmax=args.tmax)
    if len(t) == 0:
        print('No $POS lines found in log'); return

    if args.ref == 'first':
        lat0, lon0, h0 = float(lat[0]), float(lon[0]), float(h[0])
    else:
        lat0, lon0, h0 = float(np.median(lat)), float(np.median(lon)), float(np.median(h))

    E, N, U = llh_to_enu(lat, lon, h, lat0, lon0, h0)
    # Auto-scale NE to km if span is too large to avoid 1e6 scientific ticks
    span = max(np.max(np.abs(E)), np.max(np.abs(N))) if len(E) else 0.0
    scale = 1.0
    unit = '(m)'
    if span > 5e3:
        scale = 1e-3
        unit = '(km)'
    E_s = E * scale
    N_s = N * scale
    H = np.sqrt(E**2 + N**2)

    # stats
    h50, h95 = np.percentile(H, [50, 95])
    v50, v95 = np.percentile(np.abs(U), [50, 95])

    fig = plt.figure(figsize=(10,6))
    gs = fig.add_gridspec(2, 1, height_ratios=[3,1.2])
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[1,0])

    # NE track
    ax1.plot(E_s, N_s, '-o', ms=3, lw=1.0, color='#00cfe8')
    ax1.plot(0, 0, 'x', ms=8, color='lime', label='ref')
    ax1.set_xlabel(f'East {unit}')
    ax1.set_ylabel(f'North {unit}')
    ax1.grid(True, lw=0.3)
    # Only enforce equal aspect for small local tracks
    if span <= 5e3:
        ax1.set_aspect('equal', adjustable='datalim')
    # Disable scientific notation on axes for readability
    ax1.ticklabel_format(style='plain', useOffset=False)
    ax1.legend(loc='best', fontsize=8)
    ax1.text(0.02, 0.98, f'Num fixes: {len(t)}\nH err (50%,95%) = ({h50:.1f},{h95:.1f}) m',
             transform=ax1.transAxes, va='top', ha='left', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='0.8', alpha=0.7))

    # Up time series
    ax2.plot(t, U, '-o', ms=3, lw=1.0, color='#00cfe8')
    ax2.axhline(0.0, color='lime', lw=1.0, label='ref')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Up (m)')
    ax2.grid(True, lw=0.3)
    ax2.text(0.02, 0.95, f'V err (50%,95%) = ({v50:.1f},{v95:.1f}) m',
             transform=ax2.transAxes, va='top', ha='left', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='0.8', alpha=0.7))

    fig.tight_layout()
    fig.savefig(args.o, dpi=150)
    print(f'Saved {args.o}')

if __name__ == '__main__':
    main()
