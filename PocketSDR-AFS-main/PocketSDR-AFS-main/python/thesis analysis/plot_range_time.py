#!/usr/bin/env python3
import argparse, re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

RNG_RE = re.compile(r"^\$RNG,([^,]+),([^,]+),([^,]+),([^,]+)")
SIM_RE = re.compile(r"^\$SIMRNG,([^,]+),([^,]+),([^,]+)")

def parse_log(path, prn_filter=None):
    by_prn = defaultdict(lambda: {"t": [], "obs": [], "geom": []})
    with open(path, 'r', errors='ignore') as f:
        for line in f:
            m = RNG_RE.match(line)
            if not m: continue
            try:
                t    = float(m.group(1))
                prn  = int(m.group(2))
                obs  = float(m.group(3))
                geom = float(m.group(4))
            except ValueError:
                continue
            if prn_filter and prn not in prn_filter: continue
            d = by_prn[prn]
            d["t"].append(t); d["obs"].append(obs); d["geom"].append(geom)
    return by_prn

def parse_simrng(path):
    sim = defaultdict(dict)  # prn -> {rounded_time: geom}
    with open(path, 'r', errors='ignore') as f:
        for line in f:
            m = SIM_RE.match(line)
            if not m: continue
            try:
                t   = float(m.group(1))
                prn = int(m.group(2))
                g   = float(m.group(3))
            except ValueError:
                continue
            sim[prn][round(t,3)] = g
    return sim

def main():
    ap = argparse.ArgumentParser(description="Plot pseudorange and geometric range vs time per PRN")
    ap.add_argument('log', help='receiver log file (from pocket_trk -log, contains $RNG)')
    ap.add_argument('--simrng', help='optional afs_sim rng log (contains $SIMRNG)', default=None)
    ap.add_argument('-prn', nargs='*', type=int, default=None, help='filter PRNs')
    ap.add_argument('-o', default='range_time.png', help='output image')
    args = ap.parse_args()

    prn_filter = set(args.prn) if args.prn else None
    data = parse_log(args.log, prn_filter)
    sim = parse_simrng(args.simrng) if args.simrng else None
    if not data:
        print('No $RNG lines found. Build receiver with this logging and run with -log.'); return

    n = len(data)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 3*nrows), squeeze=False)
    axlist = axes.flatten()
    for k, (prn, d) in enumerate(sorted(data.items())):
        ax = axlist[k]
        # sort by time to find the last epoch robustly
        order = np.argsort(d["t"]) if len(d["t"]) else []
        t = np.array(d["t"])[order]
        obs = np.array(d["obs"])[order]
        geom = np.array(d["geom"])[order]

        # override geometric range using simrng if provided
        if sim and prn in sim:
            gmap = sim[prn]
            geom2 = []
            for tt in t:
                geom2.append(gmap.get(round(float(tt),3), np.nan))
            geom = np.array(geom2)

        ax.plot(t, obs, '-', lw=1.0, label='Pseudorange')
        ax.plot(t, geom, '-', lw=1.0, label='Geometric (sim)' if sim else 'Geometric')
        ax.set_title(f'PRN {prn}')
        ax.set_xlabel('Time (s)'); ax.set_ylabel('Range (m)')
        ax.grid(True, lw=0.3)
        ax.legend(fontsize=8)
        # annotate last-epoch values on the plot
        if len(t):
            tl, pl, gl = float(t[-1]), float(obs[-1]), float(geom[-1])
            ax.text(0.98, 0.02,
                    f't={tl:.1f}s\nP={pl:.3f} m\nG={gl:.3f} m\nΔ={pl-gl:.3f} m',
                    transform=ax.transAxes, ha='right', va='bottom', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='0.8', alpha=0.7))
    for j in range(k+1, len(axlist)):
        fig.delaxes(axlist[j])
    fig.tight_layout()
    fig.savefig(args.o, dpi=150)
    print(f'Saved {args.o}')

if __name__ == '__main__':
    main()
