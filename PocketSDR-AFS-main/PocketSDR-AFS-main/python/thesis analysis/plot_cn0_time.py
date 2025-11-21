#!/usr/bin/env python3
import argparse, re
import matplotlib.pyplot as plt
from collections import defaultdict

# Parse $CN0 or fallback to $CH lines from -log output
CN0_RE = re.compile(r"^\$CN0,([^,]+),([^,]+),([^,]+),([^,]+)")
CH_RE  = re.compile(r"^\$CH,([^,]+),([^,]+),([^,]+),([^,]+),([^,]+)")

def parse_log(path, sig_filter=None, prns_filter=None):
    by_prn = defaultdict(lambda: {"t": [], "cn0": [], "sig": None})
    samples = []
    with open(path, 'r', errors='ignore') as f:
        for line in f:
            m = CN0_RE.match(line)
            if m:
                try:
                    t   = float(m.group(1))
                    sig = m.group(2).strip()
                    prn = int(m.group(3))
                    cn0 = float(m.group(4))
                except ValueError:
                    continue
                if sig_filter and sig not in sig_filter: continue
                if prns_filter and prn not in prns_filter: continue
                by_prn[prn]["t"].append(t)
                by_prn[prn]["cn0"].append(cn0)
                by_prn[prn]["sig"] = sig
                samples.append((t, prn, cn0))
                continue
            m = CH_RE.match(line)
            if m:
                # $CH,time,sig,prn,lock,cn0,...
                try:
                    t   = float(m.group(1))
                    sig = m.group(2).strip()
                    prn = int(m.group(3))
                    cn0 = float(m.group(5))
                except ValueError:
                    continue
                if sig_filter and sig not in sig_filter: continue
                if prns_filter and prn not in prns_filter: continue
                by_prn[prn]["t"].append(t)
                by_prn[prn]["cn0"].append(cn0)
                by_prn[prn]["sig"] = sig
                samples.append((t, prn, cn0))
    return by_prn, samples

def main():
    ap = argparse.ArgumentParser(description="Plot CN0 vs time per PRN from pocket_trk -log output")
    ap.add_argument('log', help='log file path (from -log)')
    ap.add_argument('-sig', nargs='*', default=None, help='filter signals, e.g., AFSD AFSP')
    ap.add_argument('-prn', nargs='*', type=int, default=None, help='filter PRNs')
    ap.add_argument('-o', default='cn0_time.png', help='output image file')
    args = ap.parse_args()

    sig_filter = set(args.sig) if args.sig else None
    prn_filter = set(args.prn) if args.prn else None
    data, samples = parse_log(args.log, sig_filter, prn_filter)
    if not data:
        print('No $CN0/$CH lines found. Ensure -log is enabled.'); return

    fig, ax = plt.subplots(figsize=(10,5))
    for prn, d in sorted(data.items()):
        if not d["t"]: continue
        lbl = f'PRN {prn}' + (f' ({d["sig"]})' if d["sig"] else '')
        ax.plot(d["t"], d["cn0"], '-', lw=1.0, label=lbl)
    thresh = 35.0
    ax.axhline(thresh, color='tomato', lw=1.0, ls='--', label=f'{thresh:.0f} dB-Hz threshold')

    counts = defaultdict(set)
    for t, prn, cn0 in samples:
        if cn0 >= thresh:
            key = round(t, 3)
            counts[key].add(prn)
    ax2 = None
    if counts:
        times = sorted(counts.keys())
        values = [len(counts[t]) for t in times]
        ax2 = ax.twinx()
        ax2.step(times, values, where='post', color='black', lw=1.0, alpha=0.6,
                 label='Sat ≥35 dB-Hz')
        ax2.set_ylabel('Satellites in fix')
        ax2.set_ylim(0, max(values) + 1)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('C/N0 (dB-Hz)')
    ax.set_title('CN0 over Time by PRN')
    ax.grid(True, lw=0.3)

    handles, labels = ax.get_legend_handles_labels()
    if ax2:
        h2, l2 = ax2.get_legend_handles_labels()
        handles += h2
        labels += l2
    ax.legend(handles, labels, ncol=3, fontsize=8)

    fig.tight_layout()
    fig.savefig(args.o, dpi=150)
    print(f'Saved {args.o}')

if __name__ == '__main__':
    main()
