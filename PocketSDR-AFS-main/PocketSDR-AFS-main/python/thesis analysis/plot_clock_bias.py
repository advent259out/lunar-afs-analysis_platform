#!/usr/bin/env python3
import argparse, re
import matplotlib.pyplot as plt

CLK_RE = re.compile(r"^\$CLK,([^,]+),([^,]+),([^,]+),([^,]+)")

def parse_log(path):
    t, bs, bm, dr = [], [], [], []
    with open(path, 'r', errors='ignore') as f:
        for line in f:
            m = CLK_RE.match(line)
            if not m: continue
            try:
                t.append(float(m.group(1)))
                bs.append(float(m.group(2)))
                bm.append(float(m.group(3)))
                dr.append(float(m.group(4)))
            except ValueError:
                continue
    return t, bs, bm, dr

def main():
    ap = argparse.ArgumentParser(description='Plot receiver clock bias over time from $CLK logs')
    ap.add_argument('log', help='log file from pocket_trk -log')
    ap.add_argument('-u', choices=['s','m'], default='s', help='y-axis unit: bias in seconds or meters')
    ap.add_argument('-o', default='clock_bias_time.png', help='output image')
    args = ap.parse_args()

    t, bs, bm, dr = parse_log(args.log)
    if not t:
        print('No $CLK lines found. Make sure you are using the AFS solver with -log enabled.'); return

    y = bs if args.u == 's' else bm
    yl = 'Clock Bias (s)' if args.u == 's' else 'Clock Bias (m)'

    fig, ax1 = plt.subplots(figsize=(10,4))
    ax1.plot(t, y, '-', lw=1.2, label=yl)
    ax1.set_xlabel('Time (s)'); ax1.set_ylabel(yl)
    ax1.grid(True, lw=0.3)
    # annotate last values
    ax1.text(0.98, 0.02,
             (f't={t[-1]:.1f}s\n' +
              (f'Bias={bs[-1]:.9f}s\n' if args.u=='s' else f'Bias={bm[-1]:.3f}m\n') +
              f'Drift={dr[-1]:.3e} s/s'),
             transform=ax1.transAxes, ha='right', va='bottom', fontsize=8,
             bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='0.8', alpha=0.7))
    fig.tight_layout(); fig.savefig(args.o, dpi=150)
    print(f'Saved {args.o}')

if __name__ == '__main__':
    main()

