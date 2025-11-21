#!/usr/bin/env python3
"""Plot SNR vs. time from pocket_trk log and report RMS."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


CH_RE = re.compile(
    r"^\$CH,([^,]+),([^,]+),(\d+),(\d+),([\d\.-]+),([\d\.-]+)")


def parse_log(path: Path, sig_filter: str | None, prn_filter: int | None) -> Dict[int, List[tuple[float, float]]]:
    data: Dict[int, List[tuple[float, float]]] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("$CH"):
                continue
            m = CH_RE.match(line)
            if not m:
                continue
            time = float(m.group(1))
            sig = m.group(2).strip()
            prn = int(m.group(3))
            cn0 = float(m.group(5))
            try:
                snr = float(m.group(6))
            except ValueError:
                continue
            if not np.isfinite(snr):
                continue
            if sig_filter and sig != sig_filter:
                continue
            if prn_filter and prn != prn_filter:
                continue
            data.setdefault(prn, []).append((time, snr))
    return data


def plot_series(series: Dict[int, List[tuple[float, float]]], out_path: Path) -> None:
    plt.figure(figsize=(10, 4))
    for prn, pts in sorted(series.items()):
        t = np.array([p[0] for p in pts])
        s = np.array([p[1] for p in pts])
        plt.plot(t, s, label=f"PRN {prn}", lw=1.0)
    plt.xlabel("Time (s)")
    plt.ylabel("SNR (dB)")
    plt.title("PocketSDR channel SNR")
    plt.grid(True, alpha=0.3)
    if len(series) <= 8:
        plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def rms_db(values_db: np.ndarray) -> float:
    if values_db.size == 0:
        return float("nan")
    lin = 10.0 ** (values_db / 10.0)
    rms_lin = np.sqrt(np.mean(lin ** 2))
    return 10.0 * np.log10(rms_lin)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot SNR time series from pocket_trk log")
    ap.add_argument("log", type=Path, help="pocket_trk log file (with $CH lines)")
    ap.add_argument("--sig", help="filter signal ID (e.g., AFSD)")
    ap.add_argument("--prn", type=int, help="filter PRN")
    ap.add_argument("--out", type=Path, default=Path("HALO/snr_vs_time.png"), help="output plot path")
    args = ap.parse_args()

    series = parse_log(args.log, args.sig, args.prn)
    if not series:
        raise SystemExit("No matching $CH entries found")

    plot_series(series, args.out)

    for prn, pts in sorted(series.items()):
        snr_db = np.array([p[1] for p in pts])
        print(f"PRN {prn}: {len(pts)} samples, RMS SNR = {rms_db(snr_db):.2f} dB")


if __name__ == "__main__":
    main()
