"""Compare estimated positions in PocketSDR logs with a truth trajectory.

The script parses lines containing "$POS" from a PocketSDR log file and truth
positions from another file, aligns the epochs, and plots how far each solution
moves away from the initial truth position. The bottom pane shows the residual
(estimated minus truth).

Supported truth formats:
* PocketSDR log with `$POS` entries.
* CSV or whitespace separated text with columns:
  time_seconds, latitude_or_x, longitude_or_y, height_or_z.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


@dataclass
class EpochECEF:
    time: float
    x: float
    y: float
    z: float


def lla_to_ecef(lat_deg: float, lon_deg: float, h_m: float) -> Tuple[float, float, float]:
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)

    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    x = (n + h_m) * cos_lat * cos_lon
    y = (n + h_m) * cos_lat * sin_lon
    z = (n * (1.0 - WGS84_E2) + h_m) * sin_lat
    return x, y, z


def parse_pos_file(path: Path) -> List[Tuple[float, float, float, float]]:
    entries: List[Tuple[float, float, float, float]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("$POS"):
                parts = [p.strip() for p in line.split(',')]
                if len(parts) < 11:
                    continue
                try:
                    time = float(parts[1])
                    lat = float(parts[8])
                    lon = float(parts[9])
                    hgt = float(parts[10])
                except ValueError:
                    continue
                entries.append((time, lat, lon, hgt))
                continue
            tokens = [tok for tok in line.replace(',', ' ').split() if tok]
            if len(tokens) < 4:
                continue
            try:
                time = float(tokens[0])
                c1 = float(tokens[1])
                c2 = float(tokens[2])
                c3 = float(tokens[3])
            except ValueError:
                continue
            entries.append((time, c1, c2, c3))
    if not entries:
        raise ValueError(f"No usable data found in {path}")
    entries.sort(key=lambda item: item[0])
    return entries


def detect_coordinate_system(entries: Sequence[Tuple[float, float, float, float]]) -> str:
    sample = entries[: min(len(entries), 10)]
    lat_like = [abs(item[1]) for item in sample]
    lon_like = [abs(item[2]) for item in sample]
    if any(val > 400.0 for val in lat_like + lon_like):
        return "ecef"
    if all(val <= 90.5 for val in lat_like) and all(val <= 180.5 for val in lon_like):
        return "lla"
    return "ecef"


def entries_to_ecef(entries: Sequence[Tuple[float, float, float, float]]) -> List[EpochECEF]:
    coord_type = detect_coordinate_system(entries)
    ecef_entries: List[EpochECEF] = []
    for time, a, b, c in entries:
        if coord_type == "ecef":
            ecef_entries.append(EpochECEF(time, a, b, c))
        else:
            x, y, z = lla_to_ecef(a, b, c)
            ecef_entries.append(EpochECEF(time, x, y, z))
    return ecef_entries


def align_epochs(
    est: Sequence[EpochECEF],
    truth: Sequence[EpochECEF],
    precision: int,
) -> Tuple[List[EpochECEF], List[EpochECEF], List[float]]:
    key_fn = lambda value: round(value, precision)

    def to_dict(series: Sequence[EpochECEF]) -> Dict[float, EpochECEF]:
        result: Dict[float, EpochECEF] = {}
        for item in series:
            key = key_fn(item.time)
            if key not in result:
                result[key] = item
            else:
                if abs(item.time - key) < abs(result[key].time - key):
                    result[key] = item
        return result

    est_dict = to_dict(est)
    truth_dict = to_dict(truth)
    common = sorted(set(est_dict) & set(truth_dict))
    if not common:
        raise ValueError("No overlapping epochs between estimated and truth data")

    est_aligned = [est_dict[key] for key in common]
    truth_aligned = [truth_dict[key] for key in common]
    times = [truth_dict[key].time for key in common]
    return est_aligned, truth_aligned, times


def displacement_series(series: Sequence[EpochECEF], origin: EpochECEF) -> List[float]:
    displacements: List[float] = []
    for item in series:
        dx = item.x - origin.x
        dy = item.y - origin.y
        dz = item.z - origin.z
        displacements.append(math.sqrt(dx * dx + dy * dy + dz * dz))
    return displacements


def build_plot(times: Sequence[float], truth: Sequence[float], est: Sequence[float], output: Path) -> None:
    residual = [e - t for e, t in zip(est, truth)]
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, constrained_layout=True)

    ax_top.plot(times, truth, label="Truth", color="tab:blue")
    ax_top.plot(times, est, label="Estimated", color="tab:orange", linestyle="--")
    ax_top.set_ylabel("Displacement from start (m)")
    ax_top.grid(True, which="both", linestyle=":", linewidth=0.5)
    ax_top.legend()

    ax_bottom.plot(times, residual, label="Estimated - Truth", color="tab:red")
    ax_bottom.axhline(0.0, color="black", linewidth=0.8, linestyle=":")
    ax_bottom.set_xlabel("Time (s)")
    ax_bottom.set_ylabel("Residual (m)")
    ax_bottom.grid(True, which="both", linestyle=":", linewidth=0.5)

    fig.suptitle("Position Change Comparison")
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare PocketSDR position solutions with truth trajectory")
    parser.add_argument("--log", type=Path, default=Path("app/pocket_trk/log.txt"), help="PocketSDR log file containing $POS entries")
    parser.add_argument("--truth", type=Path, required=True, help="Truth trajectory file")
    parser.add_argument("--output", type=Path, default=Path("position_comparison.png"), help="Output plot path")
    parser.add_argument("--time-precision", type=int, default=3, help="Decimal places when matching epochs (default: 3)")
    args = parser.parse_args()

    est_entries = parse_pos_file(args.log)
    truth_entries = parse_pos_file(args.truth)

    est_ecef = entries_to_ecef(est_entries)
    truth_ecef = entries_to_ecef(truth_entries)

    est_aligned, truth_aligned, times = align_epochs(est_ecef, truth_ecef, args.time_precision)
    origin = truth_aligned[0]
    truth_disp = displacement_series(truth_aligned, origin)
    est_disp = displacement_series(est_aligned, origin)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    build_plot(times, truth_disp, est_disp, args.output)
    print(f"Saved comparison plot with {len(times)} epochs to {args.output}")


if __name__ == "__main__":
    main()
