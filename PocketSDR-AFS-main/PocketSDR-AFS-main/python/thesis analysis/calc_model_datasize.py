#!/usr/bin/env python3
"""
Compute the number of parameters and nominal data size (in bits/bytes) required
to broadcast orbit models using:

  (a) Cartesian polynomials (Chebyshev-like per-axis polynomials)
  (b) Kepler + empirical RSW accelerations

The counts reflect the models used in plot_feasible_interval.py:
  - Polynomial model: independent degree-n polynomials for x, y, z plus segment
    metadata (t0, duration).
  - Keplerian model: osculating Kepler elements at reference epoch combined with
    empirical accelerations expressed as Fourier series in the RSW frame with
    harmonics up to n_kep.

For each combination, the script prints:
    * number of (real-valued) parameters
    * total bit budget using GNSS-like quantization assumptions

Defaults (can be overridden by CLI options):

  Polynomial model:
    - 3 axes × (order+1) coefficients, each quantized with 17 bits
      (similar to Chebyshev designs in LCNS-style studies)
    - Segment metadata {t0, dt}: 2 fields × 16 bits each

  Kepler + empirical model:
    - 7 main Keplerian "geometry" parameters:
        a, e, inc, RAAN, argp, mean_anom, mean_motion
      each quantized with 32 bits (GPS-like)
    - epoch (t0): 16 bits (toe-style)
    - Empirical accelerations in R,S,W:
        for each axis: 1 constant + 2 * n_kep sin/cos coefficients,
        each coefficient quantized with 16 bits.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List


@dataclass
class ModelSize:
    name: str
    params: int   # logical real-valued parameters
    bits: int     # total bits after quantization

    def size_bytes(self) -> float:
        return self.bits / 8.0


def polynomial_params(
    order: int,
    include_metadata: bool = True,
    bits_coeff: int = 17,
    bits_meta: int = 16,
) -> ModelSize:
    """
    Polynomial (Chebyshev-like) model:

        - 3 axes × (order+1) coefficients
        - Optional {t0, dt} metadata (2 values per segment)

    bits_coeff: bits per polynomial coefficient (default 17)
    bits_meta : bits per metadata field (default 16)
    """
    n_coeff = 3 * (order + 1)
    n_meta = 2 if include_metadata else 0

    params = n_coeff + n_meta
    bits = n_coeff * bits_coeff + n_meta * bits_meta

    name = f"Polynomial (n={order})"
    return ModelSize(name=name, params=params, bits=bits)


def kepler_empirical_params(
    n_kep: int,
    bits_kepler_main: int = 32,
    bits_epoch: int = 16,
    bits_empirical: int = 16,
) -> ModelSize:
    """
    Kepler + empirical RSW accelerations model.

    Parameters (logical / continuous):

      - 7 main Kepler-like geometry parameters:
          a, e, inc, RAAN, argp, mean_anom, mean_motion  -> 7 params
      - epoch (t0)                                       -> 1 param
      - Empirical accelerations in R, S, W:
          each axis has 1 + 2 * n_kep coefficients
            (constant + sin/cos pairs for k=1..n_kep)
          total empirical params = 3 * (1 + 2*n_kep)

    Quantization:

      - Each main Kepler parameter uses bits_kepler_main bits (default: 32)
      - Epoch uses bits_epoch bits (default: 16)
      - Each empirical coefficient uses bits_empirical bits (default: 16)
    """
    n_kepler_main = 7   # a, e, inc, raan, argp, mean_anom, mean_motion
    n_epoch = 1

    n_empirical = 3 * (1 + 2 * n_kep)  # R,S,W each: 1 + 2*n_kep

    params = n_kepler_main + n_epoch + n_empirical

    bits_kepler = n_kepler_main * bits_kepler_main + n_epoch * bits_epoch
    bits_emp = n_empirical * bits_empirical
    bits = bits_kepler + bits_emp

    name = f"Kepler + RSW empirical (n_kep={n_kep})"
    return ModelSize(name=name, params=params, bits=bits)


def format_bytes(num_bytes: float) -> str:
    if num_bytes < 1024:
        return f"{num_bytes:.1f} B"
    num_kib = num_bytes / 1024.0
    if num_kib < 1024:
        return f"{num_kib:.2f} KiB"
    num_mib = num_kib / 1024.0
    return f"{num_mib:.2f} MiB"


def main():
    parser = argparse.ArgumentParser(
        description="Calculate broadcast data size requirements (bits/bytes) "
                    "for polynomial vs Kepler+RSW-empirical orbit models."
    )
    parser.add_argument(
        "--poly-orders", type=int, nargs="+", default=[5, 7, 9, 11],
        help="Polynomial degrees to evaluate (default: 5 7 9 11).",
    )
    parser.add_argument(
        "--kep-orders", type=int, nargs="+", default=[1, 3],
        help="n_kep harmonics to evaluate for Kepler model (default: 1 3).",
    )
    # Quantization options
    parser.add_argument(
        "--poly-coeff-bits", type=int, default=17,
        help="Bits per polynomial coefficient (default: 17).",
    )
    parser.add_argument(
        "--poly-meta-bits", type=int, default=16,
        help="Bits per polynomial metadata field (default: 16).",
    )
    parser.add_argument(
        "--kep-main-bits", type=int, default=32,
        help="Bits per main Kepler parameter (default: 32).",
    )
    parser.add_argument(
        "--kep-epoch-bits", type=int, default=16,
        help="Bits for Kepler epoch (t0) parameter (default: 16).",
    )
    parser.add_argument(
        "--kep-emp-bits", type=int, default=16,
        help="Bits per empirical RSW coefficient (default: 16).",
    )

    args = parser.parse_args()

    models: List[ModelSize] = []

    # Polynomial (Chebyshev-like) models
    for n in args.poly_orders:
        models.append(
            polynomial_params(
                order=n,
                include_metadata=True,
                bits_coeff=args.poly_coeff_bits,
                bits_meta=args.poly_meta_bits,
            )
        )

    # Kepler + empirical models
    for nk in args.kep_orders:
        models.append(
            kepler_empirical_params(
                n_kep=nk,
                bits_kepler_main=args.kep_main_bits,
                bits_epoch=args.kep_epoch_bits,
                bits_empirical=args.kep_emp_bits,
            )
        )

    print("Broadcast model size comparison (GNSS-like quantization):")
    print(f"{'Model':40s} {'Params':>8s} {'Bits':>10s} {'Bytes':>12s}")
    print("-" * 76)
    for model in models:
        bytes_ = model.size_bytes()
        print(
            f"{model.name:40s} "
            f"{model.params:8d} "
            f"{model.bits:10d} "
            f"{format_bytes(bytes_):>12s}"
        )


if __name__ == "__main__":
    main()
