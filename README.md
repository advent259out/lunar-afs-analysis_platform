# lunar-afs-analysis\_platform



The platform is modified based on  https://github.com/osqzss/PocketSDR-AFS and https://github.com/osqzss/LANS-AFS-SIM and the propogation model is quoted from https://github.com/Quent2G/High-precision-Analyser-of-Lunar-Orbits



The code for the analysis in the thesis is in ~\\PocketSDR-AFS-main\\PocketSDR-AFS-main\\python.



**Major Upgrades Since LANS-AFS-SIM**



1. HALO Track \& Chebyshev Support



Added CSV loader for HALO truth tracks and Chebyshev ephemeris handling (afs\_sim.c:94-420, 372-430, 1617-1668).

New CLI options: -halo, -cheb <file>, -chebgen <out>, -chebdt, -chebN let you ingest truth trajectories (propogated by HALO), propagate via Chebyshev segments, or generate new Chebyshev tables for PocketSDR.

2\. Enhanced CLI \& Logging



Expanded afs\_sim usage (afs\_sim.c (lines 1314-1505)): receiver motion (-vel, -acc), logging hooks (-truth, -rnglog, -orbitlog), elevation mask and CN0 overrides (-elvmask, -cn0), multipath modeling (-mp a,b,c), channel emulator (-chan), and debug switches (-dbgprn, -dbghalo, -dbgcmp).

Truth samples and pseudorange geometry now stream to optional logs (afs\_sim.c (lines 1794-2060)).

3\. Channel Emulation \& Multipath



Introduced per-sample interpolation of carrier/code rates and gain to mimic live channel dynamics (afs\_sim.c (lines 2143-2265)) when -chan is enabled.

Added multipath jitter model tied to elevation (afs\_sim.c (lines 1760-1776)) with tunable sigma curve via -mp.

Receiver Motion \& Acceleration



User can simulate moving receivers (NEU velocity and optional acceleration ramp) through new CLI and propagation logic (afs\_sim.c (lines 1370-1410), 1790-1885).

Documentation \& Outputs



Generated files such as truth\_track\*.csv, sim\_rng.csv, cheb\*.txt, and per-run logs are now first-class outputs.



**Major Upgrades Since PocketSDR-AFS**



1. Corresponded updates to adapt different ephemeris in decoding and postioning used in **LANS-AFS-SIM , can use -help or just refer to the code.**



**2.      Support for long-time coherent integration for pilot channel**

