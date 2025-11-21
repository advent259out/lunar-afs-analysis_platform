// Simple Chebyshev ephemeris support for PocketSDR-AFS
// Optional: load polynomial segments from a text file and evaluate pos/vel.

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Load Chebyshev ephemeris file.
// Format per line:
//   PRN t0 dt N cx0 cx1 .. cxN cy0 .. cyN cz0 .. czN
// Units: t in seconds, position meters. Velocity derived from coefficients.
// Returns number of segments loaded, or 0 on failure.
int sdr_cheb_load(const char *file);

// Query sat position/velocity at time tsec (sec-of-week or any epoch in same units
// as t0/dt in the file). clk[0],clk[1] are set to 0.
// Returns 1 if found a segment and outputs pos(m), vel(m/s), else 0.
int sdr_cheb_satpos(int prn, double tsec, double pos[3], double vel[3], double clk[2]);

// Return 1 if a Chebyshev DB is loaded.
int sdr_cheb_available(void);

// Query segment window for a given time. Returns 1 and outputs t0, dt, order
// if a segment containing tsec is found for PRN; otherwise returns 0.
int sdr_cheb_get_seg(int prn, double tsec, double *t0, double *dt, int *order);

#ifdef __cplusplus
}
#endif
