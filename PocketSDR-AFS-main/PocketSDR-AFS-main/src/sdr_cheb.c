// Simple Chebyshev ephemeris support for PocketSDR-AFS

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "pocket_sdr.h"
#include "sdr_cheb.h"

#define CHEB_MAX_SAT  64
#define CHEB_MAX_SEG  128

typedef struct {
    int prn;
    double t0;    // segment start time (s)
    double dt;    // segment length (s)
    int n;        // order (highest degree)
    double *cx;   // (n+1)
    double *cy;   // (n+1)
    double *cz;   // (n+1)
} cheb_seg_t;

typedef struct {
    int prn;
    int nseg;
    cheb_seg_t *seg[CHEB_MAX_SEG];
} cheb_sat_t;

static cheb_sat_t cheb_db[CHEB_MAX_SAT];
static int cheb_loaded = 0;

static void cheb_free(void)
{
    for (int i=0;i<CHEB_MAX_SAT;i++) {
        for (int j=0;j<cheb_db[i].nseg;j++) {
            cheb_seg_t *s = cheb_db[i].seg[j];
            if (!s) continue;
            free(s->cx); free(s->cy); free(s->cz); free(s);
            cheb_db[i].seg[j] = NULL;
        }
        cheb_db[i].nseg = 0; cheb_db[i].prn = 0;
    }
    cheb_loaded = 0;
}

static cheb_sat_t *get_sat_slot(int prn)
{
    if (prn <= 0 || prn > CHEB_MAX_SAT) return NULL;
    cheb_sat_t *s = &cheb_db[prn-1];
    if (s->prn == 0) s->prn = prn;
    return s;
}

static int parse_line(char *line, cheb_seg_t **out)
{
    int prn, N;
    double t0, dt;
    // First four tokens
    char *p = line;
    if (sscanf(p, "%d %lf %lf %d", &prn, &t0, &dt, &N) != 4) return 0;
    // Move pointer past first four tokens
    int tok = 0; while (*p && tok < 4) { if (*p==' '||*p=='\t'||*p==',') { tok++; while (*p==' '||*p=='\t'||*p==',') p++; } else p++; }
    // allocate segment
    cheb_seg_t *seg = (cheb_seg_t*)calloc(1, sizeof(cheb_seg_t));
    if (!seg) return 0;
    seg->prn = prn; seg->t0 = t0; seg->dt = dt; seg->n = N;
    int M = N+1;
    seg->cx = (double*)calloc(M, sizeof(double));
    seg->cy = (double*)calloc(M, sizeof(double));
    seg->cz = (double*)calloc(M, sizeof(double));
    if (!seg->cx || !seg->cy || !seg->cz) { free(seg->cx); free(seg->cy); free(seg->cz); free(seg); return 0; }

    // expect 3*(N+1) coeffs
    for (int i=0;i<M;i++) { if (sscanf(p, "%lf", &seg->cx[i])!=1) { free(seg->cx); free(seg->cy); free(seg->cz); free(seg); return 0; }
        while (*p && *p!=' ' && *p!='\t' && *p!=',') p++; while (*p==' '||*p=='\t'||*p==',') p++; }
    for (int i=0;i<M;i++) { if (sscanf(p, "%lf", &seg->cy[i])!=1) { free(seg->cx); free(seg->cy); free(seg->cz); free(seg); return 0; }
        while (*p && *p!=' ' && *p!='\t' && *p!=',') p++; while (*p==' '||*p=='\t'||*p==',') p++; }
    for (int i=0;i<M;i++) { if (sscanf(p, "%lf", &seg->cz[i])!=1) { free(seg->cx); free(seg->cy); free(seg->cz); free(seg); return 0; }
        while (*p && *p!=' ' && *p!='\t' && *p!=',') p++; while (*p==' '||*p=='\t'||*p==',') p++; }

    *out = seg; return 1;
}

int sdr_cheb_load(const char *file)
{
    FILE *fp = fopen(file, "rt");
    if (!fp) return 0;
    cheb_free();
    char line[16384];
    int nseg_total = 0;
    while (fgets(line, sizeof(line), fp)) {
        if (line[0]=='#' || strlen(line)<6) continue;
        cheb_seg_t *seg = NULL;
        if (!parse_line(line, &seg)) continue;
        cheb_sat_t *sat = get_sat_slot(seg->prn);
        if (!sat || sat->nseg >= CHEB_MAX_SEG) { free(seg->cx); free(seg->cy); free(seg->cz); free(seg); continue; }
        sat->seg[sat->nseg++] = seg; nseg_total++;
    }
    fclose(fp);
    cheb_loaded = (nseg_total > 0);
    return nseg_total;
}

int sdr_cheb_available(void) { return cheb_loaded; }

// Evaluate position and velocity using Clenshaw and derivative via U polynomials
static void cheb_eval_seg(const cheb_seg_t *seg, double t, double *pos, double *vel)
{
    // Map to [-1,1]
    double tau = 2.0*(t - seg->t0)/seg->dt - 1.0;
    if (tau < -1.0) tau = -1.0; if (tau > 1.0) tau = 1.0;
    int N = seg->n;
    // Clenshaw for T_n
    auto eval_series = [&](const double *c){
        double bkp1=0.0,bkp2=0.0; // Clenshaw backward
        for (int k=N; k>=1; k--) { double bk = 2.0*tau*bkp1 - bkp2 + c[k]; bkp2=bkp1; bkp1=bk; }
        double f = tau*bkp1 - bkp2 + c[0];
        // derivative using U_{k-1}
        double d=0.0; // sum k c_k U_{k-1}(tau)
        double ukm1=1.0; // U_0
        double uk=2.0*tau; // U_1
        for (int k=1;k<=N;k++) {
            double Ukm1 = (k==1)?1.0:ukm1; // U_{k-1}
            d += k * c[k] * Ukm1;
            double ukp1 = 2.0*tau*uk - ukm1; ukm1 = uk; uk = ukp1; // iterate
        }
        double dfdt = (2.0/seg->dt) * d; // chain rule dtau/dt = 2/dt
        return std::pair<double,double>(f, dfdt);
    };
    // C doesn't have lambdas; reimplement without lambdas
}

// C implementation of series eval
static void series_eval(const double *c, int N, double tau, double dt, double *f, double *dfdt)
{
    double bkp1=0.0,bkp2=0.0;
    for (int k=N; k>=1; k--) {
        double bk = 2.0*tau*bkp1 - bkp2 + c[k];
        bkp2=bkp1; bkp1=bk;
    }
    *f = tau*bkp1 - bkp2 + c[0];
    // derivative sum k c_k U_{k-1}(tau)
    double dsum = 0.0;
    double Ukm2 = 0.0; // U_{-1}=0
    double Ukm1 = 1.0; // U_0=1
    for (int k=1;k<=N;k++) {
        dsum += k * c[k] * Ukm1;
        double Uk = 2.0*tau*Ukm1 - Ukm2;
        Ukm2 = Ukm1; Ukm1 = Uk;
    }
    *dfdt = (2.0/dt) * dsum;
}

static int find_seg(const cheb_sat_t *sat, double tsec)
{
    for (int i=0;i<sat->nseg;i++) {
        const cheb_seg_t *s = sat->seg[i];
        if (tsec >= s->t0 && tsec <= s->t0 + s->dt) return i;
    }
    return -1;
}

int sdr_cheb_satpos(int prn, double tsec, double pos[3], double vel[3], double clk[2])
{
    if (!cheb_loaded) return 0;
    if (prn <= 0 || prn > CHEB_MAX_SAT) return 0;
    cheb_sat_t *sat = &cheb_db[prn-1];
    if (sat->nseg <= 0) return 0;
    int idx = find_seg(sat, tsec);
    if (idx < 0) return 0;
    cheb_seg_t *s = sat->seg[idx];
    double tau = 2.0*(tsec - s->t0)/s->dt - 1.0; if (tau < -1.0) tau=-1.0; if (tau>1.0) tau=1.0;
    double fx, dfx, fy, dfy, fz, dfz;
    series_eval(s->cx, s->n, tau, s->dt, &fx, &dfx);
    series_eval(s->cy, s->n, tau, s->dt, &fy, &dfy);
    series_eval(s->cz, s->n, tau, s->dt, &fz, &dfz);
    pos[0]=fx; pos[1]=fy; pos[2]=fz;
    vel[0]=dfx; vel[1]=dfy; vel[2]=dfz;
    if (clk) { clk[0]=0.0; clk[1]=0.0; }
    return 1;
}

int sdr_cheb_get_seg(int prn, double tsec, double *t0, double *dt, int *order)
{
    if (!cheb_loaded) return 0;
    if (prn <= 0 || prn > CHEB_MAX_SAT) return 0;
    cheb_sat_t *sat = &cheb_db[prn-1];
    if (sat->nseg <= 0) return 0;
    int idx = find_seg(sat, tsec);
    if (idx < 0) return 0;
    cheb_seg_t *s = sat->seg[idx];
    if (t0) *t0 = s->t0;
    if (dt) *dt = s->dt;
    if (order) *order = s->n;
    return 1;
}
