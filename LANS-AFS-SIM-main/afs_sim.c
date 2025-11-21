#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <stdint.h>
#include <ctype.h>
#include "afs_nav.h"
#include "afs_rand.h"

#define DEMO_L1

#ifdef DEMO_L1
#define LAMBDA (0.190293672798365)
#define CARR_FREQ (1575.42e6) // = 1.023MHz * 1540.0 = 5.115MHz * 308.0
#define I_CODE_FREQ (1.023e6) // BPSK(1)
#define CARR_TO_I_CODE (1.0/1540.0)
#define Q_CODE_FREQ (5.115e6) // BPSK(5)
#define CARR_TO_Q_CODE (1.0/308.0)
#else
#define LAMBDA (0.120300597746093)
#define CARR_FREQ (2492.028e6) // = 1.023MHz * 2436.0 = 5.115MHz * 487.2
#define I_CODE_FREQ (1.023e6) // BPSK(1)
#define CARR_TO_I_CODE (1.0/2436.0)
#define Q_CODE_FREQ (5.115e6) // BPSK(5)
#define CARR_TO_Q_CODE (1.0/487.2)
#endif

#define MAX_CHAR (512)
#define MAX_SAT (12)

#define PI 3.1415926535898
#define R2D 57.2957795131

#define GM_MOON 4.9028e12
#define R_MOON 1737.4e3

#define SECONDS_IN_WEEK 604800.0
#define SECONDS_IN_HALF_WEEK 302400.0

#define SPEED_OF_LIGHT 2.99792458e8

#define POW2_M19 1.907348632812500e-6
#define POW2_M31 4.656612873077393e-10
#define POW2_M32 2.328306436538696e-10
#define POW2_M43 1.136868377216160e-13

#define GAIN_SCALE (128)
#define CHEB_MAX_N  32
#define AFS_TOI_WRAP 500  // keep TOI monotonic across 500-frame table (matches PocketSDR decoder)

int scode[4][4] = {{1,1,1,0},{0,1,1,1},{1,0,1,1},{1,1,0,1}};
int tcode[210][1500];

int sinT[] = {
    2,   5,   8,  11,  14,  17,  20,  23,  26,  29,  32,  35,  38,  41,  44,  47,
    50,  53,  56,  59,  62,  65,  68,  71,  74,  77,  80,  83,  86,  89,  91,  94,
    97, 100, 103, 105, 108, 111, 114, 116, 119, 122, 125, 127, 130, 132, 135, 138,
    140, 143, 145, 148, 150, 153, 155, 157, 160, 162, 164, 167, 169, 171, 173, 176,
    178, 180, 182, 184, 186, 188, 190, 192, 194, 196, 198, 200, 202, 204, 205, 207,
    209, 210, 212, 214, 215, 217, 218, 220, 221, 223, 224, 225, 227, 228, 229, 230,
    232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 241, 242, 243, 244, 244, 245,
    245, 246, 247, 247, 248, 248, 248, 249, 249, 249, 249, 250, 250, 250, 250, 250,
    250, 250, 250, 250, 250, 249, 249, 249, 249, 248, 248, 248, 247, 247, 246, 245,
    245, 244, 244, 243, 242, 241, 241, 240, 239, 238, 237, 236, 235, 234, 233, 232,
    230, 229, 228, 227, 225, 224, 223, 221, 220, 218, 217, 215, 214, 212, 210, 209,
    207, 205, 204, 202, 200, 198, 196, 194, 192, 190, 188, 186, 184, 182, 180, 178,
    176, 173, 171, 169, 167, 164, 162, 160, 157, 155, 153, 150, 148, 145, 143, 140,
    138, 135, 132, 130, 127, 125, 122, 119, 116, 114, 111, 108, 105, 103, 100,  97,
    94,  91,  89,  86,  83,  80,  77,  74,  71,  68,  65,  62,  59,  56,  53,  50,
    47,  44,  41,  38,  35,  32,  29,  26,  23,  20,  17,  14,  11,   8,   5,   2,
    -2,  -5,  -8, -11, -14, -17, -20, -23, -26, -29, -32, -35, -38, -41, -44, -47,
    -50, -53, -56, -59, -62, -65, -68, -71, -74, -77, -80, -83, -86, -89, -91, -94,
    -97,-100,-103,-105,-108,-111,-114,-116,-119,-122,-125,-127,-130,-132,-135,-138,
    -140,-143,-145,-148,-150,-153,-155,-157,-160,-162,-164,-167,-169,-171,-173,-176,
    -178,-180,-182,-184,-186,-188,-190,-192,-194,-196,-198,-200,-202,-204,-205,-207,
    -209,-210,-212,-214,-215,-217,-218,-220,-221,-223,-224,-225,-227,-228,-229,-230,
    -232,-233,-234,-235,-236,-237,-238,-239,-240,-241,-241,-242,-243,-244,-244,-245,
    -245,-246,-247,-247,-248,-248,-248,-249,-249,-249,-249,-250,-250,-250,-250,-250,
    -250,-250,-250,-250,-250,-249,-249,-249,-249,-248,-248,-248,-247,-247,-246,-245,
    -245,-244,-244,-243,-242,-241,-241,-240,-239,-238,-237,-236,-235,-234,-233,-232,
    -230,-229,-228,-227,-225,-224,-223,-221,-220,-218,-217,-215,-214,-212,-210,-209,
    -207,-205,-204,-202,-200,-198,-196,-194,-192,-190,-188,-186,-184,-182,-180,-178,
    -176,-173,-171,-169,-167,-164,-162,-160,-157,-155,-153,-150,-148,-145,-143,-140,
    -138,-135,-132,-130,-127,-125,-122,-119,-116,-114,-111,-108,-105,-103,-100, -97,
    -94, -91, -89, -86, -83, -80, -77, -74, -71, -68, -65, -62, -59, -56, -53, -50,
    -47, -44, -41, -38, -35, -32, -29, -26, -23, -20, -17, -14, -11,  -8,  -5,  -2
};

// ------------------------
// HALO track support (CSV)
// ------------------------
// Optional: supply external J2000 Moon-centered tracks from HALO to override
// Kepler propagation. Use -halo <csv> to enable.

typedef struct {
    int n;            // number of samples
    double *t;        // time [s]
    double (*pos)[3]; // position [m]
    double (*vel)[3]; // velocity [m/s]
} track_t;

typedef struct {
    int prn;     // PRN id (1-based)
    track_t tr;  // track samples
    int valid;   // 1 if loaded and has >=2 samples
} halo_trk_t;

static int use_halo_tracks = 0;
static int use_cheb = 0;               // use Chebyshev for satpos
static char cheb_file[MAX_CHAR] = {0};
static int do_cheb_gen = 0;            // export Chebyshev file
static char cheb_gen_file[MAX_CHAR] = {0};
static double cheb_dt = 90.0;          // segment length (s)
static int cheb_N = 11;                // order

static int use_multipath = 0;
static double mp_a = 0.0, mp_b = 0.0, mp_c = 0.0;

static double multipath_offset(double elevation_rad)
{
    if (!use_multipath) return 0.0;
    double phi_deg = elevation_rad * R2D;
    double sigma = mp_a + mp_b * exp(mp_c * phi_deg);
    if (sigma <= 0.0) return 0.0;
    double gauss = (double)randn() / 1250.0; // randn() sigma ≈ 1250
    return sigma * gauss;
}

// Multipath model parameters -------------------------------------------------
static char halo_track_file[MAX_CHAR] = {0}; // supports comma/semicolon-separated list
static halo_trk_t halo_db[MAX_SAT] = {0};
static double halo_tmin = 0.0, halo_tmax = 0.0;

static void halo_free(void)
{
    for (int i = 0; i < MAX_SAT; i++) {
        if (halo_db[i].tr.t) free(halo_db[i].tr.t);
        if (halo_db[i].tr.pos) free(halo_db[i].tr.pos);
        if (halo_db[i].tr.vel) free(halo_db[i].tr.vel);
        memset(&halo_db[i], 0, sizeof(halo_db[i]));
    }
}

static void halo_reset(void)
{
    halo_free();
    halo_tmin = 0.0;
    halo_tmax = 0.0;
}

static int halo_prn_index(int prn)
{
    if (prn <= 0 || prn > MAX_SAT) return -1;
    return prn - 1;
}

static int halo_append(track_t *tr, double ts, const double p[3], const double v[3])
{
    int newn = tr->n + 1;
    int grow = (tr->n == 0) || ((tr->n & 1023) == 0);
    if (grow) {
        int cap = ((newn + 1023) / 1024) * 1024;
        double *nt = (double*)realloc(tr->t, sizeof(double) * cap);
        double (*np)[3] = (double (*)[3])realloc(tr->pos, sizeof(double) * 3 * cap);
        double (*nv)[3] = (double (*)[3])realloc(tr->vel, sizeof(double) * 3 * cap);
        if (!nt || !np || !nv) return 0;
        tr->t = nt; tr->pos = np; tr->vel = nv;
    }
    tr->t[tr->n] = ts;
    tr->pos[tr->n][0] = p[0]; tr->pos[tr->n][1] = p[1]; tr->pos[tr->n][2] = p[2];
    tr->vel[tr->n][0] = v[0]; tr->vel[tr->n][1] = v[1]; tr->vel[tr->n][2] = v[2];
    tr->n = newn;
    return 1;
}

static void halo_sort(track_t *tr)
{
    for (int i = 1; i < tr->n; i++) {
        double ti = tr->t[i];
        double pi[3] = { tr->pos[i][0], tr->pos[i][1], tr->pos[i][2] };
        double vi[3] = { tr->vel[i][0], tr->vel[i][1], tr->vel[i][2] };
        int j = i - 1;
        while (j >= 0 && tr->t[j] > ti) {
            tr->t[j+1] = tr->t[j];
            tr->pos[j+1][0] = tr->pos[j][0]; tr->pos[j+1][1] = tr->pos[j][1]; tr->pos[j+1][2] = tr->pos[j][2];
            tr->vel[j+1][0] = tr->vel[j][0]; tr->vel[j+1][1] = tr->vel[j][1]; tr->vel[j+1][2] = tr->vel[j][2];
            j--;
        }
        tr->t[j+1] = ti;
        tr->pos[j+1][0] = pi[0]; tr->pos[j+1][1] = pi[1]; tr->pos[j+1][2] = pi[2];
        tr->vel[j+1][0] = vi[0]; tr->vel[j+1][1] = vi[1]; tr->vel[j+1][2] = vi[2];
    }
}

// Append a HALO CSV into current tracks (no reset)
static int halo_add_csv(const char *fname)
{
    FILE *fp = fopen(fname, "rt");
    if (!fp) return 0;
    char line[1024];
    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == '#' || strlen(line) < 3) continue;
        int prn; double t, xkm, ykm, zkm, vxkms, vykms, vzkms;
        int n = sscanf(line, "%d , %lf , %lf , %lf , %lf , %lf , %lf , %lf",
                       &prn, &t, &xkm, &ykm, &zkm, &vxkms, &vykms, &vzkms);
        if (n != 8) n = sscanf(line, "%d %lf %lf %lf %lf %lf %lf %lf",
                                &prn, &t, &xkm, &ykm, &zkm, &vxkms, &vykms, &vzkms);
        if (n != 8) continue;
        int idx = halo_prn_index(prn);
        if (idx < 0) continue;
        halo_db[idx].prn = prn;
        double p[3] = { xkm*1000.0, ykm*1000.0, zkm*1000.0 };
        double v[3] = { vxkms*1000.0, vykms*1000.0, vzkms*1000.0 };
        if (halo_tmin == 0.0 || t < halo_tmin) halo_tmin = t;
        if (t > halo_tmax) halo_tmax = t;
        if (!halo_append(&halo_db[idx].tr, t, p, v)) { fclose(fp); return 0; }
        halo_db[idx].valid = 1;
    }
    fclose(fp);
    return 1;
}

// Expand a spec possibly containing a numeric range like prefix1-8suffix and add all files
static int halo_add_from_spec(const char *spec)
{
    if (!spec || !*spec) return 0;
    const char *dash = spec;
    while ((dash = strchr(dash, '-')) != NULL) {
        const char *l = dash - 1;
        const char *r = dash + 1;
        if (l < spec || !isdigit((unsigned char)*l) || !isdigit((unsigned char)*r)) { dash++; continue; }
        while (l > spec && isdigit((unsigned char)*(l-1))) l--; // leftmost digit of left number
        const char *r2 = r;
        while (isdigit((unsigned char)*r2)) r2++; // one past right number

        int width = (int)(dash - l);
        long start = 0, end = 0;
        for (const char *p = l; p < dash; p++) start = start * 10 + (*p - '0');
        for (const char *p = r; p < r2; p++) end = end * 10 + (*p - '0');

        char prefix[MAX_CHAR], suffix[MAX_CHAR];
        size_t prelen = (size_t)(l - spec);
        size_t suflen = strlen(spec) - (size_t)(r2 - spec);
        if (prelen >= sizeof(prefix)) prelen = sizeof(prefix) - 1;
        if (suflen >= sizeof(suffix)) suflen = sizeof(suffix) - 1;
        memcpy(prefix, spec, prelen); prefix[prelen] = '\0';
        memcpy(suffix, r2, suflen); suffix[suflen] = '\0';

        int loaded = 0;
        int step = (start <= end) ? 1 : -1;
        for (long v = start; (step > 0) ? (v <= end) : (v >= end); v += step) {
            char num[32];
            snprintf(num, sizeof(num), "%0*ld", width, v);
            char fname[MAX_CHAR];
            snprintf(fname, sizeof(fname), "%s%s%s", prefix, num, suffix);
            if (halo_add_csv(fname)) loaded = 1;
        }
        return loaded;
    }
    // no range pattern, treat as single file
    return halo_add_csv(spec);
}

static int halo_interp(const track_t *tr, double tq, double pos[3], double vel[3])
{
    if (tr->n < 2) return 0;
    if (tq <= tr->t[0]) {
        pos[0]=tr->pos[0][0]; pos[1]=tr->pos[0][1]; pos[2]=tr->pos[0][2];
        vel[0]=tr->vel[0][0]; vel[1]=tr->vel[0][1]; vel[2]=tr->vel[0][2];
        return 1;
    }
    if (tq >= tr->t[tr->n-1]) {
        pos[0]=tr->pos[tr->n-1][0]; pos[1]=tr->pos[tr->n-1][1]; pos[2]=tr->pos[tr->n-1][2];
        vel[0]=tr->vel[tr->n-1][0]; vel[1]=tr->vel[tr->n-1][1]; vel[2]=tr->vel[tr->n-1][2];
        return 1;
    }
    int lo=0, hi=tr->n-1;
    while (hi-lo>1) { int mid=(lo+hi)/2; if (tr->t[mid]<=tq) lo=mid; else hi=mid; }
    double a=(tq-tr->t[lo])/(tr->t[hi]-tr->t[lo]);
    for (int k=0;k<3;k++) {
        pos[k]=tr->pos[lo][k]*(1.0-a)+tr->pos[hi][k]*a;
        vel[k]=tr->vel[lo][k]*(1.0-a)+tr->vel[hi][k]*a;
    }
    return 1;
}

static int satpos_halo(int prn, double tsec, double *pos, double *vel, double *clk)
{
    int idx = halo_prn_index(prn);
    if (idx < 0 || !halo_db[idx].valid) return 0;
    /* If track has (approximately) uniform sampling, quantize query time to track grid
       so we directly use CSV samples instead of interpolating. */
    const track_t *tr = &halo_db[idx].tr;
    if (tr->n >= 2) {
        double dt = tr->t[1] - tr->t[0];
        if (dt > 0.0) {
            double n = floor(((tsec - tr->t[0]) / dt) + 0.5); /* nearest sample index */
            tsec = tr->t[0] + n * dt;
        }
    }
    if (!halo_interp(tr, tsec, pos, vel)) return 0;
    clk[0] = 0.0; clk[1] = 0.0; // no clock correction
    return 1;
}

// ------------------------
// Chebyshev ephemeris (simple local loader/evaluator)
// ------------------------
typedef struct {
    int prn;
    double t0, dt; // time span [t0,t0+dt]
    int n;         // order
    double *cx, *cy, *cz; // (n+1)
} cheb_seg_t;

typedef struct {
    int prn;
    int nseg;
    cheb_seg_t **seg;
} cheb_sat_t;

static cheb_sat_t cheb_db_local[MAX_SAT] = {0};

static void cheb_db_free_local(void)
{
    for (int i=0;i<MAX_SAT;i++) {
        cheb_sat_t *sat = &cheb_db_local[i];
        if (!sat->seg) continue;
        for (int j=0;j<sat->nseg;j++) {
            cheb_seg_t *s = sat->seg[j];
            if (!s) continue;
            free(s->cx); free(s->cy); free(s->cz); free(s);
        }
        free(sat->seg); sat->seg=NULL; sat->nseg=0; sat->prn=0;
    }
}

static cheb_sat_t *cheb_get_sat_local(int prn)
{
    if (prn<=0 || prn>MAX_SAT) return NULL;
    cheb_sat_t *sat = &cheb_db_local[prn-1];
    if (sat->prn == 0) sat->prn = prn;
    return sat;
}

static int cheb_parse_line_local(char *line, cheb_seg_t **out)
{
    int prn, N; double t0, dt; char *p=line;
    if (sscanf(p, "%d %lf %lf %d", &prn, &t0, &dt, &N) != 4) return 0;
    int tok=0; while (*p && tok<4) { if (*p==' '||*p=='\t'||*p==',') { tok++; while (*p==' '||*p=='\t'||*p==',') p++; } else p++; }
    cheb_seg_t *s = (cheb_seg_t*)calloc(1,sizeof(cheb_seg_t)); if (!s) return 0;
    s->prn=prn; s->t0=t0; s->dt=dt; s->n=N; int M=N+1;
    s->cx=(double*)calloc(M,sizeof(double)); s->cy=(double*)calloc(M,sizeof(double)); s->cz=(double*)calloc(M,sizeof(double));
    if (!s->cx||!s->cy||!s->cz) { free(s->cx); free(s->cy); free(s->cz); free(s); return 0; }
    for (int i=0;i<M;i++){ if (sscanf(p, "%lf", &s->cx[i])!=1){ free(s->cx); free(s->cy); free(s->cz); free(s); return 0;} while(*p&&*p!=' '&&*p!='\t'&&*p!=',')p++; while(*p==' '||*p=='\t'||*p==',')p++; }
    for (int i=0;i<M;i++){ if (sscanf(p, "%lf", &s->cy[i])!=1){ free(s->cx); free(s->cy); free(s->cz); free(s); return 0;} while(*p&&*p!=' '&&*p!='\t'&&*p!=',')p++; while(*p==' '||*p=='\t'||*p==',')p++; }
    for (int i=0;i<M;i++){ if (sscanf(p, "%lf", &s->cz[i])!=1){ free(s->cx); free(s->cy); free(s->cz); free(s); return 0;} while(*p&&*p!=' '&&*p!='\t'&&*p!=',')p++; while(*p==' '||*p=='\t'||*p==',')p++; }
    *out = s; return 1;
}

static void cheb_series_eval_local(const double *c, int N, double tau, double dt, double *f, double *dfdt)
{
    double b1=0.0, b2=0.0;
    for (int k=N; k>=1; k--) { double b = 2.0*tau*b1 - b2 + c[k]; b2=b1; b1=b; }
    *f = tau*b1 - b2 + c[0];
    double Ukm2=0.0, Ukm1=1.0, dsum=0.0;
    for (int k=1;k<=N;k++) { dsum += k * c[k] * Ukm1; double Uk = 2.0*tau*Ukm1 - Ukm2; Ukm2=Ukm1; Ukm1=Uk; }
    *dfdt = (2.0/dt) * dsum;
}

int cheb_load(const char *fname)
{
    FILE *fp = fopen(fname, "rt"); if (!fp) return 0;
    cheb_db_free_local();
    char line[16384]; int nseg=0;
    while (fgets(line, sizeof(line), fp)) {
        if (line[0]=='#' || strlen(line)<6) continue;
        cheb_seg_t *seg=NULL; if (!cheb_parse_line_local(line, &seg)) continue;
        cheb_sat_t *sat = cheb_get_sat_local(seg->prn); if (!sat) { free(seg->cx); free(seg->cy); free(seg->cz); free(seg); continue; }
        sat->seg = (cheb_seg_t**)realloc(sat->seg, sizeof(cheb_seg_t*)*(sat->nseg+1));
        sat->seg[sat->nseg++] = seg; nseg++;
    }
    fclose(fp);
    return nseg;
}

int satpos_cheb(int prn, double tsec, double *pos, double *vel, double *clk)
{
    if (prn<=0 || prn>MAX_SAT) return 0;
    cheb_sat_t *sat = &cheb_db_local[prn-1]; if (sat->nseg<=0) return 0;
    for (int i=0;i<sat->nseg;i++) {
        cheb_seg_t *s = sat->seg[i];
        if (tsec < s->t0 || tsec > s->t0 + s->dt) continue;
        double tau = 2.0*(tsec - s->t0)/s->dt - 1.0; if (tau<-1.0)tau=-1.0; if (tau>1.0)tau=1.0;
        double fx,dfx,fy,dfy,fz,dfz;
        cheb_series_eval_local(s->cx,s->n,tau,s->dt,&fx,&dfx);
        cheb_series_eval_local(s->cy,s->n,tau,s->dt,&fy,&dfy);
        cheb_series_eval_local(s->cz,s->n,tau,s->dt,&fz,&dfz);
        pos[0]=fx; pos[1]=fy; pos[2]=fz; vel[0]=dfx; vel[1]=dfy; vel[2]=dfz; if (clk) { clk[0]=0.0; clk[1]=0.0; }
        return 1;
    }
    return 0;
}

int cosT[] = {
    250, 250, 250, 250, 250, 249, 249, 249, 249, 248, 248, 248, 247, 247, 246, 245,
    245, 244, 244, 243, 242, 241, 241, 240, 239, 238, 237, 236, 235, 234, 233, 232,
    230, 229, 228, 227, 225, 224, 223, 221, 220, 218, 217, 215, 214, 212, 210, 209,
    207, 205, 204, 202, 200, 198, 196, 194, 192, 190, 188, 186, 184, 182, 180, 178,
    176, 173, 171, 169, 167, 164, 162, 160, 157, 155, 153, 150, 148, 145, 143, 140,
    138, 135, 132, 130, 127, 125, 122, 119, 116, 114, 111, 108, 105, 103, 100,  97,
    94,  91,  89,  86,  83,  80,  77,  74,  71,  68,  65,  62,  59,  56,  53,  50,
    47,  44,  41,  38,  35,  32,  29,  26,  23,  20,  17,  14,  11,   8,   5,   2,
    -2,  -5,  -8, -11, -14, -17, -20, -23, -26, -29, -32, -35, -38, -41, -44, -47,
    -50, -53, -56, -59, -62, -65, -68, -71, -74, -77, -80, -83, -86, -89, -91, -94,
    -97,-100,-103,-105,-108,-111,-114,-116,-119,-122,-125,-127,-130,-132,-135,-138,
    -140,-143,-145,-148,-150,-153,-155,-157,-160,-162,-164,-167,-169,-171,-173,-176,
    -178,-180,-182,-184,-186,-188,-190,-192,-194,-196,-198,-200,-202,-204,-205,-207,
    -209,-210,-212,-214,-215,-217,-218,-220,-221,-223,-224,-225,-227,-228,-229,-230,
    -232,-233,-234,-235,-236,-237,-238,-239,-240,-241,-241,-242,-243,-244,-244,-245,
    -245,-246,-247,-247,-248,-248,-248,-249,-249,-249,-249,-250,-250,-250,-250,-250,
    -250,-250,-250,-250,-250,-249,-249,-249,-249,-248,-248,-248,-247,-247,-246,-245,
    -245,-244,-244,-243,-242,-241,-241,-240,-239,-238,-237,-236,-235,-234,-233,-232,
    -230,-229,-228,-227,-225,-224,-223,-221,-220,-218,-217,-215,-214,-212,-210,-209,
    -207,-205,-204,-202,-200,-198,-196,-194,-192,-190,-188,-186,-184,-182,-180,-178,
    -176,-173,-171,-169,-167,-164,-162,-160,-157,-155,-153,-150,-148,-145,-143,-140,
    -138,-135,-132,-130,-127,-125,-122,-119,-116,-114,-111,-108,-105,-103,-100, -97,
    -94, -91, -89, -86, -83, -80, -77, -74, -71, -68, -65, -62, -59, -56, -53, -50,
    -47, -44, -41, -38, -35, -32, -29, -26, -23, -20, -17, -14, -11,  -8,  -5,  -2,
    2,   5,   8,  11,  14,  17,  20,  23,  26,  29,  32,  35,  38,  41,  44,  47,
    50,  53,  56,  59,  62,  65,  68,  71,  74,  77,  80,  83,  86,  89,  91,  94,
    97, 100, 103, 105, 108, 111, 114, 116, 119, 122, 125, 127, 130, 132, 135, 138,
    140, 143, 145, 148, 150, 153, 155, 157, 160, 162, 164, 167, 169, 171, 173, 176,
    178, 180, 182, 184, 186, 188, 190, 192, 194, 196, 198, 200, 202, 204, 205, 207,
    209, 210, 212, 214, 215, 217, 218, 220, 221, 223, 224, 225, 227, 228, 229, 230,
    232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 241, 242, 243, 244, 244, 245,
    245, 246, 247, 247, 248, 248, 248, 249, 249, 249, 249, 250, 250, 250, 250, 250
};

typedef struct
{
    int wn;
    int itow;
    int toi;
    double fsec;
} afstime_t;

typedef struct
{
    int week;
    double sec;
} gpstime_t;

typedef struct
{
    int y;
    int m;
    int d;
    int hh;
    int mm;
    double sec;
} datetime_t;

typedef struct
{
    int vflg;

    gpstime_t toe;
    gpstime_t toc;

    double ecc;   // Eccentricity
    double sqrta; // SQRT(A) 
    double m0;    // Mean Anom
    double omg0;  // Right Ascen at Week
    double inc0;  // Orbital Inclination
    double aop;   // Argument of Perigee

    double af0;
    double af1;

    // Working variables
    double n; // Mean motion
    double A; // Semi-major axis
    double sq1e2; // sqrt(1-e^2)

} ephem_t;

// Forward declaration for Chebyshev fitting (defined later in file)
static int cheb_fit_segment(FILE *cfp, int prn, int week, const ephem_t *eph,
                            double t0, double dt, int N);

typedef struct
{
    int code[2046];
    double f_code;
    double code_phase;
    int C; // spreading code, signal level

    int ibit; // data bit counter
    int iframe; // data frame counter
    int D; // data signal level
    uint8_t data[2][6000];

} Ich_t;

typedef struct
{
    int code[10230];
    double f_code;
    double code_phase;
    int C; // spreading code, signal level
    int S; // seconday code
    int T; // tertiary code

    int ibit; // seconday code bit counter
    int ichip; // tertiary code chip counter

} Qch_t;

typedef struct
{
    int prn;
    Ich_t I;
    Qch_t Q;
    gpstime_t g0;
    double f_carr;
    double carr_phase;
    int* iq_buff;
    double azel[2];
    double multipath_bias;
    int has_multipath_bias;
} channel_t;

typedef struct
{
    gpstime_t g;
    double range;
    double rate;
} range_t;

int hex2bin(char hex, int* bin)
{
	int val;

	if (hex >= '0' && hex <= '9') {
		val = hex - '0';
	}
	else if (hex >= 'A' && hex <= 'F') {
		val = hex - 'A' + 10;
	}
	else if (hex >= 'a' && hex <= 'f') {
		val = hex - 'a' + 10;
	}
	else { // Invalid hex character
		return 0;
	}

	for (int i = 3; i >= 0; i--) {
		bin[i] = val % 2;
		val /= 2;
	}

	return 1;
}

int readTertiary(char* fname)
{
	FILE* fp;
	char str[MAX_CHAR];
	int nsv;

	if (NULL == (fp = fopen(fname, "rt"))) {
		return 0;
	}

	// Ignore the first line
	fgets(str, MAX_CHAR, fp);

	for (nsv = 0; nsv < 210; nsv++) {

		if (NULL == fgets(str, MAX_CHAR, fp))
			break;

		for (int i = 0; i < 375; i++) {
			hex2bin(str[i + 1], tcode[nsv] + i * 4); // Skip the first charactor
		}
	}

	fclose(fp);

	return nsv;
}

void icodegen(int* code, int prn)
{
    // Logic Level | Signal Level
    //      0      |      +1
    //      1      |      -1

    int delay[] = {
       1845, 1071,  170, 2035, 1214, 1292, 1284, 1894, 1537,  735, //   1 -  10
        561, 1789, 1453,  196, 1040,  326, 1787,  982, 1030, 1380, //  11 -  20
       1932, 1188,  390,  714,  303, 1001,  707, 1984,  139,  182, //  21 -  30
       1891, 1247, 1434, 2000, 1843,  865,  616,  514,  449, 1173, //  31 -  40
         24, 1383, 1940, 1594, 1765,  752,  145, 1615, 1666, 1372, //  41 -  50
       1634, 1068, 1181,  879, 1153, 1621,  927, 1848,  402,  413, //  51 -  60
       1090,  657,  609, 1547,  370,  271, 1353,  635,  299,  697, //  61 -  70
        152,  678, 1329,   15, 1974, 1884, 1868,  277,  302,    9, //  71 -  80
        603, 1583,  848, 1234, 1568,  510, 1303, 1921,  823, 1187, //  81 -  90
       1299,  824,  672, 2034, 1388,   13,  223, 1840, 1161, 1132, //  91 - 100
        365,    2,  924, 1373,  959,  220, 1542,  188,  264,  453, // 101 - 110
         68,  715,   75, 1095,  938, 1316,  394, 1156,  166,  969, // 111 - 120
        269,  179,  957,  400,  625, 1513, 1796,  100, 1660, 1454, // 121 - 130
       1613, 1064,  844,  518,  320,  661, 2031,  694, 1143, 1167, // 131 - 140
       1885,  833, 1601,  903,  399, 1896,  899,  133,  556,  331, // 141 - 150
        198,  212, 1024, 1070, 1972, 1573,  884, 1177, 1691,  533, // 151 - 160
        480,  751,  447,  734,  973,  857, 1767, 1548, 1876,  614, // 161 - 170
       1017, 1978,  275, 1141, 1252, 1952, 1714, 1067,  557,  522, // 171 - 180
       1159,  545, 1580,  610,  935, 1134,  780,  691, 1038, 1418, // 181 - 190
        295,  916, 1654,  624,  706, 1033, 1633,  790, 1451, 1300, // 191 - 200
        459,  106,  861, 1541,  114, 1381, 1945, 1069,  242,  356  // 201 - 210
    };

    static int g1[2047], g2[2047];
    int r1[11] = { 0 }, r2[11] = { 0 }, c1, c2;
    int i, j;

    for (i = 0; i < 11; i++)
        r1[i] = r2[i] = -1;

    for (i = 0; i < 2047; i++) {
        g1[i] = r1[10];
        g2[i] = r2[10];
        c1 = r1[1] * r1[10];
        c2 = r2[1] * r2[4] * r2[7] * r2[10];

        for (j = 10; j > 0; j--) {
            r1[j] = r1[j - 1];
            r2[j] = r2[j - 1];
        }
        r1[0] = c1;
        r2[0] = c2;
    }

    for (i = 0, j = 2047 - delay[prn - 1]; i < 2046; i++, j++) {
        //code[i] = (1 - g1[i] * g2[j % 2047]) / 2; // Logic level
        code[i] = g1[i] * g2[j % 2047]; // Signal level
    }

    return;
}

static char legendre[10223] = { 0 };

void gen_legendre_sequence()
{
    int i;
    for (i = 0; i < 10223; i++) {
        legendre[i] = 1;
    }

    for (i = 0; i < 10224; i++) {
        legendre[(i * i) % 10223] = -1;
    }
    legendre[0] = 1;

    return;
}

void qcodegen(int* code, int prn) // L1CP code (IS-GPS-800)
{
    // Kudos to Taro Suzuki: https://github.com/taroz/GNSS-SDRLIB/blob/master/src/sdrcode.c

    static const short weil[] = { /* Weil Index */
       5111, 5109, 5108, 5106, 5103, 5101, 5100, 5098, 5095, 5094, /*   1- 10 */
       5093, 5091, 5090, 5081, 5080, 5069, 5068, 5054, 5044, 5027, /*  11- 20 */
       5026, 5014, 5004, 4980, 4915, 4909, 4893, 4885, 4832, 4824, /*  21- 30 */
       4591, 3706, 5092, 4986, 4965, 4920, 4917, 4858, 4847, 4790, /*  31- 40 */
       4770, 4318, 4126, 3961, 3790, 4911, 4881, 4827, 4795, 4789, /*  41- 50 */
       4725, 4675, 4539, 4535, 4458, 4197, 4096, 3484, 3481, 3393, /*  51- 60 */
       3175, 2360, 1852, 5065, 5063, 5055, 5012, 4981, 4952, 4934, /*  61- 70 */
       4932, 4786, 4762, 4640, 4601, 4563, 4388, 3820, 3687, 5052, /*  71- 80 */
       5051, 5047, 5039, 5015, 5005, 4984, 4975, 4974, 4972, 4962, /*  81- 90 */
       4913, 4907, 4903, 4833, 4778, 4721, 4661, 4660, 4655, 4623, /*  91-100 */
       4590, 4548, 4461, 4442, 4347, 4259, 4256, 4166, 4155, 4109, /* 101-110 */
       4100, 4023, 3998, 3979, 3903, 3568, 5088, 5050, 5020, 4990, /* 111-120 */
       4982, 4966, 4949, 4947, 4937, 4935, 4906, 4901, 4872, 4865, /* 121-130 */
       4863, 4818, 4785, 4781, 4776, 4775, 4754, 4696, 4690, 4658, /* 131-140 */
       4607, 4599, 4596, 4530, 4524, 4451, 4441, 4396, 4340, 4335, /* 141-150 */
       4296, 4267, 4168, 4149, 4097, 4061, 3989, 3966, 3789, 3775, /* 151-160 */
       3622, 3523, 3515, 3492, 3345, 3235, 3169, 3157, 3082, 3072, /* 161-170 */
       3032, 3030, 4582, 4595, 4068, 4871, 4514, 4439, 4122, 4948, /* 171-180 */
       4774, 3923, 3411, 4745, 4195, 4897, 3047, 4185, 4354, 5077, /* 181-190 */
       4042, 2111, 4311, 5024, 4352, 4678, 5034, 5085, 3646, 4868, /* 191-200 */
       3668, 4211, 2883, 2850, 2815, 2542, 2492, 2376, 2036, 1920  /* 201-210 */
    };

    static const short insert[] = { /* Insertion Index */
        412,  161,    1,  303,  207, 4971, 4496,    5, 4557,  485, /*   1- 10 */
        253, 4676,    1,   66, 4485,  282,  193, 5211,  729, 4848, /*  11- 20 */
        982, 5955, 9805,  670,  464,   29,  429,  394,  616, 9457, /*  21- 30 */
       4429, 4771,  365, 9705, 9489, 4193, 9947,  824,  864,  347, /*  31- 40 */
        677, 6544, 6312, 9804,  278, 9461,  444, 4839, 4144, 9875, /*  41- 50 */
        197, 1156, 4674,10035, 4504,    5, 9937,  430,    5,  355, /*  51- 60 */
        909, 1622, 6284, 9429,   77,  932, 5973,  377,10000,  951, /*  61- 70 */
       6212,  686, 9352, 5999, 9912, 9620,  635, 4951, 5453, 4658, /*  71- 80 */
       4800,   59,  318,  571,  565, 9947, 4654,  148, 3929,  293, /*  81- 90 */
        178,10142, 9683,  137,  565,   35, 5949,    2, 5982,  825, /*  91-100 */
       9614, 9790, 5613,  764,  660, 4870, 4950, 4881, 1151, 9977, /* 101-110 */
       5122,10074, 4832,   77, 4698, 1002, 5549, 9606, 9228,  604, /* 111-120 */
       4678, 4854, 4122, 9471, 5026,  272, 1027,  317,  691,  509, /* 121-130 */
       9708, 5033, 9938, 4314,10140, 4790, 9823, 6093,  469, 1215, /* 131-140 */
        799,  756, 9994, 4843, 5271, 9661, 6255, 5203,  203,10070, /* 141-150 */
         30,  103, 5692,   32, 9826,   76,   59, 6831,  958, 1471, /* 151-160 */
      10070,  553, 5487,   55,  208,  645, 5268, 1873,  427,  367, /* 161-170 */
       1404, 5652,    5,  368,  451, 9595, 1030, 1324,  692, 9819, /* 171-180 */
       4520, 9911,  278,  642, 6330, 5508, 1872, 5445,10131,  422, /* 181-190 */
       4918,  787, 9864, 9753, 9859,  328,    1, 4733,  164,  135, /* 191-200 */
        174,  132,  538,  176,  198,  595,  574,  321,  596,  491  /* 201-210 */
    };

    char weilcode[10223] = { 0 };
    int i, j, ind;
    int w = weil[prn - 1], p = insert[prn - 1] - 1;
    static const char insertbit[7] = {-1, 1, 1, -1, 1, -1, -1};

    /* Generate Legendre Sequence */
    if (!legendre[0]) gen_legendre_sequence();

    for (i = 0; i < 10223; i++) {
        ind = (i + w) % 10223;
        weilcode[i] = -legendre[i] * legendre[ind];
    }

    /* Insert bits */
    for (i = 0; i < p; i++) 
        code[i] = weilcode[i];
    
    for (j = 0; j < 7; j++) 
        code[i++] = insertbit[j];
    
    for (i = p + 7; i < 10230; i++)
        code[i] = weilcode[i - 7];

    for (i = 0; i < 10230; i++) {
        //code[i] = (1 + code[i]) / 2; // Logic level
        code[i] = -1 * code[i]; // Signal level
    }

    return;
}

void timeadd(gpstime_t *g, double dt)
{
    g->sec += dt;

    if (g->sec < 0.0) {
        g->sec += SECONDS_IN_WEEK;
        g->week -= 1;
    }
    else if (g->sec >= SECONDS_IN_WEEK) {
        g->sec -= SECONDS_IN_WEEK;
        g->week += 1;
    }

    return;
}

void gpst2afst(const gpstime_t *gpst, afstime_t *afst)
{
    afst->wn = gpst->week;
    afst->fsec = gpst->sec;

    if (afst->fsec < 0.0) {
        afst->fsec += SECONDS_IN_WEEK;
        afst->wn -= 1;
    }
    else if (afst->fsec >= SECONDS_IN_WEEK) {
        afst->fsec -= SECONDS_IN_WEEK;
        afst->wn += 1;
    }
    
    afst->itow = (int)(afst->fsec / 1200.0);
    afst->fsec -= (double)(afst->itow) * 1200.0;
    
    afst->toi = (int)(afst->fsec / 12.0);
    afst->fsec -= (double)(afst->toi) * 12.0;

    return;
}

void gpst2date(const gpstime_t* g, datetime_t* t)
{
    // Convert Julian day number to calendar date
    int c = (int)(7 * g->week + floor(g->sec / 86400.0) + 2444245.0) + 1537;
    int d = (int)((c - 122.1) / 365.25);
    int e = 365 * d + d / 4;
    int f = (int)((c - e) / 30.6001);

    t->d = c - e - (int)(30.6001 * f);
    t->m = f - 1 - 12 * (f / 14);
    t->y = d - 4715 - ((7 + t->m) / 10);

    t->hh = ((int)(g->sec / 3600.0)) % 24;
    t->mm = ((int)(g->sec / 60.0)) % 60;
    t->sec = g->sec - 60.0 * floor(g->sec / 60.0);

    return;
}

void llh2xyz(double* llh, double* xyz)
{
    double a, h, tmp;

    a = R_MOON;
    h = a + llh[2];

    tmp = h * cos(llh[0]);
    xyz[0] = tmp * cos(llh[1]);
    xyz[1] = tmp * sin(llh[1]);
    xyz[2] = h * sin(llh[0]);

    return;
}

void ltcmat(double* llh, double t[3][3])
{
    double slat, clat;
    double slon, clon;

    slat = sin(llh[0]);
    clat = cos(llh[0]);
    slon = sin(llh[1]);
    clon = cos(llh[1]);

    t[0][0] = -slat * clon;
    t[0][1] = -slat * slon;
    t[0][2] = clat;
    t[1][0] = -slon;
    t[1][1] = clon;
    t[1][2] = 0.0;
    t[2][0] = clat * clon;
    t[2][1] = clat * slon;
    t[2][2] = slat;

    return;
}

// update receiver position assuming constant north/east/up velocity (m/s)
static void update_rcv_pos(double dt, double *llh, double *xyz, const double *vel_neu)
{
    if (dt <= 0.0) return;

    double vn = vel_neu[0];
    double ve = vel_neu[1];
    double vu = vel_neu[2];

    if (vn == 0.0 && ve == 0.0 && vu == 0.0) {
        return; // static receiver
    }

    double cos_lat = cos(llh[0]);
    double radius = R_MOON + llh[2];

    if (fabs(cos_lat) < 1e-6) {
        cos_lat = (cos_lat >= 0.0) ? 1e-6 : -1e-6;
    }

    llh[0] += (vn / radius) * dt;             // latitude (rad)
    llh[1] += (ve / (radius * cos_lat)) * dt; // longitude (rad)
    llh[2] += vu * dt;                        // height (m)

    llh2xyz(llh, xyz);
}

// speed profile helper for uniform acceleration → (optional) cruise → uniform deceleration
// Returns speed magnitude along a fixed direction at absolute time t in [0,T]
static double ramp_speed_profile(double t, double T, double acc, double vmax)
{
    if (t < 0.0) {
        t = 0.0;
    } else if (t > T) {
        t = T;
    }
    if (acc <= 0.0 || vmax <= 0.0) return vmax; // degenerate, treat as constant
    double t_acc_lim = vmax / acc;           // time to reach vmax if possible
    double t_acc = (t_acc_lim < 0.5*T) ? t_acc_lim : (0.5*T); // triangle if too short
    double v_peak = acc * t_acc;             // equals vmax in trapezoid, else triangle peak
    double t_dec_start = T - t_acc;
    if (t <= t_acc) {
        return acc * t;                      // accelerate
    } else if (t < t_dec_start) {
        return v_peak;                       // cruise (may be 0 length for triangle)
    } else {
        double tau = T - t;                  // time remaining to end
        return acc * tau;                    // decelerate to 0 at T
    }
}

void xyz2neu(double* xyz, double t[3][3], double* neu)
{
    neu[0] = t[0][0] * xyz[0] + t[0][1] * xyz[1] + t[0][2] * xyz[2];
    neu[1] = t[1][0] * xyz[0] + t[1][1] * xyz[1] + t[1][2] * xyz[2];
    neu[2] = t[2][0] * xyz[0] + t[2][1] * xyz[1] + t[2][2] * xyz[2];

    return;
}

void neu2azel(double* azel, double* neu)
{
    double ne;

    azel[0] = atan2(neu[1], neu[0]);
    if (azel[0] < 0.0)
        azel[0] += (2.0 * PI);

    ne = sqrt(neu[0] * neu[0] + neu[1] * neu[1]);
    azel[1] = atan2(neu[2], ne);

    return;
}

void satpos(ephem_t eph, gpstime_t g, double* pos, double* vel, double* clk)
{
    double tk;
    double mk;
    double ek;
    double ekold;
    double OneMinusecosE;
    double cek, sek;
    double ekdot;
    double uk;
    double cuk, suk;
    double ukdot;
    double rk;
    double rkdot;
    double ik;
    double cik, sik;
    double xpk, ypk;
    double xpkdot, ypkdot;
    double ok;
    double cok, sok;

    tk = g.sec - eph.toe.sec;

    if (tk > SECONDS_IN_HALF_WEEK)
        tk -= SECONDS_IN_WEEK;
    else if (tk < -SECONDS_IN_HALF_WEEK)
        tk += SECONDS_IN_WEEK;

    // Mean anomaly
    mk = eph.m0 + eph.n * tk;

    // Eccentric anomaly
    ek = mk;
    ekold = ek + 1.0;

    OneMinusecosE = 1.0;

    while (fabs(ek - ekold) > 1.0E-14)
    {
        ekold = ek;
        OneMinusecosE = 1.0 - eph.ecc * cos(ekold);
        ek = ek + (mk - ekold + eph.ecc * sin(ekold)) / OneMinusecosE;
    }

    sek = sin(ek);
    cek = cos(ek);

    ekdot = eph.n / OneMinusecosE;

    // True anomaly + Argument of perigee
    uk = atan2(eph.sq1e2 * sek, cek - eph.ecc) + eph.aop;
    suk = sin(uk);
    cuk = cos(uk);
    ukdot = eph.sq1e2 * ekdot / OneMinusecosE;

    // Range and range rate
    rk = eph.A * OneMinusecosE;
    rkdot = eph.A * eph.ecc * sek * ekdot;

    xpk = rk * cuk;
    ypk = rk * suk;
    xpkdot = rkdot * cuk - ypk * ukdot;
    ypkdot = rkdot * suk + xpk * ukdot;

    // Inclination
    ik = eph.inc0;

    sik = sin(ik);
    cik = cos(ik);

    // RAAN
    ok = eph.omg0;
    sok = sin(ok);
    cok = cos(ok);

    // Moon-centered inertial coordinates
    pos[0] = xpk * cok - ypk * cik * sok;
    pos[1] = xpk * sok + ypk * cik * cok;
    pos[2] = ypk * sik;

    vel[0] = xpkdot * cok - ypkdot * cik * sok;
    vel[1] = xpkdot * sok + ypkdot * cik * cok;
    vel[2] = ypkdot * sik;

    // Satellite clock correction
    tk = g.sec - eph.toc.sec;

    if (tk > SECONDS_IN_HALF_WEEK)
        tk -= SECONDS_IN_WEEK;
    else if (tk < -SECONDS_IN_HALF_WEEK)
        tk += SECONDS_IN_WEEK;

    clk[0] = eph.af0 + tk * eph.af1;
    clk[1] = eph.af1;
}

int readAlmanac(ephem_t eph[], const char* fname)
{
    int nsat = 0;

    FILE* fp;
    int sv;
    char str[MAX_CHAR];

    if (NULL == (fp = fopen(fname, "rt")))
        return(-1);

    for (sv = 0; sv < MAX_SAT; sv++)
        eph[sv].vflg = 0;

    while (1)
    {
        if (NULL == fgets(str, MAX_CHAR, fp))
            break;

        if (strlen(str) < 25)
            continue; // Skip empty line

        if (str[0] == '*')
        {
            // ID
            if (NULL == fgets(str, MAX_CHAR, fp))
                break;

            sv = atoi(str + 26) - 1;

            if (sv<0 || sv>MAX_SAT)
                break;

            // Health
            if (NULL == fgets(str, MAX_CHAR, fp))
                break;

            // Eccentricity
            if (NULL == fgets(str, MAX_CHAR, fp))
                break;

            eph[sv].ecc = atof(str + 26);

            // Time of Applicability(s)
            if (NULL == fgets(str, MAX_CHAR, fp))
                break;

            eph[sv].toe.sec = atof(str + 26);

            // Orbital Inclination(rad)
            if (NULL == fgets(str, MAX_CHAR, fp))
                break;

            eph[sv].inc0 = atof(str + 26);

            // Rate of Right Ascen(r/s)
            if (NULL == fgets(str, MAX_CHAR, fp))
                break;

            // SQRT(A)  (m 1 / 2)
            if (NULL == fgets(str, MAX_CHAR, fp))
                break;

            eph[sv].sqrta = atof(str + 26);

            // Right Ascen at Week(rad)
            if (NULL == fgets(str, MAX_CHAR, fp))
                break;

            eph[sv].omg0 = atof(str + 26);

            // Argument of Perigee(rad)
            if (NULL == fgets(str, MAX_CHAR, fp))
                break;

            eph[sv].aop = atof(str + 26);

            // Mean Anom(rad)
            if (NULL == fgets(str, MAX_CHAR, fp))
                break;

            eph[sv].m0 = atof(str + 26);

            // Af0(s)
            if (NULL == fgets(str, MAX_CHAR, fp))
                break;

            eph[sv].af0 = atof(str + 26);

            // Af1(s/s)
            if (NULL == fgets(str, MAX_CHAR, fp))
                break;

            eph[sv].af1 = atof(str + 26);

            // Week
            if (NULL == fgets(str, MAX_CHAR, fp))
                break;

            eph[sv].toe.week = atoi(str + 26);
            // GPS week number rollover on April 6, 2019.
            eph[sv].toe.week += 2048;

            eph[sv].toc = eph[sv].toe;

            // Valid almanac
            eph[sv].vflg = 1;
            nsat++;

            // Update the working variables
            eph[sv].A = eph[sv].sqrta * eph[sv].sqrta;
            eph[sv].n = sqrt(GM_MOON / (eph[sv].A * eph[sv].A * eph[sv].A));
            eph[sv].sq1e2 = sqrt(1.0 - eph[sv].ecc * eph[sv].ecc);
        }
    }

    return (nsat);
}

void eph2sbf(const ephem_t eph, uint8_t *syms)
{
    uint32_t toe;
    uint32_t toc;
    uint32_t ecc;
	uint32_t sqrta;
	int32_t m0;
	int32_t omg0;
	int32_t inc0;
	int32_t aop;
    int32_t af0;
	int32_t af1;

	toc = (uint32_t)(eph.toc.sec/16.0);
    ecc = (uint32_t)(eph.ecc/POW2_M32);
	sqrta = (uint32_t)(eph.sqrta/POW2_M19);
    inc0 = (int32_t)(eph.inc0/POW2_M31/PI);
	omg0 = (int32_t)(eph.omg0/POW2_M31/PI);
	aop = (int32_t)(eph.aop/POW2_M31/PI);
    m0 = (int32_t)(eph.m0/POW2_M31/PI);

    toe = (uint32_t)(eph.toe.sec/16.0);
    af0 = (int32_t)(eph.af0 / POW2_M31);
	af1 = (int32_t)(eph.af1 / POW2_M43);
    
    sdr_unpack_data(toe, 16, syms);
    sdr_unpack_data(ecc, 32, syms + 16);
    sdr_unpack_data(sqrta, 32, syms + 48);
    sdr_unpack_data(inc0, 32, syms + 80);
    sdr_unpack_data(omg0, 32, syms + 112);
    sdr_unpack_data(aop, 32, syms + 144);
    sdr_unpack_data(m0, 32, syms + 176);
    sdr_unpack_data(toc, 16, syms + 208);
    sdr_unpack_data(af0, 22, syms + 224);
    sdr_unpack_data(af1, 16, syms + 246);

    return;
}


void subVect(double* y, double* x1, double* x2)
{
    y[0] = x1[0] - x2[0];
    y[1] = x1[1] - x2[1];
    y[2] = x1[2] - x2[2];

    return;
}

double normVect(double* x)
{
    return(sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]));
}

double dotProd(double* x1, double* x2)
{
    return(x1[0] * x2[0] + x1[1] * x2[1] + x1[2] * x2[2]);
}

void computeRange(range_t* rho, ephem_t eph, gpstime_t g, double xyz[])
{
    double pos[3], vel[3], clk[2];
    double los[3];
    double tau;
    double range, rate;

    // SV position at time of the pseudorange observation.
    satpos(eph, g, pos, vel, clk);

    // Receiver to satellite vector and light-time.
    subVect(los, pos, xyz);
    tau = normVect(los) / SPEED_OF_LIGHT;

    // Extrapolate the satellite position backwards to the transmission time.
    pos[0] -= vel[0] * tau;
    pos[1] -= vel[1] * tau;
    pos[2] -= vel[2] * tau;

    // New observer to satellite vector and satellite range.
    subVect(los, pos, xyz);
    range = normVect(los);

    // Pseudorange
    rho->range = range - SPEED_OF_LIGHT * clk[0];

    // Relative velocity of SV and receiver
    rate = dotProd(vel, los) / range;

    // Pseudorange rate
    rho->rate = rate;

    // Time of application
    rho->g = g;

    return;
}

void computeCodePhase(channel_t* chan, range_t rho0, range_t rho1, double dt)
{
    double ms;
    double rhorate;
    int ibit, iframe;
    int sv;
    gpstime_t gt;
    afstime_t afst;

    // Pseudorange rate
    rhorate = (rho1.range - rho0.range) / dt;

    // Carrier and code frequency
    chan->f_carr = -rhorate / LAMBDA;
    chan->I.f_code = I_CODE_FREQ + chan->f_carr * CARR_TO_I_CODE;
    chan->Q.f_code = Q_CODE_FREQ + chan->f_carr * CARR_TO_Q_CODE;

    // Signal transmission time
    gt.sec = rho0.g.sec;
    gt.week = rho0.g.week;
    timeadd(&gt, -rho0.range / SPEED_OF_LIGHT);
    gpst2afst(&gt, &afst);
    
    iframe = afst.toi; // 1 frame = 12 sec
    ms = afst.fsec * 1000.0; // Fractional milliseconds within a frame
    ibit = (int)(ms / 2.0); // 1 bit = 1 code = 2 ms
    ms -= ibit * 2.0; // Fractional milliseconds within a code

    // Spreading code
    chan->I.code_phase = ms / 2.0 * 2046.0; // 1 chip = 2 ms
    chan->I.C = chan->I.code[(int)chan->I.code_phase];

    chan->Q.code_phase = ms / 2.0 * 10230.0; // 1 chip = 2 ms
    chan->Q.C = chan->Q.code[(int)chan->Q.code_phase];

    // Navigation message
    chan->I.ibit = ibit;
    chan->I.iframe = iframe;

    // Logic Level | Signal Level
    //      0      |      +1
    //      1      |      -1

    //chan->I.D = -2 * chan->I.data[iframe % 2][ibit] + 1;
    chan->I.D = -2 * chan->I.data[0][ibit] + 1;

    // Seconday code
    sv = chan->prn - 1;
    chan->Q.ibit = ibit % 4;
    chan->Q.S = -2 * scode[sv % 4][chan->Q.ibit] + 1;

    // Tertiary code
    chan->Q.ichip = (int)(afst.fsec * 1000.0 / 8.0);
    chan->Q.T = -2 * tcode[sv][chan->Q.ichip] + 1;

    return;
}

void bitncpy(uint8_t *syms1, const uint8_t *syms2, int n)
{
    //for (int i = 0; i < n; i++) {
    //    syms1[i] = syms2[i];
    //}

    memcpy(syms1, syms2, n);   

    return;
}

static void log_truth_sample(FILE *fp, double time_rel, const double *llh, const double *xyz)
{
    if (!fp) return;
    fprintf(fp, "%.3f,%.9f,%.9f,%.3f,%.3f,%.3f,%.3f\n",
        time_rel, llh[0] * R2D, llh[1] * R2D, llh[2], xyz[0], xyz[1], xyz[2]);
}

void printUsage(void)
{
    printf("Usage: afs_sim [-t tsec] [-s freq] [-e feph] [-b bits] [-l lat:lon:hgt] [-vel vn:ve:vu] [-acc a] [-truth file] [-rnglog file] [-orbitlog file] [-elvmask deg] [-chan] [-cn0 dBHz] [-halo spec] [-cheb file] [-chebgen file] [-chebdt sec] [-chebN N] [-dbgprn N] [-dbghalo N] [-mp a,b,c] fout\n");
    printf("  -halo spec : CSV file, list (comma/semicolon), multiple -halo, or range pattern like path/halo_prn1-8.csv\n");

    printf("  -mp a,b,c : enable multipath jitter (sigma=a+b*exp(c*el_deg))\n");
    exit(0);
}

int main(int argc, char** argv)
{
    FILE *fp, *truth_fp = NULL, *rng_fp = NULL, *orbit_fp = NULL;
    const char *fout = "", *feph="", *ftruth = NULL;
    double tsec = 1.0, freq = 12.0e6;
    int nbits = 16;
    double llh[3] = { 0.0, 0.0, -(R_MOON + 1.0e3) };
    double xyz[3];
    int ret;

    int sv;
    int neph;
    ephem_t eph[MAX_SAT];
    gpstime_t g0;
    datetime_t t0;

    int i;
    static channel_t chan[MAX_SAT]; // Avoid warning C6262
    double tmat[3][3];
    int nsat;
    double pos[3], vel[3], clk[2];
    double los[3];
    double neu[3];
    double azel[2];
    double elvmask = 0.0 / R2D; // elevation mask in rad (can override by -elvmask)

    int iq_buff_size;
    double delt;
    void* iq_buff = NULL;

    gpstime_t grx;
    clock_t tstart, tend;
    range_t rho, rho0[MAX_SAT];
    afstime_t afst; // AFS frame/time info for main

    int isim, nsim;
    int isamp;
    int ph;
    int ip, qp;
    int I, Q;

    int sample = 0;
    int thresh = 1; // 2-bit ADC threshold
    int twobitADC = 0;

    double path_loss;
    int gain;
    int noise_scale;

    int inv_q = 0; // Inverse Q sign flag
    int dbg_prn = 0; // 0=off; >0 print satpos() at g0 for PRN (Kepler)
    int dbg_prn_halo = 0; // 0=off; >0 print satpos_halo() at g0 for PRN
    int dbg_cmp_prn = 0; // 0=off; >0 compare HALO vs Kepler az/el/range/dopp at g0

    double vel_neu[3] = {0.0, 0.0, 0.0}; // receiver velocity (N,E,U) m/s
    int moving = 0;
    // Optional ramp motion along vel_neu direction
    double acc_mag = 0.0; // [m/s^2]
    int use_acc_ramp = 0;
    double vdir[3] = {0.0, 0.0, 0.0};
    double vmax_mag = 0.0;
    int use_chan_emul = 0;
    int use_cn0 = 0;
    double cn0_dB = 0.0;
    double cn0_lin = 0.0;
    double gain_accum_ref = 0.0;

    for (i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-t") && i + 1 < argc) {
            tsec = atof(argv[++i]);
        }
        else if (!strcmp(argv[i], "-s") && i + 1 < argc) {
            freq = atof(argv[++i]);
        }
        else if (!strcmp(argv[i], "-e") && i + 1 < argc) {
            feph = argv[++i];
        }
        else if (!strcmp(argv[i], "-b") && i + 1 < argc) {
            nbits = atoi(argv[++i]);
            if (nbits != 2) {
                fprintf(stderr, "ERROR: Invalid ADC bits for -b option.\n");
                exit(-1);
            }
        }
        else if (!strcmp(argv[i], "-l") && i + 1 < argc) {
            ret = sscanf(argv[++i], "%lf:%lf:%lf", &llh[0], &llh[1], &llh[2]);
            if (ret != 3) {
                fprintf(stderr, "ERROR: Wrong format for -l option.\n");
                exit(-1);
            }
            llh[0] /= R2D;
            llh[1] /= R2D;
        }
        else if (!strcmp(argv[i], "-vel") && i + 1 < argc) {
            ret = sscanf(argv[++i], "%lf:%lf:%lf", &vel_neu[0], &vel_neu[1], &vel_neu[2]);
            if (ret != 3) {
                fprintf(stderr, "ERROR: Wrong format for -vel option.\n");
                exit(-1);
            }
            moving = (vel_neu[0] != 0.0 || vel_neu[1] != 0.0 || vel_neu[2] != 0.0);
        }
        else if (!strcmp(argv[i], "-acc") && i + 1 < argc) {
            acc_mag = atof(argv[++i]);
            if (acc_mag < 0.0) acc_mag = -acc_mag;
        }
        else if (!strcmp(argv[i], "-InQ")) {
            inv_q = 1; // Set inverse Q sign flag for MAX2771
        }
        else if (!strcmp(argv[i], "-chan")) {
            use_chan_emul = 1;
        }
        else if (!strcmp(argv[i], "-cn0") && i + 1 < argc) {
            cn0_dB = atof(argv[++i]);
            use_cn0 = 1;
        }
        else if (!strcmp(argv[i], "-truth") && i + 1 < argc) {
            ftruth = argv[++i];
        }
        else if (!strcmp(argv[i], "-rnglog") && i + 1 < argc) {
            const char *frng = argv[++i];
            rng_fp = fopen(frng, "w");
            if (!rng_fp) {
                fprintf(stderr, "ERROR: Failed to open -rnglog file.\n");
                exit(1);
            }
            fprintf(rng_fp, "# $SIMRNG,time_s,prn,geom_range_m\n");
        }
        else if (!strcmp(argv[i], "-orbitlog") && i + 1 < argc) {
            const char *forbit = argv[++i];
            orbit_fp = fopen(forbit, "w");
            if (!orbit_fp) {
                fprintf(stderr, "ERROR: Failed to open -orbitlog file.\n");
                exit(1);
            }
            fprintf(orbit_fp, "time_s,prn,x_m,y_m,z_m,vx_mps,vy_mps,vz_mps\n");
        }
        else if (!strcmp(argv[i], "-elvmask") && i + 1 < argc) {
            elvmask = atof(argv[++i]) / R2D; // degrees to radians
        }
        else if (!strcmp(argv[i], "-halo") && i + 1 < argc) {
            const char *arg = argv[++i];
            size_t len = strlen(halo_track_file);
            if (len > 0) {
                if (len < MAX_CHAR-1) { halo_track_file[len++] = ','; halo_track_file[len] = '\0'; }
            }
            strncat(halo_track_file, arg, MAX_CHAR-1 - strlen(halo_track_file));
            halo_track_file[MAX_CHAR-1] = '\0';
            use_halo_tracks = 1;
        }
        else if (!strcmp(argv[i], "-dbghalo") && i + 1 < argc) {
            dbg_prn_halo = atoi(argv[++i]);
        }
        else if (!strcmp(argv[i], "-dbgprn") && i + 1 < argc) {
            dbg_prn = atoi(argv[++i]);
        }
        else if (!strcmp(argv[i], "-cheb") && i + 1 < argc) {
            strncpy(cheb_file, argv[++i], MAX_CHAR-1);
            cheb_file[MAX_CHAR-1] = '\0';
            use_cheb = 1;
        }
        else if (!strcmp(argv[i], "-chebgen") && i + 1 < argc) {
            do_cheb_gen = 1;
            strncpy(cheb_gen_file, argv[++i], MAX_CHAR-1);
            cheb_gen_file[MAX_CHAR-1] = '\0';
        }
        else if (!strcmp(argv[i], "-chebdt") && i + 1 < argc) {
            cheb_dt = atof(argv[++i]);
            if (cheb_dt <= 0.0) {
                fprintf(stderr, "ERROR: Invalid -chebdt (must be > 0).\n");
                exit(-1);
            }
        }
        else if (!strcmp(argv[i], "-mp") && i + 1 < argc) {
            if (sscanf(argv[++i], "%lf,%lf,%lf", &mp_a, &mp_b, &mp_c) != 3) {
                fprintf(stderr, "ERROR: Invalid -mp a,b,c\n");
                exit(-1);
            }
            use_multipath = 1;
        }
        else if (!strcmp(argv[i], "-chebN") && i + 1 < argc) {
            cheb_N = atoi(argv[++i]);
            if (cheb_N < 1 || cheb_N > CHEB_MAX_N) {
                fprintf(stderr, "ERROR: Invalid -chebN (1..%d).\n", CHEB_MAX_N);
                exit(-1);
            }
        }
        else if (!strcmp(argv[i], "-dbgcmp") && i + 1 < argc) {
            dbg_cmp_prn = atoi(argv[++i]);
        }
        //else if (argv[i][0] == '-' && argv[i][1] != '\0') {
        else if (argv[i][0] == '-') {
            printUsage();
        }
        else {
            fout = argv[i];
        }
    }

    if (!*fout) {
        fprintf(stderr, "ERROR: Specify output file.\n");
        exit(-1);
    }

    if (!readTertiary("008_Weil1500hex210prns.txt")) {
		printf("ERROR: Failed to read tertiary code file.\n");
		exit(-1);
	}

    // Set user location

    if (llh[2] < -(R_MOON)) {
        // Default user location; Shackleton crater
        llh[0] = -89.66 / R2D;
        llh[1] = 129.20 / R2D;
        llh[2] = 100.0;
    }

    llh2xyz(llh, xyz);

    printf("xyz = %11.1f, %11.1f, %11.1f\n", xyz[0], xyz[1], xyz[2]);
    printf("llh = %11.6f, %11.6f, %11.1f\n", llh[0] * R2D, llh[1] * R2D, llh[2]);

    // Setup ramp profile if requested
    vmax_mag = sqrt(vel_neu[0]*vel_neu[0] + vel_neu[1]*vel_neu[1] + vel_neu[2]*vel_neu[2]);
    if (vmax_mag > 0.0) { vdir[0] = vel_neu[0]/vmax_mag; vdir[1] = vel_neu[1]/vmax_mag; vdir[2] = vel_neu[2]/vmax_mag; }
    use_acc_ramp = (moving && acc_mag > 0.0 && vmax_mag > 0.0);
    if (use_acc_ramp) {
        double t_acc_lim = vmax_mag / acc_mag;
        double t_acc = (t_acc_lim < 0.5*tsec) ? t_acc_lim : (0.5*tsec);
        printf("Receiver ramp: vmax=%.3f m/s, acc=%.3f m/s^2, t_acc=%.2f s, %s profile\n",
               vmax_mag, acc_mag, t_acc, (t_acc_lim < 0.5*tsec) ? "trapezoid" : "triangle");
    }
    else if (moving) {
        printf("Receiver velocity (N,E,U) = %.3f, %.3f, %.3f m/s\n",
               vel_neu[0], vel_neu[1], vel_neu[2]);
    }

    // Read ephemeris
    if (!*feph) {
        feph = "default_almanac.txt";
    }
    neph = readAlmanac(eph, feph);

    if (neph == -1) {
        fprintf(stderr, "ERROR: Failed to open ephemeris file.\n");
        exit(1);
    }
    
    // Set simulation start time

    g0.week = -1;

    if (neph > 0) {
        for (sv = 0; sv < MAX_SAT; sv++) {
            if (g0.week < 0 && eph[sv].vflg == 1) {
                g0 = eph[sv].toe;
                break;
            }
        }
    }
    else
    {
        fprintf(stderr, "ERROR: No valid ephemeris has been found.\n");
        exit(1);
    }

    gpst2date(&g0, &t0);
    printf("Start time = %4d/%02d/%02d,%02d:%02d:%02.0f (%d:%.0f)\n", t0.y, t0.m, t0.d, t0.hh, t0.mm, t0.sec, g0.week, g0.sec);

    /* Optional: load HALO tracks (supports multiple CSV via comma/semicolon-separated list
       or multiple -halo options) before debug prints */
    if (use_halo_tracks) {
        char list[MAX_CHAR];
        strncpy(list, halo_track_file, MAX_CHAR-1); list[MAX_CHAR-1] = '\0';
        halo_reset();
        int loaded = 0;
        const char *delims = ",; \t";
        char *tok = strtok(list, delims);
        while (tok) {
            if (*tok) {
                if (!halo_add_from_spec(tok)) {
                    fprintf(stderr, "ERROR: Failed to read HALO track CSV: %s\n", tok);
                } else {
                    loaded = 1;
                }
            }
            tok = strtok(NULL, delims);
        }
        if (!loaded) {
            fprintf(stderr, "ERROR: No HALO tracks loaded (files: %s)\n", halo_track_file);
            exit(1);
        }
        for (int i = 0; i < MAX_SAT; i++) if (halo_db[i].valid) halo_sort(&halo_db[i].tr);
        /* align start time to first track sample */
        if (halo_tmin > 0.0) g0.sec = halo_tmin;
    }
    if (use_cheb) {
        int nseg = cheb_load(cheb_file);
        if (nseg <= 0) {
            fprintf(stderr, "ERROR: Failed to read Chebyshev ephemeris: %s\n", cheb_file);
            use_cheb = 0; // fallback
        }
    }

    // Optional: generate Chebyshev ephemeris from current source (HALO/Kepler)
    if (do_cheb_gen) {
        FILE *cfp = fopen(cheb_gen_file, "wt");
        if (!cfp) {
            fprintf(stderr, "ERROR: failed to open -chebgen output: %s\n", cheb_gen_file);
        } else {
            double t_start = g0.sec;
            double t_end   = g0.sec + tsec;
            int nseg_out = 0;
            for (int prn=1; prn<=MAX_SAT; prn++) {
                int has_src = 0;
                if (use_halo_tracks) {
                    int idx = halo_prn_index(prn);
                    has_src = (idx>=0 && halo_db[idx].valid);
                } else {
                    has_src = (eph[prn-1].vflg==1);
                }
                if (!has_src) continue;
                for (double t0s=t_start; t0s < t_end-1e-9; t0s += cheb_dt) {
                    if (cheb_fit_segment(cfp, prn, g0.week, eph, t0s, cheb_dt, cheb_N)) nseg_out++;
                }
            }
            fclose(cfp);
            printf("CHEB exported: %d segments -> %s (dt=%.1f,N=%d)\n", nseg_out, cheb_gen_file, cheb_dt, cheb_N);
        }
    }

    /* AFS time and debug prints */
    gpst2afst(&g0, &afst);
    printf("AFS time: WN = %d, ITOW = %d, TOI = %d, fsec = %.1f\n", afst.wn, afst.itow, afst.toi, afst.fsec);

    if (dbg_prn >= 1 && dbg_prn <= MAX_SAT && eph[dbg_prn-1].vflg == 1) {
        double ptest[3], vtest[3], ctest[2];
        satpos(eph[dbg_prn-1], g0, ptest, vtest, ctest);
        printf("DBG satpos Kepler PRN %02d @g0: pos_km = [% .6f % .6f % .6f], vel_kms = [% .9f % .9f % .9f]\n",
               dbg_prn,
               ptest[0]/1000.0, ptest[1]/1000.0, ptest[2]/1000.0,
               vtest[0]/1000.0, vtest[1]/1000.0, vtest[2]/1000.0);
    }
    if (dbg_prn >= 1 && dbg_prn <= MAX_SAT && use_cheb) {
        double pc[3], vc[3], cc[2]={0};
        if (satpos_cheb(dbg_prn, g0.sec, pc, vc, cc)) {
            printf("DBG satpos Cheb    PRN %02d @g0: pos_km = [% .6f % .6f % .6f], vel_kms = [% .9f % .9f % .9f]\n",
                dbg_prn, pc[0]/1000.0, pc[1]/1000.0, pc[2]/1000.0, vc[0]/1000.0, vc[1]/1000.0, vc[2]/1000.0);
        }
    }
    if (dbg_prn_halo >= 1 && dbg_prn_halo <= MAX_SAT) {
        if (use_halo_tracks) {
            double pth[3], vth[3], cth[2] = {0};
            if (satpos_halo(dbg_prn_halo, g0.sec, pth, vth, cth)) {
                printf("DBG satpos HALO   PRN %02d @g0: pos_km = [% .6f % .6f % .6f], vel_kms = [% .9f % .9f % .9f]\n",
                       dbg_prn_halo,
                       pth[0]/1000.0, pth[1]/1000.0, pth[2]/1000.0,
                       vth[0]/1000.0, vth[1]/1000.0, vth[2]/1000.0);
            } else {
                printf("DBG satpos HALO   PRN %02d @g0: not in track or no data\n", dbg_prn_halo);
            }
        }
    }

    /* Optional: direct comparison of HALO vs Kepler az/el/range/dopp for the same PRN at g0 */
    if (dbg_cmp_prn >= 1 && dbg_cmp_prn <= MAX_SAT) {
        int sv_cmp = dbg_cmp_prn - 1;
        double azel_h[2]={0}, azel_k[2]={0};
        range_t rh={0}, rk={0};
        int have_h = 0, have_k = 0;
        if (use_halo_tracks) {
            double ph[3], vh[3], clkh[2]={0}, losh[3], neu_h[3];
            if (satpos_halo(dbg_cmp_prn, g0.sec, ph, vh, clkh)) {
                subVect(losh, ph, xyz);
                double tauh = normVect(losh)/SPEED_OF_LIGHT;
                ph[0]-=vh[0]*tauh; ph[1]-=vh[1]*tauh; ph[2]-=vh[2]*tauh;
                subVect(losh, ph, xyz);
                double rgeom = normVect(losh);
                rh.range = rgeom - SPEED_OF_LIGHT*clkh[0];
                rh.rate  = dotProd(vh, losh) / (rgeom + 1e-9);
                xyz2neu(losh, tmat, neu_h);
                neu2azel(azel_h, neu_h);
                have_h = 1;
            }
        }
        if (eph[sv_cmp].vflg == 1) {
            double pk[3], vk[3], clkk[2], losk[3], neu_k[3];
            satpos(eph[sv_cmp], g0, pk, vk, clkk);
            subVect(losk, pk, xyz);
            double tauk = normVect(losk)/SPEED_OF_LIGHT;
            pk[0]-=vk[0]*tauk; pk[1]-=vk[1]*tauk; pk[2]-=vk[2]*tauk;
            subVect(losk, pk, xyz);
            double rgeom = normVect(losk);
            rk.range = rgeom - SPEED_OF_LIGHT*clkk[0];
            rk.rate  = dotProd(vk, losk) / (rgeom + 1e-9);
            xyz2neu(losk, tmat, neu_k);
            neu2azel(azel_k, neu_k);
            have_k = 1;
        }
        if (have_h || have_k) {
            printf("DBG CMP PRN %02d @g0:\n", dbg_cmp_prn);
            if (have_h) printf("  HALO: az=%.6f deg, el=%.6f deg, range=%.3f m, dopp=%.1f Hz\n",
                               azel_h[0]*R2D, azel_h[1]*R2D, rh.range, -rh.rate/LAMBDA);
            else       printf("  HALO: (no data)\n");
            if (have_k) printf("  KEPL: az=%.6f deg, el=%.6f deg, range=%.3f m, dopp=%.1f Hz\n",
                               azel_k[0]*R2D, azel_k[1]*R2D, rk.range, -rk.rate/LAMBDA);
            else       printf("  KEPL: (no eph)\n");
        }
    }

    // Check visible satellites

    for (i = 0; i < MAX_SAT; i++)
        chan[i].prn = 0; // Idle channel

    ltcmat(llh, tmat);

    nsat = 0;

    for (sv = 0; sv < MAX_SAT; sv++) {
        int have = 0;
        if (use_halo_tracks) {
            if (satpos_halo(sv + 1, g0.sec, pos, vel, clk)) have = 1;
        }
        else if (use_cheb) {
            if (satpos_cheb(sv + 1, g0.sec, pos, vel, clk)) have = 1;
        }
        else if (eph[sv].vflg == 1) {
            satpos(eph[sv], g0, pos, vel, clk);
            have = 1;
        }
        if (!have) continue;

        subVect(los, pos, xyz);
        xyz2neu(los, tmat, neu);
        neu2azel(azel, neu);

        if (azel[1] > elvmask) {
            chan[nsat].prn = sv + 1;

            chan[nsat].azel[0] = azel[0];
            chan[nsat].azel[1] = azel[1];
            chan[nsat].multipath_bias = 0.0;
            chan[nsat].has_multipath_bias = 0;

            nsat++; // Number of visible satellites
        }
    }

    printf("Number of channels = %d\n", nsat);

    // Baseband signal buffer and output file

    freq = floor(freq / 10.0);
    iq_buff_size = (int)freq; // samples per 0.1sec
    freq *= 10.0;

    delt = 1.0 / freq;

    // Allocate I/Q buffer
    if (nbits == 2) {
        iq_buff = (signed char*)calloc(2 * iq_buff_size, 1);
    }
    else {
        iq_buff = (short*)calloc(2 * iq_buff_size, 2);
    }

    if (iq_buff == NULL) {
        fprintf(stderr, "ERROR: Faild to allocate global IQ buffer.\n");
        exit(1);
    }
    
    if (ftruth) {
        truth_fp = fopen(ftruth, "w");
        if (!truth_fp) {
            fprintf(stderr, "ERROR: Failed to open truth file.\n");
            exit(1);
        }
        fprintf(truth_fp, "# time_s,lat_deg,lon_deg,height_m,x_m,y_m,z_m\n");
        log_truth_sample(truth_fp, 0.0, llh, xyz);
    }

    if (NULL == (fp = fopen(fout, "wb"))) { // Open output file
        fprintf(stderr, "ERROR: Failed to open output file.\n");
        exit(1);
    }

    // Initialize signals

    grx = g0; // Initial reception time

    for (i = 0; i < nsat; i++) {

        // Code generation
        icodegen(chan[i].I.code, chan[i].prn);
        qcodegen(chan[i].Q.code, chan[i].prn);

        // Allocate I/Q buffer
        chan[i].iq_buff = (int*)calloc(2 * iq_buff_size, sizeof(int));

        if (chan[i].iq_buff == NULL)
        {
            fprintf(stderr, "ERROR: Faild to allocate channel IQ buffer.\n");
            exit(1);
        }
    }

    // Generate frames and data bits

    for (i = 0; i < nsat; i++) {

        chan[i].g0 = g0; // Data bit reference time
    }

    // Initialize carrier phase
    for (i = 0; i < nsat; i++) {
                chan[i].carr_phase = 0.0;
    }

    // Initial pseudorange
    printf("SV    AZ    EL     RANGE     DOPP\n");
    for (i = 0; i < nsat; i++) {
        sv = chan[i].prn - 1;
        if (use_halo_tracks) {
            double pos_h[3], vel_h[3], clk_h[2] = {0};
            range_t rtmp;
            if (satpos_halo(chan[i].prn, grx.sec, pos_h, vel_h, clk_h)) {
                double los0[3];
                subVect(los0, pos_h, xyz);
                double tau0 = normVect(los0) / SPEED_OF_LIGHT;
                pos_h[0] -= vel_h[0] * tau0;
                pos_h[1] -= vel_h[1] * tau0;
                pos_h[2] -= vel_h[2] * tau0;
                subVect(los0, pos_h, xyz);
                double neu_now[3];
                xyz2neu(los0, tmat, neu_now);
                neu2azel(chan[i].azel, neu_now);
                double rgeom0 = normVect(los0);
                rtmp.range = rgeom0 - SPEED_OF_LIGHT * clk_h[0];
                rtmp.rate  = dotProd(vel_h, los0) / (rtmp.range + 1e-9);
                rtmp.g = grx;
                double mp_off = 0.0;
                if (use_multipath) {
                    if (!chan[i].has_multipath_bias) {
                        chan[i].multipath_bias = multipath_offset(chan[i].azel[1]);
                        chan[i].has_multipath_bias = 1;
                    }
                    mp_off = chan[i].multipath_bias;
                }
                rtmp.range += mp_off;
                rho0[sv] = rtmp;
                if (rng_fp) fprintf(rng_fp, "$SIMRNG,%.3f,%d,%.3f\n", grx.sec - g0.sec, chan[i].prn, rgeom0);
            } else {
                memset(&rho0[sv], 0, sizeof(rho0[sv]));
            }
        }
        else if (use_cheb) {
            double pos_h[3], vel_h[3], clk_h[2] = {0};
            range_t rtmp;
            if (satpos_cheb(chan[i].prn, grx.sec, pos_h, vel_h, clk_h)) {
                double los0[3];
                subVect(los0, pos_h, xyz);
                double tau0 = normVect(los0) / SPEED_OF_LIGHT;
                pos_h[0] -= vel_h[0] * tau0;
                pos_h[1] -= vel_h[1] * tau0;
                pos_h[2] -= vel_h[2] * tau0;
                subVect(los0, pos_h, xyz);
                double neu_now[3];
                xyz2neu(los0, tmat, neu_now);
                neu2azel(chan[i].azel, neu_now);
                double rgeom0 = normVect(los0);
                rtmp.range = rgeom0 - SPEED_OF_LIGHT * clk_h[0];
                rtmp.rate  = dotProd(vel_h, los0) / (rgeom0 + 1e-9);
                rtmp.g = grx;
                double mp_off = 0.0;
                if (use_multipath) {
                    if (!chan[i].has_multipath_bias) {
                        chan[i].multipath_bias = multipath_offset(chan[i].azel[1]);
                        chan[i].has_multipath_bias = 1;
                    }
                    mp_off = chan[i].multipath_bias;
                }
                rtmp.range += mp_off;
                rho0[sv] = rtmp;
                if (rng_fp) fprintf(rng_fp, "$SIMRNG,%.3f,%d,%.3f\n", grx.sec - g0.sec, chan[i].prn, rgeom0);
            } else {
                memset(&rho0[sv], 0, sizeof(rho0[sv]));
            }
        }
        else {
            // Also compute geometric range for logging
            double pos_k[3], vel_k[3], clk_k[2];
            satpos(eph[sv], grx, pos_k, vel_k, clk_k);
            double los0[3];
            subVect(los0, pos_k, xyz);
            double tau0 = normVect(los0) / SPEED_OF_LIGHT;
            pos_k[0] -= vel_k[0] * tau0; pos_k[1] -= vel_k[1] * tau0; pos_k[2] -= vel_k[2] * tau0;
            subVect(los0, pos_k, xyz);
            double neu_now[3];
            xyz2neu(los0, tmat, neu_now);
            neu2azel(chan[i].azel, neu_now);
            double rgeom0 = normVect(los0);
            computeRange(&rho0[sv], eph[sv], grx, xyz);
            double mp_off = 0.0;
            if (use_multipath) {
                if (!chan[i].has_multipath_bias) {
                    chan[i].multipath_bias = multipath_offset(chan[i].azel[1]);
                    chan[i].has_multipath_bias = 1;
                }
                mp_off = chan[i].multipath_bias;
            }
            rho0[sv].range += mp_off;
            if (rng_fp) fprintf(rng_fp, "$SIMRNG,%.3f,%d,%.3f\n", grx.sec - g0.sec, chan[i].prn, rgeom0);
        }
        { double pl0 = 5200000.0 / rho0[sv].range; int g0tmp = (int)(pl0 * (double)GAIN_SCALE); gain_accum_ref += (double)g0tmp; }

        printf("%02d %6.1f %5.1f %10.1f %+8.1f\n", chan[i].prn, 
            chan[i].azel[0] * R2D, chan[i].azel[1] * R2D, rho0[sv].range, -rho0[sv].rate/LAMBDA);
    }

    // Insert synchronization pattern 

    const uint8_t sync[9] ={0xCC, 0x63, 0xF7, 0x45, 0x36, 0xF4, 0x9E, 0x04, 0xA0}; // left justified

    for (i = 0; i < nsat; i++) {
        sdr_unpack_bits(sync, 68, chan[i].I.data[0]); // synchronization pattern
    }

    // Insert subframe 1

    uint8_t AFS_SB1[52];
    
    // The TOI corresponds to the node time epoch at the leading edge of the "next" 12-second frame.
    int toi_mod = (afst.toi + 1) % AFS_TOI_WRAP;
    generate_BCH_AFS_SF1(AFS_SB1, 0, toi_mod); // subframe 1 for frame ID 0

    for (i = 0; i < nsat; i++) {
        bitncpy(chan[i].I.data[0] + 68, AFS_SB1, 52);
    }

    // Insert subframe 2-4

    uint8_t AFS_SB234[5880];
    uint8_t syms[5880];

    for (i = 0; i < 846; i++) {
        syms[i] = i%2; // test pattern 0,1,0,1,...
    }
    append_CRC24(syms, 870);
    encode_LDPC_AFS_SF3(syms, AFS_SB234 + 2400); // Subframe 3

    encode_LDPC_AFS_SF3(syms, AFS_SB234 + 4140); // Subframe 4

    for (i = 0; i < nsat; i++) {

        for (int j = 0; j < 1176; j++) {
            syms[j] = j%2; // test pattern 0,1,0,1,...
        }
        uint32_t data = ((uint32_t)afst.wn<<9) | (afst.itow & 0x1ff);
        sdr_unpack_data(data, 22, syms); // Insert WN and ITOW
        
        // Insert ephemeris
        sv = chan[i].prn - 1;
        eph2sbf(eph[sv], syms + 22);

        append_CRC24(syms, 1200);
        encode_LDPC_AFS_SF2(syms, AFS_SB234); // Subframe 2
 
        interleave_AFS_SF234(AFS_SB234, syms); // Interleaving

        bitncpy(chan[i].I.data[0] + 120, syms, 5880);
    }

    // Generate baseband signals

    tstart = clock();

    printf("Generating baseband signals...\n");

    printf("\rTime = %4.1f", grx.sec - g0.sec);
    fflush(stdout);

    // Update receiver time
    grx.sec += 0.1;

    nsim = (int)((tsec - 0.1) * 10.0); // 10Hz update rate

    // 2-bit ADC threshold
    thresh = (int)(1250.0 * sqrt((double)nsat)); // 1-sigma value of thermal noise
    // Noise amplitude scale
    noise_scale = (int)(3000 / 1449.0 * sqrt((double)nsat) * GAIN_SCALE); 


    // Seed the gaussian noise generator
    srandn(time(NULL));

    for (isim = 0; isim < nsim; isim++) {
        if (moving) {
            double vel_step[3];
            if (use_acc_ramp) {
                double t_center = (grx.sec - g0.sec) - 0.05; // midpoint of this 0.1s step
                if (t_center < 0.0) t_center = 0.0;
                double vmag_now = ramp_speed_profile(t_center, tsec, acc_mag, vmax_mag);
                vel_step[0] = vdir[0] * vmag_now;
                vel_step[1] = vdir[1] * vmag_now;
                vel_step[2] = vdir[2] * vmag_now;
            } else {
                vel_step[0] = vel_neu[0];
                vel_step[1] = vel_neu[1];
                vel_step[2] = vel_neu[2];
            }
            update_rcv_pos(0.1, llh, xyz, vel_step);
            ltcmat(llh, tmat);

            for (int k = 0; k < nsat; k++) {
                int sv_idx = chan[k].prn - 1;
                if (sv_idx < 0) continue;
                if (use_halo_tracks) {
                    if (!satpos_halo(chan[k].prn, grx.sec, pos, vel, clk)) continue;
                }
                else if (use_cheb) {
                    if (!satpos_cheb(chan[k].prn, grx.sec, pos, vel, clk)) continue;
                }
                else {
                    satpos(eph[sv_idx], grx, pos, vel, clk);
                }
                subVect(los, pos, xyz);
                /* Light-time correction for az/el LOS consistency */
                double tau = normVect(los) / SPEED_OF_LIGHT;
                pos[0] -= vel[0] * tau; pos[1] -= vel[1] * tau; pos[2] -= vel[2] * tau;
                subVect(los, pos, xyz);
                xyz2neu(los, tmat, neu);
                neu2azel(chan[k].azel, neu);
            }
        }

        log_truth_sample(truth_fp, grx.sec - g0.sec, llh, xyz);
        if (orbit_fp) {
            for (i = 0; i < nsat; i++) {
                int sv_idx = chan[i].prn - 1;
                if (sv_idx < 0 || sv_idx >= MAX_SAT) continue;
                if (eph[sv_idx].vflg != 1) continue;
                double pos_k[3], vel_k[3], clk_k[2];
                satpos(eph[sv_idx], grx, pos_k, vel_k, clk_k);
                fprintf(orbit_fp, "%.3f,%d,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f\n",
                    grx.sec - g0.sec, chan[i].prn,
                    pos_k[0], pos_k[1], pos_k[2],
                    vel_k[0], vel_k[1], vel_k[2]);
            }
            fflush(orbit_fp);
        }
        #pragma omp parallel for private(sv, rho, path_loss, gain, I, Q, ip, qp, isamp)
        for (i = 0; i < nsat; i++) {

            // Refresh code phase and data bit counters
            sv = chan[i].prn - 1;

            // Current pseudorange
            if (use_halo_tracks) {
                double pos_h[3], vel_h[3], clk_h[2]={0};
                if (satpos_halo(chan[i].prn, grx.sec, pos_h, vel_h, clk_h)) {
                    double los_h[3];
                    subVect(los_h, pos_h, xyz);
                    double tau_h = normVect(los_h) / SPEED_OF_LIGHT;
                    pos_h[0]-=vel_h[0]*tau_h; pos_h[1]-=vel_h[1]*tau_h; pos_h[2]-=vel_h[2]*tau_h;
                    subVect(los_h, pos_h, xyz);
                    double rgeom_h = normVect(los_h);
                    rho.range = rgeom_h - SPEED_OF_LIGHT*clk_h[0];
                    rho.rate  = dotProd(vel_h, los_h) / (rgeom_h + 1e-9);
                    rho.g = grx;
                    if (rng_fp) fprintf(rng_fp, "$SIMRNG,%.3f,%d,%.3f\n", grx.sec - g0.sec, chan[i].prn, rgeom_h);
                } else { memset(&rho,0,sizeof(rho)); rho.g=grx; }
            }
            else if (use_cheb) {
                double pos_h[3], vel_h[3], clk_h[2]={0};
                if (satpos_cheb(chan[i].prn, grx.sec, pos_h, vel_h, clk_h)) {
                    double los_h[3];
                    subVect(los_h, pos_h, xyz);
                    double tau_h = normVect(los_h) / SPEED_OF_LIGHT;
                    pos_h[0]-=vel_h[0]*tau_h; pos_h[1]-=vel_h[1]*tau_h; pos_h[2]-=vel_h[2]*tau_h;
                    subVect(los_h, pos_h, xyz);
                    double rgeom_h = normVect(los_h);
                    rho.range = rgeom_h - SPEED_OF_LIGHT*clk_h[0];
                    rho.rate  = dotProd(vel_h, los_h) / (rgeom_h + 1e-9);
                    rho.g = grx;
                    if (rng_fp) fprintf(rng_fp, "$SIMRNG,%.3f,%d,%.3f\n", grx.sec - g0.sec, chan[i].prn, rgeom_h);
                } else { memset(&rho,0,sizeof(rho)); rho.g=grx; }
            }
            else {
                // Kepler path + log geometric range
                double pos_k[3], vel_k[3], clk_k[2];
                satpos(eph[sv], grx, pos_k, vel_k, clk_k);
                double losk[3]; subVect(losk, pos_k, xyz);
                double tauk = normVect(losk)/SPEED_OF_LIGHT;
                pos_k[0]-=vel_k[0]*tauk; pos_k[1]-=vel_k[1]*tauk; pos_k[2]-=vel_k[2]*tauk;
                subVect(losk, pos_k, xyz);
                double rgeom_k = normVect(losk);
                computeRange(&rho, eph[sv], grx, xyz);
                if (rng_fp) fprintf(rng_fp, "$SIMRNG,%.3f,%d,%.3f\n", grx.sec - g0.sec, chan[i].prn, rgeom_k);
            }

            if (use_multipath && chan[i].has_multipath_bias) {
                rho.range += chan[i].multipath_bias;
            }

            // Update code phase and data bit counters
            computeCodePhase(&chan[i], rho0[sv], rho, 0.1);

            // Save current pseudorange
            rho0[sv] = rho;

            // Path loss
            path_loss = 5200000.0 / rho.range;

            // Gain
            gain = (int)(path_loss * (double)GAIN_SCALE);

                        // Channel emulator interpolation setup
            double f_carr0 = chan[i].f_carr, I_code0 = chan[i].I.f_code, Q_code0 = chan[i].Q.f_code;
            double f_carr1 = f_carr0,       I_code1 = I_code0,          Q_code1 = Q_code0;
            double gain0_lin = (double)GAIN_SCALE * path_loss;
            double gain1_lin = gain0_lin;
            if (use_chan_emul) {
                gpstime_t g_next = grx;
                g_next.sec += 0.1;
                if (g_next.sec >= SECONDS_IN_WEEK) g_next.sec -= SECONDS_IN_WEEK;
                range_t rho_next;
                if (use_halo_tracks) {
                    double pos_h2[3], vel_h2[3], clk_h2[2]={0};
                    if (satpos_halo(chan[i].prn, g_next.sec, pos_h2, vel_h2, clk_h2)) {
                        double los2[3];
                        subVect(los2, pos_h2, xyz);
                        double tau2 = normVect(los2) / SPEED_OF_LIGHT;
                        pos_h2[0]-=vel_h2[0]*tau2; pos_h2[1]-=vel_h2[1]*tau2; pos_h2[2]-=vel_h2[2]*tau2;
                        subVect(los2, pos_h2, xyz);
                        double rgeom2 = normVect(los2);
                        rho_next.range = rgeom2 - SPEED_OF_LIGHT*clk_h2[0];
                        rho_next.rate  = dotProd(vel_h2, los2) / (rgeom2 + 1e-9);
                        rho_next.g = g_next;
                    } else { memset(&rho_next,0,sizeof(rho_next)); rho_next.g=g_next; }
                } else if (use_cheb) {
                    double pos_h2[3], vel_h2[3], clk_h2[2]={0};
                    if (satpos_cheb(chan[i].prn, g_next.sec, pos_h2, vel_h2, clk_h2)) {
                        double los2[3];
                        subVect(los2, pos_h2, xyz);
                        double tau2 = normVect(los2) / SPEED_OF_LIGHT;
                        pos_h2[0] -= vel_h2[0] * tau2; pos_h2[1] -= vel_h2[1] * tau2; pos_h2[2] -= vel_h2[2] * tau2;
                        subVect(los2, pos_h2, xyz);
                        double rgeom2 = normVect(los2);
                        rho_next.range = rgeom2 - SPEED_OF_LIGHT*clk_h2[0];
                        rho_next.rate  = dotProd(vel_h2, los2) / (rgeom2 + 1e-9);
                        rho_next.g = g_next;
                    } else { memset(&rho_next,0,sizeof(rho_next)); rho_next.g=g_next; }
                } else {
                    computeRange(&rho_next, eph[sv], g_next, xyz);
                }
                if (use_multipath && chan[i].has_multipath_bias) {
                    rho_next.range += chan[i].multipath_bias;
                }
                channel_t tmp = chan[i];
                computeCodePhase(&tmp, rho, rho_next, 0.1);
                f_carr1 = tmp.f_carr;
                I_code1 = tmp.I.f_code;
                Q_code1 = tmp.Q.f_code;
                double path_loss_next = 5200000.0 / rho_next.range;
                gain1_lin = (double)GAIN_SCALE * path_loss_next;
            }
            int denom = (iq_buff_size > 1) ? (iq_buff_size - 1) : 1;
            double df_carr = (f_carr1 - f_carr0) * 10.0;
            double dI_code = (I_code1 - I_code0) * 10.0;
            double dQ_code = (Q_code1 - Q_code0) * 10.0;
                        for (isamp = 0; isamp < iq_buff_size; isamp++) {

                ph = (int)floor(chan[i].carr_phase * 512.0);

                I = chan[i].I.C * chan[i].I.D;
                Q = chan[i].Q.C * chan[i].Q.S * chan[i].Q.T;
                double alpha  = use_chan_emul ? ((double)isamp / (double)denom) : 0.0;
                double gain_s = use_chan_emul ? (gain0_lin + (gain1_lin - gain0_lin) * alpha) : (double)gain;
                int    igain_s = (int)(gain_s + 0.5);
                double t = isamp * delt;
                ip = igain_s * (I * cosT[ph] - Q * sinT[ph]);
                qp = igain_s * (I * sinT[ph] + Q * cosT[ph]);

                if (inv_q == 1 && nbits == 2)
                    qp *= -1; // Inverse Q sign to emulate MAX2771 outputs

                // Store I/Q samples into buffer
                chan[i].iq_buff[isamp * 2] = ip;
                chan[i].iq_buff[isamp * 2 + 1] = qp;

                // Update code phase
                                double I_code_s = use_chan_emul ? (I_code0 + dI_code * (t - 0.05)) : chan[i].I.f_code;
                double Q_code_s = use_chan_emul ? (Q_code0 + dQ_code * (t - 0.05)) : chan[i].Q.f_code;
                chan[i].I.code_phase += I_code_s * delt;
                chan[i].Q.code_phase += Q_code_s * delt;

                if (chan[i].I.code_phase >= 2046.0) {
                    chan[i].I.code_phase -= 2046.0;

                    chan[i].I.ibit++;
                    if (chan[i].I.ibit >= 6000) {
                        chan[i].I.ibit -= 6000;
                        chan[i].I.iframe++;
                    }

                    // Update navigation message data bit
                    chan[i].I.D = -2 * chan[i].I.data[0][chan[i].I.ibit] + 1;
                }

                if (chan[i].Q.code_phase >= 10230.0) {
                    chan[i].Q.code_phase -= 10230.0;

                    chan[i].Q.ibit++;
                    if (chan[i].Q.ibit >= 4) { // Secondary code period = 8 ms = 4 codes
                        chan[i].Q.ibit -= 4;
                        chan[i].Q.ichip++; // Tertiary code chip = Secondary code period

                        // Update tertiary code
                        chan[i].Q.T = -2 * tcode[sv][chan[i].Q.ichip % 1500] + 1;
                    }

                    // Update secondary code
                    chan[i].Q.S = -2 * scode[sv % 4][chan[i].Q.ibit] + 1;
                }

                // Set currnt code chip
                chan[i].I.C = chan[i].I.code[(int)chan[i].I.code_phase];
                chan[i].Q.C = chan[i].Q.code[(int)chan[i].Q.code_phase];

                // Update carrier phase
                                double f_carr_s = use_chan_emul ? (f_carr0 + df_carr * (t - 0.05)) : chan[i].f_carr;
                chan[i].carr_phase += f_carr_s * delt;

                if (chan[i].carr_phase >= 1.0)
                chan[i].carr_phase -= 1.0;
                else if (chan[i].carr_phase < 0.0)
                chan[i].carr_phase += 1.0;
            }
        }

        if (nbits == 2) {
            #pragma omp parallel for private(i, sample, twobitADC)
            for (isamp = 0; isamp < 2 * iq_buff_size; isamp++)
            {
                sample = 0;
                for (i = 0; i < nsat; i++)
                    sample += chan[i].iq_buff[isamp];

                sample += randn() * noise_scale; // Add thermal noise
                sample /= GAIN_SCALE;

                if (sample >= 0) {
                    twobitADC = (sample > thresh) ? +3 : +1;
                }
                else {
                    twobitADC = (sample < -thresh) ? -3 : -1;
                }
                ((signed char*)iq_buff)[isamp] = (signed char)twobitADC;
            }
            fwrite(iq_buff, 1, 2 * iq_buff_size, fp);
        }
        else {
            #pragma omp parallel for private(i, sample)
            for (isamp = 0; isamp < 2 * iq_buff_size; isamp++)
            {
                sample = 0;
                for (i = 0; i < nsat; i++)
                    sample += chan[i].iq_buff[isamp];

                sample /= GAIN_SCALE;

                ((short*)iq_buff)[isamp] = (short)sample;
            }
            fwrite(iq_buff, 2, 2 * iq_buff_size, fp);
        }

        // Update TOI in SB1 at every 12 seconds
        int igrx = (int)(grx.sec*10.0+0.5);
        
        if (igrx%120==110)
        {
            afst.toi++;

            int toi_mod = (afst.toi + 1) % AFS_TOI_WRAP;
            generate_BCH_AFS_SF1(AFS_SB1, 0, toi_mod); // subframe 1 for frame ID 0

            for (i = 0; i < nsat; i++) {
                bitncpy(chan[i].I.data[0] + 68, AFS_SB1, 52);
            }

            //printf("\nUpdate Data Frames: TOI = %d\n", afst.toi);
        }

        // Update receiver time
        grx.sec += 0.1;

        // Update time counter
        printf("\rTime = %4.1f", grx.sec - g0.sec);
        fflush(stdout);
    }

    tend = clock();

    printf("\nDone!\n");

    // Free I/Q buffers
    free(iq_buff);
    for (i = 0; i < nsat; i++) {
        free(chan[i].iq_buff);
    }
        
    // Close output file
    fclose(fp);

    // Close truth file if requested
    if (truth_fp) {
        fclose(truth_fp);
    }
    if (rng_fp) {
        fclose(rng_fp);
    }
    if (orbit_fp) {
        fclose(orbit_fp);
    }

    // Process time
    printf("Process time = %.1f[sec]\n", (double)(tend - tstart) / CLOCKS_PER_SEC);

	return 0;
}


// ------------------------
// Chebyshev export (fit from HALO/Kepler)
// ------------------------
static int lin_solve(int n, double *A, double *b, double *x)
{
    // Gaussian elimination with partial pivoting on A (n x n), solve A x = b
    // A is modified in-place. b not preserved. x output.
    for (int i=0;i<n;i++) x[i]=0.0;
    // forward elimination
    for (int k=0;k<n;k++) {
        // pivot
        int piv = k; double amax=fabs(A[k*n+k]);
        for (int r=k+1;r<n;r++){ double v=fabs(A[r*n+k]); if (v>amax){amax=v;piv=r;} }
        if (amax < 1e-18) return 0;
        if (piv!=k){
            for (int c=k;c<n;c++){ double tmp=A[k*n+c]; A[k*n+c]=A[piv*n+c]; A[piv*n+c]=tmp; }
            double tb=b[k]; b[k]=b[piv]; b[piv]=tb;
        }
        double akk=A[k*n+k];
        for (int r=k+1;r<n;r++){
            double f=A[r*n+k]/akk; if (fabs(f)<1e-22) continue;
            for (int c=k;c<n;c++) A[r*n+c]-=f*A[k*n+c];
            b[r]-=f*b[k];
        }
    }
    // back substitution
    for (int i=n-1;i>=0;i--){
        double sum=b[i];
        for (int c=i+1;c<n;c++) sum-=A[i*n+c]*x[c];
        double aii=A[i*n+i]; if (fabs(aii)<1e-18) return 0; x[i]=sum/aii;
    }
    return 1;
}

static void cheb_build_row(double tau, int N, double *T)
{
    T[0]=1.0; if (N==0) return; T[1]=tau; for (int k=2;k<=N;k++) T[k]=2.0*tau*T[k-1]-T[k-2];
}

static int get_satpos_src(int prn, int week, double tsec, const ephem_t *eph, double pos[3])
{
    double vel[3], clk[2];
    if (use_halo_tracks) {
        if (!satpos_halo(prn, tsec, pos, vel, clk)) return 0;
        return 1;
    }
    int sv = prn-1; if (sv<0 || sv>=MAX_SAT) return 0; if (eph[sv].vflg!=1) return 0;
    gpstime_t g; g.week=week; g.sec=tsec; satpos((ephem_t)eph[sv], g, pos, vel, clk);
    return 1;
}

static int cheb_fit_segment(FILE *cfp, int prn, int week, const ephem_t *eph,
                            double t0, double dt, int N)
{
    const double ds = 0.1; // sampling step
    int M = N+1;
    // accumulate normal equations A = sum(T T^T), bx/by/bz = sum(T * f)
    double *A = (double*)calloc(M*M, sizeof(double));
    double *bx = (double*)calloc(M, sizeof(double));
    double *by = (double*)calloc(M, sizeof(double));
    double *bz = (double*)calloc(M, sizeof(double));
    if (!A||!bx||!by||!bz){ free(A); free(bx); free(by); free(bz); return 0; }
    int ns=0; double Trow[CHEB_MAX_N+1];
    for (double t=t0; t<=t0+dt+1e-9; t+=ds){
        double pos[3]; if (!get_satpos_src(prn, week, t, eph, pos)) continue;
        double tau = 2.0*(t - t0)/dt - 1.0; if (tau<-1.0) tau=-1.0; if (tau>1.0) tau=1.0;
        cheb_build_row(tau, N, Trow);
        // A += T T^T
        for (int i=0;i<M;i++){
            for (int j=0;j<M;j++) A[i*M+j] += Trow[i]*Trow[j];
        }
        for (int i=0;i<M;i++){
            bx[i]+=Trow[i]*pos[0]; by[i]+=Trow[i]*pos[1]; bz[i]+=Trow[i]*pos[2];
        }
        ns++;
    }
    if (ns < M+2){ free(A); free(bx); free(by); free(bz); return 0; }
    // solve for each axis
    double *Acpy = (double*)malloc(M*M*sizeof(double));
    double *rhs = (double*)malloc(M*sizeof(double));
    double *ax = (double*)malloc(M*sizeof(double));
    double *ay = (double*)malloc(M*sizeof(double));
    double *az = (double*)malloc(M*sizeof(double));
    if (!Acpy||!rhs||!ax||!ay||!az){ free(A); free(bx); free(by); free(bz); free(Acpy); free(rhs); free(ax); free(ay); free(az); return 0; }
    memcpy(Acpy,A,M*M*sizeof(double)); memcpy(rhs,bx,M*sizeof(double)); if (!lin_solve(M,Acpy,rhs,ax)){ }
    memcpy(Acpy,A,M*M*sizeof(double)); memcpy(rhs,by,M*sizeof(double)); if (!lin_solve(M,Acpy,rhs,ay)){ }
    memcpy(Acpy,A,M*M*sizeof(double)); memcpy(rhs,bz,M*sizeof(double)); if (!lin_solve(M,Acpy,rhs,az)){ }
    // write line
    fprintf(cfp, "%d %.3f %.3f %d", prn, t0, dt, N);
    for (int i=0;i<M;i++) fprintf(cfp, " %.16g", ax[i]);
    for (int i=0;i<M;i++) fprintf(cfp, " %.16g", ay[i]);
    for (int i=0;i<M;i++) fprintf(cfp, " %.16g", az[i]);
    fprintf(cfp, "\n");
    free(A); free(bx); free(by); free(bz); free(Acpy); free(rhs); free(ax); free(ay); free(az);
    return 1;
}



















