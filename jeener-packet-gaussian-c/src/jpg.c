#include "jpg.h"

#include <math.h>
#include <complex.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

extern void dpotrf_(char *uplo, int *n, double *a, int *lda, int *info);
extern void zheev_(char *job, char *uplo, int *n, double complex *a, int *lda,
                   double *w, double complex *work, int *lwork, double *rwork, int *info);

enum { N = JPG_N_SPINS, SD = JPG_STATE_DIM, CD = JPG_COV_DIM };

static const double POS[8][3] = {
    {0,0,0},{1,0,0},{1,1,0},{0,1,0},
    {0,0,1},{1,0,1},{1,1,1},{0,1,1}
};
static const int NBR_COUNT[8] = {3,3,3,3,3,3,3,3};
static const int NBR[8][3] = {
    {1,3,4}, {0,2,5}, {1,3,6}, {0,2,7},
    {0,5,7}, {1,4,6}, {2,5,7}, {3,4,6}
};
static const int PAIRS[12][2] = {
    {0,1},{0,3},{0,4},{1,2},{1,5},{2,3},
    {2,6},{3,7},{4,5},{4,7},{5,6},{6,7}
};
static const double B_INH[8] = {
    -0.04 * 2.0 * 3.141592653589793238462643383279502884,
    -0.03 * 2.0 * 3.141592653589793238462643383279502884,
    -0.02 * 2.0 * 3.141592653589793238462643383279502884,
    -0.01 * 2.0 * 3.141592653589793238462643383279502884,
     0.01 * 2.0 * 3.141592653589793238462643383279502884,
     0.02 * 2.0 * 3.141592653589793238462643383279502884,
     0.03 * 2.0 * 3.141592653589793238462643383279502884,
     0.04 * 2.0 * 3.141592653589793238462643383279502884
};

static double T4[8][8][3][3];
static pthread_once_t init_once = PTHREAD_ONCE_INIT;

static double dmax(double a, double b) { return a > b ? a : b; }
static double dmin(double a, double b) { return a < b ? a : b; }
static double clip(double x, double a, double b) { return dmin(dmax(x, a), b); }

static double vdot3(const double a[3], const double b[3]) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

static double vnorm3(const double a[3]) {
    return sqrt(vdot3(a, a));
}

static void vcross3(const double a[3], const double b[3], double out[3]) {
    out[0] = a[1]*b[2] - a[2]*b[1];
    out[1] = a[2]*b[0] - a[0]*b[2];
    out[2] = a[0]*b[1] - a[1]*b[0];
}

static void mat_zero(double *A, int n, int m) {
    memset(A, 0, (size_t)n * (size_t)m * sizeof(double));
}

static void mat_eye(double *A, int n) {
    mat_zero(A, n, n);
    for (int i = 0; i < n; i++) A[i*n + i] = 1.0;
}

static void mat_copy(double *dst, const double *src, int n, int m) {
    memcpy(dst, src, (size_t)n * (size_t)m * sizeof(double));
}

static double mat_norm1(const double *A, int n) {
    double best = 0.0;
    for (int j = 0; j < n; j++) {
        double s = 0.0;
        for (int i = 0; i < n; i++) s += fabs(A[i*n + j]);
        if (s > best) best = s;
    }
    return best;
}

static void mat_mul(const double *A, const double *B, double *C, int n) {
    mat_zero(C, n, n);
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            double a = A[i*n+k];
            for (int j = 0; j < n; j++) C[i*n+j] += a*B[k*n+j];
        }
    }
}

static void mat_transpose(const double *A, double *AT, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            AT[j*n + i] = A[i*n + j];
}

static void mat_symmetrize(double *A, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            double v = 0.5 * (A[i*n + j] + A[j*n + i]);
            A[i*n + j] = v;
            A[j*n + i] = v;
        }
    }
}

static int mat_expm_taylor(const double *A, int n, double *E) {
    if (n < 1 || n > 2*CD) return 1;
    for (int i = 0; i < n*n; i++) if (!isfinite(A[i])) return 1;
    double norm = mat_norm1(A, n);
    if (!isfinite(norm)) return 1;
    int s = 0;
    if (norm > 0.5) s = (int)ceil(log(norm / 0.5) / log(2.0));
    if (s < 0) s = 0;
    double scale = ldexp(1.0, -s);

    double B[4*CD*CD], term[4*CD*CD], tmp[4*CD*CD];
    for (int i = 0; i < n*n; i++) B[i] = A[i] * scale;
    mat_eye(E, n);
    mat_eye(term, n);
    for (int k = 1; k <= 80; k++) {
        mat_mul(term, B, tmp, n);
        double invk = 1.0 / (double)k;
        double maxabs = 0.0;
        for (int i = 0; i < n*n; i++) {
            term[i] = tmp[i] * invk;
            E[i] += term[i];
            maxabs = dmax(maxabs, fabs(term[i]));
        }
        if (maxabs < 1e-16) break;
    }
    for (int r = 0; r < s; r++) {
        mat_mul(E, E, tmp, n);
        mat_copy(E, tmp, n, n);
    }
    for (int i = 0; i < n*n; i++) if (!isfinite(E[i])) return 1;
    return 0;
}

static void init_geometry(void) {
    memset(T4, 0, sizeof(T4));
    for (int j = 0; j < N; j++) {
        for (int ii = 0; ii < NBR_COUNT[j]; ii++) {
            int g = NBR[j][ii];
            double r[3] = {
                POS[j][0] - POS[g][0],
                POS[j][1] - POS[g][1],
                POS[j][2] - POS[g][2]
            };
            double rn = vnorm3(r);
            for (int a = 0; a < 3; a++) r[a] /= rn;
            for (int a = 0; a < 3; a++) {
                for (int b = 0; b < 3; b++) {
                    T4[j][g][a][b] = 3.0 * r[a] * r[b] - (a == b ? 1.0 : 0.0);
                }
            }
        }
    }
}

static void ensure_init(void) {
    pthread_once(&init_once, init_geometry);
}

void jpg_default_options(jpg_options *opt) {
    memset(opt, 0, sizeof(*opt));
    opt->tip = 0.05;
    opt->pattern = 0;
    opt->kmax = 0.03;
    opt->ent_nk = 13;
    opt->ent_nt = 17;
    opt->ent_window0 = 0.0;
    opt->ent_window1 = 1.0;
    opt->total_time = 8.0;
    opt->npts = 2401;
    opt->native_nth = 0.0;
    opt->native_nbath = 0.0;
    opt->reg_warn = 1e-6;
    opt->aggregate_mean = 0;
    opt->threads = 0;
    opt->loop.loops = 0;
    opt->loop.dynamic_mix = 1;
}

static void initial_state(const jpg_options *opt, double y[SD]) {
    double mhat[3] = {0.0, -sin(opt->tip), cos(opt->tip)};
    double mnorm = vnorm3(mhat);
    for (int a = 0; a < 3; a++) mhat[a] /= mnorm;
    for (int j = 0; j < N; j++) {
        double s = 1.0;
        if (opt->pattern == 0) {
            int parity = (int)(POS[j][0] + POS[j][1] + POS[j][2]);
            s = (parity % 2) ? -1.0 : 1.0;
        }
        double Nj[3] = {s * opt->nvec[0], s * opt->nvec[1], s * opt->nvec[2]};
        double par = vdot3(Nj, mhat);
        for (int a = 0; a < 3; a++) Nj[a] -= par * mhat[a];
        double nmag = vnorm3(Nj);
        double mmag = sqrt(dmax(1.0 - nmag*nmag, 0.0));
        for (int a = 0; a < 3; a++) {
            y[3*j + a] = mmag * mhat[a];
            y[3*N + 3*j + a] = Nj[a];
        }
    }
}

static void safe_unit(const double v[3], const double fallback[3], double out[3]) {
    double n = vnorm3(v);
    if (isfinite(n) && n > 1e-15) {
        for (int a = 0; a < 3; a++) out[a] = v[a] / n;
        return;
    }
    double nf = vnorm3(fallback);
    if (isfinite(nf) && nf > 1e-15) {
        for (int a = 0; a < 3; a++) out[a] = fallback[a] / nf;
        return;
    }
    out[0] = 0.0; out[1] = 0.0; out[2] = 1.0;
}

static void renorm_packet(double M[3], double Nv[3]) {
    double nF_raw[3], nB_raw[3], fbF[3], fbB[3], nF[3], nB[3];
    for (int a = 0; a < 3; a++) {
        nF_raw[a] = M[a] + Nv[a];
        nB_raw[a] = M[a] - Nv[a];
        fbF[a] = M[a] + Nv[a];
        fbB[a] = M[a] - Nv[a];
    }
    safe_unit(nF_raw, fbF, nF);
    safe_unit(nB_raw, fbB, nB);
    for (int a = 0; a < 3; a++) {
        M[a] = 0.5 * (nF[a] + nB[a]);
        Nv[a] = 0.5 * (nF[a] - nB[a]);
    }
}

static void renorm_state(double y[SD]) {
    for (int j = 0; j < N; j++) {
        double Mj[3], Nj[3];
        for (int a = 0; a < 3; a++) {
            Mj[a] = y[3*j + a];
            Nj[a] = y[3*N + 3*j + a];
        }
        renorm_packet(Mj, Nj);
        for (int a = 0; a < 3; a++) {
            y[3*j + a] = Mj[a];
            y[3*N + 3*j + a] = Nj[a];
        }
    }
}

static double poly_sum(const double c[5], int order, double x, int start) {
    double out = 0.0;
    for (int m = start; m <= order && m <= 4; m++) out += c[m] * pow(x, m);
    return out;
}

static double poly_sum_kappa_edge(const double c[5], int order, double x) {
    double out = 0.0;
    for (int m = 2; m <= order && m <= 4; m++) out += c[m] * pow(x, m - 1);
    return out;
}

static void loop_locals(const jpg_loop_config *cfg, const double M[8][3],
                        const double Nv[8][3], double k,
                        double Bshift[8], double lam[8]) {
    for (int j = 0; j < N; j++) {
        Bshift[j] = 0.0; lam[j] = 0.0;
    }
    if (cfg->loops <= 0 || k == 0.0) return;
    for (int j = 0; j < N; j++) {
        double x = vdot3(Nv[j], Nv[j]);
        Bshift[j] = k * poly_sum(cfg->alpha, cfg->loops, x, 1);
        lam[j] = k * poly_sum(cfg->beta, cfg->loops, x, 1);
        Bshift[j] *= 1.0 + cfg->alpha_k2 * k * k;
        lam[j] *= 1.0 + cfg->beta_k2 * k * k;
    }
    (void)M;
}

static void rhs_mn(const jpg_loop_config *cfg, const double y[SD], double k,
                   double n_amp_base, double dy[SD]) {
    double M[8][3], Nv[8][3], BjM[8][3], BjN[8][3];
    for (int j = 0; j < N; j++) {
        for (int a = 0; a < 3; a++) {
            M[j][a] = y[3*j + a];
            Nv[j][a] = y[3*N + 3*j + a];
            BjM[j][a] = 0.0;
            BjN[j][a] = 0.0;
        }
    }

    double Bshift[8], lam[8];
    loop_locals(cfg, M, Nv, k, Bshift, lam);

    for (int j = 0; j < N; j++) {
        for (int ii = 0; ii < NBR_COUNT[j]; ii++) {
            int g = NBR[j][ii];
            double xavg = 0.5 * (vdot3(Nv[j], Nv[j]) + vdot3(Nv[g], Nv[g]));
            double gmm = 1.0, gmn = 1.0;
            if (cfg->loops > 0) {
                gmm = clip(1.0 + poly_sum(cfg->sigma, cfg->loops, xavg, 1), 1e-3, 10.0);
                if (cfg->loops >= 4) {
                    gmn = clip(1.0 + poly_sum_kappa_edge(cfg->kappa, cfg->loops, xavg), 1e-3, 10.0);
                }
            }
            for (int a = 0; a < 3; a++) {
                double sm = 0.0, sn = 0.0;
                for (int b = 0; b < 3; b++) {
                    sm += T4[j][g][a][b] * M[g][b];
                    sn += T4[j][g][a][b] * Nv[g][b];
                }
                BjM[j][a] += gmm * sm;
                BjN[j][a] += gmn * sn;
            }
        }
    }

    for (int j = 0; j < N; j++) {
        double Bm[3] = {k * BjM[j][0], k * BjM[j][1], B_INH[j] + Bshift[j] + k * BjM[j][2]};
        double kNa = cfg->dynamic_mix ? k * sqrt(dmax(vdot3(Nv[j], Nv[j]), 0.0)) : k * n_amp_base;
        if (cfg->dynamic_mix) kNa = dmin(kNa, 2.0);
        double Bn[3] = {-kNa * BjN[j][0], -kNa * BjN[j][1], -kNa * BjN[j][2]};
        double nb = vnorm3(Bn);
        if (nb > 10.0) for (int a = 0; a < 3; a++) Bn[a] *= 10.0 / nb;

        double c1[3], c2[3], c3[3], c4[3];
        vcross3(M[j], Bm, c1);
        vcross3(Nv[j], Bn, c2);
        vcross3(Nv[j], Bm, c3);
        vcross3(M[j], Bn, c4);
        double lamx = lam[j] * (1.0 + cfg->beta_anis);
        double lamy = lam[j] * (1.0 - cfg->beta_anis);
        for (int a = 0; a < 3; a++) {
            dy[3*j + a] = c1[a] + c2[a];
            dy[3*N + 3*j + a] = c3[a] + c4[a];
        }
        dy[3*j + 0] -= lamx * M[j][0];
        dy[3*j + 1] -= lamy * M[j][1];
        dy[3*N + 3*j + 0] -= lamx * Nv[j][0];
        dy[3*N + 3*j + 1] -= lamy * Nv[j][1];
    }
}

static void tangent_basis(const double m[3], double e1[3], double e2[3]) {
    double nh[3] = {m[0], m[1], m[2]};
    double nn = vnorm3(nh);
    if (nn < 1e-12) { nh[0] = 0; nh[1] = 0; nh[2] = 1; }
    else for (int a = 0; a < 3; a++) nh[a] /= nn;
    double ref[3] = {0, 0, 1};
    if (fabs(nh[2]) >= 0.9) { ref[0] = 1; ref[1] = 0; ref[2] = 0; }
    double dot = vdot3(ref, nh);
    for (int a = 0; a < 3; a++) e1[a] = ref[a] - dot * nh[a];
    double e1n = vnorm3(e1);
    if (e1n < 1e-12) { e1[0] = 0; e1[1] = 1; e1[2] = 0; e1n = 1; }
    for (int a = 0; a < 3; a++) e1[a] /= e1n;
    vcross3(nh, e1, e2);
    double e2n = vnorm3(e2);
    for (int a = 0; a < 3; a++) e2[a] /= e2n;
}

static void state_tangent_basis(const double y[SD], double inject[SD*CD], double proj[CD*(3*N)]) {
    mat_zero(inject, SD, CD);
    mat_zero(proj, CD, 3*N);
    for (int j = 0; j < N; j++) {
        double m[3] = { y[3*j], y[3*j+1], y[3*j+2] };
        double e1[3], e2[3];
        tangent_basis(m, e1, e2);
        double scale = sqrt(dmax(vnorm3(m), 1e-15));
        int cm = 2*j;
        for (int a = 0; a < 3; a++) {
            inject[(3*j+a)*CD + cm] = scale * e1[a];
            inject[(3*j+a)*CD + cm + 1] = scale * e2[a];
            proj[cm*(3*N) + 3*j+a] = e1[a] / scale;
            proj[(cm+1)*(3*N) + 3*j+a] = e2[a] / scale;
        }
        /* Rotate the response with M so the injected displacement is tangent. */
        const double *Nv = y + 3*N + 3*j;
        double d1 = scale*vdot3(Nv, e1)/vdot3(m, m);
        double d2 = scale*vdot3(Nv, e2)/vdot3(m, m);
        for (int a = 0; a < 3; a++) {
            inject[(3*N+3*j+a)*CD+cm] = -m[a]*d1;
            inject[(3*N+3*j+a)*CD+cm+1] = -m[a]*d2;
        }
    }
}

static void packet_drift_matrix(const jpg_options *opt, const double y[SD], double k, double A[CD*CD]) {
    double inject[SD*CD], proj[CD*(3*N)];
    state_tangent_basis(y, inject, proj);
    mat_zero(A, CD, CD);
    double n_amp = sqrt(opt->nvec[0]*opt->nvec[0] + opt->nvec[1]*opt->nvec[1] + opt->nvec[2]*opt->nvec[2]);
    double eps = 1e-6;
    for (int col = 0; col < CD; col++) {
        double yp[SD], ym[SD], fp[SD], fm[SD];
        for (int i = 0; i < SD; i++) {
            double d = eps * inject[i*CD + col];
            yp[i] = y[i] + d;
            ym[i] = y[i] - d;
        }
        renorm_state(yp);
        renorm_state(ym);
        rhs_mn(&opt->loop, yp, k, n_amp, fp);
        rhs_mn(&opt->loop, ym, k, n_amp, fm);
        for (int row = 0; row < CD; row++) {
            double s = 0.0;
            for (int i = 0; i < 3*N; i++) {
                double dmdot = (fp[i] - fm[i]) / (2.0 * eps);
                s += proj[row*(3*N) + i] * dmdot;
            }
            A[row*CD + col] = s;
        }
    }
}

static void packet_diffusion_matrix(const jpg_options *opt, const double y[SD], double k, double D[CD*CD]) {
    double inject[SD*CD], proj[CD*(3*N)];
    state_tangent_basis(y, inject, proj);
    mat_zero(D, CD, CD);
    double M[8][3], Nv[8][3];
    for (int j = 0; j < N; j++) {
        for (int a = 0; a < 3; a++) {
            M[j][a] = y[3*j+a];
            Nv[j][a] = y[3*N+3*j+a];
        }
    }
    double Bshift[8], lam[8];
    loop_locals(&opt->loop, M, Nv, k, Bshift, lam);
    double therm = 2.0 * dmax(opt->native_nbath, 0.0) + 1.0;
    for (int j = 0; j < N; j++) {
        int blk = 2*j;
        double rate = dmax(lam[j], 0.0);
        double lamx = dmax(rate * (1.0 + opt->loop.beta_anis), 0.0);
        double lamy = dmax(rate * (1.0 - opt->loop.beta_anis), 0.0);
        double Dlab[3][3] = {{therm*lamx,0,0},{0,therm*lamy,0},{0,0,0}};
        for (int a = 0; a < 2; a++) {
            for (int b = 0; b < 2; b++) {
                double s = 0.0;
                for (int u = 0; u < 3; u++) {
                    for (int v = 0; v < 3; v++) {
                        s += proj[(blk+a)*(3*N) + 3*j+u] * Dlab[u][v] * inject[(3*j+v)*CD + blk+b];
                    }
                }
                D[(blk+a)*CD + blk+b] = s;
            }
        }
    }
    mat_symmetrize(D, CD);
    (void)Bshift;
}

static int gaussian_step(const double A[CD*CD], const double D[CD*CD], double dt,
                               double Phi[CD*CD], double Qd[CD*CD], int *has_qd) {
    double dnorm = mat_norm1(D, CD);
    if (dnorm < 1e-18) {
        double Adt[CD*CD];
        for (int i = 0; i < CD*CD; i++) Adt[i] = A[i] * dt;
        if (mat_expm_taylor(Adt, CD, Phi)) return 1;
        mat_zero(Qd, CD, CD);
        *has_qd = 0;
        return 0;
    }
    int n2 = 2 * CD;
    double M[4*CD*CD] = {0}, E[4*CD*CD], top[CD*CD], PT[CD*CD];
    for (int i = 0; i < CD; i++) {
        for (int j = 0; j < CD; j++) {
            M[i*n2 + j] = A[i*CD + j];
            M[i*n2 + CD + j] = D[i*CD + j];
            M[(CD+i)*n2 + CD+j] = -A[j*CD + i];
        }
    }
    for (int i = 0; i < n2*n2; i++) M[i] *= dt;
    if (mat_expm_taylor(M, n2, E)) return 1;
    for (int i = 0; i < CD; i++) {
        for (int j = 0; j < CD; j++) {
            Phi[i*CD + j] = E[i*n2 + j];
            top[i*CD + j] = E[i*n2 + CD + j];
        }
    }
    mat_transpose(Phi, PT, CD);
    mat_mul(top, PT, Qd, CD);
    mat_symmetrize(Qd, CD);
    *has_qd = 1;
    return 0;
}

static void cov_step(double V[CD*CD], const double Phi[CD*CD], const double Qd[CD*CD], int has_qd) {
    double tmp[CD*CD], PT[CD*CD], out[CD*CD];
    mat_transpose(Phi, PT, CD);
    mat_mul(Phi, V, tmp, CD);
    mat_mul(tmp, PT, out, CD);
    if (has_qd) {
        for (int i = 0; i < CD*CD; i++) out[i] += Qd[i];
    }
    mat_symmetrize(out, CD);
    mat_copy(V, out, CD, CD);
}

static void symplectic_form(double *Om, int nm) {
    int n = 2 * nm;
    mat_zero(Om, n, n);
    for (int j = 0; j < nm; j++) {
        Om[(2*j)*n + (2*j+1)] = 1.0;
        Om[(2*j+1)*n + (2*j)] = -1.0;
    }
}

static int hermitian_eigs(double complex *A, int n, double *w) {
    double complex work[64*CD];
    double rwork[3*CD];
    int lwork = 64*CD, info = 0;
    char job = 'N', uplo = 'L';
    if (n < 1 || n > CD) return 1;
    for (int i = 0; i < n*n; i++)
        if (!isfinite(creal(A[i])) || !isfinite(cimag(A[i]))) return 1;
    zheev_(&job, &uplo, &n, A, &n, w, work, &lwork, rwork, &info);
    return info;
}

static int symplectic_eigs(const double *V, int nm, double *nu) {
    int n = 2*nm, info = 0;
    double L[CD*CD], w[CD];
    double complex H[CD*CD];
    char uplo = 'L';
    if (nm < 1 || nm > N) return 1;
    for (int r = 0; r < n; r++)
        for (int c = 0; c < n; c++) {
            L[c*n+r] = 0.5*(V[r*n+c] + V[c*n+r]);
            if (!isfinite(L[c*n+r])) return 1;
        }
    dpotrf_(&uplo, &n, L, &n, &info);
    if (info) return info;
    for (int c = 0; c < n; c++)
        for (int r = 0; r < c; r++) L[c*n+r] = 0;
    /* i L^T Omega L is Hermitian and has the Williamson eigenvalues +/- nu. */
    for (int r = 0; r < n; r++)
        for (int c = 0; c < n; c++) {
            double sum = 0;
            for (int j = 0; j < nm; j++)
                sum += L[r*n+2*j]*L[c*n+2*j+1] - L[r*n+2*j+1]*L[c*n+2*j];
            H[c*n+r] = I*sum;
        }
    info = hermitian_eigs(H, n, w);
    if (info) return info;
    for (int j = 0; j < nm; j++) nu[j] = 0.5*(w[nm+j] - w[nm-1-j]);
    return 0;
}

static void subcov(const double *V, const int *modes, int nm, double *S) {
    int n = 2 * nm;
    for (int a = 0; a < nm; a++) {
        for (int qa = 0; qa < 2; qa++) {
            int ia = 2*modes[a] + qa;
            for (int b = 0; b < nm; b++) {
                for (int qb = 0; qb < 2; qb++) {
                    int ib = 2*modes[b] + qb;
                    S[(2*a+qa)*n + (2*b+qb)] = V[ia*CD + ib];
                }
            }
        }
    }
}

static void partial_transpose(double *V, int nm, int mode) {
    int n = 2 * nm;
    int p = 2*mode + 1;
    for (int j = 0; j < n; j++) {
        V[p*n + j] *= -1.0;
        V[j*n + p] *= -1.0;
    }
}

static double physicality_margin(const double *V, int nm) {
    int n = 2*nm;
    double complex H[CD*CD];
    double w[CD];
    if (nm < 1 || nm > N) return NAN;
    for (int r = 0; r < n; r++)
        for (int c = 0; c < n; c++)
            H[c*n+r] = 0.5*(V[r*n+c] + V[c*n+r]);
    for (int j = 0; j < nm; j++) {
        H[(2*j+1)*n+2*j] += 0.5*I;
        H[(2*j)*n+2*j+1] -= 0.5*I;
    }
    return hermitian_eigs(H, n, w) ? NAN : w[0];
}

static double regularize(double *V, int nm) {
    int n = 2 * nm;
    double margin = physicality_margin(V, nm);
    if (!isfinite(margin)) return NAN;
    if (margin >= -1e-10) return 0.0;
    double delta = -margin + 1e-9;
    for (int i = 0; i < n; i++) V[i*n + i] += delta;
    return delta;
}

static int logneg_cov(const double *V, int nm, int pt_mode, double *nu_min, double *ln) {
    int n = 2 * nm;
    double W[64], nu[4];
    if (nm < 1 || nm > 4 || pt_mode < 0 || pt_mode >= nm) return 1;
    mat_copy(W, V, n, n);
    partial_transpose(W, nm, pt_mode);
    if (symplectic_eigs(W, nm, nu)) return 1;
    *nu_min = nu[0];
    *ln = 0.0;
    for (int i = 0; i < nm; i++) {
        double val = 2.0 * dmax(nu[i], 1e-16);
        double add = -log(val) / log(2.0);
        if (nu[i] < 0.5 - 1e-10 && add > 0.0) *ln += add;
    }
    return 0;
}

static double duan_pair(const double V[16]) {
    double e[4] = {1,0,1,0};
    double f[4] = {0,1,0,-1};
    double se = 0.0, sf = 0.0;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            se += e[i] * V[i*4+j] * e[j];
            sf += f[i] * V[i*4+j] * f[j];
        }
    }
    return se + sf;
}

static int row_metrics_for_cov(const double Vfull[CD*CD], int aggregate_mean,
                                double *pair_ln, double *pair_nu,
                                double *pair_du, double *multi_ln, double *multi_fi,
                                double *multi_nu, double *added_noise) {
    double V[CD*CD];
    mat_copy(V, Vfull, CD, CD);
    *added_noise = regularize(V, N);
    if (!isfinite(*added_noise)) return 1;

    *pair_ln = 0.0;
    *pair_nu = aggregate_mean ? 0.0 : INFINITY;
    *pair_du = aggregate_mean ? 0.0 : INFINITY;
    for (int p = 0; p < 12; p++) {
        int modes[2] = {PAIRS[p][0], PAIRS[p][1]};
        double S[16], nu, ln;
        subcov(V, modes, 2, S);
        if (logneg_cov(S, 2, 0, &nu, &ln)) return 1;
        double du = duan_pair(S);
        if (aggregate_mean) {
            *pair_ln += ln/12.0;
            *pair_nu += nu/12.0;
            *pair_du += du/12.0;
        } else {
            *pair_ln = dmax(*pair_ln, ln);
            *pair_nu = dmin(*pair_nu, nu);
            *pair_du = dmin(*pair_du, du);
        }
    }

    *multi_ln = 0.0;
    *multi_fi = 0.0;
    *multi_nu = aggregate_mean ? 0.0 : INFINITY;
    for (int a = 0; a < N - 2; a++) {
        for (int b = a + 1; b < N - 1; b++) {
            for (int c = b + 1; c < N; c++) {
                int modes[3] = {a,b,c};
                double S[36], nu, ln;
                subcov(V, modes, 3, S);
                double worst_nu = 0.0, weakest_ln = INFINITY;
                for (int mode = 0; mode < 3; mode++) {
                    if (logneg_cov(S, 3, mode, &nu, &ln)) return 1;
                    worst_nu = dmax(worst_nu, nu);
                    weakest_ln = dmin(weakest_ln, ln);
                }
                double fi = worst_nu < 0.5 - 1e-10 ? 0.5 - worst_nu : 0.0;
                if (aggregate_mean) {
                    *multi_ln += weakest_ln/56.0;
                    *multi_fi += fi/56.0;
                    *multi_nu += worst_nu/56.0;
                } else {
                    *multi_ln = dmax(*multi_ln, weakest_ln);
                    *multi_fi = dmax(*multi_fi, fi);
                    *multi_nu = dmin(*multi_nu, worst_nu);
                }
            }
        }
    }
    return 0;
}

typedef struct {
    const jpg_options *opt;
    jpg_result *res;
    int next;
    int failed;
    pthread_mutex_t lock;
} work_state;

static double generator_cp_margin(const double A[CD*CD], const double D[CD*CD]) {
    double Om[CD*CD], AO[CD*CD], w[CD];
    double complex H[CD*CD];
    symplectic_form(Om, N);
    mat_mul(A, Om, AO, CD);
    for (int r = 0; r < CD; r++)
        for (int c = 0; c < CD; c++)
            H[c*CD+r] = D[r*CD+c] - 0.5*I*(AO[r*CD+c] - AO[c*CD+r]);
    return hermitian_eigs(H, CD, w) ? NAN : w[0];
}

static int compute_row(const jpg_options *opt, int ik, jpg_row_result *row) {
    memset(row, 0, sizeof(*row));
    double k = opt->ent_nk <= 1 ? 0.0 : opt->kmax * (double)ik / (double)(opt->ent_nk - 1);
    row->k = k;
    int start = (int)(opt->ent_window0 * (double)(opt->npts - 1));
    int end = (int)(opt->ent_window1 * (double)(opt->npts - 1));
    if (end < start) end = start;
    int t_idx[JPG_MAX_T];
    for (int jt = 0; jt < opt->ent_nt; jt++) {
        double step = opt->ent_nt <= 1 ? 0.0 : (double)(end-start)/(double)(opt->ent_nt-1);
        t_idx[jt] = (int)((double)start + jt*step);
        if (opt->ent_nt > 1 && jt == opt->ent_nt-1) t_idx[jt] = end;
    }

    double y[SD], Phi[CD*CD], Qd[CD*CD], V[CD*CD];
    int has_qd = 0;
    initial_state(opt, y);
    renorm_state(y);
    double response_max = 0;
    for (int j = 0; j < N; j++) response_max = dmax(response_max, vnorm3(y+3*N+3*j));
    row->response_invariant = response_max <= 1e-15;
    double dt = opt->total_time / (double)(opt->npts - 1);
    double A[CD*CD], D[CD*CD];
    packet_drift_matrix(opt, y, k, A);
    packet_diffusion_matrix(opt, y, k, D);
    row->generator_cp_margin = generator_cp_margin(A, D);
    if (!isfinite(row->generator_cp_margin)) return 1;

    mat_zero(V, CD, CD);
    double qv = dmax(opt->native_nth, 0.0) + 0.5;
    for (int j = 0; j < CD; j++) V[j*CD + j] = qv;

    double pair_end = 0, multi_end = 0;
    double pair_best = 0, multi_best = 0;
    double pair_sum = 0, multi_sum = 0;
    double pair_nu_min = 1e100, pair_du_min = 1e100, multi_nu_min = 1e100;
    double add_max = 0.0;
    int ti = 0, count = 0, cached_gap = -1;
    row->uncertainty_margin_min = INFINITY;
    for (int jt = 0; jt < opt->ent_nt; jt++) {
        int gap = t_idx[jt] - ti;
        if (gap) {
            if (gap != cached_gap) {
                if (gaussian_step(A, D, gap*dt, Phi, Qd, &has_qd)) return 1;
                cached_gap = gap;
            }
            cov_step(V, Phi, Qd, has_qd);
        }
        ti = t_idx[jt];
        double margin = physicality_margin(V, N);
        if (!isfinite(margin)) return 1;
        row->uncertainty_margin_min = dmin(row->uncertainty_margin_min, margin);
        {
            double pln, pnu, pdu, mln, mfi, mnu, add;
            if (row_metrics_for_cov(V, opt->aggregate_mean, &pln, &pnu, &pdu, &mln, &mfi, &mnu, &add)) return 1;
            add_max = dmax(add_max, add);
            if (row->response_invariant && (pln > 1e-10 || mfi > 1e-10)) row->null_control_violation = 1;
            if (add > opt->reg_warn || row->generator_cp_margin < -1e-8) continue;
            pair_end = pln;
            multi_end = mfi;
            pair_best = dmax(pair_best, pln);
            multi_best = dmax(multi_best, mfi);
            pair_sum += pln;
            multi_sum += mfi;
            pair_nu_min = dmin(pair_nu_min, pnu);
            pair_du_min = dmin(pair_du_min, pdu);
            multi_nu_min = dmin(multi_nu_min, mnu);
            count++;
        }
    }

    row->pair_end = pair_end;
    row->pair_best = pair_best;
    row->pair_mean = count > 0 ? pair_sum / count : 0.0;
    row->pair_nu_min = pair_nu_min;
    row->pair_duan_min = pair_du_min;
    row->multi_end = multi_end;
    row->multi_best = multi_best;
    row->multi_mean = count > 0 ? multi_sum / count : 0.0;
    row->multi_nu_min = multi_nu_min;
    row->added_noise_max = add_max;
    row->valid_count = count;
    row->sample_count = opt->ent_nt;
    if (!count) {
        row->pair_end = row->pair_best = row->pair_mean = NAN;
        row->pair_nu_min = row->pair_duan_min = NAN;
        row->multi_end = row->multi_best = row->multi_mean = row->multi_nu_min = NAN;
    }
    return 0;
}

static void *worker_main(void *arg) {
    work_state *ws = (work_state *)arg;
    for (;;) {
        pthread_mutex_lock(&ws->lock);
        int ik = ws->next++;
        pthread_mutex_unlock(&ws->lock);
        if (ik >= ws->opt->ent_nk) break;
        if (compute_row(ws->opt, ik, &ws->res->rows[ik])) {
            pthread_mutex_lock(&ws->lock);
            ws->failed = 1;
            pthread_mutex_unlock(&ws->lock);
        }
    }
    return NULL;
}

int jpg_run_benchmark(const jpg_options *opt_in, jpg_result *out) {
    if (!opt_in || !out) return 1;
    ensure_init();
    jpg_options opt = *opt_in;
    if (opt.ent_nk < 1 || opt.ent_nk > JPG_MAX_K) return 2;
    if (opt.ent_nt < 1 || opt.ent_nt > JPG_MAX_T) return 3;
    if (opt.npts < 2) return 4;
    if (opt.loop.loops < 0 || opt.loop.loops > 4 || opt.threads < 0) return 4;
    if (!isfinite(opt.kmax) || opt.kmax < 0 || !isfinite(opt.total_time) || opt.total_time < 0
        || !isfinite(opt.tip) || !isfinite(opt.native_nth) || opt.native_nth < 0
        || !isfinite(opt.native_nbath) || opt.native_nbath < 0
        || !isfinite(opt.reg_warn) || opt.reg_warn < 0
        || !isfinite(opt.ent_window0) || !isfinite(opt.ent_window1)
        || opt.ent_window0 < 0 || opt.ent_window1 > 1 || opt.ent_window1 < opt.ent_window0
        || (opt.pattern != 0 && opt.pattern != 1)
        || (opt.aggregate_mean != 0 && opt.aggregate_mean != 1)
        || (opt.loop.dynamic_mix != 0 && opt.loop.dynamic_mix != 1)) return 4;
    for (int a = 0; a < 3; a++) if (!isfinite(opt.nvec[a])) return 4;
    double mhat[3] = {0, -sin(opt.tip), cos(opt.tip)}, nperp[3];
    double parallel = vdot3(opt.nvec, mhat);
    for (int a = 0; a < 3; a++) nperp[a] = opt.nvec[a] - parallel*mhat[a];
    if (!isfinite(vnorm3(nperp)) || vnorm3(nperp) >= 1.0) return 4;
    if (!isfinite(opt.loop.alpha_k2) || !isfinite(opt.loop.beta_k2)
        || !isfinite(opt.loop.beta_anis)) return 4;
    for (int a = 0; a < 5; a++)
        if (!isfinite(opt.loop.alpha[a]) || !isfinite(opt.loop.beta[a])
            || !isfinite(opt.loop.sigma[a]) || !isfinite(opt.loop.kappa[a])) return 4;

    out->nk = opt.ent_nk;
    out->nt = opt.ent_nt;

    int threads = opt.threads;
    if (threads <= 0) {
        long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
        threads = ncpu > 0 ? (int)ncpu : 1;
    }
    if (threads < 1) threads = 1;
    if (threads > opt.ent_nk) threads = opt.ent_nk;

    work_state ws;
    ws.opt = &opt;
    ws.res = out;
    ws.next = 0;
    ws.failed = 0;
    if (pthread_mutex_init(&ws.lock, NULL)) return 5;
    pthread_t *th = calloc((size_t)threads, sizeof(pthread_t));
    int created = 0;
    if (th) {
        for (; created < threads; created++)
            if (pthread_create(&th[created], NULL, worker_main, &ws)) break;
    }
    if (created < threads) worker_main(&ws);
    for (int i = 0; i < created; i++) {
        if (pthread_join(th[i], NULL)) {
            pthread_mutex_lock(&ws.lock);
            ws.failed = 1;
            pthread_mutex_unlock(&ws.lock);
        }
    }
    free(th);
    pthread_mutex_destroy(&ws.lock);
    return ws.failed ? 5 : 0;
}

static void json_number(const char *key, double value, int comma) {
    printf("\"%s\": ", key);
    if (isfinite(value)) printf("%.17g", value);
    else printf("null");
    if (comma) printf(", ");
}

void jpg_print_json(const jpg_options *opt, const jpg_result *res) {
    printf("{\n");
    printf("  \"program\": \"jeener_packet_c\",\n");
    printf("  \"model\": \"frozen-native-packet-gaussian\",\n");
    printf("  \"schema_version\": 2,\n");
    printf("  \"interpretation\": \"computed covariance; paper null-control discrepancy unresolved\",\n");
    printf("  \"aggregate\": \"%s\",\n", opt->aggregate_mean ? "mean" : "max");
    printf("  \"backend\": \"cpu-pthreads\",\n");
    printf("  \"parameters\": {\"kmax\": %.17g, \"T\": %.17g, \"npts\": %d, \"tip\": %.17g, "
           "\"pattern\": \"%s\", \"window\": [%.17g, %.17g], \"nth\": %.17g, \"nbath\": %.17g, "
           "\"reg_warn\": %.17g, \"dynamic_mix\": %d, \"alpha_k2\": %.17g, \"beta_k2\": %.17g, \"beta_anis\": %.17g, ",
           opt->kmax, opt->total_time, opt->npts, opt->tip, opt->pattern ? "uniform" : "staggered",
           opt->ent_window0, opt->ent_window1, opt->native_nth, opt->native_nbath, opt->reg_warn,
           opt->loop.dynamic_mix, opt->loop.alpha_k2, opt->loop.beta_k2, opt->loop.beta_anis);
    const char *channels[] = {"alpha", "beta", "sigma", "kappa"};
    const double *coeffs[] = {opt->loop.alpha, opt->loop.beta, opt->loop.sigma, opt->loop.kappa};
    for (int c = 0; c < 4; c++) {
        printf("\"%s\": [", channels[c]);
        for (int j = c == 3 ? 2 : 1; j <= 4; j++) printf("%.17g%s", coeffs[c][j], j == 4 ? "" : ", ");
        printf("]%s", c == 3 ? "},\n" : ", ");
    }
    printf("  \"nvec\": [%.17g, %.17g, %.17g],\n", opt->nvec[0], opt->nvec[1], opt->nvec[2]);
    printf("  \"loops\": %d,\n", opt->loop.loops);
    printf("  \"ent_nk\": %d,\n", res->nk);
    printf("  \"ent_nt\": %d,\n", res->nt);
    printf("  \"rows\": [\n");
    for (int i = 0; i < res->nk; i++) {
        const jpg_row_result *r = &res->rows[i];
        const char *keys[] = {"k", "pair_end", "pair_best", "pair_mean", "pair_nu_min", "pair_duan_min",
            "multi_end", "multi_best", "multi_mean", "multi_nu_min", "added_noise_max",
            "uncertainty_margin_min", "generator_cp_margin"};
        double vals[] = {r->k, r->pair_end, r->pair_best, r->pair_mean, r->pair_nu_min, r->pair_duan_min,
            r->multi_end, r->multi_best, r->multi_mean, r->multi_nu_min, r->added_noise_max,
            r->uncertainty_margin_min, r->generator_cp_margin};
        printf("    {");
        for (size_t j = 0; j < sizeof(vals)/sizeof(vals[0]); j++) json_number(keys[j], vals[j], 1);
        printf("\"valid_count\": %d, \"sample_count\": %d, \"response_invariant\": %s, \"null_control_violation\": %s}%s\n",
            r->valid_count, r->sample_count, r->response_invariant ? "true" : "false",
            r->null_control_violation ? "true" : "false", i == res->nk - 1 ? "" : ",");
    }
    const jpg_row_result *last = &res->rows[res->nk - 1];
    printf("  ],\n");
    printf("  \"summary\": {");
    json_number("pair_endpoint", last->pair_end, 1);
    json_number("pair_mean_endpoint_row", last->pair_mean, 1);
    json_number("multi_endpoint", last->multi_end, 1);
    json_number("multi_mean_endpoint_row", last->multi_mean, 0);
    printf("}\n");
    printf("}\n");
}

void jpg_print_csv(const jpg_options *opt, const jpg_result *res) {
    (void)opt;
    printf("k,pair_end,pair_best,pair_mean,pair_nu_min,pair_duan_min,multi_end,multi_best,multi_mean,multi_nu_min,added_noise_max,uncertainty_margin_min,generator_cp_margin,valid_count,sample_count,response_invariant,null_control_violation\n");
    for (int i = 0; i < res->nk; i++) {
        const jpg_row_result *r = &res->rows[i];
        printf("%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%d,%d,%d,%d\n",
               r->k, r->pair_end, r->pair_best, r->pair_mean, r->pair_nu_min, r->pair_duan_min,
               r->multi_end, r->multi_best, r->multi_mean, r->multi_nu_min, r->added_noise_max,
               r->uncertainty_margin_min, r->generator_cp_margin, r->valid_count, r->sample_count,
               r->response_invariant, r->null_control_violation);
    }
}
