#include "../src/jpg.c"
#include <assert.h>

static void near(double actual, double expected, double tol) {
    if (!isfinite(actual) || fabs(actual-expected) > tol) {
        fprintf(stderr, "expected %.17g, got %.17g\n", expected, actual);
        exit(1);
    }
}

int main(void) {
    double V[CD*CD] = {0}, nu[N], ln, nmin;
    for (int j = 0; j < CD; j++) V[j*CD+j] = 0.5;
    assert(!symplectic_eigs(V,N,nu));
    for (int j = 0; j < N; j++) near(nu[j],0.5,1e-13);
    near(physicality_margin(V,N),0,1e-13);
    double bad[4] = {-1,0,0,-1};
    near(physicality_margin(bad,1),-1.5,1e-13);
    assert(symplectic_eigs(bad,1,nu));
    near(regularize(bad,1),1.5+1e-9,1e-13);
    assert(physicality_margin(bad,1) >= 0);

    double r = 0.7, c = cosh(2*r)/2, s = sinh(2*r)/2;
    double pair[16] = {c,0,s,0, 0,c,0,-s, s,0,c,0, 0,-s,0,c};
    assert(!logneg_cov(pair,2,0,&nmin,&ln));
    near(nmin,exp(-2*r)/2,1e-13);
    near(ln,2*r/log(2.0),1e-12);
    for (int a = 0; a < 4; a++)
        for (int b = 0; b < 4; b++) V[a*CD+b] = pair[a*4+b];
    double pln,pnu,pdu,mln,mfi,mnu,noise;
    assert(!row_metrics_for_cov(V,0,&pln,&pnu,&pdu,&mln,&mfi,&mnu,&noise));
    near(pln,2*r/log(2.0),1e-12);
    near(mfi,0,1e-12);
    near(mln,0,1e-12);

    double A[4] = {0,-0.3,0.3,0}, E[4];
    assert(!mat_expm_taylor(A,2,E));
    near(E[0],cos(0.3),1e-14);
    near(E[1],-sin(0.3),1e-14);
    A[0] = NAN;
    assert(mat_expm_taylor(A,2,E));

    double drift[CD*CD] = {0}, diffusion[CD*CD] = {0}, phi[CD*CD], q[CD*CD];
    for (int j = 0; j < CD; j++) { drift[j*CD+j] = -0.3; diffusion[j*CD+j] = 0.48; }
    int has_q;
    assert(!gaussian_step(drift,diffusion,2.0,phi,q,&has_q));
    assert(has_q);
    for (int j = 0; j < CD*CD; j++) {
        near(phi[j], j%(CD+1) == 0 ? exp(-0.6) : 0,1e-13);
        near(q[j], j%(CD+1) == 0 ? 0.8*(-expm1(-1.2)) : 0,1e-13);
    }
    assert(generator_cp_margin(drift,diffusion) > 0);
    memset(diffusion,0,sizeof(diffusion));
    assert(generator_cp_margin(drift,diffusion) < 0);

    jpg_options o;
    jpg_result *result = malloc(sizeof(*result));
    assert(result);
    jpg_default_options(&o);
    o.ent_nk = 2; o.ent_nt = 7; o.npts = 3; o.total_time = 0.2;
    assert(!jpg_run_benchmark(&o,result));
    assert(result->rows[1].valid_count == 7);
    assert(result->rows[1].response_invariant);
    assert(result->rows[1].null_control_violation);
    o.nvec[0] = 1.1;
    assert(jpg_run_benchmark(&o,result));
    o.nvec[0] = 0; o.ent_window0 = -0.1;
    assert(jpg_run_benchmark(&o,result));
    free(result);
    puts("C analytic covariance, partition, sampling and domain tests passed");
    return 0;
}
