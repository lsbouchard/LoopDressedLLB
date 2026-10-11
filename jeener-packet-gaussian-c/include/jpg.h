#ifndef JPG_H
#define JPG_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define JPG_N_SPINS 8
#define JPG_VEC3 3
#define JPG_STATE_DIM (2 * JPG_N_SPINS * JPG_VEC3)
#define JPG_COV_DIM (2 * JPG_N_SPINS)
#define JPG_MAX_K 4096
#define JPG_MAX_T 4096

typedef struct {
    int loops;
    double alpha[5];
    double beta[5];
    double sigma[5];
    double kappa[5];
    double alpha_k2;
    double beta_k2;
    double beta_anis;
    int dynamic_mix;
} jpg_loop_config;

typedef struct {
    double nvec[3];
    double tip;
    int pattern; /* 0 staggered, 1 uniform */
    double kmax;
    int ent_nk;
    int ent_nt;
    double ent_window0;
    double ent_window1;
    double total_time;
    int npts;
    double native_nth;
    double native_nbath;
    double reg_warn;
    int aggregate_mean;
    int threads;
    jpg_loop_config loop;
} jpg_options;

typedef struct {
    double k;
    double pair_end;
    double pair_best;
    double pair_mean;
    double pair_nu_min;
    double pair_duan_min;
    double multi_end;
    double multi_best;
    double multi_mean;
    double multi_nu_min;
    double added_noise_max;
    double uncertainty_margin_min;
    double generator_cp_margin;
    int valid_count;
    int sample_count;
    int response_invariant;
    int null_control_violation;
} jpg_row_result;

typedef struct {
    int nk;
    int nt;
    jpg_row_result rows[JPG_MAX_K];
} jpg_result;

void jpg_default_options(jpg_options *opt);
int jpg_run_benchmark(const jpg_options *opt, jpg_result *out);
void jpg_print_json(const jpg_options *opt, const jpg_result *res);
void jpg_print_csv(const jpg_options *opt, const jpg_result *res);

#ifdef __cplusplus
}
#endif

#endif
