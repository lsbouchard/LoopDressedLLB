#include "jpg.h"
#include "jpg_metal.h"
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int number(const char *s, double *v) {
    char *end;
    errno = 0;
    *v = strtod(s, &end);
    return s != end && !*end && !errno && isfinite(*v);
}

static void usage(void) {
    puts("usage: jeener_packet_c [options]\n\n"
         "Frozen packet covariance calculation. See AUDIT.md for the unresolved paper null control.\n"
         "  --Nvec Nx Ny Nz --kmax K --loops 0..4\n"
         "  --alpha a1 [a2 a3 a4] --beta b1 [b2 b3 b4]\n"
         "  --sigma s1 [s2 s3 s4] --kappa k2 [k3 k4]\n"
         "  --alpha-k2 X --beta-k2 X --beta-anis X --dynamic-mix 0|1\n"
         "  --tip X --pattern staggered|uniform\n"
         "  --ent-NK N --ent-NT N --ent-window a:b --T X --npts N\n"
         "  --ent-native-nth X --ent-native-nbath X --ent-native-reg-warn X\n"
         "  --ent-aggregate max|mean --threads N --format json|csv\n"
         "  --backend cpu|metal --metal-shader PATH\n\n"
         "Metal only probes shader availability; numerical work uses CPU pthreads.\n"
         "Only frozen/full, nearest-neighbor pairs and all triplet partitions are supported.\n"
         "Unknown options and unsupported scientific settings are errors.");
}

int main(int argc, char **argv) {
    jpg_options o;
    jpg_default_options(&o);
    const char *format = "json", *backend = "cpu", *shader = "shaders/jeener_kernels.metal", *bad = "";
    struct { const char *name; double *p; } real_opts[] = {
        {"--kmax", &o.kmax}, {"--T", &o.total_time}, {"--tip", &o.tip},
        {"--alpha-k2", &o.loop.alpha_k2}, {"--beta-k2", &o.loop.beta_k2}, {"--beta-anis", &o.loop.beta_anis},
        {"--ent-native-nth", &o.native_nth}, {"--ent-native-nbath", &o.native_nbath},
        {"--ent-native-reg-warn", &o.reg_warn}};
    struct { const char *name; int *p; } int_opts[] = {
        {"--loops", &o.loop.loops}, {"--npts", &o.npts}, {"--threads", &o.threads},
        {"--ent-NK", &o.ent_nk}, {"--ent-NT", &o.ent_nt}, {"--dynamic-mix", &o.loop.dynamic_mix}};
    for (int i = 1; i < argc; i++) {
        const char *key = argv[i];
        bad = key;
        if (!strcmp(key,"--help") || !strcmp(key,"-h")) { usage(); return 0; }
        double *coeff = NULL;
        int first = 1, count = 4;
        if (!strcmp(key,"--alpha")) coeff = o.loop.alpha;
        if (!strcmp(key,"--beta")) coeff = o.loop.beta;
        if (!strcmp(key,"--sigma")) coeff = o.loop.sigma;
        if (!strcmp(key,"--kappa")) { coeff = o.loop.kappa; first = 2; count = 3; }
        if (coeff) {
            memset(coeff,0,5*sizeof(double));
            int used = 0;
            while (i + 1 < argc && used < count) {
                double v;
                if (!number(argv[i+1],&v)) break;
                coeff[first+used++] = v;
                i++;
            }
            if (!used) goto invalid;
            continue;
        }
        if (!strcmp(key,"--Nvec")) {
            if (i+3 >= argc) goto invalid;
            for (int j = 0; j < 3; j++) if (!number(argv[++i],&o.nvec[j])) goto invalid;
            continue;
        }
        if (i+1 >= argc) goto invalid;
        const char *v = argv[++i];
        int found = 0;
        for (size_t j = 0; j < sizeof(real_opts)/sizeof(real_opts[0]); j++)
            if (!strcmp(key,real_opts[j].name)) {
                if (!number(v,real_opts[j].p)) goto invalid;
                found = 1;
            }
        for (size_t j = 0; j < sizeof(int_opts)/sizeof(int_opts[0]); j++)
            if (!strcmp(key,int_opts[j].name)) {
                char *end;
                errno = 0;
                long n = strtol(v,&end,10);
                if (end == v || *end || errno || n < INT_MIN || n > INT_MAX) goto invalid;
                *int_opts[j].p = (int)n;
                found = 1;
            }
        if (found) continue;
        if (!strcmp(key,"--pattern")) {
            if (strcmp(v,"uniform") && strcmp(v,"staggered")) goto invalid;
            o.pattern = !strcmp(v,"uniform");
        } else if (!strcmp(key,"--ent-aggregate")) {
            if (strcmp(v,"mean") && strcmp(v,"max")) goto invalid;
            o.aggregate_mean = !strcmp(v,"mean");
        } else if (!strcmp(key,"--ent-window")) {
            int used = 0;
            if (sscanf(v,"%lf:%lf%n",&o.ent_window0,&o.ent_window1,&used) != 2 || v[used]) goto invalid;
        } else if (!strcmp(key,"--format")) {
            if (strcmp(v,"json") && strcmp(v,"csv")) goto invalid;
            format = v;
        } else if (!strcmp(key,"--backend")) {
            if (strcmp(v,"cpu") && strcmp(v,"metal")) goto invalid;
            backend = v;
        } else if (!strcmp(key,"--metal-shader")) shader = v;
        else if (!strcmp(key,"--ent-native-model")) { if (strcmp(v,"frozen")) goto invalid; }
        else if (!strcmp(key,"--ent-native-source")) { if (strcmp(v,"full")) goto invalid; }
        else goto invalid;
    }
    if (!strcmp(backend,"metal")) {
        char msg[1024];
        if (!jpg_metal_available()) fputs("warning: Metal unavailable; using CPU pthreads\n",stderr);
        else {
            int ok = jpg_metal_compile_shader_file(shader,msg,sizeof(msg));
            fprintf(stderr,"%s: %s; numerical calculation uses CPU pthreads\n",ok ? "info" : "warning",msg);
        }
    }
    jpg_result result;
    int rc = jpg_run_benchmark(&o,&result);
    if (rc) { fprintf(stderr,"benchmark failed (code %d): invalid domain or numerical failure\n",rc); return rc; }
    int null_warning = 0, invalid_warning = 0;
    for (int i = 0; i < result.nk; i++) {
        null_warning |= result.rows[i].null_control_violation;
        invalid_warning |= result.rows[i].valid_count < result.nt;
    }
    if (null_warning) fputs("warning: computed N=0 covariance violates the paper's separable null control; see AUDIT.md\n",stderr);
    if (invalid_warning) fputs("warning: covariance repair or generator complete-positivity checks failed; inspect validity diagnostics\n",stderr);
    if (!strcmp(format,"csv")) jpg_print_csv(&o,&result);
    else jpg_print_json(&o,&result);
    return 0;
invalid:
    fprintf(stderr,"invalid or unsupported option/value: %s\n",bad);
    return 2;
}
