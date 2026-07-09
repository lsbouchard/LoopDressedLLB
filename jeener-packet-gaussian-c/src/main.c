#include "jpg.h"
#include "jpg_metal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int is_opt(const char *s, const char *name) {
    return s && strcmp(s, name) == 0;
}

static int parse_doubles(int argc, char **argv, int *i, double *dst, int maxn) {
    int n = 0;
    while (*i + 1 < argc && n < maxn) {
        const char *s = argv[*i + 1];
        if (s[0] == '-' && !(s[1] >= '0' && s[1] <= '9') && s[1] != '.') break;
        dst[n++] = atof(s);
        (*i)++;
    }
    return n;
}

static void usage(FILE *f) {
    fprintf(f,
        "usage: jeener_packet_c [options]\n"
        "\n"
        "Native packet-Gaussian benchmark engine in C.\n"
        "\n"
        "Main options:\n"
        "  --Nvec Nx Ny Nz              response vector\n"
        "  --kmax K                     maximum k value\n"
        "  --loops N                    loop order 0..4\n"
        "  --alpha a1 [a2 a3 a4]        alpha coefficients\n"
        "  --beta b1 [b2 b3 b4]         beta coefficients\n"
        "  --sigma s1 [s2 s3 s4]        sigma coefficients\n"
        "  --kappa k2 [k3 k4]           kappa coefficients\n"
        "  --tip X                      tip angle, radians\n"
        "  --pattern staggered|uniform  N-pattern\n"
        "  --ent-NK N                   number of k samples\n"
        "  --ent-NT N                   number of time bins\n"
        "  --ent-window a:b             time-window fraction\n"
        "  --T X                        total time\n"
        "  --npts N                     time steps\n"
        "  --ent-native-nth X           initial thermal occupancy\n"
        "  --threads N                  pthread worker count\n"
        "  --format json|csv            output format\n"
        "  --backend cpu|metal          requested backend\n"
        "  --metal-shader PATH          optional Metal shader source path\n"
        "  -h, --help                   show this help\n"
        "\n"
        "Unrecognized plotting options from the Python workflow are accepted and ignored.\n");
}

int main(int argc, char **argv) {
    jpg_options opt;
    jpg_default_options(&opt);
    const char *format = "json";
    const char *backend = "cpu";
    const char *metal_shader = "shaders/jeener_kernels.metal";

    for (int i = 1; i < argc; i++) {
        if (is_opt(argv[i], "-h") || is_opt(argv[i], "--help")) {
            usage(stdout);
            return 0;
        } else if (is_opt(argv[i], "--Nvec")) {
            double v[3] = {0,0,0};
            int n = parse_doubles(argc, argv, &i, v, 3);
            for (int j = 0; j < n && j < 3; j++) opt.nvec[j] = v[j];
        } else if (is_opt(argv[i], "--kmax")) {
            if (++i < argc) opt.kmax = atof(argv[i]);
        } else if (is_opt(argv[i], "--loops")) {
            if (++i < argc) opt.loop.loops = atoi(argv[i]);
        } else if (is_opt(argv[i], "--alpha")) {
            double v[4] = {0,0,0,0};
            int n = parse_doubles(argc, argv, &i, v, 4);
            for (int j = 0; j < n; j++) opt.loop.alpha[j + 1] = v[j];
        } else if (is_opt(argv[i], "--beta")) {
            double v[4] = {0,0,0,0};
            int n = parse_doubles(argc, argv, &i, v, 4);
            for (int j = 0; j < n; j++) opt.loop.beta[j + 1] = v[j];
        } else if (is_opt(argv[i], "--sigma")) {
            double v[4] = {0,0,0,0};
            int n = parse_doubles(argc, argv, &i, v, 4);
            for (int j = 0; j < n; j++) opt.loop.sigma[j + 1] = v[j];
        } else if (is_opt(argv[i], "--kappa")) {
            double v[3] = {0,0,0};
            int n = parse_doubles(argc, argv, &i, v, 3);
            for (int j = 0; j < n; j++) opt.loop.kappa[j + 2] = v[j];
        } else if (is_opt(argv[i], "--alpha-k2")) {
            if (++i < argc) opt.loop.alpha_k2 = atof(argv[i]);
        } else if (is_opt(argv[i], "--beta-k2")) {
            if (++i < argc) opt.loop.beta_k2 = atof(argv[i]);
        } else if (is_opt(argv[i], "--beta-anis")) {
            if (++i < argc) opt.loop.beta_anis = atof(argv[i]);
        } else if (is_opt(argv[i], "--dynamic-mix")) {
            if (++i < argc) opt.loop.dynamic_mix = atoi(argv[i]);
        } else if (is_opt(argv[i], "--tip")) {
            if (++i < argc) opt.tip = atof(argv[i]);
        } else if (is_opt(argv[i], "--pattern")) {
            if (++i < argc) opt.pattern = strcmp(argv[i], "uniform") == 0 ? 1 : 0;
        } else if (is_opt(argv[i], "--ent-NK")) {
            if (++i < argc) opt.ent_nk = atoi(argv[i]);
        } else if (is_opt(argv[i], "--ent-NT")) {
            if (++i < argc) opt.ent_nt = atoi(argv[i]);
        } else if (is_opt(argv[i], "--ent-window")) {
            if (++i < argc) {
                double a = 0.8, b = 1.0;
                sscanf(argv[i], "%lf:%lf", &a, &b);
                opt.ent_window0 = a;
                opt.ent_window1 = b;
            }
        } else if (is_opt(argv[i], "--T")) {
            if (++i < argc) opt.total_time = atof(argv[i]);
        } else if (is_opt(argv[i], "--npts")) {
            if (++i < argc) opt.npts = atoi(argv[i]);
        } else if (is_opt(argv[i], "--ent-native-nth")) {
            if (++i < argc) opt.native_nth = atof(argv[i]);
        } else if (is_opt(argv[i], "--ent-native-nbath")) {
            if (++i < argc) opt.native_nbath = atof(argv[i]);
        } else if (is_opt(argv[i], "--ent-native-reg-warn")) {
            if (++i < argc) opt.reg_warn = atof(argv[i]);
        } else if (is_opt(argv[i], "--ent-aggregate")) {
            if (++i < argc) opt.aggregate_mean = strcmp(argv[i], "mean") == 0;
        } else if (is_opt(argv[i], "--threads")) {
            if (++i < argc) opt.threads = atoi(argv[i]);
        } else if (is_opt(argv[i], "--format")) {
            if (++i < argc) format = argv[i];
        } else if (is_opt(argv[i], "--backend")) {
            if (++i < argc) backend = argv[i];
        } else if (is_opt(argv[i], "--metal-shader")) {
            if (++i < argc) metal_shader = argv[i];
        } else if (is_opt(argv[i], "--mode") || is_opt(argv[i], "--ent") ||
                   is_opt(argv[i], "--ent-save") || is_opt(argv[i], "--ent-native-model") ||
                   is_opt(argv[i], "--ent-native-source") || is_opt(argv[i], "--fig-dpi") ||
                   is_opt(argv[i], "--fig-fontsize") || is_opt(argv[i], "--fig-cmap") ||
                   is_opt(argv[i], "--fig-transparent") || is_opt(argv[i], "--no-show") ||
                   is_opt(argv[i], "--outdir") || is_opt(argv[i], "--export-data")) {
            if (i + 1 < argc && argv[i + 1][0] != '-') i++;
        } else {
            fprintf(stderr, "warning: ignoring option %s\n", argv[i]);
        }
    }

    if (strcmp(backend, "metal") == 0) {
        char msg[1024];
        if (!jpg_metal_available()) {
            fprintf(stderr, "warning: Metal backend not available; using CPU pthread backend\n");
        } else if (jpg_metal_compile_shader_file(metal_shader, msg, sizeof(msg))) {
            fprintf(stderr, "info: %s; using CPU pthread backend for validated benchmark path\n", msg);
        } else {
            fprintf(stderr, "warning: %s; using CPU pthread backend\n", msg);
        }
    }

    jpg_result res;
    int rc = jpg_run_benchmark(&opt, &res);
    if (rc != 0) {
        fprintf(stderr, "benchmark failed with code %d\n", rc);
        return rc;
    }

    if (strcmp(format, "csv") == 0) jpg_print_csv(&opt, &res);
    else jpg_print_json(&opt, &res);
    return 0;
}
