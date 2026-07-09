#!/usr/bin/env bash
set -euo pipefail

CODE="${CODE:-jeener-packet-gaussian}"
OUTDIR="${OUTDIR:-pra_benchmark_outputs}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
mkdir -p "$OUTDIR"

tag_ent () {
  local s="$1"
  mv -f "$OUTDIR/ent_pairwise_gaussian_logneg_heatmap.png" "$OUTDIR/ent_pairwise_gaussian_logneg_heatmap_${s}.png"
  mv -f "$OUTDIR/ent_pairwise_gaussian_nupt_heatmap.png" "$OUTDIR/ent_pairwise_gaussian_nupt_heatmap_${s}.png"
  mv -f "$OUTDIR/ent_pairwise_gaussian_duan_heatmap.png" "$OUTDIR/ent_pairwise_gaussian_duan_heatmap_${s}.png"
  mv -f "$OUTDIR/ent_multipartite_lnmin_heatmap.png" "$OUTDIR/ent_multipartite_lnmin_heatmap_${s}.png"
  mv -f "$OUTDIR/ent_multipartite_full_insep_heatmap.png" "$OUTDIR/ent_multipartite_full_insep_heatmap_${s}.png"
}

run_case () {
  local tag="$1"
  shift
  "$CODE" "$@" --outdir "$OUTDIR" --no-show 1 2>&1 | tee "$OUTDIR/${tag}.log"
  tag_ent "$tag"
}

run_case classical_control_x000_T8 \
  --mode eigen \
  --kmax 0.03 \
  --loops 0 \
  --Nvec 0 0 0 \
  --ent 1 \
  --ent-save 1 \
  --ent-native-model frozen \
  --ent-native-source full \
  --ent-NK 4 \
  --ent-NT 9 \
  --T 8.0 \
  --npts 801 \
  --fig-dpi 400 \
  --fig-fontsize 11 \
  --fig-cmap magma \
  --fig-transparent 0

run_case fig3_native_corr_x030_T8 \
  --mode eigen \
  --kmax 0.03 \
  --loops 0 \
  --Nvec 0.30 0 0 \
  --tip 0.05 \
  --pattern staggered \
  --ent 1 \
  --ent-save 1 \
  --ent-native-model frozen \
  --ent-native-source full \
  --ent-window 0.0:1.0 \
  --ent-NK 13 \
  --ent-NT 17 \
  --T 8.0 \
  --npts 2401 \
  --fig-dpi 800 \
  --fig-fontsize 12 \
  --fig-cmap magma \
  --fig-transparent 0

run_case fig4_sigma_x030_T8 \
  --mode eigen \
  --kmax 0.03 \
  --loops 4 \
  --sigma 2.00 5.00 10.00 \
  --Nvec 0.30 0 0 \
  --tip 0.05 \
  --pattern staggered \
  --ent 1 \
  --ent-save 1 \
  --ent-native-model frozen \
  --ent-native-source full \
  --ent-window 0.0:1.0 \
  --ent-NK 13 \
  --ent-NT 17 \
  --T 8.0 \
  --npts 2401 \
  --fig-dpi 800 \
  --fig-fontsize 12 \
  --fig-cmap magma \
  --fig-transparent 0

run_case fig5_aggr_x030_T8 \
  --mode eigen \
  --kmax 0.03 \
  --loops 4 \
  --alpha 5.00 5.00 5.00 5.00 \
  --sigma 5.00 10.00 20.00 \
  --kappa 5.00 10.00 20.00 \
  --Nvec 0.30 0 0 \
  --tip 0.05 \
  --pattern staggered \
  --ent 1 \
  --ent-save 1 \
  --ent-native-model frozen \
  --ent-native-source full \
  --ent-window 0.0:1.0 \
  --ent-NK 13 \
  --ent-NT 17 \
  --T 8.0 \
  --npts 2401 \
  --fig-dpi 800 \
  --fig-fontsize 12 \
  --fig-cmap magma \
  --fig-transparent 0

run_case app_thermal_nth010_x030_T8 \
  --mode eigen \
  --kmax 0.03 \
  --loops 0 \
  --Nvec 0.30 0 0 \
  --tip 0.05 \
  --pattern staggered \
  --ent 1 \
  --ent-save 1 \
  --ent-native-model frozen \
  --ent-native-source full \
  --ent-native-nth 0.10 \
  --ent-window 0.0:1.0 \
  --ent-NK 13 \
  --ent-NT 17 \
  --T 8.0 \
  --npts 2401 \
  --fig-dpi 800 \
  --fig-fontsize 12 \
  --fig-cmap magma \
  --fig-transparent 0
