#!/usr/bin/env bash
set -euo pipefail

bin/jeener_packet_c \
  --kmax 0.03 \
  --loops 0 \
  --Nvec 0.30 0 0 \
  --tip 0.05 \
  --pattern staggered \
  --ent-NK 13 \
  --ent-NT 17 \
  --ent-window 0.0:1.0 \
  --T 8.0 \
  --npts 2401 \
  --threads "${THREADS:-4}" \
  --format json
