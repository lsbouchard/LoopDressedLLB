# Jeener Packet Gaussian C

This repository contains a C implementation of the native packet-Gaussian
benchmark path for the Jeener cube code. It is intended for reproducing the
validated frozen native packet-Gaussian PRA benchmark cases from the Python
release, without the plotting and exploratory workflow layers.

The portable backend uses POSIX threads to parallelize the k-grid. The macOS
build includes a low-level Metal availability layer and shader source. The
regression-tested numerical path is the portable CPU implementation.

## Build

macOS:

```bash
make
```

Linux:

```bash
make LDLIBS="-lm -lpthread -llapack -lblas"
```

The build requires BLAS/LAPACK for the small eigenvalue problems used in the
Gaussian symplectic spectra.

## Optional Metal Shader Probe

Metal shaders are optional. The validated benchmark path is CPU/Pthreads. On
macOS, `--backend metal` asks the program to runtime-compile the shader source
through the Metal framework and then continue with the CPU/Pthreads numerical
path:

```bash
bin/jeener_packet_c --backend metal --format json
```

No offline `.metallib` build is required, and the command-line `metal` compiler
from full Xcode is not required. If Metal or the shader source is unavailable,
the program prints a warning and uses the CPU/Pthreads backend.

## Run

```bash
bin/jeener_packet_c \
  --kmax 0.03 \
  --loops 0 \
  --Nvec 0.30 0 0 \
  --ent-NK 13 \
  --ent-NT 17 \
  --T 8.0 \
  --npts 2401 \
  --threads 8 \
  --format json
```

## Regression Test

```bash
make test
```

The regression script runs the frozen Python benchmark path and compares the C
output for the native, sigma-enhanced, stronger coherent, and thermal benchmark
cases. By default it uses:

```text
/Users/bouchard/Python/Jeener/jeener_packet_gaussian.py
```

Set `JPG_PYTHON_REFERENCE=/path/to/jeener_packet_gaussian.py` to compare against
another frozen Python reference file.

## License

This repository is released under a non-commercial license. Commercial use is
not permitted without prior written permission. The software is provided as-is,
without warranty, maintenance obligations, or technical support.
