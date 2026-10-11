# Jeener Packet Gaussian C

This repository contains a C implementation of the native packet-Gaussian
benchmark path for the Jeener cube code. It computes the corrected frozen
packet covariance, without the plotting and exploratory workflow layers.

The covariance equations do not reproduce the paper's separable `N=0` null
control. See [the audit](../AUDIT.md) before interpreting these results. The
corrected software benchmarks are listed in the repository-root README.

The portable backend uses POSIX threads to parallelize the k-grid. The macOS
build includes a low-level Metal availability layer and shader source. The
regression-tested numerical path is the portable CPU implementation.

## Build

macOS:

```bash
make
```

On macOS, the SDK is selected with `xcrun` to keep it consistent with the
active developer toolchain. `MACOS_SDK=/path/to/sdk` overrides that selection.

Build without any Metal framework dependency:

```bash
make clean
make METAL=0
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

The tests check analytic Gaussian states and every row of full-precision C
and Python results, including null, thermal, dissipative, aggregation, sampling,
and input-validation controls. The Python reference defaults to the sibling
codebase:

```text
../jeener-packet-gaussian-python/src/jeener_packet_gaussian/simulator.py
```

Set `JPG_PYTHON_REFERENCE=/path/to/jeener_packet_gaussian.py` to compare against
another compatible Python reference file. `make test-numerics` runs the C
analytic tests without Python.

## Output Validity

JSON includes complete-positivity and uncertainty margins, added covariance
noise, accepted/sample counts, and a flag for violation of the paper's null
control. Invalid samples are excluded; metrics are `null` if no samples remain.
Endpoint metrics refer to the last accepted sample. `--ent-aggregate max|mean`
applies to both pairs and triplets; every triplet uses its weakest bipartition.
Unknown options and unsupported scientific configurations are errors.

## License

This repository is released under a non-commercial license. Commercial use is
not permitted without prior written permission. The software is provided as-is,
without warranty, maintenance obligations, or technical support.
