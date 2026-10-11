# LoopDressedLLB

Research code for loop-dressed mesoscopic spin-packet dynamics and
packet-mode Gaussian diagnostics used in the PRA benchmark workflow.

## Numerical Status

The October 2026 audit found an unresolved disagreement with the paper's
`N=0` null control. The covariance equations produce nonzero entanglement on
that branch. The earlier Python command line replaced those results with a
separable baseline; C did not. That override has been removed. This release
does not claim to reproduce the paper's null control.

The audit also corrected constrained tangent perturbations, incomplete
multipartite partition enumeration, covariance physicality checks, C mean
aggregation, and duplicate time sampling. These corrections change numerical
benchmarks. See [AUDIT.md](AUDIT.md) for evidence, definitions, and limitations.

This repository contains two release codebases:

| Folder | Contents |
|---|---|
| `jeener-packet-gaussian-python/` | Python research code, plotting, numerical tests, and documentation. |
| `jeener-packet-gaussian-c/` | Frozen packet-covariance solver, with POSIX/Pthreads parallelism and optional runtime Metal shader probing on macOS. |

The legacy notebook `spinpacket_041426.ipynb` is retained for provenance.

## Authors

Louis-S. Bouchard and Arianna Wu.

## Python Release

From the repository root:

```bash
cd jeener-packet-gaussian-python
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
jeener-packet-gaussian --help
```

Run the Python tests:

```bash
python -m unittest discover -s tests
```

The Python manual and benchmark commands are in:

```text
jeener-packet-gaussian-python/docs/
jeener-packet-gaussian-python/examples/
```

## C Release

Build the C benchmark engine:

```bash
cd jeener-packet-gaussian-c
make
```

Run a representative native benchmark:

```bash
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
  --threads 8 \
  --format json
```

On macOS, `--backend metal` optionally asks the program to runtime-compile the
included Metal shader source through the Metal framework. The validated
benchmark calculation remains the CPU/Pthreads path. If Metal is unavailable,
the program falls back to CPU/Pthreads.

```bash
bin/jeener_packet_c --backend metal --format json
```

## Regression Check

The tests include analytic Gaussian states and compare full-precision C and
Python metrics over every sampled coupling, including zero response, zero
coupling, thermal and dissipative cases, mean aggregation, and repeated times.
From `jeener-packet-gaussian-c/`:

```bash
JPG_PYTHON_REFERENCE=../jeener-packet-gaussian-python/src/jeener_packet_gaussian/simulator.py \
make test
```

Corrected computed endpoints at `k=0.03`, `t=8`, tip `0.05`, staggered pattern,
max aggregation over nearest-neighbor pairs and over all triplets. Each triplet
is tested across all three bipartitions. These are code regression values, not
validation of the paper:

| Scenario | Pairwise endpoint | Multipartite endpoint |
|---|---:|---:|
| Response invariant, `N=0`, loops `0` | `E_N ~= 0.7111304971` | `FI ~= 0.1907552563` |
| Native, `N=(0.30,0,0)`, loops `0` | `E_N ~= 0.6797119581` | `FI ~= 0.1873368018` |
| Sigma-enhanced coherent run | `E_N ~= 0.7696156077` | `FI ~= 0.2096115635` |
| Stronger coherent regime | `E_N ~= 0.8400339883` | `FI ~= 0.2332852550` |
| Thermal run, `--ent-native-nth 0.10` | `E_N ~= 0.4166775523` | `FI ~= 0.1248041621` |

The paper describes averaging over pairs. With `--ent-aggregate mean`, the native
pair endpoint is approximately `0.3645674060`. The same switch also averages
over triplets (`FI ~= 0.0226921933`); C does not offer separate aggregation
switches for pairs and triplets. The default remains `max` and is explicit in
the C JSON output.

## License And Support

This repository is released under a non-commercial license. Commercial use is
not permitted without prior written permission. The software is provided as-is,
without warranty, maintenance obligations, or technical support.
