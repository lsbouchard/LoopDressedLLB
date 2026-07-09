# Jeener Packet Gaussian

`jeener-packet-gaussian` is a research code release for the Jeener-cube packet
simulator used in the PRA benchmark workflow.

The executable source is preserved at:

```text
src/jeener_packet_gaussian/simulator.py
```

Release source SHA256:

```text
4892b31a6bb43573c0c5051fbe5f587aa3caef14966adf51a26fc6be889c5722
```

## What It Does

The code simulates an eight-spin Jeener cube using mesoscopic packet variables
`M` and `N`. It includes:

- deterministic packet dynamics;
- loop-dressed corrections through `alpha`, `beta`, `sigma`, and `kappa`;
- optional stochastic torque and short-memory damping;
- certified native packet-mode Gaussian entanglement diagnostics;
- an optional engineered quadratic-Gaussian benchmark layer.

The implementation prepares user-specified initial conditions directly on the
rotated coherent-state constraint surface,

```text
|M_j|^2 + |N_j|^2 = R^2,     M_j . N_j = 0.
```

For example, `--Nvec 0.30 0 0` initializes each packet with `|N| = 0.30`.

## Install

From the repository root:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Alternatively, install only the runtime requirements and run the source file:

```bash
python -m pip install -r requirements.txt
python src/jeener_packet_gaussian/simulator.py --help
```

After installation, the command-line entry point is:

```bash
jeener-packet-gaussian --help
```

You can also run it as a module:

```bash
python -m jeener_packet_gaussian --help
```

For batch or headless runs, use Matplotlib's non-GUI backend:

```bash
export MPLBACKEND=Agg
```

## Minimal Execution Test

```bash
MPLBACKEND=Agg \
jeener-packet-gaussian \
  --mode eigen \
  --kmax 0.03 \
  --loops 0 \
  --Nvec 0 0 0 \
  --ent 1 \
  --ent-native-model frozen \
  --ent-NK 2 \
  --ent-NT 3 \
  --T 1.0 \
  --npts 101 \
  --no-show 1 \
  --outdir smoke_outputs
```

## PRA Benchmark Commands

The full benchmark command suite is in:

```text
examples/pra_benchmark_commands.sh
```

The complete user manual is in:

```text
docs/jeener_packet_gaussian_manual.pdf
docs/jeener_packet_gaussian_manual.tex
```

Representative endpoint values for the `frozen` native packet-Gaussian model:

| Scenario | Pairwise endpoint | Multipartite endpoint |
|---|---:|---:|
| Invariant branch, `N=0`, loops `0` | `E_N = 0` | `FI = 0` |
| Native benchmark, `N=(0.30,0,0)`, loops `0` | `E_N ~= 0.6468` | `FI ~= 0.2394` |
| Sigma-enhanced coherent run | `E_N ~= 0.7349` | `FI ~= 0.2631` |
| Stronger coherent regime | `E_N ~= 0.8091` | `FI ~= 0.2787` |
| Thermal run, `--ent-native-nth 0.10` | `E_N ~= 0.3838` | `FI ~= 0.1873` |

## Tests

```bash
python -m unittest discover -s tests
```

The test suite checks importability, the release source identity, the exact
constrained initialization, and the CLI help path. It also runs under `pytest`
when `pytest` is available.

## License

This repository is released under a non-commercial license. Commercial use is
not permitted without prior written permission. The software is provided as-is,
without warranty, maintenance obligations, or technical support.
