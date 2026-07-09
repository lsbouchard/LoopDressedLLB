# LoopDressedLLB

Research code for loop-dressed mesoscopic spin-packet dynamics and
packet-mode Gaussian diagnostics used in the PRA benchmark workflow.

This repository contains two release codebases:

| Folder | Contents |
|---|---|
| `jeener-packet-gaussian-python/` | Full Python research release, including plotting, benchmark commands, tests, documentation, and the frozen simulator source. |
| `jeener-packet-gaussian-c/` | C implementation of the validated frozen native packet-Gaussian benchmark path, with POSIX/Pthreads parallelism and optional runtime Metal shader probing on macOS. |

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

The C regression test compares the C output against the frozen Python simulator.
From `jeener-packet-gaussian-c/`:

```bash
JPG_PYTHON_REFERENCE=../jeener-packet-gaussian-python/src/jeener_packet_gaussian/simulator.py \
make test
```

Representative benchmark endpoints for the frozen native packet-Gaussian model:

| Scenario | Pairwise endpoint | Multipartite endpoint |
|---|---:|---:|
| Native benchmark, `N=(0.30,0,0)`, loops `0` | `E_N ~= 0.6468` | `FI ~= 0.2394` |
| Sigma-enhanced coherent run | `E_N ~= 0.7349` | `FI ~= 0.2631` |
| Stronger coherent regime | `E_N ~= 0.8091` | `FI ~= 0.2787` |
| Thermal run, `--ent-native-nth 0.10` | `E_N ~= 0.3838` | `FI ~= 0.1873` |

## License And Support

This repository is released under a non-commercial license. Commercial use is
not permitted without prior written permission. The software is provided as-is,
without warranty, maintenance obligations, or technical support.
