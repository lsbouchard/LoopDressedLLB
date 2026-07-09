#!/usr/bin/env python3
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = Path(os.environ.get("JPG_PYTHON_REFERENCE", "/Users/bouchard/Python/Jeener/jeener_packet_gaussian.py"))
BIN = ROOT / "bin" / "jeener_packet_c"


CASES = [
    ("native", ["--loops", "0"], {}),
    ("sigma", ["--loops", "4", "--sigma", "2.00", "5.00", "10.00"], {}),
    (
        "strong",
        [
            "--loops",
            "4",
            "--alpha",
            "5.00",
            "5.00",
            "5.00",
            "5.00",
            "--sigma",
            "5.00",
            "10.00",
            "20.00",
            "--kappa",
            "5.00",
            "10.00",
            "20.00",
        ],
        {},
    ),
    ("thermal", ["--loops", "0", "--ent-native-nth", "0.10"], {}),
]


def run_c(extra):
    cmd = [
        str(BIN),
        "--kmax",
        "0.03",
        "--Nvec",
        "0.30",
        "0",
        "0",
        "--tip",
        "0.05",
        "--pattern",
        "staggered",
        "--ent-NK",
        "13",
        "--ent-NT",
        "17",
        "--ent-window",
        "0.0:1.0",
        "--T",
        "8.0",
        "--npts",
        "2401",
        "--threads",
        "4",
        "--format",
        "json",
    ] + extra
    out = subprocess.check_output(cmd, text=True, cwd=ROOT)
    return json.loads(out)


def run_python(extra):
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    outdir = ROOT / "tests" / "out"
    outdir.mkdir(parents=True, exist_ok=True)
    mpldir = outdir / "matplotlib"
    mpldir.mkdir(parents=True, exist_ok=True)
    env["MPLCONFIGDIR"] = str(mpldir)
    cmd = [
        sys.executable,
        str(PY),
        "--mode",
        "eigen",
        "--kmax",
        "0.03",
        "--Nvec",
        "0.30",
        "0",
        "0",
        "--tip",
        "0.05",
        "--pattern",
        "staggered",
        "--ent",
        "1",
        "--ent-save",
        "0",
        "--ent-native-model",
        "frozen",
        "--ent-native-source",
        "full",
        "--ent-window",
        "0.0:1.0",
        "--ent-NK",
        "13",
        "--ent-NT",
        "17",
        "--T",
        "8.0",
        "--npts",
        "2401",
        "--no-show",
        "1",
        "--outdir",
        str(outdir),
    ] + extra
    text = subprocess.check_output(cmd, text=True, env=env, cwd=ROOT)
    pair = None
    multi = None
    section = None
    for line in text.splitlines():
        if "=== Pairwise Gaussian summary" in line:
            section = "pair"
            continue
        if "=== Multipartite Gaussian summary" in line:
            section = "multi"
            continue
        if "=== Whole-window" in line or "best over certified" in line:
            section = None
            continue
        if section == "pair" and "k=0.0300" in line and "n_valid=" in line:
            m = re.search(r"\sE_N=([0-9.]+)", line)
            if m:
                pair = float(m.group(1))
        if section == "multi" and "k=0.0300" in line and "FI_score=" in line and "n_valid=" in line:
            m = re.search(r"FI_score=([0-9.]+)", line)
            if m:
                multi = float(m.group(1))
    if pair is None or multi is None:
        raise RuntimeError("could not parse Python output")
    return {"pair_endpoint": pair, "multi_endpoint": multi}


def main():
    if not PY.exists():
        raise SystemExit(
            "Python reference simulator not found. Set JPG_PYTHON_REFERENCE to the "
            "frozen jeener_packet_gaussian.py path before running this regression."
        )
    failures = []
    for name, extra, _meta in CASES:
        c = run_c(extra)["summary"]
        p = run_python(extra)
        dp = abs(c["pair_endpoint"] - p["pair_endpoint"])
        dm = abs(c["multi_endpoint"] - p["multi_endpoint"])
        print(
            f"{name:8s} pair C={c['pair_endpoint']:.6f} PY={p['pair_endpoint']:.6f} "
            f"diff={dp:.3e} | multi C={c['multi_endpoint']:.6f} PY={p['multi_endpoint']:.6f} diff={dm:.3e}"
        )
        if dp > 5e-3 or dm > 5e-3:
            failures.append((name, dp, dm))
    if failures:
        raise SystemExit(f"regression failures: {failures}")


if __name__ == "__main__":
    main()
