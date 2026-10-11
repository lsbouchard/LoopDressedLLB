import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jeener_packet_gaussian import PROGRAM_NAME, PROGRAM_VERSION
from jeener_packet_gaussian import simulator


SIMULATOR = SRC / "jeener_packet_gaussian" / "simulator.py"


class SourceIntegrityTests(unittest.TestCase):
    def test_package_identity(self):
        self.assertEqual(PROGRAM_NAME, "jeener_packet_gaussian")
        self.assertEqual(PROGRAM_VERSION, "2026.10.10")

    def test_exact_constrained_initialization_for_pra_benchmark(self):
        y = simulator.init_MN(Nvec=(0.30, 0.0, 0.0), tip=0.05, pattern="staggered")
        n = simulator.N_SPINS
        m = y[: 3 * n].reshape(n, 3)
        response = y[3 * n :].reshape(n, 3)

        self.assertTrue(np.allclose(np.linalg.norm(response, axis=1), 0.30, atol=1e-12))
        self.assertTrue(np.allclose(np.sum(m * response, axis=1), 0.0, atol=1e-12))
        self.assertTrue(
            np.allclose(
                np.linalg.norm(m, axis=1) ** 2 + np.linalg.norm(response, axis=1) ** 2,
                1.0,
                atol=1e-12,
            )
        )

    def test_invariant_branch_initializes_with_zero_response(self):
        y = simulator.init_MN(Nvec=(0.0, 0.0, 0.0), tip=0.05, pattern="staggered")
        n = simulator.N_SPINS
        response = y[3 * n :].reshape(n, 3)

        self.assertTrue(np.allclose(response, 0.0, atol=1e-15))

    def test_module_cli_help_runs(self):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(SRC)
        env.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig-test"))

        result = subprocess.run(
            [sys.executable, "-m", "jeener_packet_gaussian", "--help"],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )

        self.assertIn("Jeener 8-spin spectral clustering", result.stdout)
        self.assertIn("--ent-native-model", result.stdout)

    def test_native_cli_uses_computed_covariance(self):
        env = os.environ.copy()
        env['PYTHONPATH'] = str(SRC)
        env['MPLBACKEND'] = 'Agg'
        with tempfile.TemporaryDirectory() as output:
            base = [sys.executable,'-m','jeener_packet_gaussian','--mode','eigen',
                    '--kmax','0.03','--Nvec','0','0','0','--ent','1','--ent-save','0',
                    '--ent-native-model','frozen','--ent-NK','2','--ent-NT','3',
                    '--ent-window','0:1','--T','8','--npts','31','--no-show','1',
                    '--outdir',output]
            for extra,expected in [([], 'E_N=0.7111'),
                                   (['--ent-layer','multi'], 'FI_score=0.1908'),
                                   (['--ent-native-model','none'], 'no entanglement result is reported')]:
                result = subprocess.run(base+extra,env=env,text=True,capture_output=True,check=True,timeout=30)
                self.assertIn(expected,result.stdout)


if __name__ == "__main__":
    unittest.main()
