import hashlib
import os
import subprocess
import sys
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
RELEASE_SHA256 = "4892b31a6bb43573c0c5051fbe5f587aa3caef14966adf51a26fc6be889c5722"


class SourceIntegrityTests(unittest.TestCase):
    def test_package_identity(self):
        self.assertEqual(PROGRAM_NAME, "jeener_packet_gaussian")
        self.assertEqual(PROGRAM_VERSION, "2026.7.9")

    def test_release_source_hash(self):
        digest = hashlib.sha256(SIMULATOR.read_bytes()).hexdigest()
        self.assertEqual(digest, RELEASE_SHA256)

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


if __name__ == "__main__":
    unittest.main()
