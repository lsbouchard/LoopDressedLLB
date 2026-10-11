import itertools
import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from numpy.testing import assert_allclose
from scipy.linalg import expm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jeener_packet_gaussian import simulator as s


def squeezed_pair(r):
    c, z = np.cosh(2*r)/2, np.sinh(2*r)/2
    return np.array([[c,0,z,0], [0,c,0,-z], [z,0,c,0], [0,-z,0,c]])


class CovarianceTests(unittest.TestCase):
    def test_backaction_diagnostic_uses_applied_edge_kernel(self):
        original = copy.deepcopy(s.LOOP_COEFFS)
        try:
            s.LOOP_COEFFS['kappa'].update({2:5.0,3:10.0,4:20.0})
            y = s.init_MN(Nvec=(0.3,0,0))
            _,_,diag = s._native_packet_exact_step(y,0.1,0.03,0.3,4,0)
            expected = 1 + 5*0.09 + 10*0.09**2 + 20*0.09**3
            self.assertAlmostEqual(diag['packet_backaction_max'],expected)
            self.assertEqual(diag['packet_vertex_max'],1)
        finally:
            s.LOOP_COEFFS = original

    def test_roundoff_is_not_counted_as_entanglement(self):
        mask = np.ones((1,3),dtype=bool)
        values = np.array([[0.5-1e-15,0.5,0.5+1e-15]])
        self.assertEqual(s._row_entangled_fraction(values,mask,'lt',0.5)[0],0)

    def test_thermal_attenuation_exact_step(self):
        A, D = -0.3*np.eye(4), 0.48*np.eye(4)
        for t in (1e-7,2.0,8.0):
            phi,q = s._engineered_exact_step_operator(A,D,t)
            assert_allclose(phi,np.exp(-0.3*t)*np.eye(4),atol=1e-13)
            assert_allclose(q,0.8*(-np.expm1(-0.6*t))*np.eye(4),atol=1e-13)

    def test_vacuum_and_thermal(self):
        for nm in (1,2,3,8):
            for v in (0.5,0.8,2.0):
                V = v*np.eye(2*nm)
                assert_allclose(s._symplectic_eigenvalues(V),v,atol=1e-13)
                self.assertAlmostEqual(s._covariance_physicality_margin(V),v-0.5)
                self.assertEqual(s._gaussian_logneg_from_cov(V,[0])[1],0)

    def test_squeezed_pair_analytic(self):
        for r in (0.1,0.7,2.0):
            V = squeezed_pair(r)
            assert_allclose(s._symplectic_eigenvalues(V),0.5,atol=1e-12)
            nu, ln = s._gaussian_logneg_from_cov(V,[0])
            assert_allclose(nu,0.5*np.exp(-2*r),atol=1e-12)
            assert_allclose(ln,2*r/np.log(2),atol=1e-10)

    def test_random_symplectic_invariance(self):
        rng = np.random.default_rng(904)
        O = s._symplectic_form(4)
        for _ in range(5):
            G = rng.normal(size=(8,8))*0.1
            S = expm(O @ (G+G.T))
            spectrum = np.array([0.5,0.7,0.9,1.1])
            V = S @ np.diag(np.repeat(spectrum,2)) @ S.T
            assert_allclose(s._symplectic_eigenvalues(V),spectrum,atol=1e-12)

    def test_indefinite_covariance_rejected(self):
        self.assertLess(s._covariance_physicality_margin(-np.eye(2)),0)
        with self.assertRaises(np.linalg.LinAlgError):
            s._symplectic_eigenvalues(-np.eye(2))
        for V in (np.eye(3),np.full((2,2),np.nan),np.ones((2,3))):
            with self.assertRaises(ValueError):
                s._covariance_physicality_margin(V)

    def test_minimal_isotropic_repair(self):
        V = np.diag([0.01,4.0])
        margin = s._covariance_physicality_margin(V)
        repaired, after, noise = s._regularize_covariance_if_needed(V)
        self.assertAlmostEqual(noise,-margin+1e-9)
        self.assertGreaterEqual(after,0)
        self.assertLess(s._covariance_physicality_margin(repaired-2e-9*np.eye(2)),0)
        self.assertEqual(s._regularize_covariance_if_needed(np.eye(2)/2)[2],0)

    def test_all_bipartitions(self):
        for nm in range(2,9):
            parts = s._subset_bipartitions(tuple(range(nm)))
            self.assertEqual(len(parts),2**(nm-1)-1)
            keys = {frozenset((frozenset(a),frozenset(b))) for a,b in parts}
            self.assertEqual(len(parts),len(keys))

    def test_biseparable_triplet_is_not_fully_inseparable(self):
        V = np.eye(6)/2
        V[:4,:4] = squeezed_pair(0.7)
        for modes in itertools.permutations(range(3)):
            idx = [q for j in modes for q in (2*j,2*j+1)]
            metrics = s._multimode_metrics_from_cov(V[np.ix_(idx,idx)],(0,1,2))
            self.assertEqual(metrics['FI_SCORE'],0)
            self.assertEqual(metrics['LN_MIN'],0)

    def test_connected_triplet_permutation_invariance(self):
        O = s._symplectic_form(3)
        G = np.zeros((6,6))
        G[0,2] = G[2,0] = 0.4
        G[2,4] = G[4,2] = 0.6
        S = expm(O@G)
        V = S@S.T/2
        expected = s._multimode_metrics_from_cov(V,(0,1,2))['FI_SCORE']
        self.assertGreater(expected,0)
        for modes in itertools.permutations(range(3)):
            idx = [q for j in modes for q in (2*j,2*j+1)]
            self.assertAlmostEqual(s._multimode_metrics_from_cov(V[np.ix_(idx,idx)],(0,1,2))['FI_SCORE'],expected)

    def test_zero_coupling_is_separable(self):
        V = s.simulate_native_linearized_covariance_bins(0,(0.3,0,0),0,8,11,[0,5,10])
        for covariance in V:
            for a,b in itertools.combinations(range(8),2):
                assert_allclose(covariance[2*a:2*a+2,2*b:2*b+2],0,atol=1e-12)
                self.assertEqual(s._pair_metrics_from_cov(s._subcovariance(covariance,(a,b)))['LN'],0)

    def test_null_generator_against_analytic_classical_jacobian(self):
        y = s.init_MN(Nvec=(0,0,0))
        M = y[:24].reshape(8,3)
        inject, project = s._state_tangent_basis_packet(y)
        field = 0.03*np.einsum('jgik,gk->ji',s.T4,M)
        field[:,2] += s.B_INH
        jacobian = np.zeros((24,24))
        for j in range(8):
            for g in range(8):
                block = 0.03*s._cross_matrix(M[j])@s.T4[j,g]
                if j == g:
                    block -= s._cross_matrix(field[j])
                jacobian[3*j:3*j+3,3*g:3*g+3] = block
        analytic = project@jacobian@inject[:24]
        numerical = s._native_packet_drift_matrix(y,0.03,0,0)
        assert_allclose(analytic,numerical,atol=1e-10)

    def test_response_invariant_is_not_overridden(self):
        args = SimpleNamespace(ent_native_model='frozen',tip=0.05,pattern='staggered')
        status = s._native_entanglement_status((0,0,0),args)
        self.assertTrue(status['computed'])
        self.assertFalse(status['certified'])
        self.assertTrue(status['classical_invariant'])
        y = s.init_MN(Nvec=(0,0,0))
        assert_allclose(s._rhs_MN_drift(y,0.03,0,0)[24:],0,atol=0)
        A = s._native_packet_drift_matrix(y,0.03,0,0)
        O = s._symplectic_form(8)
        assert_allclose(A@O+O@A.T,0,atol=1e-10)
        self.assertGreater(np.linalg.norm(A+A.T),0.1)
        S = expm(A*8)
        V = s.simulate_native_linearized_covariance_bins(0.03,(0,0,0),0,8,31,[30])[0]
        assert_allclose(V,S@S.T/2,atol=1e-11)
        ln = max(s._pair_metrics_from_cov(s._subcovariance(V,p))['LN'] for p in s._neighbor_pairs())
        self.assertGreater(ln,0.7)
        small = s.simulate_native_linearized_covariance_bins(0.03,(1e-8,0,0),0,8,31,[30])[0]
        assert_allclose(small,V,atol=1e-8)

    def test_duplicate_indices_and_grid_independence(self):
        v = s.simulate_native_linearized_covariance_bins(0.03,(0.3,0,0),0,8,3,[0,0,1,1,2,2])
        assert_allclose(v[::2],v[1::2],atol=0)
        dense = s.simulate_native_linearized_covariance_bins(0.03,(0.3,0,0),0,8,2401,[0,1200,2400])
        assert_allclose(v[::2],dense,atol=1e-11)

    def test_dynamic_duplicate_indices_and_channel_diagnostics(self):
        V,d = s.simulate_native_trajectory_covariance_bins(0.03,(0.3,0,0),0,0.02,3,[0,0,1,2,2],return_diag=True)
        assert_allclose(V[0],V[1],atol=0)
        assert_allclose(V[3],V[4],atol=0)
        self.assertTrue(np.all(np.isfinite(d['step_cp_margin'])))
        self.assertGreater(np.min(d['step_cp_margin']),-1e-8)

    def test_tangent_lift_preserves_constraints_and_zero_time_identity(self):
        y = s.init_MN(Nvec=(0.1,0.2,0.1))
        J,P = s._state_tangent_basis_packet(y)
        M, N = y[:24].reshape(8,3), y[24:].reshape(8,3)
        for a in range(16):
            dm,dn = J[:24,a].reshape(8,3),J[24:,a].reshape(8,3)
            assert_allclose(np.sum(M*dm+N*dn,axis=1),0,atol=1e-13)
            assert_allclose(np.sum(N*dm+M*dn,axis=1),0,atol=1e-13)
        V = s.simulate_native_trajectory_covariance_bins(.03,(0.3,0,0),0,0.0,3,[0,1,2])
        assert_allclose(V,np.broadcast_to(np.eye(16)/2,V.shape),atol=1e-8)

    def test_invalid_sampling_and_initial_state(self):
        for idx in ([-1],[3],[2,1],[0.5],[]):
            with self.assertRaises(ValueError):
                s.simulate_native_linearized_covariance_bins(0.03,(0.3,0,0),0,8,3,idx)
        with self.assertRaises(ValueError):
            s.init_MN(Nvec=(np.nan,0,0))
        with self.assertRaises(ValueError):
            s.simulate_native_linearized_covariance_bins(0.03,(1,0,0),0,8,3,[0])


if __name__ == '__main__':
    unittest.main()
