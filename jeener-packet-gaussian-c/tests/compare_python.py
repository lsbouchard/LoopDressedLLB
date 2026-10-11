#!/usr/bin/env python3
"""Full-precision C/Python comparisons and command-line regression tests."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "loopdressed-test-mpl"))
import numpy as np
from numpy.testing import assert_allclose

ROOT = Path(__file__).resolve().parents[1]
PY = Path(os.environ.get("JPG_PYTHON_REFERENCE", ROOT.parent /
    "jeener-packet-gaussian-python/src/jeener_packet_gaussian/simulator.py"))
BIN = Path(os.environ.get("JPG_C_BINARY", ROOT / "bin/jeener_packet_c"))
spec = importlib.util.spec_from_file_location("reference_simulator", PY)
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
COEFFS, CONFIG = copy.deepcopy(s.LOOP_COEFFS), copy.deepcopy(s.CFG)

DEFAULT = dict(nvec=(0.3,0,0), loops=0, kmax=0.03, nk=3, nt=7, npts=2401,
               total=8.0, window=(0,1), tip=0.05, pattern='staggered', nth=0.0,
               nbath=0.0, aggregate='max', beta_anis=0.0, threads=2,
               dynamic_mix=1, alpha_k2=0.0, beta_k2=0.0)
CASES = {
    'native': {},
    'null': {'nvec':(0,0,0)},
    'near_null': {'nvec':(1e-8,0,0)},
    'uncoupled': {'kmax':0.0},
    'sigma': {'loops':4,'sigma':[2,5,10]},
    'strong': {'loops':4,'alpha':[5,5,5,5],'sigma':[5,10,20],'kappa':[5,10,20]},
    'thermal': {'nth':0.1},
    'damping': {'loops':4,'beta':[0.5,0.2,0.1,0.05], 'nbath':0.2},
    'anisotropic': {'loops':4,'beta':[0.5], 'beta_anis':0.2, 'nbath':0.4},
    'mean': {'aggregate':'mean'},
    'tilted_uniform': {'nvec':(0.1,0.2,0.1), 'tip':0.4, 'pattern':'uniform'},
    'duplicates': {'npts':3,'nt':9,'total':0.8,'window':(0.4,0.6)},
    'endpoint_only': {'nt':1,'window':(1,1)},
    'zero_duration': {'total':0.0},
    'thermal_null': {'nvec':(0,0,0),'nth':0.1},
    'static_mix': {'dynamic_mix':0,'kmax':10.0,'total':0.01},
    'k2_dressing': {'loops':4,'alpha':[2,1], 'beta':[0.2], 'alpha_k2':2.0,'beta_k2':3.0},
    'non_cp_gain': {'loops':1,'beta':[-1.0]},
}


def run_c(cfg, extra=()):
    c = DEFAULT | cfg
    args = [str(BIN), '--Nvec', *map(str,c['nvec'])]
    mapping = {'loops':'--loops','kmax':'--kmax','nk':'--ent-NK','nt':'--ent-NT',
        'npts':'--npts','total':'--T','tip':'--tip','pattern':'--pattern',
        'nth':'--ent-native-nth','nbath':'--ent-native-nbath','aggregate':'--ent-aggregate',
        'threads':'--threads','beta_anis':'--beta-anis','dynamic_mix':'--dynamic-mix',
        'alpha_k2':'--alpha-k2','beta_k2':'--beta-k2'}
    for key, flag in mapping.items():
        args += [flag,str(c[key])]
    args += ['--ent-window', ':'.join(map(str,c['window']))]
    for key in ('alpha','beta','sigma','kappa'):
        if key in c:
            args += ['--'+key,*map(str,c[key])]
    result = subprocess.run(args + list(extra),capture_output=True,text=True,check=True,timeout=60)
    return json.loads(result.stdout, parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))


def run_python(cfg):
    c = DEFAULT | cfg
    s.LOOP_COEFFS = copy.deepcopy(COEFFS)
    s.CFG = copy.deepcopy(CONFIG)
    for key in ('beta_anis','dynamic_mix','alpha_k2','beta_k2'):
        s.CFG[key] = c[key]
    for channel in ('alpha','beta','sigma','kappa'):
        for i,value in enumerate(c.get(channel,[]),2 if channel == 'kappa' else 1):
            s.LOOP_COEFFS[channel][i] = value
    return s.native_linearized_collect_metrics_over_kt(
        Nvec=c['nvec'],k_values=np.linspace(0,c['kmax'],c['nk']),T_TOT=c['total'],
        N_PTS=c['npts'],tip=c['tip'],pattern=c['pattern'],loop_order=c['loops'],
        rng=np.random.default_rng(123),time_window_frac=c['window'],ent_NT=c['nt'],
        pairs=s._neighbor_pairs(),ent_S=100,multi_subsets=s._subset_combinations(3),
        native_nth=c['nth'],native_nbath=c['nbath'],aggregate_mode=c['aggregate'],
        dynamic_background=False)


class RegressionTests(unittest.TestCase):
    def test_all_rows_and_metrics(self):
        for name,cfg in CASES.items():
            with self.subTest(case=name):
                c, p = run_c(cfg), run_python(cfg)
                for i,row in enumerate(c['rows']):
                    mask = np.isfinite(p['pair']['LN'][i]) & (p['diag']['added_noise'][i] <= 1e-6)
                    self.assertEqual(row['valid_count'],int(mask.sum()))
                    self.assertEqual(row['sample_count'],(DEFAULT | cfg)['nt'])
                    assert_allclose(row['generator_cp_margin'],p['diag']['generator_cp_margin'][i,0],atol=2e-9)
                    assert_allclose(row['added_noise_max'],p['diag']['added_noise'][i].max(),atol=2e-8)
                    values = {'pair':p['pair']['LN'][i], 'multi':p['multi']['FI_SCORE'][i]}
                    for key, series in values.items():
                        for suffix, fn in [('end',lambda x:x[-1]),('best',np.max),('mean',np.mean)]:
                            actual = row[key+'_'+suffix]
                            if mask.any():
                                assert_allclose(actual,fn(series[mask]),atol=2e-8,rtol=2e-8,
                                                err_msg=f'{name} row {i} {key}_{suffix}')
                            else:
                                self.assertIsNone(actual)
                    for key,series in [('pair_nu_min',p['pair']['NU_PT'][i]),
                                       ('pair_duan_min',p['pair']['DUAN'][i]),
                                       ('multi_nu_min',p['multi']['NU_WORST'][i])]:
                        if mask.any(): assert_allclose(row[key],np.min(series[mask]),atol=2e-8,rtol=2e-8)
                        else: self.assertIsNone(row[key])
                print(f'{name:16s} pair={c["summary"]["pair_endpoint"]} FI={c["summary"]["multi_endpoint"]}',flush=True)

    def test_thread_determinism(self):
        a = run_c({'threads':1})
        b = run_c({'threads':4})
        self.assertEqual(a['rows'],b['rows'])

    def test_frozen_grid_independence(self):
        a = run_c({'npts':13})
        b = run_c({'npts':1000003})
        for ar,br in zip(a['rows'],b['rows']):
            assert_allclose(ar['pair_end'],br['pair_end'],atol=1e-10)
            assert_allclose(ar['multi_end'],br['multi_end'],atol=1e-10)

    def test_null_reporting_and_zero_coupling(self):
        c = run_c({'nvec':(0,0,0)})
        self.assertFalse(c['rows'][0]['null_control_violation'])
        self.assertTrue(c['rows'][-1]['null_control_violation'])
        self.assertGreater(c['rows'][-1]['pair_end'],0.7)
        self.assertEqual(c['rows'][0]['pair_end'],0)
        self.assertEqual(c['rows'][0]['multi_end'],0)

    def test_invalid_cli(self):
        cases = [['--npts','1'],['--ent-window','-1:1'],['--ent-window','0.8:0.2'],
            ['--ent-window','0:2'],['--T','nan'],['--T','-1'],['--T','1junk'],
            ['--T'],['--loops','5'],['--threads','-1'],['--threads','1.5'],
            ['--pattern','oops'],['--format','oops'],['--backend','cuda'],
            ['--Nvec','0','0'],['--Nvec','1.1','0','0'],['--Nvec','1','0','0'],
            ['--sigma'],['--sigma','nan'],['--ent-native-nth','-1'],
            ['--ent-native-model','linearized'],['--ent-native-source','cross'],['--bogus','1']]
        for args in cases:
            with self.subTest(args=args):
                result = subprocess.run([str(BIN),*args],capture_output=True,text=True,timeout=10)
                self.assertNotEqual(result.returncode,0)
                self.assertEqual(result.stdout,'')


if __name__ == '__main__':
    unittest.main(verbosity=2)
