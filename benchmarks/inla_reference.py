"""Direct published BYM2/INLA checks; no Astrolabe adapter or new inference method."""
import json
from pathlib import Path
import sys
import tempfile
import numpy as np
import pandas as pd
import scipy.linalg as la
import scipy.sparse as sp
import pyinla
from importlib.metadata import version

if version('pyinla') != '0.1.11' or not pyinla.is_binary_installed():
    raise RuntimeError('Explicitly install pyINLA 0.1.11 and its engine before running this reference')

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
out = args.output.resolve()
out.parent.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(811)
n_core, per_core = 8, 12
adj = np.zeros((per_core, per_core))
for i in range(per_core):
    for j in range(i):
        if abs(i//4-j//4)+abs(i%4-j%4) == 1:
            adj[i, j] = adj[j, i] = 1
q = np.diag(adj.sum(axis=1))-adj
qplus = la.pinvh(q)
qplus /= np.exp(np.log(np.diag(qplus)).mean())
graph = sp.block_diag([adj]*n_core, format='csr')
spatial_cov = la.block_diag(*([qplus]*n_core))
n = graph.shape[0]
donor = np.repeat(np.arange(n_core//2), per_core*2)
z = np.eye(n_core//2)[donor]
x = np.tile(np.arange(per_core)%4, n_core).astype(float)
x = (x-x.mean())/x.std()
design = np.column_stack([np.ones(n), x])
variance = .1**2*np.eye(n)+.3**2*(.5*spatial_cov+.5*np.eye(n))+.2**2*z@z.T
response = design@np.array([1., .5])+rng.multivariate_normal(np.zeros(n), variance)
data = pd.DataFrame({'y': response, 'x': x, 'cell': np.arange(1, n+1), 'donor': donor+1})
model = {'response': 'y', 'fixed': ['1', 'x'], 'random': [
    {'id': 'cell', 'model': 'bym2', 'graph': graph, 'scale.model': True,
     'adjust.for.con.comp': True, 'hyper': [
         {'initial': float(np.log(1/.3**2)), 'fixed': True},
         {'initial': 0., 'fixed': True}]},
    {'id': 'donor', 'model': 'iid', 'hyper': [
        {'initial': float(np.log(1/.2**2)), 'fixed': True}]}]}
control = {'family': {'hyper': [{'id': 'prec', 'initial': float(np.log(1/.1**2)), 'fixed': True}]}}
result = pyinla.pyinla(model=model, family='gaussian', data=data, control=control)
vinv_x = la.solve(variance, design, assume_a='pos')
posterior_cov = la.inv(design.T@vinv_x+np.diag([0., .001]))
posterior_mean = posterior_cov@vinv_x.T@response
summary = result.summary_fixed.loc[['(Intercept)', 'x']]
report = {'backend': 'pyINLA 0.1.11', 'seed': 811, 'n_cells': n, 'n_cores': n_core,
          'n_donors': n_core//2, 'gaussian': {'direct': summary.to_dict(),
          'analytic_mean': posterior_mean.tolist(), 'analytic_sd': np.sqrt(np.diag(posterior_cov)).tolist()}}
import hashlib
from pyinla.binary import BinaryManager, get_binary_version
report['engine'] = get_binary_version()
launcher = Path(BinaryManager().get_binary_path())
engine = launcher
if launcher.suffix == '.run':
    engine = next(path for path in (launcher.parent/'bin/inla', launcher.parent/'inla.mkl', launcher.parent/'inla') if path.is_file())
report['engine_sha256'] = hashlib.sha256(engine.read_bytes()).hexdigest()
report['launcher_sha256'] = hashlib.sha256(launcher.read_bytes()).hexdigest()
build_info = launcher.parent/'BUILDINFO'
report['build_info'] = build_info.read_text() if build_info.exists() else 'unavailable'
report['gaussian']['max_mean_error'] = float(np.max(np.abs(summary['mean'].to_numpy()-posterior_mean)))
report['gaussian']['max_sd_error'] = float(np.max(np.abs(summary['sd'].to_numpy()-np.sqrt(np.diag(posterior_cov)))))
np.testing.assert_allclose(summary['mean'], posterior_mean, atol=1e-5, rtol=1e-5)
np.testing.assert_allclose(summary['sd'], np.sqrt(np.diag(posterior_cov)), atol=1e-5, rtol=1e-4)
out.write_text(json.dumps(report, indent=2))
print(json.dumps(report), flush=True)

# A count-model computation check uses the documented NB2 likelihood, a
# scaled disconnected BYM2 field and one IID effect shared across donor cores.
eta = design@np.array([2., .5])+rng.multivariate_normal(np.zeros(n), variance-.1**2*np.eye(n))
offset = np.log(rng.uniform(.5, 2., n))
mu = np.exp(eta+offset)
data['y'] = rng.negative_binomial(10, 10/(10+mu))
for term in model['random']:
    term.pop('hyper')
model['random'][1]['hyper'] = [{'prior': 'pc.prec', 'param': [1., .01]}]
report['donor_sd_prior'] = [1., .01]
from pyinla.pc_bym import pc_bym_phi
prior = pc_bym_phi(graph=graph.toarray(), alpha=.5, u=.5, scale_model=True,
                   adjust_for_con_comp=True, return_as_table=True)
table = np.column_stack([prior['theta'], prior['logprior']])
model['random'][0]['hyper'] = [{}, {'prior': 'table: '+ ' '.join(format(v, '.17g') for v in table.ravel(order='F'))}]
temporary = tempfile.TemporaryDirectory(prefix='inla-reference-')
work = Path(temporary.name)
counts = pyinla.pyinla(model=model, family='nbinomial', data=data, offset=offset,
    control={'inla': {'strategy':'simplified.laplace', 'int.strategy':'ccd'}},
    keep=True, working_directory=str(work.resolve()), num_threads=1)
report['negative_binomial'] = {'planted_effect': .5, 'fixed': counts.summary_fixed.to_dict(),
                               'hyperparameters': counts.summary_hyperpar.to_dict()}
from pyinla.fmesher_io import read_fmesher_file
artifact = work/'inla.model'
line = next(v for v in (artifact/'Model.ini').read_text().splitlines() if v.startswith('prior1 = table:'))
filename = line.replace(chr(92), '/').rsplit('/', 1)[-1]
actual_prior = read_fmesher_file(str(artifact/'data.files'/filename))
np.testing.assert_array_equal(actual_prior, table)
report['prior_table'] = {'shape': list(table.shape), 'exact_match': True,
    'integrated_density': float(np.trapezoid(np.exp(table[:, 1]), table[:, 0])),
    'cdf_at_phi_half': float(np.trapezoid(np.exp(table[table[:,0] <= 0, 1]), table[table[:,0] <= 0, 0]))}
report['R_loaded'] = any(name == 'rpy2' or name.startswith('rpy2.') for name in sys.modules)
out.write_text(json.dumps(report, indent=2))
print(json.dumps(report['negative_binomial']), flush=True)
temporary.cleanup()
