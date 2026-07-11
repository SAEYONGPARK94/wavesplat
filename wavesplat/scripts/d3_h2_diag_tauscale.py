"""H2 post-hoc diagnostic (NO GATE): anatomy of the accounting-estimation gap.

Question. HA accounting predicts ~ +0.009 annulus-elevation surplus from the
mask; the hand EP realizes +0.002 (ns), and the run-1 task-loss net went
NEGATIVE. Is the surplus reachable by ANY site strength at all?

Probe: scale the converged EP sites by c in {0, 1/4, 1/2, 1, 2, 4} (mu = nu/
tau kept fixed) and score the H1 annulus instrument, paired, on a FRESH probe
set (seeds 500-507; run-1 val was consumed, run-2 val must stay untouched).
Also: return-sites-only and miss-sites-only at c=1.

Readings.
  - gain(c) maximized near c=1 at ~ +0.002  -> EP already sits at the top of
    its 1-parameter family; the accounting surplus is unreachable by site
    strength => the +0.007 gap is PHANTOM (overconfident-posterior accounting),
    not inference loss. The learning target never existed.
  - gain(c*) ~ +0.009 for some c* != 1     -> the EP is mis-scaled; inference
    loss is real and a trivial calibration (not a net) recovers it.
  - return-only vs miss-only decomposes WHERE any effect lives.

Post-hoc diagnostic, disclosed as such. No verdict authority over H2 gates.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

import d3_g0b_selection_aware as g0b
from d3_g0b_selection_aware import DEV, DT
from d3_h1_mask_destination import sea_dir
from d3_h2_taskloss_sites import add_annulus, ann_nrmse
from d3_n5_site_head import scene_tensors, teacher_sites, solve_with

OUT = 'results/direction3/h2_diag_tauscale.json'
SEEDS = list(range(500, 508))
CS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]

g0b.sea = sea_dir(180.0)
rng = np.random.default_rng(913)

from d3_s2_mask_information import grad_design
from d3_door1b_field_bias import R_MIN, R_MAX


def perm_sites(sc, rng):
    """break field correspondence: shuffle MISS footprints within range bins
    (H1/G0b's G3 control), leaving returns, ranges, T, weights untouched."""
    sp = sc['sitesP'].clone()
    ret = sc['is_ret'].cpu().numpy()
    mr = sc['sites_r'].cpu().numpy()
    idx = np.where(~ret)[0]
    edges = np.linspace(R_MIN, R_MAX, 13)
    b = np.clip(np.digitize(mr[idx], edges) - 1, 0, 11)
    xy = sp[:, :2].cpu().numpy()
    for j in range(12):
        m = idx[b == j]
        if len(m) > 1:
            xy[m] = xy[rng.permutation(m)]
    sp[:, :2] = torch.tensor(xy, device=sp.device, dtype=sp.dtype)
    return sp


rows = []
for sd in SEEDS:
    sc = scene_tensors(sd, rng)
    add_annulus(sc, rng)
    a0 = solve_with(sc, None, None, None)
    e0 = float(ann_nrmse(sc, a0))
    G, tau, nu = teacher_sites(sc)
    r = {'seed': sd, 'naive': e0, 'gains': {}}
    for c in CS:
        ac = solve_with(sc, G, c * tau, c * nu)
        r['gains'][str(c)] = e0 - float(ann_nrmse(sc, ac))
    ret = sc['is_ret']
    for name, m in (('ret_only', ret), ('miss_only', ~ret)):
        tm, nm = tau.clone(), nu.clone()
        tm[~m] = 0.0
        nm[~m] = 0.0
        ac = solve_with(sc, G, tm, nm)
        r['gains'][name] = e0 - float(ann_nrmse(sc, ac))
    # PAIRED perm control: same scene, field link broken, fresh EP
    sp = perm_sites(sc, rng)
    sc2 = dict(sc)
    sc2['sitesP'] = sp
    Gp, taup, nup = teacher_sites(sc2)
    for c in (1.0, 4.0):
        ac = solve_with(sc2, Gp, c * taup, c * nup)
        r['gains'][f'perm{c:g}'] = e0 - float(ann_nrmse(sc2, ac))
    rows.append(r)
    print(f"seed {sd} naive {e0:.4f} | " +
          " ".join(f"c{k} {v:+.4f}" for k, v in r['gains'].items()),
          flush=True)
    del G, tau, nu, Gp, taup, nup, sc, sc2
    torch.cuda.empty_cache()

print("\n" + "=" * 70)
keys = [str(c) for c in CS] + ['ret_only', 'miss_only', 'perm1', 'perm4']
summ = {}
for k in keys:
    v = np.array([r['gains'][k] for r in rows])
    se = v.std(ddof=1) / np.sqrt(len(v))
    summ[k] = [float(v.mean()), float(se)]
    print(f"c = {k:9s} gain {v.mean():+.5f} +- {se:.5f}")
json.dump({'rows': rows, 'summary': summ}, open(OUT, 'w'), indent=2)
print(f"wrote {OUT}")
