"""
shortlambda_revalidate.py — the #1 re-validation: does "weak short-lambda components are
unrecoverable by ALL methods at ALL apertures" survive on a Nyquist-adequate dataset?

Why this one first. That sentence (tmp/NOVELTY_ROUND3_execution_2026-07-23.md, the ANM
feasibility gate) is currently the *empirical* leg of the project's information-wall
argument. It was measured on `data/`, where `wavesim/mesh_adequacy.py` fails 0/102 episodes
and content shorter than ~14 m was not representable by the raycast mesh at all. If the weak
short-lambda component was never rendered into the truth field, "no method recovers it" is
circular.

The regenerated set `data_finemesh_2026-07-23/` (200x167, dx = 1.206 m) is a PAIRED
replacement: same seeds -> wave components bit-match the shipped files (verified 100/100).
So the same episode can be run old vs new with everything else held fixed.

Method: for each episode, run the same greedy + joint-refine component estimator used by the
ANM gate (tmp/blasso_feasibility.py) on the OLD and NEW file, match recovered atoms to the
true `config/wave_components`, and report recovery of (i) all components and (ii) the weakest
and the shortest-wavelength component specifically -- the ones the claim is about.

Pre-registered decision rule (stated before the numbers):
  - Weak/short components recovered on NEW but not OLD -> the claim was a RENDERING
    ARTIFACT. The information-wall's empirical leg must be re-derived, and this is a
    significant correction.
  - Still unrecovered on NEW -> the claim SURVIVES the strongest available challenge. That
    is the outcome that matters most for the thesis, and it should be reported as such
    (it does not *prove* the wall -- it removes the most obvious confound).
  - Recovery improves but only partially -> report the fraction and restate the claim with
    the mesh caveat rather than keeping the absolute phrasing.
Secondary: field nRMSE old vs new, as context only -- the pilot already showed the aggregate
headline does not move, so nRMSE is NOT the metric this question turns on.

Note on variance: the greedy/joint estimator here is the deterministic numpy pipeline from
blasso_feasibility (module-level seeded RNG), not the torch/Adam tracker whose bimodal
CV-53% variance is documented in tmp/ESTIMATOR_VARIANCE_RESULT_2026-07-23.md. Determinism is
asserted, not assumed: the script re-runs one episode twice and checks identity.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, 'tmp')
import numpy as np
import h5py
from blasso_feasibility import (load_obs, greedy, joint_refine, field_nrmse,
                                build_grid, corr_at, match_to_gt)

NEW_DIR = 'data_finemesh_2026-07-23'
OLD_DIR = 'data'
RESULT = 'tmp/shortlambda_revalidate_results.json'

EPISODES = [
    ('episode_BF3_2comp_seed0_uniform.h5', 2),
    ('episode_BF3_5comp_seed0_uniform.h5', 5),
    ('episode_BF3_5comp_seed1_uniform.h5', 5),
    ('episode_BF4_15comp_seed0_uniform.h5', 15),
]


def run_one(path, n_true, nsub=2500):
    obs = load_obs(path)
    rng = np.random.default_rng(0)
    M = len(obs['x'])
    sub = rng.choice(M, size=min(nsub, M), replace=False)
    gridk = build_grid()
    lam = 0.03 * np.abs(corr_at(*gridk, obs['x'][sub], obs['y'][sub], obs['t'][sub],
                                obs['z'][sub], obs['depth']).max())
    atoms = joint_refine(greedy(obs, n_true + 1, gridk, sub), obs, sub, lam)
    rows = match_to_gt(atoms, obs['wc'])          # per-true-component match records
    wc = obs['wc']
    km = np.hypot(wc[:, 2], wc[:, 3])
    lam_m = 2 * np.pi / np.maximum(km, 1e-9)
    i_weak = int(np.argmin(wc[:, 0]))             # weakest amplitude
    i_short = int(np.argmin(lam_m))               # shortest wavelength
    hits = [bool(r[6]) for r in rows]
    fr = obs['nfr'] // 2
    return dict(n_true=n_true, n_hit=int(sum(hits)),
                weak_hit=hits[i_weak], weak_A=float(wc[i_weak, 0]),
                weak_lambda=float(lam_m[i_weak]),
                short_hit=hits[i_short], short_A=float(wc[i_short, 0]),
                short_lambda=float(lam_m[i_short]),
                nrmse=float(field_nrmse(atoms, obs, fr)))


res = json.load(open(RESULT)) if os.path.exists(RESULT) else {}

# determinism check (asserted, not assumed)
if 'determinism' not in res:
    p = os.path.join(NEW_DIR, EPISODES[0][0])
    a = run_one(p, EPISODES[0][1]); b = run_one(p, EPISODES[0][1])
    res['determinism'] = dict(first=a['nrmse'], second=b['nrmse'],
                              identical=bool(a['nrmse'] == b['nrmse']))
    json.dump(res, open(RESULT, 'w'), indent=2)
    print(f"determinism check: {a['nrmse']:.6f} vs {b['nrmse']:.6f} -> "
          f"{'IDENTICAL' if a['nrmse']==b['nrmse'] else 'NON-DETERMINISTIC (invalidates paired read)'}",
          flush=True)

for fn, n_true in EPISODES:
    res.setdefault(fn, {})
    for tag, d in (('old', OLD_DIR), ('new', NEW_DIR)):
        if tag in res[fn]:
            continue
        p = os.path.join(d, fn)
        if not os.path.exists(p):
            print(f'  missing {p}', flush=True); continue
        t0 = time.time()
        res[fn][tag] = run_one(p, n_true)
        json.dump(res, open(RESULT, 'w'), indent=2)
        r = res[fn][tag]
        print(f"  {fn:<40} {tag:<3} recovered {r['n_hit']}/{r['n_true']}  "
              f"weak(A={r['weak_A']:.3f},lam={r['weak_lambda']:.1f}m)={'HIT' if r['weak_hit'] else 'MISS'}  "
              f"short(lam={r['short_lambda']:.1f}m)={'HIT' if r['short_hit'] else 'MISS'}  "
              f"nRMSE={r['nrmse']:.4f}  [{time.time()-t0:.0f}s]", flush=True)
    if 'old' in res[fn] and 'new' in res[fn]:
        o, n = res[fn]['old'], res[fn]['new']
        print(f"{fn:<40} OLD->NEW  recovered {o['n_hit']}->{n['n_hit']} / {o['n_true']}   "
              f"weak {'HIT' if o['weak_hit'] else 'MISS'}->{'HIT' if n['weak_hit'] else 'MISS'}   "
              f"short {'HIT' if o['short_hit'] else 'MISS'}->{'HIT' if n['short_hit'] else 'MISS'}   "
              f"nRMSE {o['nrmse']:.4f}->{n['nrmse']:.4f}\n", flush=True)
