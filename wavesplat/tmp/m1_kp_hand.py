"""M-1 — kp↔hand A↔B unification check. Prereg = tmp/M1_kp_hand_prereg_2026-07-22.md.
Pure re-analysis of results/direction3/e60k_hos_arena.json (no new sim).
Tests whether B-2 task-loss gain (T-L, T-N) and the failure axis (hand) scale with
under-determination proxies (kp=2pi/lam_p, lam_p, steepness), same direction as E-58.
Usage: python3 tmp/m1_kp_hand.py
"""
import json
import numpy as np
from scipy import stats

d = json.load(open('results/direction3/e60k_hos_arena.json'))
rows = d['rows']

# --- aggregate 160 rows -> 32 seas (mean over 5 draws); sea = cluster unit ---
seas = {}
for r in rows:
    s = r['sea']
    seas.setdefault(s, []).append(r)

recs = []
for s, rs in seas.items():
    T = np.mean([[r[f'T_s{i}'] for i in range(3)] for r in rs])
    L = np.mean([[r[f'L_s{i}'] for i in range(3)] for r in rs])
    N = np.mean([[r[f'N_s{i}'] for i in range(3)] for r in rs])
    hand = np.mean([r['hand'] for r in rs])
    lam_p = np.mean([r['lam_p'] for r in rs])
    steep = np.mean([r['steepness'] for r in rs])
    Tp = rs[0]['Tp']; Hs = rs[0]['Hs']
    recs.append(dict(sea=s, hand=hand, Tp=Tp, Hs=Hs, lam_p=lam_p,
                     kp=2*np.pi/lam_p, steep=steep,
                     gain_TL=T-L, gain_TN=T-N, T=T, L=L, N=N))

n = len(recs)
arr = {k: np.array([r[k] for r in recs]) for k in
       ('hand', 'lam_p', 'kp', 'steep', 'Tp', 'Hs', 'gain_TL', 'gain_TN')}
print(f"n_seas={n}  (failure seas hand<0.5: {(arr['hand']<0.5).sum()})")


def spearman_ci(x, y, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    rho = stats.spearmanr(x, y).correlation
    idx = np.arange(len(x))
    bs = []
    for _ in range(B):
        j = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(x[j])) < 3 or len(np.unique(y[j])) < 3:
            continue
        bs.append(stats.spearmanr(x[j], y[j]).correlation)
    bs = np.array(bs)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return rho, lo, hi


pairs = [
    ('PRED1a gain_TL ~ lam_p  (expect >0)', 'gain_TL', 'lam_p'),
    ('PRED1b gain_TL ~ kp     (expect <0)', 'gain_TL', 'kp'),
    ('       gain_TN ~ lam_p  (expect >0)', 'gain_TN', 'lam_p'),
    ('PRED2a hand    ~ lam_p  (expect <0)', 'hand', 'lam_p'),
    ('PRED2b hand    ~ kp     (expect >0)', 'hand', 'kp'),
    ('PRED3  gain_TL ~ hand   (expect <0, sanity)', 'gain_TL', 'hand'),
    ('       gain_TL ~ steep  (?)', 'gain_TL', 'steep'),
    ('       hand    ~ steep  (?)', 'hand', 'steep'),
    ('       gain_TL ~ Tp     (?)', 'gain_TL', 'Tp'),
    ('       hand    ~ Tp     (?)', 'hand', 'Tp'),
]
print("\n=== Spearman (n=32 seas) + sea-bootstrap 95% CI ===")
out = {}
for lab, a, b in pairs:
    rho, lo, hi = spearman_ci(arr[a], arr[b])
    excl0 = (lo > 0) or (hi < 0)
    print(f"  {lab:44s} rho={rho:+.3f}  CI[{lo:+.3f},{hi:+.3f}]  {'*0-excl' if excl0 else ''}")
    out[lab.strip()] = dict(rho=float(rho), ci=[float(lo), float(hi)], excl0=bool(excl0))

# within-failure-seas view (hand<0.5)
fw = arr['hand'] < 0.5
print(f"\n=== within failure seas (hand<0.5, n={fw.sum()}) — descriptive only (underpowered) ===")
for a in ('lam_p', 'kp', 'Tp', 'Hs', 'steep'):
    rho = stats.spearmanr(arr[a][fw], arr['gain_TL'][fw]).correlation
    print(f"  gain_TL ~ {a:6s}  rho={rho:+.3f}  (n={fw.sum()}, no CI/label)")

# direction-consistency summary vs prereg
print("\n=== prereg direction check ===")
checks = {
    'PRED1a gain_TL~lam_p >0': arr and stats.spearmanr(arr['gain_TL'], arr['lam_p']).correlation > 0,
    'PRED1b gain_TL~kp    <0': stats.spearmanr(arr['gain_TL'], arr['kp']).correlation < 0,
    'PRED2a hand~lam_p    <0': stats.spearmanr(arr['hand'], arr['lam_p']).correlation < 0,
    'PRED2b hand~kp       >0': stats.spearmanr(arr['hand'], arr['kp']).correlation > 0,
    'PRED3  gain_TL~hand  <0': stats.spearmanr(arr['gain_TL'], arr['hand']).correlation < 0,
}
for k, v in checks.items():
    print(f"  {k}: {'MATCH' if v else 'MISS'}")

json.dump(dict(n_seas=n, pairs=out,
               pred_match={k: bool(v) for k, v in checks.items()},
               recs=recs),
          open('results/direction3/m1_kp_hand.json', 'w'), indent=1, default=float)
print("\nsaved -> results/direction3/m1_kp_hand.json")
