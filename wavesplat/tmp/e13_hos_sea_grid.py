"""E-13 — give the HOS leg of the gradient enough SEAS to be testable.

PRE-REGISTERED (gates fixed before any number is produced).
Motivation: docs/E12_KP_FLOOR_AUDIT_prereg_2026-07-14.md section 10.4.

The centrepiece claim is a three-stage gradient: learning LOSES where physics is
exact (clean linear), and WINS where physics is misspecified (HOS nonlinear, real
stereo). The middle leg is the one that isolates the *physics* misspecification,
because HOS and clean share the observation operator and differ only in the wave
physics. It is therefore the load-bearing leg of the thesis sentence.

E-12e found that leg rests on TWO seas:
    d3_a1_field_hos.py:41-42   SEEDS=[0,1,2,3], HS_LIST=[1.2, 3.0]
    d3_f1_hos_door1.py:104     HS, TP, DEPTH, SPREAD = 1.2, 8.0, 50.0, 30.0
                               WDIR = 180.0
Tp, direction and directional spread are CONSTANTS. The two "sea states" differ
only in significant wave height. The seed is not a cluster (seed 0 appears at both
Hs values, so it does not name a sea), so the cluster is the sea state, k=2, and
cluster-robust inference is not even defined (t(1) critical value = 12.7).

This runs a real sea grid: Hs x Tp, with the original (Hs, Tp=8) cells included so
the new run CONTAINS the old one as a reproduction check.

CLUSTER = the sea state (Hs, Tp). Seeds within a sea are phase realisations.

--- GATES (frozen before running) ---------------------------------------------
G1 (primary): delta = net - hand (whole-field corr) > 0 on the pooled grid, with
    BOTH honest small-cluster tests agreeing at alpha=0.05:
      - sea-mean t, df = k-1
      - Webb 6-point wild cluster bootstrap-t
    PASS  -> the HOS leg is established ACROSS SEAS; the gradient's middle leg
             holds and the thesis sentence survives.
    FAIL  -> the HOS leg is NOT established. The "learning wins under
             misspecification" claim then rests on the real stage alone, and the
             gradient must be re-written as a two-point claim with HOS reported
             as directionally consistent but underpowered.

G2 (mechanism, secondary -- reported either way, not a gate on G1):
    does delta grow with wave steepness Hs/lambda_p (i.e. with the amount of
    nonlinearity the linear physics prior gets wrong)? Spearman rho over the sea
    states, plus a sea-clustered slope. A positive dose-response is the mechanism
    the thesis asserts; a flat or negative one means the HOS win is not coming
    from nonlinearity and the mechanism story must change.

Reproduction check: the (Hs 1.2, Tp 8) and (Hs 3.0, Tp 8) cells must reproduce
the frozen a1_field_hos_sparse600.json values for the same seeds.

Observation budget: 600 (E-9 unified sparse budget) -- NOT the 2000 default.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

os.environ.setdefault('A1_N_OBS', '600')          # must be set before A1 import

import d3_f1_hos_door1 as F1                       # noqa: E402
import d3_a1_field_hos as A1                       # noqa: E402
from d3_prior_net_eval import get_net, DEVICE      # noqa: E402
from d3_l1_s2head_train import SmultHead           # noqa: E402
from d3_cluster_se import cluster_se               # noqa: E402
import torch                                       # noqa: E402

G = 9.81
OUT = 'results/direction3/e13_hos_sea_grid.json'
CKPT = 'results/direction3/prior_net_l1s2bw.pt'
DEPTH = 50.0

# (Tp, Hs) sea grid. Steepness Hs/lambda_p is kept <= 0.04 so HOS M=3 stays in
# its validity range. Tp=8 rows are the FROZEN cells (reproduction check).
SEAS = [
    (7.0, 1.2), (7.0, 2.0), (7.0, 3.0),
    (8.0, 1.2), (8.0, 3.0),                        # <- the frozen pair
    (9.0, 1.2), (9.0, 2.0), (9.0, 3.0), (9.0, 4.0),
    (11.0, 1.2), (11.0, 2.0), (11.0, 3.0), (11.0, 4.0),
    (13.0, 2.0), (13.0, 4.0),
]


def lam_p(Tp, depth=DEPTH):
    om = 2 * np.pi / Tp
    k = om ** 2 / G
    for _ in range(80):
        k = om ** 2 / (G * np.tanh(k * depth))
    return float(2 * np.pi / k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=2)
    ap.add_argument('--seas', type=int, default=len(SEAS))
    ap.add_argument('--out', default=OUT)
    a = ap.parse_args()

    net = get_net(CKPT)
    head = SmultHead().to(DEVICE)
    head.load_state_dict(torch.load(CKPT, map_location=DEVICE)['head'])
    head.eval()

    rows = []
    for (Tp, Hs) in SEAS[:a.seas]:
        F1.TP = float(Tp)                # hos_run + sea_components read TP live
        lam = lam_p(Tp)
        for seed in range(a.seeds):
            t0 = time.time()
            rng = np.random.default_rng(1000 + seed)
            r = A1.run_cell(seed, float(Hs), net, head, rng)
            r.update(Tp=float(Tp), lam_p=lam, steepness=float(Hs) / lam,
                     sea=f'Tp{Tp:g}_Hs{Hs:g}')
            rows.append(r)
            print(f"[Tp {Tp:4.1f} Hs {Hs:3.1f} seed {seed}] "
                  f"steep {Hs/lam:.4f}  hand {r['hand']:+.3f}  "
                  f"net {r['learned']:+.3f}  d {r['learned']-r['hand']:+.4f}  "
                  f"[{time.time()-t0:.0f}s]", flush=True)
            json.dump(rows, open(a.out, 'w'), indent=1)   # checkpoint each cell

    d = np.array([r['learned'] - r['hand'] for r in rows])
    g = np.array([r['sea'] for r in rows])
    seas = sorted(set(g))
    k = len(seas)
    sea_means = np.array([d[g == s].mean() for s in seas])
    if k < 3:
        print(f"\nonly k={k} sea(s) — inference not defined; cells written to "
              f"{a.out}")
        json.dump({'rows': rows}, open(a.out, 'w'), indent=2)
        return

    # --- G1: the two honest small-cluster tests -------------------------------
    t_sea, p_sea = stats.ttest_1samp(sea_means, 0.0)
    se_cl, _ = cluster_se(d, g)

    def cr_t(y):
        mu_ = y.mean()
        acc = sum((y[g == s].sum() - (g == s).sum() * mu_) ** 2 for s in seas)
        # finite-cluster correction; with ONE cell per sea n == k and the (n-1)/
        # (n-k) factor is undefined -- fall back to c=1, exactly as
        # d3_cluster_se.cluster_se does. (With 1 cell/sea the cluster-robust SE
        # is just the sample SE of the k sea deltas, which is what we want.)
        n_ = len(y)
        c = (k / (k - 1)) * ((n_ - 1) / (n_ - k)) if (k > 1 and n_ > k) else 1.0
        se_ = np.sqrt(c * acc / n_ ** 2)
        return mu_ / se_ if se_ > 0 else np.inf

    rng = np.random.default_rng(0)
    WEBB = np.array([-np.sqrt(1.5), -1., -np.sqrt(.5),
                     np.sqrt(.5), 1., np.sqrt(1.5)])
    idx = {s: i for i, s in enumerate(seas)}
    ci = np.array([idx[s] for s in g])
    t_obs = cr_t(d)
    wb = np.array([cr_t(d * rng.choice(WEBB, k)[ci]) for _ in range(10000)])
    p_wild = float((np.abs(wb) >= abs(t_obs)).mean())

    print(f"\n{'='*80}\nE-13  n={len(d)} cells, k={k} SEAS "
          f"(was k=2)\n{'='*80}")
    print(f"  pooled delta      {d.mean():+.4f}   sign {(d>0).sum()}/{len(d)}"
          f"   sea sign {(sea_means>0).sum()}/{k}")
    print(f"  sea-mean t({k-1})   {t_sea:+.2f}   p {p_sea:.4f}")
    print(f"  wild boot-t (Webb) t {t_obs:+.2f}   p {p_wild:.4f}")
    g1 = bool(d.mean() > 0 and p_sea < 0.05 and p_wild < 0.05)
    print(f"\n  *** G1: {'PASS — HOS leg established ACROSS SEAS' if g1 else 'FAIL — HOS leg NOT established'}")

    # --- G2: dose-response in steepness ---------------------------------------
    st = np.array([r['steepness'] for r in rows])
    sea_st = np.array([st[g == s].mean() for s in seas])
    rho, p_rho = stats.spearmanr(sea_st, sea_means)
    sl = np.polyfit(sea_st, sea_means, 1)[0]
    print(f"\n  G2 mechanism — does the learning gain grow with steepness?")
    print(f"     Spearman rho {rho:+.3f} (p {p_rho:.4f}) over {k} seas, "
          f"slope {sl:+.3f} per unit Hs/lam")
    for s in seas:
        print(f"       {s:<14} steep {st[g==s].mean():.4f}  "
              f"d {d[g==s].mean():+.4f}")

    # --- reproduction check ---------------------------------------------------
    try:
        fz = json.load(open('results/direction3/a1_field_hos_sparse600.json'))
        print('\n  reproduction check vs frozen a1_field_hos_sparse600.json '
              '(Tp=8 cells):')
        for r in rows:
            if r['Tp'] != 8.0:
                continue
            m = [x for x in fz if x['seed'] == r['seed']
                 and abs(x['Hs'] - r['Hs']) < 1e-6]
            if m:
                print(f"     Hs {r['Hs']} seed {r['seed']}: "
                      f"new d {r['learned']-r['hand']:+.4f}  vs  "
                      f"frozen d {m[0]['learned']-m[0]['hand']:+.4f}")
    except FileNotFoundError:
        pass

    json.dump({'rows': rows,
               'summary': dict(n=len(d), k_seas=k, delta=float(d.mean()),
                               sea_means={s: float(m) for s, m in
                                          zip(seas, sea_means)},
                               t_sea=float(t_sea), p_sea=float(p_sea),
                               t_wild=float(t_obs), p_wild=p_wild,
                               G1_pass=g1, spearman_rho=float(rho),
                               spearman_p=float(p_rho), slope=float(sl))},
              open(a.out, 'w'), indent=2)
    print(f"\n-> {a.out}")


if __name__ == '__main__':
    sys.exit(main())
