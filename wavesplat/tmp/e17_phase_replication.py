"""E-17 -- phase-seed replication of E-16's short-lambda sign flip.

E-16's one non-negotiable caveat: every cell was a SINGLE phase realization
(seed 0). The headline physics evidence -- at short lambda the net LOSES on the
exact-linear field (Delta<0) but WINS on the nonlinear M3 field (Delta>0),
dd>0 -- therefore rested on 2 wavelengths x 1 phase. This re-runs the short-lam
seas across INDEPENDENT phase realizations (seed = the JONSWAP phase/direction
draw in sea_components AND hos_run), rebuilding BOTH arms fresh per seed (no
E-15 reuse), and asks: does the sign flip (Delta_lin<0 < Delta_M3, dd>0) hold
across phases?

Gate: dd>0 across the (sea x seed) cells, clustered on phase seed; and the
per-cell sign flip rate. PASS -> the short-lam physics claim generalizes beyond
seed 0. FAIL/mixed -> it was a single-realization artifact; report honestly.
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
sys.path.insert(0, 'tmp')
os.environ.setdefault('A1_N_OBS', '600')

import torch                                                     # noqa: E402
from e16_physics_operator import build_sea_linear, build_sea_M, one_draw  # noqa
from e15_hos_multidraw import CKPT, lam_p                        # noqa: E402
from d3_prior_net_eval import get_net, DEVICE                    # noqa: E402
from d3_l1_s2head_train import SmultHead                         # noqa: E402

OUT = 'results/direction3/e17_phase_replication.json'
# short-lam regime seas where E-16 saw the sign flip (Tp6/7). Steeper ones where
# the flip is clearest; a mid/long-lam control sea confirms dd~0 there too.
SHORT = [(6.0, 2.0), (7.0, 2.0), (7.0, 3.0)]
CONTROL = [(11.0, 3.0)]          # long-lam: expect dd ~ 0 across phases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    ap.add_argument('--draws', type=int, default=6)
    ap.add_argument('--out', default=OUT)
    a = ap.parse_args()

    net = get_net(CKPT)
    head = SmultHead().to(DEVICE)
    head.load_state_dict(torch.load(CKPT, map_location=DEVICE)['head'])
    head.eval()

    rows = []
    for tag, seas in (('short', SHORT), ('control', CONTROL)):
        for (Tp, Hs) in seas:
            lam = lam_p(Tp)
            for seed in a.seeds:
                t0 = time.time()
                sea_l = build_sea_linear(Tp, Hs, seed)
                sea_3 = build_sea_M(Tp, Hs, seed, 3)
                dl, d3 = [], []
                for r in range(a.draws):
                    rng = np.random.default_rng(500000 + 31 * r)
                    hl, nl, _ = one_draw(sea_l, rng, net, head)
                    dl.append(nl - hl)
                    rng = np.random.default_rng(500000 + 31 * r)
                    h3, n3, _ = one_draw(sea_3, rng, net, head)
                    d3.append(n3 - h3)
                dl, d3 = np.array(dl), np.array(d3)
                dlin, dm3 = float(dl.mean()), float(d3.mean())
                flip = bool(dlin < 0 and dm3 > 0)
                rows.append(dict(regime=tag, sea=f'Tp{Tp:g}_Hs{Hs:g}', Tp=Tp,
                                 Hs=Hs, lam=lam, seed=seed, d_lin=dlin,
                                 d_m3=dm3, dd=dm3 - dlin, flip=flip))
                print(f"[{tag:7s} Tp{Tp:g} Hs{Hs:g} lam{lam:.0f} seed{seed}] "
                      f"Dlin {dlin:+.3f} Dm3 {dm3:+.3f} dd {dm3-dlin:+.3f} "
                      f"flip={flip} [{time.time()-t0:.0f}s]", flush=True)
                json.dump(rows, open(a.out, 'w'), indent=1)

    # inference on the short-lam regime, clustered on phase seed
    sr = [r for r in rows if r['regime'] == 'short']
    dd = np.array([r['dd'] for r in sr])
    seeds = sorted({r['seed'] for r in sr})
    # per-seed mean dd (cluster = phase realization)
    seed_dd = np.array([np.mean([r['dd'] for r in sr if r['seed'] == s])
                        for s in seeds])
    t_s, p_s = stats.ttest_1samp(seed_dd, 0.0)
    nflip = sum(r['flip'] for r in sr)
    ctrl = [r for r in rows if r['regime'] == 'control']
    ctrl_dd = np.array([r['dd'] for r in ctrl])

    print(f"\n{'='*70}\nE-17 phase replication\n{'='*70}")
    print(f"  SHORT-lam: {len(sr)} cells ({len(seeds)} phase seeds)")
    print(f"    dd pooled {dd.mean():+.4f}  per-seed-mean t({len(seeds)-1}) "
          f"{t_s:+.2f} p {p_s:.4f}   sign-flip {nflip}/{len(sr)} cells")
    print(f"    d_lin mean {np.mean([r['d_lin'] for r in sr]):+.4f} "
          f"(neg {sum(r['d_lin']<0 for r in sr)}/{len(sr)}) | "
          f"d_m3 mean {np.mean([r['d_m3'] for r in sr]):+.4f} "
          f"(pos {sum(r['d_m3']>0 for r in sr)}/{len(sr)})")
    print(f"  CONTROL long-lam: dd {ctrl_dd.mean():+.4f} "
          f"(n={len(ctrl_dd)}) -- expect ~0")
    gate = bool(dd.mean() > 0 and p_s < 0.05)
    print(f"  *** short-lam physics generalizes across phases: "
          f"{'PASS' if gate else 'NOT CONFIRMED'}")
    json.dump({'rows': rows, 'short_dd': float(dd.mean()),
               'short_t': float(t_s), 'short_p': float(p_s),
               'n_flip': nflip, 'n_short': len(sr),
               'control_dd': float(ctrl_dd.mean()), 'gate': gate},
              open(a.out, 'w'), indent=2)
    print(f"\n-> {a.out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
