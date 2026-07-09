"""Wall-clock cost: WaveSplat (analytic prior) vs WavePrior (learned prior).

The "analytic is cheaper" intuition assumes the network is the expensive part.
Both paths share the same physics MMSE over the same bank; the analytic path
additionally val-selects a prior power exponent and runs val-gated SBL-EM,
while the learned path replaces all of that with one forward pass. So which
is cheaper is an empirical question, not an obvious one.

Timed on the real YS17 scene at a control-relevant query load.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..'))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, 'scripts')

from d3_compare_gifs import scene_real
from d3_sa_nomp import mmse_bank_estimate
from d3_prior_net_eval import get_net, net_gamma_mmse

NET_CKPT = 'results/direction3/prior_net_v3.pt'
REPS = 5


def timeit(fn, reps=REPS):
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts)), float(np.std(ts))


def main():
    sc = scene_real('YS17', win=3)
    P, z, depth, Hs, kp = sc['P'], sc['z'], sc['depth'], sc['Hs'], sc['kp']
    s0 = 0.05 * Hs
    net = get_net(NET_CKPT)

    # Control-relevant query load: the encounter point at leads 0..2 s.
    # This is what the feed-forward controller actually consumes.
    Pq_ctrl = sc['Pq'][-len(sc['leads']):][:3]
    # Full map load, for reference (situational-awareness product).
    Pq_map = sc['Pq']

    print(f"scene: {len(z)} obs, depth {depth} m, Hs {Hs:.2f} m")
    print(f"query loads: control {len(Pq_ctrl)} pts, map {len(Pq_map)} pts\n")

    a_m_ctrl = None
    for label, Pq in (('control band (3 queries)', Pq_ctrl),
                      ('full map (%d queries)' % len(Pq_map), Pq_map)):
        a_m, a_s = timeit(lambda: mmse_bank_estimate(
            P, z, depth, Pq, sigma0=s0, kp_hint=kp, seed=0))
        b_m, b_s = timeit(lambda: net_gamma_mmse(
            net, P, z, depth, Pq, sigma0=s0, seed=0, kp_hint=kp))
        if a_m_ctrl is None:
            a_m_ctrl = a_m
        print(f"[{label}]")
        print(f"  WaveSplat (analytic) : {1e3*a_m:8.1f} ms  (sd {1e3*a_s:.1f})")
        print(f"  WavePrior (learned)  : {1e3*b_m:8.1f} ms  (sd {1e3*b_s:.1f})")
        print(f"  ratio learned/analytic: {b_m/a_m:.2f}x\n")

    # Where does each path spend its time?
    import torch
    from d3_prior_net_eval import NGRID, KREL, DEVICE
    from d3_sa_nomp import beamform_power, split_masks
    m1, m2 = split_masks(P, mode='legs')
    _, _, p1 = beamform_power(P[m1], z[m1], depth, KREL * kp, NGRID)
    _, _, p2 = beamform_power(P[m2], z[m2], depth, KREL * kp, NGRID)
    p1 = (p1 / max(p1.max(), 1e-12)).reshape(NGRID, NGRID)
    p2 = (p2 / max(p2.max(), 1e-12)).reshape(NGRID, NGRID)
    pm = np.minimum(p1, p2)
    X = np.stack([p1, p2, pm,
                  np.log10(p1 + 1e-6) / 6 + 1, np.log10(p2 + 1e-6) / 6 + 1,
                  np.log10(pm + 1e-6) / 6 + 1]).astype(np.float32)
    with torch.no_grad():
        f_m, _ = timeit(lambda: net(torch.tensor(X[None], device=DEVICE)))
    bf_m, _ = timeit(lambda: beamform_power(P, z, depth, KREL * kp, NGRID))
    # analytic without the SBL-EM refinement, to price that stage
    noem_m, _ = timeit(lambda: mmse_bank_estimate(
        P, z, depth, Pq_ctrl, sigma0=s0, kp_hint=kp, seed=0, max_em=0))
    print(f"net forward pass alone   : {1e3*f_m:.2f} ms")
    print(f"one beamform power map   : {1e3*bf_m:.2f} ms  "
          f"(both methods need these)")
    print(f"analytic with max_em=0   : {1e3*noem_m:.1f} ms  "
          f"(vs {1e3*a_m_ctrl:.1f} ms with EM)")


if __name__ == '__main__':
    sys.exit(main())
