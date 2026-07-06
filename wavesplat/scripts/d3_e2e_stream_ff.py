"""
d3_e2e_stream_ff.py — E2E control closure (2026-07-06, roadmap C3):
translate "encounter forecast corr" into the control currency
"heave-acceleration reduction %" on the Candela C-8 simulator.

Loop: true linear sea (head seas, BF3/BF4, 15-comp JONSWAP as in the
hydrofoil paper scenarios) -> 20 s of forward-fan Falcon obs while the
ship runs at cruise speed -> StreamMMSE state at t=0 -> the filter's
bank coefficients ARE a wave-component list (A, omega, kx, ky, phi) ->
plug into the EXISTING C-8 FF controller via ff_info_provider ->
compare heave accel RMS: PD-only vs PD+GT-FF (perfect knowledge) vs
PD+Stream-FF (our pipeline). Metric: fraction of the perfect-knowledge
benefit retained.

Usage: python3 scripts/d3_e2e_stream_ff.py [--seeds 2]
"""
import os
import sys
import json
import argparse

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from wavesim.wave_generator import jonswap_components
from hydrofoil.config import candela_c8
from hydrofoil.controller import FeedforwardConfig, PDControllerConfig
from hydrofoil.sim import run_simulation, _compute_feedforward_info
from hydrofoil.wavesplat_bridge import WaveSplatComponent
from d3_sa_fim_identifiability import omega_of
from d3_stream_seqmmse import make_bank, gamma_beamform, StreamMMSE

G = 9.81
OUT = 'results/direction3/e2e_stream_ff.json'
SEAS = {'BF3': dict(Hs=0.88, Tp=5.0), 'BF4': dict(Hs=1.88, Tp=7.5)}
# paper-scenario params (hydrofoil_feedforward run_phase2 head-sea rows)


def estimate_components(comps_true, U, Hs, Tp, depth, seed):
    """20 s forward-fan warmup on the true sea -> StreamMMSE state at
    t=0 -> component list for the FF controller."""
    kx = np.array([c.kx for c in comps_true])
    ky = np.array([c.ky for c in comps_true])
    A = np.array([c.A for c in comps_true])
    ph = np.array([c.phi for c in comps_true])
    om = omega_of(kx, ky, depth)

    def eta(x, y, t):
        ps = kx[None] * np.atleast_1d(x)[:, None] \
            + ky[None] * np.atleast_1d(y)[:, None] \
            - om[None] * np.atleast_1d(t)[:, None] + ph[None]
        return (A[None] * np.cos(ps)).sum(1)

    rng = np.random.default_rng(seed)
    dt, m = 0.1, 12
    kp = (2 * np.pi / Tp) ** 2 / G
    # ship runs +x at U (sim convention); fan looks +x ahead;
    # true waves propagate -x (head seas)
    Pw, zw = [], []
    for t in np.arange(-20.0, 0.0, dt):
        sx = U * t
        rr = rng.uniform(20.0, 100.0, m)
        aa = rng.uniform(-np.pi / 3, np.pi / 3, m)
        xs = sx + rr * np.cos(aa)
        ys = rr * np.sin(aa)
        Pw.append(np.column_stack([xs, ys, np.full(m, t)]))
        zw.append(eta(xs, ys, t) + rng.normal(0, 0.02, m))
    Pw, zw = np.vstack(Pw), np.concatenate(zw)
    ii = np.random.default_rng(1).choice(len(zw), 600, False)
    kxb, kyb, omb = make_bank(kp, depth)
    gam, s2 = gamma_beamform(kxb, kyb, omb, Pw[ii], zw[ii], 0.05 * Hs,
                             depth)
    st = StreamMMSE(kxb, kyb, omb, gam, s2, tau=None, prune=400)
    for k in range(len(zw) // m):
        st.step(Pw[k * m:(k + 1) * m], zw[k * m:(k + 1) * m])
    B = len(st.kxb)
    a_c, c_c = st.w[:B], st.w[B:]
    Ae = np.hypot(a_c, c_c)
    pe = np.arctan2(-c_c, a_c)
    keep = np.argsort(Ae)[-100:]          # top-100: sim cost is O(n_comp)
    return [WaveSplatComponent(A=float(Ae[i]), omega=float(st.om[i]),
                               kx=float(st.kxb[i]), ky=float(st.kyb[i]),
                               phi=float(pe[i]))
            for i in keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=2)
    a = ap.parse_args()
    cfg = candela_c8()
    U = cfg['sim'].speed
    depth = cfg['sim'].depth
    out = {}
    for sea, sc in SEAS.items():
        rows = []
        for seed in range(a.seeds):
            comps = jonswap_components(Hs=sc['Hs'], Tp=sc['Tp'],
                                       n_components=15, depth=depth,
                                       direction_deg=180.0,
                                       direction_spread_deg=20.0,
                                       seed=7000 + seed)
            est = estimate_components(comps, U, sc['Hs'], sc['Tp'],
                                      depth, seed)
            ff = FeedforwardConfig(preview_dt=0.5, K_ff=0.8,
                                   noise_std=0.0, slope_noise_std=0.0)

            def stream_provider(state, front_foil, rear_foil, sim,
                                _components, t, heading, ff_cfg):
                return _compute_feedforward_info(
                    state, front_foil, rear_foil, sim, est, t,
                    heading, ff_cfg)

            res = {}
            for name, kw in (
                    ('PD', dict(feedforward=None)),
                    ('GT_FF', dict(feedforward=ff)),
                    ('Stream_FF', dict(feedforward=ff,
                                       ff_info_provider=stream_provider))):
                r = run_simulation(cfg, comps, vessel_heading=0.0,
                                   controller=PDControllerConfig(), **kw)
                mask = r.time > 5.0
                res[name] = float(np.std(r.heave_accel[mask]))
            red_gt = 1 - res['GT_FF'] / res['PD']
            red_st = 1 - res['Stream_FF'] / res['PD']
            frac = red_st / red_gt if red_gt > 1e-6 else np.nan
            rows.append(dict(seed=seed, a_pd=res['PD'], a_gt=res['GT_FF'],
                             a_st=res['Stream_FF'], red_gt=red_gt,
                             red_st=red_st, frac_of_perfect=frac))
            print(f"[{sea} seed{seed}] a_RMS: PD {res['PD']:.3f} | GT_FF "
                  f"{res['GT_FF']:.3f} (-{red_gt:.0%}) | Stream_FF "
                  f"{res['Stream_FF']:.3f} (-{red_st:.0%}) -> "
                  f"{frac:.0%} of perfect", flush=True)
        out[sea] = rows
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"saved {OUT}")


if __name__ == '__main__':
    main()
