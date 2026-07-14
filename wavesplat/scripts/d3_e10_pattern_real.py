"""E-10 (real arm): does the headline survive if the returns land somewhere else?

Prereg: docs/E10_PATTERN_prereg_2026-07-13.md (gates frozen before numbers).

Every synthetic and real result in this project draws its observations uniformly
over the surface the sensor sees.  Nothing justifies that.  At 1550 nm over water
at grazing incidence the return mechanism is genuinely unknown -- the repo's own
radiometry has a category error (E-6b) -- and if returns come from foam and
microscale roughness they land on crests and steep faces, not everywhere.

Sparsifying to 600 obs/window (E-9) restored the return *count*.  It did nothing
about the *pattern*.  This tests the pattern, on the real stereo records, which
give eta at every pixel and so let the crest/slope-conditional laws be applied to
real seas rather than only to the simulator.

Observation COUNT is held at 600 in every arm; only the draw probability moves.
Everything else -- window construction, hints, arms, queries -- is
d3_gate_promotion_eval verbatim.
"""
import json
import os
import sys
import time

import numpy as np
from scipy import stats
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from d3_cluster_se import cluster_se
from d3_falcon_real_operator import RECORDS, bilinear, resolve_direction  # noqa: E402,E501
from d3_gate_promotion_eval import RECS, W                                # noqa: E402
from d3_gpu_mmse import mmse_gpu                                          # noqa: E402
from d3_l1_s2head_train import SmultHead                                  # noqa: E402
from d3_prior_net_eval import DEVICE, get_net, net_gamma_mmse             # noqa: E402
from d3_sa_multirecord import MRecord                                     # noqa: E402
from realdata_analysis import directional_spread                          # noqa: E402

G = 9.81
OUT = 'results/direction3/e10_pattern_real.json'
N_OBS = 600          # E-9 sparse budget -- identical in every arm
ARMS = ['S0_uniform', 'S1_crest', 'S2_slope', 'S3_crest_only']


def weights(arm, eta, idx, dx, dy):
    """Draw probability on the valid in-fan pixels (flat indices `idx`)."""
    flat = eta.ravel()
    e = flat[idx]
    if arm == 'S0_uniform':
        return np.ones(len(e))
    if arm == 'S1_crest':
        # foam and breaking ride the crests: 4x odds above the upper quartile
        return 1.0 + 3.0 * (e >= np.nanquantile(e, 0.75))
    if arm == 'S2_slope':
        gy, gx = np.gradient(np.nan_to_num(eta), dy, dx)
        s = np.hypot(gx, gy).ravel()[idx]
        med = max(float(np.nanmedian(s)), 1e-9)
        return np.clip(s / med, 0.0, 20.0) ** 2 + 1e-6
    if arm == 'S3_crest_only':
        # pure-foam upper bound.  The literature calls this an exaggeration:
        # whitecap coverage is 0.01-1% of the surface (Monahan), and foam
        # reflectance falls off into the SWIR.  It bounds, it does not model.
        return (e >= np.nanquantile(e, 0.90)).astype(float) + 1e-9
    raise ValueError(arm)


def window_data(rec, cfg, i0, nfw, n_lead, wi, arm):
    """d3_gate_promotion_eval.window_data, with the draw law swapped."""
    rng = np.random.default_rng(700 + wi)
    Zmem = rec.window(i0 - nfw + 1, nfw)
    Zfut = rec.window(i0, n_lead + 1)
    t0 = float(rec.t[i0])
    Hs = 4.0 * float(np.nanstd(Zmem))
    spread, lam_p, pdir_ax, kp = directional_spread(
        np.nan_to_num(Zmem[-1]), rec.dx, rec.dy)
    om_p = np.sqrt(G * kp * np.tanh(kp * cfg['depth']))
    pdir, _ = resolve_direction(Zmem, rec, kp, pdir_ax, om_p / kp)
    h = -np.array([np.cos(np.radians(pdir)), np.sin(np.radians(pdir))])
    cx = 0.5 * (rec.gx.min() + rec.gx.max())
    cy = 0.5 * (rec.gy.min() + rec.gy.max())
    Xg, Yg = np.meshgrid(rec.gx, rec.gy)
    W_max, U = 40.0, cfg['U']
    T_tr = W_max + cfg['lead_max']

    def ship_pos(t):
        return np.array([cx, cy]) + U * (t - (t0 - W_max) - T_tr / 2) * h

    r0, r1 = cfg['r_fan']
    P_all, z_all = [], []
    for j in range(nfw):
        t_j = float(rec.t[i0 - nfw + 1 + j])
        sp = ship_pos(t_j)
        RX, RY = Xg - sp[0], Yg - sp[1]
        R = np.hypot(RX, RY)
        ca = (RX * h[0] + RY * h[1]) / np.maximum(R, 1e-9)
        m = ((R >= r0) & (R <= r1) & (ca >= 0.5) & np.isfinite(Zmem[j]))
        idx = np.flatnonzero(m)
        if len(idx) == 0:
            continue
        w = weights(arm, Zmem[j], idx, rec.dx, rec.dy)
        w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
        if w.sum() <= 0:
            continue
        k = min(12, len(idx))
        pick = rng.choice(idx, k, replace=False, p=w / w.sum())
        iy, ix = np.unravel_index(pick, Zmem[j].shape)
        P_all.append(np.column_stack([rec.gx[ix], rec.gy[iy],
                                      np.full(len(pick), t_j)]))
        z_all.append(Zmem[j][iy, ix] + rng.normal(0, 0.02, len(pick)))

    P_all, z_all = np.vstack(P_all), np.concatenate(z_all)
    mm = P_all[:, 2] >= t0 - W
    P, z = P_all[mm], z_all[mm]
    if len(z) > N_OBS:
        ii = np.random.default_rng(0).choice(len(z), N_OBS, False)
        P, z = P[ii], z[ii]

    leads = np.arange(0.0, cfg['lead_max'] + 1e-9, 1.0)
    Pq = np.array([[*ship_pos(t0 + l_), t0 + l_] for l_ in leads])
    gt_q = np.array([bilinear(Zfut[int(round(l_ / rec.dt))],
                              rec.gx, rec.gy, px, py)
                     for (px, py, _t), l_ in zip(Pq, leads)])
    return P, z, Pq, gt_q, leads, Hs, kp


def main():
    net = get_net('results/direction3/prior_net_l1s2bw.pt')
    head = SmultHead().to(DEVICE)
    head.load_state_dict(torch.load('results/direction3/prior_net_l1s2bw.pt',
                                    map_location=DEVICE)['head'])
    head.eval()

    rows = []
    for name in RECS:
        cfg = RECORDS[name]
        rec = MRecord(cfg['path'], stride=cfg['stride'])
        nfw = int(round(40.0 / rec.dt))
        n_lead = int(round(cfg['lead_max'] / rec.dt))
        i0s = np.linspace(nfw + 1, rec.T_all - n_lead - 2, 12).astype(int)
        for wi, i0 in enumerate(i0s):
            t0 = time.time()
            row = dict(record=name, wi=wi)
            ok = True
            for arm in ARMS:
                P, z, Pq, gt_q, leads, Hs, kp = window_data(
                    rec, cfg, int(i0), nfw, n_lead, wi, arm)
                v = np.isfinite(gt_q) & (leads <= 10.0)
                if v.sum() < 5:
                    ok = False
                    break
                sig0 = 0.05 * Hs
                with torch.no_grad():
                    ph = mmse_gpu(
                        torch.tensor(P, device=DEVICE, dtype=torch.float32),
                        torch.tensor(z, device=DEVICE, dtype=torch.float32),
                        cfg['depth'],
                        torch.tensor(Pq, device=DEVICE, dtype=torch.float32),
                        sig0, kp_hint=kp).cpu().numpy()
                    pn = np.asarray(net_gamma_mmse(
                        net, P, z, cfg['depth'], Pq, sig0, seed=int(i0),
                        kp_hint=kp, use_em=False, smult_head=head))
                hh = float(np.corrcoef(ph[v], gt_q[v])[0, 1])
                nn = float(np.corrcoef(pn[v], gt_q[v])[0, 1])
                row[arm] = dict(hand=hh, learned=nn, delta=nn - hh,
                                n_obs=int(len(z)))
            if not ok:
                continue
            rows.append(row)
            print(f'[{name} w{wi}] '
                  + '  '.join(f"{a.split('_')[0]} d={row[a]['delta']:+.3f}"
                              for a in ARMS)
                  + f'  [{time.time() - t0:.0f}s]', flush=True)

    # E-12e (2026-07-14): this used se = std/sqrt(n) over 60 WINDOWS and a 1.96
    # critical value. The windows are clustered in 5 records, so the iid SE is
    # not valid, and on G clusters the reference is t(G-1), not the normal.
    # Re-judged with CR1 + t(4): S0_uniform p 0.046, S1_crest p 0.083,
    # S2_slope p 0.173, S3_crest_only p 0.105 -- the E-10 FAIL STANDS (only the
    # uniform reference arm reaches significance; no pattern arm does).
    g = [r['record'] for r in rows]
    summ = {}
    for a in ARMS:
        d = np.array([r[a]['delta'] for r in rows])
        se_iid = float(d.std(ddof=1) / np.sqrt(len(d)))
        se, n_clu = cluster_se(d, g)
        tcrit = float(stats.t.ppf(0.975, n_clu - 1))
        summ[a] = dict(
            n=len(d), n_clusters=n_clu, delta=float(d.mean()),
            se=se, se_iid=se_iid, t=float(d.mean() / se),
            p=float(2 * stats.t.sf(abs(d.mean() / se), n_clu - 1)),
            ci_lo=float(d.mean() - tcrit * se),
            ci_hi=float(d.mean() + tcrit * se),
            hand=float(np.mean([r[a]['hand'] for r in rows])),
            learned=float(np.mean([r[a]['learned'] for r in rows])),
            n_obs=int(np.mean([r[a]['n_obs'] for r in rows])))

    res = dict(date='2026-07-13',
               prereg='docs/E10_PATTERN_prereg_2026-07-13.md',
               n_obs_budget=N_OBS, arms=ARMS, summary=summ, windows=rows,
               gate_G_E10_2=bool(summ['S1_crest']['ci_lo'] > 0
                                 and summ['S2_slope']['ci_lo'] > 0))
    json.dump(res, open(OUT, 'w'), indent=1)

    print(f'\n{"=" * 76}\n[E-10 real] {len(rows)} windows, {N_OBS} obs each, '
          f'pattern varied, count fixed')
    print(f'{"arm":16s} {"hand":>7} {"learned":>8} {"delta":>9} '
          f'{"95% CI":>20} {"t":>6}')
    for a in ARMS:
        s = summ[a]
        print(f'{a:16s} {s["hand"]:7.3f} {s["learned"]:8.3f} {s["delta"]:+9.4f}'
              f'  [{s["ci_lo"]:+.4f},{s["ci_hi"]:+.4f}] {s["t"]:6.2f}')
    print(f'\nG-E10-2 (real headline survives crest AND slope patterns): '
          f'{"PASS" if res["gate_G_E10_2"] else "FAIL"}')
    print(f'-> {OUT}')


if __name__ == '__main__':
    main()
