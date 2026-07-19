"""E-B2K PRIMARY ARENA -- 3-arm loss-swap on the frozen E-15 HOS grid.

Pre-registration: tmp/E60K_PREREG_v3_FROZEN_2026-07-19.md (FROZEN before this ran).
  arena   = E-15 grid, ALL 32 seas (zero selection freedom), 5 draws per sea,
            draw rng stream IDENTICAL to E-15 (first 5 of its 8) so the arena is
            the already-published one, not a new one.
  arms    = T (task loss) / L (in-window held-out z-residual, CV-type) /
            N (true ML-II marginal likelihood), 3 training seeds each = 9 models.
  cluster = sea (k=32).  failure window = hand < 0.5 (07-12 frozen definition).
  All 9 models see the SAME observations and the SAME queries within a draw --
  the draw is made once and shared, which is what makes this a fair duel.

Correctness gate (runs first, aborts on failure): the multi-model draw must
reproduce e15_hos_multidraw.one_draw bit-for-bit for the same sea/rng/model, so
that adding arms did not perturb the frozen sampling path.
"""
import json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')
sys.path.insert(0, 'tmp')

from e15_hos_multidraw import (           # noqa: E402
    SEAS, build_sea, one_draw, lam_p, T_MANUAL_50, N_OBS, LEADS,
    DEPTH, NOISE, R_MIN, FOV_H, z_at, mmse_gpu, DEVICE)
from d3_prior_net_eval import get_net, net_gamma_mmse    # noqa: E402
from d3_l1_s2head_train import SmultHead                 # noqa: E402

OUT = 'results/direction3/e60k_hos_arena.json'
ARMS = {f'{a}_s{s}': f'results/direction3/b2deep_{a}_s{s}.pt'
        for a in ('T', 'L', 'N') for s in (0, 1, 2)}
DRAWS = 5


def load_model(ck):
    net = get_net(ck)
    head = SmultHead().to(DEVICE)
    head.load_state_dict(torch.load(ck, map_location=DEVICE)['head'])
    head.eval()
    return net, head


def draw_multi(sea, rng, models):
    """one_draw's sampling body VERBATIM, then every model on that one draw.

    rng consumption order is unchanged from one_draw (models consume no rng:
    net_gamma_mmse is called with seed=0), so the observation draw is identical
    to E-15's. Verified by the gate in main()."""
    P, zv, rr = [], [], []
    for sc in sea['scans']:                        # 1) return-power lottery
        pw = sc['amp'] * rng.exponential(1.0, len(sc['amp']))
        keep = pw > T_MANUAL_50
        P.append(np.column_stack([sc['xy'][keep],
                                  np.full(int(keep.sum()), sc['trel'])]))
        zv.append(sc['z'][keep])
        rr.append(sc['r'][keep])
    P, zv, rr = np.vstack(P), np.concatenate(zv), np.concatenate(rr)
    zv = zv + rng.normal(0, NOISE, len(zv))        # 2) measurement noise

    edges = np.linspace(R_MIN, rr.max() + 1e-6, 13)   # 3) range-stratified 600
    b = np.clip(np.digitize(rr, edges) - 1, 0, 11)
    take = []
    for j in range(12):
        idx = np.where(b == j)[0]
        k = min(len(idx), max(-(-N_OBS // 12), 1))
        if k:
            take.append(rng.choice(idx, k, replace=False))
    sel = np.concatenate(take)
    Ps, zs = P[sel], zv[sel]

    rf = np.sqrt(rng.uniform((15.0 / 200.0) ** 2, 1.0, 800)) * 200.0  # 4) queries
    af = rng.uniform(-np.deg2rad(FOV_H), np.deg2rad(FOV_H), 800)
    qx, qy = rf * np.cos(af), rf * np.sin(af)
    Q, gt = [], []
    for li, l_ in enumerate(LEADS):
        gt.append(z_at(sea['leads'][li], qx, qy))
        Q.append(np.column_stack([qx, qy, np.full(800, l_)]))
    Q, gt = np.vstack(Q), np.concatenate(gt)

    kp = sea['kp']
    with torch.no_grad():
        ph = mmse_gpu(torch.tensor(Ps, device=DEVICE, dtype=torch.float32),
                      torch.tensor(zs, device=DEVICE, dtype=torch.float32),
                      DEPTH,
                      torch.tensor(Q, device=DEVICE, dtype=torch.float32),
                      NOISE, kp_hint=kp).cpu().numpy()
    out = {'hand': float(np.corrcoef(ph, gt)[0, 1]), 'n_obs': int(len(zs))}
    for name, (net, head) in models.items():
        pn = np.asarray(net_gamma_mmse(net, Ps, zs, DEPTH, Q, NOISE, seed=0,
                                       kp_hint=kp, use_em=False,
                                       smult_head=head))
        out[name] = float(np.corrcoef(pn, gt)[0, 1])
    return out


def main():
    missing = [k for k, v in ARMS.items() if not os.path.exists(v)]
    if missing:
        print(f'ABORT -- missing checkpoints: {missing}')
        return 1
    models = {k: load_model(v) for k, v in ARMS.items()}
    print(f'loaded {len(models)} models: {sorted(models)}', flush=True)

    # ---- correctness gate --------------------------------------------------
    print('gate: draw_multi must reproduce one_draw bit-for-bit '
          '(Tp 8, Hs 1.2, rng 7, model T_s0)...', flush=True)
    sea0 = build_sea(8.0, 1.2, 0)
    a = draw_multi(sea0, np.random.default_rng(7), {'T_s0': models['T_s0']})
    h_ref, n_ref, nobs_ref = one_draw(sea0, np.random.default_rng(7),
                                      *models['T_s0'])
    dh, dn = abs(a['hand'] - h_ref), abs(a['T_s0'] - n_ref)
    print(f"  multi  hand {a['hand']:+.10f}  net {a['T_s0']:+.10f}  "
          f"n_obs {a['n_obs']}")
    print(f"  ref    hand {h_ref:+.10f}  net {n_ref:+.10f}  n_obs {nobs_ref}")
    print(f"  d_hand {dh:.2e}  d_net {dn:.2e}")
    if dh > 1e-12 or dn > 1e-12 or a['n_obs'] != nobs_ref:
        print('  *** GATE FAILED -- adding arms perturbed the draw. Aborting.')
        return 1
    print('  *** GATE PASSED\n', flush=True)

    rows = []
    if os.path.exists(OUT):                       # resumable
        rows = json.load(open(OUT)).get('rows', [])
    done = {(r['sea'], r['draw']) for r in rows}

    for si, (Tp, Hs) in enumerate(SEAS):
        tag = f'Tp{Tp:g}_Hs{Hs:g}'
        if all((tag, r) in done for r in range(DRAWS)):
            continue
        t0 = time.time()
        sea = sea0 if (Tp == 8.0 and Hs == 1.2) else build_sea(Tp, Hs, 0)
        lam = lam_p(Tp)
        for r in range(DRAWS):
            if (tag, r) in done:
                continue
            # SAME rng stream as E-15 (900000 + 7919*si + 31*r)
            res = draw_multi(sea, np.random.default_rng(900_000 + 7919 * si
                                                        + 31 * r), models)
            res.update(sea=tag, Tp=float(Tp), Hs=float(Hs), lam_p=lam,
                       steepness=float(Hs) / lam, draw=r)
            rows.append(res)
        json.dump({'config': dict(seas=len(SEAS), draws=DRAWS, arms=sorted(ARMS),
                                  prereg='tmp/E60K_PREREG_v3_FROZEN_2026-07-19.md'),
                   'rows': rows}, open(OUT, 'w'), indent=1)
        sub = [x for x in rows if x['sea'] == tag]
        tl = np.mean([np.mean([x[f'T_s{s}'] - x[f'L_s{s}'] for s in (0, 1, 2)])
                      for x in sub])
        tn = np.mean([np.mean([x[f'T_s{s}'] - x[f'N_s{s}'] for s in (0, 1, 2)])
                      for x in sub])
        hh = np.mean([x['hand'] for x in sub])
        print(f'[{si + 1:2d}/32 {tag:12s}] hand {hh:+.3f}  T-L {tl:+.4f}  '
              f'T-N {tn:+.4f}  [{time.time() - t0:.0f}s]', flush=True)

    print(f'\nwrote {OUT}  ({len(rows)} rows)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
