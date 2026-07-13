"""E-3: re-judge the clean-linear stage of the misspecification gradient on an
adequately rendered mesh.

Prereg: docs/E2_MESH_AUDIT_prereg_2026-07-13.md (gates frozen before numbers).

The published stage value -- d(learned - hand) = -0.041 over 35 episodes, the
left anchor of fig_gradient -- was measured on episodes whose raycast mesh
cannot represent their own short components.  Here the same 35 seas are scored
twice with the arms held fixed: once as stored (mesh 40x30) and once re-rendered
(mesh 180x150).  Waves and truth are bit-identical across the pair; only the
observation operator differs.

Arms are imported verbatim from the frozen a2 evaluator, so nothing about the
estimators moves.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

import h5py                                                     # noqa: E402
from d3_a2_dir180_eval import (LEADS, load_episode, load_net_head,  # noqa: E402
                               score_episode)

COARSE = 'data'
FINE = 'tmp/finemesh_clean'
LIST = 'tmp/clean_stage_episodes.txt'
OUT = 'results/direction3/e3_clean_rejudge.json'


def operator_error(path):
    """std of (LiDAR return z - analytic truth at the same x,y,t).

    The E-1 statistic, per episode: how much of what the sensor reports is not
    the wave.  Truth is evaluated from the stored components, never from a grid.
    """
    from wavesim.gt_generator import evaluate_field
    from wavesim.wave_generator import WaveComponent
    with h5py.File(path, 'r') as h:
        c = h['config/wave_components'][:]
        pts = h['frames/lidar_points'][:]
        msk = h['frames/lidar_valid_mask'][:]
        ts = h['frames/timestamps'][:]
    comps = [WaveComponent(A=r[0], omega=r[1], kx=r[2], ky=r[3], phi=r[4],
                           Q=r[5]) for r in c]
    errs = []
    for j in range(0, len(ts), 10):               # every 10th frame is plenty
        m = msk[j]
        if not m.any():
            continue
        p = pts[j][m]
        z_true, _, _, _ = evaluate_field(comps, p[:, 0], p[:, 1], float(ts[j]))
        errs.append(p[:, 2] - z_true)
    e = np.concatenate(errs)
    return float(e.std())


def judge(tag, root, names, net, head, n_obs=2000):
    recs = []
    for i, n in enumerate(names):
        path = os.path.join(root, n)
        rng = np.random.default_rng(11 + i)       # paired: same stream per sea
        ep = load_episode(path, rng, n_obs=n_obs)
        s = score_episode(ep, net, head)
        d = float(np.mean(s['corr_learned']) - np.mean(s['corr_hand']))
        recs.append(dict(episode=n, sea=ep['sea'], n_comp=ep['ncomp'],
                         corr_hand=s['corr_hand'], corr_learned=s['corr_learned'],
                         delta=d, op_err=operator_error(path)))
        print(f'  [{tag}] {i + 1}/{len(names)} {n} d={d:+.3f}', flush=True)
    return recs


def summarize(recs):
    d = np.array([r['delta'] for r in recs])
    by_sea = {}
    for sea in sorted({r['sea'] for r in recs}):
        m = [r for r in recs if r['sea'] == sea]
        by_sea[sea] = dict(
            n=len(m),
            delta=float(np.mean([r['delta'] for r in m])),
            op_err=float(np.mean([r['op_err'] for r in m])),
            hand=float(np.mean([np.mean(r['corr_hand']) for r in m])),
            learned=float(np.mean([np.mean(r['corr_learned']) for r in m])))
    return dict(
        n=len(recs), delta_mean=float(d.mean()),
        delta_se=float(d.std(ddof=1) / np.sqrt(len(d))),
        learned_wins=int((d > 0).sum()),
        hand_mean=float(np.mean([np.mean(r['corr_hand']) for r in recs])),
        learned_mean=float(np.mean([np.mean(r['corr_learned']) for r in recs])),
        op_err_mean=float(np.mean([r['op_err'] for r in recs])),
        by_sea=by_sea)


def main():
    names = [l.strip() for l in open(LIST) if l.strip()]
    missing = [n for n in names if not os.path.exists(os.path.join(FINE, n))]
    if missing:
        raise SystemExit(f'fine-mesh regeneration incomplete: {len(missing)} '
                         f'missing, e.g. {missing[:2]}')

    net, head = load_net_head()
    t0 = time.time()
    print(f'[E-3] judging {len(names)} episodes, coarse vs fine mesh', flush=True)
    fine = judge('fine', FINE, names, net, head)
    coarse = judge('coarse', COARSE, names, net, head)
    # E-9 (PI directive): validation is quoted at the originally assumed sparse
    # budget -- 50 returns/frame x 12 frames = 600 obs/window.  The 2000-obs arm
    # is the optimistic-radiometry case.
    sparse = judge('fine_sparse', FINE, names, net, head, n_obs=600)

    sf, sc, ss = summarize(fine), summarize(coarse), summarize(sparse)
    paired = np.array([f['delta'] for f in fine]) - \
             np.array([c['delta'] for c in coarse])

    res = dict(
        date='2026-07-13', prereg='docs/E2_MESH_AUDIT_prereg_2026-07-13.md',
        leads=LEADS, n_episodes=len(names),
        published_coarse_delta=-0.041,
        coarse=sc, fine=sf, fine_sparse=ss,
        shift=dict(mean=float(paired.mean()),
                   se=float(paired.std(ddof=1) / np.sqrt(len(paired)))),
        gates=dict(
            G_E3_1_narrative_holds=bool(sf['delta_mean'] < 0),
            G_E3_2_number_stable=bool(
                abs(sf['delta_mean'] - sc['delta_mean']) <= 0.03),
            G_E3_3_intervention=bool(sf['op_err_mean'] < sc['op_err_mean'])),
        episodes=dict(coarse=coarse, fine=fine, fine_sparse=sparse))
    json.dump(res, open(OUT, 'w'), indent=1)

    print(f'\n{"=" * 62}\n[E-3] {len(names)} episodes, {time.time() - t0:.0f}s')
    print(f'  coarse mesh : d = {sc["delta_mean"]:+.4f} '
          f'(hand {sc["hand_mean"]:.3f} / learned {sc["learned_mean"]:.3f}) '
          f'op_err {sc["op_err_mean"]:.4f} m')
    print(f'  fine   mesh : d = {sf["delta_mean"]:+.4f} '
          f'(hand {sf["hand_mean"]:.3f} / learned {sf["learned_mean"]:.3f}) '
          f'op_err {sf["op_err_mean"]:.4f} m')
    print(f'  paired shift: {paired.mean():+.4f} +- {res["shift"]["se"]:.4f}')
    print(f'  ★fine, 600 obs (SPARSE, the regime the thesis assumes):'
          f' d = {ss["delta_mean"]:+.4f} +- {ss["delta_se"]:.4f}'
          f'  (learned wins {ss["learned_wins"]}/{ss["n"]})')
    for g, v in res['gates'].items():
        print(f'  {"PASS" if v else "FAIL"}  {g}')
    print(f'-> {OUT}')


if __name__ == '__main__':
    main()
