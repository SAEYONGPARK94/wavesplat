"""fable INDEPENDENT eval of prior_net_compdiverse vs prior_net_v3 (dual-verification
of Opus's tmp/betternet_eval.py — deliberately NOT reusing it, and deliberately using
DIFFERENT held-out data than its default n=16/last-3-shards slice, to check the PASS/FAIL
verdict is not an artifact of one particular episode/shard subset).

Design choices that differ from betternet_eval.py:
  - SPARSE: betternet_eval.py takes sorted(glob 15comp))[:16] = seed0-3 x
    {fixedT,fixedT_dir180,slope,uniform} (4 independent underlying sea draws x 4 geometry
    replicates -- fixedT/slope/uniform for a given seed share the IDENTICAL wave_components,
    confirmed by direct h5 inspection). This script instead uses the 9 held-out 15comp
    episodes NOT in that slice: seed4 x{fixedT,fixedT_dir180,slope,uniform} (4 files, 1 new
    independent seed) + seed{5,6,7,8,9}_fixedT_dir180 (5 files, 5 more independent seeds).
    -> 6 independent underlying sea realizations (vs Opus's 4), same n_components=15/BF4.
  - BROADBAND: betternet_eval.py uses the LAST 3 of 4 bwtask shards (s001,s002,s003) with
    per=40. This script uses shard s000 (the one Opus's slice never touches) with per=80 for
    more n. bwtask predates prior_net_v3 (bwtask_s000 mtime 2026-07-08, v3 mtime 2026-07-05)
    so it is genuinely held out for v3 regardless of which shard; using a disjoint shard from
    Opus's just removes any possibility of the two evals sharing exact scenes.
  - Reports per-seed medians (not just pooled) to expose pseudo-replication risk (Opus's own
    n=16 pools 4 geometry-replicates per seed as if independent).

Usage: python3 tmp/BETTERNET_fable_indep_eval.py --ckpt results/direction3/prior_net_compdiverse.pt
Out: results/direction3/betternet_fable_indep_eval.json
"""
import os, sys, glob, json, time, argparse
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, 'scripts'); sys.path.insert(0, 'tmp')
from scripts.d1_crb import DEVICE, DT, load_ep, collect_obs, predict
from d3_prior_net_eval import get_net, net_gamma_mmse

SIGMA = 0.02
V3 = 'results/direction3/prior_net_v3.pt'

HELD_OUT_SPARSE = sorted(
    glob.glob('data/episode_BF4_15comp_seed4_*.h5') +
    glob.glob('data/episode_BF4_15comp_seed[5-9]_*.h5')
)


def dom_kp(th):
    w = th[:, 0].abs(); i = int(torch.argmax(w))
    return float(torch.hypot(th[i, 1], th[i, 2]))


def corr(a, b):
    a, b = np.asarray(a), np.asarray(b)
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 3 else np.nan


def seed_of(path):
    b = os.path.basename(path)
    return b.split('seed')[1].split('_')[0]


def eval_sparse_heldout(net_v3, net_c, paths, R=6, nobs=250, obs_seed=1):
    """Same synthetic-obs recipe as gap_ladder/betternet_eval (predict(theta)+noise,
    beamform-kp net-MMSE, non-oracle), but on the held-out episode list and a DIFFERENT
    obs-subsample/phase RNG seed (1, not 0) so this is not a bit-identical rerun."""
    rows = []
    for p in paths:
        ep = load_ep(p)
        t_now = float(ep['ts'][len(ep['ts']) // 2])
        th0 = ep['theta']; N = th0.shape[0]; depth = ep['depth']
        x, y, t = collect_obs(ep, t_now)
        g = torch.Generator(device=DEVICE).manual_seed(obs_seed)
        if x.numel() > nobs:
            idx = torch.randperm(x.numel(), device=DEVICE, generator=g)[:nobs]
            x, y, t = x[idx], y[idx], t[idx]
        gx, gy = ep['gx'], ep['gy']
        stride = max(1, (gx.numel() * gy.numel()) // 400)
        Gx, Gy = torch.meshgrid(gx, gy, indexing='xy')
        qx, qy = Gx.reshape(-1)[::stride], Gy.reshape(-1)[::stride]
        cv, cc = [], []
        for _ in range(R):
            thr = th0.clone()
            thr[:, 3] = torch.rand(N, dtype=DT, device=DEVICE, generator=g) * 2 * np.pi
            qt = torch.full_like(qx, float(t_now))
            truth = predict(thr, qx, qy, qt, depth).cpu().numpy()
            s = predict(thr, x, y, t, depth)
            z = (s + SIGMA * torch.randn(s.shape, dtype=DT, device=DEVICE, generator=g)).cpu().numpy()
            P = torch.stack([x, y, t], 1).cpu().numpy()
            Q = torch.stack([qx, qy, qt], 1).cpu().numpy()
            cv.append(corr(net_gamma_mmse(net_v3, P, z, depth, Q, SIGMA, seed=obs_seed,
                                          kp_hint=None, use_em=False, smult_fixed=3.25), truth))
            cc.append(corr(net_gamma_mmse(net_c, P, z, depth, Q, SIGMA, seed=obs_seed,
                                          kp_hint=None, use_em=False, smult_fixed=3.25), truth))
        rows.append(dict(ep=os.path.basename(p), seed=seed_of(p),
                         v3=float(np.nanmean(cv)), cand=float(np.nanmean(cc))))
    return rows


def eval_broadband_heldout(net_v3, net_c, shard='tmp/prior_train/bwtask_s000.npz', per=80):
    d = np.load(shard, allow_pickle=True)
    rv, rc = [], []
    n = min(per, len(d['P']))
    for i in range(n):
        P, z, Q, gt = d['P'][i], d['z'][i], d['Q'][i], d['gt_q'][i]
        kp = d['meta'][i][0]
        pv = net_gamma_mmse(net_v3, P, z, 50.0, Q, SIGMA, seed=0, kp_hint=kp, use_em=False, smult_fixed=3.25)
        pc = net_gamma_mmse(net_c, P, z, 50.0, Q, SIGMA, seed=0, kp_hint=kp, use_em=False, smult_fixed=3.25)
        rv.append(corr(pv, gt)); rc.append(corr(pc, gt))
    return np.array(rv), np.array(rc)


def ci(d, B=4000, seed=0):
    d = d[np.isfinite(d)]
    rng = np.random.default_rng(seed)
    bs = [np.mean(rng.choice(d, len(d))) for _ in range(B)]
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), int((d > 0).sum()), len(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='results/direction3/prior_net_compdiverse.pt')
    a = ap.parse_args()
    t0 = time.time()
    net_v3 = get_net(V3)
    net_c = get_net(a.ckpt)

    print(f"=== fable independent eval ({a.ckpt}) vs prior_net_v3 ===")
    print(f"held-out sparse episodes (n={len(HELD_OUT_SPARSE)}): "
          f"{[os.path.basename(p) for p in HELD_OUT_SPARSE]}")

    rows = eval_sparse_heldout(net_v3, net_c, HELD_OUT_SPARSE)
    for r in rows:
        print(f"  {r['ep']:38s} seed{r['seed']}  v3 {r['v3']:.3f}  cand {r['cand']:.3f}  "
              f"d {r['cand']-r['v3']:+.3f}")
    sv = np.array([r['v3'] for r in rows]); sc = np.array([r['cand'] for r in rows])
    dm, lo, hi, nb, nn = ci(sc - sv)
    print(f"\nSPARSE (held-out, pooled n={nn}, PSEUDO-REPLICATED across geometry variants):")
    print(f"  v3 median {np.nanmedian(sv):.3f}  cand median {np.nanmedian(sc):.3f}")
    print(f"  Delta(cand-v3) {dm:+.3f} CI[{lo:+.3f},{hi:+.3f}] win {nb}/{nn}")
    # per-seed aggregation (6 independent sea draws) -- the honest n for CI purposes
    seeds = sorted(set(r['seed'] for r in rows))
    per_seed_d = []
    for sd in seeds:
        vs = [r['cand'] - r['v3'] for r in rows if r['seed'] == sd]
        per_seed_d.append(float(np.mean(vs)))
    per_seed_d = np.array(per_seed_d)
    dm2, lo2, hi2, nb2, nn2 = ci(per_seed_d)
    print(f"\nSPARSE per-INDEPENDENT-SEED (n={len(seeds)} unique sea draws, honest n):")
    print(f"  seed deltas: {dict(zip(seeds, [round(float(x),3) for x in per_seed_d]))}")
    print(f"  Delta(cand-v3) {dm2:+.3f} CI[{lo2:+.3f},{hi2:+.3f}] win {nb2}/{nn2}")

    bv, bc = eval_broadband_heldout(net_v3, net_c)
    dmb, lob, hib, nbb, nnb = ci(bc - bv)
    print(f"\nBROADBAND (held-out shard s000, n={len(bv)}):")
    print(f"  v3 median {np.nanmedian(bv):.3f}  cand median {np.nanmedian(bc):.3f}")
    print(f"  Delta(cand-v3) {dmb:+.3f} CI[{lob:+.3f},{hib:+.3f}] win {nbb}/{nnb}")

    pass_pooled = dm >= 0.05 and lo > 0 and dmb >= -0.02
    pass_perseed = dm2 >= 0.05 and lo2 > 0 and dmb >= -0.02
    print(f"\n=== VERDICT (pooled, Opus's bar) === {'PASS' if pass_pooled else 'FAIL'}")
    print(f"=== VERDICT (per-seed honest-n, same bar) === {'PASS' if pass_perseed else 'FAIL'}")
    print(f"[{time.time()-t0:.0f}s]")

    json.dump(dict(sparse_rows=rows, sparse_pooled=dict(v3=float(np.nanmedian(sv)), cand=float(np.nanmedian(sc)),
              delta=dm, ci=[lo, hi], win=f"{nb}/{nn}"),
              sparse_perseed=dict(seed_deltas=dict(zip(seeds, [float(x) for x in per_seed_d])),
              delta=dm2, ci=[lo2, hi2], win=f"{nb2}/{nn2}"),
              broadband=dict(v3=float(np.nanmedian(bv)), cand=float(np.nanmedian(bc)), delta=dmb,
              ci=[lob, hib], win=f"{nbb}/{nnb}"),
              pass_pooled=pass_pooled, pass_perseed=pass_perseed, ckpt=a.ckpt),
              open('results/direction3/betternet_fable_indep_eval.json', 'w'), indent=1, default=float)
    print("saved -> results/direction3/betternet_fable_indep_eval.json")


if __name__ == '__main__':
    main()
