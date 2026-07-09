"""
d68 — multi-seed hardening of the posterior-UQ conclusion (d67 was 1 training run/variant).

Build the HOS eval windows ONCE, then train N_SEEDS vanilla and N_SEEDS physics-feature posteriors
(different init + data RNG) and evaluate each on the same windows. Report per-seed lead-calibration
(mean |±1σ−0.68| across lead) and nowcast accuracy, with mean ± SE and a paired vanilla-vs-physics
comparison. Confirms whether "vanilla forecast-trained NN ≥ physics-feature NN" is real or run-noise.

Run: python3 scripts/d68_multiseed_posterior.py [--steps 3000] [--seeds 4]
"""
import os, sys, json, time, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from d67_physics_posterior_uq import (PosteriorNet, gen_batch_fc, coverage, LEADS,
                                      DEVICE, SEAS, CAL_SEEDS, TEST_SEEDS, temp)
import d62_predictable_zone_test as Z
from scipy.stats import wilcoxon


def train_seeded(qdim, seed, steps, B=24, lr=3e-4):
    torch.manual_seed(seed); np.random.seed(seed)
    model = PosteriorNet(qdim=qdim).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for s in range(steps):
        obs, q, cov, qz, Hs = gen_batch_fc(B)
        mean, logv = model(obs, q, cov); logv = logv.clamp(-8, 4)
        nll = 0.5 * ((mean - qz) ** 2 / logv.exp() + logv)
        loss = (nll.mean(1) / (Hs ** 2 + 1e-6)).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    model.eval(); return model


def build_eval_windows():
    """Build HOS forecast windows once; cache obs tensor + per-lead (query, coverage, GT)."""
    Z.SIGMA = 0.02; Z.HORIZONS = LEADS
    gx = np.linspace(10, 250, 80); gy = np.linspace(-100, 100, 60)
    Xg, Yg = np.meshgrid(gx, gy); grid = np.column_stack([Xg.ravel(), Yg.ravel()])
    W = []
    for tag, Hs, Tp in SEAS:
        for seed in CAL_SEEDS + TEST_SEEDS:
            w = Z.build_window(Hs, Tp, seed, gx, gy, M=3)
            kx, ky, A, phi = Z.estimate(w['P'], w['z'], w['t_now'])
            Hs_n = float(w['gt_at'][0][1].std() * 4); t0obs = w['P'][:, 2].min()
            ob = torch.tensor(np.column_stack([w['P'][:, 0], w['P'][:, 1], w['P'][:, 2] - t0obs,
                                               w['z']])[None], dtype=torch.float32, device=DEVICE)
            leads = []
            for h in LEADS:
                st = int(round(h / 0.1))
                if st not in w['gt_at']:
                    continue
                t_star, GT = w['gt_at'][st]
                q = torch.tensor(np.column_stack([grid, np.full(len(grid), float(t_star - t0obs))])[None],
                                 dtype=torch.float32, device=DEVICE)
                cov = coverage(grid[:, 0], grid[:, 1], t_star, w['P'][:, 0], w['P'][:, 1], w['P'][:, 2], kx, ky, A)
                covt = torch.tensor(cov[None], dtype=torch.float32, device=DEVICE)
                leads.append(dict(h=h, q=q, cov=covt, gt=GT.ravel()))
            W.append(dict(seed=seed, Hs=Hs_n, ob=ob, leads=leads))
    return W


def eval_model(model, W):
    """Return (nowcast nRMSE on test, mean |±1σ−0.68| across lead on test) after cal-temperature."""
    out = {}
    for w in W:
        rec = []
        with torch.no_grad():
            for ld in w['leads']:
                mean, logv = model(w['ob'], ld['q'], ld['cov'] if model.qdim == 4 else None)
                m = mean[0].cpu().numpy(); sg = (0.5 * logv[0]).exp().cpu().numpy()
                rec.append(dict(h=ld['h'], err=np.abs(m - ld['gt']) / w['Hs'], sig=sg / w['Hs'],
                                rmse=float(np.sqrt(np.mean((m - ld['gt']) ** 2)) / w['Hs'])))
        out[w['seed']] = rec
    # calibrate temperature on CAL seeds (pooled), evaluate on TEST
    ce = np.concatenate([r['err'] for s in CAL_SEEDS for w in W if w['seed'] == s for r in out[s]])
    cs = np.concatenate([r['sig'] for s in CAL_SEEDS for w in W if w['seed'] == s for r in out[s]])
    sT = temp(ce, cs)
    devs, nrs = [], []
    for h in LEADS:
        es, ss = [], []
        for s in TEST_SEEDS:
            for w in W:
                if w['seed'] != s:
                    continue
                for r in out[s]:
                    if r['h'] == h:
                        es.append(r['err']); ss.append(r['sig'])
        if not es:
            continue
        e = np.concatenate(es); sg = np.concatenate(ss) * sT
        devs.append(abs(float((e < sg).mean()) - 0.683))
    for s in TEST_SEEDS:
        for w in W:
            if w['seed'] == s:
                nrs += [r['rmse'] for r in out[s] if r['h'] == 0.0]
    return float(np.mean(nrs)), float(np.mean(devs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=3000); ap.add_argument('--seeds', type=int, default=4)
    a = ap.parse_args(); t0 = time.time()
    print("building HOS eval windows once...", flush=True)
    W = build_eval_windows()
    res = {'vanilla': dict(dev=[], nrmse=[]), 'physics': dict(dev=[], nrmse=[])}
    for seed in range(a.seeds):
        for nm, qd in [('vanilla', 3), ('physics', 4)]:
            m = train_seeded(qd, seed, a.steps)
            nr, dev = eval_model(m, W)
            res[nm]['dev'].append(dev); res[nm]['nrmse'].append(nr)
            print(f"  seed {seed} {nm:8s}: lead-cal dev {dev:.3f}  nowcast nRMSE {nr:.3f}  ({time.time()-t0:.0f}s)", flush=True)

    summ = {}
    for nm in res:
        d = np.array(res[nm]['dev']); n = np.array(res[nm]['nrmse'])
        summ[nm] = dict(dev_mean=float(d.mean()), dev_se=float(d.std(ddof=1)/np.sqrt(len(d))),
                        nrmse_mean=float(n.mean()), nrmse_se=float(n.std(ddof=1)/np.sqrt(len(n))))
    # paired test (physics vs vanilla lead-cal dev)
    try:
        wstat = wilcoxon(res['physics']['dev'], res['vanilla']['dev'])
        pval = float(wstat.pvalue)
    except Exception:
        pval = None
    json.dump(dict(steps=a.steps, seeds=a.seeds, raw=res, summary=summ, paired_p=pval),
              open('results/direction3/multiseed_posterior.json', 'w'), indent=2)

    print("\n================ MULTI-SEED POSTERIOR UQ (n={} seeds) ================".format(a.seeds))
    for nm in ['vanilla', 'physics']:
        s = summ[nm]
        print(f"  {nm:8s}: lead-cal dev {s['dev_mean']:.3f} ± {s['dev_se']:.3f}   "
              f"nowcast nRMSE {s['nrmse_mean']:.3f} ± {s['nrmse_se']:.3f}")
    better = 'vanilla' if summ['vanilla']['dev_mean'] < summ['physics']['dev_mean'] else 'physics'
    print(f"  paired Wilcoxon p(physics vs vanilla dev) = {pval}")
    print(f"  => better lead-calibration: {better}; physics feature "
          f"{'does NOT help' if summ['physics']['dev_mean'] >= summ['vanilla']['dev_mean'] - 0.003 else 'helps'}")
    print(f"saved results/direction3/multiseed_posterior.json   ({time.time()-t0:.0f}s)")


if __name__ == '__main__':
    main()
