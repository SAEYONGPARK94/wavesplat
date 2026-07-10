"""N1: a critic that ranks physics estimators, trained where the truth is free.

E1 moved the bottleneck. Give the gate a family of twenty certified estimators
-- four bands crossed with five levels of trust in the learned prior -- and a
better one is genuinely in there: the family's oracle beats the hand rule by
about +0.021, the same size as everything else we established today. But the
gate harvests only a third of it, and three no-learning fixes (single split,
4-fold random, 4-fold spatial block) all leave +0.016 on the table. Adding
folds made it WORSE, so the held-out observation residual is not a noisy
criterion. It is a biased one.

Bias is what a learned critic is for.

    features (observations only)  ->  MLP  ->  predicted FIELD error
    choose the candidate with the smallest prediction

The truth is needed to train it, and synthetic seas hand it over for free --
the axis G1 showed is the binding one (labelled seas), and synthesis has an
unlimited supply. The truth is never needed at run time.

WHY THIS SURVIVES TODAY'S CONSTRAINTS, each of which killed an earlier attempt:

  Gaussian optimality       the critic emits no field. It ranks.
  13 real seas              it trains and validates on synthetic ones.
  learned prior's OOD harm  every candidate is a certified MMSE solve.
  destroyed calibration     it never touches the posterior variance.
  adversarial fragility     its worst case is picking a bad band, and that is
                            bounded by the family's oracle gap and by B1.

FEATURES, all computable at run time from the observations alone:
   0  held-out residual / std(z)      the hand rule's entire criterion
   1  training residual / std(z)      how much of the fit is memorisation
   2  alpha                           how much of the learned prior is trusted
   3  band low edge  / kp
   4  band high edge / kp
   5  log10(bank atoms)               model size
   6  beta = min(1, std(z~)/std(p))   the certificate's shrinkage
   7  std(p) / std(z~)                amplitude the fit wants
   8  prior mass below 0.35 kp        is there sub-band energy?
   9  prior mass above 1.6 kp
  10  prior peakiness (max/mean)      how concentrated the beamform prior is

LABEL: the candidate's field nRMSE, CENTERED within its sea. Centering removes
how hard the sea is, which the critic cannot know and must not learn; what is
left is exactly the ranking.

PRE-REGISTERED, on the same 30 real record-window cells:
  1. beat the hand rule by >= +0.010 (paired)
  2. cut the selection overfit from +0.016 to <= +0.008
  3. lose <= 0.005 on the calm records (YS17, AA15b)
  4. NEGATIVE CONTROL: a critic trained without features 0-1 must collapse.
     If it does not, it is not reading the residual and something is leaking.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from d3_gpu_mmse import mmse_gpu, build_bank, beamform_prior, design
from d3_l1_learned_prior import RadialGain, sea, obs, WIDE, CANDS, dm, S0
from d3_l3_trust_and_coverage import ood_sea
from d3_g3_adversarial_sea import geometry, DEV, DEPTH, KP, SIGMA
from d3_a2_real_revalidation import draw, HELD
import d3_static_field as SF

OUT = 'results/direction3/n1_critic.json'
CKPT = 'results/direction3/n1_critic.pt'
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
NFEAT = 11
RECS = ['AA14', 'AA15a', 'AA15b', 'LJ18g', 'YS17']
WINDOWS = (0, 1, 2, 3, 4, 5)
RSEEDS = (11, 12, 13)
CALM = ('YS17', 'AA15b')


class Critic(nn.Module):
    def __init__(self, nf=NFEAT, h=64):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(nf, h), nn.SiLU(), nn.Linear(h, h),
                               nn.SiLU(), nn.Linear(h, 1))

    def forward(self, x):
        return self.f(x).squeeze(-1)


def gains_for(net, kp, depth):
    out = {}
    for band in CANDS:
        kxb, kyb, om = build_bank(kp, band, kp / 10.0, depth, DEV,
                                  torch.float32)
        kmag = torch.hypot(kxb, kyb)
        with torch.no_grad():
            g = torch.exp(net.f(torch.log(kmag / kp).unsqueeze(1))
                          .squeeze(1).clamp(-6.0, 6.0))
        for a in ALPHAS:
            out[(band, a)] = (None if a == 0.0 else g ** a, kmag, om)
    return out


def features(P, z, tr, he, kp, depth, band, gain, alpha, kmag, sig0):
    """Everything here is available at run time. The truth appears nowhere."""
    zc = z - z.mean()
    sz = float(zc.std()) + 1e-9
    with torch.no_grad():
        pv = mmse_gpu(P[tr], z[tr], depth, P[he], sigma0=sig0, kp_hint=kp,
                      band=band, prior_gain=gain)
        r_he = float(torch.sqrt(((pv - z[he]) ** 2).mean())) / sz
        pt = mmse_gpu(P[tr], z[tr], depth, P[tr], sigma0=sig0, kp_hint=kp,
                      band=band, prior_gain=gain)
        r_tr = float(torch.sqrt(((pt - z[tr]) ** 2).mean())) / sz

        kxb, kyb, om = build_bank(kp, band, kp / 10.0, depth, DEV,
                                  torch.float32)
        pw = beamform_prior(P, z, kxb, kyb, om)
        if gain is not None:
            pw = pw * gain
        pw = pw / pw.sum()
        lo = float(pw[kmag < 0.35 * kp].sum())
        hi = float(pw[kmag > 1.6 * kp].sum())
        peak = float(pw.max() * len(pw))

        pr = mmse_gpu(P, z, depth, P, sigma0=sig0, kp_hint=kp, band=band,
                      prior_gain=gain)
        spr = float((pr - pr.mean()).std()) + 1e-9
        beta = min(1.0, sz / spr)
    return [r_he, r_tr, alpha, band[0], band[1], np.log10(len(kxb)),
            beta, spr / sz, lo, hi, peak]


def field_ratio(P, z, Pq, gt, kp, depth, band, gain, sig0):
    with torch.no_grad():
        pr = mmse_gpu(P, z, depth, Pq, sigma0=sig0, kp_hint=kp, band=band,
                      prior_gain=gain)
        pr = pr - pr.mean()
        return float(torch.sqrt(((pr - gt) ** 2).mean()) / gt.std())


def synth_sea(rng):
    r = rng.random()
    if r < 0.35:
        return sea(rng, 'swell')
    if r < 0.7:
        return sea(rng, 'inband')
    if r < 0.85:
        return ood_sea(rng, 1.5, 2.4)
    return ood_sea(rng, 0.2, 2.4)


def build_dataset(net, n_sea, gen, seed0=0):
    P, Pq = geometry()
    gains = gains_for(net, KP, DEPTH)
    X, Y, S = [], [], []
    for i in range(n_sea):
        rng = np.random.default_rng(seed0 + i)
        s = synth_sea(rng)
        z, gt = obs(P, Pq, s, gen)
        n = len(z)
        idx = torch.randperm(n, device=DEV, generator=gen)
        tr, he = idx[:int((1 - HELD) * n)], idx[int((1 - HELD) * n):]
        sig0 = S0
        for (band, a), (g, kmag, om) in gains.items():
            X.append(features(P, z, tr, he, KP, DEPTH, band, g, a, kmag, sig0))
            Y.append(field_ratio(P, z, Pq, gt, KP, DEPTH, band, g, sig0))
            S.append(i)
        if (i + 1) % 25 == 0:
            print(f"  sea {i+1}/{n_sea}", flush=True)
    X, Y, S = np.array(X), np.array(Y), np.array(S)
    for i in np.unique(S):                        # centre within each sea
        Y[S == i] -= Y[S == i].mean()
    return X, Y, S


def train(X, Y, S, mask_resid=False, steps=3000, seed=0, rank_loss=True,
          drop_alpha=True):
    """MSE on the centered label does NOT induce a ranking.

    The first version regressed the centered field error and reached
    argmin-correct = 12% on held-out synthetic seas (chance is 5% with twenty
    candidates), while the no-residual control matched it at 12% with a LOWER
    validation MSE. Reading those two together: the critic was predicting each
    candidate's average behaviour across the training distribution -- alpha and
    the bank size explain most of the centered label variance there -- and never
    looked at the sea. On real records it then chose alpha = 0.82, exactly the
    heavily-learned prior that A showed is harmful.

    Two fixes, both aimed at forcing a per-sea judgement:
      rank_loss    a pairwise margin loss on candidates WITHIN a sea. Ordering
                   is what the selector needs; MSE is a detour.
      drop_alpha   alpha (and the band edges) let the critic answer without
                   looking at the observations. Remove that shortcut.
    """
    Xt = X.copy()
    if drop_alpha:
        Xt[:, 2:5] = 0.0                          # alpha, band lo, band hi
    if mask_resid:
        Xt[:, 0:2] = 0.0                          # the negative control
    mu, sd = Xt.mean(0), Xt.std(0) + 1e-9
    Xn = (Xt - mu) / sd
    torch.manual_seed(seed)
    net = Critic().to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    xt = torch.tensor(Xn, device=DEV, dtype=torch.float32)
    yt = torch.tensor(Y, device=DEV, dtype=torch.float32)
    seas = np.unique(S)
    tr_s = set(seas[: int(0.8 * len(seas))])
    m = torch.tensor(np.isin(S, list(tr_s)), device=DEV)
    tr_seas = [i for i in seas if i in tr_s]
    sea_idx = {i: torch.tensor(np.where(S == i)[0], device=DEV)
               for i in tr_seas}
    rng = np.random.default_rng(seed)
    for st in range(steps):
        if rank_loss:
            # sample seas; within each, all pairs get a margin proportional to
            # their true gap. The critic never sees an absolute difficulty.
            batch = rng.choice(tr_seas, size=min(16, len(tr_seas)),
                               replace=False)
            loss = 0.0
            for i in batch:
                k = sea_idx[i]
                p, y = net(xt[k]), yt[k]
                dp = p[:, None] - p[None, :]
                dy = y[:, None] - y[None, :]
                loss = loss + torch.relu(-(dp * torch.sign(dy))
                                         + dy.abs()).mean()
            loss = loss / len(batch)
        else:
            p = net(xt[m])
            loss = ((p - yt[m]) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sch.step()
    with torch.no_grad():
        pv = net(xt[~m])
        vl = float(((pv - yt[~m]) ** 2).mean()) if not rank_loss else float('nan')
        # Two measures. argmin-correct is what the first gate used, and it is
        # the wrong one: with twenty candidates several are near-ties, so
        # missing the exact winner costs almost nothing. REGRET -- how much
        # field error the choice costs against the sea's own best candidate --
        # is what the selector is for, and it is directly comparable to the
        # residual rule (argmin of feature 0), which is the baseline we never
        # thought to measure on synthetic seas.
        ok, reg, reg_res = 0, [], []
        hs = S[~m.cpu().numpy()]
        yy = Y[~m.cpu().numpy()]
        pp = pv.cpu().numpy()
        r0 = Xt[~m.cpu().numpy(), 0]              # held-out residual feature
        for i in np.unique(hs):
            k = hs == i
            best = yy[k].min()
            ok += int(np.argmin(pp[k]) == np.argmin(yy[k]))
            reg.append(yy[k][np.argmin(pp[k])] - best)
            reg_res.append(yy[k][np.argmin(r0[k])] - best)
        acc = ok / len(np.unique(hs))
    return net, mu, sd, vl, acc, float(np.mean(reg)), float(np.mean(reg_res))


def pick(net, mu, sd, feats, mask_resid=False, drop_alpha=True):
    F = np.array(feats)
    if drop_alpha:
        F[:, 2:5] = 0.0
    if mask_resid:
        F[:, 0:2] = 0.0
    with torch.no_grad():
        p = net(torch.tensor((F - mu) / sd, device=DEV, dtype=torch.float32))
    return int(torch.argmin(p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seas', type=int, default=300)
    ap.add_argument('--steps', type=int, default=3000)
    a_ = ap.parse_args()

    prior = RadialGain().to(DEV)
    prior.load_state_dict(torch.load('results/direction3/l1_prior_gain.pt'))
    prior.eval()
    gen = torch.Generator(device=DEV).manual_seed(0)

    print(f"building critic training set: {a_.seas} synthetic seas x 20 "
          f"candidates", flush=True)
    X, Y, S = build_dataset(prior, a_.seas, gen)
    print(f"  {len(X)} rows, label std {Y.std():.4f}\n", flush=True)

    crit, mu, sd, vl, acc, reg, reg_res = train(X, Y, S, False, a_.steps)
    ctrl, mu2, sd2, vl2, acc2, reg2, _ = train(X, Y, S, True, a_.steps)
    print(f"held-out synthetic seas:")
    print(f"  {'':<18}{'argmin':>9}{'REGRET':>10}")
    print(f"  {'residual rule':<18}{'-':>9}{reg_res:>10.4f}   <- the baseline")
    print(f"  {'critic':<18}{acc*100:>8.0f}%{reg:>10.4f}")
    print(f"  {'no-residual ctl':<18}{acc2*100:>8.0f}%{reg2:>10.4f}")
    print(f"\nregret = field nRMSE of the chosen candidate minus the sea's best."
          f"\nSTAGED GATE (revised, and the revision is recorded): the first gate"
          f"\nasked for argmin-correct >= 35%, which is the wrong quantity --"
          f"\nwith twenty near-tied candidates, missing the winner costs almost"
          f"\nnothing. The gate is regret: the critic must cut the residual"
          f"\nrule's regret by half on synthetic seas before it touches a real one."
          f"\n", flush=True)
    if reg > 0.5 * reg_res or reg2 < reg + 0.005:
        print(f"SYNTHETIC GATE FAILED: critic regret {reg:.4f} vs residual "
              f"{reg_res:.4f} (need <= {0.5*reg_res:.4f});\n"
              f"control regret {reg2:.4f} (must be >> critic). "
              f"Real evaluation skipped.")
        with open(OUT, 'w') as f:
            json.dump({'synth': dict(argmin_acc=acc, ctl_argmin_acc=acc2,
                                     regret=reg, regret_ctl=reg2,
                                     regret_residual=reg_res),
                       'synthetic_gate': False}, f, indent=2)
        return 1

    # ---------------- real evaluation ------------------------------------
    t = lambda x: torch.tensor(x, device=DEV, dtype=torch.float32)
    print(f"{'record':<8}{'win':>5}{'hand':>9}{'CRITIC':>9}{'ctl':>8}"
          f"{'fam orc':>9}{'alpha':>7}")
    cells, per = [], {}
    for rec in RECS:
        rows = []
        for w in WINDOWS:
            try:
                sc = SF.scene_real(rec, win=w)
            except Exception:
                continue
            Xq, Yq = np.meshgrid(sc['gx'], sc['gy'])
            eta = sc['make_eta']()
            g_all = eta(Xq.ravel(), Yq.ravel(), sc['t0'])
            cx, cy = sc['center']
            inside, _ = SF.fan_mask(Xq, Yq, cx, cy, sc['hvec'])
            m = inside.ravel() & np.isfinite(g_all)
            if m.sum() < 150:
                continue
            Pq = t(np.column_stack([Xq.ravel()[m], Yq.ravel()[m],
                                    np.full(m.sum(), sc['t0'])]))
            gtq = t(g_all[m])
            gtq = gtq - gtq.mean()
            if float(gtq.std()) < 1e-6:
                continue
            kp, dep, sig0 = sc['kp'], sc['depth'], 0.05 * sc['Hs']
            gains = gains_for(prior, kp, dep)
            cell = []
            for ds in RSEEDS:
                rng = np.random.default_rng(1000 + ds)
                P, Z = draw(sc, rng)
                if len(Z) < 200:
                    continue
                Pt, zt = t(P), t(Z)
                n = len(zt)
                idx = torch.randperm(n, device=DEV, generator=gen)
                tr, he = idx[:int((1 - HELD) * n)], idx[int((1 - HELD) * n):]
                keys, feats, rats = [], [], []
                for (band, al), (g, kmag, om) in gains.items():
                    keys.append((band, al, g))
                    feats.append(features(Pt, zt, tr, he, kp, dep, band, g, al,
                                          kmag, sig0))
                    rats.append(field_ratio(Pt, zt, Pq, gtq, kp, dep, band, g,
                                            sig0))
                feats, rats = np.array(feats), np.array(rats)
                hand_i = int(np.argmin([f[0] for f, k in zip(feats, keys)
                                        if k[1] == 0.0]))
                hand_keys = [i for i, k in enumerate(keys) if k[1] == 0.0]
                hand = rats[hand_keys[hand_i]]
                ci = pick(crit, mu, sd, feats)
                oi = pick(ctrl, mu2, sd2, feats, True)
                cell.append((hand, rats[ci], rats[oi], rats.min(),
                             keys[ci][1]))
            if not cell:
                continue
            c = np.array(cell).mean(0)
            print(f"{rec:<8}{w:>5}{c[0]:>9.3f}{c[1]:>9.3f}{c[2]:>8.3f}"
                  f"{c[3]:>9.3f}{c[4]:>7.2f}", flush=True)
            rows.append(c)
            cells.append(c)
        if rows:
            r = np.array(rows)
            per[rec] = dict(hand=r[:, 0].mean(), critic=r[:, 1].mean(),
                            ctrl=r[:, 2].mean(), oracle=r[:, 3].mean(),
                            alpha=r[:, 4].mean())

    a = np.array(cells)
    d = a[:, 0] - a[:, 1]
    se = d.std(ddof=1) / np.sqrt(len(d))
    of = a[:, 1] - a[:, 3]                        # critic gap to family oracle
    calm = np.array([per[r]['hand'] - per[r]['critic'] for r in CALM
                     if r in per])
    dctl = a[:, 0] - a[:, 2]
    print(f"\nper record: {'':<2}" + "  ".join(
        f"{k}: hand {v['hand']:.3f} critic {v['critic']:.3f} "
        f"({v['hand']-v['critic']:+.3f})" for k, v in per.items()))
    print(f"\npaired over {len(d)} real cells:")
    print(f"  critic beats hand by {d.mean():+.4f} ± {se:.4f} (se)   "
          f"t = {d.mean()/se:+.2f}   wins {int((d>0).sum())}/{len(d)}")
    print(f"  critic gap to family oracle: {of.mean():+.4f}   "
          f"(E1's residual gate left +0.016 over the hand rule's own gap)")
    print(f"  no-residual control beats hand by {dctl.mean():+.4f}   "
          f"(must be << critic)")
    print(f"  mean alpha chosen: {a[:,4].mean():.2f}")

    g1 = d.mean() >= 0.010
    g3 = (calm.min() >= -0.005) if len(calm) else False
    g4 = dctl.mean() < d.mean() - 0.005
    print(f"\n=== N1 gates (pre-registered) ===")
    print(f"  1) beat hand by >= +0.010     : {'PASS' if g1 else 'FAIL'}"
          f"  ({d.mean():+.4f})")
    print(f"  3) calm records lose <= 0.005 : {'PASS' if g3 else 'FAIL'}"
          f"  (worst {calm.min():+.4f})")
    print(f"  4) negative control collapses : {'PASS' if g4 else 'FAIL'}"
          f"  (ctl {dctl.mean():+.4f} vs critic {d.mean():+.4f})")

    torch.save({'critic': crit.state_dict(), 'mu': mu, 'sd': sd}, CKPT)
    os.makedirs('results/direction3', exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump({'synth': dict(val_mse=vl, argmin_acc=acc,
                                 ctl_val_mse=vl2, ctl_argmin_acc=acc2),
                   'per_record': {k: {kk: float(vv) for kk, vv in v.items()}
                                  for k, v in per.items()},
                   'paired_gain': float(d.mean()), 'se': float(se),
                   'ctl_gain': float(dctl.mean()),
                   'critic_oracle_gap': float(of.mean()),
                   'gates': dict(beat=bool(g1), calm=bool(g3),
                                 negctl=bool(g4))}, f, indent=2)
    return 0


if __name__ == '__main__':
    sys.exit(main())
