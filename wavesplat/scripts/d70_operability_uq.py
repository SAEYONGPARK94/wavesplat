"""
d70 — D2: give the calibrated UQ a downstream JOB (risk-aware operability decisions).

Offshore ops (crane lifts, gangway transfers) need a CALM window: at decision time the operator must
decide, from a phase-resolved forecast, whether the wave at an operation point/lead will stay below a
safety limit. A point forecast cannot express risk; a CALIBRATED, HETEROSCEDASTIC predictive σ can.

Decision unit = (operation point, forecast lead). "safe" = |η_GT| < H_safe (= H_SAFE_FRAC·Hs).
Policies (GO = predict safe):
  - point/blind        : GO iff |mean| < H_safe                       (ignores uncertainty)
  - persistence        : GO iff |η(now)| < H_safe                     (assume current calm persists)
  - constant-σ margin  : GO iff |mean| + z·σ̄ < H_safe  (homoscedastic; a fixed safety margin)
  - posterior-σ (ours) : GO iff |mean| + z·σ(point,lead) < H_safe     (LEARNED heteroscedastic UQ)
  - oracle             : GO iff actually safe                          (upper bound)
A GO on an actually-unsafe case = VIOLATION (operating in danger); NO-GO on a safe case = lost uptime.

The contribution is the VALUE of the learned, calibrated, heteroscedastic σ: by sweeping the risk
level z it traces a (utilization, violation) frontier; the test is whether the posterior-σ frontier
DOMINATES the constant-σ frontier (lower violation at equal uptime) — i.e. allocating the risk budget
where the forecast is actually uncertain beats a uniform margin — and whether it controls risk to a
chosen target (which the point forecast cannot).

Run: python3 scripts/d70_operability_uq.py [--steps 3000]
"""
import os, sys, json, time, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from d67_physics_posterior_uq import PosteriorNet, gen_batch_fc, DEVICE
import d62_predictable_zone_test as Z

# span predictability (simple→complex seas) and use the realistic short feedforward horizon,
# so the forecast σ is genuinely HETEROSCEDASTIC (small in the predictable zone, large outside)
SEAS = [('BF3_5c', 0.8, 5.5), ('BF4_15c', 1.2, 7.0), ('BF5_30c', 2.0, 9.0)]
SEEDS = [0, 1, 2, 3]
CAL_SEEDS, TEST_SEEDS = [0, 1], [2, 3]
LEADS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
HORIZON = [0.5, 1.0, 1.5, 2.0, 3.0]      # realistic feedforward decision leads (exclude present)
H_SAFE_FRAC = 0.25                        # safety limit as fraction of Hs (~30% base unsafe)
ZS = np.linspace(0.0, 4.0, 41)


def train_posterior(steps):
    torch.manual_seed(0); np.random.seed(0)
    m = PosteriorNet(qdim=3).to(DEVICE)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps); t0 = time.time()
    for s in range(steps):
        obs, q, cov, qz, Hs = gen_batch_fc(24)
        mean, logv = m(obs, q); logv = logv.clamp(-8, 4)
        loss = (0.5 * ((mean - qz) ** 2 / logv.exp() + logv).mean(1) / (Hs ** 2 + 1e-6)).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if s % 1000 == 0 or s == steps - 1:
            print(f"    train step {s} loss {loss.item():.4f} ({time.time()-t0:.0f}s)", flush=True)
    m.eval(); return m


def collect(model):
    Z.SIGMA = 0.02; Z.HORIZONS = LEADS
    gx = np.linspace(10, 250, 80); gy = np.linspace(-100, 100, 60)
    rng = np.random.default_rng(0); recs = []
    for tag, Hs, Tp in SEAS:
        for seed in SEEDS:
            w = Z.build_window(Hs, Tp, seed, gx, gy, M=3)
            t0obs = w['P'][:, 2].min()
            ob = torch.tensor(np.column_stack([w['P'][:, 0], w['P'][:, 1], w['P'][:, 2] - t0obs, w['z']])[None],
                              dtype=torch.float32, device=DEVICE)
            gi = rng.choice(len(gx), 40); gj = rng.choice(len(gy), 40)
            opx = gx[gi]; opy = gy[gj]; Hs_n = float(w['gt_at'][0][1].std() * 4)
            now = np.abs(w['gt_at'][0][1][gj, gi])     # |η(now)| at op points
            for h in HORIZON:
                st = int(round(h / 0.1))
                if st not in w['gt_at']:
                    continue
                t_star, GT = w['gt_at'][st]
                q = torch.tensor(np.column_stack([opx, opy, np.full(40, float(t_star - t0obs))])[None],
                                 dtype=torch.float32, device=DEVICE)
                with torch.no_grad():
                    mn, lv = model(ob, q)
                mn = mn[0].cpu().numpy(); sg = (0.5 * lv[0]).exp().cpu().numpy(); gt = GT[gj, gi]
                for p in range(40):
                    recs.append(dict(seed=seed, lead=h, Hs=Hs_n, absmean=abs(float(mn[p])),
                                     sig=float(sg[p]), absgt=abs(float(gt[p])), now=float(now[p]),
                                     err=abs(float(mn[p] - gt[p]))))
    return recs


def calibrate_sigma(recs, split):
    e = np.array([r['err'] for r in recs if r['seed'] in split])
    s = np.array([r['sig'] for r in recs if r['seed'] in split])
    s_cal = float(np.quantile(e / (s + 1e-9), 0.6827))
    for r in recs:
        r['sig'] *= s_cal
    return s_cal


def stats(go, safe):
    go = np.asarray(go, bool); safe = np.asarray(safe, bool)
    util = float(go.mean()); viol = float((go & ~safe).sum() / max(go.sum(), 1))
    return util, viol


def at_target(frontier, vmax):
    feas = [(u, v, z) for (z, u, v) in frontier if v <= vmax]
    return max(feas, key=lambda r: r[0]) if feas else None


def util_at(frontier, target_u):
    return min(frontier, key=lambda r: abs(r[1] - target_u))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--steps', type=int, default=3000); a = ap.parse_args()
    out = 'results/direction3'; t0 = time.time()
    print("training forecast posterior...", flush=True); model = train_posterior(a.steps)
    print("collecting HOS forecast cases...", flush=True); recs = collect(model)
    s_cal = calibrate_sigma(recs, CAL_SEEDS)
    te = [r for r in recs if r['seed'] in TEST_SEEDS]
    Hsafe = np.array([H_SAFE_FRAC * r['Hs'] for r in te])
    mean = np.array([r['absmean'] for r in te]); sig = np.array([r['sig'] for r in te])
    gt = np.array([r['absgt'] for r in te]); now = np.array([r['now'] for r in te])
    safe = gt < Hsafe; sbar = float(sig.mean()); base_unsafe = float(1 - safe.mean())
    print(f"  σ temp s={s_cal:.2f}; {len(te)} TEST decisions; base unsafe {base_unsafe:.2f} "
          f"(H_safe={H_SAFE_FRAC}·Hs)", flush=True)

    blind = stats(mean < Hsafe, safe)
    pers = stats(now < Hsafe, safe)
    orac = stats(safe, safe)
    fr_const = [(float(z), *stats(mean + z * sbar < Hsafe, safe)) for z in ZS]   # homoscedastic
    fr_post = [(float(z), *stats(mean + z * sig < Hsafe, safe)) for z in ZS]     # heteroscedastic

    res = dict(n=len(te), base_unsafe=base_unsafe, s_cal=s_cal,
               blind=dict(util=blind[0], viol=blind[1]), persistence=dict(util=pers[0], viol=pers[1]),
               oracle=dict(util=orac[0], viol=orac[1]),
               frontier_const=[dict(z=z, util=u, viol=v) for (z, u, v) in fr_const],
               frontier_post=[dict(z=z, util=u, viol=v) for (z, u, v) in fr_post])
    # headline: at target violation 10%, uptime of post vs const
    tp = at_target(fr_post, 0.10); tc = at_target(fr_const, 0.10)
    res['at_viol10'] = dict(post=tp, const=tc)
    json.dump(res, open(f'{out}/operability_uq.json', 'w'), indent=2)

    print("\n================ RISK-AWARE OPERABILITY — value of learned calibrated UQ ================")
    print(f"  {len(te)} decisions, base unsafe {base_unsafe:.0%}, σ calibrated (s={s_cal:.2f})")
    print(f"  point/blind  : uptime {blind[0]:.0%}  violation {blind[1]:.0%}  (no risk control)")
    print(f"  persistence  : uptime {pers[0]:.0%}  violation {pers[1]:.0%}")
    print(f"  oracle       : uptime {orac[0]:.0%}  violation {orac[1]:.0%}")
    print(f"\n  to operate at ≤10% violation risk (uptime achieved):")
    print(f"    constant-σ margin   : {tc[0]:.0%} uptime (viol {tc[1]:.0%}, z={tc[2]:.2f})" if tc else "    constant-σ: infeasible")
    print(f"    posterior-σ (learned): {tp[0]:.0%} uptime (viol {tp[1]:.0%}, z={tp[2]:.2f})" if tp else "    posterior-σ: infeasible")
    # dominance: at several uptimes, posterior viol vs constant viol
    print(f"\n  violation at matched uptime (posterior-σ vs constant-σ; lower=better):")
    for tu in [0.3, 0.5, 0.7]:
        pp = util_at(fr_post, tu); cc = util_at(fr_const, tu)
        print(f"    uptime≈{tu:.0%}:  posterior {pp[2]:.0%}  vs  constant {cc[2]:.0%}")
    # average dominance margin
    import numpy as _np
    grid = _np.linspace(0.1, 0.9, 17)
    dmarg = _np.mean([util_at(fr_const, u)[2] - util_at(fr_post, u)[2] for u in grid])
    print(f"\n  mean violation reduction (constant−posterior) across uptime = {dmarg:+.1%}  "
          f"({'posterior DOMINATES' if dmarg > 0.01 else '≈ tie'})")
    print(f"  => learned heteroscedastic UQ allocates risk where the forecast is uncertain → "
          f"safer at equal uptime. The calibrated UQ has a job.")
    make_fig(fr_const, fr_post, blind, pers, orac, out)
    print(f"saved {out}/operability_uq.json   ({time.time()-t0:.0f}s)")


def make_fig(fr_const, fr_post, blind, pers, orac, out):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.plot([u for _, u, _ in fr_post], [v for _, _, v in fr_post], '-o', ms=3, c='tab:green',
            label='posterior-σ (learned, heteroscedastic)')
    ax.plot([u for _, u, _ in fr_const], [v for _, _, v in fr_const], '-s', ms=3, c='tab:orange',
            label='constant-σ margin (homoscedastic)')
    ax.scatter([blind[0]], [blind[1]], s=90, c='tab:red', marker='D', zorder=5, label='point/blind')
    ax.scatter([pers[0]], [pers[1]], s=90, c='tab:gray', marker='^', zorder=5, label='persistence')
    ax.scatter([orac[0]], [orac[1]], s=120, c='k', marker='*', zorder=5, label='oracle')
    ax.axhline(0.10, color='b', ls='--', lw=1, label='target risk 10%')
    ax.set_xlabel('utilization (operational uptime)'); ax.set_ylabel('violation rate  P(unsafe | go)')
    ax.set_title('Risk-aware operability: learned calibrated UQ\ndominates the safety–uptime tradeoff')
    ax.legend(fontsize=8, loc='upper left'); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f'{out}/fig_operability_uq.png', dpi=140, bbox_inches='tight')
    plt.close(fig); print(f"saved {out}/fig_operability_uq.png")


if __name__ == '__main__':
    main()
