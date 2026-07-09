"""
d71 — D2 (option 1): feedforward wave-induced MOTION COMPENSATION from the phase-resolved forecast.

A platform's wave-induced motion is m(t) ≈ η(x_op, t) (heave proxy, RAO≈1). An actuator has reaction
latency τ: to cancel motion at time t it must be commanded at t−τ, i.e. from a τ-ahead forecast.
After feedforward compensation the RESIDUAL motion = m(t) − command. So:
  - no compensation : residual = m(t)              (RMS = full motion)
  - persistence FF  : command = m(t−τ)  → residual = m(t) − m(t−τ)   (assume motion persists)
  - forecast FF     : command = m̂(t|t−τ) → residual = m(t) − m̂ = forecast error
  - oracle FF       : command = m(t)     → residual = 0
The forecast has CONTROL value iff forecast-FF leaves less residual motion than persistence-FF. The
posterior σ then directly predicts the residual (since residual = forecast error, and σ is the
calibrated forecast-error std) → the operator knows in advance how well compensation will work.

Reports residual-motion RMS (as % of uncompensated motion) vs actuator latency τ, for each method,
plus the σ↔residual calibration. `scripts/d71_motion_compensation.py`.
"""
import os, sys, json, time, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from d67_physics_posterior_uq import PosteriorNet, gen_batch_fc, DEVICE
import d62_predictable_zone_test as Z

SEAS = [('BF3_5c', 0.8, 5.5), ('BF4_15c', 1.2, 7.0), ('BF5_30c', 2.0, 9.0)]
SEEDS = [0, 1, 2, 3]
CAL_SEEDS, TEST_SEEDS = [0, 1], [2, 3]
TAUS = [0.5, 1.0, 1.5, 2.0, 3.0]          # actuator latency / forecast lead [s]


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
    """Per (op point, τ): signed forecast mean m̂, σ, true motion m(t_now+τ), current motion m(t_now)."""
    Z.SIGMA = 0.02; Z.HORIZONS = [0.0] + TAUS
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
            now = w['gt_at'][0][1][gj, gi]                  # signed m(t_now)
            for tau in TAUS:
                st = int(round(tau / 0.1))
                if st not in w['gt_at']:
                    continue
                t_star, GT = w['gt_at'][st]
                q = torch.tensor(np.column_stack([opx, opy, np.full(40, float(t_star - t0obs))])[None],
                                 dtype=torch.float32, device=DEVICE)
                with torch.no_grad():
                    mn, lv = model(ob, q)
                mn = mn[0].cpu().numpy(); sg = (0.5 * lv[0]).exp().cpu().numpy(); m_true = GT[gj, gi]
                for p in range(40):
                    recs.append(dict(seed=seed, tau=tau, Hs=Hs_n, mhat=float(mn[p]), sig=float(sg[p]),
                                     m=float(m_true[p]), now=float(now[p])))
    return recs


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--steps', type=int, default=3000); a = ap.parse_args()
    out = 'results/direction3'; t0 = time.time()
    print("training forecast posterior...", flush=True); model = train_posterior(a.steps)
    print("collecting...", flush=True); recs = collect(model)
    # σ temperature on cal seeds (so σ predicts residual honestly)
    ec = np.array([abs(r['m'] - r['mhat']) for r in recs if r['seed'] in CAL_SEEDS])
    sc = np.array([r['sig'] for r in recs if r['seed'] in CAL_SEEDS])
    s_cal = float(np.quantile(ec / (sc + 1e-9), 0.6827))
    te = [r for r in recs if r['seed'] in TEST_SEEDS]

    per_tau = {}
    for tau in TAUS:
        rs = [r for r in te if r['tau'] == tau]
        if not rs:
            continue
        m = np.array([r['m'] for r in rs]); mhat = np.array([r['mhat'] for r in rs])
        now = np.array([r['now'] for r in rs]); sig = np.array([r['sig'] for r in rs]) * s_cal
        rms_full = np.sqrt(np.mean(m ** 2))
        res_pers = np.sqrt(np.mean((m - now) ** 2))
        res_fc = np.sqrt(np.mean((m - mhat) ** 2))
        # σ predicts residual: coverage of |forecast error| within σ
        err = np.abs(m - mhat); cov1 = float((err < sig).mean())
        corr = float(np.corrcoef(sig, err)[0, 1])
        per_tau[str(tau)] = dict(rms_full=float(rms_full),
                                 res_persist=float(res_pers), res_forecast=float(res_fc),
                                 # as % of uncompensated motion REMAINING (lower=better cancellation)
                                 pct_persist=float(res_pers / rms_full), pct_forecast=float(res_fc / rms_full),
                                 cov1=cov1, sig_err_corr=corr)
    res = dict(s_cal=s_cal, n_test=len(te), per_tau=per_tau)
    json.dump(res, open(f'{out}/motion_compensation.json', 'w'), indent=2)

    print("\n================ FEEDFORWARD MOTION COMPENSATION (HOS, test) ================")
    print(f"  residual motion REMAINING as % of uncompensated (lower = better cancellation):")
    print(f"  {'τ (latency)':>11s}  {'persistence FF':>15s}  {'forecast FF':>12s}  {'forecast better by':>18s}")
    for tau in TAUS:
        if str(tau) not in per_tau:
            continue
        r = per_tau[str(tau)]
        gain = (r['pct_persist'] - r['pct_forecast'])
        print(f"  {tau:>9.1f}s   {r['pct_persist']:>13.0%}    {r['pct_forecast']:>10.0%}     "
              f"{gain:>+15.0%}  pts")
    avg_p = np.mean([per_tau[str(t)]['pct_persist'] for t in TAUS if str(t) in per_tau])
    avg_f = np.mean([per_tau[str(t)]['pct_forecast'] for t in TAUS if str(t) in per_tau])
    print(f"\n  mean residual: persistence {avg_p:.0%}  vs  forecast {avg_f:.0%}  "
          f"(forecast cancels {(avg_p-avg_f):+.0%} pts more motion)")
    cov = np.mean([per_tau[str(t)]['cov1'] for t in TAUS if str(t) in per_tau])
    print(f"  σ calibration (residual within ±1σ): {cov:.0%} (ideal 68%) — σ predicts the residual motion.")
    verdict = ("forecast BEATS persistence" if avg_f < avg_p - 0.02 else
               "forecast ≈ persistence" if abs(avg_f - avg_p) <= 0.02 else "forecast WORSE")
    print(f"  => {verdict}: the phase-resolved forecast {'adds control value' if avg_f<avg_p-0.02 else 'does not clearly add value'} over persistence.")
    make_fig(per_tau, out)
    print(f"saved {out}/motion_compensation.json   ({time.time()-t0:.0f}s)")


def make_fig(per_tau, out):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    taus = [float(t) for t in sorted(per_tau, key=float)]
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.axhline(1.0, color='k', ls=':', lw=1, label='no compensation (100%)')
    ax.plot(taus, [per_tau[str(t)]['pct_persist'] for t in taus], 's-', c='tab:gray', label='persistence FF')
    ax.plot(taus, [per_tau[str(t)]['pct_forecast'] for t in taus], 'o-', c='tab:green', label='forecast FF (posterior)')
    ax.set_xlabel('actuator latency τ [s]'); ax.set_ylabel('residual motion (fraction of uncompensated)')
    ax.set_ylim(0, 1.1); ax.set_title('Feedforward motion compensation:\nforecast vs persistence (lower = more motion cancelled)')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f'{out}/fig_motion_compensation.png', dpi=140, bbox_inches='tight')
    plt.close(fig); print(f"saved {out}/fig_motion_compensation.png")


if __name__ == '__main__':
    main()
