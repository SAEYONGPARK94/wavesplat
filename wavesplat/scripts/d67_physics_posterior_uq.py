"""
d67 — #2: integrate the physics UQ into the amortized posterior's σ-head.

d58 found the learned posterior's σ is slightly OVER-confident at long forecast lead (the same
lead-structured miscalibration the unified physics UQ fixes, d64). Here we give the learned σ-head
the physics: each query carries a COVERAGE feature

    cov(q) = exp(−d_geo(q)/ℓ),   d_geo = distance from q to the observed cloud advected to the query
                                  time by the (dominant) group velocity,

so the network can condition its uncertainty on physical predictability. We train two otherwise
identical fast amortized posteriors on continuous-spectrum FORECAST queries:
    - vanilla   : query = (x, y, t)
    - physics   : query = (x, y, t, cov)
and compare ±1σ/±2σ coverage ACROSS forecast lead on held-out NONLINEAR HOS windows, against the
non-learned unified physics UQ (d64) as reference. Goal: one fast forward pass giving accurate mean
+ lead-STABLE calibrated UQ.

Run: python3 scripts/d67_physics_posterior_uq.py [--steps 4000]
"""
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wavesim.wave_generator import jonswap_components
from d23_main_results import K_TP, panel
from d40_learned_acquisition import pool_xy
from d52_F4_neural_operator import XS, YS, TS, ZS, DEVICE
import d62_predictable_zone_test as Z
from wavesplat.model.tnse import render_wave_field

DEPTH = 50.0; G = 9.81
ELL = 40.0                       # coverage length scale (m); a network-input scale, learned from
LEADS = [0.0, 2.0, 4.0, 6.0]


def cg_np(k):
    kd = k * DEPTH; th = np.tanh(kd); w = np.sqrt(G * k * th + 1e-30)
    return G * (th + kd * (1 - th ** 2)) / (2 * w + 1e-30)


def coverage(qx, qy, t_star, ox, oy, ot, kx, ky, A):
    """cov(q)=exp(−d_geo/ELL), d_geo = dist to obs advected to t_star by the dominant group velocity."""
    ang = np.arctan2(np.sum(A ** 2 * ky), np.sum(A ** 2 * kx))
    di = int(np.argmax(A ** 2)); kdom = np.hypot(kx[di], ky[di]); cgd = cg_np(np.array([kdom]))[0]
    ax = ox + cgd * np.cos(ang) * (t_star - ot)
    ay = oy + cgd * np.sin(ang) * (t_star - ot)
    from scipy.spatial.distance import cdist
    d = cdist(np.column_stack([qx, qy]), np.column_stack([ax, ay])).min(1)
    return np.exp(-d / ELL).astype(np.float32)


def gen_batch_fc(B, M_obs=200, Q=400, lead_max=6.0):
    pool = pool_xy()
    obs = np.zeros((B, M_obs, 4), np.float32); qpt = np.zeros((B, Q, 3), np.float32)
    cov = np.zeros((B, Q), np.float32); qz = np.zeros((B, Q), np.float32); Hsb = np.zeros(B, np.float32)
    for b in range(B):
        Hs = np.random.uniform(0.6, 2.0); Tp = np.random.uniform(5.5, 9.5)
        comp = jonswap_components(Hs=Hs, Tp=Tp, n_components=64, depth=DEPTH, seed=int(np.random.randint(1e9)))
        A = np.array([c.A for c in comp]); om = np.array([c.omega for c in comp])
        kx = np.array([c.kx for c in comp]); ky = np.array([c.ky for c in comp]); ph = np.array([c.phi for c in comp])
        Tobs = K_TP * Tp; tstar = Tobs + np.random.uniform(0, lead_max)
        oi = np.random.randint(0, len(pool), M_obs); ox, oy = pool[oi, 0], pool[oi, 1]
        ot = np.random.uniform(0, Tobs, M_obs)
        oz = (A[None]*np.cos(kx[None]*ox[:,None]+ky[None]*oy[:,None]-om[None]*ot[:,None]+ph[None])).sum(1)
        oz += np.random.normal(0, 0.02, M_obs)
        obs[b] = np.column_stack([ox, oy, ot, oz])
        qx = np.random.uniform(10, 250, Q); qy = np.random.uniform(-100, 100, Q)
        qz[b] = (A[None]*np.cos(kx[None]*qx[:,None]+ky[None]*qy[:,None]-om[None]*tstar+ph[None])).sum(1)
        qpt[b] = np.column_stack([qx, qy, np.full(Q, tstar)])
        cov[b] = coverage(qx, qy, tstar, ox, oy, ot, kx, ky, A); Hsb[b] = Hs
    t = lambda a: torch.tensor(a, device=DEVICE)
    return t(obs), t(qpt), t(cov), t(qz), t(Hsb)


class PosteriorNet(nn.Module):
    def __init__(self, qdim=3, d=160, heads=4, enc=4, dec=3):
        super().__init__()
        self.oenc = nn.Sequential(nn.Linear(4, d), nn.GELU(), nn.Linear(d, d))
        el = nn.TransformerEncoderLayer(d, heads, 4*d, batch_first=True, activation='gelu')
        self.enc = nn.TransformerEncoder(el, enc)
        self.qenc = nn.Sequential(nn.Linear(qdim, d), nn.GELU(), nn.Linear(d, d))
        dl = nn.TransformerDecoderLayer(d, heads, 4*d, batch_first=True, activation='gelu')
        self.dec = nn.TransformerDecoder(dl, dec)
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2))
        self.qdim = qdim

    def forward(self, obs, q, cov=None):
        o = obs.clone(); o[..., 0] /= XS; o[..., 1] /= YS; o[..., 2] /= TS; o[..., 3] /= ZS
        ctx = self.enc(self.oenc(o))
        qn = q.clone(); qn[..., 0] /= XS; qn[..., 1] /= YS; qn[..., 2] /= TS
        if self.qdim == 4:
            qn = torch.cat([qn, cov[..., None]], -1)
        h = self.dec(self.qenc(qn), ctx); out = self.head(h)
        return out[..., 0], out[..., 1]


def train(qdim, steps, B=24, lr=3e-4):
    torch.manual_seed(0)
    model = PosteriorNet(qdim=qdim).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps); t0 = time.time()
    for s in range(steps):
        obs, q, cov, qz, Hs = gen_batch_fc(B)
        mean, logv = model(obs, q, cov); logv = logv.clamp(-8, 4)
        nll = 0.5 * ((mean - qz)**2 / logv.exp() + logv)
        loss = (nll.mean(1) / (Hs**2 + 1e-6)).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if s % 1000 == 0 or s == steps-1:
            print(f"    [qdim{qdim}] step {s:4d} loss {loss.item():.4f} ({time.time()-t0:.0f}s)", flush=True)
    model.eval(); return model


# ---- HOS forecast evaluation ----
SEAS = [('BF4_15c', 1.2, 7.0), ('BF5_30c', 2.0, 9.0)]
CAL_SEEDS, TEST_SEEDS = [0, 1], [2, 3]


def eval_hos(models):
    Z.SIGMA = 0.02; Z.HORIZONS = LEADS
    gx = np.linspace(10, 250, 80); gy = np.linspace(-100, 100, 60)
    Xg, Yg = np.meshgrid(gx, gy); grid = np.column_stack([Xg.ravel(), Yg.ravel()])
    recs = []  # per (seed,lead): dict of err + sigma per model + unified physics
    for tag, Hs, Tp in SEAS:
        for seed in CAL_SEEDS + TEST_SEEDS:
            w = Z.build_window(Hs, Tp, seed, gx, gy, M=3)
            kx, ky, A, phi = Z.estimate(w['P'], w['z'], w['t_now'])
            th = np.column_stack([A, kx, ky, phi]); Hs_n = float(w['gt_at'][0][1].std()*4)
            ob = torch.tensor(np.column_stack([w['P'][:,0], w['P'][:,1], w['P'][:,2]-w['P'][:,2].min(),
                                               w['z']])[None], dtype=torch.float32, device=DEVICE)
            t0obs = w['P'][:,2].min()
            for h in LEADS:
                st = int(round(h/0.1))
                if st not in w['gt_at']:
                    continue
                t_star, GT = w['gt_at'][st]
                qt = float(t_star - t0obs)
                q = torch.tensor(np.column_stack([grid, np.full(len(grid), qt)])[None], dtype=torch.float32, device=DEVICE)
                cov = coverage(grid[:,0], grid[:,1], t_star, w['P'][:,0], w['P'][:,1], w['P'][:,2], kx, ky, A)
                covt = torch.tensor(cov[None], dtype=torch.float32, device=DEVICE)
                rec = dict(seed=seed, lead=h, gt=GT.ravel(), Hs=Hs_n)
                with torch.no_grad():
                    for nm, mdl in models.items():
                        mean, logv = mdl(ob, q, covt if mdl.qdim == 4 else None)
                        m = mean[0].cpu().numpy(); sg = (0.5*logv[0]).exp().cpu().numpy()
                        rec[f'mean_{nm}'] = m; rec[f'sig_{nm}'] = sg
                # unified physics UQ (non-learned reference): use RELAX mean + d64 sigma
                sigF = Z.fim_sigma_map(th, w['P'][:,0], w['P'][:,1], w['P'][:,2], gx, gy, t_star).ravel()
                dgeo = Z.geo_distance_map(kx, ky, w['P'][:,0], w['P'][:,1], w['P'][:,2], gx, gy, t_star).ravel()
                c = np.exp(-dgeo/26.0)
                rec['sig_phys'] = np.sqrt(c*sigF**2 + (1-c)*(Hs_n/4.0)**2)
                rec['mean_relax'] = render_wave_field(kx, ky, A, phi, DEPTH, gx, gy, t_star).ravel()
                recs.append(rec)
    return recs


def temp(err, sig, target=0.6827):
    lo, hi = 1e-3, 1e3
    for _ in range(50):
        s = np.sqrt(lo*hi)
        if np.mean(err < s*sig) < target: lo = s
        else: hi = s
    return np.sqrt(lo*hi)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--steps', type=int, default=4000); a = ap.parse_args()
    print("training vanilla posterior...", flush=True);   mv = train(3, a.steps)
    print("training physics-informed posterior...", flush=True); mp = train(4, a.steps)
    models = {'vanilla': mv, 'physics': mp}
    print("evaluating on HOS forecast windows...", flush=True)
    recs = eval_hos(models)

    methods = ['vanilla', 'physics', 'phys']   # phys = non-learned unified physics UQ
    err_of = {'vanilla': 'mean_vanilla', 'physics': 'mean_physics', 'phys': 'mean_relax'}
    # calibrate one global temperature per method on CAL seeds (to ±1σ=0.68)
    cal = [r for r in recs if r['seed'] in CAL_SEEDS]
    s_of = {}
    for mth in methods:
        e = np.concatenate([np.abs(r[err_of[mth]] - r['gt'])/r['Hs'] for r in cal])
        sg = np.concatenate([r[f'sig_{mth}']/r['Hs'] for r in cal])
        s_of[mth] = temp(e, sg)
    # accuracy (nRMSE) of learned means on TEST
    test = [r for r in recs if r['seed'] in TEST_SEEDS]
    acc = {}
    for mth in ['vanilla', 'physics', 'phys']:
        nr = [np.sqrt(np.mean((r[err_of[mth]] - r['gt'])**2))/r['Hs'] for r in test if r['lead'] == 0.0]
        acc[mth] = float(np.mean(nr))
    # coverage vs lead on TEST
    per_lead = {}
    for h in LEADS:
        rs = [r for r in test if r['lead'] == h]
        if not rs: continue
        row = {}
        for mth in methods:
            e = np.concatenate([np.abs(r[err_of[mth]] - r['gt'])/r['Hs'] for r in rs])
            sg = np.concatenate([r[f'sig_{mth}']/r['Hs'] for r in rs]) * s_of[mth]
            row[mth] = dict(cov1=float((e < sg).mean()), cov2=float((e < 2*sg).mean()))
        per_lead[str(h)] = row

    res = dict(steps=a.steps, ell=ELL, s_of=s_of, accuracy_nowcast=acc, per_lead=per_lead)
    json.dump(res, open('results/direction3/physics_posterior_uq.json', 'w'), indent=2)

    print("\n================ PHYSICS-INFORMED POSTERIOR σ-HEAD (HOS forecast, TEST) ================")
    print(f"  nowcast accuracy nRMSE:  vanilla {acc['vanilla']:.3f}  physics {acc['physics']:.3f}  "
          f"(RELAX mean {acc['phys']:.3f})")
    print(f"  (all temperature-matched to ±1σ=0.68 on calibration seeds)")
    print(f"\n  ±1σ coverage vs lead  (ideal 0.68):")
    print(f"    lead    vanilla   physics   unified-phys")
    for h in LEADS:
        if str(h) in per_lead:
            r = per_lead[str(h)]
            print(f"    {h:.0f}s     {r['vanilla']['cov1']:.2f}      {r['physics']['cov1']:.2f}      {r['phys']['cov1']:.2f}")
    dev = {m: np.mean([abs(per_lead[str(h)][m]['cov1']-0.683) for h in LEADS if str(h) in per_lead]) for m in methods}
    print(f"\n  mean |±1σ−0.68| across lead:  vanilla {dev['vanilla']:.3f}  physics {dev['physics']:.3f}  "
          f"unified-phys {dev['phys']:.3f}  (lower=better lead-calibration)")
    print(f"  => physics-informed σ-head {'IMPROVES' if dev['physics'] < dev['vanilla'] - 0.005 else '≈/✗'} "
          f"lead-calibration vs vanilla")
    make_fig(per_lead, 'results/direction3')
    print("saved results/direction3/physics_posterior_uq.json")


def make_fig(per_lead, out):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    hs = [float(h) for h in sorted(per_lead, key=float)]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.axhline(0.683, color='k', ls='--', lw=1, label='ideal 0.68')
    for m, c in [('vanilla', 'tab:red'), ('physics', 'tab:green'), ('phys', 'tab:blue')]:
        ax.plot(hs, [per_lead[str(h)][m]['cov1'] for h in hs], 'o-', color=c,
                label={'vanilla': 'vanilla NN', 'physics': 'physics-informed NN', 'phys': 'unified physics (non-learned)'}[m])
    ax.set_xlabel('forecast lead [s]'); ax.set_ylabel('±1σ coverage'); ax.set_ylim(0.45, 1.0)
    ax.set_title('Physics-informed σ-head stays lead-calibrated (HOS forecast)')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f'{out}/fig_physics_posterior_uq.png', dpi=140, bbox_inches='tight')
    plt.close(fig); print(f"saved {out}/fig_physics_posterior_uq.png")


if __name__ == '__main__':
    main()
