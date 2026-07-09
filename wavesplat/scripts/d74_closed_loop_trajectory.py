"""
d74 — Closed-loop informative-TRAJECTORY control for synthetic-aperture wave estimation.

The doctoral positive: stationary ship-LiDAR is directionally information-limited (CRB); the ship's
own motion synthesizes the missing cross-range aperture (d72). Here we close the loop: a maneuvering
platform (USV / ocean basin; Froude-scaled to a tank) chooses its HEADING over an observation window
to MINIMIZE the field-reconstruction CRB, under a turn-rate constraint.

Policies compared (heading sequence over K steps):
  - stationary         : ship fixed (the information-limited baseline)
  - straight-along     : straight, parallel to dominant wave (surfing — expected worst)
  - straight-cross     : straight, perpendicular to dominant wave (the d72-naive-optimal)
  - greedy             : myopic — each step pick the turn minimizing the NEXT-step CRB
  - CEM (non-myopic)   : optimize the whole heading sequence (cross-entropy method) — the planner
  - learned (ours)     : a fast policy net imitating the non-myopic CEM planner (one forward/step)

Questions: (A) does trajectory control beat stationary on ACHIEVED reconstruction? (achievability)
(B) does non-myopic CEM beat greedy? (is there headroom learning can exploit?) (C) can the learned
policy match CEM at a fraction of the cost? (the learning-central contribution)

Speeds sweep the Froude-scaled range (full-scale 2-8 m/s ↔ 1:25 tank 0.4-1.6 m/s). All results saved
progressively. Run: python3 scripts/d74_closed_loop_trajectory.py [--episodes N] [--train_eps N]
"""
import os, sys, json, math, time, argparse
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from d1_crb import DEVICE, DT, jacobian, fisher, field_design, omega_of_k, cg_of_k
from wavesplat.model.tnse import fit_amplitudes_phases, render_wave_field

G = 9.81; DEPTH = 50.0; SIGMA = 0.02
DT_STEP = 0.5            # control/observation step [s] (full scale)
T_OBS = 20.0            # observation window [s]
K = int(round(T_OBS / DT_STEP))    # number of steps
L_FOOT = 20.0          # per-step LiDAR footprint patch size [m]
M_STEP = 30            # observations per step
OMEGA_MAX = math.radians(15.0)     # max turn rate [rad/s] (realistic vessel ~10-20°/s)
ACTIONS = np.radians(np.array([-15, -8, -3, 0, 3, 8, 15]))   # per-step heading change options [rad]
SPEEDS_FULL = [2.0, 5.0, 8.0]      # full-scale ship speeds [m/s] (=tank 0.4/1.0/1.6 at 1:25)


# ---------------- sea + platform ----------------
def make_sea(seed, N=8, spread_deg=22, mean_dir_deg=None):
    rng = np.random.default_rng(seed)
    md = rng.uniform(0, 360) if mean_dir_deg is None else mean_dir_deg
    ang = np.radians(md + rng.uniform(-spread_deg, spread_deg, N))
    kmag = rng.uniform(0.05, 0.12, N)
    A = 0.25 * np.exp(-rng.uniform(0, 1.2, N))
    phi = rng.uniform(0, 2 * math.pi, N)
    return dict(A=A, kx=kmag * np.cos(ang), ky=kmag * np.sin(ang), phi=phi,
                om=np.sqrt(G * kmag * np.tanh(kmag * DEPTH)), kmag=kmag, mean_dir=math.radians(md))


def dom_dir(sea):
    return math.atan2(np.sum(sea['A'] ** 2 * sea['ky']), np.sum(sea['A'] ** 2 * sea['kx']))


def theta_tensor(sea):
    return torch.tensor(np.stack([sea['A'], sea['kx'], sea['ky'], sea['phi']], 1), dtype=DT, device=DEVICE)


def fly_path(sea, speed, headings, gen_z=False, seed=0):
    """Given a per-step heading sequence, fly the ship and gather observations along the path."""
    rng = np.random.default_rng(seed + 777)
    px, py, t = 0.0, 0.0, 0.0
    xs, ys, ts, zs = [], [], [], []
    for j in range(K):
        psi = headings[j]
        ox = px + rng.uniform(-L_FOOT / 2, L_FOOT / 2, M_STEP)
        oy = py + rng.uniform(-L_FOOT / 2, L_FOOT / 2, M_STEP)
        xs.append(ox); ys.append(oy); ts.append(np.full(M_STEP, t))
        if gen_z:
            psi_w = (sea['kx'][None] * ox[:, None] + sea['ky'][None] * oy[:, None]
                     - sea['om'][None] * t + sea['phi'][None])
            z = (sea['A'][None] * np.cos(psi_w)).sum(1) + rng.normal(0, SIGMA, M_STEP)
            zs.append(z)
        px += speed * math.cos(psi) * DT_STEP; py += speed * math.sin(psi) * DT_STEP; t += DT_STEP
    x = np.concatenate(xs); y = np.concatenate(ys); tt = np.concatenate(ts)
    z = np.concatenate(zs) if gen_z else None
    return x, y, tt, z


# ---------------- objective: field-reconstruction CRB ----------------
def recon_grid(sea):
    # reconstruction target: a region around the path/sea at the final time
    gx = torch.linspace(-30, 120, 24, dtype=DT, device=DEVICE)
    gy = torch.linspace(-80, 80, 24, dtype=DT, device=DEVICE)
    return gx, gy


def field_crb(sea, x, y, t, t_star):
    th = theta_tensor(sea); n = th.shape[0] * 4
    xt = torch.tensor(x, dtype=DT, device=DEVICE); yt = torch.tensor(y, dtype=DT, device=DEVICE)
    ttt = torch.tensor(t, dtype=DT, device=DEVICE)
    F = fisher(th, xt, yt, ttt, DEPTH, SIGMA)
    gx, gy = recon_grid(sea)
    Gm = field_design(th, gx, gy, t_star, DEPTH)
    lam = 1e-8 * torch.trace(F) / n
    Finv = torch.linalg.inv(F + lam * torch.eye(n, dtype=DT, device=F.device))
    Hs = 4 * math.sqrt((sea['A'] ** 2 / 2).sum())
    var = torch.trace(Finv @ Gm)
    return float(torch.sqrt(torch.clamp(var, min=0))) / Hs       # field nRMSE CRB


def eval_headings(sea, speed, headings):
    x, y, t, _ = fly_path(sea, speed, headings)
    return field_crb(sea, x, y, t, t_star=T_OBS)


# ---------------- policies ----------------
def headings_straight(sea, rel_deg):
    """Constant heading at rel_deg relative to dominant wave direction."""
    psi = dom_dir(sea) + math.radians(rel_deg)
    return np.full(K, psi)


def headings_stationary(sea):
    return None   # handled specially (speed 0)


def headings_greedy(sea, speed):
    """Myopic: each step choose the turn (within turn-rate) that minimizes the running field CRB."""
    psi = dom_dir(sea) + math.radians(90)   # start cross-track
    chosen = []
    px = py = t = 0.0; obs = [[], [], []]
    rng = np.random.default_rng(1)
    for j in range(K):
        best, best_psi = 1e9, psi
        for da in ACTIONS:
            psj = psi + da
            ox = px + rng.uniform(-L_FOOT / 2, L_FOOT / 2, M_STEP)
            oy = py + rng.uniform(-L_FOOT / 2, L_FOOT / 2, M_STEP)
            tx = obs[0] + list(ox); ty = obs[1] + list(oy); tt = obs[2] + [t] * M_STEP
            c = field_crb(sea, np.array(tx), np.array(ty), np.array(tt), t_star=T_OBS)
            if c < best:
                best, best_psi = c, psj
        chosen.append(best_psi); psi = best_psi
        ox = px + rng.uniform(-L_FOOT / 2, L_FOOT / 2, M_STEP)
        oy = py + rng.uniform(-L_FOOT / 2, L_FOOT / 2, M_STEP)
        obs[0] += list(ox); obs[1] += list(oy); obs[2] += [t] * M_STEP
        px += speed * math.cos(best_psi) * DT_STEP; py += speed * math.sin(best_psi) * DT_STEP; t += DT_STEP
    return np.array(chosen)


def headings_cem(sea, speed, iters=4, pop=48, elite=10, seed=0):
    """Non-myopic: optimize the per-step turn sequence (Δψ) with the cross-entropy method."""
    rng = np.random.default_rng(seed + 11)
    mean = np.zeros(K); std = np.full(K, math.radians(10))
    psi0 = dom_dir(sea) + math.radians(90)
    best_h, best_c = None, 1e9
    for _ in range(iters):
        dpop = np.clip(rng.normal(mean, std, (pop, K)), -OMEGA_MAX * DT_STEP, OMEGA_MAX * DT_STEP)
        costs = []
        for p in range(pop):
            h = psi0 + np.cumsum(dpop[p])
            costs.append(eval_headings(sea, speed, h))
        costs = np.array(costs); idx = np.argsort(costs)[:elite]
        mean = dpop[idx].mean(0); std = dpop[idx].std(0) + 1e-3
        if costs[idx[0]] < best_c:
            best_c = costs[idx[0]]; best_h = psi0 + np.cumsum(dpop[idx[0]])
    return best_h


# ---------------- learned policy (imitate CEM, non-myopic, fast) ----------------
class Policy(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(5, d), nn.GELU(), nn.Linear(d, d), nn.GELU(),
                                 nn.Linear(d, len(ACTIONS)))

    def forward(self, s):
        return self.net(s)


def policy_state(sea, psi, j, speed):
    """Compact state: heading rel to dominant dir (cos,sin), time-to-go, spread, speed-norm."""
    rel = psi - dom_dir(sea)
    spread = float(np.std(np.arctan2(sea['ky'], sea['kx'])))
    return np.array([math.cos(rel), math.sin(rel), 1 - j / K, spread, speed / 8.0], np.float32)


def gen_imitation_data(speed, n_eps, seed0=0):
    """Roll out CEM, record (state, nearest-action) pairs along the optimal trajectory."""
    S, Aidx = [], []
    for e in range(n_eps):
        sea = make_sea(seed0 + e)
        h = headings_cem(sea, speed, seed=seed0 + e)
        psi = h[0] - (h[1] - h[0]) if K > 1 else h[0]
        prev = dom_dir(sea) + math.radians(90)
        for j in range(K):
            da = h[j] - prev
            ai = int(np.argmin(np.abs(ACTIONS - da)))
            S.append(policy_state(sea, prev, j, speed)); Aidx.append(ai)
            prev = h[j]
    return np.array(S, np.float32), np.array(Aidx, np.int64)


def train_policy(speed, n_eps, steps=1500):
    S, A = gen_imitation_data(speed, n_eps)
    Xs = torch.tensor(S, device=DEVICE); Ya = torch.tensor(A, device=DEVICE)
    pol = Policy().to(DEVICE); opt = torch.optim.AdamW(pol.parameters(), lr=2e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    for s in range(steps):
        idx = torch.randint(0, len(Xs), (256,), device=DEVICE)
        out = pol(Xs[idx]); loss = lossf(out, Ya[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    pol.eval(); return pol


def headings_learned(sea, speed, pol):
    psi = dom_dir(sea) + math.radians(90); chosen = []
    for j in range(K):
        s = torch.tensor(policy_state(sea, psi, j, speed)[None], device=DEVICE)
        with torch.no_grad():
            ai = int(pol(s).argmax())
        psi = psi + float(ACTIONS[ai]); chosen.append(psi)
    return np.array(chosen)


# ---------------- achievability: actual RELAX reconstruction ----------------
def achieved_nrmse(sea, speed, headings, seed=0):
    x, y, t, z = fly_path(sea, speed, headings, gen_z=True, seed=seed)
    obs = np.column_stack([x, y, t, z])
    if len(obs) > 1500:
        obs = obs[np.random.default_rng(0).choice(len(obs), 1500, replace=False)]
    # use TRUE wavenumbers (oracle ID) to isolate the APERTURE effect on reconstruction
    A, phi = fit_amplitudes_phases(sea['kx'], sea['ky'], obs, DEPTH, lambda_reg=0.5)
    gxn = np.linspace(-30, 120, 40); gyn = np.linspace(-80, 80, 40)
    fld = render_wave_field(sea['kx'], sea['ky'], A, phi, DEPTH, gxn, gyn, T_OBS)
    Xg, Yg = np.meshgrid(gxn, gyn)
    psi_w = (sea['kx'][None, None] * Xg[..., None] + sea['ky'][None, None] * Yg[..., None]
             - sea['om'][None, None] * T_OBS + sea['phi'][None, None])
    gt = (sea['A'][None, None] * np.cos(psi_w)).sum(-1)
    Hs = 4 * math.sqrt((sea['A'] ** 2 / 2).sum())
    return float(np.sqrt(np.mean((fld - gt) ** 2)) / Hs)


# ---------------- evaluation harness ----------------
OUT = 'results/direction1'
POLICIES = ['stationary', 'straight-along', 'straight-cross', 'greedy', 'CEM', 'learned']


def headings_for(name, sea, speed, pol=None):
    if name == 'straight-along':
        return headings_straight(sea, 0)
    if name == 'straight-cross':
        return headings_straight(sea, 90)
    if name == 'greedy':
        return headings_greedy(sea, speed)
    if name == 'CEM':
        return headings_cem(sea, speed)
    if name == 'learned':
        return headings_learned(sea, speed, pol)
    return headings_straight(sea, 90)   # placeholder (stationary handled via speed=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=12)
    ap.add_argument('--train_eps', type=int, default=30)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    if a.smoke:
        global K, SPEEDS_FULL
        SPEEDS_FULL = [5.0]
        a.episodes = 2; a.train_eps = 3
    os.makedirs(OUT, exist_ok=True); t0 = time.time()
    results = dict(config=dict(K=K, dt=DT_STEP, T_obs=T_OBS, speeds=SPEEDS_FULL,
                               omega_max_deg=math.degrees(OMEGA_MAX), episodes=a.episodes),
                   crb={}, achieved={}, learned_time_ms=None)

    # ---- Stage A+B: per speed, train learned policy, then eval all policies on CRB ----
    for sp in SPEEDS_FULL:
        print(f"\n=== speed {sp} m/s (tank {sp/5:.2f} m/s @1:25) ===", flush=True)
        print(f"  training learned policy (imitate CEM, {a.train_eps} eps)...", flush=True)
        pol = train_policy(sp, a.train_eps); print(f"   trained ({time.time()-t0:.0f}s)", flush=True)
        crb_sp = {p: [] for p in POLICIES}
        for e in range(a.episodes):
            sea = make_sea(10000 + e)            # held-out episodes (train used 0..train_eps)
            crb_sp['stationary'].append(eval_headings(sea, 0.0, headings_straight(sea, 90)))
            for p in ['straight-along', 'straight-cross', 'greedy', 'CEM']:
                crb_sp[p].append(eval_headings(sea, sp, headings_for(p, sea, sp)))
            crb_sp['learned'].append(eval_headings(sea, sp, headings_learned(sea, sp, pol)))
            if (e + 1) % 4 == 0:
                print(f"   ep {e+1}/{a.episodes} ({time.time()-t0:.0f}s)", flush=True)
        results['crb'][str(sp)] = {p: dict(mean=float(np.mean(crb_sp[p])), se=float(np.std(crb_sp[p], ddof=1)/np.sqrt(len(crb_sp[p])))) for p in POLICIES}
        json.dump(results, open(f'{OUT}/closed_loop_trajectory.json', 'w'), indent=2)
        line = "   field-CRB: " + "  ".join(f"{p}={results['crb'][str(sp)][p]['mean']:.3f}" for p in POLICIES)
        print(line, flush=True)
        # time the learned policy (one full trajectory)
        if results['learned_time_ms'] is None:
            sea = make_sea(99999); t1 = time.time()
            for _ in range(10): headings_learned(sea, sp, pol)
            results['learned_time_ms'] = (time.time() - t1) * 100

    # ---- Stage C: achievability (actual RELAX reconstruction) on key policies, headline speed ----
    print(f"\n=== achievability (RELAX nRMSE) at {SPEEDS_FULL[-1]} m/s ===", flush=True)
    sp = SPEEDS_FULL[-1]; pol = train_policy(sp, a.train_eps)
    for p in ['stationary', 'straight-along', 'straight-cross', 'greedy', 'CEM', 'learned']:
        vals = []
        for e in range(min(a.episodes, 8)):
            sea = make_sea(20000 + e)
            spd = 0.0 if p == 'stationary' else sp
            h = headings_straight(sea, 90) if p == 'stationary' else headings_for(p, sea, sp, pol)
            vals.append(achieved_nrmse(sea, spd, h, seed=e))
        results['achieved'][p] = dict(mean=float(np.mean(vals)), se=float(np.std(vals, ddof=1)/np.sqrt(len(vals))))
        print(f"   {p:16s} nRMSE {results['achieved'][p]['mean']:.3f}", flush=True)
    json.dump(results, open(f'{OUT}/closed_loop_trajectory.json', 'w'), indent=2)

    write_verdict(results); make_fig(results)
    print(f"\nDONE ({time.time()-t0:.0f}s). saved {OUT}/closed_loop_trajectory.json", flush=True)
    return results


def write_verdict(r):
    sp = str(SPEEDS_FULL[-1]); c = r['crb'][sp]; ach = r.get('achieved', {})
    def g(d, p): return d.get(p, {}).get('mean', float('nan'))
    lines = []
    lines.append(f"# VERDICT — closed-loop informative-trajectory control (2026-06-30)\n")
    lines.append("Closed-loop platform-heading control to minimize field-reconstruction CRB; learned")
    lines.append("non-myopic policy (imitating CEM) vs baselines. Froude-scaled (full 2-8 m/s ↔ tank 0.4-1.6).\n")
    lines.append(f"## Field-CRB by policy (headline speed {sp} m/s, lower=better)")
    for p in POLICIES:
        lines.append(f"- {p:16s}: {g(c,p):.3f} ± {c.get(p,{}).get('se',0):.3f}")
    lines.append("")
    if ach:
        lines.append("## Achieved reconstruction nRMSE (RELAX, oracle wavenumbers)")
        for p in POLICIES:
            lines.append(f"- {p:16s}: {g(ach,p):.3f}")
        lines.append("")
    # honest readout
    stat, cross, greedy, cem, learned = g(c,'stationary'), g(c,'straight-cross'), g(c,'greedy'), g(c,'CEM'), g(c,'learned')
    lines.append("## Honest readout")
    lines.append(f"- (A) achievability: trajectory control vs stationary — cross {cross:.3f} vs stationary {stat:.3f} "
                 f"({'IMPROVES' if cross < stat*0.95 else '≈/✗'}).")
    lines.append(f"- (B) non-myopic headroom: CEM {cem:.3f} vs greedy {greedy:.3f} "
                 f"({'CEM better → learning has a target' if cem < greedy*0.97 else '≈ greedy → little non-myopic headroom'}).")
    lines.append(f"- (C) learned: {learned:.3f} vs CEM {cem:.3f} / greedy {greedy:.3f}; "
                 f"inference {r.get('learned_time_ms',0):.1f} ms/trajectory "
                 f"({'matches/approaches planner fast' if learned < greedy*1.03 else 'underperforms — honest'}).")
    lines.append(f"- surfing check: straight-along {g(c,'straight-along'):.3f} vs straight-cross {cross:.3f} "
                 f"(along should be worse).")
    open(f'{OUT}/closed_loop_trajectory_VERDICT.md', 'w').write("\n".join(lines))
    print(f"saved {OUT}/closed_loop_trajectory_VERDICT.md")


def make_fig(r):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))
    speeds = [float(s) for s in r['crb']]
    for p, col in zip(POLICIES, ['gray', 'tab:red', 'tab:blue', 'tab:orange', 'tab:purple', 'tab:green']):
        a1.plot(speeds, [r['crb'][str(s)][p]['mean'] for s in speeds], 'o-', color=col, label=p)
    a1.set_xlabel('platform speed [m/s]'); a1.set_ylabel('field-reconstruction CRB (nRMSE)')
    a1.set_title('Closed-loop trajectory control: field-CRB by policy'); a1.legend(fontsize=8); a1.grid(alpha=0.3)
    if r.get('achieved'):
        ps = [p for p in POLICIES if p in r['achieved']]
        a2.bar(range(len(ps)), [r['achieved'][p]['mean'] for p in ps],
               color=['gray', 'tab:red', 'tab:blue', 'tab:orange', 'tab:purple', 'tab:green'][:len(ps)])
        a2.set_xticks(range(len(ps))); a2.set_xticklabels(ps, rotation=30, ha='right', fontsize=8)
        a2.set_ylabel('achieved nRMSE'); a2.set_title('Achievability (RELAX reconstruction)')
        a2.grid(alpha=0.3, axis='y')
    fig.suptitle('Learned informative-trajectory control for synthetic-aperture wave estimation', fontsize=11)
    fig.tight_layout(); fig.savefig(f'{OUT}/fig_closed_loop_trajectory.png', dpi=140, bbox_inches='tight')
    plt.close(fig); print(f"saved {OUT}/fig_closed_loop_trajectory.png")


if __name__ == '__main__':
    main()
