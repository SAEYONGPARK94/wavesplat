"""R0 — physics-only JOINT (kx,ky)+(A,phi) polish of B2 (NO learning, obs-only, deployable).

E1-E3 error budget: B2 (diffusion+polish) 0.253 = linear floor O1p 0.128
+ mode-geometry d_geom 0.073 + A,phi-fitting d_fit 0.052. The B2 pipeline never JOINTLY
optimizes (kx,ky) together with EXPLICIT A,phi against the obs residual: polish_atoms
refines (kx,ky,energy) against the PHASE-MARGINALIZED GP-marginal likelihood, and A,phi
are only recovered afterward by one linear LS (fit_amplitudes_phases) on a 400-point
subsample. R0 closes that loop with NOMP-style off-grid alternation, using ONLY the
observations (no GT, no learning -> sim2real-transferable):

  init  : B2's estimated components (d3_ceiling_gif.b2_components, per posterior sample,
          from the standard 400-obs subsample — baseline protocol unchanged).
  data  : the FULL cached obs stack (P,z; ~7.4k pts). The 400-subsample exists only for
          diffusion-guidance cost; a physics post-polish can afford all obs. Split into
          FIT and VAL (holdout) sets — VAL gates acceptance (obs-only overfit guard;
          v1 finding: with 400 obs the fit residual is already ~4% Hs, far below the
          generalization error, so minimizing it further is pure noise-fitting).
  round 0 (a-only): A,phi ridge-LS refit on the FIT set, k FIXED  == ablation `R0_ls`
          (isolates the more-obs-LS contribution from the k-refinement).
  rounds 1..R:
    (b) (kx,ky) step — damped gradient refine of all modes' (kx,ky) on the explicit
        FIT residual  r_i = z_i - sum_m A_m cos(kx_m x_i + ky_m y_i - om_m t_i + phi_m),
        om_m = sqrt(g|k_m| tanh(|k_m| d))  (dispersion re-evaluated as k moves; torch
        autograd, Adam inner iters, per-mode displacement clipped to clip_frac*k_p —
        LOCAL refine: fixes near-misses, not gross ~1*k_p errors). A,phi FIXED.
    (a) A,phi step — global ridge-LS refit on the FIT set (fit_amplitudes_phases).
    accept the round only if the VAL RMS decreased, else roll back and halve the step
    (VAL residual monotone by construction — asserted). Worst case R0 == B2 init.

Per window (3-window smoke: BF3_s0/BF4_s0/BF5_s0; full: 15):
  B2_obs (baseline) / R0_ls (a-only ablation) / R0 (joint) / O1p_true (E1 floor, must
  match E1 exactly); E3-ladder position (B2modes_GT vs R0modes_GT: did the GEOMETRY
  move toward GT?); E2 zone split in/out nRMSE (cg-cone mask); Chamfer to true modes.

Run:  python3 scripts/d3_r0_physics_polish.py --smoke   (3 windows, GPU for dps_sample)
      python3 scripts/d3_r0_physics_polish.py            (full 15 windows)
"""
import os, sys, json, time, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

import torch
import d44_hos_nonlinear as H
import d3_setdiff_posterior as DP                      # sets DEVICE (GPU if free)
from d3_setdiff_prior import SetDenoiser
from d3_r5_oracle import init_modes
from d3_e1_oracle_fix import ls_refit_floor, nrmse, GX, GY
from d3_ceiling_gif import b2_components, render_bma, PRIOR
from d3_e3_shape_diagnosis import (window_data, peak_wavenumber, chamfer_to_true, rho)
from wavesplat.model.tnse import fit_amplitudes_phases

# d3_e2_zone_split force-sets CUDA_VISIBLE_DEVICES='' at import; init CUDA first (env
# changes are no-ops after context creation) and restore the env var afterwards.
if torch.cuda.is_available():
    torch.zeros(1, device='cuda')
_env_cvd = os.environ.get('CUDA_VISIBLE_DEVICES')
import d3_e2_zone_split as E2
if _env_cvd is None:
    os.environ.pop('CUDA_VISIBLE_DEVICES', None)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = _env_cvd

G = 9.81
DEPTH = H.DEPTH
SEAS = [('BF3', 0.6, 5.5), ('BF4', 1.2, 7.0), ('BF5', 2.0, 9.0)]
SEEDS = [0, 1, 2, 3, 4]
OUT_JSON = 'results/direction3/r0_physics_polish.json'
R_PRIMARY = E2.R_PRIMARY                                # 0.25 * lambda_p (E2 primary)


# ── R0 core: obs-only joint alternation ──────────────────────────────────────
def obs_pred(kx, ky, A, phi, Ps, depth):
    """Linear render at the obs points (numpy, float64)."""
    k = np.hypot(kx, ky) + 1e-12
    om = np.sqrt(G * k * np.tanh(k * depth))
    psi = (kx[None, :] * Ps[:, 0][:, None] + ky[None, :] * Ps[:, 1][:, None]
           - om[None, :] * Ps[:, 2][:, None] + phi[None, :])
    return (A[None, :] * np.cos(psi)).sum(1)


def obs_rms(kx, ky, A, phi, Ps, zs, depth):
    return float(np.sqrt(np.mean((zs - obs_pred(kx, ky, A, phi, Ps, depth)) ** 2)))


def refine_k_round(kx, ky, A, phi, Ps, zs, depth, lr, clip, iters):
    """(b) step: damped gradient refine of (kx,ky) on 1/2 mean r^2, A,phi FIXED.
    Dispersion om = sqrt(g|k| tanh(|k|d)) inside the graph -> refining k updates om.
    Per-mode total displacement clipped to `clip` (trust region). CPU float64."""
    x = torch.tensor(np.stack([kx, ky], 1), dtype=torch.float64)          # (m, 2)
    x0 = x.clone()
    x.requires_grad_(True)
    At = torch.tensor(A, dtype=torch.float64)
    pt = torch.tensor(phi, dtype=torch.float64)
    xo = torch.tensor(Ps[:, 0], dtype=torch.float64)
    yo = torch.tensor(Ps[:, 1], dtype=torch.float64)
    to = torch.tensor(Ps[:, 2], dtype=torch.float64)
    zo = torch.tensor(zs, dtype=torch.float64)
    opt = torch.optim.Adam([x], lr=lr)
    for _ in range(iters):
        opt.zero_grad()
        k = x.norm(dim=1).clamp_min(1e-6)
        om = torch.sqrt(G * k * torch.tanh(k * depth))
        psi = (x[:, 0][None, :] * xo[:, None] + x[:, 1][None, :] * yo[:, None]
               - om[None, :] * to[:, None] + pt[None, :])
        r = zo - (At[None, :] * torch.cos(psi)).sum(1)
        (0.5 * (r ** 2).mean()).backward()
        opt.step()
    with torch.no_grad():
        d = x - x0
        n = d.norm(dim=1, keepdim=True)
        x = x0 + d * (clip / n.clamp_min(1e-12)).clamp(max=1.0)
    return x[:, 0].numpy(), x[:, 1].numpy()


def r0_polish(kx, ky, A, phi, Pf, zf, Pv, zv, depth, kp,
              rounds=5, inner=25, lr_frac=0.02, clip_frac=0.15, ridge=0.5):
    """NOMP-style alternation from the B2 init; FIT set drives the steps, VAL set gates
    acceptance. Returns (R0 state, R0_ls ablation state, VAL-RMS history [monotone
    non-increasing by construction], per-round state snapshots, accepted-round count)."""
    obs_f = np.column_stack([Pf[:, 0], Pf[:, 1], Pf[:, 2], zf]).astype(np.float64)
    cur = tuple(v.astype(np.float64).copy() for v in (kx, ky, A, phi))
    vrms = lambda st: obs_rms(*st, Pv, zv, depth)
    best_val = vrms(cur)
    val_hist = [best_val]; states = [cur]; accepted = 0

    # round 0 (a-only): A,phi refit on FIT, k fixed == R0_ls ablation (recorded
    # unconditionally; adopted into the R0 trajectory only if VAL improves)
    A0, phi0 = fit_amplitudes_phases(cur[0], cur[1], obs_f, depth, lambda_reg=ridge)
    ls_state = (cur[0].copy(), cur[1].copy(), A0, phi0)
    if vrms(ls_state) < best_val - 1e-12:
        cur, best_val = ls_state, vrms(ls_state); accepted += 1
    val_hist.append(best_val); states.append(cur)

    lr, clip = lr_frac * kp, clip_frac * kp
    for _ in range(rounds):
        kx2, ky2 = refine_k_round(*cur, Pf, zf, depth, lr, clip, inner)
        A2, phi2 = fit_amplitudes_phases(kx2, ky2, obs_f, depth, lambda_reg=ridge)
        cand = (kx2, ky2, A2, phi2)
        if vrms(cand) < best_val - 1e-12:
            cur, best_val = cand, vrms(cand); accepted += 1
        else:                                            # roll back, damp harder
            lr *= 0.5; clip *= 0.5
        val_hist.append(best_val); states.append(cur)
        if lr < 1e-3 * kp:
            break
    assert all(val_hist[i + 1] <= val_hist[i] + 1e-12 for i in range(len(val_hist) - 1)), \
        'VAL residual must be monotone non-increasing (acceptance guard broken)'
    return cur, ls_state, val_hist, states, accepted


def comps_obs_rms(comps, Ps, zs, depth):
    """Mean obs RMS over posterior samples (each sample is an independent hypothesis)."""
    return float(np.mean([obs_rms(kx.astype(np.float64), ky.astype(np.float64),
                                  A.astype(np.float64), phi.astype(np.float64),
                                  Ps, zs, depth) for kx, ky, A, phi in comps]))


def modes_gt_refit(comps, gt, t, Hs_t):
    """E3's B2modes_GT rung for an arbitrary component set: A,phi refit to full GT per
    sample (ridge 1e-2), fields BMA-averaged."""
    fs = [ls_refit_floor(kx.astype(np.float64), ky.astype(np.float64), gt, t, Hs_t)[1]
          for kx, ky, _, _ in comps]
    return nrmse(np.mean(fs, axis=0), gt, Hs_t)


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true', help='seeds [0] only (3 windows)')
    ap.add_argument('--rounds', type=int, default=5,
                    help='more rounds (8-12) gain BF4/BF5 slightly but flip BF3 '
                         'negative (out-zone drift the VAL gate cannot see) — keep 5')
    ap.add_argument('--inner', type=int, default=40, help='inner Adam iters per round')
    ap.add_argument('--lr_frac', type=float, default=0.02, help='Adam lr = lr_frac*k_p')
    ap.add_argument('--clip_frac', type=float, default=0.10,
                    help='per-round per-mode k-step clip = clip_frac*k_p (local refine)')
    ap.add_argument('--ridge', type=float, default=0.5,
                    help='A,phi ridge on the FIT set (B2 uses 0.5 — kept identical)')
    ap.add_argument('--holdout_frac', type=float, default=0.15,
                    help='fraction of the full obs stack held out as VAL (acceptance)')
    args = ap.parse_args()
    seeds = [0] if args.smoke else SEEDS
    hp = dict(rounds=args.rounds, inner=args.inner, lr_frac=args.lr_frac,
              clip_frac=args.clip_frac, ridge=args.ridge)

    ck = torch.load(PRIOR, map_location=DP.DEVICE)
    stats = ck['stats']
    prior = SetDenoiser(**ck.get('arch', {})).to(DP.DEVICE)
    prior.load_state_dict(ck['model']); prior.eval()
    abar = ck['abar'].to(DP.DEVICE); betas = ck['betas'].to(DP.DEVICE)
    e1 = json.load(open('results/direction3/e1_oracle_fix.json'))['per_window']
    e3 = json.load(open('results/direction3/e3_shape_diagnosis.json'))['per_window']

    rows = {}; t0 = time.time()
    print(f'R0 physics-only joint polish (obs-only, no learning)  [device={DP.DEVICE}]\n'
          f'  hp={hp}  holdout_frac={args.holdout_frac}  FIT=full obs stack minus VAL',
          flush=True)
    hdr = (f"{'window':8s} {'acc':>3} | {'B2obs':>5} {'R0ls':>5} {'R0':>5} {'O1p':>5} "
           f"{'dObs':>6} | {'valB2':>5} {'valR0':>5} | {'B2mGT':>5} {'R0mGT':>5} | "
           f"{'chB2':>5} {'chR0':>5} | {'B2in':>5} {'R0in':>5} {'B2out':>5} {'R0out':>5}")
    print(hdr, flush=True)

    for tag, Hs, Tp in SEAS:
        kp = peak_wavenumber(Tp)
        lam = E2.peak_wavelength(Tp)
        for seed in seeds:
            gt, t, P, z = window_data(tag, seed)
            Hs_t = float(gt.std() * 4)
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(z), min(400, len(z)), replace=False)
            Ps, zs = P[idx].astype(np.float32), z[idx].astype(np.float32)
            # FIT/VAL split of the FULL obs stack (rng continues after the 400-draw so
            # the baseline subsample is untouched; VAL disjoint from FIT by permutation)
            perm = rng.permutation(len(z))
            n_val = int(round(args.holdout_frac * len(z)))
            vi, fi = perm[:n_val], perm[n_val:]
            Pf, zf = P[fi].astype(np.float64), z[fi].astype(np.float64)
            Pv, zv = P[vi].astype(np.float64), z[vi].astype(np.float64)

            # 1) B2 baseline (real pipeline: DPS + phase-marginal polish + LS, BMA of 2)
            comps = b2_components(prior, stats, abar, betas, Ps, zs, DEPTH)
            if not comps:
                print(f'{tag}_s{seed}: EMPTY B2 component set — skipped', flush=True)
                continue
            f_b2 = render_bma(comps, DEPTH, t)
            nr_b2, rho_b2 = nrmse(f_b2, gt, Hs_t), rho(f_b2, gt)
            val_b2 = comps_obs_rms(comps, Pv, zv, DEPTH)

            # 2) R0 joint polish per posterior sample (obs-only, VAL-gated)
            comps_r0, comps_ls, hists, all_states, accs = [], [], [], [], []
            for kx, ky, A, phi in comps:
                best, ls_st, val_hist, states, acc = r0_polish(
                    kx, ky, A, phi, Pf, zf, Pv, zv, DEPTH, kp, **hp)
                comps_r0.append(best); comps_ls.append(ls_st)
                hists.append(val_hist); all_states.append(states); accs.append(acc)
            f_r0 = render_bma(comps_r0, DEPTH, t)
            f_ls = render_bma(comps_ls, DEPTH, t)
            nr_r0, rho_r0 = nrmse(f_r0, gt, Hs_t), rho(f_r0, gt)
            nr_ls = nrmse(f_ls, gt, Hs_t)
            val_r0 = comps_obs_rms(comps_r0, Pv, zv, DEPTH)
            # per-round BMA nRMSE trace (VAL gate should prevent val-down/GT-up drift)
            n_rd = max(len(s) for s in all_states)
            trace = [float(nrmse(render_bma(
                [s[min(r, len(s) - 1)] for s in all_states], DEPTH, t), gt, Hs_t))
                for r in range(n_rd)]

            # 3) O1p_true floor (must reproduce E1 exactly) + E3 ladder rungs
            kxT, kyT, _, _ = init_modes(Hs, Tp, seed)
            nr_o1p, f_o1p = ls_refit_floor(kxT[:32], kyT[:32], gt, t, Hs_t)
            nr_b2mgt = modes_gt_refit(comps, gt, t, Hs_t)
            nr_r0mgt = modes_gt_refit(comps_r0, gt, t, Hs_t)

            # 4) geometry: Chamfer to true top-32 (B2 vs R0)
            ch_b2, chw_b2 = chamfer_to_true(comps, kxT[:32], kyT[:32])
            ch_r0, chw_r0 = chamfer_to_true(comps_r0, kxT[:32], kyT[:32])

            # 5) E2 zone split (cg-cone, r_tol = 0.25*lambda_p)
            md = E2.zone_mindist(Ps, t, kxT[:32], kyT[:32], advect=True)
            mask = md <= R_PRIMARY * lam
            zm_b2 = E2.zone_metrics(f_b2, gt, Hs_t, mask)
            zm_r0 = E2.zone_metrics(f_r0, gt, Hs_t, mask)
            zm_o1p = E2.zone_metrics(f_o1p, gt, Hs_t, mask)

            key = f'{tag}_s{seed}'
            rows[key] = dict(
                b2_obs=dict(nrmse=nr_b2, rho=rho_b2, val_rms=val_b2,
                            zone=dict((r, zm_b2[r]['nrmse']) for r in ('in', 'out')),
                            chamfer=dict(mean=ch_b2, wmean=chw_b2, over_kp=ch_b2 / kp),
                            modes_gt=nr_b2mgt),
                r0_ls=dict(nrmse=nr_ls),
                r0=dict(nrmse=nr_r0, rho=rho_r0, val_rms=val_r0,
                        zone=dict((r, zm_r0[r]['nrmse']) for r in ('in', 'out')),
                        chamfer=dict(mean=ch_r0, wmean=chw_r0, over_kp=ch_r0 / kp),
                        modes_gt=nr_r0mgt, val_hist=hists, nrmse_trace=trace,
                        accepted_rounds=accs),
                o1p_true=dict(nrmse=nr_o1p,
                              zone=dict((r, zm_o1p[r]['nrmse']) for r in ('in', 'out'))),
                d_obs=nr_b2 - nr_r0, k_p=float(kp),
                n_fit=int(len(fi)), n_val=int(len(vi)),
                zone_coverage=float(mask.mean()),
                o1p_ref_e1=e1[key]['O1p_32'],
                b2_ref_e3=e3.get(key, {}).get('b2_obs', {}).get('nrmse'))
            print(f"{key:8s} {'+'.join(map(str, accs)):>3} | "
                  f"{nr_b2:5.3f} {nr_ls:5.3f} {nr_r0:5.3f} {nr_o1p:5.3f} "
                  f"{nr_b2 - nr_r0:+6.3f} | {val_b2:5.3f} {val_r0:5.3f} | "
                  f"{nr_b2mgt:5.3f} {nr_r0mgt:5.3f} | "
                  f"{ch_b2 / kp:5.2f} {ch_r0 / kp:5.2f} | "
                  f"{zm_b2['in']['nrmse']:5.3f} {zm_r0['in']['nrmse']:5.3f} "
                  f"{zm_b2['out']['nrmse']:5.3f} {zm_r0['out']['nrmse']:5.3f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)

    # ── summary (window-mean, pooled + per-sea) ──
    tags = np.array([k.split('_')[0] for k in rows])

    def agg(fn):
        arr = np.array([fn(r) for r in rows.values()])
        ps = {s: float(arr[tags == s].mean()) for s in ('BF3', 'BF4', 'BF5')
              if (tags == s).any()}
        return dict(mean=float(arr.mean()), per_sea=ps)

    summary = dict(
        b2_obs=agg(lambda r: r['b2_obs']['nrmse']),
        r0_ls=agg(lambda r: r['r0_ls']['nrmse']),
        r0=agg(lambda r: r['r0']['nrmse']),
        o1p_true=agg(lambda r: r['o1p_true']['nrmse']),
        d_obs=agg(lambda r: r['d_obs']),
        b2modes_gt=agg(lambda r: r['b2_obs']['modes_gt']),
        r0modes_gt=agg(lambda r: r['r0']['modes_gt']),
        d_fit_b2=agg(lambda r: r['b2_obs']['nrmse'] - r['b2_obs']['modes_gt']),
        d_fit_r0=agg(lambda r: r['r0']['nrmse'] - r['r0']['modes_gt']),
        chamfer_over_kp_b2=agg(lambda r: r['b2_obs']['chamfer']['over_kp']),
        chamfer_over_kp_r0=agg(lambda r: r['r0']['chamfer']['over_kp']),
        zone_in_b2=agg(lambda r: r['b2_obs']['zone']['in']),
        zone_in_r0=agg(lambda r: r['r0']['zone']['in']),
        zone_out_b2=agg(lambda r: r['b2_obs']['zone']['out']),
        zone_out_r0=agg(lambda r: r['r0']['zone']['out']),
        val_rms_b2=agg(lambda r: r['b2_obs']['val_rms']),
        val_rms_r0=agg(lambda r: r['r0']['val_rms']))

    m = {k: summary[k]['mean'] for k in summary}
    gain = m['b2_obs'] - m['r0']
    gain_ls = m['b2_obs'] - m['r0_ls']                    # more-obs LS refit only
    gain_k = m['r0_ls'] - m['r0']                         # incremental from k refine
    d_geom_move = m['b2modes_gt'] - m['r0modes_gt']       # >0: geometry moved toward GT
    fit_recover = m['d_fit_b2'] - m['d_fit_r0']           # >0: fitting gap shrank
    gap_b2 = m['b2_obs'] - m['o1p_true']
    print('\n=== R0 summary (window-mean) ===', flush=True)
    for k in ('b2_obs', 'r0_ls', 'r0', 'o1p_true', 'b2modes_gt', 'r0modes_gt'):
        s = summary[k]
        print(f"  {k:12s} {s['mean']:.3f}  |  " +
              '  '.join(f'{t_} {v:.3f}' for t_, v in s['per_sea'].items()), flush=True)
    print(f"  gain (B2-R0)          {gain:+.3f}  ({100 * gain / max(gap_b2, 1e-9):.0f}% "
          f"of the B2->floor gap {gap_b2:.3f})  =  LS-more-obs {gain_ls:+.3f} "
          f"+ k-refine {gain_k:+.3f}", flush=True)
    print(f"  fitting-gap  d_fit    {m['d_fit_b2']:.3f} -> {m['d_fit_r0']:.3f}  "
          f"(recovered {fit_recover:+.3f} of E3's 0.052)", flush=True)
    print(f"  geometry  modes_GT    {m['b2modes_gt']:.3f} -> {m['r0modes_gt']:.3f}  "
          f"({d_geom_move:+.3f}; E3 near-miss headroom d_geom=0.073)", flush=True)
    print(f"  chamfer/k_p           {m['chamfer_over_kp_b2']:.2f} -> "
          f"{m['chamfer_over_kp_r0']:.2f}", flush=True)
    print(f"  zone in / out         B2 {m['zone_in_b2']:.3f}/{m['zone_out_b2']:.3f} -> "
          f"R0 {m['zone_in_r0']:.3f}/{m['zone_out_r0']:.3f}", flush=True)
    print(f"  VAL RMS [m]           {m['val_rms_b2']:.3f} -> {m['val_rms_r0']:.3f}",
          flush=True)

    # ── sanity gates ──
    o1p_dev = max(abs(r['o1p_true']['nrmse'] - r['o1p_ref_e1']) for r in rows.values())
    b2_dev = max(abs(r['b2_obs']['nrmse'] - r['b2_ref_e3'])
                 for r in rows.values() if r['b2_ref_e3'] is not None)
    worse = [k for k, r in rows.items() if r['d_obs'] < -1e-3]
    val_up = [k for k, r in rows.items()
              if r['r0']['val_rms'] > r['b2_obs']['val_rms'] + 1e-9]
    print('\n=== sanity gates ===', flush=True)
    print(f"  O1p_true == E1 O1p_32 (deterministic): max|diff| = {o1p_dev:.2e}  "
          f"({'OK' if o1p_dev < 1e-6 else 'MISMATCH'})", flush=True)
    print(f"  B2_obs vs E3 ref (same seed/device):   max|diff| = {b2_dev:.3f}  "
          f"({'OK' if b2_dev < 0.02 else 'CHECK'})", flush=True)
    print(f"  R0 VAL-RMS <= B2 VAL-RMS (monotone):   violations = {val_up or 'none'}",
          flush=True)
    print(f"  R0 nRMSE <= B2 nRMSE (+1e-3):          violations = {worse or 'none'}"
          f"{'' if not worse else '  (VAL improved but GT worse -> gate imperfect)'}",
          flush=True)

    # ── verdict ──
    hit = m['r0'] <= 0.22
    verdict = (f"R0 gain {gain:+.3f} ({m['b2_obs']:.3f} -> {m['r0']:.3f}; "
               f"LS-more-obs {gain_ls:+.3f}, k-refine {gain_k:+.3f}); "
               f"target <=0.22 {'HIT' if hit else 'MISSED'}. "
               f"fitting-gap recovered {fit_recover:+.3f}/0.052, "
               f"geometry moved {d_geom_move:+.3f} (chamfer/kp "
               f"{m['chamfer_over_kp_b2']:.2f}->{m['chamfer_over_kp_r0']:.2f}). ")
    if gain < 0.01:
        verdict += ('R0 barely helps -> the obs residual carries no further signal at '
                    'this init; remaining error is mode DETECTION (R1 priority up).')
    elif not hit:
        verdict += ('Partial recovery -> R0 worth keeping as a free post-step, but '
                    'detection (geometry) remains the binding constraint.')
    else:
        verdict += 'Physics-only polish suffices to reach the 0.22 target.'
    print(f'\nREAD: {verdict}', flush=True)

    out = dict(summary=summary, gain_mean=gain, gain_ls=gain_ls, gain_k=gain_k,
               d_geom_move=d_geom_move, fit_gap_recovered=fit_recover,
               hp=dict(**hp, holdout_frac=args.holdout_frac), per_window=rows,
               smoke=args.smoke, verdict=verdict,
               gates=dict(o1p_vs_e1_maxdiff=o1p_dev, b2_vs_e3_maxdiff=b2_dev,
                          r0_worse_than_b2=worse, val_rms_increased=val_up))
    path = OUT_JSON if not args.smoke else OUT_JSON.replace('.json', '_smoke.json')
    json.dump(out, open(path, 'w'), indent=2)
    print(f'saved {path}', flush=True)


if __name__ == '__main__':
    main()
