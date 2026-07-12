"""
d3_d18_ctrl_tune.py -- D-18: controller tuning + cause decomposition after
the D-17 FAIL ("frozen-forecast FF loses to persistence").

Pre-registration: docs/D17_CONTROL_prereg_2026-07-12.md, "D-18 spec" clause
(frozen 2026-07-12). Executes the four frozen items.

WHY THE ACTUATOR-DELAY MODEL (design note, read this):
A smoke test of the literal "tau = FeedforwardConfig.preview_dt" reading found
even the GT ceiling degrades with tau (GT z 0.187->0.229 as tau 0.5->3): with
NO plant lag in the sim, a larger spatial preview just adds error to everyone,
so no tau regime can favour forecast (degenerate FAIL). The spec's stated
expectation -- "persistence collapses at large tau while forecast-FF holds"
plus "true delay simulation" -- requires a REAL delay. So tau is modelled as an
ACTUATOR INFORMATION DELAY: the flap command ACTIVE at time t was ISSUED at
t-tau and must cancel the disturbance active at t, i.e. the wave at the foil's
CURRENT position (x_cg(t)+x_foil, t). Then:
  GT           perfect tau-ahead forecast -> exact current-cancel; tau-flat CEILING
  persistence  hold the wave it last measured tau ago:
                 wave at (x_cg(t)-U*tau + x_foil, t - tau)  ["current wave from tau ago"]
  hand-frozen  ONE estimate (obs [-20,0]) predicting the current disturbance
  hand-stream  re-estimate every REEST_PERIOD s from the trailing 20 s window,
                 issue time = t-tau -> obs [t-tau-20, t-tau]
This is the textbook forecast-value setup: persistence degrades with the
horizon tau; a good forecast holds; the streaming fan tracks the ship so the
current disturbance stays inside the observed region (fixes D-17 caveat 1).

(1) streaming re-estimation arm (hand/mmse_gpu, GPU). learned arm SKIPPED
    (CPU ~2.4 s/call; 30 re-est x cells prohibitive) -- reported.
(2) tau sweep {0.5,1,2,3} s as the actuator delay above.
(3) lp_alpha {0.05,0.2,1.0}: Stage A reproduces the 07-06 GT-FF preview sweep
    (shipped default provider) to confirm the controller's lp optimum, then
    fixes best lp for Stage B.
(4) UQ-gating: only on a (tau) where streaming beats persistence; else skipped.

Metric: residual heave RMS over t>5 s (z primary, accel secondary).
recovery_z = (persist - arm)/(persist - GT).

Gate G-D18-1: streaming z RMS < persistence z RMS at some tau* -> PASS
(scope "preview-forecast FF has control value for actuator lead tau >= tau*").
Loss at every tau -> FAIL.

Usage: python3 scripts/d3_d18_ctrl_tune.py [--seeds 2] [--reest 1.0]
"""
import os
import sys
import json
import time
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from wavesim.wave_generator import jonswap_components
from hydrofoil.config import candela_c8
from hydrofoil.controller import (FeedforwardConfig, PDControllerConfig,
                                  FeedforwardWaveInfo)
from hydrofoil.sim import run_simulation
from hydrofoil.disturbance import (wave_slope_x_at_point,
                                   wave_vertical_velocity_at_point)
from d3_sa_fim_identifiability import omega_of
from d3_gpu_mmse import mmse_gpu

G = 9.81
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT = 'results/direction3/d18_ctrl_tune.json'

# BF3/BF4 = d3_e2e_stream_ff.SEAS convention (D-17 coverage). BF5 added.
SEAS = {'BF3': dict(Hs=0.88, Tp=5.0),
        'BF4': dict(Hs=1.88, Tp=7.5),
        'BF5': dict(Hs=3.00, Tp=9.0)}

TAUS = (0.5, 1.0, 2.0, 3.0)
ALPHAS = (0.05, 0.2, 1.0)
OBS_NOISE = 0.02          # Falcon 2 cm 1-sigma elevation noise [m]
K_FF = 0.8
EPS_X = 0.5               # spatial FD step for slope [m]
EPS_T = 0.05              # temporal FD step for vertical velocity [s]
WARM = 20.0              # observation window length [s]


# ---------------------------------------------------------------- observations
def collect_obs_window(comps, U, depth, t_start, t_end, seed_pos, n_obs=600):
    """Forward-fan LiDAR obs over [t_start, t_end]; geometry identical to
    d3_d17_ctrl_closedloop.collect_obs (ship +x at U, fan +x ahead r 20-100 m,
    +/-60 deg, head seas, +2 cm noise) so the fan tracks the ship forward."""
    kx = np.array([c.kx for c in comps]); ky = np.array([c.ky for c in comps])
    A = np.array([c.A for c in comps]); ph = np.array([c.phi for c in comps])
    om = omega_of(kx, ky, depth)

    def eta(x, y, t):
        ps = (kx[None] * np.atleast_1d(x)[:, None]
              + ky[None] * np.atleast_1d(y)[:, None]
              - om[None] * np.atleast_1d(t)[:, None] + ph[None])
        return (A[None] * np.cos(ps)).sum(1)

    rng = np.random.default_rng(seed_pos)
    dt, m = 0.1, 12
    Pw, zw = [], []
    for t in np.arange(t_start, t_end, dt):
        sx = U * t
        rr = rng.uniform(20.0, 100.0, m)
        aa = rng.uniform(-np.pi / 3, np.pi / 3, m)
        xs = sx + rr * np.cos(aa)
        ys = rr * np.sin(aa)
        Pw.append(np.column_stack([xs, ys, np.full(m, t)]))
        zw.append(eta(xs, ys, t) + rng.normal(0, OBS_NOISE, m))
    Pw, zw = np.vstack(Pw), np.concatenate(zw)
    ii = rng.choice(len(zw), min(n_obs, len(zw)), False)
    return Pw[ii], zw[ii]


# ---------------------------------------------------------------- FF injection
def build_query_current(cfg, n_steps, dt, U):
    """Query points at the foil's CURRENT position (x_cg + x_foil, t) -- the
    disturbance the FF active at t must cancel. tau-INDEPENDENT. For each step
    and foil: 4 points (x+-EPS_X, t+-EPS_T). Block order
    [F x+,x-,t+,t- , R x+,x-,t+,t-]."""
    front, rear = cfg['front_foil'], cfg['rear_foil']
    t_arr = np.arange(n_steps) * dt
    x_cg = U * t_arr
    blocks = []
    for foil in (front, rear):
        xf = x_cg + foil.x_pos
        z0 = np.zeros(n_steps)
        blocks.append(np.column_stack([xf + EPS_X, z0, t_arr]))
        blocks.append(np.column_stack([xf - EPS_X, z0, t_arr]))
        blocks.append(np.column_stack([xf, z0, t_arr + EPS_T]))
        blocks.append(np.column_stack([xf, z0, t_arr - EPS_T]))
    return np.vstack(blocks), t_arr


def field_to_drive(eta_flat, n_steps, U):
    """(8*n_steps,) field -> slope/w per foil + raw FF drive per step.
    raw = -K*(-slope - w/U), exactly hydrofoil.sim's formula."""
    e = eta_flat.reshape(8, n_steps)
    slope_f = (e[0] - e[1]) / (2 * EPS_X); w_f = (e[2] - e[3]) / (2 * EPS_T)
    slope_r = (e[4] - e[5]) / (2 * EPS_X); w_r = (e[6] - e[7]) / (2 * EPS_T)
    raw_f = -K_FF * (-slope_f - w_f / U)
    raw_r = -K_FF * (-slope_r - w_r / U)
    return slope_f, w_f, slope_r, w_r, raw_f, raw_r


def make_provider(eta_flat, n_steps, dt, U):
    slope_f, w_f, slope_r, w_r, _, _ = field_to_drive(eta_flat, n_steps, U)

    def provider(state, ff_, rf_, sim, comps, t, hd, fc):
        i = min(max(int(round(t / dt)), 0), n_steps - 1)
        return FeedforwardWaveInfo(slope_front=float(slope_f[i]),
                                   slope_rear=float(slope_r[i]),
                                   w_front=float(w_f[i]),
                                   w_rear=float(w_r[i]))
    return provider


def make_gated_provider(eta_flat, sig_flat, n_steps, dt, U, sig_thresh):
    """UQ-gated lookup: attenuate FF drive when forecast std at that step
    exceeds sig_thresh (gain = min(1, thresh/sig))."""
    slope_f, w_f, slope_r, w_r, _, _ = field_to_drive(eta_flat, n_steps, U)
    s = sig_flat.reshape(8, n_steps)
    sig_f = 0.5 * (s[0] + s[1]); sig_r = 0.5 * (s[4] + s[5])

    def provider(state, ff_, rf_, sim, comps, t, hd, fc):
        i = min(max(int(round(t / dt)), 0), n_steps - 1)
        gf = min(1.0, sig_thresh / (sig_f[i] + 1e-9))
        gr = min(1.0, sig_thresh / (sig_r[i] + 1e-9))
        return FeedforwardWaveInfo(slope_front=float(slope_f[i] * gf),
                                   slope_rear=float(slope_r[i] * gr),
                                   w_front=float(w_f[i] * gf),
                                   w_rear=float(w_r[i] * gr))
    return provider


def make_persist_provider(comps, U, tau):
    """Persistence with actuator delay tau: the command active at t is what was
    measured tau ago -> GT wave at the foil's position tau ago (x_cg(t)-U*tau
    + x_foil) at time t - tau.  tau -> 0 recovers current-cancel; tau large
    decorrelates (temporal + spatial staleness)."""
    def provider(state, ff_, rf_, sim, comps_, t, hd, fc):
        ts = t - tau
        res = {}
        for name, foil in (('front', ff_), ('rear', rf_)):
            xf = state.x - U * tau + foil.x_pos
            res[name] = (wave_slope_x_at_point(comps, xf, 0.0, ts),
                         wave_vertical_velocity_at_point(comps, xf, 0.0, ts))
        return FeedforwardWaveInfo(slope_front=res['front'][0],
                                   slope_rear=res['rear'][0],
                                   w_front=res['front'][1],
                                   w_rear=res['rear'][1])
    return provider


def true_eta_at(comps, Q):
    kx = np.array([c.kx for c in comps]); ky = np.array([c.ky for c in comps])
    A = np.array([c.A for c in comps]); ph = np.array([c.phi for c in comps])
    om = np.array([c.omega for c in comps])
    ps = (kx[None] * Q[:, 0:1] + ky[None] * Q[:, 1:2]
          - om[None] * Q[:, 2:3] + ph[None])
    return (A[None] * np.cos(ps)).sum(1)


# ---------------------------------------------------------------- estimators
def est_hand(P, z, Qsub, depth, kp, seed, want_var=False):
    Pt = torch.tensor(P, device=DEV, dtype=torch.float32)
    zt = torch.tensor(z, device=DEV, dtype=torch.float32)
    Qt = torch.tensor(Qsub, device=DEV, dtype=torch.float32)
    if want_var:
        eta, var = mmse_gpu(Pt, zt, depth, Qt, OBS_NOISE, kp, seed=seed,
                            return_var=True)
        return (eta.detach().cpu().numpy(),
                np.sqrt(np.clip(var.detach().cpu().numpy(), 0, None)))
    eta = mmse_gpu(Pt, zt, depth, Qt, OBS_NOISE, kp, seed=seed)
    return eta.detach().cpu().numpy()


def frozen_field(comps, U, depth, kp, seed, Q):
    """One estimate from the [-20,0] warmup, predicting the current
    disturbance for the whole run (tau-independent forecast quality)."""
    P, z = collect_obs_window(comps, U, depth, -WARM, 0.0, seed_pos=seed)
    return est_hand(P, z, Q, depth, kp, seed)


def stream_field(comps, U, depth, kp, seed, Q, n_steps, dt, period, tau,
                 want_var=False, timing=None):
    """Re-estimate every `period` s. The command active at step t is issued at
    t-tau, so it uses the estimate from obs [t_iss-20, t_iss] with
    t_iss = max(0, floor((t-tau)/period)*period)."""
    Qm = Q.reshape(8, n_steps, 3)
    e = np.zeros((8, n_steps)); sg = np.zeros((8, n_steps))
    t_arr = np.arange(n_steps) * dt
    issue = np.maximum(0.0, np.floor((t_arr - tau) / period) * period)
    for t_iss in np.unique(issue):
        P, z = collect_obs_window(comps, U, depth, t_iss - WARM, t_iss,
                                  seed_pos=seed * 1000 + int(round(t_iss)))
        idx = np.where(issue == t_iss)[0]
        Qsub = Qm[:, idx, :].reshape(-1, 3)
        t0 = time.time()
        if want_var:
            eta, sig = est_hand(P, z, Qsub, depth, kp, seed, want_var=True)
            sg[:, idx] = sig.reshape(8, len(idx))
        else:
            eta = est_hand(P, z, Qsub, depth, kp, seed)
        if timing is not None:
            timing.append((time.time() - t0) * 1e3)
        e[:, idx] = eta.reshape(8, len(idx))
    if want_var:
        return e.reshape(-1), sg.reshape(-1)
    return e.reshape(-1)


# ---------------------------------------------------------------- sim helpers
def rms(x):
    return float(np.sqrt(np.mean(x ** 2)))


def run_arm(cfg, comps, ff_cfg, provider=None):
    r = run_simulation(cfg, comps, vessel_heading=0.0,
                       controller=PDControllerConfig(),
                       feedforward=ff_cfg, ff_info_provider=provider)
    mask = r.time > 5.0
    return rms(r.heave_accel[mask]), rms(r.z[mask])


def skill_vs_range(pred_raw_f, gt_raw_f, t_arr):
    out = []
    for lo, hi in [(0, 5), (5, 10), (10, 20), (20, 30)]:
        m = (t_arr >= lo) & (t_arr < hi)
        a, b = pred_raw_f[m], gt_raw_f[m]
        c = (float('nan') if a.std() < 1e-12 or b.std() < 1e-12
             else float(np.corrcoef(a, b)[0, 1]))
        out.append(dict(t_lo=lo, t_hi=hi, corr=c))
    return out


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=2)
    ap.add_argument('--reest', type=float, default=1.0)
    a = ap.parse_args()

    cfg = candela_c8()
    U = cfg['sim'].speed; depth = cfg['sim'].depth
    dt = cfg['sim'].dt
    n_steps = int(cfg['sim'].duration / dt) + 1
    t_arr = np.arange(n_steps) * dt
    period = a.reest
    ff0 = dict(K_ff=K_FF, noise_std=0.0, slope_noise_std=0.0)

    if DEV.type == 'cuda':
        _w = torch.randn(600, 3, device=DEV)
        _ = mmse_gpu(_w, torch.randn(600, device=DEV), depth,
                     _w[:8].clone(), 0.02, 0.1, seed=0)
        torch.cuda.synchronize()

    print(f"device {DEV}  seeds {a.seeds}  reest {period}s  seas {list(SEAS)}",
          flush=True)
    t_wall = time.time()

    # ---------- Stage A: GT-FF preview sweep lp_alpha x tau (07-06 repro) -----
    # Shipped default provider (spatial preview_dt = tau) to pick the lp optimum.
    # best_lp = lp maximizing its OWN best-preview reduction (07-06 picks the lp
    # at the controller's optimal margin, not averaged over bad large previews).
    stageA = []
    grid = {(al, tau): [] for al in ALPHAS for tau in TAUS}
    for sea, sc in SEAS.items():
        for seed in range(a.seeds):
            comps = jonswap_components(Hs=sc['Hs'], Tp=sc['Tp'],
                                       n_components=15, depth=depth,
                                       direction_deg=180.0,
                                       direction_spread_deg=20.0,
                                       seed=7000 + seed)
            base_z = run_arm(cfg, comps, None)[1]
            for al in ALPHAS:
                for tau in TAUS:
                    ff = FeedforwardConfig(preview_dt=tau, lp_alpha=al, **ff0)
                    z = run_arm(cfg, comps, ff)[1]
                    red = 100 * (1 - z / base_z)
                    stageA.append(dict(sea=sea, seed=seed, lp_alpha=al,
                                       tau=tau, gt_z_red_pct=red))
                    grid[(al, tau)].append(red)
    grid_mean = {k: float(np.mean(v)) for k, v in grid.items()}
    lp_best_red = {al: max(grid_mean[(al, tau)] for tau in TAUS)
                   for al in ALPHAS}
    best_lp = max(lp_best_red, key=lp_best_red.get)
    best_tau_A = max(TAUS, key=lambda tau: grid_mean[(best_lp, tau)])
    lp_mean = lp_best_red
    print("[Stage A] GT-FF preview sweep z-red%% (mean over sea/seed):",
          flush=True)
    for al in ALPHAS:
        print("   lp %4g: " % al + "  ".join(
            f"tau{tau:g} {grid_mean[(al, tau)]:+.1f}" for tau in TAUS),
              flush=True)
    print(f"   -> best_lp={best_lp:g} (best at tau={best_tau_A:g}, "
          f"{grid_mean[(best_lp, best_tau_A)]:+.1f}%)", flush=True)

    # ---------- Stage B: actuator-delay arms across tau at best_lp -----------
    rows = []
    stream_ms = []
    for sea, sc in SEAS.items():
        kp = (2 * np.pi / sc['Tp']) ** 2 / G
        for seed in range(a.seeds):
            comps = jonswap_components(Hs=sc['Hs'], Tp=sc['Tp'],
                                       n_components=15, depth=depth,
                                       direction_deg=180.0,
                                       direction_spread_deg=20.0,
                                       seed=7000 + seed)
            Q, _ = build_query_current(cfg, n_steps, dt, U)
            ffB = FeedforwardConfig(preview_dt=0.0, lp_alpha=best_lp, **ff0)

            # tau-independent arms (compute once)
            base = run_arm(cfg, comps, None)
            gt_eta = true_eta_at(comps, Q)
            _, _, _, _, gt_raw, _ = field_to_drive(gt_eta, n_steps, U)
            z_gt = run_arm(cfg, comps, ffB, make_provider(gt_eta, n_steps, dt, U))
            froz = frozen_field(comps, U, depth, kp, seed, Q)
            _, _, _, _, froz_raw, _ = field_to_drive(froz, n_steps, U)
            z_froz = run_arm(cfg, comps, ffB, make_provider(froz, n_steps, dt, U))
            skill_frozen = skill_vs_range(froz_raw, gt_raw, t_arr)

            for tau in TAUS:
                z_per = run_arm(cfg, comps, ffB,
                                make_persist_provider(comps, U, tau))
                strm = stream_field(comps, U, depth, kp, seed, Q, n_steps, dt,
                                    period, tau, timing=stream_ms)
                _, _, _, _, strm_raw, _ = field_to_drive(strm, n_steps, U)
                z_str = run_arm(cfg, comps, ffB,
                                make_provider(strm, n_steps, dt, U))

                res = dict(noFF=base, persistence=z_per, hand_frozen=z_froz,
                           hand_stream=z_str, GT=z_gt)
                bz = base[1]
                gap = res['persistence'][1] - res['GT'][1]
                row = dict(
                    sea=sea, seed=seed, tau=tau, lp_alpha=best_lp, kp=float(kp),
                    z_rms={k: res[k][1] for k in res},
                    accel_rms={k: res[k][0] for k in res},
                    z_red_pct={k: 100 * (1 - res[k][1] / bz) for k in res},
                    recovery_z={k: float((res['persistence'][1] - res[k][1])
                                         / gap) if abs(gap) > 1e-9
                                else float('nan')
                                for k in ('hand_frozen', 'hand_stream', 'GT')},
                    skill_frozen=skill_frozen,
                    skill_stream=skill_vs_range(strm_raw, gt_raw, t_arr))
                rows.append(row)
                print(f"[{sea} s{seed} tau{tau:g}] z RMS  "
                      + " ".join(f"{k[:5]} {res[k][1]:.4f}" for k in
                                 ('noFF', 'persistence', 'hand_frozen',
                                  'hand_stream', 'GT'))
                      + f" | str<per "
                      f"{res['hand_stream'][1] < res['persistence'][1]}",
                      flush=True)

    # ---------- aggregate + gate --------------------------------------------
    ARMS = ('noFF', 'persistence', 'hand_frozen', 'hand_stream', 'GT')

    def agg(field, arm, tau):
        return float(np.mean([r[field][arm] for r in rows if r['tau'] == tau]))

    per_tau = {}
    for tau in TAUS:
        sub = [r for r in rows if r['tau'] == tau]
        d = {k: dict(z_rms=agg('z_rms', k, tau),
                     z_red_pct=agg('z_red_pct', k, tau)) for k in ARMS}
        for k in ('hand_frozen', 'hand_stream', 'GT'):
            d[k]['recovery_z'] = float(np.mean([r['recovery_z'][k]
                                                for r in sub]))
        allpass = all(r['z_rms']['hand_stream'] < r['z_rms']['persistence']
                      for r in sub)
        d['stream_beats_persist'] = dict(
            stream_mean=d['hand_stream']['z_rms'],
            persist_mean=d['persistence']['z_rms'],
            mean_pass=bool(d['hand_stream']['z_rms']
                           < d['persistence']['z_rms']),
            all_seed_pass=bool(allpass),
            PASS=bool(d['hand_stream']['z_rms'] < d['persistence']['z_rms']
                      and allpass))
        per_tau[tau] = d

    passing = [tau for tau in TAUS if per_tau[tau]['stream_beats_persist']['PASS']]
    tau_star = min(passing) if passing else None
    gate = dict(name='G-D18-1 streaming preview-FF beats persistence',
                metric='z (heave displacement) RMS, mean + all-seed',
                per_tau={str(tau): per_tau[tau]['stream_beats_persist']
                         for tau in TAUS},
                passing_taus=passing, tau_star=tau_star,
                PASS=bool(tau_star is not None),
                verdict=('PASS: preview-forecast FF has control value for '
                         f'actuator lead tau >= {tau_star}s' if tau_star
                         else 'FAIL: no preview value at any tau for this '
                         'controller class at 11.3 m/s'))

    def skill_agg(key):
        out = []
        for b in range(len(rows[0][key])):
            cs = [r[key][b]['corr'] for r in rows
                  if not np.isnan(r[key][b]['corr'])]
            out.append(dict(t_lo=rows[0][key][b]['t_lo'],
                            t_hi=rows[0][key][b]['t_hi'],
                            corr=float(np.mean(cs)) if cs else float('nan')))
        return out
    skill = dict(frozen=skill_agg('skill_frozen'),
                 streaming=skill_agg('skill_stream'))

    # ---------- (4) UQ-gating, conditional ----------------------------------
    uq = dict(applied=False)
    if tau_star is not None:
        uq.update(applied=True, tau=tau_star, rows=[])
        for sea, sc in SEAS.items():
            kp = (2 * np.pi / sc['Tp']) ** 2 / G
            for seed in range(a.seeds):
                comps = jonswap_components(Hs=sc['Hs'], Tp=sc['Tp'],
                                           n_components=15, depth=depth,
                                           direction_deg=180.0,
                                           direction_spread_deg=20.0,
                                           seed=7000 + seed)
                Q, _ = build_query_current(cfg, n_steps, dt, U)
                ffB = FeedforwardConfig(preview_dt=0.0, lp_alpha=best_lp, **ff0)
                strm, sig = stream_field(comps, U, depth, kp, seed, Q, n_steps,
                                         dt, period, tau_star, want_var=True)
                thr = float(np.median(sig) * 1.5)
                z_plain = run_arm(cfg, comps, ffB,
                                  make_provider(strm, n_steps, dt, U))[1]
                z_gate = run_arm(cfg, comps, ffB,
                                 make_gated_provider(strm, sig, n_steps, dt,
                                                     U, thr))[1]
                uq['rows'].append(dict(sea=sea, seed=seed, sig_thresh=thr,
                                       z_stream=z_plain, z_stream_gated=z_gate))
        uq['z_stream_mean'] = float(np.mean([r['z_stream'] for r in uq['rows']]))
        uq['z_gated_mean'] = float(np.mean([r['z_stream_gated']
                                            for r in uq['rows']]))
        uq['helps'] = bool(uq['z_gated_mean'] < uq['z_stream_mean'])
    else:
        uq['reason'] = ('streaming did not beat persistence at any tau; per '
                        'spec (4) UQ-gating applies only on a passing tau -> '
                        'skipped')

    out = dict(
        config=dict(seas=SEAS, seeds=a.seeds, taus=list(TAUS),
                    alphas=list(ALPHAS), best_lp=best_lp, reest_period=period,
                    U=float(U), depth=float(depth), obs_noise=OBS_NOISE,
                    K_ff=K_FF, warmup_window=WARM, coverage='BF3,BF4,BF5',
                    learned_arm='SKIPPED (CPU ~2.4s/call; 30 re-est x cells '
                    'prohibitive) -- hand only',
                    metric='residual heave RMS over t>5s; z primary',
                    tau_model='actuator INFORMATION delay: command active at t '
                    'was issued at t-tau, cancels the current disturbance at '
                    '(x_cg(t)+x_foil, t)',
                    arms='GT=exact current-cancel (tau-flat ceiling); '
                    'persistence=GT wave at (x_cg-U*tau+x_foil, t-tau); '
                    'hand_frozen=1 warmup estimate; hand_stream=re-est every '
                    'reest_period, issue=t-tau',
                    stageA='shipped default-provider preview_dt=tau sweep, '
                    'lp optimum only'),
        stageA=stageA,
        stageA_grid_mean={f"lp{al:g}_tau{tau:g}": grid_mean[(al, tau)]
                          for al in ALPHAS for tau in TAUS},
        stageA_lp_best_red=lp_best_red, stageA_best_tau=best_tau_A,
        rows=rows,
        per_tau={str(k): v for k, v in per_tau.items()},
        skill_vs_range=skill, gate=gate, uq_gating=uq,
        wall_time_s=float(time.time() - t_wall),
        stream_latency_ms=dict(mean=float(np.mean(stream_ms)),
                               max=float(np.max(stream_ms)),
                               n=len(stream_ms)))
    json.dump(out, open(OUT, 'w'), indent=1, default=float)

    # ---------- print summary ----------------------------------------------
    print("\n===== per-tau z RMS (m) at lp=%g =====" % best_lp)
    print(f"{'tau':>5} {'noFF':>8} {'persist':>8} {'frozen':>8} "
          f"{'stream':>8} {'GT':>8}  {'str<per':>8}  {'recov_str':>9}")
    for tau in TAUS:
        d = per_tau[tau]
        print(f"{tau:>5g} {d['noFF']['z_rms']:>8.4f} "
              f"{d['persistence']['z_rms']:>8.4f} "
              f"{d['hand_frozen']['z_rms']:>8.4f} "
              f"{d['hand_stream']['z_rms']:>8.4f} {d['GT']['z_rms']:>8.4f}  "
              f"{str(d['stream_beats_persist']['PASS']):>8}  "
              f"{d['hand_stream']['recovery_z']:>+9.2f}")
    print("\n----- forecast skill (front-FF drive corr) vs time bin -----")
    for lab in ('frozen', 'streaming'):
        print(f"  {lab:9s} " + "  ".join(
            f"{b['t_lo']}-{b['t_hi']}s:{b['corr']:+.2f}" for b in skill[lab]))
    print(f"\nG-D18-1: passing tau = {passing}  tau*={tau_star}  "
          f"-> {'PASS' if gate['PASS'] else 'FAIL'}")
    print(f"  {gate['verdict']}")
    if uq['applied']:
        print(f"UQ-gating @tau*={uq['tau']:g}: stream {uq['z_stream_mean']:.4f}"
              f" -> gated {uq['z_gated_mean']:.4f}  helps={uq['helps']}")
    else:
        print(f"UQ-gating: skipped ({uq['reason']})")
    print(f"\nstream latency mean {np.mean(stream_ms):.0f} ms "
          f"(max {np.max(stream_ms):.0f}, n={len(stream_ms)})")
    print(f"wall time {time.time() - t_wall:.0f}s   saved {OUT}")


if __name__ == '__main__':
    main()
