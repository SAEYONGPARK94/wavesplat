"""
d3_d17_ctrl_closedloop.py -- D-17: pre-designed hydrofoil controller in
closed loop with the estimator feedforward (FF) arms.

Pre-registration: docs/D17_CONTROL_prereg_2026-07-12.md (frozen 2026-07-12).

Five FF arms on the shipped Candela C-8 PD+FF controller:
  no-FF        PD only (feedforward=None)
  persistence  true wave, preview_dt=0 (no-forecast competitor)
  hand-FF      mmse_gpu(kp_hint) encounter forecast
  learned-FF   net_gamma_mmse + smult_head encounter forecast (incumbent)
  GT-FF        true wave, full preview (ceiling)

FF interface: the arms are FIELD predictors -- (P,z,depth,Pq)->eta at Pq.
They do NOT hand back RELAX-style components. So instead of the component
path (_compute_feedforward_info over a WaveSplatComponent list, as
d3_e2e_stream_ff does for StreamMMSE), we inject the tau-lead encounter
forecast DIRECTLY: we run the estimator ONCE on a 20 s forward-fan warmup,
then query its frozen forecast field eta_est(x,y,t) at the future foil
preview positions across the whole run (one batched call), and turn that
into the controller's FeedforwardWaveInfo(slope, w) by central finite
differences (slope = d eta/dx, w = d eta/dt -- exactly the quantities
disturbance.py's GT path computes analytically). The per-control-step FF
cost is then a table lookup; the estimator solve is the one-time cost that
the G-D17-3 budget is measured against.

Metric: residual heave RMS (accel primary, z reported) over t>5 s, as %
of the no-FF (uncompensated) baseline; recovery = (persist-arm)/(persist-GT).

Usage: python3 scripts/d3_d17_ctrl_closedloop.py [--seeds 2]
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
from d3_sa_fim_identifiability import omega_of
from d3_gpu_mmse import mmse_gpu
from d3_prior_net_eval import net_gamma_mmse
from d3_prior_net_train import PriorNet
from d3_l1_s2head_train import SmultHead

G = 9.81
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT = 'results/direction3/d17_ctrl_eval.json'
CKPT = 'results/direction3/prior_net_l1s2bw.pt'

# d3_e2e_stream_ff.SEAS convention (BF3, BF4). BF2/BF5 not covered -- see
# report. Hs/Tp from the hydrofoil paper head-sea scenario rows.
SEAS = {'BF3': dict(Hs=0.88, Tp=5.0), 'BF4': dict(Hs=1.88, Tp=7.5)}

OBS_NOISE = 0.02          # Falcon 2 cm 1-sigma elevation noise [m]
PREVIEW_DT = 0.5          # shipped FeedforwardConfig default [s]
K_FF = 0.8
LP_ALPHA = 0.05
EPS_X = 0.5               # spatial FD step for slope [m]
EPS_T = 0.05              # temporal FD step for vertical velocity [s]


def collect_obs(comps_true, U, depth, seed, n_obs=600):
    """20 s forward-fan warmup on the true linear sea -> (P, z, kp).

    Identical geometry to d3_e2e_stream_ff.estimate_components: ship runs
    +x at U, fan looks +x ahead (r 20-100 m, +/-60 deg), head seas.
    """
    kx = np.array([c.kx for c in comps_true])
    ky = np.array([c.ky for c in comps_true])
    A = np.array([c.A for c in comps_true])
    ph = np.array([c.phi for c in comps_true])
    om = omega_of(kx, ky, depth)

    def eta(x, y, t):
        ps = (kx[None] * np.atleast_1d(x)[:, None]
              + ky[None] * np.atleast_1d(y)[:, None]
              - om[None] * np.atleast_1d(t)[:, None] + ph[None])
        return (A[None] * np.cos(ps)).sum(1)

    rng = np.random.default_rng(seed)
    dt, m = 0.1, 12
    Pw, zw = [], []
    for t in np.arange(-20.0, 0.0, dt):
        sx = U * t
        rr = rng.uniform(20.0, 100.0, m)
        aa = rng.uniform(-np.pi / 3, np.pi / 3, m)
        xs = sx + rr * np.cos(aa)
        ys = rr * np.sin(aa)
        Pw.append(np.column_stack([xs, ys, np.full(m, t)]))
        zw.append(eta(xs, ys, t) + rng.normal(0, OBS_NOISE, m))
    Pw, zw = np.vstack(Pw), np.concatenate(zw)
    ii = np.random.default_rng(1).choice(len(zw), n_obs, False)
    return Pw[ii], zw[ii]


def build_query(cfg, n_steps, dt, U, preview_dt):
    """Foil preview query points for the whole run: for each timestep and
    each foil, 4 points (x+-EPS_X for slope, t+-EPS_T for w) at the future
    preview position (x_cg + x_foil + U*preview_dt).  Heading = 0.
    Returns Q (8*n_steps, 3) ordered [front x+,x-,t+,t-, rear x+,x-,t+,t-]."""
    front, rear = cfg['front_foil'], cfg['rear_foil']
    t_arr = np.arange(n_steps) * dt
    x_cg = U * t_arr                              # state.x advances U*dt/step
    dxp = U * preview_dt
    blocks = []
    for foil in (front, rear):
        xf = x_cg + foil.x_pos + dxp
        z0 = np.zeros(n_steps)
        blocks.append(np.column_stack([xf + EPS_X, z0, t_arr]))
        blocks.append(np.column_stack([xf - EPS_X, z0, t_arr]))
        blocks.append(np.column_stack([xf, z0, t_arr + EPS_T]))
        blocks.append(np.column_stack([xf, z0, t_arr - EPS_T]))
    return np.vstack(blocks), t_arr


def make_provider(eta_flat, n_steps, dt):
    """eta_flat: (8*n_steps,) forecast field at build_query points. Returns
    an ff_info_provider that looks up central-difference slope/w by step."""
    e = eta_flat.reshape(8, n_steps)
    slope_f = (e[0] - e[1]) / (2 * EPS_X)
    w_f = (e[2] - e[3]) / (2 * EPS_T)
    slope_r = (e[4] - e[5]) / (2 * EPS_X)
    w_r = (e[6] - e[7]) / (2 * EPS_T)

    def provider(state, ff_, rf_, sim, comps, t, hd, fc):
        i = int(round(t / dt))
        i = min(max(i, 0), n_steps - 1)
        return FeedforwardWaveInfo(slope_front=float(slope_f[i]),
                                   slope_rear=float(slope_r[i]),
                                   w_front=float(w_f[i]),
                                   w_rear=float(w_r[i]))
    return provider


def rms(x):
    return float(np.sqrt(np.mean(x ** 2)))


def run_arm(cfg, comps, ff_cfg, provider=None):
    """One closed-loop sim -> (accel_rms, z_rms) over settle window t>5 s."""
    r = run_simulation(cfg, comps, vessel_heading=0.0,
                       controller=PDControllerConfig(),
                       feedforward=ff_cfg, ff_info_provider=provider)
    mask = r.time > 5.0
    return rms(r.heave_accel[mask]), rms(r.z[mask])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=2)
    a = ap.parse_args()

    cfg = candela_c8()
    U = cfg['sim'].speed
    depth = cfg['sim'].depth
    n_steps = int(cfg['sim'].duration / cfg['sim'].dt) + 1
    dt = cfg['sim'].dt

    ck = torch.load(CKPT, map_location=DEV)
    net = PriorNet().to(DEV); net.load_state_dict(ck['model']); net.eval()
    head = SmultHead().to(DEV); head.load_state_dict(ck['head']); head.eval()
    print(f"device {DEV}  ckpt {CKPT} step {ck['step']}", flush=True)

    ff = FeedforwardConfig(preview_dt=PREVIEW_DT, K_ff=K_FF, lp_alpha=LP_ALPHA,
                           noise_std=0.0, slope_noise_std=0.0)
    ff_persist = FeedforwardConfig(preview_dt=0.0, K_ff=K_FF, lp_alpha=LP_ALPHA,
                                   noise_std=0.0, slope_noise_std=0.0)

    # CUDA warmup so the first timed estimator call is not context/compile
    if DEV.type == 'cuda':
        _w = torch.randn(600, 3, device=DEV)
        _ = mmse_gpu(_w, torch.randn(600, device=DEV), depth,
                     _w[:8].clone(), 0.02, 0.1, seed=0)
        torch.cuda.synchronize()

    ARMS = ('noFF', 'persistence', 'hand', 'learned', 'GT')
    rows = []
    ff_ms = {'hand': [], 'learned': []}          # full-traj forecast precompute
    stream_ms = {'hand': [], 'learned': []}       # per-window streaming latency

    for sea, sc in SEAS.items():
        kp = (2 * np.pi / sc['Tp']) ** 2 / G
        for seed in range(a.seeds):
            comps = jonswap_components(Hs=sc['Hs'], Tp=sc['Tp'],
                                       n_components=15, depth=depth,
                                       direction_deg=180.0,
                                       direction_spread_deg=20.0,
                                       seed=7000 + seed)
            P, z = collect_obs(comps, U, depth, seed)
            sigma0 = OBS_NOISE
            Q, _ = build_query(cfg, n_steps, dt, U, PREVIEW_DT)

            # --- hand arm forecast field (one mmse_gpu call) ---
            Pt = torch.tensor(P, device=DEV, dtype=torch.float32)
            zt = torch.tensor(z, device=DEV, dtype=torch.float32)
            Qt = torch.tensor(Q, device=DEV, dtype=torch.float32)
            if DEV.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.time()
            eta_hand = mmse_gpu(Pt, zt, depth, Qt, sigma0, kp, seed=seed)
            if DEV.type == 'cuda':
                torch.cuda.synchronize()
            ms_hand = (time.time() - t0) * 1e3
            eta_hand = eta_hand.cpu().numpy()
            ff_ms['hand'].append(ms_hand)

            # --- learned arm forecast field (one net_gamma_mmse call) ---
            t0 = time.time()
            eta_net = net_gamma_mmse(net, P, z, depth, Q, sigma0, seed=seed,
                                     kp_hint=kp, use_em=False, smult_head=head)
            ms_net = (time.time() - t0) * 1e3
            ff_ms['learned'].append(ms_net)

            # --- streaming latency: one estimate producing a control-horizon
            #     forecast (8 preview pts = one step). This is the deployment
            #     per-window FF-production cost that G-D17-3 measures; the
            #     full-trajectory call above is an offline precompute. ---
            Q8 = Q[:8]
            Q8t = torch.tensor(Q8, device=DEV, dtype=torch.float32)
            if DEV.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.time()
            _ = mmse_gpu(Pt, zt, depth, Q8t, sigma0, kp, seed=seed)
            if DEV.type == 'cuda':
                torch.cuda.synchronize()
            stream_ms['hand'].append((time.time() - t0) * 1e3)
            t0 = time.time()
            _ = net_gamma_mmse(net, P, z, depth, Q8, sigma0, seed=seed,
                               kp_hint=kp, use_em=False, smult_head=head)
            stream_ms['learned'].append((time.time() - t0) * 1e3)

            prov_hand = make_provider(eta_hand, n_steps, dt)
            prov_net = make_provider(eta_net, n_steps, dt)

            res = {}
            res['noFF'] = run_arm(cfg, comps, None)
            res['persistence'] = run_arm(cfg, comps, ff_persist)
            res['hand'] = run_arm(cfg, comps, ff, prov_hand)
            res['learned'] = run_arm(cfg, comps, ff, prov_net)
            res['GT'] = run_arm(cfg, comps, ff)

            base_a = res['noFF'][0]
            base_z = res['noFF'][1]
            row = dict(sea=sea, seed=seed, kp=float(kp),
                       accel_rms={k: res[k][0] for k in ARMS},
                       z_rms={k: res[k][1] for k in ARMS},
                       accel_red_pct={k: 100 * (1 - res[k][0] / base_a)
                                      for k in ARMS},
                       z_red_pct={k: 100 * (1 - res[k][1] / base_z)
                                  for k in ARMS},
                       ff_ms=dict(hand=ms_hand, learned=ms_net))
            # recovery = (persist - arm)/(persist - GT), each metric
            for mi, mkey in ((0, 'recovery_accel'), (1, 'recovery_z')):
                gap = res['persistence'][mi] - res['GT'][mi]
                row[mkey] = {
                    k: float((res['persistence'][mi] - res[k][mi]) / gap)
                    if abs(gap) > 1e-9 else float('nan')
                    for k in ('hand', 'learned', 'GT')}
            rows.append(row)
            print(f"[{sea} s{seed}] accel RMS (m/s^2): "
                  + "  ".join(f"{k} {res[k][0]:.3f}" for k in ARMS)
                  + f"  | red%: persist {row['accel_red_pct']['persistence']:+.0f} "
                  f"hand {row['accel_red_pct']['hand']:+.0f} "
                  f"learn {row['accel_red_pct']['learned']:+.0f} "
                  f"GT {row['accel_red_pct']['GT']:+.0f}"
                  f"  | recov learn {row['recovery_accel']['learned']:+.2f}"
                  f"  | ff_ms hand {ms_hand:.0f} net {ms_net:.0f}", flush=True)

    # ------------- aggregate + gates -------------
    def arm_mean(field, arm):
        return float(np.mean([r[field][arm] for r in rows]))

    per_arm = {}
    for k in ARMS:
        per_arm[k] = dict(
            accel_rms=arm_mean('accel_rms', k),
            z_rms=arm_mean('z_rms', k),
            accel_red_pct=arm_mean('accel_red_pct', k),
            z_red_pct=arm_mean('z_red_pct', k))
    for k in ('hand', 'learned', 'GT'):
        for mkey in ('recovery_accel', 'recovery_z'):
            per_arm[k][mkey] = float(np.mean([r[mkey][k] for r in rows]))

    # G-D17-1 (main): learned-FF < persistence-FF residual heave RMS.
    # Evaluated on both metrics; "heave motion" (prereg wording) -> z primary.
    def gate1(metric):
        rk = 'accel_rms' if metric == 'accel' else 'z_rms'
        lm = per_arm['learned'][rk]
        pm = per_arm['persistence'][rk]
        allp = all(r[rk]['learned'] < r[rk]['persistence'] for r in rows)
        return dict(metric=metric, learned_mean=lm, persistence_mean=pm,
                    mean_pass=bool(lm < pm), all_seed_pass=bool(allp),
                    PASS=bool(lm < pm and allp))
    g1 = dict(name='G-D17-1 forecast control value (main)',
              primary='z (heave displacement RMS)',
              z=gate1('z'), accel=gate1('accel'),
              PASS=bool(gate1('z')['PASS']),
              verdict_robust=bool(gate1('z')['PASS'] == gate1('accel')['PASS']))

    def gate2(metric):
        rk = 'accel_rms' if metric == 'accel' else 'z_rms'
        return dict(metric=metric, hand_mean=per_arm['hand'][rk],
                    learned_mean=per_arm['learned'][rk],
                    learned_minus_hand=float(per_arm['learned'][rk]
                                             - per_arm['hand'][rk]),
                    hand_better=bool(per_arm['hand'][rk]
                                     < per_arm['learned'][rk]))
    g2 = dict(name='G-D17-2 estimator->control transfer (descriptive)',
              note='clean synthetic: hand-win is normal (07-12 sign-reversal law)',
              z=gate2('z'), accel=gate2('accel'))

    max_stream = float(np.max(stream_ms['hand'] + stream_ms['learned']))
    g3 = dict(name='G-D17-3 FF production budget',
              metric='per-window streaming estimate latency, ms (<=100)',
              note=('streaming latency = one estimator call producing a '
                    'control-horizon forecast (8 preview pts). The full %d-pt '
                    'trajectory forecast is an offline precompute, reported '
                    'separately (forecast_precompute_ms); per-control-step FF '
                    'extraction from the cached forecast is a table lookup.'
                    % Q.shape[0]),
              hand_stream_ms=stream_ms['hand'],
              learned_stream_ms=stream_ms['learned'],
              max_stream_ms=max_stream,
              forecast_precompute_ms=dict(hand=ff_ms['hand'],
                                          learned=ff_ms['learned']),
              PASS=bool(max_stream <= 100.0))

    out = dict(
        config=dict(seas=SEAS, seeds=a.seeds, coverage='BF3,BF4 only '
                    '(d3_e2e_stream_ff.SEAS); BF2 and BF5 NOT covered',
                    preview_dt=PREVIEW_DT, K_ff=K_FF, lp_alpha=LP_ALPHA,
                    obs_noise=OBS_NOISE, ckpt=CKPT, U=float(U),
                    metric='residual heave RMS over t>5s; z primary, accel secondary',
                    ff_injection='frozen forecast field, central-diff slope/w'),
        rows=rows, per_arm=per_arm,
        gates=dict(g1=g1, g2=g2, g3=g3))
    json.dump(out, open(OUT, 'w'), indent=1, default=float)

    print("\n===== per-arm mean (z RMS m; accel RMS m/s^2; red% vs no-FF) =====")
    for k in ARMS:
        pa = per_arm[k]
        rec = (f"  recov(z) {pa['recovery_z']:+.2f}"
               if 'recovery_z' in pa else "")
        print(f"  {k:12s} z {pa['z_rms']:.4f} ({pa['z_red_pct']:+.0f}%)"
              f"   accel {pa['accel_rms']:.3f} ({pa['accel_red_pct']:+.0f}%){rec}")
    for lab, gg in (('z', g1['z']), ('accel', g1['accel'])):
        print(f"\nG-D17-1[{lab}] learned {gg['learned_mean']:.4f} vs "
              f"persist {gg['persistence_mean']:.4f}: mean {gg['mean_pass']} "
              f"all-seed {gg['all_seed_pass']} -> "
              f"{'PASS' if gg['PASS'] else 'FAIL'}")
    for lab, gg in (('z', g2['z']), ('accel', g2['accel'])):
        print(f"G-D17-2[{lab}] hand {gg['hand_mean']:.4f} learned "
              f"{gg['learned_mean']:.4f} (learned-hand "
              f"{gg['learned_minus_hand']:+.4f}; "
              f"{'hand better' if gg['hand_better'] else 'learned better'})")
    print(f"G-D17-3 streaming latency max {max_stream:.0f} ms -> "
          f"{'PASS' if g3['PASS'] else 'FAIL'}  "
          f"(hand {np.mean(stream_ms['hand']):.0f} / "
          f"net {np.mean(stream_ms['learned']):.0f} ms mean)")
    print(f"saved {OUT}")


if __name__ == '__main__':
    main()
