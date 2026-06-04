#!/usr/bin/env python3
"""
KalmanNet × WaveSplat — Stage 2: Overfit Test.

목표: 1 episode의 여러 subsequence로 KalmanNet이 overfit 가능한지 확인.
     NLL 수렴 + tracking nRMSE 측정 → Go/No-go #2.

Design:
  - 1 episode → init 1회 → 전체 tracking frames에서 subsequence 추출
  - 각 subsequence 길이: SEQ_LEN=10 frames, stride=SEQ_STRIDE=5
  - 200 epoch 학습 (overfit 의도)
  - Learnable log_sigma + gradient clipping
  - 평가: 각 frame 후 GT grid에서 nRMSE 계산

Usage:
    cd wavesplat && python3 -u scripts/kalmannet_stage2_overfit.py
"""

import sys, os, time, json
import numpy as np
import torch
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.optimizer.kalmannet import KalmanNet
from wavesplat.model.dispersion import dispersion_omega
from wavesplat.evaluation.metrics import nrmse
from wavesplat.run import set_seed

# ─── Config ───────────────────────────────────────────────────────────────────
EPISODE = 'episode_BF4_5comp_seed0_uniform.h5'  # test episode로 overfit
TP_FACTOR  = 2.0     # init window (Tp * factor seconds)
SEQ_LEN    = 10      # tracking frames per sequence
SEQ_STRIDE = 5       # stride between sequences
N_EPOCHS   = 300
LR         = 5e-4
GRAD_CLIP  = 1.0
LOG_SIGMA0 = -2.5    # initial log_sigma (~0.08 m)

N_MAX  = 30
OBS_MAX = 80

SEED = 42


# ─── Helper: forward model ─────────────────────────────────────────────────────

def forward_lidar(kx, ky, w, phi, omega, x_obs, y_obs, t):
    """η(x,y,t) = Σ w_n cos(kx_n x + ky_n y - ω_n t + φ_n) at obs points."""
    x = x_obs[:, None]; y = y_obs[:, None]
    arg = kx[None, :] * x + ky[None, :] * y - omega[None, :] * t + phi[None, :]
    return (w[None, :] * torch.cos(arg)).sum(-1)


def eval_grid_nrmse(kx, ky, w, phi, omega, t, gt_grid_x, gt_grid_y, gt_field, Hs):
    """Predicted wave field vs GT on full grid."""
    xx, yy = torch.meshgrid(gt_grid_x, gt_grid_y, indexing='ij')
    x_flat = xx.flatten(); y_flat = yy.flatten()
    with torch.no_grad():
        eta_pred = forward_lidar(kx, ky, w, phi, omega, x_flat, y_flat, t)
    pred_field = eta_pred.reshape(len(gt_grid_x), len(gt_grid_y)).T.numpy()
    return nrmse(pred_field, gt_field.numpy(), Hs)


# ─── Build subsequences from episode ─────────────────────────────────────────

def build_subsequences(h5_path, seq_len, stride, tp_factor, seed):
    """Run init once, extract all tracking frames, split into subsequences."""
    from wavesplat.optimizer.sequential_update import SequentialUpdater

    set_seed(seed)
    loader = EpisodeDataLoader(h5_path)

    cfg = dict(
        init_method='v2_konly', use_dispersion=True, use_confidence=True,
        adaptive_n=True, tracking_window=3, seed=seed,
        depth=float(loader.depth),
    )
    updater = SequentialUpdater(cfg)
    init_frames = loader.get_init_window(tp_factor=tp_factor)
    updater.initialize(
        init_frames,
        x_range=(float(loader.gt_grid_x[0]), float(loader.gt_grid_x[-1])),
        y_range=(float(loader.gt_grid_y[0]), float(loader.gt_grid_y[-1])),
    )

    with torch.no_grad():
        init_state = {k: getattr(updater.model, k).detach().clone()
                      for k in ('kx', 'ky', 'w', 'phi')}
    N = init_state['kx'].shape[0]
    omega = dispersion_omega(
        torch.sqrt(init_state['kx']**2 + init_state['ky']**2 + 1e-12),
        float(loader.depth)
    )

    tracking_frames = loader.get_tracking_frames(tp_factor=tp_factor)
    all_frames = []
    t_prev = float(init_frames[-1].t) if init_frames else 0.0
    for f in tracking_frames:
        if len(f.x_obs) == 0:
            continue
        all_frames.append({
            't':        float(f.t),
            'dt':       float(f.t) - t_prev,
            'x_obs':    f.x_obs.detach().clone(),
            'y_obs':    f.y_obs.detach().clone(),
            'eta_obs':  f.z_obs.detach().clone(),
            'gt_field': f.gt_field.detach().clone(),
        })
        t_prev = float(f.t)

    # Split into subsequences
    subs = []
    for start in range(0, max(1, len(all_frames) - seq_len + 1), stride):
        sub = all_frames[start:start + seq_len]
        if len(sub) == seq_len:
            subs.append(sub)

    meta = {
        'episode':   os.path.basename(h5_path),
        'N':         N,
        'Hs':        float(loader.Hs_true),
        'Tp':        float(loader.Tp_true),
        'depth':     float(loader.depth),
        'n_frames':  len(all_frames),
        'n_subs':    len(subs),
    }
    return init_state, omega, subs, loader.gt_grid_x, loader.gt_grid_y, meta


# ─── Pack one subsequence for KalmanNet ───────────────────────────────────────

def pack_sequence(frames, init_state, omega, n_max, obs_max):
    """Return per-frame tensors needed for KalmanNet rollout."""
    N = init_state['kx'].shape[0]
    state_mask_base = torch.zeros(2 * n_max)
    state_mask_base[:N] = 1.0        # w slots
    state_mask_base[n_max:n_max + N] = 1.0  # phi slots

    packed = []
    for f in frames:
        M = f['x_obs'].shape[0]
        innov_mask = torch.zeros(obs_max)
        innov_mask[:min(M, obs_max)] = 1.0

        packed.append({
            't':          f['t'],
            'dt':         f['dt'],
            'x_obs':      f['x_obs'][:obs_max],
            'y_obs':      f['y_obs'][:obs_max],
            'eta_obs_pad': torch.cat([f['eta_obs'][:obs_max],
                                      torch.zeros(max(0, obs_max - M))]),
            'innov_mask': innov_mask,
            'gt_field':   f['gt_field'],
        })
    return packed, state_mask_base


# ─── One epoch rollout + loss ─────────────────────────────────────────────────

def rollout_sequence(model, packed_frames, init_state, omega, state_mask,
                     log_sigma, n_max, obs_max, hidden):
    """Roll KalmanNet through one sequence. Return NLL loss."""
    N = init_state['kx'].shape[0]
    sigma = log_sigma.exp()

    w   = init_state['w'].clone()
    phi = init_state['phi'].clone()
    kx  = init_state['kx']
    ky  = init_state['ky']

    def pad_n(x):
        return torch.cat([x, torch.zeros(n_max - x.shape[0])]) if x.shape[0] < n_max else x[:n_max]

    total_nll = 0.0
    for fr in packed_frames:
        # ── Phase advance ──
        phi = phi + omega * fr['dt']

        # ── Predict at actual obs points → pad to obs_max ──
        M_actual = fr['x_obs'].shape[0]
        eta_pred = forward_lidar(kx, ky, w, phi, omega,
                                 fr['x_obs'], fr['y_obs'], fr['t'])
        eta_pred_pad = torch.cat([eta_pred.detach(),
                                   torch.zeros(obs_max - M_actual)])  # (obs_max,)

        # ── Innovation (obs_max dim) ──
        innov_pad = fr['eta_obs_pad'] - eta_pred_pad          # (obs_max,)

        # ── KalmanNet step ──
        state_prev_pad = torch.cat([pad_n(w.detach()), pad_n(phi.detach())]).unsqueeze(0)
        phi_pred_pad   = state_prev_pad.clone()
        innov_b = innov_pad.unsqueeze(0)           # (1, obs_max)
        innov_m = fr['innov_mask'].unsqueeze(0)    # (1, obs_max)
        state_m = state_mask.unsqueeze(0)          # (1, 2N_max)

        delta, hidden = model(innov_b, innov_m, state_prev_pad, phi_pred_pad, state_m, hidden)
        delta = delta.squeeze(0)                   # (2N_max,)

        # ── State update ──
        w   = w   + delta[:N]
        phi = phi + delta[n_max:n_max + N]

        # ── NLL at valid obs ──
        M_valid = int(fr['innov_mask'].sum().item())
        eta_upd = forward_lidar(kx, ky, w, phi, omega,
                                fr['x_obs'][:M_valid], fr['y_obs'][:M_valid], fr['t'])
        residual = fr['eta_obs_pad'][:M_valid] - eta_upd
        nll = 0.5 * (residual / sigma).pow(2).mean() + log_sigma
        total_nll = total_nll + nll

    return total_nll / len(packed_frames), w, phi, hidden


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    h5_path  = os.path.join(data_dir, EPISODE)

    print("=" * 70)
    print("KalmanNet × WaveSplat — Stage 2: Overfit Test")
    print("=" * 70)

    set_seed(SEED)

    # ── Build subsequences ──
    print("\n[Building sequences from episode: %s]" % EPISODE)
    init_state, omega, subsequences, gt_grid_x, gt_grid_y, meta = build_subsequences(
        h5_path, seq_len=SEQ_LEN, stride=SEQ_STRIDE,
        tp_factor=TP_FACTOR, seed=SEED,
    )
    N = meta['N']
    Hs = meta['Hs']
    print("  Hs=%.2fm, Tp=%.2fs, N=%d components" % (Hs, meta['Tp'], N))
    print("  Total tracking frames: %d" % meta['n_frames'])
    print("  Subsequences: %d (len=%d, stride=%d)" % (meta['n_subs'], SEQ_LEN, SEQ_STRIDE))

    # ── Model ──
    model = KalmanNet(n_max=N_MAX, obs_max=OBS_MAX, embed_dim=64, hidden_dim=128)
    log_sigma = torch.nn.Parameter(torch.tensor(LOG_SIGMA0))
    params = list(model.parameters()) + [log_sigma]
    optimizer = optim.AdamW(params, lr=LR, weight_decay=1e-5)
    print("\n  KalmanNet: %d params, log_sigma=%.2f (σ=%.3f m)" % (
        sum(p.numel() for p in model.parameters()), LOG_SIGMA0, np.exp(LOG_SIGMA0)))

    # Pack all subsequences
    packed_subs = []
    state_mask = None
    for sub in subsequences:
        pf, sm = pack_sequence(sub, init_state, omega, N_MAX, OBS_MAX)
        packed_subs.append(pf)
        state_mask = sm

    # ── Training ──
    print("\n[Training %d epochs, %d subsequences]" % (N_EPOCHS, len(packed_subs)))
    best_nll = float('inf')
    best_model_state = None
    t0 = time.time()

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        epoch_nll = 0.0

        for pf in packed_subs:
            hidden = model.init_hidden(1)
            optimizer.zero_grad()
            loss, _, _, _ = rollout_sequence(
                model, pf, init_state, omega, state_mask, log_sigma, N_MAX, OBS_MAX, hidden)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
            optimizer.step()
            epoch_nll += loss.item()

        avg_nll = epoch_nll / len(packed_subs)
        if avg_nll < best_nll:
            best_nll = avg_nll
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_log_sigma = log_sigma.item()

        if epoch % 50 == 0 or epoch == 1:
            elapsed = time.time() - t0
            print("  Epoch %3d/%d: NLL=%.4f  σ=%.4f  best=%.4f  [%.0fs]" % (
                epoch, N_EPOCHS, avg_nll, log_sigma.exp().item(), best_nll, elapsed))

    # ── Evaluation with best model ──
    print("\n[Evaluation — tracking nRMSE with best model]")
    model.load_state_dict(best_model_state)
    model.eval()

    # Evaluate on all subsequences (overfit check)
    all_nrmse = []
    for sub_idx, (sub, pf) in enumerate(zip(subsequences, packed_subs)):
        hidden = model.init_hidden(1)
        w   = init_state['w'].clone()
        phi = init_state['phi'].clone()
        kx  = init_state['kx']
        ky  = init_state['ky']

        sub_nrmse = []
        for fr_idx, (fr_raw, fr_packed) in enumerate(zip(sub, pf)):
            # Phase advance
            phi = phi + omega * fr_raw['dt']

            # KalmanNet step
            M_actual = fr_packed['x_obs'].shape[0]
            eta_pred = forward_lidar(kx, ky, w, phi, omega,
                                     fr_packed['x_obs'], fr_packed['y_obs'], fr_raw['t'])
            eta_pred_pad = torch.cat([eta_pred.detach(), torch.zeros(OBS_MAX - M_actual)])
            innov_pad = fr_packed['eta_obs_pad'] - eta_pred_pad

            def pad(x): return torch.cat([x, torch.zeros(N_MAX - x.shape[0])]) if x.shape[0] < N_MAX else x[:N_MAX]
            state_prev_pad = torch.cat([pad(w.detach()), pad(phi.detach())]).unsqueeze(0)
            phi_pred_pad   = state_prev_pad.clone()

            with torch.no_grad():
                delta, hidden = model(
                    innov_pad.unsqueeze(0), fr_packed['innov_mask'].unsqueeze(0),
                    state_prev_pad, phi_pred_pad, state_mask.unsqueeze(0), hidden)
            delta = delta.squeeze(0)
            N = kx.shape[0]
            w   = w   + delta[:N]
            phi = phi + delta[N_MAX:N_MAX + N]

            # nRMSE on GT grid
            val = eval_grid_nrmse(kx, ky, w, phi, omega,
                                   fr_raw['t'], gt_grid_x, gt_grid_y,
                                   fr_raw['gt_field'], Hs)
            if not np.isnan(val):
                sub_nrmse.append(val)

        if sub_nrmse:
            all_nrmse.append(float(np.mean(sub_nrmse)))

    avg_tracking_nrmse = float(np.mean(all_nrmse)) if all_nrmse else float('nan')
    print("  Per-subsequence nRMSE: " + " ".join("%.4f" % v for v in all_nrmse))
    print("  Avg tracking nRMSE: %.4f" % avg_tracking_nrmse)
    print("  Best NLL: %.4f  σ=%.4f" % (best_nll, np.exp(best_log_sigma)))
    print("\n  Reference: WaveSplat overall 0.223, ETKF 0.319")

    # ── Save results ──
    out = {
        'version': 'kalmannet_stage2_overfit_v1',
        'episode': EPISODE,
        'n_subsequences': len(packed_subs),
        'seq_len': SEQ_LEN,
        'n_epochs': N_EPOCHS,
        'best_nll': best_nll,
        'best_log_sigma': best_log_sigma,
        'avg_tracking_nrmse': avg_tracking_nrmse,
        'per_sub_nrmse': all_nrmse,
        'meta': meta,
        'config': {
            'N_MAX': N_MAX, 'OBS_MAX': OBS_MAX, 'LR': LR,
            'GRAD_CLIP': GRAD_CLIP, 'LOG_SIGMA0': LOG_SIGMA0,
        },
    }
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, 'kalmannet_stage2_overfit.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print("\nSaved → %s" % out_path)

    # ── Go/No-go verdict ──
    print("\n" + "=" * 70)
    if avg_tracking_nrmse < 0.30:
        print("Go/No-go #2: PASS ✓  (nRMSE=%.4f < 0.30)" % avg_tracking_nrmse)
    else:
        print("Go/No-go #2: FAIL ✗  (nRMSE=%.4f ≥ 0.30)" % avg_tracking_nrmse)
    print("=" * 70)


if __name__ == '__main__':
    main()
