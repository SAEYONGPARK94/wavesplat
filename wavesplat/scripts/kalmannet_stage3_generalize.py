#!/usr/bin/env python3
"""
KalmanNet × WaveSplat — Stage 3: Cross-Episode Generalization.

목표: 다양한 에피소드로 훈련 후 held-out test에서 nRMSE 측정.
     WaveSplat(0.223) / ETKF(0.319) 대비 성능 확인 → Go/No-go #3.

Split (같은 test set as WaveSplat/ConvCNP):
  TEST  (6 ep): BF3_5comp_seed0_slope, BF4_5/15/30comp_seed0_uniform,
                BF5_15/30comp_seed0_uniform
  VAL   (4 ep): seed4 episodes
  TRAIN (나머지 ~100 ep): BF2-5, seed0-3 (test/val 제외)

Usage:
    cd wavesplat && python3 -u scripts/kalmannet_stage3_generalize.py
"""

import sys, os, time, json, random
import numpy as np
import torch
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wavesplat.optimizer.kalmannet import KalmanNet
from wavesplat.model.dispersion import dispersion_omega
from wavesplat.evaluation.metrics import nrmse
from wavesplat.run import set_seed

# ─── Episode splits ────────────────────────────────────────────────────────────
TEST_EPISODES = [
    'episode_BF3_5comp_seed0_slope.h5',
    'episode_BF4_5comp_seed0_uniform.h5',
    'episode_BF4_15comp_seed0_uniform.h5',
    'episode_BF4_30comp_seed0_uniform.h5',
    'episode_BF5_15comp_seed0_uniform.h5',
    'episode_BF5_30comp_seed0_uniform.h5',
]

VAL_EPISODES = [
    'episode_BF3_5comp_seed4_uniform.h5',
    'episode_BF4_5comp_seed4_uniform.h5',
    'episode_BF4_15comp_seed4_uniform.h5',
    'episode_BF5_15comp_seed4_uniform.h5',
]

# ─── Config ────────────────────────────────────────────────────────────────────
TP_FACTOR   = 2.0
SEQ_LEN     = 10
SEQ_STRIDE  = 10    # non-overlapping for training (reduces correlation)
N_EPOCHS    = 200
LR          = 3e-4
GRAD_CLIP   = 1.0
LOG_SIGMA0  = -2.5
PATIENCE    = 30    # early stopping on val NLL

N_MAX   = 30
OBS_MAX = 80
SEED    = 42


# ─── Reuse helpers from stage2 ─────────────────────────────────────────────────

def forward_lidar(kx, ky, w, phi, omega, x_obs, y_obs, t):
    x = x_obs[:, None]; y = y_obs[:, None]
    arg = kx[None, :] * x + ky[None, :] * y - omega[None, :] * t + phi[None, :]
    return (w[None, :] * torch.cos(arg)).sum(-1)


def eval_grid_nrmse(kx, ky, w, phi, omega, t, gt_grid_x, gt_grid_y, gt_field, Hs):
    xx, yy = torch.meshgrid(gt_grid_x, gt_grid_y, indexing='ij')
    x_flat = xx.flatten(); y_flat = yy.flatten()
    with torch.no_grad():
        eta_pred = forward_lidar(kx, ky, w, phi, omega, x_flat, y_flat, t)
    pred_field = eta_pred.reshape(len(gt_grid_x), len(gt_grid_y)).T.numpy()
    return nrmse(pred_field, gt_field.numpy(), Hs)


def build_subsequences(h5_path, seq_len, stride, tp_factor, seed):
    from wavesplat.data_loader import EpisodeDataLoader
    from wavesplat.optimizer.sequential_update import SequentialUpdater

    set_seed(seed)
    loader = EpisodeDataLoader(h5_path)
    cfg = dict(init_method='v2_konly', use_dispersion=True, use_confidence=True,
               adaptive_n=True, tracking_window=3, seed=seed,
               depth=float(loader.depth))
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
        float(loader.depth))

    tracking_frames = loader.get_tracking_frames(tp_factor=tp_factor)
    all_frames, t_prev = [], float(init_frames[-1].t) if init_frames else 0.0
    for f in tracking_frames:
        if len(f.x_obs) == 0: continue
        all_frames.append({
            't': float(f.t), 'dt': float(f.t) - t_prev,
            'x_obs': f.x_obs.detach().clone(),
            'y_obs': f.y_obs.detach().clone(),
            'eta_obs': f.z_obs.detach().clone(),
            'gt_field': f.gt_field.detach().clone(),
        })
        t_prev = float(f.t)

    subs = []
    for start in range(0, max(1, len(all_frames) - seq_len + 1), stride):
        sub = all_frames[start:start + seq_len]
        if len(sub) == seq_len:
            subs.append(sub)

    meta = {'episode': os.path.basename(h5_path), 'N': N,
            'Hs': float(loader.Hs_true), 'Tp': float(loader.Tp_true),
            'depth': float(loader.depth), 'n_subs': len(subs)}
    return init_state, omega, subs, loader.gt_grid_x, loader.gt_grid_y, meta


def pack_sequence(frames, init_state, n_max, obs_max):
    N = init_state['kx'].shape[0]
    state_mask = torch.zeros(2 * n_max)
    state_mask[:N] = 1.0
    state_mask[n_max:n_max + N] = 1.0

    packed = []
    for f in frames:
        M = f['x_obs'].shape[0]
        innov_mask = torch.zeros(obs_max)
        innov_mask[:min(M, obs_max)] = 1.0
        packed.append({
            't': f['t'], 'dt': f['dt'],
            'x_obs': f['x_obs'][:obs_max],
            'y_obs': f['y_obs'][:obs_max],
            'eta_obs_pad': torch.cat([f['eta_obs'][:obs_max],
                                       torch.zeros(max(0, obs_max - M))]),
            'innov_mask': innov_mask,
            'gt_field': f['gt_field'],
        })
    return packed, state_mask


def rollout_nll(model, packed_frames, init_state, omega, state_mask,
                log_sigma, n_max, obs_max, hidden):
    N = init_state['kx'].shape[0]
    sigma = log_sigma.exp()
    w, phi = init_state['w'].clone(), init_state['phi'].clone()
    kx, ky = init_state['kx'], init_state['ky']

    def pad_n(x):
        return torch.cat([x, torch.zeros(n_max - x.shape[0])]) if x.shape[0] < n_max else x[:n_max]

    total_nll = 0.0
    for fr in packed_frames:
        phi = phi + omega * fr['dt']
        M_actual = fr['x_obs'].shape[0]
        eta_pred = forward_lidar(kx, ky, w, phi, omega,
                                 fr['x_obs'], fr['y_obs'], fr['t'])
        eta_pred_pad = torch.cat([eta_pred.detach(),
                                   torch.zeros(obs_max - M_actual)])
        innov_pad = fr['eta_obs_pad'] - eta_pred_pad

        sp = torch.cat([pad_n(w.detach()), pad_n(phi.detach())]).unsqueeze(0)
        delta, hidden = model(innov_pad.unsqueeze(0), fr['innov_mask'].unsqueeze(0),
                              sp, sp.clone(), state_mask.unsqueeze(0), hidden)
        delta = delta.squeeze(0)
        w   = w   + delta[:N]
        phi = phi + delta[n_max:n_max + N]

        M_valid = int(fr['innov_mask'].sum().item())
        eta_upd = forward_lidar(kx, ky, w, phi, omega,
                                fr['x_obs'][:M_valid], fr['y_obs'][:M_valid], fr['t'])
        residual = fr['eta_obs_pad'][:M_valid] - eta_upd
        total_nll = total_nll + 0.5 * (residual / sigma).pow(2).mean() + log_sigma

    return total_nll / len(packed_frames), hidden


@torch.no_grad()
def rollout_nrmse(model, packed_frames, init_state, omega, state_mask,
                  n_max, obs_max, gt_grid_x, gt_grid_y, Hs):
    N = init_state['kx'].shape[0]
    w, phi = init_state['w'].clone(), init_state['phi'].clone()
    kx, ky = init_state['kx'], init_state['ky']
    hidden = model.init_hidden(1)

    def pad_n(x):
        return torch.cat([x, torch.zeros(n_max - x.shape[0])]) if x.shape[0] < n_max else x[:n_max]

    ep_nrmse = []
    for fr in packed_frames:
        phi = phi + omega * fr['dt']
        M_actual = fr['x_obs'].shape[0]
        eta_pred = forward_lidar(kx, ky, w, phi, omega,
                                 fr['x_obs'], fr['y_obs'], fr['t'])
        eta_pred_pad = torch.cat([eta_pred.detach(),
                                   torch.zeros(obs_max - M_actual)])
        innov_pad = fr['eta_obs_pad'] - eta_pred_pad

        sp = torch.cat([pad_n(w.detach()), pad_n(phi.detach())]).unsqueeze(0)
        delta, hidden = model(innov_pad.unsqueeze(0), fr['innov_mask'].unsqueeze(0),
                              sp, sp.clone(), state_mask.unsqueeze(0), hidden)
        delta = delta.squeeze(0)
        w   = w   + delta[:N]
        phi = phi + delta[n_max:n_max + N]

        val = eval_grid_nrmse(kx, ky, w, phi, omega,
                               fr['t'], gt_grid_x, gt_grid_y, fr['gt_field'], Hs)
        if not np.isnan(val):
            ep_nrmse.append(val)

    return float(np.mean(ep_nrmse)) if ep_nrmse else float('nan')


# ─── Load all episodes ────────────────────────────────────────────────────────

def load_all(episode_names, data_dir, seq_len, stride, tp_factor, seed):
    items = []
    for name in episode_names:
        path = os.path.join(data_dir, name)
        if not os.path.exists(path):
            continue
        try:
            init_state, omega, subs, gx, gy, meta = build_subsequences(
                path, seq_len, stride, tp_factor, seed)
            if not subs:
                continue
            packed_list = []
            state_mask = None
            for sub in subs:
                pf, sm = pack_sequence(sub, init_state, N_MAX, OBS_MAX)
                packed_list.append(pf)
                state_mask = sm
            items.append({
                'init_state': init_state, 'omega': omega,
                'packed_list': packed_list, 'state_mask': state_mask,
                'gt_grid_x': gx, 'gt_grid_y': gy,
                'Hs': meta['Hs'], 'name': meta['episode'],
            })
        except Exception as e:
            print("  SKIP %s: %s" % (name, e))
    return items


def discover_train_episodes(data_dir, test_eps, val_eps):
    exclude = set(test_eps) | set(val_eps)
    all_h5 = [f for f in os.listdir(data_dir) if f.endswith('.h5')]
    # Use BF2-BF5 only (BF6-BF7 may be harder/OOD)
    train = [f for f in all_h5 if f not in exclude
             and any(f.startswith('episode_BF%d' % b) for b in [2, 3, 4, 5])]
    return sorted(train)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    set_seed(SEED)

    print("=" * 70)
    print("KalmanNet × WaveSplat — Stage 3: Cross-Episode Generalization")
    print("=" * 70)

    train_names = discover_train_episodes(data_dir, TEST_EPISODES, VAL_EPISODES)
    print("\n  Train episodes: %d" % len(train_names))
    print("  Val  episodes: %d" % len(VAL_EPISODES))
    print("  Test episodes: %d" % len(TEST_EPISODES))

    print("\n[Loading train sequences...]")
    t0 = time.time()
    train_items = load_all(train_names, data_dir, SEQ_LEN, SEQ_STRIDE, TP_FACTOR, SEED)
    total_train_seqs = sum(len(it['packed_list']) for it in train_items)
    print("  Loaded %d episodes, %d total sequences (%.0fs)" % (
        len(train_items), total_train_seqs, time.time() - t0))

    print("[Loading val sequences...]")
    val_items = load_all(VAL_EPISODES, data_dir, SEQ_LEN, SEQ_STRIDE, TP_FACTOR, SEED)
    print("  Loaded %d val episodes" % len(val_items))

    # ── Model ──
    model = KalmanNet(n_max=N_MAX, obs_max=OBS_MAX, embed_dim=64, hidden_dim=128)
    log_sigma = torch.nn.Parameter(torch.tensor(LOG_SIGMA0))
    params = list(model.parameters()) + [log_sigma]
    optimizer = optim.AdamW(params, lr=LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)
    print("\n  KalmanNet: %d params" % sum(p.numel() for p in model.parameters()))

    # ── Training ──
    print("\n[Training %d epochs, early stopping patience=%d]" % (N_EPOCHS, PATIENCE))
    best_val_nll = float('inf')
    best_state = None
    best_log_sigma = LOG_SIGMA0
    no_improve = 0

    # Build flat list of (init_state, omega, pf, state_mask) for shuffling
    all_train_seqs = []
    for it in train_items:
        for pf in it['packed_list']:
            all_train_seqs.append((it['init_state'], it['omega'], pf, it['state_mask']))

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        random.shuffle(all_train_seqs)
        epoch_nll = 0.0

        for init_s, omega, pf, smask in all_train_seqs:
            hidden = model.init_hidden(1)
            optimizer.zero_grad()
            loss, _ = rollout_nll(model, pf, init_s, omega, smask,
                                   log_sigma, N_MAX, OBS_MAX, hidden)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
            optimizer.step()
            epoch_nll += loss.item()

        scheduler.step()
        avg_train_nll = epoch_nll / len(all_train_seqs)

        # Val NLL
        model.eval()
        val_nll_total, val_count = 0.0, 0
        for it in val_items:
            for pf in it['packed_list']:
                with torch.no_grad():
                    hidden = model.init_hidden(1)
                    loss, _ = rollout_nll(model, pf, it['init_state'], it['omega'],
                                          it['state_mask'], log_sigma, N_MAX, OBS_MAX, hidden)
                val_nll_total += loss.item()
                val_count += 1
        val_nll = val_nll_total / max(val_count, 1)

        improved = ""
        if val_nll < best_val_nll:
            best_val_nll = val_nll
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_log_sigma = log_sigma.item()
            no_improve = 0
            improved = " *"
            # Save checkpoint
            ckpt_dir = os.path.join(data_dir, '..', 'checkpoints')
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(best_state, os.path.join(ckpt_dir, 'kalmannet_v1_best.pt'))
        else:
            no_improve += 1

        if epoch % 20 == 0 or epoch == 1:
            elapsed = time.time() - t0
            print("  Epoch %3d/%d: train=%.4f val=%.4f σ=%.4f [%.0fs]%s" % (
                epoch, N_EPOCHS, avg_train_nll, val_nll,
                log_sigma.exp().item(), elapsed, improved))

        if no_improve >= PATIENCE:
            print("  Early stop at epoch %d (patience=%d)" % (epoch, PATIENCE))
            break

    print("  Best val NLL: %.4f" % best_val_nll)

    # ── Test evaluation ──
    print("\n[Test Evaluation — best model]")
    model.load_state_dict(best_state)
    model.eval()

    print("[Loading test episodes (full sequences)...]")
    test_items = load_all(TEST_EPISODES, data_dir, SEQ_LEN, SEQ_STRIDE, TP_FACTOR, SEED)

    results = {}
    all_nrmse = []
    for it in test_items:
        ep_nrmses = []
        for pf in it['packed_list']:
            val = rollout_nrmse(model, pf, it['init_state'], it['omega'],
                                it['state_mask'], N_MAX, OBS_MAX,
                                it['gt_grid_x'], it['gt_grid_y'], it['Hs'])
            if not np.isnan(val):
                ep_nrmses.append(val)
        avg = float(np.mean(ep_nrmses)) if ep_nrmses else float('nan')
        ep_name = it['name'].replace('episode_', '').replace('.h5', '')
        results[ep_name] = avg
        all_nrmse.append(avg)
        print("  %s: nRMSE=%.4f" % (ep_name, avg))

    overall = float(np.mean(all_nrmse))
    print("\n  Avg test nRMSE: %.4f" % overall)
    print("  Reference: WaveSplat 0.223 | ETKF 0.319 | ConvCNP 0.244")

    # ── Save ──
    out = {
        'version': 'kalmannet_stage3_generalize_v1',
        'best_val_nll': best_val_nll,
        'test_avg_nrmse': overall,
        'test_per_episode': results,
        'n_train_episodes': len(train_items),
        'n_train_seqs': total_train_seqs,
        'best_log_sigma': best_log_sigma,
        'config': {
            'N_MAX': N_MAX, 'OBS_MAX': OBS_MAX, 'SEQ_LEN': SEQ_LEN,
            'SEQ_STRIDE': SEQ_STRIDE, 'LR': LR, 'N_EPOCHS': N_EPOCHS,
            'PATIENCE': PATIENCE,
        },
    }
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    out_path = os.path.join(results_dir, 'kalmannet_stage3_generalize.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print("Saved → %s" % out_path)

    print("\n" + "=" * 70)
    if overall < 0.223:
        print("Go/No-go #3: PASS ✓  WaveSplat BEATEN! (%.4f < 0.223)" % overall)
    elif overall < 0.319:
        print("Go/No-go #3: PASS ✓  Beats ETKF (%.4f < 0.319)" % overall)
    else:
        print("Go/No-go #3: FAIL ✗  (%.4f ≥ 0.319)" % overall)
    print("=" * 70)


if __name__ == '__main__':
    main()
