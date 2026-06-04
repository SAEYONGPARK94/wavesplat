#!/usr/bin/env python3
"""
KalmanNet × WaveSplat — Stage 3 v2: Improved Generalization.

Stage 3 v1 문제:
  1. 입력 정규화 없음 → Hs(0.6~2.0m) 차이로 encoder 혼란
  2. 데이터 로딩 7153s → 캐싱 없음
  3. 과적합 (val NLL 1.47→9.04) → Dropout 부재

v2 개선:
  1. Hs 정규화: w/Hs, innov/Hs, phi/(2π) → encoder 입력 O(1)
  2. 시퀀스 캐싱: wavesplat/cache/kalmannet/ 에 .pt 저장
  3. KalmanNet에 Dropout(p=0.1)
  4. 더 작은 모델: embed=48, hidden=96

Usage:
    cd wavesplat && python3 -u scripts/kalmannet_stage3_v2.py
"""

import sys, os, time, json, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wavesplat.optimizer.kalmannet import KalmanNet
from wavesplat.model.dispersion import dispersion_omega
from wavesplat.evaluation.metrics import nrmse
from wavesplat.run import set_seed

# ─── Episode splits (동일) ─────────────────────────────────────────────────────
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
TP_FACTOR  = 2.0
SEQ_LEN    = 10
SEQ_STRIDE = 10
N_EPOCHS   = 200
LR         = 3e-4
GRAD_CLIP  = 1.0
LOG_SIGMA0 = -2.5
PATIENCE   = 40

N_MAX   = 30
OBS_MAX = 80
TWO_PI  = 2.0 * np.pi
SEED    = 42

# Smaller model with dropout
EMBED_DIM  = 48
HIDDEN_DIM = 96
DROPOUT    = 0.1


# ─── KalmanNet with dropout ────────────────────────────────────────────────────

class KalmanNetDrop(nn.Module):
    """KalmanNet + Dropout for regularization."""
    def __init__(self, n_max=30, obs_max=80, embed_dim=48, hidden_dim=96, dropout=0.1):
        super().__init__()
        self.n_max = n_max
        self.obs_max = obs_max
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.innov_encoder = nn.Sequential(
            nn.Linear(obs_max * 2, embed_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(2 * n_max * 3, embed_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )
        self.gru = nn.GRU(
            input_size=2 * embed_dim, hidden_size=hidden_dim,
            num_layers=1, batch_first=False, dropout=0.0,
        )
        self.delta_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2 * n_max),
        )

    def init_hidden(self, batch_size, device='cpu'):
        return torch.zeros(1, batch_size, self.hidden_dim, device=device)

    def forward(self, innov, innov_mask, state_prev, state_pred, state_mask, hidden):
        innov_in = torch.cat([innov * innov_mask, innov_mask], dim=-1)
        e_innov  = self.innov_encoder(innov_in)
        state_in = torch.cat([state_prev, state_pred, state_mask], dim=-1)
        e_state  = self.state_encoder(state_in)
        gru_in   = torch.cat([e_innov, e_state], dim=-1).unsqueeze(0)
        gru_out, hidden_new = self.gru(gru_in, hidden)
        delta = self.delta_decoder(gru_out.squeeze(0))
        return delta * state_mask, hidden_new


# ─── Forward model ─────────────────────────────────────────────────────────────

def forward_lidar(kx, ky, w, phi, omega, x_obs, y_obs, t):
    x = x_obs[:, None]; y = y_obs[:, None]
    arg = kx[None, :] * x + ky[None, :] * y - omega[None, :] * t + phi[None, :]
    return (w[None, :] * torch.cos(arg)).sum(-1)


def eval_grid_nrmse(kx, ky, w, phi, omega, t, gx, gy, gt_field, Hs):
    xx, yy = torch.meshgrid(gx, gy, indexing='ij')
    xf, yf = xx.flatten(), yy.flatten()
    with torch.no_grad():
        eta = forward_lidar(kx, ky, w, phi, omega, xf, yf, t)
    pred = eta.reshape(len(gx), len(gy)).T.numpy()
    return nrmse(pred, gt_field.numpy(), Hs)


# ─── Caching ───────────────────────────────────────────────────────────────────

def cache_path(data_dir, episode_name):
    cache_dir = os.path.join(data_dir, '..', 'cache', 'kalmannet')
    os.makedirs(cache_dir, exist_ok=True)
    tag = episode_name.replace('.h5', '')
    return os.path.join(cache_dir, '%s_sl%d_ss%d.pt' % (tag, SEQ_LEN, SEQ_STRIDE))


def build_or_load(h5_path, data_dir, seq_len, stride, tp_factor, seed):
    cp = cache_path(data_dir, os.path.basename(h5_path))
    if os.path.exists(cp):
        return torch.load(cp, map_location='cpu', weights_only=False)

    from wavesplat.data_loader import EpisodeDataLoader
    from wavesplat.optimizer.sequential_update import SequentialUpdater
    set_seed(seed)
    loader = EpisodeDataLoader(h5_path)
    cfg = dict(init_method='v2_konly', use_dispersion=True, use_confidence=True,
               adaptive_n=True, tracking_window=3, seed=seed, depth=float(loader.depth))
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
            'x_obs': f.x_obs.detach().clone(), 'y_obs': f.y_obs.detach().clone(),
            'eta_obs': f.z_obs.detach().clone(), 'gt_field': f.gt_field.detach().clone(),
        })
        t_prev = float(f.t)

    subs = []
    for start in range(0, max(1, len(all_frames) - seq_len + 1), stride):
        sub = all_frames[start:start + seq_len]
        if len(sub) == seq_len:
            subs.append(sub)

    data = {
        'init_state': init_state, 'omega': omega, 'subsequences': subs,
        'gt_grid_x': loader.gt_grid_x, 'gt_grid_y': loader.gt_grid_y,
        'Hs': float(loader.Hs_true), 'Tp': float(loader.Tp_true),
        'depth': float(loader.depth), 'N': N,
        'episode': os.path.basename(h5_path),
    }
    torch.save(data, cp)
    return data


def pack(subs, init_state, Hs, n_max, obs_max):
    N = init_state['kx'].shape[0]
    state_mask = torch.zeros(2 * n_max)
    state_mask[:N] = 1.0
    state_mask[n_max:n_max + N] = 1.0

    packed_list = []
    for sub in subs:
        frames = []
        for f in sub:
            M = f['x_obs'].shape[0]
            innov_mask = torch.zeros(obs_max)
            innov_mask[:min(M, obs_max)] = 1.0
            frames.append({
                't': f['t'], 'dt': f['dt'],
                'x_obs': f['x_obs'][:obs_max], 'y_obs': f['y_obs'][:obs_max],
                'eta_obs_pad': torch.cat([f['eta_obs'][:obs_max],
                                           torch.zeros(max(0, obs_max - M))]),
                'innov_mask': innov_mask, 'gt_field': f['gt_field'],
            })
        packed_list.append(frames)
    return packed_list, state_mask


# ─── Normalized rollout ────────────────────────────────────────────────────────

def pad_n(x, n_max):
    if x.shape[0] < n_max:
        return torch.cat([x, torch.zeros(n_max - x.shape[0])])
    return x[:n_max]


def rollout_nll(model, packed_frames, init_state, omega, state_mask,
                log_sigma, Hs, n_max, obs_max, hidden):
    """Normalized rollout: w/Hs, innov/Hs, phi/(2π)."""
    N = init_state['kx'].shape[0]
    sigma = log_sigma.exp()
    w, phi = init_state['w'].clone(), init_state['phi'].clone()
    kx, ky = init_state['kx'], init_state['ky']

    total_nll = 0.0
    for fr in packed_frames:
        phi = phi + omega * fr['dt']

        M_actual = fr['x_obs'].shape[0]
        eta_pred = forward_lidar(kx, ky, w, phi, omega,
                                 fr['x_obs'], fr['y_obs'], fr['t'])
        eta_pred_pad = torch.cat([eta_pred.detach(),
                                   torch.zeros(obs_max - M_actual)])
        innov_pad = (fr['eta_obs_pad'] - eta_pred_pad) / (Hs + 1e-6)  # normalized

        # Normalized state features
        w_norm   = pad_n(w.detach() / (Hs + 1e-6), n_max)
        phi_norm = pad_n(phi.detach() / TWO_PI, n_max)
        sp = torch.cat([w_norm, phi_norm]).unsqueeze(0)

        delta, hidden = model(innov_pad.unsqueeze(0), fr['innov_mask'].unsqueeze(0),
                              sp, sp.clone(), state_mask.unsqueeze(0), hidden)
        delta = delta.squeeze(0)

        # Denormalize delta: first N_max are Δ(w/Hs), next N_max are Δ(phi/(2π))
        w   = w   + delta[:N] * Hs
        phi = phi + delta[n_max:n_max + N] * TWO_PI

        M_valid = int(fr['innov_mask'].sum().item())
        eta_upd = forward_lidar(kx, ky, w, phi, omega,
                                fr['x_obs'][:M_valid], fr['y_obs'][:M_valid], fr['t'])
        residual = fr['eta_obs_pad'][:M_valid] - eta_upd
        total_nll = total_nll + 0.5 * (residual / sigma).pow(2).mean() + log_sigma

    return total_nll / len(packed_frames), hidden


@torch.no_grad()
def rollout_nrmse_eval(model, packed_frames, init_state, omega, state_mask,
                       Hs, n_max, obs_max, gx, gy):
    N = init_state['kx'].shape[0]
    w, phi = init_state['w'].clone(), init_state['phi'].clone()
    kx, ky = init_state['kx'], init_state['ky']
    hidden = model.init_hidden(1)
    ep_nrmse = []
    for fr in packed_frames:
        phi = phi + omega * fr['dt']
        M_actual = fr['x_obs'].shape[0]
        eta_pred = forward_lidar(kx, ky, w, phi, omega,
                                 fr['x_obs'], fr['y_obs'], fr['t'])
        eta_pred_pad = torch.cat([eta_pred, torch.zeros(obs_max - M_actual)])
        innov_pad = (fr['eta_obs_pad'] - eta_pred_pad) / (Hs + 1e-6)

        w_norm   = pad_n(w / (Hs + 1e-6), n_max)
        phi_norm = pad_n(phi / TWO_PI, n_max)
        sp = torch.cat([w_norm, phi_norm]).unsqueeze(0)

        delta, hidden = model(innov_pad.unsqueeze(0), fr['innov_mask'].unsqueeze(0),
                              sp, sp.clone(), state_mask.unsqueeze(0), hidden)
        delta = delta.squeeze(0)
        w   = w   + delta[:N] * Hs
        phi = phi + delta[n_max:n_max + N] * TWO_PI

        val = eval_grid_nrmse(kx, ky, w, phi, omega, fr['t'],
                               gx, gy, fr['gt_field'], Hs)
        if not np.isnan(val):
            ep_nrmse.append(val)

    return float(np.mean(ep_nrmse)) if ep_nrmse else float('nan')


# ─── Load episodes with caching ───────────────────────────────────────────────

def load_episodes(names, data_dir, verbose=True):
    items = []
    for name in names:
        path = os.path.join(data_dir, name)
        if not os.path.exists(path): continue
        cp = cache_path(data_dir, name)
        cached = os.path.exists(cp)
        try:
            data = build_or_load(path, data_dir, SEQ_LEN, SEQ_STRIDE, TP_FACTOR, SEED)
            pl, sm = pack(data['subsequences'], data['init_state'], data['Hs'], N_MAX, OBS_MAX)
            if not pl: continue
            items.append({**data, 'packed_list': pl, 'state_mask': sm})
            if verbose:
                print("  %s [%s] N=%d Hs=%.1f seqs=%d" % (
                    name[:40], 'cache' if cached else 'built',
                    data['N'], data['Hs'], len(pl)))
        except Exception as e:
            print("  SKIP %s: %s" % (name, e))
    return items


def discover_train(data_dir, test_eps, val_eps):
    exclude = set(test_eps) | set(val_eps)
    return sorted([f for f in os.listdir(data_dir)
                   if f.endswith('.h5') and f not in exclude
                   and any(f.startswith('episode_BF%d' % b) for b in [2,3,4,5])])


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    set_seed(SEED)

    print("=" * 70)
    print("KalmanNet × WaveSplat — Stage 3 v2 (Normalized + Dropout + Cache)")
    print("=" * 70)

    train_names = discover_train(data_dir, TEST_EPISODES, VAL_EPISODES)
    print("\n  Train: %d | Val: %d | Test: %d episodes" % (
        len(train_names), len(VAL_EPISODES), len(TEST_EPISODES)))

    t0 = time.time()
    print("\n[Loading train episodes (with cache)...]")
    train_items = load_episodes(train_names, data_dir, verbose=False)
    print("  Loaded %d episodes, %d seqs (%.0fs)" % (
        len(train_items), sum(len(it['packed_list']) for it in train_items),
        time.time() - t0))

    print("[Loading val episodes...]")
    val_items = load_episodes(VAL_EPISODES, data_dir, verbose=False)
    print("  Loaded %d episodes" % len(val_items))

    # ── Model ──
    model = KalmanNetDrop(n_max=N_MAX, obs_max=OBS_MAX,
                          embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, dropout=DROPOUT)
    log_sigma = nn.Parameter(torch.tensor(LOG_SIGMA0))
    params = list(model.parameters()) + [log_sigma]
    optimizer = optim.AdamW(params, lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)
    n_params = sum(p.numel() for p in model.parameters())
    print("\n  KalmanNetDrop: %d params, embed=%d, hidden=%d, dropout=%.1f" % (
        n_params, EMBED_DIM, HIDDEN_DIM, DROPOUT))

    # Flat training list
    all_train = []
    for it in train_items:
        for pf in it['packed_list']:
            all_train.append((it['init_state'], it['omega'], pf,
                               it['state_mask'], it['Hs']))

    # ── Training ──
    print("\n[Training %d epochs, patience=%d]" % (N_EPOCHS, PATIENCE))
    best_val_nll = float('inf')
    best_state = None; best_log_sigma = LOG_SIGMA0; no_improve = 0

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        random.shuffle(all_train)
        ep_nll = 0.0
        for init_s, omega, pf, smask, Hs in all_train:
            hidden = model.init_hidden(1)
            optimizer.zero_grad()
            loss, _ = rollout_nll(model, pf, init_s, omega, smask,
                                   log_sigma, Hs, N_MAX, OBS_MAX, hidden)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
            optimizer.step()
            ep_nll += loss.item()
        scheduler.step()
        avg_train = ep_nll / len(all_train)

        model.eval()
        v_total, v_count = 0.0, 0
        for it in val_items:
            for pf in it['packed_list']:
                with torch.no_grad():
                    hidden = model.init_hidden(1)
                    loss, _ = rollout_nll(model, pf, it['init_state'], it['omega'],
                                          it['state_mask'], log_sigma, it['Hs'],
                                          N_MAX, OBS_MAX, hidden)
                v_total += loss.item(); v_count += 1
        val_nll = v_total / max(v_count, 1)

        improved = ""
        if val_nll < best_val_nll:
            best_val_nll = val_nll
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_log_sigma = log_sigma.item()
            no_improve = 0; improved = " *"
            ckpt_dir = os.path.join(data_dir, '..', 'checkpoints')
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(best_state, os.path.join(ckpt_dir, 'kalmannet_v2_best.pt'))
        else:
            no_improve += 1

        if epoch % 20 == 0 or epoch == 1:
            print("  Epoch %3d/%d: train=%.4f val=%.4f σ=%.4f [%.0fs]%s" % (
                epoch, N_EPOCHS, avg_train, val_nll,
                log_sigma.exp().item(), time.time() - t0, improved))

        if no_improve >= PATIENCE:
            print("  Early stop @ epoch %d" % epoch); break

    print("  Best val NLL: %.4f" % best_val_nll)

    # ── Test ──
    print("\n[Test Evaluation]")
    model.load_state_dict(best_state)
    model.eval()
    test_items = load_episodes(TEST_EPISODES, data_dir, verbose=False)

    results = {}; all_nrmse = []
    for it in test_items:
        sub_nrmses = []
        for pf in it['packed_list']:
            val = rollout_nrmse_eval(model, pf, it['init_state'], it['omega'],
                                     it['state_mask'], it['Hs'], N_MAX, OBS_MAX,
                                     it['gt_grid_x'], it['gt_grid_y'])
            if not np.isnan(val): sub_nrmses.append(val)
        avg = float(np.mean(sub_nrmses)) if sub_nrmses else float('nan')
        ep_name = it['episode'].replace('episode_', '').replace('.h5', '')
        results[ep_name] = avg; all_nrmse.append(avg)
        print("  %s: %.4f" % (ep_name, avg))

    overall = float(np.mean(all_nrmse))
    print("\n  Avg test nRMSE: %.4f" % overall)
    print("  Reference: WaveSplat 0.223 | ETKF 0.319 | ConvCNP 0.244")

    out = {
        'version': 'kalmannet_stage3_v2',
        'best_val_nll': best_val_nll, 'test_avg_nrmse': overall,
        'test_per_episode': results, 'best_log_sigma': best_log_sigma,
        'n_train_episodes': len(train_items),
        'config': {'N_MAX': N_MAX, 'OBS_MAX': OBS_MAX, 'EMBED_DIM': EMBED_DIM,
                   'HIDDEN_DIM': HIDDEN_DIM, 'DROPOUT': DROPOUT,
                   'SEQ_LEN': SEQ_LEN, 'SEQ_STRIDE': SEQ_STRIDE, 'LR': LR},
    }
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    out_path = os.path.join(results_dir, 'kalmannet_stage3_v2.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print("Saved → %s" % out_path)

    print("\n" + "=" * 70)
    if overall < 0.223:
        print("PASS ✓  WaveSplat BEATEN! (%.4f < 0.223)" % overall)
    elif overall < 0.244:
        print("PASS ✓  Beats ConvCNP (%.4f < 0.244)" % overall)
    elif overall < 0.319:
        print("PASS ✓  Beats ETKF (%.4f < 0.319)" % overall)
    else:
        print("FAIL ✗  (%.4f ≥ 0.319)" % overall)
    print("=" * 70)


if __name__ == '__main__':
    main()
