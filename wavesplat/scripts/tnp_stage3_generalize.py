"""
tnp_stage3_generalize.py

Stage 3: Cross-episode generalization — Go/No-go #3
Train on multiple episodes, evaluate on held-out test episodes.
Same test set as WaveSplat/ConvCNP/Diffusion for fair comparison.
"""
import os
import sys
import time
import json
from glob import glob

import numpy as np
import torch

# Reuse model + dataset from Stage 2
sys.path.insert(0, os.path.dirname(__file__))
from tnp_stage2_overfit import (
    VanillaTNP, WaveTNPDataset, collate_fn, load_episode, evaluate_reconstruction,
)

# ─── Episode split (same test set as all previous experiments) ───
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


def get_episode_split(data_dir):
    """Split episodes into train/val/test."""
    all_eps = sorted(glob(os.path.join(data_dir, 'episode_BF*.h5')))
    test_set = set(TEST_EPISODES)
    val_set = set(VAL_EPISODES)

    train, val, test = [], [], []
    for p in all_eps:
        name = os.path.basename(p)
        if name in test_set:
            test.append(p)
        elif name in val_set:
            val.append(p)
        elif 'BF6' not in name and 'BF7' not in name:
            # Exclude BF6/BF7 from train (save for OOD test later)
            train.append(p)
    return train, val, test


def evaluate_per_episode(model, episode_paths, device, n_eval_frames=10, window=3):
    """Evaluate nRMSE per episode."""
    results = {}
    for ep_path in episode_paths:
        ep_name = os.path.basename(ep_path).replace('episode_', '').replace('.h5', '')
        ep = load_episode(ep_path)
        eval_res = evaluate_reconstruction(model, [ep], device, n_frames=n_eval_frames, window=window)
        results[ep_name] = eval_res['avg_nrmse']
        print(f'    {ep_name}: nRMSE={eval_res["avg_nrmse"]:.4f}, amp={eval_res["avg_amp_ratio"]:.3f}')
    avg = np.mean(list(results.values()))
    print(f'    Average: {avg:.4f}')
    return results, avg


def train_stage3(
    data_dir,
    epochs=300,
    batch_size=4,
    lr=3e-4,
    embed_dim=128,
    num_heads=4,
    num_layers=4,
    max_target=2048,
    patience=50,
    device='cuda',
):
    train_paths, val_paths, test_paths = get_episode_split(data_dir)
    print(f'Split: train={len(train_paths)}, val={len(val_paths)}, test={len(test_paths)}')

    # Load episodes
    print('Loading train episodes...')
    train_eps = [load_episode(p) for p in train_paths]
    print('Loading val episodes...')
    val_eps = [load_episode(p) for p in val_paths]

    train_ds = WaveTNPDataset(train_eps, window=3, stride=3, max_target=max_target, seed=42)
    val_ds = WaveTNPDataset(val_eps, window=3, stride=3, max_target=max_target, seed=123)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0,
    )
    print(f'Train tasks: {len(train_ds)}, Val tasks: {len(val_ds)}')
    print(f'Batches/epoch: train={len(train_loader)}, val={len(val_loader)}')

    model = VanillaTNP(
        dim_x=3, dim_y=1,
        embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model: VanillaTNP, {n_params:,} params ({n_params/1e6:.2f}M)')

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float('inf')
    no_improve = 0
    history = {'train': [], 'val': []}
    t_start = time.time()
    ckpt_path = '/tmp/tnp_stage3_best.pt'

    for epoch in range(epochs):
        # Train
        model.train()
        train_losses = []
        for xc, yc, xt, yt, mask in train_loader:
            xc, yc, xt, yt, mask = [t.to(device) for t in [xc, yc, xt, yt, mask]]
            optimizer.zero_grad()
            loss = model.nll_loss(xc, yc, xt, yt, context_mask=mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # Val
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xc, yc, xt, yt, mask in val_loader:
                xc, yc, xt, yt, mask = [t.to(device) for t in [xc, yc, xt, yt, mask]]
                loss = model.nll_loss(xc, yc, xt, yt, context_mask=mask)
                val_losses.append(loss.item())

        scheduler.step()
        train_avg = np.mean(train_losses)
        val_avg = np.mean(val_losses)
        history['train'].append(train_avg)
        history['val'].append(val_avg)

        if val_avg < best_val_loss:
            best_val_loss = val_avg
            torch.save(model.state_dict(), ckpt_path)
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            elapsed = time.time() - t_start
            print(f'  epoch {epoch+1:4d}/{epochs}: train={train_avg:.4f}  val={val_avg:.4f}  '
                  f'best_val={best_val_loss:.4f}  lr={scheduler.get_last_lr()[0]:.2e}  [{elapsed:.0f}s]')

        if no_improve >= patience:
            print(f'  Early stopping at epoch {epoch+1} (patience={patience})')
            break

    elapsed = time.time() - t_start
    print(f'\nTraining done in {elapsed:.0f}s. Best val NLL: {best_val_loss:.4f}')

    # ── Test evaluation ──
    print('\n── Test Set Evaluation ──')
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()

    test_results, test_avg = evaluate_per_episode(
        model, test_paths, device, n_eval_frames=10, window=3,
    )

    # Save results
    results = {
        'stage': 'Stage 3 generalization',
        'n_train_episodes': len(train_paths),
        'n_val_episodes': len(val_paths),
        'n_test_episodes': len(test_paths),
        'n_train_tasks': len(train_ds),
        'n_params': n_params,
        'epochs_trained': epoch + 1,
        'best_val_nll': best_val_loss,
        'test_avg_nrmse': test_avg,
        'test_per_episode': test_results,
        'training_time_s': elapsed,
        'loss_history_every10': {
            'train': [history['train'][i] for i in range(0, len(history['train']), 10)],
            'val': [history['val'][i] for i in range(0, len(history['val']), 10)],
        },
        'comparison': {
            'WaveSplat': 0.223,
            'ConvCNP_v2': 0.244,
            'WaveDiffusion_v3': 0.260,
        },
    }
    out_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'tnp_stage3_generalize.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved: {out_path}')

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--max_target', type=int, default=2048)
    parser.add_argument('--patience', type=int, default=50)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    train_stage3(
        data_dir=data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        max_target=args.max_target,
        patience=args.patience,
        device=device,
    )
