"""
eval_tnse.py — T-NSE evaluation against GP K_disp and WaveSplat MAP.

Loads trained T-NSE checkpoint and evaluates on 30 test episodes.
Reports nRMSE, VR, Pearson per sea-state category.

Usage:
  python scripts/eval_tnse.py --ckpt results/tnse_stage2_best.pt
"""

import os, sys, json, time, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.model.tnse import TNSE, fit_amplitudes_phases, render_wave_field, MAX_PTS
from wavesplat.model.tnse_dataset import TNSEDataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
W = 10
N_TARGETS = 30

SEA_STATES = [('BF3', 5), ('BF4', 5), ('BF4', 15), ('BF4', 30), ('BF5', 15), ('BF5', 30)]
TEST_FILES = [(f'{ss}_{nc}c_s{s}', f'episode_{ss}_{nc}comp_seed{s}_uniform.h5')
              for ss, nc in SEA_STATES for s in range(5)]


def metrics(pred, gt, Hs):
    err = pred - gt
    nRMSE   = float(np.sqrt(np.mean(err**2)) / max(Hs, 1e-6))
    sp, sg  = float(np.std(pred)), float(np.std(gt))
    rho     = float(np.corrcoef(pred.ravel(), gt.ravel())[0, 1]) if sp > 1e-8 and sg > 1e-8 else 0.0
    vr      = float((sp / sg)**2) if sg > 1e-8 else 0.0
    skill   = float(1 - np.mean(err**2) / max(np.mean(gt**2), 1e-12))
    return nRMSE, rho, vr, skill


def eval_episode(tag, h5_name, model, cfg):
    loader = EpisodeDataLoader(os.path.join('data', h5_name))
    Hs = loader.Hs_true; depth = loader.depth
    gx = loader.gt_grid_x.numpy(); gy = loader.gt_grid_y.numpy()
    Tp = loader.Tp_true
    x_scale = float(gx.max() - gx.min()) or 250.0

    frames = loader.get_tracking_frames(tp_factor=2.0)
    n = len(frames)
    targets = list(range(W, min(n - 1, W + N_TARGETS)))

    r_list, rho_list, vr_list, sk_list = [], [], [], []
    inf_times = []

    model.eval()
    with torch.no_grad():
        for ti in targets:
            # Build input tokens
            frames_tokens = np.zeros((W, MAX_PTS, 5), dtype=np.float32)
            frames_mask   = np.ones( (W, MAX_PTS),    dtype=bool)
            frame_ids_np  = np.arange(W, dtype=np.int64)
            window_obs    = []

            t0_win = float(frames[ti - W + 1].t)   # relative-time ref (matches training)
            for fi, j in enumerate(range(ti - W + 1, ti + 1)):
                f = frames[j]
                x_raw = f.x_obs.numpy(); y_raw = f.y_obs.numpy()
                z_raw = f.z_obs.numpy(); conf  = f.confidence.numpy()
                t_raw = float(f.t)
                M = min(len(x_raw), MAX_PTS)
                if M == 0: continue
                frames_tokens[fi, :M, 0] = x_raw[:M] / x_scale
                frames_tokens[fi, :M, 1] = y_raw[:M] / x_scale
                frames_tokens[fi, :M, 2] = (t_raw - t0_win) / Tp
                frames_tokens[fi, :M, 3] = z_raw[:M] / Hs
                frames_tokens[fi, :M, 4] = conf[:M]
                frames_mask[fi, :M]      = False
                window_obs.extend(zip(x_raw[:M], y_raw[:M], [t_raw]*M, z_raw[:M]))

            ft = torch.from_numpy(frames_tokens).unsqueeze(0).to(DEVICE)
            fm = torch.from_numpy(frames_mask).unsqueeze(0).to(DEVICE)
            fi_t = torch.from_numpy(frame_ids_np).unsqueeze(0).to(DEVICE)

            t0 = time.time()
            kx_pred, ky_pred, conf_pred = model(ft, fm, fi_t)
            kx = kx_pred[0].cpu().numpy()
            ky = ky_pred[0].cpu().numpy()

            # Algebraic A/phi fit from window observations
            obs = np.array(window_obs, dtype=np.float32)
            if len(obs) < 2 * len(kx):
                continue
            A, phi = fit_amplitudes_phases(kx, ky, obs, depth, lambda_reg=0.5)
            inf_times.append(time.time() - t0)

            # Render at target frame
            target = frames[ti + 1]
            t_star = float(target.t)
            gt = target.gt_field.numpy()
            pred = render_wave_field(kx, ky, A, phi, depth, gx, gy, t_star)

            r, rho, vr, sk = metrics(pred, gt, Hs)
            r_list.append(r); rho_list.append(rho); vr_list.append(vr); sk_list.append(sk)

    return {
        'tag': tag,
        'nRMSE':   float(np.mean(r_list))   if r_list else 1.0,
        'Pearson': float(np.mean(rho_list)) if rho_list else 0.0,
        'VR':      float(np.mean(vr_list))  if vr_list else 0.0,
        'Skill':   float(np.mean(sk_list))  if sk_list else -1.0,
        'inf_ms':  float(np.mean(inf_times) * 1000) if inf_times else 0.0,
        'n_frames': len(r_list),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default='results/tnse_stage2_best.pt')
    args = parser.parse_args()

    print(f"Loading T-NSE from {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=DEVICE)
    cfg  = ckpt.get('cfg', {})

    model = TNSE(
        n_components=cfg.get('n_components', 15),
        depth=50.0,
        d_model=cfg.get('d_model', 128),
        n_heads=cfg.get('n_heads', 4),
        n_layers=cfg.get('n_layers', 4),
        d_ff=cfg.get('d_ff', 256),
        dropout=0.0,  # no dropout at eval
    ).to(DEVICE)
    model.load_state_dict(ckpt['model'])
    print(f"Checkpoint: epoch {ckpt.get('epoch')}, val nRMSE={ckpt.get('val_nRMSE', '?'):.4f}")

    print("=" * 65)
    all_results = {}
    skipped = []

    for tag, h5 in TEST_FILES:
        fpath = os.path.join('data', h5)
        if not os.path.exists(fpath) or os.path.getsize(fpath) < 1000:
            skipped.append(tag); continue

        print(f"[{tag}]", end='', flush=True)
        t0 = time.time()
        r = eval_episode(tag, h5, model, cfg)
        all_results[tag] = r
        elapsed = time.time() - t0
        print(f" nRMSE={r['nRMSE']:.4f}  P={r['Pearson']:.3f}  "
              f"VR={r['VR']:.3f}  Skill={r['Skill']:.3f}  ({elapsed:.0f}s)")

    # Per-sea-state breakdown
    ss_groups = {}
    for tag in all_results:
        ss_key = '_'.join(tag.split('_')[:2])
        ss_groups.setdefault(ss_key, []).append(tag)

    print("\n" + "=" * 65)
    print("PER-SEA-STATE SUMMARY")
    for ss, tags in sorted(ss_groups.items()):
        avg_nrmse = np.mean([all_results[t]['nRMSE'] for t in tags])
        avg_pear  = np.mean([all_results[t]['Pearson'] for t in tags])
        print(f"  {ss:12s}: nRMSE={avg_nrmse:.4f}  Pearson={avg_pear:.3f}  (n={len(tags)})")

    def gav(key):
        return float(np.mean([all_results[t][key] for t in all_results]))

    print(f"\nOVERALL ({len(all_results)}/30 episodes):")
    print(f"  T-NSE : nRMSE={gav('nRMSE'):.4f}  P={gav('Pearson'):.3f}  "
          f"VR={gav('VR'):.3f}  Skill={gav('Skill'):.3f}")
    print(f"  Inference: {gav('inf_ms'):.1f}ms / frame (neural + LS)")
    print(f"  Ref: GP K_disp ms5_reg_amp = 0.2232, WavePINO = 0.2279, Oracle = 0.1259")

    output = {
        'description': 'T-NSE evaluation, 30-episode protocol',
        'checkpoint': args.ckpt,
        'n_evaluated': len(all_results),
        'skipped': skipped,
        'per_episode': all_results,
        'per_sea_state': {ss: {
            'nRMSE': float(np.mean([all_results[t]['nRMSE'] for t in tags])),
            'Pearson': float(np.mean([all_results[t]['Pearson'] for t in tags])),
            'VR': float(np.mean([all_results[t]['VR'] for t in tags])),
        } for ss, tags in ss_groups.items()},
        'summary': {
            'nRMSE': gav('nRMSE'), 'Pearson': gav('Pearson'),
            'VR': gav('VR'), 'Skill': gav('Skill'), 'inf_ms': gav('inf_ms'),
        },
    }
    out_path = 'results/tnse_eval.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
