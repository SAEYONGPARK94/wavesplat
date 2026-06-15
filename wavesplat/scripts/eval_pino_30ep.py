"""
WavePINO S4 (MF T=3 gap=5) 재평가 — GP K_disp와 동일한 30-episode 프로토콜

동일 조건:
  - 30 episodes: seed 0-4 × {BF3_5c, BF4_5c, BF4_15c, BF4_30c, BF5_15c, BF5_30c} uniform
  - 30 tracking frames (tp_factor=2.0 이후, GP K_disp와 동일)
  - 동일 메트릭: nRMSE, VR, Pearson, Skill

결과: results/pino_s4_30ep_eval.json
"""
import os, sys, json
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wavesplat.data_loader import EpisodeDataLoader
from scripts.gp_disp_pilot import compute_metrics_simple, TEST_EPISODES_FULL, N_FRAMES

# Import WavePINO architecture from S4 script
from scripts.pino_stage4_multiframe import WavePINOMultiFrame, FNOBlock

# S4 config
FNO_WIDTH  = 16
FNO_LAYERS = 4
FNO_MODES_H = 16
FNO_MODES_W = 12
N_FRAMES_MODEL = 3
FRAME_GAP  = 5
CKPT_PATH  = 'checkpoints/wavepino_mf_T3_g5.pt'
OUT_PATH   = 'results/pino_s4_30ep_eval.json'
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_model():
    model = WavePINOMultiFrame(
        fno_width=FNO_WIDTH, fno_layers=FNO_LAYERS,
        fno_modes_h=FNO_MODES_H, fno_modes_w=FNO_MODES_W,
        n_frames=N_FRAMES_MODEL,
    ).to(DEVICE)
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state)
    model.eval()
    return model


def eval_episode_pino(h5_path, model, n_frames=N_FRAMES):
    loader = EpisodeDataLoader(h5_path)
    Hs = loader.Hs_true
    gt_x = loader.gt_grid_x.cpu()
    gt_y = loader.gt_grid_y.cpu()

    # Build target grid for model
    xx, yy = torch.meshgrid(gt_x, gt_y, indexing='ij')
    xt = torch.stack([xx.flatten(), yy.flatten()], dim=-1).unsqueeze(0).to(DEVICE)

    # Use same tracking frames as GP K_disp
    eval_frames = loader.get_tracking_frames(tp_factor=2.0)
    min_fi_needed = (N_FRAMES_MODEL - 1) * FRAME_GAP  # need history

    frame_metrics = []
    frame_count = 0

    for fi, frame in enumerate(loader):
        # Skip until we have enough history AND are in eval window
        global_fi = fi  # loader[fi] = frame
        pass

    # Use index-based access for multi-frame
    # get_tracking_frames returns frames; we need their global indices
    # Rebuild eval frame indices
    all_frames = list(loader)
    n_total = len(all_frames)

    # find tracking start index
    tp_factor = 2.0
    Tp = getattr(loader, 'Tp_true', 8.0)
    dt = getattr(loader, 'dt', 0.1)
    init_n = max(3, int(tp_factor * Tp / dt))
    start_idx = min(init_n, n_total - 1)

    count = 0
    for fi in range(start_idx, n_total):
        if count >= n_frames:
            break
        if fi < min_fi_needed:
            continue

        # Build multi-frame input
        xc_list, yc_list = [], []
        for t_idx in range(N_FRAMES_MODEL):
            src_fi = fi - (N_FRAMES_MODEL - 1 - t_idx) * FRAME_GAP
            if src_fi < 0:
                break
            fr = loader[src_fi]
            xc = torch.stack([fr.x_obs, fr.y_obs], dim=-1).unsqueeze(0).to(DEVICE)
            yc = fr.z_obs.unsqueeze(-1).unsqueeze(0).to(DEVICE) / max(Hs, 0.1)
            xc_list.append(xc)
            yc_list.append(yc)

        if len(xc_list) < N_FRAMES_MODEL:
            continue

        with torch.no_grad():
            pred = model(xc_list, yc_list, xt)

        pred_field = (pred[0, :, 0] * Hs).cpu().numpy()
        pred_field = pred_field.reshape(len(gt_x), len(gt_y)).T  # (Ny, Nx)

        fr_cur = loader[fi]
        gt = fr_cur.gt_field.cpu().numpy()
        m = compute_metrics_simple(pred_field, gt, Hs)
        frame_metrics.append(m)
        count += 1

    if not frame_metrics:
        return {'nRMSE': float('nan'), 'VR': float('nan'),
                'Pearson': float('nan'), 'Skill': float('nan')}

    return {k: float(np.nanmean([fm[k] for fm in frame_metrics]))
            for k in frame_metrics[0]}


def main():
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH) as f:
            results = json.load(f)
        done = len(results.get('per_episode', {}))
        print(f"Loaded: {done} episodes done")
    else:
        results = {'per_episode': {}, 'config': {
            'model': 'WavePINO-MF-S4', 'n_frames_model': N_FRAMES_MODEL,
            'frame_gap': FRAME_GAP, 'eval_frames': N_FRAMES,
            'protocol': '30 tracking frames after tp_factor=2.0 init window',
        }}

    per_ep = results.get('per_episode', {})

    print(f"Loading WavePINO S4 from {CKPT_PATH} on {DEVICE}")
    model = load_model()

    print("=" * 70)
    print("WavePINO S4 — 30 episodes, same protocol as GP K_disp")
    print("=" * 70)

    for h5_name in TEST_EPISODES_FULL:
        h5_path = os.path.join('data', h5_name)
        ep_key = h5_name.replace('episode_', '').replace('.h5', '')

        if ep_key in per_ep:
            m = per_ep[ep_key]
            print(f"  {ep_key:<35}: skip nRMSE={m['nRMSE']:.4f}")
            continue

        print(f"  {ep_key:<35}...", end='', flush=True)
        try:
            m = eval_episode_pino(h5_path, model)
            per_ep[ep_key] = m
            print(f" nRMSE={m['nRMSE']:.4f} VR={m['VR']:.3f} "
                  f"Pearson={m['Pearson']:.3f} Skill={m['Skill']:.3f}")
        except Exception as e:
            print(f" ERROR: {e}")
            import traceback; traceback.print_exc()

        avg = {k: float(np.nanmean([v[k] for v in per_ep.values()]))
               for k in next(iter(per_ep.values()))}
        results['per_episode'] = per_ep
        results['average'] = avg
        with open(OUT_PATH, 'w') as f:
            json.dump(results, f, indent=2)

    if 'average' in results:
        a = results['average']
        print("\n" + "=" * 70)
        print(f"WavePINO S4 FINAL ({len(per_ep)} ep): "
              f"nRMSE={a['nRMSE']:.4f} VR={a['VR']:.4f} "
              f"Pearson={a['Pearson']:.4f} Skill={a['Skill']:.4f}")
        print("=" * 70)
        print("\nFair comparison (30 ep, same protocol):")
        print(f"  WaveSplat:       nRMSE=0.2633")
        print(f"  ws_ms5_reg_amp:  nRMSE=0.2232")
        print(f"  WavePINO S4:     nRMSE={a['nRMSE']:.4f}  ← this run")
        print(f"  Oracle K_disp:   nRMSE=0.1259")

    print(f"\nSaved: {OUT_PATH}")


if __name__ == '__main__':
    main()
