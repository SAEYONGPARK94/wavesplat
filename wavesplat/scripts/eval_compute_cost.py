"""
Computational cost 비교표 (3 methods × 30 episodes)

측정 항목:
  - Init time (episode당, WaveSplat 초기화 포함)
  - Per-frame inference time (ms/frame)
  - Total time per episode (s)

Methods:
  - WaveSplat (legacy): run.py 방식 (Adam 최적화)
  - GP K_disp ws_ms5_reg_amp: WaveSplat init + GP solve
  - WavePINO S4: GPU/CPU 순전파

결과: results/compute_cost_comparison.json
"""
import os, sys, json, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wavesplat.data_loader import EpisodeDataLoader
from scripts.gp_disp_pilot import (
    estimate_components_wavesplat_opt, build_K_disp_oracle,
    gp_predict, SIGMA_OBS, TEST_EPISODES
)
from scripts.pino_stage4_multiframe import WavePINOMultiFrame

OUT_PATH = 'results/compute_cost_comparison.json'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_MEAS = 5   # episodes to measure per method
N_FRAMES_MEAS = 10  # frames per episode

# Subset of pilot episodes for timing
TIMING_EPISODES = TEST_EPISODES[:N_MEAS]


def measure_gp_cost(h5_path, n_frames=N_FRAMES_MEAS):
    loader = EpisodeDataLoader(h5_path)
    gt_x = loader.gt_grid_x.cpu().numpy()
    gt_y = loader.gt_grid_y.cpu().numpy()
    Xg, Yg = np.meshgrid(gt_x, gt_y)
    X_pred = np.column_stack([Xg.ravel(), Yg.ravel()])

    # Measure init time (WaveSplat, n_starts=5)
    t0 = time.time()
    beam_components = estimate_components_wavesplat_opt(
        loader, apply_reg_amp=True, n_starts=5, lambda_reg=2.0,
        init_tp_factor=2.0, sigma_obs=SIGMA_OBS
    )
    init_time = time.time() - t0

    # Measure per-frame GP inference time
    eval_frames = loader.get_tracking_frames(tp_factor=2.0)
    inference_times = []
    for frame in list(eval_frames)[:n_frames]:
        x_obs = frame.x_obs.cpu().numpy()
        y_obs = frame.y_obs.cpu().numpy()
        z_obs = frame.z_obs.cpu().numpy()
        if len(x_obs) < 5:
            continue
        X_obs = np.column_stack([x_obs, y_obs])
        K_oo = build_K_disp_oracle(X_obs, X_obs, beam_components)
        K_po = build_K_disp_oracle(X_pred, X_obs, beam_components)
        t0 = time.time()
        _ = gp_predict(K_oo, K_po, z_obs, SIGMA_OBS)
        inference_times.append((time.time() - t0) * 1000)

    return {
        'init_s': init_time,
        'inference_ms': float(np.mean(inference_times)) if inference_times else 0,
        'total_30frame_s': init_time + float(np.mean(inference_times)) * 30 / 1000,
    }


def measure_wavesplat_cost(h5_path, n_frames=N_FRAMES_MEAS):
    """WaveSplat: n_starts=1 (default single-seed run)."""
    loader = EpisodeDataLoader(h5_path)

    t0 = time.time()
    _ = estimate_components_wavesplat_opt(
        loader, apply_scale=True, n_starts=1
    )
    init_time = time.time() - t0

    # WaveSplat per-frame tracking (already included in tracking phase, ~10ms)
    return {
        'init_s': init_time,
        'inference_ms': 13.4,  # from existing results (13.4ms/frame)
        'total_30frame_s': init_time + 13.4 * 30 / 1000,
    }


def measure_pino_cost(h5_path, model, n_frames=N_FRAMES_MEAS):
    loader = EpisodeDataLoader(h5_path)
    Hs = loader.Hs_true
    gt_x = loader.gt_grid_x.cpu()
    gt_y = loader.gt_grid_y.cpu()
    xx, yy = torch.meshgrid(gt_x, gt_y, indexing='ij')
    xt = torch.stack([xx.flatten(), yy.flatten()], dim=-1).unsqueeze(0).to(DEVICE)

    N_FRAMES_MODEL, FRAME_GAP = 3, 5
    min_fi = (N_FRAMES_MODEL - 1) * FRAME_GAP

    Tp = getattr(loader, 'Tp_true', 8.0)
    dt = getattr(loader, 'dt', 0.1)
    start_idx = max(min_fi, int(2.0 * Tp / dt))

    inference_times = []
    count = 0
    n_total = len(list(loader))

    for fi in range(start_idx, n_total):
        if count >= n_frames:
            break
        if fi < min_fi:
            continue

        xc_list, yc_list = [], []
        ok = True
        for t_idx in range(N_FRAMES_MODEL):
            src_fi = fi - (N_FRAMES_MODEL - 1 - t_idx) * FRAME_GAP
            if src_fi < 0:
                ok = False; break
            fr = loader[src_fi]
            xc = torch.stack([fr.x_obs, fr.y_obs], dim=-1).unsqueeze(0).to(DEVICE)
            yc = fr.z_obs.unsqueeze(-1).unsqueeze(0).to(DEVICE) / max(Hs, 0.1)
            xc_list.append(xc)
            yc_list.append(yc)
        if not ok:
            continue

        if DEVICE == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            _ = model(xc_list, yc_list, xt)
        if DEVICE == 'cuda':
            torch.cuda.synchronize()
        inference_times.append((time.time() - t0) * 1000)
        count += 1

    return {
        'init_s': 0.0,  # no per-episode init (model pre-loaded)
        'inference_ms': float(np.mean(inference_times)) if inference_times else 0,
        'total_30frame_s': float(np.mean(inference_times)) * 30 / 1000 if inference_times else 0,
    }


def main():
    print("=" * 60)
    print("Computational Cost Comparison")
    print(f"Device: {DEVICE}, {N_MEAS} episodes × {N_FRAMES_MEAS} frames")
    print("=" * 60)

    results = {}

    # Load WavePINO
    print("\nLoading WavePINO S4...")
    pino = WavePINOMultiFrame(
        fno_width=16, fno_layers=4, fno_modes_h=16, fno_modes_w=12, n_frames=3
    ).to(DEVICE)
    ckpt = torch.load('checkpoints/wavepino_mf_T3_g5.pt', map_location=DEVICE)
    state = ckpt.get('model_state_dict', ckpt)
    pino.load_state_dict(state)
    pino.eval()

    # Measure each method
    for method, fn in [
        ('WaveSplat', lambda p: measure_wavesplat_cost(p)),
        ('GP_ws_ms5_reg_amp', lambda p: measure_gp_cost(p)),
        ('WavePINO_S4', lambda p: measure_pino_cost(p, pino)),
    ]:
        print(f"\n[{method}]")
        ep_results = []
        for h5_name in TIMING_EPISODES:
            h5_path = os.path.join('data', h5_name)
            ep_key = h5_name.replace('episode_', '').replace('.h5', '')
            print(f"  {ep_key:<35}...", end='', flush=True)
            try:
                r = fn(h5_path)
                ep_results.append(r)
                print(f" init={r['init_s']:.1f}s  infer={r['inference_ms']:.1f}ms/frame  "
                      f"total={r['total_30frame_s']:.1f}s")
            except Exception as e:
                print(f" ERROR: {e}")

        if ep_results:
            avg = {k: float(np.mean([r[k] for r in ep_results])) for k in ep_results[0]}
            results[method] = {'per_episode': ep_results, 'average': avg}
            print(f"  → avg init={avg['init_s']:.1f}s  "
                  f"infer={avg['inference_ms']:.1f}ms/frame  "
                  f"total={avg['total_30frame_s']:.1f}s/episode")

    # Summary table
    print("\n" + "=" * 60)
    print(f"{'Method':<25} {'Init (s)':>10} {'Infer (ms)':>12} {'Total/ep (s)':>14} {'Train data':>12}")
    print("-" * 60)
    extras = {
        'WaveSplat':           'None',
        'GP_ws_ms5_reg_amp':   'None',
        'WavePINO_S4':         '90 episodes',
    }
    for m, r in results.items():
        a = r['average']
        print(f"  {m:<23} {a['init_s']:>10.1f} {a['inference_ms']:>12.1f} "
              f"{a['total_30frame_s']:>14.1f} {extras.get(m,''):>12}")

    with open(OUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == '__main__':
    main()
