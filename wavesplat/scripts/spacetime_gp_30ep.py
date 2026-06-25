"""
spacetime_gp_30ep.py — Space-Time dispersion-fused GP: full 30-episode evaluation.

Extends spacetime_gp_practical.py (6-ep pilot) to the canonical 30-episode protocol
(6 sea states × 5 seeds, uniform raydrop) used by direction_a_report_ko.md.

Methods compared per episode:
  - per-frame GP K_disp   (static kernel, single frame)
  - space-time fused GP   (W-frame window, dispersion kernel, estimated comps)
  - WaveSplat MAP         (existing method, from cached result or live run)
  - fused oracle          (theoretical ceiling)

Output: results/spacetime_gp_30ep.json
"""
import os
import sys
import json
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from wavesplat.data_loader import EpisodeDataLoader
from scripts.gp_disp_pilot import estimate_components_wavesplat_opt, estimate_components_beamform

G = 9.81
SIGMA_OBS = 0.02
COMP_MODE = 'ws_reg_amp'  # cheapest mode: n_starts=1
W = 10                     # temporal window frames
N_TARGETS = 30             # evaluation frames per episode

# 30 episodes: 6 sea states × 5 seeds, uniform raydrop
SEA_STATES = [
    ('BF3',  5),
    ('BF4',  5),
    ('BF4', 15),
    ('BF4', 30),
    ('BF5', 15),
    ('BF5', 30),
]

TEST_EPISODES = [
    (f'{ss}_{nc}c_s{seed}', f'episode_{ss}_{nc}comp_seed{seed}_uniform.h5')
    for ss, nc in SEA_STATES
    for seed in range(5)
]


# ── Kernel & GP utilities ────────────────────────────────────────────────────

def comps_to_arrays(comps, depth):
    A = comps[:, 0]; kx = comps[:, 2]; ky = comps[:, 3]
    k = np.sqrt(kx**2 + ky**2 + 1e-12)
    omega = np.sqrt(G * k * np.tanh(k * depth))
    return A, omega, kx, ky


def oracle_arrays(loader):
    wc = np.array(loader.wave_components)
    return wc[:, 0], wc[:, 1], wc[:, 2], wc[:, 3]


def build_K_static(P1, P2, A, kx, ky):
    dx = P1[:, 0:1] - P2[None, :, 0]
    dy = P1[:, 1:2] - P2[None, :, 1]
    K = np.zeros((P1.shape[0], P2.shape[0]))
    for n in range(len(A)):
        K += 0.5 * A[n]**2 * np.cos(kx[n]*dx + ky[n]*dy)
    return K


def build_K_spacetime(P1, P2, A, omega, kx, ky):
    dx = P1[:, 0:1] - P2[None, :, 0]
    dy = P1[:, 1:2] - P2[None, :, 1]
    dt = P1[:, 2:3] - P2[None, :, 2]
    K = np.zeros((P1.shape[0], P2.shape[0]))
    for n in range(len(A)):
        K += 0.5 * A[n]**2 * np.cos(kx[n]*dx + ky[n]*dy - omega[n]*dt)
    return K


def gp_mean(K_oo, K_po, z, sigma=SIGMA_OBS):
    N = len(z)
    A = K_oo + (sigma**2 + 1e-6) * np.eye(N)
    try:
        alpha = np.linalg.solve(A, z)
    except np.linalg.LinAlgError:
        alpha = np.linalg.lstsq(A, z, rcond=None)[0]
    return K_po @ alpha


def metrics(pred, gt, Hs):
    err = pred - gt
    nRMSE = float(np.sqrt(np.mean(err**2)) / max(Hs, 1e-6))
    sp = float(np.std(pred)); sg = float(np.std(gt))
    rho = float(np.corrcoef(pred.ravel(), gt.ravel())[0, 1]) if sp > 1e-8 and sg > 1e-8 else 0.0
    vr = float((sp / sg)**2) if sg > 1e-8 else 0.0
    skill = float(1.0 - np.mean(err**2) / max(np.mean(gt**2), 1e-12))
    return nRMSE, rho, vr, skill


def frame_obs(f):
    x = f.x_obs.cpu().numpy(); y = f.y_obs.cpu().numpy(); z = f.z_obs.cpu().numpy()
    return np.column_stack([x, y, np.full_like(x, float(f.t))]), z


# ── Per-episode evaluation ───────────────────────────────────────────────────

def eval_episode(tag, h5_name):
    loader = EpisodeDataLoader(os.path.join('data', h5_name))
    Hs = loader.Hs_true; depth = loader.depth
    gx = loader.gt_grid_x.cpu().numpy(); gy = loader.gt_grid_y.cpu().numpy()
    Xg, Yg = np.meshgrid(gx, gy)
    grid_xy = np.column_stack([Xg.ravel(), Yg.ravel()])

    # Component estimation (ws_reg_amp: same as GP K_disp ms1 variant)
    t0 = time.time()
    comps = estimate_components_wavesplat_opt(
        loader, apply_reg_amp=True, n_starts=1, lambda_reg=0.5)
    est_time = time.time() - t0
    Ae, oe, kxe, kye = comps_to_arrays(comps, depth)
    Ao, oo, kxo, kyo = oracle_arrays(loader)

    frames = loader.get_tracking_frames(tp_factor=2.0)
    n = len(frames)
    targets = list(range(W, min(n, W + N_TARGETS)))

    pf_r, pf_rho, pf_vr, pf_sk = [], [], [], []
    fu_r, fu_rho, fu_vr, fu_sk = [], [], [], []
    fo_r = []

    for ti in targets:
        target = frames[ti]
        t_star = float(target.t)
        gt = target.gt_field.cpu().numpy()
        grid_pts = np.column_stack([grid_xy, np.full(len(grid_xy), t_star)])

        # per-frame static GP
        P1, z1 = frame_obs(target)
        K_oo = build_K_static(P1[:, :2], P1[:, :2], Ae, kxe, kye)
        K_po = build_K_static(grid_xy, P1[:, :2], Ae, kxe, kye)
        pred = gp_mean(K_oo, K_po, z1).reshape(gt.shape)
        r, rho, vr, sk = metrics(pred, gt, Hs)
        pf_r.append(r); pf_rho.append(rho); pf_vr.append(vr); pf_sk.append(sk)

        # space-time fused GP
        Ps, zs = [], []
        for j in range(ti - W + 1, ti + 1):
            Pj, zj = frame_obs(frames[j]); Ps.append(Pj); zs.append(zj)
        Pw = np.vstack(Ps); zw = np.concatenate(zs)
        K_oo = build_K_spacetime(Pw, Pw, Ae, oe, kxe, kye)
        K_po = build_K_spacetime(grid_pts, Pw, Ae, oe, kxe, kye)
        pred = gp_mean(K_oo, K_po, zw).reshape(gt.shape)
        r, rho, vr, sk = metrics(pred, gt, Hs)
        fu_r.append(r); fu_rho.append(rho); fu_vr.append(vr); fu_sk.append(sk)

        # fused oracle
        K_oo = build_K_spacetime(Pw, Pw, Ao, oo, kxo, kyo)
        K_po = build_K_spacetime(grid_pts, Pw, Ao, oo, kxo, kyo)
        pred = gp_mean(K_oo, K_po, zw).reshape(gt.shape)
        r, _, _, _ = metrics(pred, gt, Hs); fo_r.append(r)

    return {
        'tag': tag, 'comp_mode': COMP_MODE, 'W': W,
        'n_frames_eval': len(targets),
        'est_time_sec': round(est_time, 2),
        'perframe': {
            'nRMSE': float(np.mean(pf_r)),
            'Pearson': float(np.mean(pf_rho)),
            'VR': float(np.mean(pf_vr)),
            'Skill': float(np.mean(pf_sk)),
        },
        'fused': {
            'nRMSE': float(np.mean(fu_r)),
            'Pearson': float(np.mean(fu_rho)),
            'VR': float(np.mean(fu_vr)),
            'Skill': float(np.mean(fu_sk)),
        },
        'fused_oracle': {'nRMSE': float(np.mean(fo_r))},
        'fusion_gain_pct': float(
            (np.mean(pf_r) - np.mean(fu_r)) / max(np.mean(pf_r), 1e-6) * 100),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Space-Time Dispersion-Fused GP — 30-Episode Evaluation")
    print(f"comp_mode={COMP_MODE}, W={W}, n_targets={N_TARGETS}")
    print("=" * 70)

    all_results = {}
    skipped = []

    for tag, h5 in TEST_EPISODES:
        fpath = os.path.join('data', h5)
        if not os.path.exists(fpath) or os.path.getsize(fpath) < 1000:
            print(f"SKIP {tag} (file not ready)")
            skipped.append(tag)
            continue

        print(f"\n[{tag}]")
        t0 = time.time()
        try:
            r = eval_episode(tag, h5)
            all_results[tag] = r
            elapsed = time.time() - t0
            print(f"  per-frame: nRMSE={r['perframe']['nRMSE']:.4f}  "
                  f"P={r['perframe']['Pearson']:.3f}  VR={r['perframe']['VR']:.3f}")
            print(f"  fused    : nRMSE={r['fused']['nRMSE']:.4f}  "
                  f"P={r['fused']['Pearson']:.3f}  VR={r['fused']['VR']:.3f}  "
                  f"[{r['fusion_gain_pct']:+.1f}%]")
            print(f"  oracle   : nRMSE={r['fused_oracle']['nRMSE']:.4f}  "
                  f"({elapsed:.0f}s total)")
        except Exception as e:
            print(f"  ERROR: {e}")
            skipped.append(tag)

    if not all_results:
        print("\nNo episodes evaluated. Generate data first.")
        return

    # ── Aggregate ────────────────────────────────────────────────────────────
    def avg(key, sub):
        vals = [all_results[t][sub][key] for t in all_results if sub in all_results[t]]
        return float(np.mean(vals)) if vals else float('nan')

    # Per sea-state breakdown
    print("\n" + "=" * 70)
    print("PER-SEA-STATE BREAKDOWN (mean over seeds)")
    ss_groups = {}
    for tag in all_results:
        ss_key = '_'.join(tag.split('_')[:2])  # e.g. BF4_15c
        ss_groups.setdefault(ss_key, []).append(tag)

    for ss, tags in sorted(ss_groups.items()):
        pf = np.mean([all_results[t]['perframe']['nRMSE'] for t in tags])
        fu = np.mean([all_results[t]['fused']['nRMSE'] for t in tags])
        fo = np.mean([all_results[t]['fused_oracle']['nRMSE'] for t in tags])
        gain = (pf - fu) / max(pf, 1e-6) * 100
        print(f"  {ss:12s}: per-frame={pf:.4f}  fused={fu:.4f} [{gain:+.1f}%]  oracle={fo:.4f}")

    print("\n" + "=" * 70)
    print(f"OVERALL SUMMARY ({len(all_results)}/30 episodes)")
    pf_avg = avg('nRMSE', 'perframe')
    fu_avg = avg('nRMSE', 'fused')
    fo_avg = avg('nRMSE', 'fused_oracle')
    gain_avg = float(np.mean([all_results[t]['fusion_gain_pct'] for t in all_results]))
    print(f"  per-frame GP K_disp : nRMSE={pf_avg:.4f}  "
          f"P={avg('Pearson','perframe'):.3f}  VR={avg('VR','perframe'):.3f}  Skill={avg('Skill','perframe'):.3f}")
    print(f"  fused space-time GP : nRMSE={fu_avg:.4f}  "
          f"P={avg('Pearson','fused'):.3f}  VR={avg('VR','fused'):.3f}  Skill={avg('Skill','fused'):.3f}")
    print(f"  fused oracle        : nRMSE={fo_avg:.4f}")
    print(f"  >>> mean fusion gain: {gain_avg:+.1f}%")
    if skipped:
        print(f"  (skipped: {skipped})")

    # ── Save ─────────────────────────────────────────────────────────────────
    output = {
        'description': 'Space-time dispersion-fused GP, 30-episode evaluation',
        'config': {'comp_mode': COMP_MODE, 'W': W, 'n_targets': N_TARGETS,
                   'sigma_obs': SIGMA_OBS},
        'n_evaluated': len(all_results),
        'n_skipped': len(skipped),
        'skipped': skipped,
        'per_episode': all_results,
        'per_sea_state': {
            ss: {
                'perframe_nRMSE': float(np.mean([all_results[t]['perframe']['nRMSE'] for t in tags])),
                'fused_nRMSE': float(np.mean([all_results[t]['fused']['nRMSE'] for t in tags])),
                'oracle_nRMSE': float(np.mean([all_results[t]['fused_oracle']['nRMSE'] for t in tags])),
                'n': len(tags),
            }
            for ss, tags in ss_groups.items()
        },
        'summary': {
            'perframe_nRMSE': pf_avg,
            'perframe_Pearson': avg('Pearson', 'perframe'),
            'perframe_VR': avg('VR', 'perframe'),
            'perframe_Skill': avg('Skill', 'perframe'),
            'fused_nRMSE': fu_avg,
            'fused_Pearson': avg('Pearson', 'fused'),
            'fused_VR': avg('VR', 'fused'),
            'fused_Skill': avg('Skill', 'fused'),
            'fused_oracle_nRMSE': fo_avg,
            'mean_fusion_gain_pct': gain_avg,
        },
    }
    out_path = 'results/spacetime_gp_30ep.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
