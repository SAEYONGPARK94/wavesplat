"""ONE-episode probe: is the 90%-energy prune CAUSAL for the crowded-regime loss?

Episode BF4_30comp_seed0_uniform, seed=123 -- the OURS cell that pruned 15->9 and
scored the worst nRMSE (0.335) in tmp/relax_extctrl_2026-08-05/relax_extctrl.json.

Arm A: shipped config  (prune_threshold=0.10)   -> must reproduce 0.335 / n=9
Arm B: prune knocked out (prune_threshold=0.0)  -> grow still active, prune cannot fire
       (target_fraction=1.0 => n_keep == N => `n_keep < n_before` is False)

Instrumentation: monkeypatch model.remove_gaussians to capture the pre-prune
(w,kx,ky) and the kept index set. No production file is modified.
"""
import io, json, os, sys, contextlib
import numpy as np, h5py, torch

ROOT = '/home/vexit/wavesplat/wavesplat'
sys.path.insert(0, ROOT)
from wavesplat.run import run_estimation
from wavesplat.model.spectral_gaussian import SpectralGaussianModel

EP = 'BF4_30comp_seed0_uniform'
H5 = f'{ROOT}/data_finemesh_2026-07-23/episode_{EP}.h5'
SEED = 123
BASE = {'use_dispersion': True, 'use_confidence': True, 'adaptive_n': True,
        'tracking_window': 3, 'init_method': 'v2_konly', 'seed': SEED}

with h5py.File(H5, 'r') as f:
    WC = f['config/wave_components'][:]          # [A, omega, kx, ky, phi, Q]
GT = dict(A=WC[:, 0], kx=WC[:, 2], ky=WC[:, 3])

EVENTS = []
_orig = SpectralGaussianModel.remove_gaussians
def patched(self, keep_idx):
    EVENTS.append(dict(
        n_before=int(self.n_gaussians),
        w=self.w.detach().cpu().numpy().copy(),
        kx=self.kx.detach().cpu().numpy().copy(),
        ky=self.ky.detach().cpu().numpy().copy(),
        keep=np.asarray(keep_idx.detach().cpu()).copy()))
    return _orig(self, keep_idx)
SpectralGaussianModel.remove_gaussians = patched

def go(tag, cfg):
    EVENTS.clear()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = run_estimation(H5, cfg, verbose=True)
    log = [l for l in buf.getvalue().splitlines() if 'Adaptive' in l or 'Refine' in l]
    print(f'--- {tag}: nRMSE={r["avg_nRMSE"]:.4f} init_nrmse={r["init_nrmse"]:.4f} '
          f'final_n={r.get("n_gaussians")}')
    for l in log: print('   ', l.strip())
    return r, [dict(e) for e in EVENTS]

rA, evA = go('A shipped (prune_threshold=0.10)', dict(BASE))
rB, evB = go('B prune OFF (prune_threshold=0.0)', dict(BASE, prune_threshold=0.0))

print('\n=== prune-event forensics (arm A) ===')
out = {'A': {'nRMSE': float(rA['avg_nRMSE']), 'init_nrmse': float(rA['init_nrmse']),
             'final_n': int(rA['n_gaussians'])},
       'B': {'nRMSE': float(rB['avg_nRMSE']), 'init_nrmse': float(rB['init_nrmse']),
             'final_n': int(rB['n_gaussians'])},
       'events_A': [], 'n_events_B': len(evB)}
for e in evA:
    w, kx, ky, keep = e['w'], e['kx'], e['ky'], e['keep']
    n = e['n_before']
    dropped = np.setdiff1d(np.arange(n), keep)
    en = w ** 2
    # distance of each estimated component to nearest GT component in (kx,ky)
    dk = np.hypot(kx[:, None] - GT['kx'][None, :], ky[:, None] - GT['ky'][None, :])
    j = dk.argmin(1); dmin = dk[np.arange(n), j]; gtA = GT['A'][j]
    rec = dict(n_before=n, n_keep=int(len(keep)),
               energy_frac_kept=float(en[keep].sum() / en.sum()),
               kept=dict(idx=keep.tolist(), dmin=np.round(dmin[keep], 3).tolist(),
                         w=np.round(np.abs(w[keep]), 4).tolist(),
                         gtA=np.round(gtA[keep], 4).tolist()),
               dropped=dict(idx=dropped.tolist(), dmin=np.round(dmin[dropped], 3).tolist(),
                            w=np.round(np.abs(w[dropped]), 4).tolist(),
                            gtA=np.round(gtA[dropped], 4).tolist()))
    out['events_A'].append(rec)
    print(f"  prune {n} -> {len(keep)}, energy kept {rec['energy_frac_kept']*100:.1f}%")
    print(f"    KEPT    |k-k_gt| median={np.median(dmin[keep]):.3f} "
          f"vals={np.round(np.sort(dmin[keep]),3).tolist()}")
    print(f"    DROPPED |k-k_gt| median={np.median(dmin[dropped]):.3f} "
          f"vals={np.round(np.sort(dmin[dropped]),3).tolist()}")
    print(f"    KEPT    |w| = {np.round(np.abs(w[keep]),4).tolist()}")
    print(f"    DROPPED |w| = {np.round(np.abs(w[dropped]),4).tolist()}")
    print(f"    GT amplitudes near KEPT   : {np.round(gtA[keep],3).tolist()}")
    print(f"    GT amplitudes near DROPPED: {np.round(gtA[dropped],3).tolist()}")

print(f"\nGT: n_comp={len(GT['A'])} |k| range "
      f"{np.hypot(GT['kx'],GT['ky']).min():.3f}-{np.hypot(GT['kx'],GT['ky']).max():.3f}, "
      f"A range {GT['A'].min():.4f}-{GT['A'].max():.4f}")
print(f"\nSTORED baseline for this cell: nRMSE=0.335 final_n=9 (relax_extctrl.json)")
json.dump(out, open(f'{ROOT}/tmp/adaptn_probe_2026-08-05/probe.json','w'), indent=1)
