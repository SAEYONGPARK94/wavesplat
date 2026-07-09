"""Full 30-ep eval of iterative (2-pass) beamform-detect vs RELAX_full / DPS / oracle.
Run: python3 tmp/beamform_full_eval.py [--npass 2]
"""
import os, sys, json, argparse, numpy as np, torch
sys.path.insert(0, 'scripts')
from d3_beamform_net import DetectCNN, beamform, extract, DEVICE, M_OBS
from d17_recursive_estimator import refine_params
from wavesplat.model.tnse import fit_amplitudes_phases, render_wave_field
from d6_integrated_eval import tag_to_file
from d15_sequential_fusion import gather_frames
from d23_main_results import K_TP, panel
from wavesplat.data_loader import EpisodeDataLoader

ap = argparse.ArgumentParser(); ap.add_argument('--npass', type=int, default=2); a = ap.parse_args()
G = 9.81
m = DetectCNN().to(DEVICE); m.load_state_dict(torch.load('results/direction3/beamform_net.pt', map_location=DEVICE)); m.eval()
gp = json.load(open('results/spacetime_gp_30ep.json'))['per_episode']
relax_iter = json.load(open('results/direction3/main_results_iter.json'))
relax_full = json.load(open('results/direction3/main_results.json'))
dps = json.load(open('results/direction3/dps_highm.json'))['per_episode']  # _long @ m=3000


def _om(kx, ky, d): k = np.sqrt(kx**2 + ky**2) + 1e-9; return np.sqrt(G * k * np.tanh(k * d))
def _ro(kx, ky, A, phi, P, d):
    o = _om(kx, ky, d); psi = kx[None]*P[:, 0][:, None] + ky[None]*P[:, 1][:, None] - o[None]*P[:, 2][:, None] + phi[None]
    return (A[None] * np.cos(psi)).sum(1)


def bf_iter(P, z, depth, tgt, npass):
    t0 = P[:, 2].min(); Ps = P.copy(); Ps[:, 2] -= t0
    T = dict(tgt); T['t'] = tgt['t'] - t0
    obs = np.column_stack([Ps[:, 0], Ps[:, 1], Ps[:, 2], z]).astype(np.float32)
    akx, aky = np.zeros(0), np.zeros(0); zres = z.copy()
    for p in range(npass):
        with torch.no_grad():
            B = beamform(torch.tensor(Ps[None], device=DEVICE, dtype=torch.float32),
                         torch.tensor(zres[None], device=DEVICE, dtype=torch.float32))
            kx, ky = extract(m(B)[0])
        if len(kx): akx = np.concatenate([akx, kx]); aky = np.concatenate([aky, ky])
        if len(akx) == 0: continue
        akx, aky, fld = refine_params(akx, aky, Ps, z, depth, T)
        A, phi = fit_amplitudes_phases(akx, aky, obs, depth, lambda_reg=0.5)
        zres = z - _ro(akx, aky, A, phi, Ps, depth)
    if len(akx) == 0: return 1.0
    A, phi = fit_amplitudes_phases(akx, aky, obs, depth, lambda_reg=0.5)
    return panel(render_wave_field(akx, aky, A, phi, depth, tgt['gx'], tgt['gy'], T['t']), tgt['gt'], tgt['Hs'])['nrmse']


out = {}
for ti, tag in enumerate(gp.keys()):
    L = EpisodeDataLoader('data/' + tag_to_file(tag)); depth = L.depth; Hs = L.Hs_true
    gx = L.gt_grid_x.numpy(); gy = L.gt_grid_y.numpy(); nb = int(round(K_TP * L.Tp_true / 0.1))
    i0 = nb + 3; i1 = len(L) - 2
    if i1 - i0 < 4: i0 = max(nb + 1, len(L) - 60)
    nr = []
    for i_star in np.linspace(i0, i1, 5).astype(int):
        f = L[int(i_star)]; P, z = gather_frames(L, int(i_star), nb)
        if len(z) > M_OBS: idx = np.random.default_rng(0).choice(len(z), M_OBS, replace=False); P, z = P[idx], z[idx]
        tgt = {'gx': gx, 'gy': gy, 't': float(f.t), 'gt': f.gt_field.numpy(), 'Hs': Hs}
        nr.append(bf_iter(P, z, depth, tgt, a.npass))
    out[tag] = dict(bf=float(np.mean(nr)), relax_full=relax_iter[tag]['deployable']['nrmse'][0],
                    dps=dps[tag]['dps'], oracle=relax_full[tag]['oracle']['nrmse'][0])
    print(f"[{ti+1:2d}/30] {tag:12s} bf-iter {out[tag]['bf']:.3f} | RELAX {out[tag]['relax_full']:.3f} | "
          f"DPS {out[tag]['dps']:.3f} | ora {out[tag]['oracle']:.3f}", flush=True)

seas = {}
for tag, r in out.items():
    s = '_'.join(tag.split('_')[:2]); seas.setdefault(s, []).append(r)
print(f"\n=== per sea-state (iterative beamform-detect, {a.npass}-pass, m≤{M_OBS}) ===")
print(f"{'sea':9s} {'bf-iter':>8} {'RELAX':>7} {'DPS':>7} {'oracle':>7}")
A = {k: [] for k in ['bf', 'rf', 'dps', 'o']}
for s in sorted(seas):
    rs = seas[s]; mm = lambda f: float(np.mean([f(r) for r in rs]))
    row = (mm(lambda r: r['bf']), mm(lambda r: r['relax_full']), mm(lambda r: r['dps']), mm(lambda r: r['oracle']))
    print(f"{s:9s} {row[0]:8.3f} {row[1]:7.3f} {row[2]:7.3f} {row[3]:7.3f}")
    for k, v in zip(['bf', 'rf', 'dps', 'o'], row): A[k].append(v)
mn = lambda k: np.mean(A[k])
print(f"\nOVERALL  bf-iter {mn('bf'):.3f} | RELAX {mn('rf'):.3f} | DPS {mn('dps'):.3f} | oracle {mn('o'):.3f}")
json.dump(out, open('results/direction3/beamform_iter_eval.json', 'w'), indent=2)
print("saved results/direction3/beamform_iter_eval.json")
