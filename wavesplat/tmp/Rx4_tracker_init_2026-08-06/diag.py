"""Rx-4 STAGE 2 -- DIAGNOSIS (descriptive, no pre-registration needed).

Question: in the crowded stratum, at the ONLY shipping arm that beats the flat-sea null
(init_method='auto', analytic_ensemble_k=3), WHY does Phase 2 make a good initialisation
worse (B3: final-init = +0.0354 for AUTO vs +0.0015 for NOTRACK)?

What is instrumented (all read-only monkey-patches, undone in a finally block; production
defaults, configs/ and checkpoints untouched):

  A. TIME STRUCTURE   per-frame nRMSE trajectory vs the Phase-1 value.
  B. CHANNEL          per-frame orthogonal split m = (1-rho^2) + (sqrt(v)-rho)^2.
  C. PARAM DRIFT      ||phi_t - phi_0||, ||w_t - w_0||/||w_0||, and the DISPERSION RESIDUAL
                      max|omega_model - sqrt(g|k|tanh(|k|d))| (structurally 0; measured to
                      prove it, because the Rx-4 brief hypothesised w == angular frequency).
  D. COUNTERFACTUAL   at every frame the field is ALSO rendered with
                        (w_0, phi_0)   frozen        -- what NOTRACK would predict
                        (w_t, phi_0)   amplitude-only drift
                        (w_0, phi_t)   phase-only drift
                        (w_t, phi_t)   actual
                      => the damage is split into an amplitude channel and a phase channel
                      WITHOUT running any new cell.  kx/ky are frozen by tracking_mode
                      ='w_phi' so nothing else can move.
  E. CONCENTRATION    per-atom |dw|/|w_0| and |dphi| ranked by atom amplitude.

Runs on the ALREADY-CONSUMED seeds 1521/1522/1523 (B3's own seeds) so this probe consumes
NO new ledger seed and its `actual` trace must reproduce B3's AUTO numbers.
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
DATA = os.path.join(REPO, 'data_finemesh_2026-07-23')
OUT = os.path.join(HERE, 'diag.jsonl')

CROWDED4 = [
    'episode_BF4_15comp_seed0_uniform.h5',
    'episode_BF4_30comp_seed0_uniform.h5',
    'episode_BF5_15comp_seed0_uniform.h5',
    'episode_BF5_30comp_seed0_uniform.h5',
]
SEEDS = [1521, 1522, 1523]

# identical to run_b3.py BASE_CFG + the AUTO cell override (the shipped operating point)
AUTO_CFG = dict(init_method='auto', analytic_ensemble_k=3,
                use_dispersion=True, use_confidence=True,
                adaptive_n=True, tracking_window=3, prune_threshold=0.10, max_n=30,
                n_gaussians=[3, 8, 15], loss_thresholds=[0.1, 0.05, None],
                max_iters_per_stage=[200, 200, 500])

G = 9.81
STASH = {}


def _members(u):
    ms = getattr(u, '_members', None)
    return ms if ms else [u]


def _snap(u):
    out = []
    for m in _members(u):
        mo = m.model
        if mo is None:
            out.append(None)
            continue
        out.append({k: getattr(mo, k).detach().cpu().numpy().copy()
                    for k in ('w', 'kx', 'ky', 'phi')})
    return out


def _render(u, gx, gy, t, override):
    """Render the field with (w, phi) taken from `override` (list per member)."""
    import torch
    saved = []
    for m, ov in zip(_members(u), override):
        mo = m.model
        if mo is None or ov is None:
            saved.append(None)
            continue
        saved.append((mo.w.data.clone(), mo.phi.data.clone()))
        if ov.get('w') is not None:
            mo.w.data.copy_(torch.as_tensor(ov['w'], dtype=mo.w.dtype))
        if ov.get('phi') is not None:
            mo.phi.data.copy_(torch.as_tensor(ov['phi'], dtype=mo.phi.dtype))
    try:
        preds = [m.predict_field(gx, gy, t) for m in _members(u)]
        f = (torch.median(torch.stack(preds, 0), 0).values if len(preds) > 1
             else preds[0])
        return f.detach().cpu().numpy().copy()
    finally:
        for m, sv in zip(_members(u), saved):
            if sv is None:
                continue
            m.model.w.data.copy_(sv[0])
            m.model.phi.data.copy_(sv[1])


def run_one(job):
    sys.path.insert(0, REPO)
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ.setdefault(v, '1')
    import torch
    from wavesplat.run import run_estimation
    from wavesplat.optimizer.sequential_update import (
        SequentialUpdater, MedianEnsembleUpdater)

    base = {k: job[k] for k in ('job_id', 'episode', 'seed')}
    t0 = time.time()
    S = {'init': None, 'frames': [], 'depth': None}
    STASH.clear()
    STASH.update(S)

    orig_seq_init = SequentialUpdater.initialize
    orig_ens_init = MedianEnsembleUpdater.initialize
    orig_seq_pf = SequentialUpdater.predict_field
    orig_ens_pf = MedianEnsembleUpdater.predict_field
    depth_seen = {}

    def wrap_init(orig):
        def f(self, *a, **kw):
            r = orig(self, *a, **kw)
            if S['init'] is None:          # top-level (ensemble wraps members)
                S['init'] = _snap(self)
                S['depth'] = float(getattr(self, 'depth', 50.0))
                depth_seen['u'] = self
            return r
        return f

    busy = {'v': False}

    def wrap_pf(orig):
        def f(self, gx, gy, t):
            actual = orig(self, gx, gy, t)
            if busy['v'] or S['init'] is None or self is not depth_seen.get('u'):
                return actual
            busy['v'] = True
            try:
                cur = _snap(self)
                init = S['init']
                if any(c is None for c in cur):
                    return actual
                frozen = _render(self, gx, gy, t,
                                 [{'w': i['w'], 'phi': i['phi']} for i in init])
                w_only = _render(self, gx, gy, t,
                                 [{'w': c['w'], 'phi': i['phi']}
                                  for c, i in zip(cur, init)])
                p_only = _render(self, gx, gy, t,
                                 [{'w': i['w'], 'phi': c['phi']}
                                  for c, i in zip(cur, init)])
                d = float(S['depth'])
                drift = []
                for c, i in zip(cur, init):
                    n = min(len(c['w']), len(i['w']))
                    dw = c['w'][:n] - i['w'][:n]
                    dphi = np.angle(np.exp(1j * (c['phi'][:n] - i['phi'][:n])))
                    kmag = np.sqrt(c['kx'] ** 2 + c['ky'] ** 2 + 1e-12)
                    om_model = np.sqrt(G * kmag * np.tanh(kmag * d) + 1e-12)
                    kmag0 = np.sqrt(i['kx'][:n] ** 2 + i['ky'][:n] ** 2 + 1e-12)
                    om0 = np.sqrt(G * kmag0 * np.tanh(kmag0 * d) + 1e-12)
                    drift.append({
                        'n': int(n),
                        'dw_rel_l2': float(np.linalg.norm(dw)
                                           / max(np.linalg.norm(i['w'][:n]), 1e-12)),
                        'dphi_rms': float(np.sqrt(np.mean(dphi ** 2))),
                        'dk_l2': float(np.linalg.norm(
                            np.r_[c['kx'][:n] - i['kx'][:n],
                                  c['ky'][:n] - i['ky'][:n]])),
                        'domega_max': float(np.max(np.abs(om_model[:n] - om0)))
                        if n else 0.0,
                        'disp_resid_max': 0.0,   # filled below from the live model
                        'sumabs_w_ratio': float(np.sum(np.abs(c['w'][:n]))
                                                / max(np.sum(np.abs(i['w'][:n])), 1e-12)),
                        'dw_per_atom': [float(x) for x in dw],
                        'dphi_per_atom': [float(x) for x in dphi],
                        'w0_per_atom': [float(x) for x in i['w'][:n]],
                        'kmag0_per_atom': [float(x) for x in kmag0],
                    })
                # dispersion residual straight off the live model: |_omega() - sqrt(g k tanh kd)|
                for m, dr in zip(_members(self), drift):
                    with torch.no_grad():
                        om = m.model._omega().cpu().numpy()
                    km = np.sqrt(m.model.kx.detach().cpu().numpy() ** 2
                                 + m.model.ky.detach().cpu().numpy() ** 2 + 1e-12)
                    ref = np.sqrt(G * km * np.tanh(km * d) + 1e-12)
                    dr['disp_resid_max'] = float(np.max(np.abs(om - ref)))
                S['frames'].append({
                    't': float(t),
                    'pred_actual': actual.detach().cpu().numpy().copy(),
                    'pred_frozen': frozen,
                    'pred_w_only': w_only,
                    'pred_phi_only': p_only,
                    'drift': drift,
                })
            finally:
                busy['v'] = False
            return actual
        return f

    try:
        SequentialUpdater.initialize = wrap_init(orig_seq_init)
        MedianEnsembleUpdater.initialize = wrap_init(orig_ens_init)
        SequentialUpdater.predict_field = wrap_pf(orig_seq_pf)
        MedianEnsembleUpdater.predict_field = wrap_pf(orig_ens_pf)

        from wavesplat.data_loader import EpisodeDataLoader
        path = os.path.join(DATA, job['episode'])
        cfg = dict(AUTO_CFG)
        cfg['seed'] = job['seed']
        if job.get('notrack'):
            cfg['tracking_lr'] = 0.0
        r = run_estimation(path, cfg)

        loader = EpisodeDataLoader(path)
        Hs = float(loader.Hs_true)
        tf = loader.get_tracking_frames(tp_factor=2.0)
        gt_by_t = {round(float(f.t), 6): f.gt_field.cpu().numpy() for f in tf}

        def nrmse(a, b):
            return float(np.sqrt(np.mean((a - b) ** 2)) / Hs)

        traj = []
        # S['frames'][0] is the Phase-1 evaluation frame (last init frame); the rest are
        # the tracking frames, in order.
        for j, fr in enumerate(S['frames']):
            key = round(fr['t'], 6)
            gt = gt_by_t.get(key)
            rec = {'i': j - 1, 't': fr['t'], 'phase': 'init' if j == 0 else 'track'}
            if gt is not None:
                for tag in ('actual', 'frozen', 'w_only', 'phi_only'):
                    rec['nrmse_' + tag] = nrmse(fr['pred_' + tag], gt)
                p = fr['pred_actual'].ravel(); g = gt.ravel()
                rho = float(np.corrcoef(p, g)[0, 1])
                v = float(np.var(p) / max(np.var(g), 1e-12))
                rec['rho'] = rho
                rec['vr'] = v
                rec['decorr'] = float(1.0 - rho ** 2)
                rec['shrink'] = float((np.sqrt(max(v, 0.0)) - rho) ** 2)
            d0 = fr['drift'][0]
            rec.update({k: d0[k] for k in
                        ('n', 'dw_rel_l2', 'dphi_rms', 'dk_l2', 'domega_max',
                         'disp_resid_max', 'sumabs_w_ratio')})
            if j in (0, len(S['frames']) - 1) or j == len(S['frames']) // 2:
                rec['per_atom'] = {k: d0[k] for k in
                                   ('dw_per_atom', 'dphi_per_atom',
                                    'w0_per_atom', 'kmag0_per_atom')}
            traj.append(rec)

        frames = r['frame_results']
        return {**base, 'notrack': bool(job.get('notrack')),
                'Hs': Hs, 'n_members': len(S['init']),
                'auto_selected': r.get('auto_selected'),
                'ensemble_k': r.get('ensemble_k'),
                'N': r.get('n_gaussians', -1),
                'init_nrmse': float(r.get('init_nrmse', float('nan'))),
                'nRMSE': float(np.nanmean([f['nRMSE'] for f in frames])),
                'traj': traj,
                'wall_s': time.time() - t0}
    except Exception as e:
        import traceback
        return {**base, 'error': repr(e), 'tb': traceback.format_exc()[-2000:]}
    finally:
        SequentialUpdater.initialize = orig_seq_init
        MedianEnsembleUpdater.initialize = orig_ens_init
        SequentialUpdater.predict_field = orig_seq_pf
        MedianEnsembleUpdater.predict_field = orig_ens_pf


def _init_worker():
    sys.path.insert(0, REPO)
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ.setdefault(v, '1')


def main():
    import multiprocessing as mp
    jobs = []
    for nt in (False, True):
        for ep in CROWDED4:
            for s in SEEDS:
                tag = 'NOTRACK' if nt else 'AUTO'
                jobs.append({'job_id': f'{tag}|{ep}|{s}', 'episode': ep,
                             'seed': s, 'notrack': nt})
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if 'error' not in r:
                done.add(r['job_id'])
    todo = [j for j in jobs if j['job_id'] not in done]
    print(f'total {len(jobs)} | todo {len(todo)}', flush=True)
    if not todo:
        print('nothing to do'); return
    ctx = mp.get_context('spawn')
    t0 = time.time()
    with ctx.Pool(int(os.environ.get('WORKERS', '8')),
                  initializer=_init_worker) as pool, open(OUT, 'a') as fh:
        for i, rec in enumerate(pool.imap_unordered(run_one, todo, chunksize=1), 1):
            fh.write(json.dumps(rec) + '\n'); fh.flush()
            print(f"[{i}/{len(todo)}] {rec['job_id']} "
                  f"{rec.get('nRMSE', rec.get('error'))}", flush=True)
    print(f'DONE {(time.time()-t0)/60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
