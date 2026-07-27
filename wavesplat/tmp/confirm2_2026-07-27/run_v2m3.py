"""CONFIRM-2 arm runner: `v2_m3` = median-of-3 ANALYTIC ENSEMBLE.

Spec (frozen draft tmp/confirm2_2026-07-27/PREREG_CONFIRM2_draft.md Sec 2):

    In lane (est_seed=es, diff_seed=ds), run the STANDARD v2_konly estimation three
    times independently with est seeds {es, es+1000, es+2000}.  The ensemble
    prediction is the **pointwise (element-wise) median of the per-frame predicted
    elevation fields**; it is scored with the STANDARD metric code
    (wavesplat.evaluation.metrics.compute_all_metrics).  If any member run fails the
    whole lane is recorded as `err` (conservative).

WHY.  The analytic arm initialises component phases at random in two places, so a
fixed (episode, config) still gives a bimodal run-to-run spread (CV up to 53%,
tmp/ESTIMATOR_VARIANCE_RESULT_2026-07-23.md).  ~4% of sparse lanes land in the bad
mode and blow up 5-10x.  The documented prescription is median-of-k; this runner
implements it at the level of the *field*, not of the scalar score, so it is a
deployable estimator (no ground truth is used to combine members).

HOW THE FIELDS ARE OBTAINED WITHOUT TOUCHING wavesplat/.
`run_estimation` returns per-frame *metrics*, not the predicted fields.  Rather than
fork or re-implement its loop (which would risk protocol drift), this runner
temporarily rebinds two names in the `wavesplat.run` module namespace for the
duration of one call:

  * `wavesplat.run.compute_all_metrics` -> spy that stores a copy of `pred_field`
    (and keeps references to the exact `gt_field`, `grid_x`, `grid_y`, `Hs_true`,
    `obs_x`, `obs_y` arguments) and then delegates to the original function;
  * `wavesplat.run.nrmse` -> spy that stores the initialisation-frame prediction.

Both names are module-level imports inside run.py, each called exactly once per
frame / per run, so the capture is complete and side-effect free.  The originals are
restored in a `finally` block.  No file under wavesplat/ or wavesim/ is modified and
no default is changed; `TRACK_PHASE_PREDICT=False` and `init_method='v2_konly'` are
set explicitly at runtime, exactly as in the reference harness.

Output schema is identical to tmp/confirm40_2026-07-27/run_arms_confirm.py (episode,
est_seed, diff_seed, arm, init_nrmse, avg_nRMSE, final_n, raw_init_nrmse,
auto_selected, auto_n_selected, sec, flag, err, avg_<panel>) so score_confirm-style
scorers consume it unchanged.  Ensemble-only diagnostics are appended as extra keys
(member_seeds, member_nRMSE, member_final_n, member_sec, k, ens_sec, avg_inference_ms,
init_time_s) -- extra keys are ignored by the scorers.

Usage
    python3 tmp/confirm2_2026-07-27/run_v2m3.py \
        --data data_confirm2_2026-07-27/ --out tmp/confirm2_2026-07-27/arm_runs_v2m3.jsonl \
        --workers 6 --episodes all

`--k` exists ONLY for integrity gate I1 (k=1 must reproduce the standard single-seed
v2_konly lane bit-for-bit).  The pre-registered arm is k=3; do not judge on k!=3.
"""
import os
import sys
import json
import time
import argparse

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
# repo root on sys.path regardless of cwd (same guard as tmp/heldout40_2026-07-26/v2_arm.py)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                '..', '..')))
import warnings; warnings.filterwarnings('ignore')
from multiprocessing import Pool
import numpy as np

LANES = [(42, 0), (1, 1), (2, 2), (3, 3), (4, 4)]   # (estimator_seed, diffusion_seed)
MEMBER_OFFSETS = (0, 1000, 2000, 3000, 4000)         # frozen arm = first 3 (es, es+1000, es+2000);
                                                     # offsets 4-5 exist for k=5 characterization only
ARM = 'v2_m3'
PANEL_KEYS = ['Variance_Ratio', 'Pearson_rho', 'Skill_Score',
              'Directional_R2', 'Hs_error']


def panel_from_frames(frame_results):
    """Verbatim from tmp/confirm40_2026-07-27/run_arms_confirm.py."""
    out = {}
    for k in PANEL_KEYS:
        vals = [r[k] for r in frame_results
                if k in r and np.isfinite(r.get(k, np.nan))]
        out['avg_' + k] = float(np.mean(vals)) if vals else float('nan')
    return out


def _run_member(h5_path, est_seed):
    """One standard v2_konly run; returns (summary, init_capture, frame_captures).

    frame_captures[i] holds the exact arguments run.py handed to compute_all_metrics
    for tracking frame i, with `pred` copied out.
    """
    import wavesplat.run as wrun
    from wavesplat.evaluation.metrics import compute_all_metrics as _cam
    from wavesplat.evaluation.metrics import nrmse as _nrmse

    frames = []
    init_cap = []

    def _spy_metrics(pred_field, gt_field, grid_x, grid_y, Hs_true, **kw):
        frames.append({'pred': np.array(pred_field, copy=True), 'gt': gt_field,
                       'grid_x': grid_x, 'grid_y': grid_y, 'Hs': Hs_true, 'kw': kw})
        return _cam(pred_field, gt_field, grid_x, grid_y, Hs_true, **kw)

    def _spy_nrmse(pred, gt, Hs):
        init_cap.append({'pred': np.array(pred, copy=True), 'gt': gt, 'Hs': Hs})
        return _nrmse(pred, gt, Hs)

    cfg = {'adaptive_n': True, 'seed': est_seed, 'device': 'cpu',
           'init_method': 'v2_konly'}          # explicit; module defaults untouched
    wrun.compute_all_metrics = _spy_metrics
    wrun.nrmse = _spy_nrmse
    try:
        summary = wrun.run_estimation(h5_path, config=cfg)
    finally:
        wrun.compute_all_metrics = _cam
        wrun.nrmse = _nrmse

    if len(frames) != summary['tracking_frames']:
        raise RuntimeError(f'field capture incomplete: {len(frames)} captured vs '
                           f"{summary['tracking_frames']} tracking frames")
    return summary, (init_cap[0] if init_cap else None), frames


def _pointwise_median(stack_list):
    """Element-wise median across members, cast back to the members' dtype.

    For odd k the median IS one of the member values, so the cast is lossless and a
    k=1 ensemble is bit-identical to the single run (integrity gate I1).
    """
    dt = stack_list[0].dtype
    if len(stack_list) == 1:
        med = np.median(stack_list[0][None, ...], axis=0)
    else:
        med = np.median(np.stack(stack_list, axis=0), axis=0)
    return med.astype(dt, copy=False)


def _worker(args):
    ep, est_seed, diff_seed, data_dir, k = args
    import wavesplat.optimizer.sequential_update as su
    su.TRACK_PHASE_PREDICT = False           # phase-fix; module default untouched
    from wavesplat.evaluation.metrics import compute_all_metrics, nrmse

    t0 = time.time()
    seeds = [est_seed + off for off in MEMBER_OFFSETS[:k]]
    base = dict(episode=ep, est_seed=est_seed, diff_seed=diff_seed, arm=ARM,
                k=k, member_seeds=seeds)
    try:
        h5 = os.path.join(data_dir, ep)
        members = []
        for s in seeds:
            ts = time.time()
            summ, init_cap, frames = _run_member(h5, s)
            members.append({'seed': s, 'summary': summ, 'init': init_cap,
                            'frames': frames, 'sec': time.time() - ts})

        # --- consistency of the scoring targets across members (must be identical) ---
        n_fr = len(members[0]['frames'])
        for m in members[1:]:
            if len(m['frames']) != n_fr:
                raise RuntimeError('member frame counts differ')
        ref = members[0]['frames']
        for m in members[1:]:
            for i in range(n_fr):
                if not np.array_equal(m['frames'][i]['gt'], ref[i]['gt']):
                    raise RuntimeError(f'member GT fields differ at frame {i}')
                if m['frames'][i]['Hs'] != ref[i]['Hs']:
                    raise RuntimeError(f'member Hs differ at frame {i}')

        # --- pointwise-median ensemble, scored with the standard metric code ---
        t_ens = time.time()
        ens_frames = []
        for i in range(n_fr):
            med = _pointwise_median([m['frames'][i]['pred'] for m in members])
            met = compute_all_metrics(med, ref[i]['gt'], ref[i]['grid_x'],
                                      ref[i]['grid_y'], ref[i]['Hs'], **ref[i]['kw'])
            ens_frames.append(met)
        ens_sec = time.time() - t_ens

        avg_nrmse = float(np.mean([f['nRMSE'] for f in ens_frames]))
        panel = panel_from_frames(ens_frames)

        # init-frame ensemble (same rule) -- reporting only, not a bar
        init_nrmse = None
        if all(m['init'] is not None for m in members):
            med0 = _pointwise_median([m['init']['pred'] for m in members])
            init_nrmse = float(nrmse(med0, members[0]['init']['gt'],
                                     members[0]['init']['Hs']))

        fin = [int(m['summary']['n_gaussians']) for m in members]
        row = dict(base,
                   init_nrmse=init_nrmse,
                   avg_nRMSE=avg_nrmse,
                   final_n=int(np.median(fin)),          # members can differ; median
                   raw_init_nrmse=None,
                   auto_selected=None, auto_n_selected=None,
                   sec=time.time() - t0, flag=su.TRACK_PHASE_PREDICT, err=None,
                   member_nRMSE=[float(m['summary']['avg_nRMSE']) for m in members],
                   member_obs_loss=[float(m['summary'].get('avg_obs_loss', float('nan')))
                                    for m in members],
                   member_final_n=fin,
                   member_sec=[float(m['sec']) for m in members],
                   member_init_nrmse=[float(m['summary']['init_nrmse']) for m in members],
                   init_time_s=[float(m['summary']['init_time_s']) for m in members],
                   avg_inference_ms=[float(m['summary']['avg_inference_ms']) for m in members],
                   ens_sec=ens_sec, n_frames=n_fr,
                   **panel)
        return row
    except Exception as e:
        return dict(base, init_nrmse=None, avg_nRMSE=None, final_n=None,
                    raw_init_nrmse=None, auto_selected=None, auto_n_selected=None,
                    sec=time.time() - t0, flag=None, err=repr(e))


def select_episodes(data_dir, spec):
    eps = sorted(f for f in os.listdir(data_dir) if f.endswith('.h5'))
    if spec == 'all':
        return eps
    if spec == 'crowded':
        return [e for e in eps if '15comp' in e or '30comp' in e]
    if spec == 'onecomp':
        return [e for e in eps if '_1comp_' in e]
    if spec == 'sparse':
        return [e for e in eps if '_1comp_' in e or '_2comp_' in e]
    named = [s.strip() for s in spec.split(',') if s.strip()]
    missing = [s for s in named if s not in eps]
    if missing:
        raise SystemExit(f'episodes not found in {data_dir}: {missing}')
    return named


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='episode directory')
    ap.add_argument('--out', required=True, help='output jsonl')
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--episodes', default='all',
                    help="'all' | 'crowded' | 'onecomp' | 'sparse' | comma-separated filenames")
    ap.add_argument('--lanes', default='all',
                    help="'all' or comma-separated es:ds pairs, e.g. '42:0,4:4' "
                         '(integrity gates only; the arm is the 5 frozen lanes)')
    ap.add_argument('--k', type=int, default=3,
                    help='ensemble size; 3 = pre-registered spec. k=1 is the I1 gate.')
    a = ap.parse_args()

    if a.k < 1 or a.k > len(MEMBER_OFFSETS):
        raise SystemExit(f'--k must be 1..{len(MEMBER_OFFSETS)}')
    if a.k != 3:
        print(f'*** WARNING: k={a.k} is NOT the pre-registered arm (k=3). '
              'Integrity-gate use only. ***', flush=True)

    data_dir = a.data if a.data.endswith('/') else a.data + '/'
    eps = select_episodes(data_dir, a.episodes)
    lanes = LANES if a.lanes == 'all' else [tuple(int(x) for x in p.split(':'))
                                            for p in a.lanes.split(',')]
    print(f"data {data_dir} | episodes {len(eps)} | lanes {lanes} | k={a.k}", flush=True)

    jobs = [(ep, es, ds, data_dir, a.k) for ep in eps for (es, ds) in lanes]

    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            try:
                d = json.loads(line)
                if d.get('err') is None and d.get('k') == a.k:
                    done.add((d['episode'], d['est_seed'], d['diff_seed']))
            except Exception:
                pass
    jobs = [j for j in jobs if (j[0], j[1], j[2]) not in done]
    print(f'{len(eps)} episodes x {len(lanes)} lanes; {len(jobs)} runs to do '
          f'({len(done)} done), {a.workers} workers', flush=True)
    if not jobs:
        return

    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    t0 = time.time(); n = 0
    with open(a.out, 'a', buffering=1) as f, Pool(a.workers) as pool:
        for res in pool.imap_unordered(_worker, jobs):
            f.write(json.dumps(res) + '\n')
            n += 1
            tag = ('ERR ' + res['err'][:50] if res['err']
                   else f"nRMSE {res['avg_nRMSE']:.4f} N{res['final_n']}")
            print(f"[{n:3d}/{len(jobs)}] {ARM:7s} {res['episode']:44s} "
                  f"e{res['est_seed']}d{res['diff_seed']} {tag} "
                  f"({res['sec']:.0f}s)", flush=True)
    print(f'done: {n} runs, {time.time()-t0:.0f}s wall', flush=True)


if __name__ == '__main__':
    main()
