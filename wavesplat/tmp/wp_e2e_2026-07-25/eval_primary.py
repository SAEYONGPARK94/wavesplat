"""PRIMARY + sparse-control eval (PREREG §4,§5b): proposal-level 'cond' nowcast, median-of-5.

Reuses tmp/wp_crowded_2026-07-25/run_cond_crowded.eval_episode VERBATIM so the protocol is
byte-identical to the head-to-head cond arm (4-sample BMA + 80 polish, 3 nowcast frames, /Hs).
Runs the FINE-TUNED ckpt on the 40 crowded + 12 sparse finemesh episodes, 5 sampling seeds.
Also re-runs the CURRENT ckpt on a 3-episode x 5-seed spot check to verify the recorded
head-to-head baseline reproduces (PREREG §4). Read-only ckpts; data_finemesh_2026-07-23/ only.
"""
import os, sys, json, time, argparse
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import torch

sys.path.insert(0, 'scripts')
sys.path.insert(0, 'tmp/wp_crowded_2026-07-25')
import d3_beamform_cond_diffusion as cd
import run_cond_crowded as rcc
torch.set_num_threads(4)

DEVICE = cd.DEVICE
FINE = 'data_finemesh_2026-07-23/'
SEEDS = (0, 1, 2, 3, 4)
CROWDED = rcc.CROWDED
SPARSE = rcc.SPARSE
VERIFY_EPS = ['episode_BF4_15comp_seed0_uniform.h5',
              'episode_BF5_30comp_seed3_uniform.h5',
              'episode_BF4_30comp_seed0_slope.h5']


def load_model(ckpt):
    m = cd.CondSetDenoiser().to(DEVICE)
    m.load_state_dict(torch.load(ckpt, map_location=DEVICE)); m.eval()
    return m


def run_arm(model, stats, abar, betas, eps, out_path, label):
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            try:
                d = json.loads(line); done.add((d['episode'], d['seed']))
            except Exception:
                pass
    jobs = [(e, s) for e in eps for s in SEEDS if (e, s) not in done]
    print(f"[{label}] {len(eps)} episodes, {len(jobs)} jobs ({len(done)} done)", flush=True)
    t0 = time.time(); n = 0
    with open(out_path, 'a', buffering=1) as f:
        for (ep, s) in jobs:
            tt = time.time()
            try:
                v = rcc.eval_episode(model, stats, abar, betas, FINE + ep, s)
                err = None
            except Exception as e:
                v = None; err = repr(e)
            row = dict(episode=ep, ncomp=rcc.ncomp_of(ep), seed=s,
                       cond_polish_nrmse_Hs=v, sec=time.time() - tt, err=err)
            f.write(json.dumps(row) + '\n'); n += 1
            print(f"[{label} {n:3d}/{len(jobs)}] {ep:40s} s{s} "
                  f"{'ERR' if err else f'{v:.3f}'} ({row['sec']:.1f}s)", flush=True)
    print(f"[{label}] done {n} runs, {time.time()-t0:.0f}s", flush=True)


def verify_current(stats, abar, betas):
    """Re-run current ckpt on VERIFY_EPS x 5 seeds; compare median-of-5 to recorded cond_runs."""
    rec = {}
    for line in open('tmp/wp_crowded_2026-07-25/cond_runs.jsonl'):
        d = json.loads(line)
        if d.get('err') is None and d['episode'] in VERIFY_EPS:
            rec.setdefault(d['episode'], []).append(d['cond_polish_nrmse_Hs'])
    model = load_model('results/direction3/beamform_cond_diffusion.pt')
    out = {}
    for ep in VERIFY_EPS:
        mine = [rcc.eval_episode(model, stats, abar, betas, FINE + ep, s) for s in SEEDS]
        mmed = float(np.median(mine)); rmed = float(np.median(rec[ep]))
        out[ep] = dict(mine_median=mmed, recorded_median=rmed, delta=mmed - rmed)
        print(f"  verify {ep:42s} mine {mmed:.4f} recorded {rmed:.4f} "
              f"Δ {mmed-rmed:+.4f}", flush=True)
    ok = all(abs(v['delta']) < 0.01 for v in out.values())
    print(f"  REPRODUCIBILITY: {'PASS' if ok else 'FAIL (re-run full current arm)'}", flush=True)
    json.dump(out, open('tmp/wp_e2e_2026-07-25/verify_current.json', 'w'), indent=2)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True, help='fine-tuned ckpt to eval as ft arm')
    ap.add_argument('--set', choices=['crowded', 'sparse', 'both', 'verify'], default='both')
    a = ap.parse_args()
    stats = torch.load(cd.PRIOR_CKPT, map_location=DEVICE)['stats']
    betas = cd.cosine_betas(cd.T).to(DEVICE); abar = torch.cumprod(1 - betas, 0)

    if a.set == 'verify':
        verify_current(stats, abar, betas); return

    model = load_model(a.ckpt)
    eps = []
    if a.set in ('crowded', 'both'):
        eps += CROWDED
    if a.set in ('sparse', 'both'):
        eps += SPARSE
    run_arm(model, stats, abar, betas, eps,
            'tmp/wp_e2e_2026-07-25/ft_cond_runs.jsonl', 'ft')


if __name__ == '__main__':
    main()
