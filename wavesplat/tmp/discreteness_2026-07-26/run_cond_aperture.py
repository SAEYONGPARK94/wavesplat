"""Learned (cond-diffusion) arm on the APERTURE cells. `eval_episode` reused VERBATIM.

No code change is needed for the aperture lever: `nb = round(K_TP*Tp/0.1) = 154` is a FRAME count,
so at dt=0.4 s it spans 61.6 s instead of 15.4 s -- the same 4x widening the WS arm gets from
tp_factor 2.0->8.0, at the same frame/observation count. This is ASSERTED here, not assumed.
"""
import os, sys, json, time, argparse
import warnings; warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
sys.path.insert(0, 'scripts')
sys.path.insert(0, 'tmp/wp_crowded_2026-07-25')

import numpy as np
import torch
import d3_beamform_cond_diffusion as cd
from run_cond_crowded import eval_episode

DATA = 'tmp/discreteness_2026-07-26/data'
CELLS = ['abig', 'cbig']
SEEDS = (0, 1, 2, 3, 4)
EPS = [f'episode_BF4_15comp_seed{s}_{r}.h5'
       for s in (20, 21, 22, 23, 24) for r in ('uniform', 'slope')]


def assert_window(path):
    from wavesplat.data_loader import EpisodeDataLoader
    L = EpisodeDataLoader(path)
    nb = int(round(cd.K_TP * L.Tp_true / 0.1))
    i0, i1 = nb + 3, len(L) - 2
    sel = np.linspace(i0, i1, 3).astype(int)
    lo = max(0, int(sel[0]) - nb)
    span = float(L[int(sel[0])].t) - float(L[lo].t)
    return nb, int(sel[0]) - lo + 1, span


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='tmp/discreteness_2026-07-26/cond_runs_aperture.jsonl')
    ap.add_argument('--ckpt', default='results/direction3/beamform_cond_diffusion.pt')
    a = ap.parse_args()

    stats = torch.load(cd.PRIOR_CKPT, map_location=cd.DEVICE)['stats']
    betas = cd.cosine_betas(cd.T).to(cd.DEVICE)
    abar = torch.cumprod(1 - betas, 0)
    model = cd.CondSetDenoiser().to(cd.DEVICE)
    model.load_state_dict(torch.load(a.ckpt, map_location=cd.DEVICE))
    model.eval()

    jobs = [(c, ep, s) for c in CELLS for ep in EPS for s in SEEDS]
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            try:
                d = json.loads(line)
                if d.get('err') is None:
                    done.add((d['cell'], d['episode'], d['seed']))
            except Exception:
                pass
    jobs = [j for j in jobs if j not in done]
    jobs = [j for j in jobs if os.path.exists(f'{DATA}/cell_{j[0]}/{j[1]}')]

    probe = f"{DATA}/cell_{jobs[0][0]}/{jobs[0][1]}"
    nb, nfr, span = assert_window(probe)
    print(f"cond aperture window check: nb={nb} frames, gathered={nfr}, span={span:.1f}s", flush=True)
    assert nb == 154 and abs(span - 61.6) < 0.05, "cond aperture window assertion failed"

    print(f"cond aperture arm on {cd.DEVICE}: {len(jobs)} jobs", flush=True)
    t0 = time.time(); n = 0
    with open(a.out, 'a', buffering=1) as f:
        for (c, ep, s) in jobs:
            tt = time.time()
            try:
                v = eval_episode(model, stats, abar, betas, f'{DATA}/cell_{c}/{ep}', s)
                err = None
            except Exception as e:
                v = None; err = repr(e)
            row = dict(cell=c, episode=ep, seed=s, cond_polish_nrmse_Hs=v,
                       nb_frames=nb, window_span_s=span, sec=time.time() - tt, err=err)
            f.write(json.dumps(row) + '\n'); n += 1
            print(f"[{n:3d}/{len(jobs)}] {c:5s} {ep:38s} s{s} "
                  f"cond {'ERR' if err else f'{v:.3f}'} ({row['sec']:.1f}s)", flush=True)
    print(f"cond aperture arm done: {n} runs, {time.time()-t0:.0f}s wall", flush=True)


if __name__ == '__main__':
    main()
