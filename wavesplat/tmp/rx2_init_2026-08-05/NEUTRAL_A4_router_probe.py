"""NEUTRAL A4 probe — what would the SHIPPED default (init_method='auto') do on the 4
Rx-2 episodes?  Runs ONLY the router signal (learned proposal -> n_selected), not a full
estimation.  No pipeline state is mutated; nothing is written outside this directory.
"""
import json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.abspath(os.path.join(REPO, '..')))
DATA = os.path.join(REPO, 'data_finemesh_2026-07-23')

EPISODES = ['episode_BF4_15comp_seed0_uniform.h5', 'episode_BF4_30comp_seed0_uniform.h5',
            'episode_BF5_15comp_seed0_uniform.h5', 'episode_BF5_30comp_seed0_uniform.h5']
SEEDS = [42, 123, 456]

from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.optimizer.learned_init import propose_atoms, pool_and_select, DEFAULT_CKPT

THRESH = 8  # shipped auto_switch_thresh
out = []
for ep in EPISODES:
    p = os.path.join(DATA, ep)
    if not os.path.exists(p):
        out.append({'episode': ep, 'error': 'missing'}); continue
    loader = EpisodeDataLoader(p, device='cpu')
    frames = loader.get_init_window(tp_factor=2.0)
    for seed in SEEDS:
        t0 = time.time()
        try:
            rkx, rky, ren = propose_atoms(frames, loader.depth, ckpt=DEFAULT_CKPT,
                                          nsamp=4, diffusion_seed=seed)
            sk, _, _ = pool_and_select(rkx, rky, ren, energy_thresh=0.03,
                                       dedup_dk=0.03, max_n=30)
            n_sel = int(len(sk))
            rec = {'episode': ep, 'seed': seed, 'n_selected': n_sel,
                   'auto_would_route_to': 'learned' if n_sel >= THRESH else 'v2_konly',
                   'thresh': THRESH, 'n_frames': len(frames), 'wall_s': time.time() - t0}
        except Exception as e:
            rec = {'episode': ep, 'seed': seed, 'error': f'{type(e).__name__}: {e}'}
        print(json.dumps(rec), flush=True)
        out.append(rec)

json.dump(out, open(os.path.join(HERE, 'NEUTRAL_A4_router_probe.json'), 'w'), indent=1)
