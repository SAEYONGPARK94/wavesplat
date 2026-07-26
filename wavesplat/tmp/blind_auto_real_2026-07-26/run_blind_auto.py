"""W5 — the shipped `auto` policy routed live on the 30 real windows.

Protocol frozen in tmp/blind_auto_real_2026-07-26/PREREG.md (read §0: this is an auditable
composition record, NOT a blind test). The router is the SHIPPED
SequentialUpdater._auto_select_init, called directly on FrameData built from the real
scene's observations. Arm scores are reused from wp_real.json (declared, §1.2).

  python3 tmp/blind_auto_real_2026-07-26/run_blind_auto.py
"""
import os, sys, json, time, hashlib
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

OUT = os.path.join(ROOT, 'tmp/blind_auto_real_2026-07-26/blind_auto_real.json')
PREREG = os.path.join(ROOT, 'tmp/blind_auto_real_2026-07-26/PREREG.md')
WPREAL = os.path.join(ROOT, 'tmp/wp_real_2026-07-25/wp_real.json')

RECORDS = ['YS17', 'AA15a', 'AA14', 'AA15b', 'LJ18g']     # wp_real order
WINS = list(range(6))
COLLAPSE = {('AA14', 0), ('AA15a', 3), ('LJ18g', 3)}
PRIMARY_SEED = 42                 # wavesplat/run.py DEFAULT_CONFIG['seed'] -> diffusion seed
DIFF_SEEDS = (42, 0, 1, 2, 3, 4)  # 42 = primary (shipped); 0-4 = declared diagnostic

# frozen shipped constants (PREREG §1 / B4 lint)
SHIPPED = dict(init_method='auto', auto_switch_thresh=8, auto_guard=True,
               auto_guard_ratio=0.05, learned_energy_thresh=0.03,
               learned_dedup_dk=0.03, max_n=30, learned_nsamp=4)


def frames_from_scene(sc):
    """Format adapter ONLY: real scattered observations -> FrameData grouped by time.
    No value altered, nothing filtered, no mean removed."""
    from wavesplat.data_loader import FrameData
    P = np.asarray(sc['P'], float); z = np.asarray(sc['z'], float)
    frames = []
    for tt in np.unique(P[:, 2]):
        m = P[:, 2] == tt
        n = int(m.sum())
        frames.append(FrameData(
            t=float(tt),
            x_obs=torch.tensor(P[m, 0], dtype=torch.float32),
            y_obs=torch.tensor(P[m, 1], dtype=torch.float32),
            z_obs=torch.tensor(z[m], dtype=torch.float32),
            confidence=torch.ones(n, dtype=torch.float32),
            incidence_angles=torch.zeros(n, dtype=torch.float32)))
    return frames


def main():
    with open(PREREG, 'rb') as f:
        ph = hashlib.sha256(f.read()).hexdigest()
    print(f"# PREREG sha256 {ph[:16]}", flush=True)

    # ---- B4 lint: assert the shipped defaults are what we claim ----
    from wavesplat.optimizer.sequential_update import SequentialUpdater
    import wavesplat.optimizer.sequential_update as su
    lint = {}
    src = open(os.path.join(ROOT, 'wavesplat/optimizer/sequential_update.py')).read()
    for k, v in [('auto_switch_thresh', 8), ('auto_guard', True), ('auto_guard_ratio', 0.05)]:
        lint[k] = f"cfg.get('{k}', {v!r})" in src or f'cfg.get(\'{k}\', {v})' in src
    from wavesplat.run import DEFAULT_CONFIG
    lint['DEFAULT_CONFIG_init_method'] = DEFAULT_CONFIG.get('init_method')
    lint['DEFAULT_CONFIG_seed'] = DEFAULT_CONFIG.get('seed')
    lint['primary_diffusion_seed'] = PRIMARY_SEED
    lint['seed_matches_default'] = (DEFAULT_CONFIG.get('seed') == PRIMARY_SEED)
    print(f"# B4 lint: {lint}", flush=True)

    wp = {(r['record'], r['win']): r for r in json.load(open(WPREAL))['results']}

    import d3_compare_gifs as cg
    rows = []
    t_all = time.time()
    for rec in RECORDS:
        for win in WINS:
            tw = time.time()
            sc = cg.scene_real(rec, win=win)
            frames = frames_from_scene(sc)
            per_seed = {}
            for ds in DIFF_SEEDS:
                cfg = dict(SHIPPED)
                # seed=ds and NO explicit learned_diffusion_seed: _auto_select_init's own
                # default (self.seed) supplies the diffusion seed, exactly as shipped.
                cfg.update(depth=float(sc['depth']), seed=int(ds))
                upd = SequentialUpdater(cfg)
                route = upd._auto_select_init(frames, verbose=False)
                per_seed[str(ds)] = dict(route=route,
                                         n_selected=upd._auto_n_selected,
                                         bf_peak_ratio=upd._auto_bf_peak_ratio,
                                         guard_fired=upd._auto_guard_fired)
            prim = per_seed[str(PRIMARY_SEED)]
            h = wp[(rec, win)]
            policy = h['learned_now'] if prim['route'] == 'learned' else h['hand_now']
            r = dict(record=rec, win=win, collapse=(rec, win) in COLLAPSE,
                     route=prim['route'], n_selected=prim['n_selected'],
                     bf_peak_ratio=prim['bf_peak_ratio'],
                     guard_fired=prim['guard_fired'],
                     hand_now=h['hand_now'], learned_now=h['learned_now'],
                     policy_now=policy, per_seed=per_seed, sec=time.time() - tw)
            rows.append(r)
            agree = sum(per_seed[str(d)]["route"] == prim["route"] for d in DIFF_SEEDS)
            print(f"  [{rec} w{win}] n_sel {prim['n_selected']:>2} bfr "
                  f"{prim['bf_peak_ratio'] if prim['bf_peak_ratio'] is None else round(prim['bf_peak_ratio'],4)}"
                  f" guard {prim['guard_fired']} -> {prim['route']:9s}"
                  f" | hand {h['hand_now']:.3f} learned {h['learned_now']:.3f}"
                  f" policy {policy:.3f} | seed-agree {agree}/6 ({r['sec']:.0f}s)", flush=True)

    json.dump(dict(rows=rows, prereg=PREREG, prereg_sha256=ph, shipped=SHIPPED,
                   lint=lint, diff_seeds=list(DIFF_SEEDS),
                   arm_source=WPREAL, elapsed_s=time.time() - t_all),
              open(OUT, 'w'), indent=1)
    print(f"\nwrote {OUT}: {len(rows)} rows, {time.time()-t_all:.0f}s", flush=True)


if __name__ == '__main__':
    main()
