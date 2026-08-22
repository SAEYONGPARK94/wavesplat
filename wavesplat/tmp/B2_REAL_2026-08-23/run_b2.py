"""B-2 러너 — 실측에서 제안 표현이 무엇을 하는가. 규칙은 `gates_b2.py` 단일출처."""
import hashlib, json, os, sys, time, warnings
warnings.filterwarnings('ignore')
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
for p in (HERE, ROOT, os.path.join(ROOT, 'scripts'),
          os.path.join(ROOT, 'tmp', 'bankmmse_port_2026-07-29')):
    sys.path.insert(0, p)
os.chdir(ROOT)
import gates_b2 as G                     # noqa: E402
from scene_station import scene_station   # noqa: E402
import make_gif_bank_deployed as M        # noqa: E402
OUT = os.path.join(HERE, 'results.jsonl')


def frames_from(sc, t_shift):
    """`M.frames_from_scene` 과 같은 규약 — 관측 (P,z) 를 FrameData 로."""
    return M.frames_from_scene(sc, t_shift, dtype=torch.float32)


def run(record, arm):
    from wavesplat.optimizer.sequential_update import make_updater
    from wavesplat.run import DEFAULT_CONFIG, set_seed
    t0w = time.time()
    sc = scene_station(record, win=G.WIN)
    t_shift = float(np.asarray(sc['P'])[:, 2].min())
    t0 = float(sc['Pq'][0, 2]) - t_shift
    frames = frames_from(sc, t_shift)
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(depth=float(sc['depth']), seed=G.SEED, device='cpu')
    cfg.update(G.ARMS[arm])
    set_seed(G.SEED)
    upd = make_updater(cfg)
    res = upd.initialize(frames,
                         x_range=(float(sc['gxq'].min()), float(sc['gxq'].max())),
                         y_range=(float(sc['gyq'].min()), float(sc['gyq'].max())),
                         verbose=False)
    gxq, gyq = np.asarray(sc['gxq']), np.asarray(sc['gyq'])
    curve = []
    for li, l_ in enumerate(sc['leads']):
        g = np.asarray(sc['maps_gt'][li], float); mm = np.isfinite(g)
        if not mm.any():
            curve.append(float('nan')); continue
        Z = upd.predict_field(torch.tensor(gxq, dtype=torch.float32),
                              torch.tensor(gyq, dtype=torch.float32),
                              t0 + float(l_)).cpu().numpy().reshape(len(gyq), len(gxq))
        curve.append(float(np.sqrt(np.mean((Z[mm] - g[mm]) ** 2)) / sc['sg']))
    h = lambda a: hashlib.md5(np.ascontiguousarray(a, dtype=np.float64)).hexdigest()[:12]
    return dict(cell=arm, record=record, win=G.WIN,
                n_obs=int(len(sc['z'])), npx=int(sc['npx']),
                obs_hash=h(np.column_stack([sc['P'], sc['z']])), q_hash=h(sc['Pq']),
                n_atoms=(getattr(getattr(upd, 'model', None), 'n_gaussians', 0) or 0),
                varpro_fired=bool((res or {}).get('varpro')),
                varpro_steps=(res or {}).get('varpro_steps'),
                init_iters=(res or {}).get('n_iters'),
                real_gate=getattr(upd, '_auto_real_gate', None),
                nowcast=curve[0], leadmean=float(np.nanmean(curve)),
                curve=curve, Hs=float(sc['Hs']), sigma=float(sc['sg']),
                wall_s=time.time() - t0w)


def main():
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            try: r = json.loads(l)
            except json.JSONDecodeError: continue
            if 'error' not in r: done.add((r['cell'], r['record']))
    with open(OUT, 'a') as fh:
        for rec in G.RECORDS:
            for arm in G.ARM_ORDER:
                if (arm, rec) in done: continue
                try:
                    out = run(rec, arm)
                except Exception as e:
                    import traceback
                    out = dict(cell=arm, record=rec, error=repr(e),
                               tb=traceback.format_exc()[-900:])
                fh.write(json.dumps(out) + '\n'); fh.flush()
                tag = (f"nowcast={out['nowcast']:.4f} lead={out['leadmean']:.4f} "
                       f"N={out['n_atoms']} vp={out['varpro_fired']}"
                       if 'error' not in out else str(out['error'])[:70])
                print(f'{arm:5s} {rec:7s} {tag}', flush=True)
    rows = [json.loads(l) for l in open(OUT)]
    print('\n=== 게이트 ===')
    for gid, ok, msg in G.gates(rows):
        print(f'  [{"PASS" if ok else "FAIL"}] {gid}: {msg[:100]}')


if __name__ == '__main__':
    main()
