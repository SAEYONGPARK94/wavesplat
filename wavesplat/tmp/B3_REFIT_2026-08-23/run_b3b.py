"""B-3b 러너 — 용량-반응 (판정은 `gates_b3b.py` 에 결과 전 동결). 프로토콜은 B-3 과 **동일**.

프로토콜은 `gates_b3` 독스트링에 있다. 여기서는 **순서만** 다시 못 박는다:
lead ℓ 마다 **① 예측·채점 → ② 그 시각 관측 투입**. 절대 뒤집지 마라.
"""
import json, os, sys, time, warnings
warnings.filterwarnings('ignore')
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
for p in (HERE, ROOT, os.path.join(ROOT, 'scripts'),
          os.path.join(ROOT, 'tmp', 'B2_REAL_2026-08-23'),
          os.path.join(ROOT, 'tmp', 'bankmmse_port_2026-07-29')):
    sys.path.insert(0, p)
os.chdir(ROOT)
import gates_b3b as G                      # noqa: E402
from scene_station import scene_station     # noqa: E402
import make_gif_bank_deployed as M          # noqa: E402

OUT = os.path.join(HERE, 'results_b3b.jsonl')


def run(record, win, arm):
    from wavesplat.data_loader import FrameData
    from wavesplat.optimizer.sequential_update import make_updater
    import wavesplat.optimizer.sequential_update as SU
    from wavesplat.run import DEFAULT_CONFIG, set_seed
    t0w = time.time()
    sc = scene_station(record, win=win, noise=G.NOISE, with_future_obs=True)
    t_shift = float(np.asarray(sc['P'])[:, 2].min())
    t0 = float(sc['Pq'][0, 2]) - t_shift
    frames = M.frames_from_scene(sc, t_shift, dtype=torch.float32)
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(depth=float(sc['depth']), seed=G.SEED, device='cpu')
    cfg.update(G.ARMS[arm])

    # ★가지 단위 개입 계수기 (B-2f 의 교훈: 함수 단위로는 못 잡는다)
    calls = []
    orig = SU.SequentialUpdater._bank_fit_window
    def wrapped(self, *a, **kw):
        calls.append(1); return orig(self, *a, **kw)
    SU.SequentialUpdater._bank_fit_window = wrapped
    try:
        set_seed(G.SEED)
        upd = make_updater(cfg)
        upd.initialize(frames, x_range=(float(sc['gxq'].min()), float(sc['gxq'].max())),
                       y_range=(float(sc['gyq'].min()), float(sc['gyq'].max())), verbose=False)
        n_init = len(calls)                     # 초기화 중 호출은 개입이 아니다
        gxq, gyq = np.asarray(sc['gxq']), np.asarray(sc['gyq'])
        tx = torch.tensor(gxq, dtype=torch.float32); ty = torch.tensor(gyq, dtype=torch.float32)
        Pf = np.asarray(sc['P_fut'], float); zf = np.asarray(sc['z_fut'], float)
        per_lead = {}
        for li, l_ in enumerate(sc['leads']):
            # ① 먼저 예측·채점
            g = np.asarray(sc['maps_gt'][li], float); mm = np.isfinite(g)
            if mm.sum() > 0:
                Z = upd.predict_field(tx, ty, t0 + float(l_)).cpu().numpy().reshape(len(gyq), len(gxq))
                per_lead[float(l_)] = float(np.sqrt(np.mean((Z[mm] - g[mm]) ** 2)) / sc['sg'])
            # ② 그 다음 그 시각 관측 투입
            if li == 0:
                continue
            tt = t0 + float(l_) + t_shift
            m = np.abs(Pf[:, 2] - tt) < 1e-6
            if m.sum() == 0:
                continue
            nq = int(m.sum())
            # ⚠`frames_from_scene` 과 **같은 형식**으로 만든다(같은 dtype·같은 부가 필드).
            upd.update(FrameData(t=float(l_) + t0,
                                 x_obs=torch.tensor(Pf[m, 0], dtype=torch.float32),
                                 y_obs=torch.tensor(Pf[m, 1], dtype=torch.float32),
                                 z_obs=torch.tensor(zf[m], dtype=torch.float32),
                                 confidence=torch.ones(nq, dtype=torch.float32),
                                 incidence_angles=torch.zeros(nq, dtype=torch.float32)))
        fc = [v for k, v in per_lead.items() if k >= 2.0]
        return dict(cell=arm, record=record, win=win,
                    lead0=per_lead.get(0.0), fc_mean=float(np.mean(fc)) if fc else None,
                    per_lead={str(k): v for k, v in sorted(per_lead.items())},
                    refit_calls=len(calls) - n_init, init_fit_calls=n_init,
                    refit_readback=getattr(upd, '_bank_refit_every_s', 'MISSING'),
                    n_atoms=int(getattr(getattr(upd, 'model', None), 'n_gaussians', 0) or 0),
                    n_fut_obs=int(len(zf)), sigma=float(sc['sg']), wall_s=time.time() - t0w)
    finally:
        SU.SequentialUpdater._bank_fit_window = orig


def main():
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            try: r = json.loads(l)
            except json.JSONDecodeError: continue
            if 'error' not in r: done.add((r['cell'], r['record'], r['win']))
    with open(OUT, 'a') as fh:
        for rec in G.RECORDS:
            for win in G.WINDOWS:
                for arm in G.ARM_ORDER:
                    if (arm, rec, win) in done: continue
                    try: out = run(rec, win, arm)
                    except Exception as e:
                        import traceback
                        out = dict(cell=arm, record=rec, win=win, error=repr(e),
                                   tb=traceback.format_exc()[-600:])
                    fh.write(json.dumps(out) + '\n'); fh.flush()
                    print(f"{arm:7s} {rec:7s} w{win} " + (
                        f"lead0={out['lead0']:.4f} fc={out['fc_mean']:.4f} "
                        f"refit={out['refit_calls']} N={out['n_atoms']} {out['wall_s']:.1f}s"
                        if 'error' not in out else str(out['error'])[:70]), flush=True)


if __name__ == '__main__':
    main()
