"""B-2c — **실측의 잡음 민감도가 λ 로 매개되는가** (결과 전 판정 기재).

## 왜 이 질문인가 — 오늘의 합성 발견과 실측 한계를 잇는 시험이다
- `M-1`(런 0회): `Σ|A| ~ √N` 인데 장 RMS 는 N 무관 ⟹ **고정 λ 의 상대 무게가 자료에 따라 변한다**
- `W7`·`W17`: 창-안 이탈이 **λ 에 용량-반응**하고 **스텝을 늘리면 악화**
- `B-2b`: 실측 성능이 **관측 잡음에 지배적으로 의존**하고 **잡음 0 에서도 나빠진다**(모형 불일치)
⟹ ★가설: ***"실측의 잡음 민감도는 λ 가 자료 규모에 맞춰지지 않아서다."***
(문헌의 고전적 처방 = sqrt-lasso/scaled lasso 계열. ⛔novelty 아님 — R3-2 가 3중 선점 확인.)

## 판정 (결과 전 확정)
배포 잡음(0.02)에서 `PROP` 기준선은 **7/12** 다.
- λ 를 자료 규모에 맞춘 팔이 **≥10/12** ⟹ ★`LAMBDA-MEDIATES` (처방 성립)
- **7~9/12** ⟹ ⚠`PARTIAL`
- **≤6/12** ⟹ ⛔`LAMBDA-DOES-NOT-MEDIATE` (잡음 민감도는 λ 로 설명되지 않는다)
"""
import json, os, sys, time, warnings
warnings.filterwarnings('ignore')
import numpy as np, torch
HERE=os.path.dirname(os.path.abspath(__file__))
ROOT=os.path.abspath(os.path.join(HERE,'..','..'))
for p in (HERE, ROOT, os.path.join(ROOT,'scripts'),
          os.path.join(ROOT,'tmp','bankmmse_port_2026-07-29')): sys.path.insert(0,p)
os.chdir(ROOT)
import gates_b2 as G                    # noqa: E402
from scene_station import scene_station  # noqa: E402
import make_gif_bank_deployed as M       # noqa: E402

NOISE = 0.02          # ★배포 합성값 고정 — 자료를 깨끗하게 만들지 않는다
ARMS = {
    'LAM-BASE':  dict(cfg={}, si=False),                       # 배포 λ=1e-4 (= B-2 의 PROP)
    'LAM-SCALE': dict(cfg={}, si=True),                        # ★척도 맞춤(관측 RMS 로 곱)
    'LAM-ZERO':  dict(cfg={'lambda_l1': 0.0, 'lambda_smooth': 0.0}, si=False),  # 벌점 제거
}
OUT=os.path.join(HERE,'results_b2c.jsonl')


def run(record, arm):
    from wavesplat.optimizer.sequential_update import make_updater
    import wavesplat.optimizer.sequential_update as SU
    from wavesplat.run import DEFAULT_CONFIG, set_seed
    spec=ARMS[arm]; t0w=time.time()
    sc=scene_station(record, win=G.WIN, noise=NOISE)
    t_shift=float(np.asarray(sc['P'])[:,2].min()); t0=float(sc['Pq'][0,2])-t_shift
    frames=M.frames_from_scene(sc, t_shift, dtype=torch.float32)
    cfg=dict(DEFAULT_CONFIG); cfg.update(depth=float(sc['depth']), seed=G.SEED, device='cpu')
    cfg.update(G.ARMS['PROP']); cfg.update(spec['cfg'])
    old=SU.SCALE_INVARIANT_L1; SU.SCALE_INVARIANT_L1=bool(spec['si'])
    try:
        set_seed(G.SEED); upd=make_updater(cfg)
        res=upd.initialize(frames, x_range=(float(sc['gxq'].min()), float(sc['gxq'].max())),
                           y_range=(float(sc['gyq'].min()), float(sc['gyq'].max())), verbose=False)
        gxq,gyq=np.asarray(sc['gxq']),np.asarray(sc['gyq'])
        g=np.asarray(sc['maps_gt'][0],float); mm=np.isfinite(g)
        Z=upd.predict_field(torch.tensor(gxq,dtype=torch.float32),
                            torch.tensor(gyq,dtype=torch.float32),
                            t0).cpu().numpy().reshape(len(gyq),len(gxq))
        now=float(np.sqrt(np.mean((Z[mm]-g[mm])**2))/sc['sg'])
        return dict(cell=arm, record=record, nowcast=now, sigma=float(sc['sg']),
                    si_readback=bool(SU.SCALE_INVARIANT_L1),
                    lam_readback=float(getattr(upd,'lambda_l1', float('nan'))),
                    n_atoms=getattr(getattr(upd,'model',None),'n_gaussians',0) or 0,
                    varpro_fired=bool((res or {}).get('varpro')), wall_s=time.time()-t0w)
    finally:
        SU.SCALE_INVARIANT_L1=old


def main():
    done=set()
    if os.path.exists(OUT):
        for l in open(OUT):
            try: r=json.loads(l)
            except json.JSONDecodeError: continue
            if 'error' not in r: done.add((r['cell'],r['record']))
    with open(OUT,'a') as fh:
        for rec in G.RECORDS:
            for arm in ARMS:
                if (arm,rec) in done: continue
                try: out=run(rec,arm)
                except Exception as e:
                    import traceback
                    out=dict(cell=arm, record=rec, error=repr(e), tb=traceback.format_exc()[-600:])
                fh.write(json.dumps(out)+'\n'); fh.flush()
                print(f"{arm:10s} {rec:7s} " + (f"nowcast={out['nowcast']:.3f} "
                      f"λ={out['lam_readback']:.1e} si={out['si_readback']}"
                      if 'error' not in out else str(out['error'])[:60]), flush=True)


if __name__=='__main__':
    main()
