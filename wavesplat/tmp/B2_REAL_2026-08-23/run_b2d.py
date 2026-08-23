"""B-2d — **실측 한계가 모델 차수(원자 수)로 매개되는가** (결과 전 판정 기재).

## 왜 이 질문인가 — 남은 후보를 하나씩 지운다
- `B-2b`: 실측 성능이 **관측 잡음에 지배**되고 **잡음 0 에서도 나빠진다**(잡음 모형 불일치)
- `B-2c`: ⛔**λ 는 매개하지 않는다**(벌점을 없애도 동일) ⟹ 벌점은 후보에서 제거
- ★코드 확인: `PROP` 경로에는 **잡음 모형이 없다** — 신뢰도가 전부 1이고 벌점은 무효다.
  ⟹ **가중 없는 최소제곱**이므로 자유도가 남으면 **잡음을 그대로 적합**한다.
- 실측 소형 박스는 질의점이 **16–25개**인데 원자를 **16–30개** 세운다.
⟹ ★가설: ***"실측 한계는 모델 차수가 자료에 비해 과하기 때문이다."***

## 판정 (결과 전 확정) — 기준선 `MAXN-30` 은 7/12
- 차수를 낮춘 팔이 **≥10/12** ⟹ ★`ORDER-MEDIATES` (처방 성립)
- **8~9/12** ⟹ ⚠`PARTIAL`
- **≤7/12** ⟹ ⛔`ORDER-DOES-NOT-MEDIATE`
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

NOISE = 0.02          # ★배포 합성값 고정
ARMS = {
    'MAXN-30': dict(cfg={}, si=False),                  # 배포 기본 max_n=30 (= B-2 의 PROP)
    'MAXN-10': dict(cfg={'max_n': 10}, si=False),
    'MAXN-4':  dict(cfg={'max_n': 4}, si=False),
}
OUT=os.path.join(HERE,'results_b2d.jsonl')


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
                    max_n_intent=spec['cfg'].get('max_n', 30),
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
                      f"N={out['n_atoms']} (cap {out['max_n_intent']})"
                      if 'error' not in out else str(out['error'])[:60]), flush=True)


if __name__=='__main__':
    main()
