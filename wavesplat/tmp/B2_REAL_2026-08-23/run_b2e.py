"""B-2e — **처방 시험: 적응 문턱의 척도 불변화** (결과 전 판정 기재).

## 왜 이 질문인가 — 저장소가 이미 처방을 써 뒀다
`B-2d` 가 *"자유도가 자료에 비해 많다"* 를 남겼고 처방은 **적응적 차수**다.
`adaptive_n` 은 **이미 배포돼 있다**(기본 True). 그런데 그 판정 문턱이 문제다 —
`sequential_update.py:151-160` 주석 **축자**:
> *"`_adaptive_prune_grow` 는 **절대 잔차 RMS(m)** 를 `grow_residual_threshold`(0.05 m)와
> 비교하며 **해상 상태로 정규화하지 않는다**. ... `SCALE_INVARIANT_GROW=True` 면 관측 RMS 로
> 나눠 문턱이 *"관측 진폭 중 아직 설명되지 않은 비율"* 을 뜻한다. **참값이 필요 없으므로 배포 가능하다.**"*

★실측에서 이것이 왜 치명적인가: 우리 12 레코드의 σ 가 **0.092 ~ 2.512 m 로 27배** 갈린다.
절대 문턱 0.05 m 는 `BS11b`(σ=0.092)에서 **σ의 54%**, `LJ18a`(σ=2.51)에서 **2%** 다.
⟹ ***같은 문턱이 레코드마다 전혀 다른 뜻을 갖는다.*** `B-2b` 가 찾은 잡음 민감도와 같은 계열이다.

## 판정 (결과 전 확정) — 기준선 `GROW-ABS` 는 7/12
- `GROW-SCALED` 가 **≥10/12** ⟹ ★`SCALE-INVARIANT-GROW-WORKS` (**배포 처방**)
- **8~9/12** ⟹ ⚠`PARTIAL`
- **≤7/12** ⟹ ⛔`NO-EFFECT`
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
    'GROW-ABS':    dict(cfg={}, grow=False),   # 배포: 절대 문턱 0.05 m (= B-2 의 PROP)
    'GROW-SCALED': dict(cfg={}, grow=True),    # ★척도 불변: 관측 RMS 로 정규화
}
OUT=os.path.join(HERE,'results_b2e.jsonl')


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
    old=SU.SCALE_INVARIANT_GROW; SU.SCALE_INVARIANT_GROW=bool(spec['grow'])
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
                    grow_readback=bool(SU.SCALE_INVARIANT_GROW),

                    n_atoms=getattr(getattr(upd,'model',None),'n_gaussians',0) or 0,
                    varpro_fired=bool((res or {}).get('varpro')), wall_s=time.time()-t0w)
    finally:
        SU.SCALE_INVARIANT_GROW=old


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
                      f"N={out['n_atoms']} grow_si={out['grow_readback']}"
                      if 'error' not in out else str(out['error'])[:60]), flush=True)


if __name__=='__main__':
    main()
