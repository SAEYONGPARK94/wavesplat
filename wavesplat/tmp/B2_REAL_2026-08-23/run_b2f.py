"""B-2f — **적응적 차수 그 자체** + **결합 처방** (결과 전 판정 기재).

## ★먼저 내 프레이밍을 정정한다 (결과 전)
나는 어제 *"다음은 적응적 차수를 잰다"* 고 적었고, `B-2d`(`max_n`)를 그 축으로 여겼다.
**코드를 열어보니 아니었다**: `max_n` 은 `learned_init.py:339` 에서 **초기 제안 풀링을 자르는 상한**이고
`adaptive_n` 의 prune/grow 와 **독립 경로**다(`sequential_update.py:1209,1246` 은 grow 의 여유분 계산에만 쓴다).
⟹ ***`adaptive_n` 자체는 실측에서 아직 한 번도 안 재봤다.*** 이번이 처음이다.

## 팔 (3 × 12 = 36런, 배포 잡음 0.02 고정)
- `ADAPT-ON`   : 배포 기본값(`adaptive_n=True`) = `B-2` 의 `PROP`. 기준선 **7/12**.
- `ADAPT-OFF`  : `adaptive_n=False`. **prune+grow 기구 전체를 끈다**(순수 절제).
- `RX-COMBINED`: ★처방 = `max_n=10`(B-2d 9/12) + `SCALE_INVARIANT_GROW=True`(B-2e 8/12).
                 ⚠**둘의 합이 각각보다 낫다는 보장은 없다** — 이것도 결과 전에 적는다.

## ★개입 계수기 (W3 처방: "런 전에 개입이 0회인지 확인하라")
`_adaptive_prune_grow` 를 감싸 **호출 수 · 성분 수 변화**를 기록한다.
`ADAPT-ON` 에서 `n_before == n_after` 가 12/12 면 그 팔의 절제는 **원리적으로 무의미**하고,
판정은 `NO-EFFECT` 가 아니라 ⛔**`ADAPT-NO-INTERVENTION`** 이다(= 자료가 아니라 설계가 답을 막은 것).

## 판정 (결과 전 확정)
**PRIMARY** = `RX-COMBINED` 가 제로필드(1.0)를 이기는 레코드 수, n=12:
- **≥10/12** (명목 p=0.0193) ⟹ ⚠`COMBINED-WORKS (EXPLORATORY)`
- **≥11/12** (p=0.0032) ⟹ ★`COMBINED-WORKS (CONFIRMATORY)`
- **8~9** ⟹ ⚠`PARTIAL` · **≤7** ⟹ ⛔`NO-EFFECT`

★**다중성을 결과 전에 적는다**: 오늘 같은 12 레코드를 보는 것이 **여섯 번째**다
(B-2·B-2b·B-2c·B-2d·B-2e·B-2f). 가족단위 α=0.05/6=0.0083 ⟹ **명목 통과(10/12)는 확증이 아니다.**
그래서 위에 두 문턱을 **따로** 적었다.

**SECONDARY** = `ADAPT-OFF` vs `ADAPT-ON`(짝):
- `ADAPT-OFF` 가 **더 많이** 이기면 ★`ADAPT-HURTS`(배포된 적응 기구가 실측에서 순손해)
- **같으면**(짝 비트동점 ≥10/12) ⛔`ADAPT-NO-INTERVENTION`
- `ADAPT-OFF` 가 **덜** 이기면 `ADAPT-HELPS`
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

NOISE = 0.02          # ★배포 합성값 고정 (B-2b 가 이것이 지배 인자임을 보였으나 처방은 이 값에서 재야 한다)
ARMS = {
    'ADAPT-ON':    dict(cfg={},              grow_si=False),
    'ADAPT-OFF':   dict(cfg={'adaptive_n': False}, grow_si=False),
    'RX-COMBINED': dict(cfg={'max_n': 10},   grow_si=True),
}
ARM_ORDER = ['ADAPT-ON', 'ADAPT-OFF', 'RX-COMBINED']
OUT=os.path.join(HERE,'results_b2f.jsonl')


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

    # ★개입 계수기 — 감싸기만 하고 거동은 건드리지 않는다
    trace=[]
    orig_apg=SU.SequentialUpdater._adaptive_prune_grow
    def wrapped(self, *a, **kw):
        nb=int(getattr(self.model,'n_gaussians',0) or 0)
        out=orig_apg(self, *a, **kw)
        trace.append((nb, int(getattr(self.model,'n_gaussians',0) or 0)))
        return out
    SU.SequentialUpdater._adaptive_prune_grow=wrapped
    old_si=SU.SCALE_INVARIANT_GROW; SU.SCALE_INVARIANT_GROW=bool(spec['grow_si'])
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
                    n_atoms=int(getattr(getattr(upd,'model',None),'n_gaussians',0) or 0),
                    apg_calls=len(trace), apg_trace=trace,
                    apg_changed=sum(1 for a_,b_ in trace if a_!=b_),
                    grow_si_readback=bool(SU.SCALE_INVARIANT_GROW),
                    adaptive_readback=bool(getattr(upd,'adaptive_n',None)),
                    max_n_readback=int(getattr(upd,'max_n',-1)),
                    field_hash=float(np.sum(np.abs(Z))),      # 짝 비트동점 검출용
                    varpro_fired=bool((res or {}).get('varpro')), wall_s=time.time()-t0w)
    finally:
        SU.SCALE_INVARIANT_GROW=old_si
        SU.SequentialUpdater._adaptive_prune_grow=orig_apg


def main():
    done=set()
    if os.path.exists(OUT):
        for l in open(OUT):
            try: r=json.loads(l)
            except json.JSONDecodeError: continue
            if 'error' not in r: done.add((r['cell'],r['record']))
    with open(OUT,'a') as fh:
        for rec in G.RECORDS:
            for arm in ARM_ORDER:
                if (arm,rec) in done: continue
                try: out=run(rec,arm)
                except Exception as e:
                    import traceback
                    out=dict(cell=arm, record=rec, error=repr(e), tb=traceback.format_exc()[-600:])
                fh.write(json.dumps(out)+'\n'); fh.flush()
                print(f"{arm:12s} {rec:7s} " + (f"nowcast={out['nowcast']:.4f} N={out['n_atoms']} "
                      f"apg={out['apg_calls']}c/{out['apg_changed']}ch adapt={out['adaptive_readback']} "
                      f"maxn={out['max_n_readback']} si={out['grow_si_readback']}"
                      if 'error' not in out else str(out['error'])[:70]), flush=True)


if __name__=='__main__':
    main()
