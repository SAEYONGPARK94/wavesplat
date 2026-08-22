"""B-2b — **λ 와 잡음/σ 를 가른다** (개입: 같은 레코드에서 관측 잡음만 바꾼다).

`APERTURE_AXIS.md`: `PROP` 오차가 **파장**(ρ=−0.739, p=0.006)과 **잡음/σ**(ρ=+0.678, p=0.015)
둘 다와 상관하는데 **두 특성이 서로 얽혀 있어**(짧은 파장 레코드가 곧 저에너지) n=12 로는 못 가른다.
⟹ ★**같은 레코드에서 잡음만 바꾸면 λ 는 고정되고 잡음/σ 만 움직인다.**

**판정 (결과 전 확정)**:
- 잡음을 낮췄을 때 **무기술 레코드가 기술을 얻으면** ⟹ ★**원인은 잡음/σ**
- 잡음을 낮춰도 **그대로면** ⟹ ★**원인은 파장(또는 그와 얽힌 다른 것)이고 잡음이 아니다**
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

NOISE_LEVELS = [0.02, 0.005, 0.0]     # 배포값 · 1/4 · 무잡음
OUT = os.path.join(HERE, 'results_b2b.jsonl')


def run(record, noise):
    from wavesplat.optimizer.sequential_update import make_updater
    from wavesplat.run import DEFAULT_CONFIG, set_seed
    t0w=time.time()
    sc = scene_station(record, win=G.WIN, noise=noise)
    t_shift=float(np.asarray(sc['P'])[:,2].min()); t0=float(sc['Pq'][0,2])-t_shift
    frames=M.frames_from_scene(sc, t_shift, dtype=torch.float32)
    cfg=dict(DEFAULT_CONFIG); cfg.update(depth=float(sc['depth']), seed=G.SEED, device='cpu')
    cfg.update(G.ARMS['PROP'])                     # ★제안 표현 강제
    set_seed(G.SEED); upd=make_updater(cfg)
    res=upd.initialize(frames, x_range=(float(sc['gxq'].min()), float(sc['gxq'].max())),
                       y_range=(float(sc['gyq'].min()), float(sc['gyq'].max())), verbose=False)
    gxq,gyq=np.asarray(sc['gxq']),np.asarray(sc['gyq'])
    g=np.asarray(sc['maps_gt'][0],float); mm=np.isfinite(g)
    Z=upd.predict_field(torch.tensor(gxq,dtype=torch.float32),
                        torch.tensor(gyq,dtype=torch.float32),
                        t0).cpu().numpy().reshape(len(gyq),len(gxq))
    now=float(np.sqrt(np.mean((Z[mm]-g[mm])**2))/sc['sg'])
    return dict(record=record, noise=noise, nowcast=now, sigma=float(sc['sg']),
                nsr=noise/float(sc['sg']), lam=float(2*np.pi/sc['kp']),
                n_atoms=getattr(getattr(upd,'model',None),'n_gaussians',0) or 0,
                varpro_fired=bool((res or {}).get('varpro')), wall_s=time.time()-t0w)


def main():
    done=set()
    if os.path.exists(OUT):
        for l in open(OUT):
            try: r=json.loads(l)
            except json.JSONDecodeError: continue
            if 'error' not in r: done.add((r['record'], r['noise']))
    with open(OUT,'a') as fh:
        for rec in G.RECORDS:
            for nz in NOISE_LEVELS:
                if (rec,nz) in done: continue
                try: out=run(rec,nz)
                except Exception as e:
                    import traceback
                    out=dict(record=rec, noise=nz, error=repr(e), tb=traceback.format_exc()[-600:])
                fh.write(json.dumps(out)+'\n'); fh.flush()
                print(f"{rec:7s} noise={nz:<6} " + (f"nowcast={out['nowcast']:.3f} "
                      f"nsr={out['nsr']*100:5.1f}%" if 'error' not in out else str(out['error'])[:60]), flush=True)


if __name__=='__main__':
    main()
