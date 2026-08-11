"""실측 5레코드 최종 비교 — 현재 연구 상태가 제대로 적용된 조건에서.

## 이 표가 고치는 것
1. ★**정규화 통일**: 실측 nRMSE 는 σ 로 나누는데 합성은 Hs 로 나눈다(Hs≈4σ).
   교차 도메인 비교가 가능하도록 **Hs 단위**를 병기한다.
2. ★**AA14 · LJ18g 는 배포 bank 경로를 한 번도 통과한 적이 없다** — 여기서 처음 통과시킨다.
   기존 "실측 0.49" 는 잰 3창 중 최고 창이었다.
3. ★**배포 경로 vs 연구 경로를 나란히** — 21% 불일치가 실재함을 표에 드러낸다.
"""
import os, sys, json, warnings
warnings.filterwarnings('ignore')
import numpy as np, torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.path.insert(0, os.path.join(ROOT, 'tmp', 'bankmmse_port_2026-07-29'))
os.chdir(ROOT)

import d3_compare_gifs as cg
import run_rolling_refit as RR
import make_gif_bank_deployed as M
from d3_falcon_real_operator import RECORDS
from wavesplat.optimizer.bank_mmse import fit_bank_mmse

WIN = int(os.environ.get('P1_WIN', '3'))
OUT = 'tmp/P1_real_path_2026-08-11/final_compare.json'
res = {}

for rec in ['AA15a', 'AA15b', 'AA14', 'YS17', 'LJ18g']:
    try:
        print(f'\n=== {rec} w{WIN} ===', flush=True)
        sc = cg.scene_real(rec, win=WIN)
        st = RR.build_stream(rec, WIN)
        gxq, gyq = np.asarray(st['gxq']), np.asarray(st['gyq'])
        Xq, Yq = np.meshgrid(gxq, gyq)
        taus = np.arange(0.0, st['lead_max'] + 1e-9, 1.0)

        # Hs/sigma: dense GT std vs Hs=4*std  ->  Hs 단위 = sigma 단위 / (Hs/sigma)
        Zall = np.concatenate([np.asarray(st['Zq'][int(round(t/st['dt']))]).ravel() for t in taus])
        Zall = Zall[np.isfinite(Zall)]
        Hs_over_sigma = 4.0 * np.std(Zall) / st['sg']

        # 연구 경로
        P_, z_ = np.asarray(sc['P'], float), np.asarray(sc['z'], float)
        fits = [fit_bank_mmse(P_, z_, st['depth'], 0.2*float(np.std(z_)), kp_hint=None, seed=s)
                for s in RR.SEEDS]
        # 배포 경로
        upd, t0_dep, el, cfg = M.estimate(sc, seed=42)

        cur_res, cur_dep = [], []
        for tau in taus:
            g = np.asarray(st['Zq'][int(round(tau/st['dt']))], float); mm = np.isfinite(g)
            Q = np.column_stack([Xq.ravel(), Yq.ravel(), np.full(Xq.size, st['t0']+tau)])
            med = np.median(np.stack([f.render(Q) for f in fits],0),0).reshape(len(gyq),len(gxq))
            cur_res.append(float(np.sqrt(np.mean((med[mm]-g[mm])**2))/st['sg']))
            Z = upd.predict_field(torch.tensor(gxq, dtype=torch.float32),
                                  torch.tensor(gyq, dtype=torch.float32),
                                  t0_dep+tau).cpu().numpy().reshape(len(gyq), len(gxq))
            cur_dep.append(float(np.sqrt(np.mean((Z[mm]-g[mm])**2))/st['sg']))

        res[rec] = dict(win=WIN, route=getattr(upd,'_auto_selected',None),
                        Hs_over_sigma=float(Hs_over_sigma),
                        deployed=cur_dep, research=cur_res,
                        dep_nowcast=cur_dep[0], dep_leadmean=float(np.mean(cur_dep)),
                        res_nowcast=cur_res[0], res_leadmean=float(np.mean(cur_res)))
        print(f'  route={res[rec]["route"]}  Hs/sigma={Hs_over_sigma:.3f}')
        print(f'  배포 nowcast {cur_dep[0]:.4f}  lead-mean {np.mean(cur_dep):.4f}')
        print(f'  연구 nowcast {cur_res[0]:.4f}  lead-mean {np.mean(cur_res):.4f}')
    except Exception as e:
        print(f'  ⛔ {rec}: {type(e).__name__}: {e}', flush=True)
        res[rec] = dict(error=f'{type(e).__name__}: {e}')

json.dump(res, open(OUT,'w'), indent=1)
print(f'\n저장: {OUT}')
