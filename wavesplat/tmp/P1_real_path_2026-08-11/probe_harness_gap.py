"""★하네스 21% 불일치 규명 — 같은 창(AA15a w5)에서 두 하네스의 **구성요소**를 하나씩 대조한다.

관측: lead 0 은 0.4982 vs 0.4885 (≈2% 차) 인데 lead 14 는 0.7049 vs 0.9043 (28% 차).
⟹ 채점/정규화 차이가 아니라 **시간 전진(propagation) 차이**다. 어디서 갈리는지 찾는다.
"""
import os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.path.insert(0, os.path.join(ROOT, 'tmp', 'bankmmse_port_2026-07-29'))
os.chdir(ROOT)

import d3_compare_gifs as cg
import run_rolling_refit as RR
from measure_blind_vs_frozen import dense_gt_std

REC, WIN = 'AA15a', 5

print('# scene_real 구성...')
sc = cg.scene_real(REC, win=WIN)
print('# build_stream 구성...')
st = RR.build_stream(REC, WIN)

print()
print('=== 1) 스칼라 대조 ===')
sg_gif = dense_gt_std(REC, WIN)
print(f'  sg   : scene/gif {sg_gif:.6f}   stream {st["sg"]:.6f}   '
      f'{"동일" if abs(sg_gif-st["sg"])<1e-12 else "★다름"}')
print(f'  t0   : scene {float(sc["Pq"][0,2]):.4f}   stream {st["t0"]:.4f}')
print(f'  leads: scene {len(sc["leads"])}  stream lead_max {st["lead_max"]}')
print(f'  grid : scene {np.asarray(sc["gxq"]).shape},{np.asarray(sc["gyq"]).shape}  '
      f'stream {np.asarray(st["gxq"]).shape},{np.asarray(st["gyq"]).shape}')
print(f'  obs  : scene P {np.asarray(sc["P"]).shape}   stream P_all {np.asarray(st["P_all"]).shape}')

print()
print('=== 2) ★진실면(GT) 대조 — lead 별 ===')
gx_same = np.allclose(np.asarray(sc['gxq']), np.asarray(st['gxq']))
gy_same = np.allclose(np.asarray(sc['gyq']), np.asarray(st['gyq']))
print(f'  격자 동일: gx {gx_same}  gy {gy_same}')
for li, L in enumerate(sc['leads']):
    j = int(round(float(L) / st['dt']))
    g_gif = np.asarray(sc['maps_gt'][li], float)
    g_str = np.asarray(st['Zq'][j], float)
    if g_gif.shape != g_str.shape:
        print(f'  lead {L:4.1f}: ★모양 다름 {g_gif.shape} vs {g_str.shape}')
        continue
    m = np.isfinite(g_gif) & np.isfinite(g_str)
    d = np.abs(g_gif[m] - g_str[m])
    nv_gif = int(np.sum(np.isfinite(g_gif))); nv_str = int(np.sum(np.isfinite(g_str)))
    if li in (0, 1, 7, 14):
        print(f'  lead {L:4.1f}: 최대|Δ| {d.max():.3e}  유효화소 gif {nv_gif} / stream {nv_str}')

print()
print('=== 3) ★관측 대조 — 같은 600점인가 ===')
P_gif = np.asarray(sc['P'], float)
m = (st['P_all'][:, 2] >= st['t0'] - RR.W_MAX) & (st['P_all'][:, 2] <= st['t0'] + 1e-9)
P_win, z_win = st['P_all'][m], st['z_all'][m]
print(f'  40 s 창 안 관측: stream {len(z_win)}점 -> 600 으로 서브샘플')
ii = np.random.default_rng(0).choice(len(z_win), 600, replace=False)
P_str = P_win[ii]
print(f'  gif  P: {P_gif.shape}  t범위 [{P_gif[:,2].min():.2f}, {P_gif[:,2].max():.2f}]')
print(f'  strm P: {P_str.shape}  t범위 [{P_str[:,2].min():.2f}, {P_str[:,2].max():.2f}]')
sg_ = lambda A: set(map(tuple, np.round(A, 6)))
inter = sg_(P_gif) & sg_(P_str)
print(f'  ★교집합 {len(inter)} / 600  ({len(inter)/600*100:.1f}%)')
print()
print('=== 4) 같은 관측을 주면 예측이 일치하는가 (lead 14) ===')
from wavesplat.optimizer.bank_mmse import fit_bank_mmse
gxq, gyq = np.asarray(st['gxq']), np.asarray(st['gyq'])
Xq, Yq = np.meshgrid(gxq, gyq)
tau = 14.0
Q = np.column_stack([Xq.ravel(), Yq.ravel(), np.full(Xq.size, st['t0'] + tau)])
g = np.asarray(st['Zq'][int(round(tau/st['dt']))], float); mm = np.isfinite(g)
for label, P_, z_ in (('stream 서브샘플(rng0)', P_str, z_win[ii]),
                      ('gif 의 600점', P_gif, np.asarray(sc['z'], float))):
    s0 = 0.2*float(np.std(z_))
    fits = [fit_bank_mmse(P_, z_, st['depth'], s0, kp_hint=None, seed=s) for s in RR.SEEDS]
    med = np.median(np.stack([f.render(Q) for f in fits], 0), 0).reshape(len(gyq), len(gxq))
    nr = float(np.sqrt(np.mean((med[mm]-g[mm])**2))/st['sg'])
    print(f'  {label:24s} lead14 nRMSE = {nr:.4f}')
