"""배포 경로 vs 연구 경로 — 같은 관측·같은 격자·같은 GT 에서 예측면을 직접 비교.

§3 에서 관측이 100% 동일함을 확인했고, 직접 fit_bank_mmse+render 는 rolling_refit 값(0.7049)을
정확히 재현했다. 그런데 배포 경로(SequentialUpdater 실측게이트 -> _initialize_bank ->
predict_field)는 0.9043 을 낸다. **차이는 배포 래퍼 안에 있다.**
"""
import os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np, torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.path.insert(0, os.path.join(ROOT, 'tmp', 'bankmmse_port_2026-07-29'))
os.chdir(ROOT)

import d3_compare_gifs as cg
import run_rolling_refit as RR
import make_gif_bank_deployed as M

REC, WIN = 'AA15a', 5
sc = cg.scene_real(REC, win=WIN)
st = RR.build_stream(REC, WIN)
gxq, gyq = np.asarray(st['gxq']), np.asarray(st['gyq'])
Xq, Yq = np.meshgrid(gxq, gyq)

print('=== 배포 경로 ===')
upd, t0_dep, el, cfg = M.estimate(sc, seed=42)
print(f'  route={getattr(upd,"_auto_selected",None)}  t0_deployed={t0_dep:.4f}')
t_shift = float(np.asarray(sc['P'])[:, 2].min())
print(f'  t_shift={t_shift:.4f}   (원래 t0={float(sc["Pq"][0,2]):.4f})')

print()
print('=== 연구 경로 (직접 fit) ===')
from wavesplat.optimizer.bank_mmse import fit_bank_mmse
P_, z_ = np.asarray(sc['P'], float), np.asarray(sc['z'], float)
s0 = 0.2*float(np.std(z_))
fits = [fit_bank_mmse(P_, z_, st['depth'], s0, kp_hint=None, seed=s) for s in RR.SEEDS]
print(f'  members={len(fits)}  kp={[round(f.kp,4) for f in fits]}')

print()
print('=== lead 별 대조 ===')
print(f'  {"lead":>5} {"배포":>9} {"연구":>9} {"차":>8}')
for tau in (0.0, 1.0, 7.0, 14.0):
    g = np.asarray(st['Zq'][int(round(tau/st['dt']))], float); mm = np.isfinite(g)
    # 연구
    Q = np.column_stack([Xq.ravel(), Yq.ravel(), np.full(Xq.size, st['t0']+tau)])
    med = np.median(np.stack([f.render(Q) for f in fits],0),0).reshape(len(gyq),len(gxq))
    nr_res = float(np.sqrt(np.mean((med[mm]-g[mm])**2))/st['sg'])
    # 배포 (shifted time)
    Z = upd.predict_field(torch.tensor(gxq, dtype=torch.float32),
                          torch.tensor(gyq, dtype=torch.float32), t0_dep+tau).cpu().numpy()
    Z = np.asarray(Z).reshape(len(gyq), len(gxq))
    nr_dep = float(np.sqrt(np.mean((Z[mm]-g[mm])**2))/st['sg'])
    print(f'  {tau:5.1f} {nr_dep:9.4f} {nr_res:9.4f} {nr_dep-nr_res:+8.4f}')

print()
print('=== 5) ★원점 오류인가 속도 오류인가 ===')
print('  배포 예측을 시간 t0_dep+tau+delta 로 이동시키며 연구 곡선과 맞춰본다.')
print('  delta 가 상수면 = 시간 원점 어긋남 / delta 가 tau 에 비례하면 = 전진 속도 어긋남')
for tau in (7.0, 14.0):
    g = np.asarray(st['Zq'][int(round(tau/st['dt']))], float); mm = np.isfinite(g)
    Q = np.column_stack([Xq.ravel(), Yq.ravel(), np.full(Xq.size, st['t0']+tau)])
    med = np.median(np.stack([f.render(Q) for f in fits],0),0).reshape(len(gyq),len(gxq))
    nr_res = float(np.sqrt(np.mean((med[mm]-g[mm])**2))/st['sg'])
    best = None
    for delta in np.arange(-4.0, 4.01, 0.25):
        Z = upd.predict_field(torch.tensor(gxq, dtype=torch.float32),
                              torch.tensor(gyq, dtype=torch.float32),
                              t0_dep+tau+delta).cpu().numpy().reshape(len(gyq), len(gxq))
        nr = float(np.sqrt(np.mean((Z[mm]-g[mm])**2))/st['sg'])
        if best is None or nr < best[1]:
            best = (delta, nr)
    print(f'  tau={tau:5.1f}: 연구 {nr_res:.4f} | 배포 최선 {best[1]:.4f} @ delta={best[0]:+.2f}s')
