"""★인자 분리 — 배포 경로 vs 연구 경로가 갈리는 **정확한 지점**을 측정으로 특정한다.

배포 래퍼가 연구 호출에 대해 관측 배열에 가하는 변형은 정확히 셋뿐이다
(make_gif_bank_deployed.frames_from_scene + sequential_update._bank_obs_from_frames):
  O) 순서   : np.unique(P[:,2]) 로 프레임 묶음 -> 시간순 정렬 (연구는 rng(0) 셔플 순서)
  F) float32: FrameData 가 x/y/z 를 float32 로 양자화 -> 다시 float64
  T) 시간이동: t -> t - t_shift (t_shift = P[:,2].min())
각 인자를 하나씩 켜서 fit 의 중간 산출물(kp, pexp, smult, vr, gam)과
lead14 예측면을 대조한다.  ⛔추측 금지 — 전부 실행값.
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
from wavesplat.optimizer.bank_mmse import fit_bank_mmse

REC, WIN = 'AA15a', 5
SEEDS = RR.SEEDS

sc = cg.scene_real(REC, win=WIN)
st = RR.build_stream(REC, WIN)
gxq, gyq = np.asarray(st['gxq']), np.asarray(st['gyq'])
Xq, Yq = np.meshgrid(gxq, gyq)
depth = float(sc['depth'])

P0 = np.asarray(sc['P'], float)
z0 = np.asarray(sc['z'], float)
t_shift = float(P0[:, 2].min())


def apply_order(P, z):
    """frames_from_scene 의 재정렬 = np.unique(t) 프레임 묶음 (시간순 안정 정렬)."""
    idx = []
    for tt in np.unique(P[:, 2]):
        idx.append(np.flatnonzero(P[:, 2] == tt))
    idx = np.concatenate(idx)
    return P[idx], z[idx], idx


def apply_f32(P, z):
    Q = P.copy()
    Q[:, 0] = np.asarray(P[:, 0], np.float32).astype(float)
    Q[:, 1] = np.asarray(P[:, 1], np.float32).astype(float)
    return Q, np.asarray(z, np.float32).astype(float)


def apply_tshift(P, z):
    Q = P.copy(); Q[:, 2] = Q[:, 2] - t_shift
    return Q, z


def run(tag, P, z, t_off):
    s0 = 0.05 * 4.0 * float(np.std(z))
    fits = [fit_bank_mmse(P, z, depth, s0, kp_hint=None, seed=s) for s in SEEDS]
    tau = 14.0
    Q = np.column_stack([Xq.ravel(), Yq.ravel(),
                         np.full(Xq.size, st['t0'] + tau - t_off)])
    med = np.median(np.stack([f.render(Q) for f in fits], 0), 0) \
        .reshape(len(gyq), len(gxq))
    g = np.asarray(st['Zq'][int(round(tau / st['dt']))], float)
    mm = np.isfinite(g)
    nr = float(np.sqrt(np.mean((med[mm] - g[mm]) ** 2)) / st['sg'])
    print(f'{tag:34s} sigma0={s0:.6f} kp={[round(f.kp,4) for f in fits]} '
          f'pexp={[f.pexp for f in fits]} smult={[f.smult for f in fits]} '
          f'vr={[round(f.vr,4) for f in fits]} B={[f.n_candidates for f in fits]} '
          f'| lead14 nRMSE={nr:.4f}')
    return fits, med


print('=== 0) 기준: 연구 경로(sc P,z 그대로) ===')
run('RESEARCH (order=orig,f64,t=abs)', P0, z0, 0.0)

print()
print('=== 1) 인자 하나씩 ===')
Po, zo, idx = apply_order(P0, z0)
run('  +ORDER only', Po, zo, 0.0)
Pf, zf = apply_f32(P0, z0)
run('  +FLOAT32 only', Pf, zf, 0.0)
Pt, zt_ = apply_tshift(P0, z0)
run('  +TSHIFT only', Pt, zt_, t_shift)

print()
print('=== 2) 셋 다 (= 배포 경로 재구성) ===')
Pa, za, _ = apply_order(P0, z0)
Pa, za = apply_f32(Pa, za)
Pa, za = apply_tshift(Pa, za)
fits_all, med_all = run('  ORDER+F32+TSHIFT', Pa, za, t_shift)

print()
print('=== 3) 진짜 배포 경로와 비트 대조 ===')
import make_gif_bank_deployed as M
upd, t0_dep, el, cfg = M.estimate(sc, seed=42)
bank = upd._bank if getattr(upd, '_bank', None) is not None else upd._primary._bank
print(f'  배포 kp={[round(f.kp,4) for f in bank]} pexp={[f.pexp for f in bank]} '
      f'smult={[f.smult for f in bank]} vr={[round(f.vr,4) for f in bank]} '
      f'n_obs={[f.n_obs for f in bank]}')
Q = np.column_stack([Xq.ravel(), Yq.ravel(),
                     np.full(Xq.size, st['t0'] + 14.0 - t_shift)])
med_dep = np.median(np.stack([f.render(Q) for f in bank], 0), 0) \
    .reshape(len(gyq), len(gxq))
print(f'  재구성 vs 실제배포 최대|Δ| = {np.abs(med_all-med_dep).max():.3e}')

print()
print('=== 4) ★val 분할이 고르는 물리적 점이 다른가 ===')
n = len(z0)
vi = np.random.default_rng(42).choice(n, max(8, int(0.25 * n)), replace=False)
set_research = set(map(tuple, np.round(P0[vi], 6)))
set_deployed = set(map(tuple, np.round(Po[vi], 6)))
print(f'  seed42 val 150점: 연구 vs 배포(순서변경) 교집합 '
      f'{len(set_research & set_deployed)} / {len(vi)}')
print(f'  순서 permutation 이 항등인가: {np.array_equal(idx, np.arange(n))}')
