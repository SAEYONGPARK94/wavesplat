"""HOSGATE 코퍼스 생성 — 짝(paired) 32쌍 = 64 에피소드.  PREREG.md §2 그대로.

    python3 tmp/HOSGATE_2026-08-13/gen_hos.py smoke     # 혼잡 1쌍 + 희소 1쌍
    python3 tmp/HOSGATE_2026-08-13/gen_hos.py all       # 32쌍 (재개 가능)
    python3 tmp/HOSGATE_2026-08-13/gen_hos.py crowded   # 16쌍
    python3 tmp/HOSGATE_2026-08-13/gen_hos.py sparse    # 16쌍

짝의 정의: 같은 (sea_state, n_components, seed) 에서 **같은 (스냅된) 파랑 성분**으로
  LIN 팔 = 선형 evaluate_field,  HOS 팔 = HOS M=3 비선형 전개
를 만든다. LiDAR·잡음·레이드롭·메시·격자·duration·dt·depth 는 전부 동일하다.
차이는 **파면을 시간전개시키는 방식 하나뿐**이다.

⚠`wavesim/**` 은 **수정하지 않는다**. 필요한 세 가지 우회는 전부 이 파일 안의 몽키패치다
(PREREG §2 에 선언, 게이트 G-2 가 런 전에 증명):
 1. `pipeline.jonswap_components` -> 원본 + **HOS 격자 스냅** (양 팔 공통 성분)
 2. `pipeline.HOSSimulator.initialize_jonswap` -> `initialize_from_components(스냅 성분)`
    (원본은 `initialize_jonswap(Hs,Tp,seed)` 을 불러 **다른 바다**를 만든다 = 짝 파괴)
 3. `HOSSimulator.evaluate_at` -> **분리가능 재작성** (같은 합의 인수분해, 근사 아님).
    원본은 비선형 전개 후 16384 모드 x 38200 점 = 프레임당 35 s = **2.9 h/에피소드**.
"""
import json
import os
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # .../wavesplat/wavesplat
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import h5py                                              # noqa: E402
import wavesim.pipeline as PL                            # noqa: E402
from wavesim.config import WaveSimConfig                 # noqa: E402
from wavesim.hos_simulator import HOSSimulator           # noqa: E402

# ── 동결 상수 (PREREG §2) ───────────────────────────────────────────────────
G = 9.81
HOS_M, HOS_NX, HOS_NY, HOS_LX, HOS_LY, HOS_DT = 3, 192, 128, 600.0, 400.0, 0.02
MESH_NX, MESH_NY = 200, 167
DURATION, DT, DEPTH, N_PTS = 30.0, 0.1, 50.0, 50

CROWDED_SEEDS = (1581, 1582, 1583, 1584)
SPARSE_SEEDS = (1585, 1586, 1587, 1588)
JOBS = {
    'crowded': [(bf, nc, s) for bf in ('BF4', 'BF5') for nc in (15, 30)
                for s in CROWDED_SEEDS],
    'sparse': [(bf, nc, s) for bf in ('BF4', 'BF5') for nc in (2, 5)
               for s in SPARSE_SEEDS],
}
SMOKE_JOBS = [('BF4', 15, 1581), ('BF4', 2, 1585)]

OUT = {'LIN': os.path.join(HERE, 'data_lin'), 'HOS': os.path.join(HERE, 'data_hos')}
SUFFIX = 'uniform_ra_rf_ds'


def fname(bf, nc, seed):
    return f'episode_{bf}_{nc}comp_seed{seed}_{SUFFIX}.h5'


# ── 몽키패치 1: 성분을 HOS 스펙트럼 격자에 스냅 ─────────────────────────────
_DKX, _DKY = 2 * np.pi / HOS_LX, 2 * np.pi / HOS_LY
_KX1 = np.fft.fftfreq(HOS_NX, d=1.0 / (HOS_NX * _DKX))
_KY1 = np.fft.fftfreq(HOS_NY, d=1.0 / (HOS_NY * _DKY))
_ORIG_JC = PL.jonswap_components
_LAST = {}


def snap_components(components, depth):
    """kx,ky 를 최근접 스펙트럼 빈으로 옮기고 omega 를 분산관계로 재도출."""
    out, moved = [], []
    for c in components:
        ix = int(round(c.kx / _DKX)) % HOS_NX
        iy = int(round(c.ky / _DKY)) % HOS_NY
        kx, ky = float(_KX1[ix]), float(_KY1[iy])
        k = float(np.hypot(kx, ky))
        w = float(np.sqrt(G * k * np.tanh(np.clip(k * depth, 0, 20)))) if k > 0 else 0.0
        moved.append(float(np.hypot(kx - c.kx, ky - c.ky)))
        out.append(type(c)(A=c.A, omega=w, kx=kx, ky=ky, phi=c.phi, Q=c.Q))
    _LAST['max_snap_dk'] = max(moved) if moved else 0.0
    return out


def _patched_jonswap_components(*a, **kw):
    comps = _ORIG_JC(*a, **kw)
    sc = snap_components(comps, kw.get('depth', DEPTH))
    _LAST['components'] = sc
    return sc


# ── 몽키패치 2+3: 짝 초기화 + 분리가능 evaluate_at ──────────────────────────
class PairedHOS(HOSSimulator):
    """짝 설계용 HOS.  물리는 원본 그대로, 초기화와 평가 경로만 바꾼다."""

    def initialize_jonswap(self, Hs, Tp, **kw):          # noqa: D102  (서명 호환)
        comps = _LAST['components']
        self.initialize_from_components(comps)
        # PREREG §2: Hs 재정규화를 하지 않는다 -- 두 팔이 같은 진폭을 써야 짝이 성립한다.

    def evaluate_at(self, X, Y):
        """`HOSSimulator.evaluate_at` 와 **같은 합**을 인수분해로 계산.

        eta(x_i,y_j) = Re[ sum_pq a_pq e^{i kx_p x_j'} e^{i ky_q y_i'} ]
        make_grid 규약: 배열 shape (ny,nx), X[i,j]=x[j], Y[i,j]=y[i].
        """
        assert X.ndim == 2 and X.shape == Y.shape
        x, y = X[0, :], Y[:, 0]
        assert np.allclose(X, x[None, :]) and np.allclose(Y, y[:, None]), \
            'separable evaluate_at requires a regular meshgrid'
        Ex = np.exp(1j * np.outer(x, self.KX[:, 0]))     # (nx, Nx)
        Ey = np.exp(1j * np.outer(y, self.KY[0, :]))     # (ny, Ny)
        a = self.eta_hat
        Z = np.real(Ey @ a.T @ Ex.T)
        Zx = np.real(Ey @ (1j * self.KX * a).T @ Ex.T)
        Zy = np.real(Ey @ (1j * self.KY * a).T @ Ex.T)
        mag = np.sqrt(Zx ** 2 + Zy ** 2 + 1.0)
        return Z, -Zx / mag, -Zy / mag, 1.0 / mag


PL.jonswap_components = _patched_jonswap_components
PL.HOSSimulator = PairedHOS


# ── 진단 (PREREG §2 "사전 공개 결함" 를 에피소드마다 실측) ──────────────────
def truncation_fracs(comps):
    kxn, kyn = np.pi * HOS_NX / HOS_LX, np.pi * HOS_NY / HOS_LY
    A = np.array([c.A for c in comps], float)
    kx = np.abs([c.kx for c in comps])
    ky = np.abs([c.ky for c in comps])
    tot = (A ** 2).sum()
    f = lambda m: float((A[m] ** 2).sum() / tot) if tot > 0 else 0.0   # noqa: E731
    return dict(trunc_2k=f((2 * kx > kxn) | (2 * ky > kyn)),
                trunc_3k=f((3 * kx > kxn) | (3 * ky > kyn)),
                k_max=float(np.hypot(kx, ky).max()),
                lam_min=float(2 * np.pi / np.hypot(kx, ky).max()))


def make_cfg(hos_enabled):
    cfg = WaveSimConfig()
    cfg.mesh.nx, cfg.mesh.ny = MESH_NX, MESH_NY
    cfg.hos.enabled = hos_enabled
    cfg.hos.M, cfg.hos.Nx, cfg.hos.Ny = HOS_M, HOS_NX, HOS_NY
    cfg.hos.Lx, cfg.hos.Ly, cfg.hos.dt = HOS_LX, HOS_LY, HOS_DT
    return cfg


def log(msg, fh):
    line = f'[{time.strftime("%H:%M:%S")}] {msg}'
    print(line, flush=True)
    fh.write(line + '\n')
    fh.flush()


def gen_one(arm, bf, nc, seed, fh):
    out_dir = OUT[arm]
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, fname(bf, nc, seed))
    if os.path.exists(path):
        log(f'  skip (있음) {arm} {os.path.basename(path)}', fh)
        return None
    t0 = time.time()
    PL.run_episode(
        sea_state=bf, n_components=nc, n_pts=N_PTS, seed=seed,
        raydrop_model='uniform', duration=DURATION, dt=DT, depth=DEPTH,
        output_dir=out_dir, verbose=False, cfg=make_cfg(arm == 'HOS'),
        use_gpu_raycast=True,
        random_amplitudes=True, random_frequencies=True, decorrelate_stream=True,
    )
    el = time.time() - t0
    with h5py.File(path, 'r') as f:
        gt = f['frames/gt_wave_field']
        nfr = gt.shape[0]
        hs_first = float(4 * np.std(gt[0]))
        hs_last = float(4 * np.std(gt[-1]))
        eta_max = float(np.abs(gt[:]).max())
        attrs = dict(f['config'].attrs)
    rec = dict(status='ok', arm=arm, elapsed_s=el, n_frames=nfr,
               gt_Hs_first=hs_first, gt_Hs_last=hs_last, gt_eta_absmax=eta_max,
               mesh_verdict=str(attrs.get('mesh_verdict', '?')),
               var_under_resolved=float(attrs.get('var_under_resolved', np.nan)),
               max_snap_dk=_LAST.get('max_snap_dk'),
               **truncation_fracs(_LAST['components']))
    log(f'  ok {arm} {os.path.basename(path)} {el:.0f}s frames={nfr} '
        f'Hs {hs_first:.3f}->{hs_last:.3f} mesh={rec["mesh_verdict"]} '
        f'2k={100*rec["trunc_2k"]:.1f}% 3k={100*rec["trunc_3k"]:.1f}%', fh)
    return rec


def main():
    which = (sys.argv[1:] or ['all'])[0]
    if which == 'smoke':
        jobs, tag = SMOKE_JOBS, 'smoke'
    elif which == 'all':
        jobs, tag = JOBS['crowded'] + JOBS['sparse'], 'all'
    else:
        jobs, tag = JOBS[which], which

    mpath = os.path.join(HERE, 'gen_manifest.json')
    manifest = json.load(open(mpath)) if os.path.exists(mpath) else {}
    fh = open(os.path.join(HERE, 'gen.log'), 'a')
    log(f'=== gen {tag}: {len(jobs)} 쌍  HOS M={HOS_M} {HOS_NX}x{HOS_NY} '
        f'L=({HOS_LX:.0f},{HOS_LY:.0f}) dt={HOS_DT}', fh)

    t_start = time.time()
    for i, (bf, nc, seed) in enumerate(jobs, 1):
        key = f'{bf}_{nc}comp_seed{seed}'
        log(f'[{i}/{len(jobs)}] {key}', fh)
        for arm in ('LIN', 'HOS'):        # LIN 먼저: 같은 성분이 _LAST 에 실린다
            try:
                rec = gen_one(arm, bf, nc, seed, fh)
                if rec is not None:
                    manifest[f'{arm}/{fname(bf, nc, seed)}'] = rec
            except Exception as e:
                manifest[f'{arm}/{fname(bf, nc, seed)}'] = dict(
                    status='error', arm=arm, error=f'{type(e).__name__}: {e}')
                log(f'  ERROR {arm} {key}: {e}', fh)
                traceback.print_exc()
            json.dump(manifest, open(mpath, 'w'), indent=1)
        el = time.time() - t_start
        log(f'    누적 {el/60:.1f} min, ETA {el/i*(len(jobs)-i)/60:.0f} min', fh)

    ok = sum(1 for v in manifest.values() if v.get('status') == 'ok')
    log(f'gen {tag} 완료: ok {ok} / {len(manifest)}  ({(time.time()-t_start)/60:.1f} min)', fh)
    fh.close()


if __name__ == '__main__':
    main()
