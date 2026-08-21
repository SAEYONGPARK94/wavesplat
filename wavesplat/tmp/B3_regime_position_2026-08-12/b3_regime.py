#!/usr/bin/env python3
"""B-3 — 실측 해면은 합성의 희소 쪽인가 혼잡 쪽인가.  **추정기 런 0회.**

사전 명세 = `PRESPEC.md` (실측 값을 보기 전에 동결). 요지:
  · 조밀 해면의 2D 파수 스펙트럼에서 **PR(참여율) = (ΣS)²/ΣS²** 을 재고,
  · **개구·격자를 tier 별로 맞춰서** 합성 96 에피소드(라벨 있음)와 실측 13 레코드를 같은 자로 잰다.
  · 눈금이 갈리는지(AUC)를 **먼저** 확인하고, 갈리는 tier 에서만 실측을 배치한다.

⛔`load_surface_nc` 는 `scripts/realdata_analysis.py` 에서 **import 한다**(복사 금지).
"""
import os
import sys
import json
import glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'scripts'))

from realdata_analysis import load_surface_nc          # noqa: E402  ⛔재구현 금지

N_FRAMES = 24                       # 레코드/에피소드당 표본 프레임 (PRESPEC §3)
TIERS = {                           # (Lx, Ly, dx) — PRESPEC §2 표
    'T1': (195.0, 195.0, 3.0),
    'T2': (54.0, 54.0, 3.0),
    'T3': (13.0, 13.0, 1.0),
}
TIER_RECORDS = {                    # 어느 실측 레코드가 어느 tier 에 들어가는가 (유효 범위 기준)
    'T1': ('LJ_2018/Surfaces_20180103_093938_Grid2_short.nc',
           'YS_2017/Surfaces_20170513_050000.nc'),
    'T2': ('AA_2014/Surfaces_20140327_091000.nc',
           'AA_2015/Surfaces_20150305_103500.nc',
           'AA_2015/Surfaces_20150515_090000.nc',
           'LJ_2018/Surfaces_20180103_093938_Grid1_short.nc'),
    'T3': ('BS_2011/Surfaces_20111001_161800_short.nc',
           'BS_2011/Surfaces_20111004_113800_short.nc',
           'BS_2011/Surfaces_20111004_130700_short.nc',
           'BS_2011/Surfaces_20111004_153000_short.nc',
           'BS_2013/Surfaces_20130922_130001_short.nc',
           'BS_2013/Surfaces_20130925_121501_short.nc',
           'BS_2013/Surfaces_20130930_102001_short.nc'),
}


# ------------------------------------------------------------------ 공통 자 ---
def resample_to(field, x, y, Lx, Ly, dx, cx=None, cy=None):
    """(cx,cy) 중심의 Lx×Ly 창을 dx 격자로 쌍선형 재표본.  창 밖이면 None."""
    nx = int(round(Lx / dx)) + 1
    ny = int(round(Ly / dx)) + 1
    cx = float(np.mean(x)) if cx is None else cx
    cy = float(np.mean(y)) if cy is None else cy
    xt = cx + (np.arange(nx) - (nx - 1) / 2.0) * dx
    yt = cy + (np.arange(ny) - (ny - 1) / 2.0) * dx
    if xt[0] < x[0] - 1e-9 or xt[-1] > x[-1] + 1e-9:
        return None, None, None
    if yt[0] < y[0] - 1e-9 or yt[-1] > y[-1] + 1e-9:
        return None, None, None
    # 축별 1D 보간 (격자가 규칙적이므로 분리 가능)
    ix = np.interp(xt, x, np.arange(len(x)))
    iy = np.interp(yt, y, np.arange(len(y)))
    i0, i1 = np.floor(ix).astype(int), np.minimum(np.floor(ix).astype(int) + 1, len(x) - 1)
    j0, j1 = np.floor(iy).astype(int), np.minimum(np.floor(iy).astype(int) + 1, len(y) - 1)
    wx, wy = ix - i0, iy - j0
    A = field[np.ix_(j0, i0)] * (1 - wy)[:, None] * (1 - wx)[None, :] \
        + field[np.ix_(j0, i1)] * (1 - wy)[:, None] * wx[None, :] \
        + field[np.ix_(j1, i0)] * wy[:, None] * (1 - wx)[None, :] \
        + field[np.ix_(j1, i1)] * wy[:, None] * wx[None, :]
    return A, xt, yt


_HANN = {}


def hann2(ny, nx):
    if (ny, nx) not in _HANN:
        _HANN[(ny, nx)] = np.outer(np.hanning(ny), np.hanning(nx))
    return _HANN[(ny, nx)]


def spectral_stats(A, dx):
    """PRESPEC §3.  반환 dict(PR, H, E5, kp, n_band).  NaN 있으면 None."""
    if A is None or not np.all(np.isfinite(A)):
        return None
    ny, nx = A.shape
    s = (A - A.mean()) * hann2(ny, nx)
    S = np.abs(np.fft.fftshift(np.fft.fft2(s))) ** 2
    kx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx)) * 2 * np.pi
    ky = np.fft.fftshift(np.fft.fftfreq(ny, d=dx)) * 2 * np.pi
    K = np.hypot(*np.meshgrid(kx, ky))
    pos = K > 1e-9
    if not pos.any():
        return None
    ip = np.argmax(np.where(pos, S, -np.inf))
    kp = float(K.ravel()[ip])
    band = pos & (K >= 0.4 * kp) & (K <= 2.5 * kp)
    m = int(band.sum())
    if m < 8:
        return None
    v = S[band].astype(float)
    tot = v.sum()
    if not np.isfinite(tot) or tot <= 0:
        return None
    p = v / tot
    PR = float(tot ** 2 / (v ** 2).sum())
    H = float(-(p * np.log(np.maximum(p, 1e-300))).sum() / np.log(m))
    E5 = float(np.sort(v)[::-1][:5].sum() / tot)
    return dict(PR=PR, H=H, E5=E5, kp=kp, n_band=m, ny=ny, nx=nx)


def agg(frames_stats):
    ok = [s for s in frames_stats if s]
    if not ok:
        return None
    out = {k: float(np.median([s[k] for s in ok])) for k in ('PR', 'H', 'E5', 'kp')}
    out['n_frames_ok'] = len(ok)
    out['n_band'] = int(np.median([s['n_band'] for s in ok]))
    out['grid'] = f"{ok[0]['ny']}x{ok[0]['nx']}"
    return out


# ------------------------------------------------------------------ 합성 ---
def synth_episode(path, tier):
    import h5py
    Lx, Ly, dx = TIERS[tier]
    with h5py.File(path, 'r') as f:
        x = f['frames/gt_grid_x'][:]
        y = f['frames/gt_grid_y'][:]
        Z = f['frames/gt_wave_field']
        T = Z.shape[0]
        idx = np.linspace(0, T - 1, N_FRAMES).astype(int)
        st = []
        for i in idx:
            A, _, _ = resample_to(np.asarray(Z[i], float), x, y, Lx, Ly, dx)
            st.append(spectral_stats(A, dx))
    return agg(st)


def regime(ep):
    b = os.path.basename(ep)
    if '_2comp_' in b or '_5comp_' in b:
        return 'sparse'
    if '_15comp_' in b or '_30comp_' in b:
        return 'crowded'
    raise ValueError(ep)


# ------------------------------------------------------------------ 실측 ---
def largest_valid_box(valid):
    """유효 마스크에서 가장 큰 '전부 유효' 직사각형의 (y0,y1,x0,x1).  단순 히스토그램법."""
    ny, nx = valid.shape
    h = np.zeros(nx, int)
    best = (0, 0, 0, 0, 0)
    for i in range(ny):
        h = np.where(valid[i], h + 1, 0)
        stack = []
        for j in range(nx + 1):
            cur = h[j] if j < nx else 0
            start = j
            while stack and stack[-1][1] >= cur:
                s, ht = stack.pop()
                area = ht * (j - s)
                if area > best[0]:
                    best = (area, i - ht + 1, i, s, j - 1)
                start = s
            stack.append((start, cur))
    _, y0, y1, x0, x1 = best
    return y0, y1, x0, x1


def real_record(rel, tier):
    """⚠전체 프레임을 올리면 안 된다 — 이 레코드들은 int16 패킹 1.2~11 GB 라
    float32 언팩이 그 2배를 요구한다(LJ Grid2 = 20.2 GiB → MemoryError, RAM 15 GB).
    필요한 프레임만 읽는다.  표본 자체는 PRESPEC §3 그대로(기록 전체 균등 24개)."""
    import h5py
    Lx, Ly, dx = TIERS[tier]
    path = os.path.join(REPO, 'data', 'real_bergamasco', rel)
    with h5py.File(path, 'r') as fh:
        T_all = int(fh['Z'].shape[0])
    idx = np.linspace(0, T_all - 1, N_FRAMES).astype(int)   # 통계용 (PRESPEC §3)
    mask_idx = np.arange(min(8, T_all))                     # 유효박스용 (원 규약 유지)
    want = np.unique(np.concatenate([mask_idx, idx]))
    Z, gx, gy, t, meta = load_surface_nc(path, frames=want)
    at = {int(v): k for k, v in enumerate(want)}            # 원 프레임번호 → 적재 위치

    # ⚠파일의 mask_Z 를 믿으면 안 된다: LJ Grid2 는 격자의 31.47% 가 상수 클램프인데
    # `_FillValue` 는 0.00% 이고 mask_Z 는 100% 유효라고 보고한다(4σ=26.4 m = 비물리).
    # 일반 판정: **표본 프레임 전체에서 값이 한 번도 안 변하는 셀은 바다가 아니다.**
    frozen = np.all(Z == Z[0], axis=0)
    valid = np.all(np.isfinite(Z[[at[int(i)] for i in mask_idx]]), axis=0) & ~frozen
    y0, y1, x0, x1 = largest_valid_box(valid)
    sx, sy = gx[x0:x1 + 1], gy[y0:y1 + 1]
    ext = (float(sx[-1] - sx[0]), float(sy[-1] - sy[0])) if sx.size > 1 and sy.size > 1 else (0.0, 0.0)
    st = []
    for i in idx:
        A, _, _ = resample_to(np.asarray(Z[at[int(i)], y0:y1 + 1, x0:x1 + 1], float),
                              sx, sy, Lx, Ly, dx)
        st.append(spectral_stats(A, dx))

    diag = dict(valid_box=[ext[0], ext[1]], need=[Lx, Ly], dx_native=meta['dx'],
                fps=meta['fps'], frames_total=T_all,
                fits=bool(ext[0] >= Lx - 1e-9 and ext[1] >= Ly - 1e-9))
    a = agg(st)
    if a is None:                       # ★왜 못 쟀는지를 남긴다 (None 만 남기면 진단 불가)
        return dict(error='no usable frame', **diag)
    a.update(**diag)
    a['Hs_proxy'] = float(4 * np.nanstd(Z[at[int(idx[len(idx) // 2])], y0:y1 + 1, x0:x1 + 1]))
    return a


def auc(a, b):
    """P(X_b > X_a) — 희소(a) 대 혼잡(b).  동점 0.5."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if not a.size or not b.size:
        return float('nan')
    g = (b[:, None] > a[None, :]).sum() + 0.5 * (b[:, None] == a[None, :]).sum()
    return float(g / (a.size * b.size))


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else 'both'
    out_path = os.path.join(HERE, 'b3_result.json')
    res = json.load(open(out_path)) if os.path.exists(out_path) else {}

    if which in ('synth', 'both'):
        eps = sorted(glob.glob(os.path.join(REPO, 'data_s9_sparse_2026-08-11', '*.h5'))
                     + glob.glob(os.path.join(REPO, 'data_s9_crowded_2026-08-11', '*.h5')))
        print(f'# 합성 {len(eps)} 에피소드')
        res.setdefault('synth', {})
        for tier in TIERS:
            if tier in res['synth']:
                print(f'  {tier}: 캐시 사용'); continue
            rows = []
            for k, p in enumerate(eps, 1):
                a = synth_episode(p, tier)
                if a:
                    a.update(episode=os.path.basename(p), regime=regime(p))
                    rows.append(a)
                if k % 24 == 0:
                    print(f'  {tier} {k}/{len(eps)}', flush=True)
            res['synth'][tier] = rows
            json.dump(res, open(out_path, 'w'), indent=1)
            sp = [r['PR'] for r in rows if r['regime'] == 'sparse']
            cr = [r['PR'] for r in rows if r['regime'] == 'crowded']
            print(f'  ★{tier} 격자 {rows[0]["grid"]} 대역셀 {rows[0]["n_band"]}  '
                  f'희소 PR 중앙 {np.median(sp):.2f} [{min(sp):.2f},{max(sp):.2f}] n={len(sp)}  '
                  f'혼잡 PR 중앙 {np.median(cr):.2f} [{min(cr):.2f},{max(cr):.2f}] n={len(cr)}  '
                  f'**AUC={auc(sp, cr):.3f}**', flush=True)

    if which in ('real', 'both'):
        res.setdefault('real', {})
        for tier, recs in TIER_RECORDS.items():
            res['real'].setdefault(tier, {})
            for rel in recs:
                if rel in res['real'][tier]:
                    continue
                try:
                    a = real_record(rel, tier)
                except Exception as e:
                    a = dict(error=f'{type(e).__name__}: {e}')
                res['real'][tier][rel] = a
                print(f'  {tier} {rel:48s} '
                      + (f'PR={a["PR"]:7.2f} H={a["H"]:.3f} E5={a["E5"]:.3f} '
                         f'격자 {a["grid"]} 유효박스 {a["valid_box"][0]:.0f}x{a["valid_box"][1]:.0f} m'
                         if a and 'PR' in a else str(a)), flush=True)
                json.dump(res, open(out_path, 'w'), indent=1)

    json.dump(res, open(out_path, 'w'), indent=1)
    print('\n→ b3_result.json')


if __name__ == '__main__':
    main()
