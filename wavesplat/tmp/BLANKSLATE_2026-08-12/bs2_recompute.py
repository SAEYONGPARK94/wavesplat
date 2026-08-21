#!/usr/bin/env python3
"""
BS-2 백지 재도출: 실측 (Bergamasco stereo, 13 records)
=====================================================
논문/RESULT.md/메모리의 결론을 일절 인용하지 않고, .nc 원자료에서 직접 재계산한다.

PART A  record_inventory()  : 13 레코드 전수 — 격자/개구/프레임/fps/마스크/클램프 비율
PART B  clamp_contamination(): 클램프 셀이 과거 실측 수치에 얼마나 들어갔는지 정량화
PART C  normalisation()      : Hs vs sigma — 교차도메인 비교 단위 맞추기
PART D  rerun_pilot()        : PILOT_realdata 파일럿(iterative_relax) 재현 + naive 대조

사용:
    python3 tmp/BLANKSLATE_2026-08-12/bs2_recompute.py A
    python3 tmp/BLANKSLATE_2026-08-12/bs2_recompute.py B
    ...
모든 산출 -> tmp/BLANKSLATE_2026-08-12/bs2_*.json
"""
import os, sys, json, time, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import h5py

DATA = os.path.join(ROOT, 'data', 'real_bergamasco')
OUT = HERE
G = 9.81


def records():
    """13 records, 6 sites."""
    return sorted(glob.glob(os.path.join(DATA, '*', '*.nc')))


def rid(p):
    return os.path.basename(os.path.dirname(p)) + '/' + os.path.basename(p).replace('Surfaces_', '').replace('.nc', '')


# ------------------------------------------------------------------ PART A
def scan_record(path, n_frames=32, seed=0):
    """레코드 1개 전수 스캔. int16 RAW 로 비교 (언팩 반올림 회피)."""
    f = h5py.File(path, 'r')
    zd = f['Z']
    T_all, n0, n1 = zd.shape
    a = zd.attrs
    fill = int(a['_FillValue'][0]) if '_FillValue' in a else None
    sf = float(a['scale_factor'][0]) if 'scale_factor' in a else 1.0
    off = float(a['add_offset'][0]) if 'add_offset' in a else 0.0
    # 기록 전체에 균등 배치한 프레임
    sel = np.unique(np.linspace(0, T_all - 1, n_frames).astype(int))
    raw = np.stack([zd[i] for i in sel])                      # (K, n0, n1) int16
    K = raw.shape[0]

    # 좌표
    def _unpack2(name):
        if name not in f:
            return None
        d = f[name]
        aa = d.attrs
        s = float(aa['scale_factor'][0]) if 'scale_factor' in aa else 1.0
        o = float(aa['add_offset'][0]) if 'add_offset' in aa else 0.0
        return d[:].astype(np.float64) * s + o
    Xg2 = _unpack2('_nc4_non_coord_X')
    Yg2 = _unpack2('Y')
    # 어느 좌표 grid 가 어느 축을 따라 변하는지 *측정* 한다 (로더 주석을 믿지 않는다)
    ax0 = ax1 = None
    for g in (Xg2, Yg2):
        if g is None or g.ndim != 2:
            continue
        v0 = float(np.nanstd(g[:, 0])); v1 = float(np.nanstd(g[0, :]))
        if v0 > v1 and ax0 is None:
            ax0 = g[:, 0]
        elif v1 >= v0 and ax1 is None:
            ax1 = g[0, :]
    if ax0 is None:
        ax0 = np.arange(n0, dtype=float)
    if ax1 is None:
        ax1 = np.arange(n1, dtype=float)

    mask = f['mask_Z'][:] if 'mask_Z' in f else None
    t = f['time'][:].astype(float) if 'time' in f else np.arange(T_all)
    dt = float(np.median(np.diff(t))) if T_all > 1 else np.nan
    def _js(v):
        if isinstance(v, bytes):
            return v.decode('utf8', 'replace')
        if hasattr(v, 'tolist'):
            v = v.tolist()
        if isinstance(v, list):
            return [_js(x) for x in v]
        return v if isinstance(v, (int, float, str, type(None), bool)) else str(v)
    attrs = {str(k): _js(v) for k, v in f.attrs.items()}
    # ★ 연속 창 클램프 — 추정기가 실제로 쓰는 창(연속 프레임)에서의 클램프 비율.
    #   기록 전체 균등표본은 "가끔 움직이는" 셀을 놓치지 않지만, 실제 파이프라인이 본 것은 연속 창이다.
    win_res = {}
    for W in (32, 128):
        j0 = max(0, T_all // 2 - W // 2)
        rw = zd[j0:j0 + W]
        cw = (rw == rw[0]).all(axis=0)
        fw = (rw == fill).all(axis=0) if fill is not None else np.zeros(cw.shape, bool)
        win_res[f'clamp_frac_contig_w{W}'] = float((cw & ~fw).mean())
    f.close()

    # ---------- 결함 진단 ----------
    is_fill = (raw == fill) if fill is not None else np.zeros_like(raw, bool)
    fill_frac_all = float(is_fill.mean())                     # 전체 항목 중 _FillValue
    cell_allfill = is_fill.all(axis=0)                        # 모든 표본프레임서 fill
    # 마스크 주장
    mask_valid = (mask != 0) if mask is not None else np.ones((n0, n1), bool)
    mask_valid_frac = float(mask_valid.mean())

    # ★ 클램프 판정 = 표본 프레임 내내 RAW 값이 한 번도 변하지 않는 셀
    #   (fill 셀은 별도 계산; 여기선 fill 아닌 상수 셀)
    const = (raw == raw[0]).all(axis=0)
    clamp = const & (~cell_allfill)
    clamp_frac_grid = float(clamp.mean())                     # 전체 격자 대비
    # mask_Z 가 "유효"라 주장한 셀 중 클램프
    denom = max(int(mask_valid.sum()), 1)
    clamp_frac_of_maskvalid = float((clamp & mask_valid).sum() / denom)
    # mask_Z 가 유효라 했는데 실제로 fill 인 셀 (마스크 거짓말 #2)
    maskvalid_but_fill = float((cell_allfill & mask_valid).sum() / denom)

    # 클램프 셀이 취하는 값 (상수 클램프인지 확인)
    cl_vals = raw[0][clamp].astype(np.float64) * sf + off if clamp.any() else np.array([])
    cl_uniq = int(np.unique(raw[0][clamp]).size) if clamp.any() else 0

    # ---------- 진짜 유효 셀 ----------
    good = mask_valid & (~clamp) & (~cell_allfill)
    good_frac = float(good.mean())

    # 유효 셀의 바운딩 박스 = "진짜 개구"
    def bbox(m):
        if not m.any():
            return None
        r = np.flatnonzero(m.any(axis=1)); c = np.flatnonzero(m.any(axis=0))
        return int(r[0]), int(r[-1]), int(c[0]), int(c[-1])
    bb_mask = bbox(mask_valid)
    bb_good = bbox(good)

    d_ax0 = float(np.median(np.abs(np.diff(ax0)))) if len(ax0) > 1 else np.nan
    d_ax1 = float(np.median(np.abs(np.diff(ax1)))) if len(ax1) > 1 else np.nan

    def extent(bb):
        if bb is None:
            return [0.0, 0.0]
        r0, r1, c0, c1 = bb
        return [float((r1 - r0) * d_ax0), float((c1 - c0) * d_ax1)]
    ext_mask = extent(bb_mask)
    ext_good = extent(bb_good)

    # 클램프 셀만의 "빈틈없는 사각형" 여부 -> 가장자리 밴드인지 내부 구멍인지
    if clamp.any():
        rr = np.flatnonzero(clamp.any(axis=1)); cc = np.flatnonzero(clamp.any(axis=0))
        clamp_touches_border = bool(rr[0] == 0 or rr[-1] == n0 - 1 or cc[0] == 0 or cc[-1] == n1 - 1)
        # 내부(가장자리 아닌) 클램프 비율
        interior = np.zeros_like(clamp); interior[1:-1, 1:-1] = True
        clamp_interior_frac = float((clamp & interior).sum() / max(int(clamp.sum()), 1))
    else:
        clamp_touches_border, clamp_interior_frac = False, 0.0

    # ---------- 물리량 (진짜 유효 셀에서만) ----------
    Zf = raw.astype(np.float64) * sf + off
    Zf[:, ~good] = np.nan
    # 공간 표준편차 (프레임별) 중앙값
    sig_sp = float(np.nanmedian([np.nanstd(Zf[i]) for i in range(K)]))
    # 시간 표준편차 (셀별) 중앙값  -- 표본 프레임이 균등배치라 aliasing 주의
    sig_t = float(np.nanmedian(np.nanstd(Zf, axis=0)[good])) if good.any() else np.nan

    res = dict(
        record=rid(path), path=path, site=os.path.basename(os.path.dirname(path)),
        size_GB=round(os.path.getsize(path) / 1e9, 2),
        T_all=int(T_all), n0=int(n0), n1=int(n1),
        dt=dt, fps=float(1 / dt) if dt and np.isfinite(dt) else None,
        duration_s=float(t[-1] - t[0]) if T_all > 1 else 0.0,
        d_ax0=d_ax0, d_ax1=d_ax1,
        grid_extent=[float((n0 - 1) * d_ax0), float((n1 - 1) * d_ax1)],
        nc_attrs=attrs,
        frames_sampled=int(K),
        fill_frac_all_entries=fill_frac_all,
        cell_allfill_frac=float(cell_allfill.mean()),
        mask_Z_valid_frac=mask_valid_frac,
        clamp_frac_of_grid=clamp_frac_grid,
        clamp_frac_of_maskvalid=clamp_frac_of_maskvalid,
        maskvalid_but_allfill_frac=maskvalid_but_fill,
        clamp_n_unique_values=cl_uniq,
        clamp_value_min=float(cl_vals.min()) if cl_vals.size else None,
        clamp_value_max=float(cl_vals.max()) if cl_vals.size else None,
        clamp_touches_border=clamp_touches_border,
        clamp_interior_frac=clamp_interior_frac,
        good_frac=good_frac,
        bbox_maskvalid=bb_mask, bbox_good=bb_good,
        extent_maskvalid_m=ext_mask, extent_good_m=ext_good,
        sigma_spatial_m=sig_sp, sigma_temporal_m=sig_t,
        **win_res,
    )
    return res, Zf, good, ax0, ax1, d_ax0, d_ax1, dt


def spec_stats(fld, dax0, dax1):
    """단일 프레임 2D FFT: peak wavelength, directional spread, kp."""
    s = np.nan_to_num(fld - np.nanmean(fld))
    ny, nx = s.shape
    w = np.outer(np.hanning(ny), np.hanning(nx))
    F = np.fft.fftshift(np.fft.fft2(s * w))
    P = np.abs(F) ** 2
    k0 = np.fft.fftshift(np.fft.fftfreq(ny, d=dax0)) * 2 * np.pi
    k1 = np.fft.fftshift(np.fft.fftfreq(nx, d=dax1)) * 2 * np.pi
    K1, K0 = np.meshgrid(k1, k0)
    K = np.hypot(K0, K1)
    m = K > 1e-6
    ip = int(np.argmax(P * m))
    kp = float(K.ravel()[ip])
    lam = 2 * np.pi / kp if kp > 0 else np.nan
    band = m & (K > 0.4 * kp) & (K < 2.5 * kp)
    th = np.arctan2(K0[band], K1[band]); ww = P[band]
    C = np.sum(ww * np.cos(2 * th)) / np.sum(ww); S = np.sum(ww * np.sin(2 * th)) / np.sum(ww)
    R = np.hypot(C, S)
    spread = float(np.degrees(0.5 * np.sqrt(max(-2 * np.log(max(R, 1e-9)), 0.0))))
    # participation ratio in band (B-3 의 PR 을 독립 재구현)
    Sb = P[band]
    pr = float(Sb.sum() ** 2 / np.sum(Sb ** 2)) if Sb.size else np.nan
    return dict(kp=kp, lam_p=lam, spread_deg=spread, PR=pr, n_band_cells=int(band.sum()))


def part_A(n_frames=32):
    out = []
    for p in records():
        t0 = time.time()
        try:
            r, Zf, good, ax0, ax1, d0, d1, dt = scan_record(p, n_frames=n_frames)
        except Exception as e:
            out.append(dict(record=rid(p), error=f'{type(e).__name__}: {e}'))
            print(f'  !! {rid(p)}: {e}')
            continue
        # 스펙트럼: good 박스 안에서, 유효율이 가장 높은 프레임
        bb = r['bbox_good']
        if bb:
            r0, r1, c0, c1 = bb
            sub = Zf[:, r0:r1 + 1, c0:c1 + 1]
            nanf = np.mean(np.isnan(sub.reshape(sub.shape[0], -1)), axis=1)
            i = int(np.argmin(nanf))
            fr = np.nan_to_num(sub[i] - np.nanmean(sub[i]))
            ss = spec_stats(fr, d0, d1)
            r.update(ss)
            r['char_frame_nanfrac'] = float(nanf[i])
            r['aperture_lambda_ax0'] = float(r['extent_good_m'][0] / ss['lam_p']) if ss['lam_p'] > 0 else None
            r['aperture_lambda_ax1'] = float(r['extent_good_m'][1] / ss['lam_p']) if ss['lam_p'] > 0 else None
            # Hs 후보들
            v = sub[i][~np.isnan(sub[i])]
            r['Hs_4sigma_spatial'] = float(4 * v.std())
            # 시간 유의파고 (표본프레임은 균등배치라 진짜 시계열 아님 -> 별도 PART C 서)
        r['scan_s'] = round(time.time() - t0, 1)
        print(f"  {r['record']:52s} clamp/grid={r['clamp_frac_of_grid']*100:5.1f}%  "
              f"clamp/maskvalid={r['clamp_frac_of_maskvalid']*100:5.1f}%  "
              f"mask_valid={r['mask_Z_valid_frac']*100:5.1f}%  fill={r['fill_frac_all_entries']*100:4.1f}%  "
              f"good={r['good_frac']*100:5.1f}%  ({r['scan_s']}s)")
        out.append(r)
    json.dump(out, open(os.path.join(OUT, 'bs2_partA_inventory.json'), 'w'), indent=1)
    print(f"\n-> bs2_partA_inventory.json  ({len(out)} records)")
    return out


# ------------------------------------------------------------------ PART C
def part_C(n_frames=256):
    """Hs / sigma 비율을 레코드별로 직접 측정. 여러 정의를 병기한다."""
    rows = []
    for p in records():
        f = h5py.File(p, 'r')
        zd = f['Z']; a = zd.attrs
        T_all, n0, n1 = zd.shape
        fill = int(a['_FillValue'][0]); sf = float(a['scale_factor'][0]); off = float(a['add_offset'][0])
        mask = f['mask_Z'][:] != 0 if 'mask_Z' in f else np.ones((n0, n1), bool)
        t = f['time'][:].astype(float)
        # 연속 구간 (진짜 시계열) — 중앙에서 n_frames 연속
        i0 = max(0, T_all // 2 - n_frames // 2)
        raw = zd[i0:i0 + n_frames]
        f.close()
        dt = float(np.median(np.diff(t)))
        const = (raw == raw[0]).all(axis=0)
        allfill = (raw == fill).all(axis=0)
        good = mask & ~const & ~allfill
        Z = raw.astype(np.float64) * sf + off
        Z[(raw == fill)] = np.nan
        Z[:, ~good] = np.nan
        if not good.any():
            rows.append(dict(record=rid(p), error='no good cells')); continue
        # sigma_temporal: 셀별 시간 std -> 공간 중앙값 (버릇: 실측 논문의 sigma)
        st = np.nanstd(Z, axis=0)[good]
        sigma_t = float(np.nanmedian(st))
        # sigma_spatial: 프레임별 공간 std -> 시간 중앙값
        ss = np.array([np.nanstd(Z[i]) for i in range(Z.shape[0])])
        sigma_s = float(np.nanmedian(ss))
        # Hm0 = 4*sqrt(m0) from temporal spectrum at good cells (= 4*sigma_t if all energy in band)
        Hm0_t = 4 * sigma_t
        Hm0_s = 4 * sigma_s
        # H_1/3 zero-up-crossing on a representative central good cell timeseries
        idx = np.flatnonzero(good.ravel())
        cen = idx[len(idx) // 2]
        ts = Z.reshape(Z.shape[0], -1)[:, cen]
        ts = ts - np.nanmean(ts)
        H13 = zero_crossing_H13(ts)
        rows.append(dict(record=rid(p), n_frames=int(Z.shape[0]), dt=dt,
                         sigma_temporal=sigma_t, sigma_spatial=sigma_s,
                         Hm0_temporal=Hm0_t, Hm0_spatial=Hm0_s,
                         H13_zerocross=H13,
                         ratio_Hm0s_over_sigmat=float(Hm0_s / sigma_t) if sigma_t > 0 else None,
                         ratio_H13_over_sigmat=float(H13 / sigma_t) if (H13 and sigma_t > 0) else None,
                         ratio_sigmas_over_sigmat=float(sigma_s / sigma_t) if sigma_t > 0 else None))
        print(f"  {rid(p):52s} sig_t={sigma_t:.4f} sig_s={sigma_s:.4f} "
              f"Hm0_s/sig_t={rows[-1]['ratio_Hm0s_over_sigmat']:.3f} "
              f"H13/sig_t={rows[-1]['ratio_H13_over_sigmat'] if rows[-1]['ratio_H13_over_sigmat'] else float('nan'):.3f}")
    json.dump(rows, open(os.path.join(OUT, 'bs2_partC_normalisation.json'), 'w'), indent=1)
    print('-> bs2_partC_normalisation.json')
    return rows


def zero_crossing_H13(ts):
    ts = np.asarray(ts, float)
    ts = ts[np.isfinite(ts)]
    if ts.size < 20:
        return None
    zc = np.flatnonzero((ts[:-1] < 0) & (ts[1:] >= 0))
    if zc.size < 3:
        return None
    H = []
    for a, b in zip(zc[:-1], zc[1:]):
        seg = ts[a:b + 1]
        H.append(seg.max() - seg.min())
    H = np.sort(np.array(H))[::-1]
    n3 = max(1, len(H) // 3)
    return float(H[:n3].mean())


# ------------------------------------------------------------------ PART B
def part_B(n_frames=32):
    """클램프 오염 정량화:
       (i) 과거 로더(load_surface_nc)가 어떤 셀을 '유효'로 넘겼는가
       (ii) 그 유효 집합에서 클램프 셀이 차지하는 비율
       (iii) 클램프를 포함/제외했을 때 통계량이 얼마나 달라지는가 (sigma, Hs, PR, lam_p)
    """
    rows = []
    for p in records():
        try:
            r, Zf_good, good, ax0, ax1, d0, d1, dt = scan_record(p, n_frames=n_frames)
        except Exception as e:
            rows.append(dict(record=rid(p), error=str(e))); continue
        # 과거 로더 재현: mask_Z==0 -> NaN, _FillValue -> NaN.  클램프는 그대로 통과.
        f = h5py.File(p, 'r'); zd = f['Z']; a = zd.attrs
        T_all = zd.shape[0]
        fill = int(a['_FillValue'][0]); sf = float(a['scale_factor'][0]); off = float(a['add_offset'][0])
        sel = np.unique(np.linspace(0, T_all - 1, n_frames).astype(int))
        raw = np.stack([zd[i] for i in sel])
        mask = f['mask_Z'][:] != 0 if 'mask_Z' in f else np.ones(raw.shape[1:], bool)
        f.close()
        Zold = raw.astype(np.float64) * sf + off
        Zold[raw == fill] = np.nan
        Zold[:, ~mask] = np.nan
        old_valid = np.isfinite(Zold).all(axis=0)             # 과거 파이프라인이 쓴 셀
        const = (raw == raw[0]).all(axis=0)
        allfill = (raw == fill).all(axis=0)
        clamp = const & ~allfill
        n_old = int(old_valid.sum())
        contam = float((clamp & old_valid).sum() / max(n_old, 1))

        # 통계량 비교: 대표 프레임에서
        i = raw.shape[0] // 2
        A = Zold[i].copy()                                     # 과거(클램프 포함)
        B = Zold[i].copy(); B[clamp] = np.nan                  # 정정(클램프 제외)
        def st(fld):
            v = fld[np.isfinite(fld)]
            return dict(n=int(v.size), sigma=float(v.std()) if v.size else None,
                        Hs4=float(4 * v.std()) if v.size else None,
                        mean=float(v.mean()) if v.size else None)
        sA, sB = st(A), st(B)
        # 스펙트럼: 과거는 클램프 셀을 0(평균제거 후) 로 밀어넣는다 -> 인공 저주파/DC 누출
        def sp(fld, fillval):
            g = np.where(np.isfinite(fld), fld, fillval)
            g = g - np.nanmean(g[np.isfinite(fld)])
            g = np.where(np.isfinite(fld), g, 0.0)
            return spec_stats(g, d0, d1)
        spA = sp(A, np.nan)
        spB = sp(B, np.nan)
        rows.append(dict(record=rid(p),
                         n_cells_old_pipeline=n_old,
                         frac_grid_old_pipeline=float(n_old / old_valid.size),
                         clamp_contamination_of_old_valid=contam,
                         sigma_with_clamp=sA['sigma'], sigma_without_clamp=sB['sigma'],
                         sigma_rel_change=float((sB['sigma'] - sA['sigma']) / sA['sigma'])
                         if (sA['sigma'] and sB['sigma']) else None,
                         n_with_clamp=sA['n'], n_without_clamp=sB['n'],
                         lam_p_with_clamp=spA['lam_p'], lam_p_without_clamp=spB['lam_p'],
                         spread_with_clamp=spA['spread_deg'], spread_without_clamp=spB['spread_deg'],
                         PR_with_clamp=spA['PR'], PR_without_clamp=spB['PR']))
        rr = rows[-1]
        print(f"  {rr['record']:52s} old_valid={rr['frac_grid_old_pipeline']*100:5.1f}% of grid, "
              f"clamp contamination={contam*100:5.1f}%  "
              f"sigma {rr['sigma_with_clamp']:.4f}->{rr['sigma_without_clamp']:.4f} "
              f"({(rr['sigma_rel_change'] or 0)*100:+.1f}%)  "
              f"PR {rr['PR_with_clamp']:.1f}->{rr['PR_without_clamp']:.1f}")
    json.dump(rows, open(os.path.join(OUT, 'bs2_partB_clamp_contamination.json'), 'w'), indent=1)
    print('-> bs2_partB_clamp_contamination.json')
    return rows


# ------------------------------------------------------------------ PART D
SENTINEL = -13.939052110290529   # LJ_2018 두 grid 에서 동일한 *물리* 값 (scale_factor 는 서로 다름)


def part_D(n_frames=24, tol=1e-6):
    """센티넬(±13.939 m) 전수 탐지 — '상수 셀' 규칙보다 엄격한 판정.
       셀 단위(한 번이라도 닿음) / 항목 단위 / 완전상수 세 가지를 모두 보고한다."""
    rows = []
    for p in records():
        f = h5py.File(p, 'r'); zd = f['Z']; a = zd.attrs
        sf = float(a['scale_factor'][0]); off = float(a['add_offset'][0]); fill = int(a['_FillValue'][0])
        T = zd.shape[0]
        sel = np.unique(np.linspace(0, T - 1, n_frames).astype(int))
        raw = np.stack([zd[i] for i in sel]); f.close()
        Z = raw.astype(np.float64) * sf + off
        isfill = raw == fill
        hit = (np.abs(np.abs(Z) - abs(SENTINEL)) < 1e-3) & ~isfill
        ent = float(hit.sum() / max(int((~isfill).sum()), 1))
        cell = float(hit.any(axis=0).mean())
        allhit = float(hit.all(axis=0).mean())
        # 센티넬 제거 전/후 sigma  (프레임별 공간 std 의 중앙값)
        Za = np.where(isfill, np.nan, Z)
        Zb = np.where(isfill | hit, np.nan, Z)
        sa = float(np.nanmedian([np.nanstd(Za[i]) for i in range(Za.shape[0])]))
        sb = float(np.nanmedian([np.nanstd(Zb[i]) for i in range(Zb.shape[0])]))
        rows.append(dict(record=rid(p), sentinel_entry_frac=ent, sentinel_cell_frac=cell,
                         sentinel_allframes_cell_frac=allhit,
                         sigma_with_sentinel=sa, sigma_without_sentinel=sb,
                         Hs4_with=4 * sa, Hs4_without=4 * sb,
                         denominator_inflation=float(sa / sb) if sb > 0 else None))
        r = rows[-1]
        print(f"  {r['record']:52s} entries={ent*100:6.2f}%  cells={cell*100:6.2f}%  allframes={allhit*100:6.2f}%  "
              f"sigma {sa:.4f}->{sb:.4f}  Hs denom inflation x{r['denominator_inflation']:.3f}")
    json.dump(rows, open(os.path.join(OUT, 'bs2_partD_sentinel.json'), 'w'), indent=1)
    print('-> bs2_partD_sentinel.json')
    return rows


# ------------------------------------------------------------------ PART E
def part_E(n_frames=24):
    """개구 검열 시험: 스펙트럼 피크가 *첫 분해 가능한 파수 링* 위에 앉아 있는가?
       앉아 있으면 lam_p 는 '측정' 이 아니라 '개구에서 잘린 하한' 이다."""
    rows = []
    for p in records():
        f = h5py.File(p, 'r'); zd = f['Z']; a = zd.attrs
        sf = float(a['scale_factor'][0]); off = float(a['add_offset'][0]); fill = int(a['_FillValue'][0])
        T, n0, n1 = zd.shape
        mask = (f['mask_Z'][:] != 0) if 'mask_Z' in f else np.ones((n0, n1), bool)
        def _u2(name):
            if name not in f: return None
            d = f[name]; aa = d.attrs
            return d[:].astype(np.float64) * float(aa['scale_factor'][0]) + float(aa['add_offset'][0])
        Xg2, Yg2 = _u2('_nc4_non_coord_X'), _u2('Y')
        sel = np.unique(np.linspace(0, T - 1, n_frames).astype(int))
        raw = np.stack([zd[i] for i in sel]); f.close()
        Z = raw.astype(np.float64) * sf + off
        bad = (raw == fill) | (np.abs(np.abs(Z) - abs(SENTINEL)) < 1e-3)
        good = mask & (~bad.any(axis=0)) & (~(raw == raw[0]).all(axis=0))
        # 최대 clean 직사각형 (행/열 탐욕 절삭)
        m = good.copy()
        while m.any() and not m[np.ix_(m.any(1), m.any(0))].all():
            rs, cs = m.any(1), m.any(0)
            sub = m[np.ix_(rs, cs)]
            ri, ci = np.flatnonzero(rs), np.flatnonzero(cs)
            if sub.mean(1).min() <= sub.mean(0).min():
                m[ri[int(np.argmin(sub.mean(1)))], :] = False
            else:
                m[:, ci[int(np.argmin(sub.mean(0)))]] = False
        if not m.any():
            rows.append(dict(record=rid(p), error='no clean rectangle')); continue
        rr, cc = np.flatnonzero(m.any(1)), np.flatnonzero(m.any(0))
        r0, r1, c0, c1 = rr[0], rr[-1], cc[0], cc[-1]
        d0 = float(np.median(np.abs(np.diff(Xg2[:, 0])))) if Xg2 is not None and np.nanstd(Xg2[:, 0]) > np.nanstd(Xg2[0, :]) else float(np.median(np.abs(np.diff(Yg2[:, 0]))))
        d1 = float(np.median(np.abs(np.diff(Yg2[0, :])))) if Yg2 is not None and np.nanstd(Yg2[0, :]) >= np.nanstd(Yg2[:, 0]) else float(np.median(np.abs(np.diff(Xg2[0, :]))))
        sub = Z[:, r0:r1 + 1, c0:c1 + 1]
        ny, nx = sub.shape[1], sub.shape[2]
        L0, L1 = (ny - 1) * d0, (nx - 1) * d1
        # 프레임별 피크 반경 링 인덱스의 중앙값
        rings = []
        for i in range(sub.shape[0]):
            s = sub[i] - sub[i].mean()
            w = np.outer(np.hanning(ny), np.hanning(nx))
            P = np.abs(np.fft.fftshift(np.fft.fft2(s * w))) ** 2
            k0 = np.fft.fftshift(np.fft.fftfreq(ny, d=d0)) * 2 * np.pi
            k1 = np.fft.fftshift(np.fft.fftfreq(nx, d=d1)) * 2 * np.pi
            K1, K0 = np.meshgrid(k1, k0)
            K = np.hypot(K0, K1)
            P[K < 1e-9] = 0
            ip = int(np.argmax(P))
            i0, i1 = np.unravel_index(ip, P.shape)
            n_0 = abs(i0 - ny // 2); n_1 = abs(i1 - nx // 2)
            rings.append((np.hypot(n_0, n_1), K.ravel()[ip]))
        ringn = float(np.median([r[0] for r in rings]))
        kpm = float(np.median([r[1] for r in rings]))
        lam = 2 * np.pi / kpm if kpm > 0 else np.nan
        rows.append(dict(record=rid(p), clean_rect=[int(ny), int(nx)],
                         clean_extent_m=[L0, L1], dx=[d0, d1],
                         peak_ring_index_median=ringn, kp=kpm, lam_p=lam,
                         aperture_over_lambda=[L0 / lam, L1 / lam],
                         censored=bool(ringn <= 1.5)))
        r = rows[-1]
        print(f"  {r['record']:52s} clean {ny:4d}x{nx:4d} = {L0:6.1f}x{L1:6.1f} m  "
              f"peak_ring={ringn:5.2f}  lam_p={lam:7.1f} m  ap/lam={L0/lam:5.2f}x{L1/lam:5.2f}  "
              f"{'CENSORED (peak at first ring)' if r['censored'] else ''}")
    json.dump(rows, open(os.path.join(OUT, 'bs2_partE_aperture.json'), 'w'), indent=1)
    print('-> bs2_partE_aperture.json')
    return rows


# ------------------------------------------------------------------ PART F
def part_F(nT=2048, n_cells=64, seed=0):
    """개구 검열의 **독립** 교차확인 — 공간 FFT 를 쓰지 않는다.
       연속 시계열 -> Welch -> Tp -> 심해 분산 lam = g*Tp^2/(2pi)  (심해 = lam 의 상한)
       => aperture/lam 의 *하한*.  깊이는 nc 에 없다(전 레코드 확인) -> 하한만 말할 수 있다.
       ★ 이 경로는 2D FFT 의 Hann 누설(DC->ring1) 교락을 완전히 회피한다."""
    from scipy import signal
    rng = np.random.default_rng(seed)
    rows = []
    for p in records():
        f = h5py.File(p, 'r'); zd = f['Z']; a = zd.attrs
        sf = float(a['scale_factor'][0]); off = float(a['add_offset'][0]); fill = int(a['_FillValue'][0])
        T, n0, n1 = zd.shape
        t = f['time'][:].astype(float)
        mask = (f['mask_Z'][:] != 0) if 'mask_Z' in f else np.ones((n0, n1), bool)
        i0 = max(0, T // 2 - nT // 2)
        raw = zd[i0:i0 + min(nT, T)]
        def _u2(name):
            if name not in f: return None
            d = f[name]; aa = d.attrs
            return d[:].astype(np.float64) * float(aa['scale_factor'][0]) + float(aa['add_offset'][0])
        Xg2, Yg2 = _u2('_nc4_non_coord_X'), _u2('Y')
        f.close()
        dt = float(np.median(np.diff(t)))
        Z = raw.astype(np.float64) * sf + off
        bad = (raw == fill) | (np.abs(np.abs(Z) - abs(SENTINEL)) < 1e-3)
        good = mask & (~bad.any(axis=0)) & (~(raw == raw[0]).all(axis=0))
        idx = np.flatnonzero(good.ravel())
        if idx.size == 0:
            rows.append(dict(record=rid(p), error='no good cells')); continue
        pick = rng.choice(idx, min(n_cells, idx.size), replace=False)
        ts = Z.reshape(Z.shape[0], -1)[:, pick]
        ts = ts - ts.mean(axis=0, keepdims=True)
        fr, ps = signal.welch(ts, fs=1 / dt, nperseg=min(512, ts.shape[0]), axis=0)
        Pm = ps.mean(axis=1)
        j = int(np.argmax(Pm[1:]) + 1)
        fp = float(fr[j]); Tp = 1 / fp if fp > 0 else np.nan
        lam_deep = G * Tp ** 2 / (2 * np.pi) if np.isfinite(Tp) else np.nan
        # 개구 (센티넬 제외 clean 셀의 bbox)
        d0 = float(np.median(np.abs(np.diff(Xg2[:, 0])))) if Xg2 is not None and np.nanstd(Xg2[:, 0]) > np.nanstd(Xg2[0, :]) else float(np.median(np.abs(np.diff(Yg2[:, 0]))))
        d1 = float(np.median(np.abs(np.diff(Yg2[0, :])))) if Yg2 is not None and np.nanstd(Yg2[0, :]) >= np.nanstd(Yg2[:, 0]) else float(np.median(np.abs(np.diff(Xg2[0, :]))))
        rr, cc = np.flatnonzero(good.any(1)), np.flatnonzero(good.any(0))
        L0, L1 = (rr[-1] - rr[0]) * d0, (cc[-1] - cc[0]) * d1
        # 시간 개구도 함께: 기록 길이 / Tp
        rows.append(dict(record=rid(p), dt=dt, n_frames_used=int(ts.shape[0]),
                         fp=fp, Tp=Tp, lam_deep_upper=lam_deep,
                         aperture_m=[L0, L1],
                         ap_over_lam_lowerbound=[L0 / lam_deep, L1 / lam_deep] if np.isfinite(lam_deep) else None,
                         record_len_over_Tp=float((len(t) * dt) / Tp) if np.isfinite(Tp) else None,
                         nyquist_lam_m=[2 * d0, 2 * d1]))
        r = rows[-1]
        print(f"  {r['record']:52s} Tp={Tp:5.2f}s  lam_deep<={lam_deep:7.1f} m  "
              f"aperture {L0:6.1f}x{L1:6.1f} m  ap/lam >= {L0/lam_deep:5.2f} x {L1/lam_deep:5.2f}")
    json.dump(rows, open(os.path.join(OUT, 'bs2_partF_dispersion_aperture.json'), 'w'), indent=1)
    print('-> bs2_partF_dispersion_aperture.json')
    return rows


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'A'
    for w in which:
        print(f'\n===== PART {w} =====')
        {'A': part_A, 'B': part_B, 'C': part_C, 'D': part_D, 'E': part_E, 'F': part_F}[w]()
