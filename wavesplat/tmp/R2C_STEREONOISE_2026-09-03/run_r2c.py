"""`R2c-STEREONOISE` 러너 — ⛔**추정기 런 0회**. 두 계기의 스펙트럼을 나란히 놓는다.

⚠자료의 함정 셋을 코드가 직접 피한다(`PREREG §2`):
  · `info.fps` 는 **7개 전부 `'10 Hz'`** 인데 BS_2013 은 실제 **≈20 Hz** ⟹ `Time_UTC` 에서 도출
  · `Time_UTC` 는 **MATLAB datenum(일)** — 초로 바꾼다
  · 공통 Nyquist 밖은 비교하지 않는다
"""
import hashlib
import json
import os
import sys
import time

import numpy as np
from scipy.io import loadmat
from scipy.signal import welch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
B2DIR = os.path.join(ROOT, 'tmp', 'B2_REAL_2026-08-23')
for p in (HERE, B2DIR, ROOT, os.path.join(ROOT, 'scripts')):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(ROOT)
import gates_r2c as G                                                # noqa: E402
import scene_station as SS                                          # noqa: E402
from d3_sa_multirecord import MRecord, _unpack                      # noqa: E402

OUT = os.path.join(HERE, 'results_r2c.jsonl')
GDIR = os.path.join(ROOT, 'data', 'real_bergamasco', 'wavegauge')
_MD5 = [os.path.join(HERE, 'gates_r2c.py'),
        os.path.join(B2DIR, 'scene_station.py'),
        os.path.join(ROOT, 'scripts', 'd3_sa_multirecord.py')]


def code_md5():
    return {os.path.basename(p): hashlib.md5(open(p, 'rb').read()).hexdigest()[:12]
            for p in _MD5}


def stereo_epoch(path):
    """★스테레오 파일이 **스스로 선언한** 절대 시작 시각."""
    from datetime import datetime
    import h5py
    with h5py.File(path, 'r') as f:
        v = f.attrs['Acquisition starting date [yyyymmdd HHMMSS]']
    v = v.decode() if isinstance(v, bytes) else str(v)
    return datetime.strptime(v.strip(), '%Y%m%d %H%M%S')


def gauge_epoch(tg_datenum):
    """MATLAB datenum → `datetime`. ⛔`info.fps` 는 3건에서 거짓이므로 쓰지 않는다."""
    from datetime import datetime, timedelta
    d = float(tg_datenum)
    return datetime.fromordinal(int(d)) + timedelta(days=d % 1) - timedelta(days=366)


def gauge(fn):
    """`.mat` → (t[s], eta[n,6], fs). ⛔`info.fps` 를 쓰지 않는다."""
    d = loadmat(os.path.join(GDIR, fn))
    z = np.asarray(d['WaveGauge'], float)
    t_day = np.ravel(np.asarray(d['Time_UTC'], float))
    t = (t_day - t_day[0]) * 86400.0
    dt = float(np.median(np.diff(t)))
    # ★`t_day[0]` 을 함께 돌려준다 --- 절대 시각 도출에 필요하다(⛔선행은 버렸다).
    return t, z, 1.0 / dt, float(t_day[0])


def spec(x, fs, fmax):
    f, p = welch(x, fs=fs, nperseg=min(G.NPERSEG, len(x)))
    m = (f > 0) & (f <= fmax)
    return f[m], p[m]


def moments(f, p, band):
    m = (f >= band[0]) & (f <= band[1])
    if not m.any():
        return float('nan'), float('nan'), float('nan')
    m0 = float(np.trapezoid(p[m], f[m]))
    m2 = float(np.trapezoid(p[m] * f[m] ** 2, f[m]))
    tp = float(1.0 / f[m][int(np.argmax(p[m]))])
    t02 = float(np.sqrt(m0 / m2)) if m2 > 0 else float('nan')
    return 4.0 * np.sqrt(max(m0, 0.0)), tp, t02


def stereo_pack(record):
    cfg = SS.RECORDS_12[record]
    rec = MRecord(os.path.join(ROOT, cfg['path']), stride=cfg['stride'])
    nfw = int(round(40.0 / rec.dt))
    n_lead = int(round(14.0 / rec.dt))
    lo, hi = nfw + 1, rec.T_all - n_lead - 2
    i0 = int(np.linspace(lo, hi, 6).astype(int)[SS_WIN])
    Z = rec.window(i0 - nfw + 1, nfw)
    return rec, Z, i0, nfw


SS_WIN = 3


def loc_probe(rec, Z, gz, fs_g, fs_s, lag_center=0.0):
    """★`G-3` — 위치 대응 시도가 **널임을** 자료로 남긴다(런 0회).

    ⚠스모크 정정: 초판은 **lag 탐색을 하지 않았다** ⟹ 감사의 `r_max` 와 다른 양이었다.
    이제 lag −20…+20 s 를 훑고 그 **최대**를 쓴다(감사와 같은 정의).
    """
    g = gz[:, 0] - np.nanmean(gz[:, 0])
    n_s = Z.shape[0]
    ny, nx = Z.shape[1], Z.shape[2]
    step = max(1, min(ny, nx) // 16)
    # ★탐색 **폭**은 선행과 같고 **중심**만 참 정렬로 옮긴다(PREREG §1).
    # ⚠이름 충돌 --- `step` 은 이미 **격자 서브샘플**이다(위). lag 쪽은 다른 이름을 쓴다.
    dlag = G.lag_step(fs_g, fs_s)
    lags = lag_center + np.arange(-G.LAG_HALFSPAN_S, G.LAG_HALFSPAN_S + 1e-9, dlag)
    best, best_lag, pt_best = 0.0, float('nan'), []
    for iy in range(0, ny, step):
        for ix in range(0, nx, step):
            sser = Z[:, iy, ix]
            if np.isfinite(sser).sum() < n_s // 2:
                continue
            a = sser - np.nanmean(sser)
            sa = float(np.nanstd(a))
            if not np.isfinite(sa) or sa <= 0:
                continue
            pt = 0.0
            for L in lags:
                gi = (np.arange(n_s) / fs_s + L) * fs_g
                gi = np.round(gi).astype(int)
                m = (gi >= 0) & (gi < len(g)) & np.isfinite(a)
                if m.sum() < n_s // 2:
                    continue
                b = g[gi[m]]
                sb = float(np.std(b))
                if sb <= 0:
                    continue
                r = abs(float(np.mean(a[m] * b) / (float(np.nanstd(a[m])) * sb)))
                if r > best:
                    best, best_lag = r, float(L)
                pt = max(pt, r)
            pt_best.append(pt)
    sq = max(1, int(round(4.0 / rec.dx)))
    f0 = Z[n_s // 2]
    a, b = f0[:, :-sq].ravel(), f0[:, sq:].ravel()
    m = np.isfinite(a) & np.isfinite(b)
    ac = float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 100 else float('nan')
    # ★봉우리 피복률(보고 전용) — 선행이 **기록하지 않은** 양이다.
    cov = (float(np.mean(np.asarray(pt_best) >= G.PEAK_FRAC * best))
           if pt_best and best > 0 else float('nan'))
    return best, abs(ac), float(best_lag), cov


def ft_surrogate(x, rng):
    """★FT 위상무작위화 — 진폭 스펙트럼을 **보존**하고 위상만 깬다.

    ⛔`IAAFT` 가 아니다(`PREREG §1`): IAAFT 가 추가로 보존하는 **진폭 주변분포**는
    선형 상관 통계량에 필요 없다. 그 판단을 주장하지 않고 **왜도·첨도를 행에 기록**한다.
    """
    x = np.asarray(x, float)
    mu = float(np.nanmean(x))
    x = np.where(np.isfinite(x), x, mu)
    n = len(x)
    X = np.fft.rfft(x - mu)
    ph = rng.uniform(0.0, 2 * np.pi, len(X))
    ph[0] = 0.0
    if n % 2 == 0:
        ph[-1] = 0.0
    return np.fft.irfft(np.abs(X) * np.exp(1j * ph), n) + mu


def pseudo_gauge(Z, rec, n_g, fs_g, lag_s, win_start_s):
    """★탐지기 양성 대조 — **절대 시각으로** 합성한다.

    ⛔선행 `R2b` 는 의사 게이지를 **스테레오 창의 국소 시계**로 만들어 `loc_probe` 의
    (틀린) 규약과 정확히 일치했다 ⟹ 통과가 **당연**했다. 여기서는 창의 절대 시작
    `win_start_s` 를 더해 **게이지 레코드의 시계** 위에 놓는다.
    """
    ny, nx = Z.shape[1], Z.shape[2]
    step = max(1, min(ny, nx) // 16)              # ⛔`loc_probe` 와 같은 규약
    cand = [(y, x) for y in range(0, ny, step) for x in range(0, nx, step)
            if np.isfinite(Z[:, y, x]).all()]
    if not cand:
        raise RuntimeError('탐색 부분격자에 완전유효 점이 없다')
    cy, cx = cand[len(cand) // 2]                 # ⛔난수 아님 — 결과 전 규약
    ser = np.asarray(Z[:, cy, cx], float)
    t_s = win_start_s + np.arange(len(ser)) * rec.dt      # ★절대(레코드) 시각
    t_g = np.arange(n_g) / fs_g
    return np.interp(t_g - lag_s, t_s, ser)[:, None], int(cy), int(cx), step



def run(fn, record):
    t0w = time.time()
    tg, gz, fs_g, tg_datenum0 = gauge(fn)
    rec, Z, i0, nfw = stereo_pack(record)
    fs_s = 1.0 / rec.dt
    fmax = min(fs_g, fs_s) / 2.0

    # 스테레오 표본점 (⛔시드 결과 전 고정)
    rng = np.random.default_rng(G.SAMPLE_SEED)
    valid = np.isfinite(Z).all(axis=0)
    iy, ix = np.nonzero(valid)
    take = rng.choice(len(iy), min(G.N_SAMPLE_PTS, len(iy)), replace=False)
    Ps, Pg = [], []
    for k in take:
        f_, p_ = spec(Z[:, iy[k], ix[k]], fs_s, fmax)
        Ps.append(p_)
    fS, PS = f_, np.mean(Ps, axis=0)
    for c in range(gz.shape[1]):
        f_, p_ = spec(gz[:, c] - np.nanmean(gz[:, c]), fs_g, fmax)
        Pg.append(p_)
    fG, PG = f_, np.mean(Pg, axis=0)
    # 공통 격자로 보간
    PGi = np.interp(fS, fG, PG)

    hf = fS >= G.HF_CUT_HZ
    var_s = float(np.trapezoid(PS[hf], fS[hf])) if hf.any() else float('nan')
    var_g = float(np.trapezoid(PGi[hf], fS[hf])) if hf.any() else float('nan')
    hm0_s, tp_s, t02_s = moments(fS, PS, G.WAVE_BAND)
    hm0_g, tp_g, t02_g = moments(fS, PGi, G.WAVE_BAND)

    # ⚠스모크 정정: 전 15쌍의 최소는 **외곽-외곽 0.5 m** 를 포함해 낮게 나온다.
    #   게이지 건강성의 물리적 정의는 **중심(ch0) vs 외곽 5개**(0.25 m)다.
    xc = np.corrcoef(gz.T)
    gmin = float(np.min(xc[0, 1:]))

    def probe_with_null(gg, tag, lag_center):
        """관측 1회 + **FT 위상무작위화 귀무 `B` 회**를 같은 탐색으로 돌린다."""
        r, ac, lag, cov = loc_probe(rec, Z, gg, fs_g, fs_s, lag_center)
        rn = np.random.default_rng(G.NULL_SEED + (0 if tag == 'obs' else 1))
        nulls = []
        for _ in range(G.N_NULL):
            sur = ft_surrogate(gg[:, 0], rn)[:, None]
            nulls.append(loc_probe(rec, Z, sur, fs_g, fs_s, lag_center)[0])
        p95 = float(np.percentile(nulls, G.PCTL))
        return dict(rmax=float(r), ac=float(ac), lag=float(lag), cov=float(cov),
                    null_p95=p95, null_med=float(np.median(nulls)),
                    null_max=float(max(nulls)), exceeds=bool(r > p95))

    # ★참 정렬 = 창의 절대 시작 + 두 시계의 차(둘 다 **자료에서 도출**)
    win_start_s = (i0 - nfw + 1) * rec.dt
    se = stereo_epoch(os.path.join(ROOT, SS.RECORDS_12[record]['path']))
    ge = gauge_epoch(tg_datenum0)
    clock_off = (ge - se).total_seconds()
    align = win_start_s - clock_off
    lo_l, hi_l = align - G.LAG_HALFSPAN_S, align + G.LAG_HALFSPAN_S
    obs = probe_with_null(gz, 'obs', align)
    gc, cy, cx, gstep = pseudo_gauge(Z, rec, len(gz), fs_g, G.CTRL_LAG_S, win_start_s)
    # ★대조는 **자기 정렬**을 중심으로 훑는다 --- 의사 게이지를 레코드 시계 위에 놓고
    #   `lag_s` 를 주입했으므로 그 정점은 `win_start + lag_s` = `align + clock_off + lag_s` 다.
    #   ⛔`align` 을 그대로 쓰면 정점이 좁힌 탐색 밖으로 나간다(동결 전 스모크가 잡았다).
    ctrl_center = align + clock_off + G.CTRL_LAG_S
    ctrl = probe_with_null(gc, 'ctrl', ctrl_center)

    g0 = np.asarray(gz[:, 0], float); g0 = g0[np.isfinite(g0)]
    gz_c = g0 - g0.mean(); gsd = float(gz_c.std())
    skew = float(np.mean(gz_c ** 3) / gsd ** 3) if gsd > 0 else float('nan')
    kurt = float(np.mean(gz_c ** 4) / gsd ** 4) if gsd > 0 else float('nan')

    sync = (tg[0] * 0.0)  # 게이지 t 는 0 기준; 동기는 파일명 UTC 로 확인한다
    return dict(record=record, gauge_file=fn,
                fs_gauge=fs_g, fs_stereo=fs_s, fs_source='Time_UTC',
                f_max_used=float(fmax), sample_seed=G.SAMPLE_SEED,
                n_pts=int(len(take)), n_frames=int(Z.shape[0]),
                hf_var_stereo=var_s, hf_var_gauge=var_g,
                hf_excess_rms_mm=float(np.sqrt(max(var_s - var_g, 0.0)) * 1000.0),
                stereo_gt_gauge=bool(var_s > var_g),
                Hm0_stereo=hm0_s, Hm0_gauge=hm0_g,
                Tp_stereo=tp_s, Tp_gauge=tp_g,
                T02_stereo=t02_s, T02_gauge=t02_g,
                gauge_xcorr_min=gmin, stereo_ac_4m=obs['ac'],
                lag_step_s=float(G.lag_step(fs_g, fs_s)), n_lags=int(
                    round(2 * G.LAG_HALFSPAN_S / G.lag_step(fs_g, fs_s)) + 1),
                clock_offset_s=float(clock_off), win_start_s=float(win_start_s),
                alignment_s=float(align), lag_lo=float(lo_l), lag_hi=float(hi_l),
                span_contains_alignment=bool(lo_l <= align <= hi_l),
                ctrl_lag_injected_s=float(G.CTRL_LAG_S),
                ctrl_center_s=float(ctrl_center),
                ctrl_lag_recovered_s=float(ctrl['lag'] - align - clock_off),
                ctrl_timebase='absolute',
                obs_rmax=obs['rmax'], obs_null_p95=obs['null_p95'],
                obs_null_med=obs['null_med'], obs_null_max=obs['null_max'],
                obs_exceeds_null=obs['exceeds'], obs_lag_s=obs['lag'],
                obs_lag_minus_align=float(obs['lag'] - align),
                obs_peak_cov=obs['cov'],
                ctrl_rmax=ctrl['rmax'], ctrl_null_p95=ctrl['null_p95'],
                ctrl_exceeds_null=ctrl['exceeds'], ctrl_lag_s=ctrl['lag'],
                ctrl_peak_cov=ctrl['cov'], ctrl_point=[cy, cx],
                ctrl_point_on_grid=bool(cy % gstep == 0 and cx % gstep == 0),
                grid_step=int(gstep),
                gauge_skew=skew, gauge_kurt=kurt,
                n_null=G.N_NULL, null_seed=G.NULL_SEED,
                sync_off_samples=float(sync), nan_frac=float(np.isnan(gz).mean()),
                pid=os.getpid(), code_md5=code_md5(),
                wall_s=round(time.time() - t0w, 2))


def main():
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            if l.strip():
                done.add(json.loads(l).get('record'))
    with open(OUT, 'a') as fh:
        for fn, rec in G.PAIRS:
            if rec in done:
                continue
            r = run(fn, rec)
            fh.write(json.dumps(r, ensure_ascii=False) + '\n'); fh.flush()
            print(f"  {rec:6s} hf={r['hf_excess_rms_mm']:6.1f}mm "
                  f"obs r={r['obs_rmax']:.3f} p95={r['obs_null_p95']:.3f} "
                  f"{'초과' if r['obs_exceeds_null'] else ' 널 '} | "
                  f"대조 r={r['ctrl_rmax']:.3f} p95={r['ctrl_null_p95']:.3f} "
                  f"lag={r['ctrl_lag_s']:+.2f}s "
                  f"{'OK' if r['ctrl_exceeds_null'] else 'X '} "
                  f"피복 {r['obs_peak_cov']:.2f} {r['wall_s']:.0f}s", flush=True)
    print('✅ 완료')
    return 0


if __name__ == '__main__':
    sys.exit(main())
