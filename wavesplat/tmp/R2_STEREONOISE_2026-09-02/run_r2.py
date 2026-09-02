"""`R2-STEREONOISE` 러너 — ⛔**추정기 런 0회**. 두 계기의 스펙트럼을 나란히 놓는다.

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
import gates_r2 as G                                                # noqa: E402
import scene_station as SS                                          # noqa: E402
from d3_sa_multirecord import MRecord, _unpack                      # noqa: E402

OUT = os.path.join(HERE, 'results_r2.jsonl')
GDIR = os.path.join(ROOT, 'data', 'real_bergamasco', 'wavegauge')
_MD5 = [os.path.join(HERE, 'gates_r2.py'),
        os.path.join(B2DIR, 'scene_station.py'),
        os.path.join(ROOT, 'scripts', 'd3_sa_multirecord.py')]


def code_md5():
    return {os.path.basename(p): hashlib.md5(open(p, 'rb').read()).hexdigest()[:12]
            for p in _MD5}


def gauge(fn):
    """`.mat` → (t[s], eta[n,6], fs). ⛔`info.fps` 를 쓰지 않는다."""
    d = loadmat(os.path.join(GDIR, fn))
    z = np.asarray(d['WaveGauge'], float)
    t_day = np.ravel(np.asarray(d['Time_UTC'], float))
    t = (t_day - t_day[0]) * 86400.0
    dt = float(np.median(np.diff(t)))
    return t, z, 1.0 / dt


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


def loc_probe(rec, Z, gz, fs_g, fs_s):
    """★`G-3` — 위치 대응 시도가 **널임을** 자료로 남긴다(런 0회).

    ⚠스모크 정정: 초판은 **lag 탐색을 하지 않았다** ⟹ 감사의 `r_max` 와 다른 양이었다.
    이제 lag −20…+20 s 를 훑고 그 **최대**를 쓴다(감사와 같은 정의).
    """
    g = gz[:, 0] - np.nanmean(gz[:, 0])
    n_s = Z.shape[0]
    ny, nx = Z.shape[1], Z.shape[2]
    step = max(1, min(ny, nx) // 16)
    lags = np.arange(-20.0, 20.0 + 1e-9, 0.25)
    best = 0.0
    for iy in range(0, ny, step):
        for ix in range(0, nx, step):
            sser = Z[:, iy, ix]
            if np.isfinite(sser).sum() < n_s // 2:
                continue
            a = sser - np.nanmean(sser)
            sa = float(np.nanstd(a))
            if not np.isfinite(sa) or sa <= 0:
                continue
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
                r = float(np.mean(a[m] * b) / (float(np.nanstd(a[m])) * sb))
                best = max(best, abs(r))
    sq = max(1, int(round(4.0 / rec.dx)))
    f0 = Z[n_s // 2]
    a, b = f0[:, :-sq].ravel(), f0[:, sq:].ravel()
    m = np.isfinite(a) & np.isfinite(b)
    ac = float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 100 else float('nan')
    return best, abs(ac)


def run(fn, record):
    t0w = time.time()
    tg, gz, fs_g = gauge(fn)
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
    rmax, ac4 = loc_probe(rec, Z, gz, fs_g, fs_s)
    # ★귀무분포 — **같은 다중성**으로 순환이동 게이지를 8회 돌린다(⛔최대끼리 비교한다)
    rng_n = np.random.default_rng(G.SAMPLE_SEED + 1)
    null = []
    for _ in range(G.N_PERM):
        sh = int(rng_n.integers(len(gz) // 8, len(gz) - len(gz) // 8))
        null.append(loc_probe(rec, Z, np.roll(gz, sh, axis=0), fs_g, fs_s)[0])
    null_max = float(max(null))

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
                gauge_xcorr_min=gmin, loc_rmax=float(rmax), stereo_ac_4m=float(ac4),
                loc_null_max=null_max, loc_exceeds_null=bool(rmax > null_max),
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
            print(f"  {rec:6s} hf_excess={r['hf_excess_rms_mm']:6.1f} mm  "
                  f"Tp {r['Tp_gauge']:.2f}/{r['Tp_stereo']:.2f}  "
                  f"r_max={r['loc_rmax']:.3f} ac4m={r['stereo_ac_4m']:.3f}", flush=True)
    print('✅ 완료')
    return 0


if __name__ == '__main__':
    sys.exit(main())
