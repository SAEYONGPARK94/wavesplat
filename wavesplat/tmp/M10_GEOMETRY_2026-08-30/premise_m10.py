#!/usr/bin/env python3
"""M-10 전제 재도출 — ★**추정기 0회 · 결과 0행**. (2026-08-30)

    python3 tmp/M10_GEOMETRY_2026-08-30/premise_m10.py            # 전건 → premise_m10.json
    python3 tmp/M10_GEOMETRY_2026-08-30/premise_m10.py --check 3  # 코드 경로만 (일부 표본)

`PREREG.md` §1.2·§1.3·§3.2 의 수는 **집필 중 임시 스크립트로 실측**한 것이다.
이 파일이 그것을 **다시 만들고**, 문서와 어긋나면 `AssertionError` 다.
본보기 = `tmp/B2_REAL_2026-08-23/depth_recheck.py` · `tmp/STATUS_2026-08-28/f1_table.py` ·
`tmp/M5b_WINDOW_2026-08-28/recheck_verdict.py`.

⛔**이 파일의 수는 판정 경로가 아니다** — `EXPECTED` 는 *문서 대조용*이고, 판정은
`gates_m10.py` 가 결과 행에서만 낸다.
⛔**추정기·LETKF·학습을 부르지 않는다**(임포트도 하지 않는다).

═══════════════════════════════════════════════════════════════════════════════
★**AMENDMENT_1 (2026-08-30, 결과 0행) — `BLOCKER-P1` 반영.**
초판 `EXPECTED` 의 두 항(`synth lambda_p`·`synth median gap [lam]`)이 **BF4 전용값**이었다.
코퍼스는 **BF4 24 + BF5 24** 이고 λ_p 가 **76.504 / 126.466 m** 로 갈린다.
⟹ 정정: λ_p 76.5 → **101.485**(48 에피소드 중앙) · gap 1.15 → **0.868 λ_p** ·
   §1.3 의 *"25–30배"* → **17.9배**(최대 실측 대비).
✅**미터 단위 수는 전부 불변**(gap 중앙 88.14 m · 발자국 53.9×71.4 m · 면적 비중 9개) —
   틀린 것은 **λ 정규화 하나**였다.
★그리고 이 개정이 **검사 항목을 늘린다**(⛔줄이지 않는다): 층별 λ_p 2항 · 실측 12 레코드
   범위 2항 · 레코드 수 1항 · ★**밴드 경계의 방향 반전 2항**(§1.3-b) = **7항 신설**.
   상세 = `AMENDMENT_1.md`.
═══════════════════════════════════════════════════════════════════════════════
"""
import json
import os
import sys

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
for _p in (HERE, ROOT, os.path.join(ROOT, 'scripts'),
           os.path.join(ROOT, 'tmp', 'B2_REAL_2026-08-23')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(ROOT)

import gates_m10 as G                                            # noqa: E402

OUT = os.path.join(HERE, 'premise_m10.json')
GRAV = 9.81

# ── 문서 대조용 (⛔판정 경로 아님) — PREREG.md 의 수 ────────────────────────────
EXPECTED = {
    # §3.2 — 면적 비중 (48 에피소드, t > 2·Tp 프레임).  ★AMENDMENT_1 에서 **불변**.
    'shell_frac': {'obs:0_5': 0.0282, 'obs:5_10': 0.0269, 'obs:10_20': 0.0516,
                   'obs:20_30': 0.0537, 'obs:30_50': 0.1184, 'obs:50_75': 0.1578,
                   'obs:75_100': 0.1290, 'obs:100_150': 0.2232, 'obs:150_inf': 0.2112},
    'frac_ge_50m': 0.7212,
    # §1.3 — 질의점 → 최근접 관측 거리 (중앙), 합성.  ★미터 값은 **불변**.
    'synth_median_gap_m': 88.14,
    # ★AMENDMENT_1: 1.15 (= BF4 전용) → 0.868 (= 전 코퍼스).  ⛔초판 값 인용 금지.
    'synth_median_gap_lam': 0.868,
    # §1.2 — 관측 발자국 (합성).  ★불변.
    'synth_footprint_m': (53.9, 71.4),
    # ★AMENDMENT_1: 76.5 (= BF4 전용) → 101.485 (= 48 에피소드의 중앙).
    #   ⚠이 중앙은 **24/24 이봉 분포의 중점**이라 어느 에피소드도 갖지 않는 값이다
    #   ⟹ 크기를 말할 때는 아래 층별 값을 함께 적는다(단일 수로 인용 금지).
    'synth_lambda_p_m': 101.485,
    'synth_lambda_p_by_bf': {'BF4': 76.504, 'BF5': 126.466},     # ★신설 (층별, g·Tp²/2π)
    # §1.3 — 실측 (레코드별 중앙 gap / λ_p)
    'real_gap_lam': {'BS11a': 0.0422, 'AA14': 0.0376, 'YS17': 0.0485},
    'n_real': 12,                                                # ★신설
    'real_gap_lam_range': (0.0223, 0.0485),                      # ★신설 (12 레코드 전건)
    # ★신설 §1.3-b — **방향 반전**: `obs:0_5`(≤5 m)를 λ 로 환산하면 층마다 다르고,
    #   BF5 에서는 밴드가 실측 중앙보다 **좁다**(= 보수적 대리가 아니다).
    #   값 = 실측 12 중앙 중 그 층의 밴드 상한을 **넘는** 레코드 수.
    'n_real_above_band': {'BF4': 0, 'BF5': 2},
    'n_ep': 48,
    'ratio_min': 10.0,          # premise 의 핵심 전제(§1.3): 두 코퍼스가 한 자릿수 이상 어긋난다
}
RTOL = 0.05             # 5% — 프레임 표본 간격 등 집계 자유도를 허용하되 크기는 못 바꾼다
BAND_REAL_MAX_M = 5.0   # ⛔손복사 아님 — 아래에서 `G.EDGES` 로 대조한다(G-11 규약)


_CHECKED = []           # ★AMENDMENT_1: "N/N 통과" 를 **세어서** 찍는다.
                        #   초판은 항목 수를 어디에도 적지 않아 *"18/18"* 이 사람의 셈이었다.


def _fail(name, got, want, rtol=RTOL):
    if want == 0:
        ok = got == 0
    else:
        ok = abs(got - want) / abs(want) <= rtol
    _CHECKED.append(name)
    assert ok, ('%s: 재도출 %.6g 이 문서의 %.6g 과 어긋난다 (rtol %g)'
                % (name, got, want, rtol))


def band_upper_lam(lam_p):
    """`obs:0_5` 의 상한을 λ_p 단위로. ⛔5.0 을 손복사하지 않고 `G.EDGES` 에서 읽는다."""
    return float(G.EDGES[1]) / float(lam_p)


# ── 합성: 셸 면적 비중 · 피복 거리 · 발자국 ────────────────────────────────────
def synth(limit=None, frame_stride=10):
    eps = G.episodes()
    assert eps, 'HOS 혼잡 에피소드를 못 찾았다 — 공허 참조'
    if limit:
        eps = eps[:limit]
    keys = G.SHELL_KEYS
    edges = G.EDGES
    fr_all, gap_all, fx, fy, nobs, lam = [], [], [], [], [], []
    lam_bf = {}                # ★AMENDMENT_1: 층별 λ_p (BLOCKER-P1 의 자리)
    for name in eps:
        path = os.path.join(G.HOS, 'episode_%s.h5' % name)
        with h5py.File(path, 'r') as f:
            gx = f['frames/gt_grid_x'][:]
            gy = f['frames/gt_grid_y'][:]
            P = f['frames/lidar_points'][:]
            M = f['frames/lidar_valid_mask'][:]
            ts = f['frames/timestamps'][:]
            Tp = float(f['config'].attrs['Tp'])
        Xg, Yg = np.meshgrid(gx, gy)
        sel = np.flatnonzero(ts > 2.0 * Tp)      # ★기준선의 채점 창 규칙과 같은 규칙
        assert sel.size, '%s: 채점 프레임 0' % name
        fr_e, gap_e = [], []
        for i in sel[::frame_stride]:
            m = M[i].astype(bool)
            ox, oy = P[i, m, 0], P[i, m, 1]
            if ox.size == 0:
                continue
            md = np.sqrt((Xg[None] - ox[:, None, None]) ** 2
                         + (Yg[None] - oy[:, None, None]) ** 2).min(axis=0)
            fr_e.append([float(np.mean((md >= a) & (md < b)))
                         for a, b in zip(edges[:-1], edges[1:])])
            gap_e.append(float(np.median(md)))
            fx.append(float(ox.max() - ox.min()))
            fy.append(float(oy.max() - oy.min()))
            nobs.append(int(ox.size))
        fr_all.append(np.mean(fr_e, axis=0))
        gap_all.append(float(np.mean(gap_e)))
        lam.append(GRAV * Tp ** 2 / (2 * np.pi))
        lam_bf.setdefault(name.split('_')[0], []).append(lam[-1])   # 'BF4' / 'BF5'
    fr = np.mean(fr_all, axis=0)
    lam_m = float(np.median(lam))
    gap_m = float(np.median(gap_all))
    return dict(n_ep=len(eps),
                shell_frac={k: float(v) for k, v in zip(keys, fr)},
                frac_ge_50m=float(fr[5:].sum()),
                median_gap_m=gap_m, lambda_p_m=lam_m,
                median_gap_lam=gap_m / lam_m,
                # ★AMENDMENT_1 신설 — 코퍼스가 두 층이라는 사실을 **행에 싣는다**.
                lambda_p_by_bf={k: float(np.median(v)) for k, v in sorted(lam_bf.items())},
                n_ep_by_bf={k: len(v) for k, v in sorted(lam_bf.items())},
                footprint_m=(float(np.mean(fx)), float(np.mean(fy))),
                footprint_lam=(float(np.mean(fx)) / lam_m, float(np.mean(fy)) / lam_m),
                n_obs_per_frame=float(np.mean(nobs)),
                band_cells={b: int(round(sum(fr[keys.index(k)] for k in G.BANDS[b])
                                         * 80 * 60)) for b in G.BANDS})


# ── 실측: 같은 변수를 12 레코드에서 ───────────────────────────────────────────
def real(limit=None):
    from scene_station import scene_station, RECORDS_12       # ⛔추정기 0회
    recs = list(RECORDS_12)
    if limit:
        recs = recs[:limit]
    out = {}
    for r in recs:
        sc = scene_station(r, win=3)
        g = np.asarray(sc['maps_gt'][0], float)
        mm = np.isfinite(g)
        pts = np.column_stack([np.asarray(sc['Xq'])[mm], np.asarray(sc['Yq'])[mm]])
        P = np.asarray(sc['P'], float)
        d = np.sqrt((pts[:, None, 0] - P[None, :, 0]) ** 2
                    + (pts[:, None, 1] - P[None, :, 1]) ** 2).min(axis=1)
        lam = float(2 * np.pi / sc['kp'])
        out[r] = dict(n_query=int(mm.sum()), n_obs=int(len(P)), lambda_p_m=lam,
                      box_m=(float(sc['Xq'].max() - sc['Xq'].min()),
                             float(sc['Yq'].max() - sc['Yq'].min())),
                      median_gap_m=float(np.median(d)),
                      median_gap_lam=float(np.median(d)) / lam,
                      max_gap_m=float(d.max()), max_gap_lam=float(d.max()) / lam)
    return out


def main():
    check = None
    if '--check' in sys.argv:
        check = int(sys.argv[sys.argv.index('--check') + 1])
    s = synth(limit=check)
    r = real(limit=check)
    if check is None:
        for k, want in EXPECTED['shell_frac'].items():
            _fail('shell_frac[%s]' % k, s['shell_frac'][k], want)
        _fail('frac_ge_50m', s['frac_ge_50m'], EXPECTED['frac_ge_50m'])
        _fail('synth median gap [m]', s['median_gap_m'], EXPECTED['synth_median_gap_m'])
        _fail('synth median gap [lam]', s['median_gap_lam'], EXPECTED['synth_median_gap_lam'])
        _fail('synth lambda_p', s['lambda_p_m'], EXPECTED['synth_lambda_p_m'], 0.02)
        _fail('footprint x', s['footprint_m'][0], EXPECTED['synth_footprint_m'][0])
        _fail('footprint y', s['footprint_m'][1], EXPECTED['synth_footprint_m'][1])
        assert s['n_ep'] == EXPECTED['n_ep'], s['n_ep']
        for k, want in EXPECTED['real_gap_lam'].items():
            _fail('real gap/lam[%s]' % k, r[k]['median_gap_lam'], want)

        # ══ ★AMENDMENT_1 신설 7항 — `BLOCKER-P1` 이 다시는 조용히 지나가지 못하게 ══
        # (1)(2) 층별 λ_p — **코퍼스가 두 층**이라는 사실 자체를 기계가 지킨다.
        assert set(s['n_ep_by_bf']) == set(EXPECTED['synth_lambda_p_by_bf']), s['n_ep_by_bf']
        for bf, want in sorted(EXPECTED['synth_lambda_p_by_bf'].items()):
            _fail('synth lambda_p[%s]' % bf, s['lambda_p_by_bf'][bf], want, 0.02)
        # (3) 실측 레코드 수 — *"12 레코드"* 서술의 근거
        _fail('n_real', float(len(r)), float(EXPECTED['n_real']), 0.0)
        # (4)(5) 실측 12 전건의 중앙 gap/λ 범위 — 공시가 인용하는 그 범위
        lo, hi = EXPECTED['real_gap_lam_range']
        vals = [v['median_gap_lam'] for v in r.values()]
        _fail('real gap/lam min (12 전건)', min(vals), lo)
        _fail('real gap/lam max (12 전건)', max(vals), hi)
        # (6)(7) ★**방향 반전** — 층마다 `obs:0_5` 상한이 다르고 BF5 에서는 실측 중앙보다 좁다.
        #   ⛔이 두 항이 통과해야 §6-2 의 *"보수적 대리"* 서술을 층 없이 쓰지 못한다.
        for bf, want in sorted(EXPECTED['n_real_above_band'].items()):
            ub = band_upper_lam(s['lambda_p_by_bf'][bf])
            _fail('실측 중앙이 obs:0_5 상한(%s, %.5f λ)을 넘는 레코드 수' % (bf, ub),
                  float(sum(1 for v in vals if v > ub)), float(want), 0.0)
        # ══════════════════════════════════════════════════════════════════════

        # ★핵심 전제: 두 코퍼스가 같은 변수에서 한 자릿수 이상 어긋난다 (개정 후에도 통과)
        worst = max(vals)
        ratio = s['median_gap_lam'] / worst
        assert ratio > EXPECTED['ratio_min'], (s['median_gap_lam'], worst, ratio)
        s['ratio_to_worst_real'] = float(ratio)
        s['ratio_to_best_real'] = float(s['median_gap_lam'] / min(vals))
        s['band_upper_lam_by_bf'] = {bf: band_upper_lam(v)
                                     for bf, v in sorted(s['lambda_p_by_bf'].items())}
        json.dump(dict(synth=s, real=r, expected=EXPECTED), open(OUT, 'w'),
                  indent=1, ensure_ascii=False)
        print('premise_m10.json 기록 · 문서 대조 **%d/%d 통과**'
              % (len(_CHECKED), len(_CHECKED)))
        print('  (종전 규약 18항 + AMENDMENT_1 신설 7항. ⛔항목을 줄인 적 없다.)')
        print('  합성 gap/λ %.4f = 최대 실측의 **%.1f배** · 최소 실측의 %.1f배 (전제 문턱 >%.0f)'
              % (s['median_gap_lam'], ratio, s['ratio_to_best_real'],
                 EXPECTED['ratio_min']))
        print('  ★층별 λ_p %r · obs:0_5 상한 [λ] %r'
              % ({k: round(v, 3) for k, v in s['lambda_p_by_bf'].items()},
                 {k: round(v, 5) for k, v in s['band_upper_lam_by_bf'].items()}))
    else:
        print('[--check %d] 코드 경로만 확인 — 문서 대조·기록 생략' % check)
    print(' 합성  n_ep=%d  발자국 %.1f×%.1f m (%.2f×%.2f λ)  gap 중앙 %.1f m = %.3f λ'
          % (s['n_ep'], s['footprint_m'][0], s['footprint_m'][1],
             s['footprint_lam'][0], s['footprint_lam'][1],
             s['median_gap_m'], s['median_gap_lam']))
    print('        obs:0_5 면적 %.4f · ≥50 m 면적 %.4f · 밴드 격자점 %r'
          % (s['shell_frac']['obs:0_5'], s['frac_ge_50m'], s['band_cells']))
    for k, v in r.items():
        print(' 실측  %-6s box %.0f×%.0f m  λ %.1f m  q %d  gap 중앙 %.3f m = %.4f λ '
              '(최대 %.4f λ)' % (k, v['box_m'][0], v['box_m'][1], v['lambda_p_m'],
                                v['n_query'], v['median_gap_m'], v['median_gap_lam'],
                                v['max_gap_lam']))


if __name__ == '__main__':
    main()
