#!/usr/bin/env python3
"""★셸 면적가중 재집계 — **런 0회**. 동결 행만 읽어 논문 표를 다시 만들고 본문과 대조한다.

    python3 tmp/STATUS_2026-08-30/shell_recount.py

배경 (2026-08-30, 16인 심사 §0):
  적대적 축의 최강 공격이 *"면적의 대부분에서 비교자가 기술 바닥 근처인데 어느 게이트도
  그것을 본 적이 없다"* 였다. 그 공격은 **동결 행 재집계로 닫힌다** — 비교자가 기술을
  가진 셸만 남겨도 마진은 줄지 않고 **커진다**.

⛔이 스크립트는 **사전등록 밖의 재집계**다 ⟹ 산출은 전부 `EXPLORATORY` 이고
  라벨을 만들지 않는다. `M-10` 의 판정(`verdict_m10.json`)은 이 파일이 건드리지 않는다.

⚠**규약 공시**(⛔지우지 마라):
  · 자료 = `tmp/M10_GEOMETRY_2026-08-30/results_m10_*.jsonl` (288행 = 48 에피소드 × 2 팔 × 3 시드).
  · 에피소드 접기 = 시드 3개의 **중앙값**(`gates_m10.med_by_episode` 와 같은 규약, 셸별로 적용).
  · 셸 합성 = **면적 가중 RMS**(`gates_m10.band_rmse`, 손복사 0 — 임포트한다).
  · 비교자 기술(셸 k) = `1 - SPEC25_k / ZERO_k`  (제로필드는 행의 `zero_shell_batch`).
  · 마스크는 **셀 단위**(에피소드 × 셸)다. `<5%` 행에서 마스크가 빈 에피소드 2개는
    상대차 0 · 면적 0 으로 남긴다(그 에피소드에는 `<5%` 영역이 존재하지 않는다 = 승리 아님).
  · ⚠**면적가중 RMS 는 추정기 자신의 전역 집계(`avg_nRMSE`)와 같은 수가 아니다** —
    이 스크립트가 그 어긋남을 직접 재서 찍는다(`CONV` 행).
"""
import glob
import hashlib
import json
import os
import re
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
M10 = os.path.join(ROOT, 'tmp', 'M10_GEOMETRY_2026-08-30')
TEX = os.path.join(ROOT, 'thesis', 'thesis_main.tex')
for _p in (M10, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gates_m10 as G                                              # noqa: E402

KEYS = list(G.SHELL_KEYS)          # ⛔손복사 금지 — run_s16.SHELL_KEYS 단일출처
ARMS = ('SHIP', 'SPEC25')


def load_rows():
    out = []
    for path in sorted(glob.glob(os.path.join(M10, 'results_m10_*.jsonl'))):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
    return out


def med_shell(rows, arm, field):
    """에피소드 -> {셸: 시드 중앙값}."""
    acc = {}
    for r in rows:
        if 'error' in r or r.get('arm') != arm:
            continue
        v = r.get(field)
        if not v:
            continue
        acc.setdefault(r['episode'], []).append(v)
    return {e: {k: st.median([d[k] for d in ds if d.get(k) is not None]) for k in KEYS}
            for e, ds in acc.items()}


def med_scalar(rows, arm, key):
    acc = {}
    for r in rows:
        if 'error' in r or r.get('arm') != arm or r.get(key) is None:
            continue
        acc.setdefault(r['episode'], []).append(float(r[key]))
    return {e: st.median(v) for e, v in acc.items()}


def main():
    rows = load_rows()
    ship = med_shell(rows, 'SHIP', 'shell_rmse')
    spec = med_shell(rows, 'SPEC25', 'shell_rmse')
    fr_s = med_shell(rows, 'SHIP', 'shell_frac')
    fr_c = med_shell(rows, 'SPEC25', 'shell_frac')
    z_s = med_shell(rows, 'SHIP', 'zero_shell_batch')
    z_c = med_shell(rows, 'SPEC25', 'zero_shell_batch')
    eps = sorted(set(ship) & set(spec))
    assert len(rows) == 288 and len(eps) == 48, (len(rows), len(eps))

    def skill(e, k):
        return 1.0 - spec[e][k] / z_c[e][k]

    def recount(pred):
        """(w, n, rel_median, area_median). 마스크가 빈 에피소드는 rel 0 · area 0."""
        rel, area, w = [], [], 0
        for e in eps:
            keep = [k for k in KEYS if pred(skill(e, k))]
            if not keep:
                rel.append(0.0)
                area.append(0.0)
                continue
            a = G.band_rmse(ship[e], fr_s[e], keep)
            b = G.band_rmse(spec[e], fr_c[e], keep)
            rel.append((a - b) / b)
            area.append(sum(fr_s[e][k] for k in keep))
            if a < b:
                w += 1
        return w, len(eps), st.median(rel), st.median(area)

    table = [
        ('전체 9셸', lambda s: True),
        ('비교자 기술 >=5%', lambda s: s >= 0.05),
        ('비교자 기술 >=20%', lambda s: s >= 0.20),
        ('비교자 기술 <5%', lambda s: s < 0.05),
    ]
    out = {'batch': 'shell-recount (EXPLORATORY, 사전등록 밖)', 'n_rows': len(rows),
           'n_episodes': len(eps), 'rows': {}}
    print('# 셸 면적가중 재집계 (EXPLORATORY · 런 0회 · n=%d 에피소드)' % len(eps))
    print('%-20s %8s %8s %10s' % ('재집계', '면적', 'SHIP<SPEC25', 'rel_median'))
    for name, pred in table:
        w, n, rel, area = recount(pred)
        out['rows'][name] = dict(area=area, better=w, n=n, rel_median=rel)
        print('%-20s %7.1f%% %8s %9.2f%%' % (name, 100 * area, '%d/%d' % (w, n), 100 * rel))

    # 전역 기술 (에피소드 접기 규약 = 위와 동일)
    all9 = {a: {e: G.band_rmse(m[e], f[e], KEYS) for e in eps}
            for a, m, f in (('SHIP', ship, fr_s), ('SPEC25', spec, fr_c))}
    z9 = {a: {e: G.band_rmse(m[e], f[e], KEYS) for e in eps}
          for a, m, f in (('SHIP', z_s, fr_s), ('SPEC25', z_c, fr_c))}
    for a in ARMS:
        out.setdefault('global_skill', {})[a] = 1 - st.median([all9[a][e] / z9[a][e] for e in eps])
        print('전역 기술 %-8s = %.2f%%' % (a, 100 * out['global_skill'][a]))

    # 최외곽 셸: 면적과 비교자/제로필드 비
    out['outermost'] = dict(
        shell=KEYS[-1],
        area=st.median([fr_c[e][KEYS[-1]] for e in eps]),
        ratio_spec_over_zero=st.median([spec[e][KEYS[-1]] / z_c[e][KEYS[-1]] for e in eps]))
    print('최외곽 셸 %s: 면적 %.2f%% · 비교자/제로필드 = %.3f'
          % (out['outermost']['shell'], 100 * out['outermost']['area'],
             out['outermost']['ratio_spec_over_zero']))

    # ★규약 차이 — 면적가중 RMS 대 추정기 자신의 전역 집계
    a_own = med_scalar(rows, 'SHIP', 'avg_nRMSE')
    b_own = med_scalar(rows, 'SPEC25', 'avg_nRMSE')
    w_own, n_own, rel_own = G.paired_wins(a_own, b_own)
    w_rms, n_rms, rel_rms, _ = recount(lambda s: True)
    out['convention_gap'] = dict(
        area_weighted=dict(better=w_rms, n=n_rms, rel_median=rel_rms),
        estimator_own=dict(better=w_own, n=n_own, rel_median=rel_own),
        delta_wins=w_rms - w_own, delta_pp=100 * (rel_rms - rel_own))
    print('CONV 규약 차이: 면적가중 RMS %d/%d %.2f%% vs 추정기 자신 %d/%d %.2f%%'
          ' ⟹ %d승 · %.2f pp'
          % (w_rms, n_rms, 100 * rel_rms, w_own, n_own, 100 * rel_own,
             w_rms - w_own, 100 * (rel_rms - rel_own)))

    # ── 논문 본문과 대조 (⛔손복사 방지: 문서가 어긋나면 죽는다) ────────────────
    tex = open(TEX, encoding='utf-8').read()
    out['thesis_md5'] = hashlib.md5(open(TEX, 'rb').read()).hexdigest()
    want = []
    for name, pred in table:
        w, n, rel, area = recount(pred)
        want.append(('%d/%d' % (w, n), r'\$?%d/48\$?' % w))
        want.append(('%.2f%%' % (100 * rel), re.escape('%.2f' % abs(100 * rel))))
        want.append(('area %.1f%%' % (100 * area), re.escape('%.1f' % (100 * area))))
    want.append(('전역 기술 SPEC25', re.escape('%.1f' % (100 * out['global_skill']['SPEC25']))))
    want.append(('최외곽 비', re.escape('%.3f' % out['outermost']['ratio_spec_over_zero'])))
    missing = [lbl for lbl, pat in want if not re.search(pat, tex)]
    out['thesis_missing'] = missing
    print('\n논문 대조: %d항 중 누락 %d' % (len(want), len(missing)))
    if missing:
        print('  ⛔누락:', missing)
    with open(os.path.join(HERE, 'shell_recount.json'), 'w') as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    if missing:
        raise AssertionError('논문 본문이 재집계와 어긋난다: %s' % missing)
    print('✅ 재집계 == 논문 본문')


if __name__ == '__main__':
    main()
