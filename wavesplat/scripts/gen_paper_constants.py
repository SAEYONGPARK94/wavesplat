#!/usr/bin/env python3
"""`paper/paper_constants.tex` 생성 — ⛔논문의 수치를 **손으로 적지 마라**.

선례 = `scripts/gen_deploy_constants.py`(학위논문 배포 상수).
사유 = 이 저장소가 반복해서 대가를 치른 **「손복사 기준 상수」** 결함 부류.
`paper/wavesplat.tex` 는 철회된 헤드라인을 **라이브 22줄**에 손으로 인쇄하고 있었고,
그 22줄은 서로 독립적으로 낡을 수 있었다.

★이 스크립트는 **동결 `verdict_*.json` 에서 값을 읽고**, 아래 `PINS` 와 어긋나면
`AssertionError` 로 죽는다. ⟹ 배치가 재분석되면 논문이 **조용히** 낡지 않는다.

사용:
    python3 scripts/gen_paper_constants.py          # 검증 + 생성
    python3 scripts/gen_paper_constants.py --show   # 값만 출력
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
TMP = os.path.join(ROOT, 'tmp')
OUT = os.path.join(ROOT, 'paper', 'paper_constants.tex')


def _get(v, path):
    cur = v
    for k in path.split('.'):
        if isinstance(cur, list):
            cur = cur[int(k)]
        else:
            cur = cur[k]
    return cur


def load(rel):
    p = os.path.join(TMP, rel)
    if not os.path.exists(p):
        raise SystemExit(f'⛔동결 판정이 없다: {p}')
    return json.load(open(p))


# ─────────────────────────────────────────────────────────────────────────────
# ★인용 원자 — (매크로, 배치 verdict, json 경로, 형식, 핀)
#   ⛔핀은 "오늘 값" 이다. 값이 바뀌면 **의도적으로** 갱신하라(래칫과 같은 규약).
#   ⚠`kind`: 'int' | 'pct'(부호 포함 소수1) | 'p'(과학표기) | 'ratio'(a/b)
# ─────────────────────────────────────────────────────────────────────────────
ATOMS = [
    # ── 헤드라인 (혼잡 층, 최강 적격 격자 기준선 대비)
    ('BOneBetter',  'B1_BASELINE_2026-08-22/verdict_b1.json', 'crowded.better', 'int', 62),
    ('BOneN',       'B1_BASELINE_2026-08-22/verdict_b1.json', 'crowded.n', 'int', 64),
    ('BOneRel',     'B1_BASELINE_2026-08-22/verdict_b1.json', 'crowded.rel_median_pct', 'pct', -38.6),
    # ── 비선형 참값 · 신선 시드 확장 (확증)
    ('BTwelveBetter', 'B12_HOS_EXTEND_2026-08-23/verdict_b12.json', 'ship_vs_spec.better', 'int', 43),
    ('BTwelveN',      'B12_HOS_EXTEND_2026-08-23/verdict_b12.json', 'ship_vs_spec.n', 'int', 48),
    ('BTwelveRel',    'B12_HOS_EXTEND_2026-08-23/verdict_b12.json', 'ship_vs_spec.rel_median_pct', 'pct', -40.2),
    ('BTwelveP',      'B12_HOS_EXTEND_2026-08-23/verdict_b12.json', 'ship_vs_spec.p_one_sided', 'p', 6.84e-09),
    # ★음성대조 — 비교자에게도 기술이 있는가 (F1 교훈)
    ('BTwelveCmpBetter', 'B12_HOS_EXTEND_2026-08-23/verdict_b12.json', 'spec_vs_zero.better', 'int', 47),
    ('BTwelveCmpN',      'B12_HOS_EXTEND_2026-08-23/verdict_b12.json', 'spec_vs_zero.n', 'int', 48),
    # ── 표현 절제: 파수 연속성 (혼잡)
    ('BSixteenBetter', 'B16_KGRID_2026-08-24/verdict_b16.json', 'primary.better', 'int', 46),
    ('BSixteenN',      'B16_KGRID_2026-08-24/verdict_b16.json', 'primary.n', 'int', 48),
    ('BSixteenRel',    'B16_KGRID_2026-08-24/verdict_b16.json', 'primary.rel_median_pct', 'pct', -38.6),
    ('BSixteenP',      'B16_KGRID_2026-08-24/verdict_b16.json', 'primary.p_one_sided', 'p', 4.18e-12),
    ('BSixteenDose',   'B16_KGRID_2026-08-24/verdict_b16.json', 'dose.better', 'int', 47),
    # ── 표현 절제: 파수 연속성 (희소)
    ('BNineteenBetter', 'B19_KGRID_SPARSE_2026-08-25/verdict_b19.json', 'primary.better', 'int', 14),
    ('BNineteenN',      'B19_KGRID_SPARSE_2026-08-25/verdict_b19.json', 'primary.n', 'int', 16),
    ('BNineteenRel',    'B19_KGRID_SPARSE_2026-08-25/verdict_b19.json', 'primary.rel_median_pct', 'pct', -67.4),
]

FMT = {
    'int': lambda x: str(int(x)),
    'pct': lambda x: f'{x:+.1f}\\%'.replace('+', '') if x < 0 else f'{x:+.1f}\\%',
    'p': lambda x: _sci(x),
}


def _sci(x):
    from math import floor, log10
    if x == 0:
        return '0'
    e = int(floor(log10(abs(x))))
    m = x / 10 ** e
    return f'{m:.2f}\\times 10^{{{e}}}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--show', action='store_true')
    a = ap.parse_args()

    cache, rows, bad = {}, [], []
    for name, rel, path, kind, pin in ATOMS:
        v = cache.setdefault(rel, load(rel))
        try:
            raw = _get(v, path)
        except (KeyError, IndexError, TypeError) as e:
            bad.append((name, f'경로 없음 {path}: {e}'))
            continue
        # 핀 대조 — 정수는 정확히, 실수는 표시 자릿수에서
        if kind == 'int':
            ok = int(raw) == int(pin)
        elif kind == 'pct':
            ok = abs(round(raw, 1) - pin) < 1e-9
        else:
            ok = abs(raw - pin) / abs(pin) < 0.005
        if not ok:
            bad.append((name, f'핀 {pin} != 실측 {raw}'))
        rows.append((name, FMT[kind](raw), rel.split('/')[0]))

    if a.show or bad:
        for n, val, src in rows:
            print(f'  {n:22s} = {val:22s}  [{src}]')
    if bad:
        print('\n⛔핀 불일치:')
        for n, why in bad:
            print(f'   {n}: {why}')
        raise AssertionError(f'{len(bad)}건 — 배치가 재분석됐다면 ATOMS 의 핀을 '
                             '**의도적으로** 갱신하라(래칫 규약).')

    body = ['% ⛔자동 생성 — 손으로 고치지 마라.',
            '%    생성: python3 scripts/gen_paper_constants.py',
            '%    출처: 동결 verdict_*.json (배치명은 각 줄 주석에)',
            '%    ★이 파일이 존재하는 이유: paper/wavesplat.tex 가 철회된 헤드라인을',
            '%      라이브 22줄에 **손으로** 인쇄하고 있었고 그 줄들이 독립적으로 낡았다.', '']
    for n, val, src in rows:
        body.append(f'\\newcommand{{\\{n}}}{{{val}}}  % {src}')
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, 'w').write('\n'.join(body) + '\n')
    if not a.show:
        for n, val, src in rows:
            print(f'  \\{n} = {val}')
    print(f'\n✅{len(rows)} 상수 · 핀 불일치 0 → {os.path.relpath(OUT, ROOT)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
