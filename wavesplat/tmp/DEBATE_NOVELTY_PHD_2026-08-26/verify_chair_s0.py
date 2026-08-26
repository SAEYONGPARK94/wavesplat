"""토론 §0 의 매개 주장을 기계적으로 재현·검사한다 (2026-08-27).

목적은 **판정을 되돌릴 수 있게 만드는 것**이지 새 결과를 만드는 것이 아니다.
⛔이 스크립트가 인쇄하는 §0 재현값은 **인용 대상이 아니다** — 사전등록이
결과 전에 금지한 비교이기 때문이다(B19/PREREG.md:417,419 · B16/PREREG.md:52).

  python3 tmp/DEBATE_NOVELTY_PHD_2026-08-26/verify_chair_s0.py

검사 항목 (전부 PASS 여야 기록이 유효하다):
  C-1  두 절제 배치에 `SPEC25` 행이 0개다        -> §0 은 배치 간 결합이다
  C-2  §0 수치가 그 결합에서 정확히 재현된다      -> 위원장의 산수는 맞다
  C-3  두 배치의 `SHIP-NOW` 열이 비트 동일하다    -> 표본 불일치가 아니다
  C-4  사전등록 금지 문장이 축자 그대로 있다      -> 위반이 사실이다
  C-5  절제 팔이 `ls_fit_wphi` 를 호출한다        -> 교락이 코드에 있다
"""
import collections
import glob
import json
import math
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CROWDED = ('tmp/B16_KGRID_2026-08-24/results_b16_*.jsonl',
           'tmp/B12_HOS_EXTEND_2026-08-23/results_b12.jsonl')
SPARSE = ('tmp/B19_KGRID_SPARSE_2026-08-25/results_b19_*.jsonl',
          'tmp/B11_HOS_BASELINE_2026-08-23/results_b11b.jsonl')

# 위원장이 §0 에서 제시한 값 (결과 전에 이 스크립트에 박아 두는 것이 아니라,
# 토론 기록에서 옮겨 적은 대조값이다)
CHAIR = {'crowded': (21, 48, 0.844), 'sparse': (8, 16, 0.598)}


def load(pat):
    rows = []
    for f in sorted(glob.glob(os.path.join(ROOT, pat))):
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def per_episode(rows, arm, key='avg_nRMSE'):
    """에피소드별 추정기-시드 중앙값 (기록 배치의 집계 규약)."""
    d = collections.defaultdict(list)
    for r in rows:
        if r.get('arm') == arm and r.get(key) is not None:
            d[r['episode']].append(r[key])
    return {k: median(v) for k, v in d.items()}


def sign_p(wins, n):
    """단측 부호검정 P(X >= wins)."""
    return sum(math.comb(n, k) for k in range(wins, n + 1)) / 2 ** n


def arm_census(pat):
    c = collections.Counter(r.get('arm') for r in load(pat))
    return dict(c)


def main():
    ok = []

    # ── C-1 ──────────────────────────────────────────────────────────────
    for name, (abl, _) in (('crowded', CROWDED), ('sparse', SPARSE)):
        arms = arm_census(abl)
        ok.append(('C-1 %-7s 절제 배치에 SPEC25 0행' % name, 'SPEC25' not in arms, arms))

    # ── C-2 / C-3 ────────────────────────────────────────────────────────
    for name, (abl, base) in (('crowded', CROWDED), ('sparse', SPARSE)):
        A, B = load(abl), load(base)
        k1 = per_episode(A, 'KGRID1')
        sa = per_episode(A, 'SHIP-NOW')
        sp = per_episode(B, 'SPEC25')
        sb = per_episode(B, 'SHIP-NOW')
        common = sorted(set(k1) & set(sp))
        n = len(common)
        wins = sum(1 for e in common if k1[e] < sp[e])
        p = sign_p(wins, n)
        rel = median([(k1[e] - sp[e]) / sp[e] * 100 for e in common])
        cw, cn, cp = CHAIR[name]
        ok.append(('C-2 %-7s §0 재현 (%d/%d p=%.4f rel %+.2f%%)' % (name, wins, n, p, rel),
                   (wins, n) == (cw, cn) and abs(p - cp) < 5e-3,
                   '위원장 %d/%d p=%.3f' % (cw, cn, cp)))
        same = sum(1 for e in common if e in sb and abs(sa[e] - sb[e]) < 1e-12)
        ok.append(('C-3 %-7s SHIP 열 두 배치 비트동일 %d/%d' % (name, same, n),
                   same == n, ''))

    # ── C-4 ──────────────────────────────────────────────────────────────
    bans = [('tmp/B19_KGRID_SPARSE_2026-08-25/PREREG.md', 417, '외부 기준선 비교가 아니다'),
            ('tmp/B19_KGRID_SPARSE_2026-08-25/PREREG.md', 419, '크기를 직접 비교하지 않는다'),
            ('tmp/B16_KGRID_2026-08-24/PREREG.md', 52, '판정을 대체하지 않는다')]
    for rel_path, lineno, needle in bans:
        lines = open(os.path.join(ROOT, rel_path)).read().splitlines()
        hit = 1 <= lineno <= len(lines) and needle in lines[lineno - 1]
        ok.append(('C-4 %s:%d "%s"' % (os.path.basename(rel_path), lineno, needle),
                   hit, '' if hit else (lines[lineno - 1][:60] if lineno <= len(lines) else 'EOF')))

    # ── C-5 ──────────────────────────────────────────────────────────────
    src = open(os.path.join(ROOT, 'wavesplat/optimizer/sequential_update.py')).read()
    m = re.search(r'def _snap_k_to_grid\(.*?\n(.*?)\n    def ', src, re.S)
    body = m.group(1) if m else ''
    ok.append(('C-5 절제 팔이 ls_fit_wphi 로 (w,phi) 를 덮어쓴다',
               'ls_fit_wphi' in body and 'self.model.w.copy_' in body, ''))

    print('=== §0 검사 (2026-08-27) ===')
    for label, good, extra in ok:
        print('  %-4s %s%s' % ('PASS' if good else 'FAIL', label,
                               ('  | ' + str(extra)) if extra else ''))
    n_fail = sum(1 for _, g, _ in ok if not g)
    print('\n%d/%d PASS' % (len(ok) - n_fail, len(ok)))
    print('⛔재현값은 인용 대상이 아니다 — 사전등록이 결과 전에 금지한 비교다.')
    raise SystemExit(1 if n_fail else 0)


if __name__ == '__main__':
    main()
