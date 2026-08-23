#!/usr/bin/env python3
"""판정 문턱 검사기 — ★**세 번째 방벽 부류**: *"라벨 함수의 산술을 아무도 검사하지 않는다"*
(2026-08-23, R3 심사 토론 BLOCKER ① + 2차 F-13).

같은 날 **두 결함이 같은 뿌리에서 나왔다**:
  · **F-13** `gates_b9.label(w, n)` 가 `n` 을 **레코드 수(12)** 로 받는데 `w` 는 **창 수(/36)**
    ⟹ `LOSES-GAIN` 조건이 `w ≤ −12` = **도달 불가**. ★**불리한 라벨만** 막혔다.
  · **BLOCKER ①** `gates_b10.NOSKILL = 18` 이 `Binomial(36,½)` 의 **널 중앙값**
    (P(X≤18)=0.566) ⟹ **우연 이하 어떤 결과에서도 발화**. 형제 일곱은 전부 `n−NOMINAL`(=12).

⟹ 사람이 매번 손으로 확인할 수 없다. **기계가 본다.**

검사 (배치 디렉터리별):
  A. `gates*.py` 에서 `NOMINAL` / `CONFIRM` / `NOSKILL` / `THRESHOLD` 류 상수와 표본 크기 후보를 읽는다
  B. 각 문턱에서의 **단측 이항 p** 를 계산한다
  C. ⚠**p > 0.05 인 문턱**을 경고한다 = *"이 문턱은 우연과 구분되지 않는다"*
  D. ⚠**문자 `18` 처럼 형제 규약(`n - NOMINAL`)에서 벗어난 하드코딩**을 표시한다

⛔**이 검사기는 판정을 바꾸지 않는다.** 문턱이 **말이 되는지**만 본다.
사용:  python3 tmp/_audit/check_thresholds.py [tmp/BATCH ...]
"""
import glob, os, re, sys
from math import comb


def sign_p(w, n):
    return sum(comb(n, k) for k in range(w, n + 1)) / 2 ** n


def le_p(w, n):
    return sum(comb(n, k) for k in range(0, w + 1)) / 2 ** n


CONST = re.compile(r'^\s*([A-Z][A-Z0-9_]*)\s*=\s*(\d+)\s*(?:#.*)?$', re.M)
PAIR = re.compile(r'^\s*([A-Z][A-Z0-9_]*)\s*,\s*([A-Z][A-Z0-9_]*)\s*=\s*(\d+)\s*,\s*(\d+)', re.M)
NAMES = ('NOMINAL', 'CONFIRM', 'NOSKILL', 'THRESHOLD', 'SELFTEST_MIN', 'NEG_MIN',
         'PARTIAL_LO', 'REC_NOMINAL', 'REC_CONFIRM', 'REC_NOSKILL')
NSIZE = ('N_PAIRS', 'N_EPISODES', 'N_POOL', 'N_NEW')


def n_candidates(src, path=None):
    out = set()
    # ★import 로 표본 정의를 가져오는 배치가 많다(`from gates_b3 import RECORDS, WINDOWS, N_PAIRS`).
    # 그 경우 이 파일만 읽으면 n 을 못 찾는다 ⟹ import 원본을 한 단계 따라간다.
    if path:
        for m in re.finditer(r'^from\s+(gates_\w+)\s+import', src, re.M):
            # ⚠import 원본이 **다른 배치 디렉터리**에 있는 경우가 많다
            # (`B-10` 이 `B-3` 의 gates 를 가져온다) ⟹ 형제 디렉터리도 뒤진다.
            cands = [os.path.join(os.path.dirname(path), m.group(1) + '.py')]
            cands += sorted(glob.glob(os.path.join(
                os.path.dirname(os.path.dirname(path)), '*', m.group(1) + '.py')))
            for sib in cands:
                if os.path.exists(sib) and os.path.abspath(sib) != os.path.abspath(path):
                    out |= set(n_candidates(open(sib).read()))
                    break
    for m in CONST.finditer(src):
        if m.group(1) in NSIZE:
            out.add(int(m.group(2)))
    # RECORDS/WINDOWS 로부터 유도
    r = re.search(r'^RECORDS\s*=\s*[\[(]([^\])]*)[\])]', src, re.M | re.S)
    w = re.search(r'^WINDOWS\s*=\s*[\[(]([^\])]*)[\])]', src, re.M | re.S)
    if r:
        nr = len([x for x in r.group(1).split(',') if x.strip()])
        out.add(nr)
        if w:
            out.add(nr * len([x for x in w.group(1).split(',') if x.strip()]))
    return sorted(x for x in out if 2 <= x <= 4096)


def main(argv):
    dirs = argv[1:] or sorted({os.path.dirname(f)
                               for f in glob.glob('tmp/**/gates*.py', recursive=True)})
    warn = 0
    for d in dirs:
        for gf in sorted(glob.glob(os.path.join(d, 'gates*.py'))):
            src = open(gf).read()
            vals = {}
            for m in PAIR.finditer(src):
                vals[m.group(1)] = int(m.group(3)); vals[m.group(2)] = int(m.group(4))
            for m in CONST.finditer(src):
                if m.group(1) in NAMES:
                    vals.setdefault(m.group(1), int(m.group(2)))
            if not vals:
                continue
            ns = n_candidates(src, gf)
            if not ns:
                continue
            n = max(ns)   # ⚠후보가 여럿이면 최대를 쓴다 — 아래에 후보 전체를 인쇄한다
            rows = []
            for k, v in sorted(vals.items()):
                if not (0 < v <= n):
                    continue
                lo = 'NOSKILL' in k or k in ('PARTIAL_LO',)
                p = le_p(v, n) if lo else sign_p(v, n)
                bad = p > 0.05
                rows.append((k, v, n, p, bad, '≤' if lo else '≥'))
            if rows:
                print(f'{os.path.relpath(gf)}  (n≈{n}; 후보 {ns})')
                for k, v, n_, p, bad, op in rows:
                    # ⚠자기검정(SELFTEST)·음성대조(NEG) 문턱은 **일부러 관대한 것이 옳다**
                    # (그것은 우월성 시험이 아니라 '바닥' 이다). 그래서 따로 표시한다.
                    lenient_ok = k.startswith(('SELFTEST', 'NEG'))
                    flag = ('  ⓘ의도적으로 관대(바닥 게이트)' if bad and lenient_ok
                            else '  ⚠우연과 구분 안 됨' if bad else '')
                    if bad and lenient_ok:
                        bad = False
                    print(f'   {k:14s} {op}{v:3d}/{n_:<4d}  p={p:.4f}{flag}')
                    warn += bad
                if 'NOSKILL' in vals and 'NOMINAL' in vals and \
                        vals['NOSKILL'] != n - vals['NOMINAL']:
                    print(f'   ⚠NOSKILL={vals["NOSKILL"]} 이 형제 규약 n-NOMINAL'
                          f'={n - vals["NOMINAL"]} 과 다르다')
    print(f'\n우연과 구분되지 않는 문턱: {warn}건')
    print('⛔이 검사기는 판정을 바꾸지 않는다 — 문턱이 말이 되는지만 본다.')
    return 1 if warn else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
