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
# ★2요소만 보던 정규식이 `NOMINAL, CONFIRM, WEAK = 31, 34, 24` 를 통째로 놓쳤다
# (2026-08-23 `B-12`). ⟹ **N-튜플**로 일반화한다.
TUPLE = re.compile(r'^\s*([A-Z][A-Z0-9_]*(?:\s*,\s*[A-Z][A-Z0-9_]*)+)\s*=\s*'
                   r'(\d+(?:\s*,\s*\d+)+)\s*(?:#(?P<c1>.*))?$', re.M)
# ★배치들이 이미 **분모를 주석으로 적어 두었다** (`= 44, 48, 28   # /72`).
# 검사기가 그것을 무시하고 자기 추정 n 을 쓰면 **거짓 경고**가 난다
# (2026-08-23 `B-8b` 가 /72·/12 인데 36 으로 채점돼 3건 오탐).
# ⟹ 선언된 분모가 있으면 **그쪽이 권위**다.
# 두 표기를 다 받는다: `# /72` 와 `# (n=16, 문턱)`.
# ⚠**동결된 gates 파일은 고치지 않는다** — 이미 돌아간 배치의 md5 를 깨면 재현이 끊긴다.
# ⟹ 검사기가 배치의 표기에 맞춘다(반대가 아니다).
DENOM = re.compile(r'/\s*(\d+)|\bn\s*=\s*(\d+)')


def _den(txt):
    m = DENOM.search(txt or '')
    return int(m.group(1) or m.group(2)) if m else None
NAMES = ('NOMINAL', 'CONFIRM', 'NOSKILL', 'THRESHOLD', 'SELFTEST_MIN', 'NEG_MIN',
         'PARTIAL_LO', 'REC_NOMINAL', 'REC_CONFIRM', 'REC_NOSKILL', 'WEAK')
NSIZE = ('N_PAIRS', 'N_EPISODES', 'N_POOL', 'N_NEW', 'N_EP', 'N_REC', 'N_WIN')


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
    warn = skipped = 0
    for d in dirs:
        for gf in sorted(glob.glob(os.path.join(d, 'gates*.py'))):
            src = open(gf).read()
            vals, dens = {}, {}
            for m in TUPLE.finditer(src):
                ks = [x.strip() for x in m.group(1).split(',')]
                vs = [int(x) for x in m.group(2).split(',')]
                dm = _den(m.group('c1'))
                if len(ks) == len(vs):
                    vals.update(dict(zip(ks, vs)))
                    if dm:
                        dens.update({k: dm for k in ks})
            for m in CONST.finditer(src):
                if m.group(1) in NAMES:
                    vals.setdefault(m.group(1), int(m.group(2)))
                    line = src[m.start():src.find('\n', m.start())]
                    dm = _den(line.split('#', 1)[1]) if '#' in line else None
                    if dm:
                        dens[m.group(1)] = dm
            if not vals:
                continue
            ns = n_candidates(src, gf)
            # ★파일이 **주석으로 분모를 선언**했으면 그것도 후보다
            # (`W11` 이 `# /44` 라 적고도 검사에서 조용히 빠졌다).
            ns = sorted(set(ns) | set(dens.values()))
            if not ns:
                # ★★검사기가 **조용히 건너뛰면** "경고 0건" 이 "검사함"과 구분되지 않는다.
                # 2026-08-23 `B-12` 가 `N_EP` 라 적어 이 자리에서 소리 없이 빠졌다.
                # ⟹ 침묵을 **경고로 승격**한다.
                print(f'{os.path.relpath(gf)}  ⚠**표본 크기를 못 찾아 검사하지 못했다** '
                      f'(문턱 {sorted(vals)} 은 검사되지 않음)')
                skipped += 1
                continue
            n = max(ns)   # ⚠후보가 여럿이면 최대를 쓴다 — 아래에 후보 전체를 인쇄한다
            rows = []
            for k, v in sorted(vals.items()):
                n = dens.get(k, max(ns))      # ★선언된 분모가 있으면 그쪽이 권위
                if not (0 < v <= n):
                    continue
                # ★이름으로 방향을 **추측하지 마라** — 소스에서 **실제 사용**을 읽는다.
                # (`PARTIAL_LO` 는 이름이 `_LO` 인데 코드에선 `w >= PARTIAL_LO` 다
                #  ⟹ 이름 기반 추측이 2026-08-23 에 오탐 2건을 냈다.)
                ge = len(re.findall(r'[><]?>=?\s*' + k + r'\b', src))
                le = len(re.findall(r'<=?\s*' + k + r'\b', src))
                lo = le > ge if (ge or le) else ('NOSKILL' in k or 'WEAK' in k)
                p = le_p(v, n) if lo else sign_p(v, n)
                bad = p > 0.05
                rows.append((k, v, n, p, bad, '≤' if lo else '≥'))
            if rows:
                print(f'{os.path.relpath(gf)}  (n≈{max(ns)}; 후보 {ns}'
                      + (f'; 선언 분모 {sorted(set(dens.values()))}' if dens else '') + ')')
                for k, v, n_, p, bad, op in rows:
                    # ⚠자기검정(SELFTEST)·음성대조(NEG) 문턱은 **일부러 관대한 것이 옳다**
                    # (그것은 우월성 시험이 아니라 '바닥' 이다). 그래서 따로 표시한다.
                    # ⚠`PARTIAL*` 은 **유의 주장이 아니라 중간 띠의 이름**이다 ⟹ 관대해도 옳다.
                    lenient_ok = k.startswith(('SELFTEST', 'NEG', 'PARTIAL'))
                    flag = ('  ⓘ의도적으로 관대(유의 주장 아님)' if bad and lenient_ok
                            else '  ⚠우연과 구분 안 됨' if bad else '')
                    if bad and lenient_ok:
                        bad = False
                    print(f'   {k:14s} {op}{v:3d}/{n_:<4d}  p={p:.4f}{flag}')
                    warn += bad
                if 'NOSKILL' in vals and 'NOMINAL' in vals and \
                        dens.get('NOSKILL', max(ns)) == dens.get('NOMINAL', max(ns)):
                    nn = dens.get('NOSKILL', max(ns))
                    if vals['NOSKILL'] != nn - vals['NOMINAL']:
                        print(f'   ⚠NOSKILL={vals["NOSKILL"]} 이 형제 규약 n-NOMINAL'
                              f'={nn - vals["NOMINAL"]} 과 다르다')
    print(f'\n우연과 구분되지 않는 문턱: {warn}건 · ⚠검사 못 한 gates 파일: {skipped}건')
    print('⛔이 검사기는 판정을 바꾸지 않는다 — 문턱이 말이 되는지만 본다.')
    return 1 if (warn or skipped) else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
