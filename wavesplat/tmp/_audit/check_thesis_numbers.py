#!/usr/bin/env python3
"""논문 수치 **존재 대장** — ⛔**자문 전용. 이 도구는 검증하지 않는다.**

★★★**2026-08-30 강등.** 이 검사기는 그전까지 주장마다 `✔` 를 찍었고 `run_all.sh` 4번에
**종료코드**가 붙어 있었다. 돌연변이 시험이 그 `✔` 를 무력화했다(측정은 아래 자가진단이
**매 실행마다 다시 계산**한다 — ⛔이 문서의 수를 손으로 갱신하지 마라):

    2026-08-30 실측 (thesis_main.tex, 주석 포함 618건 · 사전 972쌍)
      w−1 로 흔들어도 통과 ........ 553/618 = 89.5%
      w±1 중 하나라도 통과 ........ 592/618 = 95.8%   ⟹ 검출력 4.2%
      퍼센트 ±1.0pp 통과 .......... 290/291 = 99.7%
      n=12·16·24·32 : 가능한 w 가 **100% 사전에 존재** ⟹ `x/12` 는 0/12 든 12/12 든 무조건 통과

⟹ ★***`✔` 는 "이 수가 옳다" 가 아니라 "저장소 어딘가에 이 두 자리 수가 있다" 였다.***
   실제 오귀속도 확인됐다 — `M-1`(2026-08-28 실행)의 수에 이 검사기가 붙인 출처는
   `B26_SEASTATE`(다른 배치)와 `DEBATE_method_2026-08-15/M2_methodology.md`
   (**M-1 이 돌기 13일 전의 토론 메모**)였고, 그러고도 `대응 없음 0건` 을 찍었다.

★**원인(2026-08-30 확정, 셋 다지만 지배적인 것은 첫째)**
  1. ★**사전이 너무 넓다.** 972쌍 중 **874쌍(89.9%)이 `.md` 산문 정규식에서만** 나온다.
     훑는 `.md` 749개 중 **422개가 토론·사전등록·개정·현황·감사 메모**다 — 거기엔 가설,
     남의 배치 수, 인용금지로 축자 인용된 수, 적대적 "만약" 수가 전부 들어 있다.
     n 이 작으면(12·16·24) 0..n 이 **전부** 채워져 검사가 **항등적으로 참**이 된다.
  2. **출처를 좁히지 않는다.** 검사는 "어딘가 있는가" 이지 "이 주장이 인용한 배치에 있는가"
     가 아니다. ⚠원고는 `% 근거: tmp/…/verdict_*.json` 주석으로 출처를 **이미 선언**하는데
     (thesis_main.tex 에 164곳) 이 검사기는 주석 줄을 건너뛰어 그것을 **읽지 않는다**.
  3. **매칭이 느슨하다.** 퍼센트는 1자리 반올림 절대값으로만 만나므로 952개 값이 실수 축을
     촘촘히 덮어 ±1.0pp 교란의 99.7%가 살아남는다.

★**왜 고치지 않고 강등했나** (셋 다 측정했다):
  · 사전을 `verdict*.json` 구조화 키로만 좁힘 → 98쌍, 그래도 n=12 포화 11/13 = **미달**.
  · `% 근거:` 주석으로 출처 한정(창 12~60줄) → 죽임 **34.4%**, 게다가 **오탐 118건**
    (한 문단의 근거 주석이 이웃 문단으로 번진다).
  · 배치 단위로 완벽히 좁혀도 한 `verdict_*.json` 안에 팔이 여럿이라 w±1 이 같이 산다.
  ⟹ ★**≥90% 에 닿으려면 "어느 verdict **키**" 까지 내려가야 하고, 그건 이미 존재하는
     다른 도구다** — `tmp/STATUS_2026-08-28/audit_thesis_numbers.py`(run_all.sh 8번,
     기대 문자열을 verdict 에서 **계산**한다). 같은 것을 두 번 만들지 않는다.

★**그래서 이 도구가 지금 하는 일 — 판정이 아니라 대장 세 장:**
  (A) ⚠**대응이 아예 없는 수** — 이것만이 이 설계가 가진 유일한 진짜 신호다.
  (B) ⚠**검출력이 없는 주장** — w±1 도 사전에 있어 이 검사가 원리적으로 못 보는 것.
  (C) ★**핀 없는 수** — 8번이 판정하는 값과 겹치지 않는 수 = **어느 검사기도 안 보는 수**.
      ⚠(C) 는 **값 일치 상한**이다(같은 `12/16` 이면 다른 배치 핀이라도 덮인 것으로 센다)
      ⟹ 실제 피복은 이보다 **낮다**. 늘어나는 쪽으로만 읽어라.

⛔**종료코드 없음(항상 0).** 빨간불이 늘 켜져 있어 신호가 죽는 것을 막기 위함이고,
  ★무엇보다 ***`✔` 가 검증으로 읽히는 것이 이 저장소 최악의 상태***이기 때문이다.
  판정은 `run_all.sh` 8번이 한다.

사용:  python3 tmp/_audit/check_thesis_numbers.py [tex 파일 ...]
"""
import glob, json, os, re, sys

COUNT = re.compile(r'(?<![\d./])(\d{1,4})\s*/\s*(\d{1,4})(?![\d/])')
PCT = re.compile(r'([-+−]?\d{1,3}\.\d{1,2})\s*\\?%')
SKIP_N = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 1000}

PINS = 'tmp/STATUS_2026-08-28/audit_thesis_numbers.py'


def collect_verdicts(root='tmp'):
    """⚠**구조화된 키만 보면 놓친다** — 판정 파일 상당수가 카운트를 `label_text` 문자열
    안에만 갖고 있고, 오래된 배치는 `RESULT*.md` 에만 있다. ⟹ **세 곳 모두** 훑는다.
    ⚠**그 너비가 곧 이 도구의 무력함**이다(위 원인 1). 좁히지 않는 이유는 위 docstring."""
    pairs, pcts = {}, {}   # (better, n) -> [출처] · |퍼센트| -> [출처]
    for f in sorted(glob.glob(os.path.join(root, '**', 'verdict*.json'), recursive=True)
                    + glob.glob(os.path.join(root, '**', '*.md'), recursive=True)):
        try:
            txt = open(f, encoding='utf-8').read()
        except Exception:
            continue
        for m in COUNT.finditer(txt):
            w, n = int(m.group(1)), int(m.group(2))
            if n not in SKIP_N and w <= n:
                pairs.setdefault((w, n), []).append(os.path.relpath(f) + '(본문)')
        for m in PCT.finditer(txt):
            pcts.setdefault(round(abs(float(m.group(1).replace('−', '-'))), 1),
                            []).append(os.path.relpath(f) + '(본문)')
    for f in sorted(glob.glob(os.path.join(root, '**', 'verdict*.json'), recursive=True)):
        try:
            v = json.load(open(f))
        except Exception:
            continue
        stack = [(os.path.relpath(f), v)]
        while stack:
            src, node = stack.pop()
            if isinstance(node, dict):
                b, n = node.get('better'), node.get('n')
                if isinstance(b, int) and isinstance(n, int):
                    pairs.setdefault((b, n), []).append(src)
                r = node.get('rel_median_pct')
                if isinstance(r, (int, float)):
                    pcts.setdefault(round(abs(r), 1), []).append(src)
                for k, val in node.items():
                    if isinstance(val, (dict, list)):
                        stack.append((f'{src}:{k}', val))
            elif isinstance(node, list):
                for it in node:
                    if isinstance(it, (dict, list)):
                        stack.append((src, it))
    return pairs, pcts


def load_pins():
    """8번 검사기가 **verdict 에서 계산**하는 기대 문자열 집합. 실패하면 빈 집합(대장 (C) 만 생략).
    ⛔손복사하지 않는다 — 8번의 CHECKS 표를 그대로 평가한다."""
    import contextlib, io
    try:
        src = open(PINS, encoding='utf-8').read()
        ns = {'__name__': '_pins'}
        head = src.split('# ---8<--- CHECKS_TABLE_END')[0]
        if 'CHECKS' not in head:
            raise RuntimeError('8번의 CHECKS_TABLE_END 표지가 사라졌다 — 잘라내기 실패')
        with contextlib.redirect_stdout(io.StringIO()):   # 8번의 머리말 출력을 삼킨다
            exec(compile(head, PINS, 'exec'), ns)
        out = set()
        for label, rel, path, fmt in ns['CHECKS']:
            try:
                out.add(re.sub(r'[+\-−\s]', '', fmt(ns['jget'](rel, path))))
            except Exception:
                pass
        return out
    except BaseException as e:   # ⚠SystemExit 도 잡는다(전체 파일을 exec 하면 8번이 종료해 버린다)
        print(f'⚠핀 표를 읽지 못했다({type(e).__name__}: {e}) — 대장 (C) 를 생략한다.')
        return set()


def scan(tex, with_comments):
    counts, percents = [], []
    for i, ln in enumerate(open(tex, encoding='utf-8').read().splitlines(), 1):
        if not with_comments and ln.lstrip().startswith('%'):
            continue
        for m in COUNT.finditer(ln):
            w, n = int(m.group(1)), int(m.group(2))
            if n in SKIP_N or w > n:
                continue
            counts.append((i, w, n))
        for m in PCT.finditer(ln):
            v = m.group(1).replace('−', '-')
            if float(v) == 0.0:
                continue
            percents.append((i, v))
    return counts, percents


def selftest(counts, percents, pairs, pcts):
    """★**검출력을 매 실행마다 다시 잰다** — 이 검사기가 스스로에 대해 하는 유일한 정직한 진술.
    ⛔문서의 수를 믿지 마라. 여기서 나온 수가 사실이다."""
    if not counts:
        return
    m1 = sum(1 for _, w, n in counts if w - 1 >= 0 and (w - 1, n) in pairs)
    m2 = sum(1 for _, w, n in counts
             if (w - 1 >= 0 and (w - 1, n) in pairs) or (w + 1 <= n and (w + 1, n) in pairs))
    mp = sum(1 for _, v in percents
             if round(abs(float(v)) - 1.0, 1) in pcts or round(abs(float(v)) + 1.0, 1) in pcts)
    print('# ── 자가 검출력 측정 (이 실행에서 계산) ' + '─' * 34)
    print(f'#   w−1 교란 생존      {m1}/{len(counts)} = {100*m1/len(counts):.1f}%')
    print(f'#   w±1 교란 생존      {m2}/{len(counts)} = {100*m2/len(counts):.1f}%'
          f'   ⟹ ★검출력 {100-100*m2/len(counts):.1f}%')
    if percents:
        print(f'#   퍼센트 ±1.0pp 생존 {mp}/{len(percents)} = {100*mp/len(percents):.1f}%')
    sat = []
    for n in (12, 16, 24, 32, 48):
        have = sum(1 for w in range(n + 1) if (w, n) in pairs)
        sat.append(f'n={n}:{have}/{n+1}')
    print(f'#   사전 포화도        {" · ".join(sat)}   (100% = 그 분모는 무조건 통과)')
    print('# ' + '─' * 72)


def main(argv):
    texs = argv[1:] or sorted(glob.glob('thesis/*.tex') + glob.glob('paper/*.tex'))
    pairs, pcts = collect_verdicts()
    pins = load_pins()
    print(f'판정 파일에서 모은 (승/전체) 쌍: {len(pairs)}개 · 퍼센트: {len(pcts)}개 '
          f'· 8번이 판정하는 고유 기대값: {len(pins)}개')
    print('⛔**이 도구는 판정하지 않는다 — 자문 전용, 종료코드 없음.** '
          '`✔` 를 쓰지 않는 이유는 파일 머리말 참조.')

    for tex in texs:
        if not os.path.exists(tex):
            continue
        counts, percents = scan(tex, with_comments=False)
        if not counts and not percents:
            continue
        print(f'\n=== {tex}   (본문 카운트 {len(counts)} · 퍼센트 {len(percents)})')
        selftest(counts, percents, pairs, pcts)

        # (A) 대응이 아예 없는 수 — 이 설계의 유일한 진짜 신호
        miss = [(i, f'{w}/{n}') for i, w, n in counts if (w, n) not in pairs]
        miss += [(i, f'{v}%') for i, v in percents if round(abs(float(v)), 1) not in pcts]
        print(f'\n  (A) ⚠**대응하는 기록이 없다**: {len(miss)}건')
        for i, c in sorted(miss):
            print(f'        {i:5d}  {c}')

        # (B) 검출력이 없는 주장
        blind = [(i, f'{w}/{n}') for i, w, n in counts
                 if (w - 1 >= 0 and (w - 1, n) in pairs) or (w + 1 <= n and (w + 1, n) in pairs)]
        print(f'  (B) ⚠**이 검사가 원리적으로 못 보는 주장**(w±1 도 사전에 있다): '
              f'{len(blind)}/{len(counts)} = {100*len(blind)/max(len(counts),1):.1f}%')

        # (C) 핀 없는 수 — 어느 검사기도 판정하지 않는 수
        if pins:
            unpin_c = [(i, f'{w}/{n}') for i, w, n in counts if f'{w}/{n}' not in pins]
            unpin_p = [(i, f'{v}%') for i, v in percents
                       if re.sub(r'[+\-\s]', '', v) not in pins]
            print(f'  (C) ★**8번의 핀과 값이 겹치지 않는 수** = 어느 검사기도 판정하지 않는다:'
                  f' 카운트 {len(unpin_c)}/{len(counts)} · 퍼센트 {len(unpin_p)}/{len(percents)}')
            print('      ⚠이 수치는 **값 일치 상한**이다 — 같은 값이면 다른 배치의 핀이라도 '
                  '덮인 것으로 세므로 **실제 피복은 더 낮다**.')
            for i, c in sorted(unpin_c)[:40]:
                print(f'        카운트 {i:5d}  {c}')
            if len(unpin_c) > 40:
                print(f'        … 외 {len(unpin_c)-40}건')

    print('\n⛔대응 없음이 곧 오류는 아니다 — 파생 수이거나 집계가 다를 수 있다. '
          '★그러나 **하나하나 유래를 댈 수 있어야 한다.**')
    print('★판정은 이 도구가 아니라 `run_all.sh` 8번 '
          '(`tmp/STATUS_2026-08-28/audit_thesis_numbers.py`) 이 한다.')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
