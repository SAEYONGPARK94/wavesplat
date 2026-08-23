#!/usr/bin/env python3
"""논문 수치 대조기 — ★**손복사 결함 부류를 원고에 적용한다**.

`tmp/_audit/find_handcopied_refs.py` 는 **게이트 파일**의 손복사를 잡는다. 그런데
**가장 비싼 손복사는 원고에 있다** — 배치가 재분석되거나 수정(AMENDMENT)되면
`verdict_*.json` 은 갱신되는데 `thesis_main.tex` 의 숫자는 조용히 낡는다.
이 저장소는 정확히 그 부류로 여러 번 대가를 치렀다(⛔인용금지 목록의 상당수가 그것이다).

하는 일:
  1. `thesis_main.tex`(와 지정 파일)에서 **`w/n` 꼴 카운트 주장**을 전부 뽑는다.
  2. `tmp/**/verdict_*.json` 을 전부 읽어 **(better, n) 쌍의 사전**을 만든다.
  3. 원고의 각 주장이 **어느 판정 파일에서 나오는지** 찾는다.
  4. ⚠**어디에도 없는 카운트**를 표시한다 = *"이 수는 자료에 대응이 없다"*.

⛔⛔**이 검사가 통과해도 "논문 수치가 옳다" 는 뜻이 아니다.** 이것은 **존재 검사**다 —
*"이 (승/전체) 쌍이 어떤 배치 기록에 실재하는가"*. **어느 배치를 인용해야 옳은지는 못 본다**
(같은 `16/16` 이 여러 배치에 있으면 아무거나 맞춰진다). 잡는 것은 **최악의 경우** 하나다:
***저장소 어디에도 대응이 없는 수*** = 손으로 만들었거나 낡은 수.

⛔**이 도구는 판정하지 않는다.** 대응이 없다고 해서 반드시 틀린 것은 아니다
(집계 방식이 다르거나, 파생 수이거나, 원고가 옳고 판정 파일이 낡았을 수 있다).
**사람이 볼 목록**을 만든다.

사용:  python3 tmp/_audit/check_thesis_numbers.py [tex 파일 ...]
"""
import glob, json, os, re, sys

COUNT = re.compile(r'(?<![\d./])(\d{1,4})\s*/\s*(\d{1,4})(?![\d/])')
# 논문에 흔한 비-카운트 분수 (버전·날짜·비율 등)를 거른다
SKIP_N = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 1000}


def collect_verdicts(root='tmp'):
    """⚠**구조화된 키만 보면 놓친다** — 판정 파일 상당수가 카운트를 `label_text` 문자열
    안에만 갖고 있고, 오래된 배치는 `RESULT*.md` 에만 있다. ⟹ **세 곳 모두** 훑는다."""
    pairs = {}          # (better, n) -> [출처, ...]
    # (c) 판정 파일과 결과 문서의 **본문 텍스트**에 적힌 w/n
    for f in sorted(glob.glob(os.path.join(root, '**', 'verdict*.json'), recursive=True)
                    + glob.glob(os.path.join(root, '**', 'RESULT*.md'), recursive=True)
                    + glob.glob(os.path.join(root, '**', 'AMENDMENT*.md'), recursive=True)):
        try:
            txt = open(f, encoding='utf-8').read()
        except Exception:
            continue
        for m in COUNT.finditer(txt):
            w, n = int(m.group(1)), int(m.group(2))
            if n not in SKIP_N and w <= n:
                pairs.setdefault((w, n), []).append(os.path.relpath(f) + '(본문)')
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
                for k, val in node.items():
                    if isinstance(val, (dict, list)):
                        stack.append((f'{src}:{k}', val))
            elif isinstance(node, list):
                for it in node:
                    if isinstance(it, (dict, list)):
                        stack.append((src, it))
    return pairs


def main(argv):
    texs = argv[1:] or sorted(glob.glob('thesis/*.tex') + glob.glob('paper/*.tex'))
    pairs = collect_verdicts()
    print(f'판정 파일에서 모은 (승/전체) 쌍: {len(pairs)}개')
    miss = hit = 0
    for tex in texs:
        if not os.path.exists(tex):
            continue
        rows = []
        for i, ln in enumerate(open(tex, encoding='utf-8').read().splitlines(), 1):
            if ln.lstrip().startswith('%'):
                continue
            for m in COUNT.finditer(ln):
                w, n = int(m.group(1)), int(m.group(2))
                if n in SKIP_N or w > n:
                    continue
                src = pairs.get((w, n))
                rows.append((i, w, n, src))
        if not rows:
            continue
        print(f'\n=== {tex}')
        for i, w, n, src in rows:
            if src:
                hit += 1
                src = sorted(src, key=lambda x: ('(본문)' in x, x))
                print(f'  {i:5d}  {w:>4d}/{n:<4d} ✔ {src[0]}' +
                      (f'  (+{len(src)-1})' if len(src) > 1 else ''))
            else:
                miss += 1
                print(f'  {i:5d}  {w:>4d}/{n:<4d} ⚠**대응하는 판정 파일이 없다**')
    print(f'\n대응 있음 {hit}건 · ⚠대응 없음 {miss}건')
    print('⛔대응 없음이 곧 오류는 아니다 — 파생 수이거나 집계가 다를 수 있다. '
          '★그러나 **하나하나 유래를 댈 수 있어야 한다.**')
    return 1 if miss else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
