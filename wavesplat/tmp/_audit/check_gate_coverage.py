"""★사전등록이 적은 게이트가 **실제로 구현됐는가**.

⛔이 검사기가 없어서 `R1b` 가 `PREREG §6` 에 적은 `G-FREEZE`·`G-BLOCK` 을 구현하지 않고도
`게이트 13/13` 을 인쇄했다 — **사전등록 게이트 수와 우연히 같아서** 누락이 보이지 않았다.
(2026-09-02 적대적 감사 M1. 저장소가 `M-4` 에서 문제 삼은 *"배치마다 세는 법이 다르다"* 부류.)

하는 일: 배치의 `PREREG.md` 에서 `G-...` 이름을 뽑고, 동결 `verdict_*.json` 의
`gates` 이름과 대조해 **사전등록에만 있는 것**을 보고한다.
⚠반대 방향(구현에만 있는 것)은 **경고**로만 낸다 — 강화는 허용되기 때문이다.
"""
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
TMP = os.path.join(ROOT, 'tmp')
# ★래칫 — 기지 결함 수. ⛔침묵시키지 말고 **의도적으로** 내려라(사유를 주석에).
#   2026-09-03 기준선: 아래 실행이 낸 값. 개선되면 이 수를 내린다.
# 2026-09-03 실측 기준선 = **17**(12 배치). 내역은 실행 출력이 낸다.
# ⚠일부는 이름 변형일 수 있다(예: `G-COV` vs `G-COVG`) — 이 검사기는 **부재**를 잡지
#   **약함**을 잡지 않는다. 줄어들면 이 수를 **의도적으로** 내리고 사유를 여기 적어라.
BASELINE = 17

_NAME = re.compile(r'`?(G-[A-Za-z0-9][A-Za-z0-9_\-]*)`?')


def prereg_gates(path):
    """`PREREG.md` 의 게이트 **표**에서만 이름을 뽑는다(산문 언급은 세지 않는다)."""
    out = []
    for line in open(path, encoding='utf-8'):
        if not line.lstrip().startswith('|'):
            continue
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        if not cells:
            continue
        m = _NAME.search(cells[0])
        if m:
            out.append(m.group(1))
    return out


def verdict_gates(paths):
    """★배치의 **모든** verdict 파일에서 이름을 모은다.

    ⛔첫 실행이 `sorted(...)[0]` 만 읽어 `T5_ENS` 의 보조 verdict(게이트 2개)를 보고
    *"14개 미구현"* 이라 보고했다 --- **검사기가 거짓을 강제할 뻔했다**(오늘 그 부류 세 번째).
    반환: (이름집합, 게이트 키를 가진 파일이 하나라도 있었는가)
    """
    names, seen_any = set(), False
    for path in paths:
        try:
            v = json.load(open(path, encoding='utf-8'))
        except Exception:
            continue
        if 'gates' not in v:
            continue
        seen_any = True
        for g in v['gates']:
            m = _NAME.search(g[0] if isinstance(g, list) else str(g))
            if m:
                names.add(m.group(1))
    return names, seen_any


def main(argv):
    dirs = argv[1:] or sorted(
        d for d in glob.glob(os.path.join(TMP, '*')) if os.path.isdir(d))
    rows, skipped, missing_total = [], [], 0
    for d in dirs:
        pre = os.path.join(d, 'PREREG.md')
        vs = sorted(glob.glob(os.path.join(d, 'verdict*.json')))
        if not os.path.exists(pre) or not vs:
            continue
        pg = sorted(set(prereg_gates(pre)))     # ⛔표에 두 번 적힌 이름을 두 번 세지 않는다
        if not pg:
            continue
        vg, has_gates = verdict_gates(vs)
        if not has_gates:
            skipped.append(os.path.basename(d))   # 옛 스키마 --- ⛔"미구현" 으로 세지 않는다
            continue
        missing = [g for g in pg if g not in vg]
        extra = sorted(g for g in vg if g not in pg)
        missing_total += len(missing)
        rows.append((os.path.basename(d), len(pg), len(vg), missing, extra))

    print('── 게이트 구현 대조 ──────────────────────────────────')
    bad = [r for r in rows if r[3]]
    for name, np_, nv, miss, extra in rows:
        if miss:
            print(f'  ⛔{name}: 사전등록 {np_} · 구현 {nv} · **미구현** {miss}'
                  + (f' · 추가 {extra}' if extra else ''))
    print(f'  배치 {len(rows)} · 미구현 게이트를 가진 배치 {len(bad)} · 미구현 총 {missing_total}')
    if skipped:
        print(f'  ⚠대조 불가(verdict 에 gates 키 없음) {len(skipped)}: {skipped[:6]}')
    if BASELINE is None:
        print('  [측정] 기준선 미설정 — 이 값을 BASELINE 에 적어라(⛔침묵이 아니라 래칫).')
        return 0
    if missing_total > BASELINE:
        print(f'  [FAIL] {missing_total} > 기준선 {BASELINE} — 새 미구현 게이트가 생겼다.')
        return 1
    if missing_total < BASELINE:
        print(f'  [FAIL] {missing_total} < 기준선 {BASELINE} — **개선이 기록되지 않았다.** '
              f'기준선을 의도적으로 내려라.')
        return 1
    print(f'  [OK  ] 미구현 {missing_total} = 기준선 (래칫 유지)')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
