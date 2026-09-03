"""★동결 판정 파일 자체의 위생 — **철회된 라벨**과 **실패 게이트를 단 채 출하된 판정**.

⛔왜 필요한가 (2026-09-03 적대적 심사가 찾았다):
  `tmp/B3_REFIT_2026-08-23/verdict_b4.json` 의 `label` 이 지금도
  `"PROPAGATION-EARNS (CONFIRMATORY)"` 인데 원장은 그것을 **철회**로 기록한다
  (기준선 `PERSIST` 가 널보다 무르다 — `ZERO < PERSIST` 36/36).
  ★그런데 `scripts/gen_paper_constants.py` 의 **입력이 바로 이 파일들**이다
  ⟹ 죽은 라벨을 남겨 두면 언젠가 인쇄된다. **파일 수준에서 탐지 가능해야 한다.**

⛔동결 파일은 **고치지 않는다**. 이 검사기는 **표식(AMENDMENT/RETRACTED 파일)** 을 요구한다.
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
TMP = os.path.join(ROOT, 'tmp')

# ★철회된 라벨 — 원장(`memory/MEMORY.md` · `reference_citation_bans.md`)에 근거가 있는 것만.
#   ⛔여기에 추가할 때는 **철회 근거를 같은 줄에** 적어라.
RETRACTED_LABELS = {
    'PROPAGATION-EARNS (CONFIRMATORY)':
        '기준선 PERSIST 가 널보다 무르다 (ZERO<PERSIST 36/36) — MEMORY.md 08-23 아크',
    'GEOM-STRATUM-SURVIVES':
        '처치 팔이 카메라 중심의 두 수평축을 뒤바꿔 받았다 — R5/AMENDMENT_1_POSTRESULT.md',
}
# 표식으로 인정하는 파일 이름 조각
MARKERS = ('AMENDMENT', 'RETRACT', 'SUPERSED')
# ★래칫 기준선. ⛔침묵시키지 말고 **의도적으로** 내려라(사유를 주석에).
#   2026-09-03 실측: 표식 없는 철회 라벨 1(`verdict_b4`) · 실패 게이트 출하 아래 목록.
# ⚠[2026-09-03] `BASELINE_UNMARKED` 1 → **0**: `B3_REFIT_2026-08-23/RETRACTED_verdict_b4.md`
#   를 같은 커밋에서 만들어 표식을 붙였다(⛔동결 판정 파일은 고치지 않았다).
BASELINE_UNMARKED = 0
# ⚠[2026-09-03] `BASELINE_FAILGATE` = **13**(첫 전수 측정값). 초판이 3 이었던 것은 근거 없는
#   추정이었다. 13 건 각각은 *"실패 게이트를 달고 `blocked_by` 가 비었다"* 이고, 그중 일부는
#   정당하다(사전 개정 스냅샷 등). ⛔한 건씩 검토해 줄이되 **줄이면 이 수를 내려라**.
BASELINE_FAILGATE = 13


def _gates(v):
    out = []
    for g in v.get('gates', []) or []:
        if isinstance(g, list) and len(g) >= 2:
            out.append((str(g[0]), bool(g[1])))
        elif isinstance(g, dict):
            # ⚠스키마가 배치마다 다르다. 첫 실행이 `name`/`pass` 판을 못 읽어 **23건을
            #   거짓 경고**했다(같은 실패 모드의 재발) ⟹ 알려진 키를 전부 받는다.
            name = g.get('id') or g.get('name') or g.get('gate') or '?'
            ok = g.get('ok')
            if ok is None:
                ok = g.get('pass')
            if ok is None:
                ok = g.get('passed')
            if ok is None:          # ⛔판정 불가면 **경고하지 않는다**(거짓 강제 금지)
                continue
            out.append((str(name), bool(ok)))
    return out


def main(argv):
    dirs = argv[1:] or sorted(d for d in glob.glob(os.path.join(TMP, '*')) if os.path.isdir(d))
    unmarked, failgate = [], []
    for d in dirs:
        marks = [os.path.basename(f) for f in glob.glob(os.path.join(d, '*'))
                 if any(m in os.path.basename(f).upper() for m in MARKERS)]
        for f in sorted(glob.glob(os.path.join(d, 'verdict*.json'))):
            try:
                v = json.load(open(f, encoding='utf-8'))
            except Exception:
                continue
            # ⛔부분문자열 대조는 **후속 배치를 오탐한다** --- 초판이
            #   `GEOM-STRATUM-SURVIVES-FIXED`(결함을 고친 배치)를 잡았다.
            #   ⟹ `배치 :: 라벨` 을 갈라 **정확 일치**를 요구한다.
            lab = str(v.get('label') or '')
            core = lab.split('::')[-1].strip()
            for bad, why in RETRACTED_LABELS.items():
                if core == bad and not marks:
                    unmarked.append((os.path.basename(d), os.path.basename(f), bad, why))
            gs = _gates(v)
            bad_g = [n for n, ok in gs if not ok]
            blocked = v.get('blocked_by') or []
            grade = str(v.get('grade') or '')
            if bad_g and not blocked and grade != 'BLOCKED':
                failgate.append((os.path.basename(d), os.path.basename(f), bad_g[:2]))

    print('── 판정 파일 위생 ────────────────────────────────────')
    for d, f, lab, why in unmarked:
        print(f'  ⛔{d}/{f}: 철회된 라벨 "{lab}" 인데 표식 파일이 없다 — {why}')
    for d, f, g in failgate:
        print(f'  ⚠{d}/{f}: 실패 게이트 {g} 를 달고 blocked_by 가 비었다')
    print(f'  표식 없는 철회 라벨 {len(unmarked)} (기준선 {BASELINE_UNMARKED}) · '
          f'실패 게이트 출하 {len(failgate)} (기준선 {BASELINE_FAILGATE})')
    if len(argv) > 1:
        print('  [측정] 디렉터리 지정 실행 --- 래칫을 적용하지 않는다.')
        return 0
    rc = 0
    for got, base, name in ((len(unmarked), BASELINE_UNMARKED, '표식 없는 철회 라벨'),
                            (len(failgate), BASELINE_FAILGATE, '실패 게이트 출하')):
        if got > base:
            print(f'  [FAIL] {name} {got} > 기준선 {base} — 새로 생겼다.'); rc = 1
        elif got < base:
            print(f'  [FAIL] {name} {got} < 기준선 {base} — **개선이 기록되지 않았다.** '
                  f'기준선을 의도적으로 내려라.'); rc = 1
    if rc == 0:
        print('  [OK  ] 두 래칫 유지')
    return rc


if __name__ == '__main__':
    sys.exit(main(sys.argv))
