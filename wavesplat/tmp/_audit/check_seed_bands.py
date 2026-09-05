"""시드 대역 충돌 검사기 — ★**아직 안 돈 형제**까지 본다.

## 왜 생겼나 (2026-09-05)
`T16` 설계 에이전트가 `SEED16 = 20261100` 을 골랐다. 그 에이전트는 형제 배치의 **동결 결과
파일**에서 시드를 뽑아 서로소를 확인했고, 그 확인은 **옳게 수행됐다**. 그런데 같은 시각
`T15` 가 `SEED15 = 20261100` 으로 동결되는 중이었고 **결과 파일이 아직 없었다**
⟹ 보이지 않았다. 실제 대역은 `T15` 20261101–228 ⊇ `T16` 20261101–160 = **60/60 충돌**.

★부류: ***"형제를 결과로만 세면, 동시에 만들어지는 형제는 안 보인다."***
⟹ 이 검사기는 **선언(소스 상수)** 과 **관측(결과 행)** 을 **둘 다** 본다.

⚠**인용도 선언으로 센다.** 다른 배치의 상수를 `SEED15 = 20261100` 꼴로 **인용만** 해도
이 검사기는 그것을 선언으로 본다. 일부러 그렇게 뒀다 — 오탐은 싸고 누락은 비싸다
(★실제로 이 충돌을 고치면서 내가 사유를 그 꼴로 적었다가 **같은 FAIL 을 재생산했다**;
저장소가 이름 붙인 **「N 에는 주석이 포함된다」** 부류의 재발이다).
⟹ 남의 상수를 인용할 때는 `=` 를 쓰지 말고 *"…의 밑변"* 처럼 적어라.

## 판정 규약
- **관측 대역** = 결과 `.jsonl` 에 실제로 실린 `seed` 값들 (⟹ 사실)
- **선언 대역** = `consts_*.py`/`run_*.py` 의 `SEED… = <정수>` 를 밑변으로, 같은 파일의
  `N_…` 상수 최대값을 폭으로 본 `[base+1, base+N]` (⟹ **추정**이며 그렇게 표시한다)
- 관측이 있으면 관측을 쓰고, 없으면 선언을 쓴다.
- ⛔밑변(`SEED… = X`)이 **같으면** 폭과 무관하게 충돌로 본다.

⛔이 검사기는 *"충돌하면 판정이 오염된다"* 고 말하지 않는다 — 블록이 다르면 오염되지 않는다.
말하는 것은 **저장소 규율(대역 서로소)의 위반**이고, `T13` 사고가 *"시드가 다르면 독립"* 이
거짓임을 이미 보였으므로 그 반대(*"시드가 같아도 괜찮다"*)도 **증명 없이 쓰지 말라**는 것이다.

사용: `python3 check_seed_bands.py [tmp 경로]`   (런 0회 · 동결 파일만 읽는다)
"""
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TMP = os.path.abspath(os.path.join(HERE, '..'))
# ⛔앵커 규칙: 산문(`.md`) 안 백틱 코드조각도 잡아야 한다 — `T16/PREREG.md:162` 의
#   `- \`SEED16 = 20261100\`, …` 이 행-앵커 정규식에 **안 잡혀서** 초판이 T16 을 통째로 놓쳤다.
SEED_RE = re.compile(r'\b(SEED[A-Z0-9_]*)\s*=\s*(\d{7,})')
N_RE = re.compile(r'^\s*(N_[A-Z0-9_]*)\s*=\s*(\d{1,5})\b', re.M)
ROW_SEED_RE = re.compile(rb'"seed"\s*:\s*(\d+)')
ROW_PAIR_RE = re.compile(rb'"block"\s*:\s*"([^"]*)".*?"seed"\s*:\s*(\d+)'
                         rb'|"seed"\s*:\s*(\d+).*?"block"\s*:\s*"([^"]*)"')
MAXLINES = 20000


def observed_by_layer(batch_dir):
    """{층: {seed: {block,…}}} — ★**배치 안 층 간** 재사용을 본다.

    `T12` 는 `SECR` 층에서 `fi` 를 1부터 다시 세어 `PRIMARY` 와 같은 밑변 위에 얹었다
    ⟹ 24개 시드가 **같은 배치 안에서 서로 다른 블록**에 붙었다. 배치 간 비교만으로는
    이것이 보이지 않는다.
    """
    out = {}
    for p in glob.glob(os.path.join(batch_dir, 'results*.jsonl')):
        for i, line in enumerate(open(p, 'rb')):
            if i >= MAXLINES:
                break
            m = ROW_PAIR_RE.search(line)
            if not m:
                continue
            lm = re.search(rb'"layer"\s*:\s*"([^"]*)"', line)
            lay = (lm.group(1).decode() if lm else os.path.basename(p))
            blk = (m.group(1) or m.group(4) or b'?').decode('utf-8', 'replace')
            v = int(m.group(2) or m.group(3))
            if v >= 10 ** 7:
                out.setdefault(lay, {}).setdefault(v, set()).add(os.path.basename(blk))
    return out


def observed(batch_dir):
    """{seed: {block, …}} — ★시드만이 아니라 **어느 블록에 붙었는지**까지 본다.

    ⛔`T12` 는 `T11` 의 `(block, seed)` 를 **일부러** 재사용한다(`G-ANCHOR` 가 비트 재현을
    요구한다). 그러므로 *"같은 시드"* 자체는 위반이 아니다 — 위반은 **같은 시드가 서로 다른
    블록에 붙는 것**이다. 초판은 min/max 구간만 봐서 그 둘을 구별하지 못했고
    **의도된 앵커 4쌍을 전부 FAIL 로 인쇄했다**(= 검사기가 「있는 것을 없다고」 한 부류).
    """
    out = {}
    for p in glob.glob(os.path.join(batch_dir, 'results*.jsonl')):
        with open(p, 'rb') as f:
            for i, line in enumerate(f):
                if i >= MAXLINES:
                    break
                m = ROW_PAIR_RE.search(line)
                if not m:
                    continue
                blk = (m.group(1) or m.group(4) or b'?').decode('utf-8', 'replace')
                v = int(m.group(2) or m.group(3))
                if v >= 10 ** 7:            # ⛔작은 추정 시드(42/43/44)는 대역이 아니다
                    out.setdefault(v, set()).add(os.path.basename(blk))
    return out


def declared(batch_dir):
    """(밑변 이름, 밑변, 추정폭) 목록. ⚠폭은 **추정**이다."""
    out = []
    for p in sorted(glob.glob(os.path.join(batch_dir, 'consts*.py'))
                    + glob.glob(os.path.join(batch_dir, 'run*.py'))
                    + glob.glob(os.path.join(batch_dir, 'PREREG.md'))):
        try:
            txt = open(p, encoding='utf-8').read()
        except Exception:                                          # noqa: BLE001
            continue
        ns = [int(v) for _, v in N_RE.findall(txt)]
        for name, val in SEED_RE.findall(txt):
            out.append((name, int(val), max(ns) if ns else None, os.path.basename(p)))
    return out


def band(base, width):
    return (base + 1, base + width) if width else (base + 1, base + 1)


def main(argv):
    # ⛔`argv[1]` 을 그냥 쓰면 `--keys` 가 **경로로 해석돼** 빈 디렉터리를 재고 `FAIL 0` 을
    #   낸다. 내가 방금 그렇게 만들었고 그 출력을 하마터면 믿을 뻔했다 —
    #   ★**옵션이 무엇을 재는지를 조용히 바꾸는** 부류(저장소의 `(x or 기본값)` 사고와 같다).
    pos = [a for a in argv[1:] if not a.startswith('-')]
    root = pos[0] if pos else TMP
    batches = {}
    for d in sorted(glob.glob(os.path.join(root, '*'))):
        if not os.path.isdir(d) or os.path.basename(d).startswith('_'):
            continue
        obs = observed(d)
        dec = declared(d)
        if not obs and not dec:
            continue
        batches[os.path.basename(d)] = (obs, dec)

    spans, base_map = {}, {}
    for name, (obs, dec) in batches.items():
        if obs:
            spans[name] = (min(obs), max(obs), 'observed', len(obs))
        elif dec:
            lo = min(band(b, w)[0] for _, b, w, _ in dec)
            hi = max(band(b, w)[1] for _, b, w, _ in dec)
            spans[name] = (lo, hi, 'declared(추정폭)', 0)
        for nm, b, w, src in dec:
            base_map.setdefault(b, []).append(f'{name}:{src}:{nm}')

    keys = []
    print('══ check_seed_bands ::', root)
    print(f'  배치 {len(spans)} · 선언 밑변 {len(base_map)}')
    fails = 0
    for b, who in sorted(base_map.items()):
        owners = sorted({w.split(':')[0] for w in who})
        if len(owners) > 1:
            fails += 1
            keys.append('BASE:' + ','.join(owners))
            print(f'  [FAIL] 같은 밑변 SEED={b} 를 **여러 배치**가 선언 → {owners}')
            print(f'         ★이것이 결과 파일만 보는 검사가 못 잡는 부류다 '
                  f'(아직 안 돈 형제). 출처 {sorted(who)}')
    names = sorted(spans)
    anchors = 0
    for name in names:
        lay = observed_by_layer(os.path.join(root, name))
        Ls = sorted(lay)
        for i, a in enumerate(Ls):
            for c in Ls[i + 1:]:
                common = set(lay[a]) & set(lay[c])
                clash = sorted(s_ for s_ in common if not (lay[a][s_] & lay[c][s_]))
                if clash:
                    # ⛔스모크 행은 **어떤 검정에도 안 들어간다** ⟹ 규율 위반이되 판정과
                    #   무관하다. 등급을 가른다(침묵시키는 것이 아니라 **분류**한다).
                    smoke = 'SMOKE' in a.upper() or 'SMOKE' in c.upper()
                    if smoke:
                        keys.append(f'WARN-INTRA:{name}:{a}|{c}')
                        print(f'  [WARN] **배치 안** {name} 층 {a} ↔ {c} — 같은 시드 다른 블록 '
                              f'{len(clash)}건 (스모크 층 — 판정에 안 들어간다)')
                        continue
                    fails += 1
                    keys.append(f'INTRA:{name}:{a}|{c}')
                    print(f'  [FAIL] **배치 안** {name} 층 {a} ↔ {c} — 같은 시드가 다른 블록에 '
                          f'{len(clash)}건 (예 {clash[0]} → {sorted(lay[a][clash[0]])} vs '
                          f'{sorted(lay[c][clash[0]])}) · ★2층이 `fi` 를 1부터 다시 셌다')
    for i, a in enumerate(names):
        for c in names[i + 1:]:
            oa, oc = batches[a][0], batches[c][0]
            common = set(oa) & set(oc)
            if not common:
                continue
            # ⛔`!=` 가 아니라 **교집합 공집합**이 위반이다 — 한쪽이 그 시드를 두 층
            #   (예: PRIMARY + SMOKE)에서 쓰면 집합이 달라도 **블록은 겹친다**.
            #   초판은 `T14 G-ANCHOR-T11` 이 `T11` **스모크** 행을 앵커로 쓴 것을
            #   위반으로 인쇄했다(= 「있는 것을 없다고」 하는 부류, 두 번째 재발).
            clash = sorted(s_ for s_ in common if not (oa[s_] & oc[s_]))
            same = len(common) - len(clash)
            if clash:
                fails += 1
                keys.append(f'CROSS:{a}|{c}')
                print(f'  [FAIL] {a} ∩ {c} — 같은 시드가 **다른 블록**에 붙었다 '
                      f'{len(clash)}건 예 {clash[:3]}')
            if same:
                anchors += 1
                print(f'  [OK  ] {a} ∩ {c} — 시드 {same}개 공유하나 **블록도 같다** '
                      f'(의도된 앵커 재현)')
    if not fails:
        print('  [OK  ] 밑변 중복 0 · 같은시드-다른블록 0'
              + (f' · 의도된 앵커 공유 {anchors}쌍' if anchors else ''))
    print('  ── 대역 원장 ' + '─' * 40)
    for n in names:
        lo, hi, kind, cnt = spans[n]
        print(f'   {n:34s} {lo}–{hi}  {kind}' + (f' · {cnt}값' if cnt else ''))
    hi_all = max((h for _, h, _, _ in spans.values()), default=0)
    print(f'  ⟹ ★다음 배치가 안전하게 쓸 밑변: **{((hi_all // 100) + 1) * 100}** 이상')
    print(f'  요약: FAIL {fails}   (⛔충돌은 판정 오염이 아니라 **규율 위반**이다 — 독스트링)')
    if '--keys' in argv:
        print('KEYS ' + ' '.join(sorted(k for k in keys if not k.startswith('WARN-'))))
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
