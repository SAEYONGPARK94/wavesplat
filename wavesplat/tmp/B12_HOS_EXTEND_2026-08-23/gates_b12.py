"""B-12 단일출처 — `B-11` 을 **n=16 → 48** 로 확장 (판정 규칙은 `B-11` 을 그대로 상속).

## ⛔먼저 이 배치가 **무엇이 아닌지** 적는다 (결과 전)
오늘 `B-8b` 에서 나는 *"표본을 더 키워 문턱을 넘기면 그것이 p-hacking 이다"* 라 적고
**세 번째 확장을 거부**했다. ⟹ **왜 여기서는 늘리는가?**
- `B-8b` 는 **미결(`SPLIT-STILL-UNDECIDED`)을 통과로 바꾸려는** 확장이었다 ⟹ 거부.
- ★**`B-12` 는 이미 통과한 결과의 *크기를 더 정확히* 재려는 확장**이다
  (`B-11`: 14/16, p=0.00209, **가족단위 통과**).
⟹ ★**따라서 이 배치는 판정을 바꾸는 것이 목적이 아니다.** 그것을 못 박기 위해
**아래에 "판정이 뒤집히면 어떻게 읽을지" 를 결과 전에 적는다.**

## 표본
`HOSGATE` 코퍼스의 **혼잡 HOS** 에피소드 전체 = 기존 시드 **1581–1584** + 신규 **1633–1640**
(`B-12` 생성기가 같은 `gen_hos` 파라미터로 만든다) = `BF4/BF5 × 15/30comp × 12 시드` = **48**.
⚠**신규 시드는 학습 코퍼스 밖**이고 참값은 **HOS M=3** 이다 ⟹ 단서 ①②가 계속 제거된 상태다.

## 팔·프로토콜
`B-11` 과 **완전히 동일**(`SHIP-NOW` = 현재 배포 · `SPEC25` = `S16` 정의 그대로 import ·
`ZERO` = 기준선 행의 `zero_nRMSE`). 추정기/필터 시드 median-of-3.

## 판정 (결과 전 확정, n=48 로 **정확 계산**)
**PRIMARY** = `SHIP-NOW` < `SPEC25` 인 에피소드 수:
- **≥34/48**(p=0.00276 ≤ 가족단위 α=0.05/17=0.00294) ⟹ ★`HEADLINE-SURVIVES (CONFIRMATORY, n=48)`
- **≥31/48**(p=0.0297) ⟹ ⚠명목 통과
- **25~30** ⟹ `UNDECIDED-AT-48`
- **≤24** ⟹ ⛔`HEADLINE-WEAKENS-AT-48`

★**결과 전 해석 규칙**(이것이 이 배치를 p-hacking 과 가르는 장치다):
- 통과하면 → ***`B-11` 의 결론은 그대로이고 크기 추정만 좁아진다.*** ⛔**"더 강해졌다" 로 쓰지 마라.**
- **미달하면** → ⛔***`B-11`(n=16)의 통과를 표본 우연으로 의심해야 한다*** — 그 경우
  **작은 표본 쪽을 버리고 큰 표본 쪽을 채택한다.** 이 문장을 지금 적어 둔다.

**SECONDARY(음성 대조, 필수)** = `SPEC25` < `ZERO`. **≤30/48 이면 기준선이 무기술** ⟹ ⛔PRIMARY 무효.
★**부차**: **신규 시드만(32개)** 따로도 낸다 — *"기존 16개가 결과를 끌고 가는가"* 를 본다.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
B11 = os.path.abspath(os.path.join(HERE, '..', 'B11_HOS_BASELINE_2026-08-23'))
for p in (HERE, B11):
    if p not in sys.path: sys.path.insert(0, p)
from gates_b11 import (HOS, S16, EST_SEEDS, FILTER_SEEDS, SPEC,          # noqa: F401
                       sign_test_one_sided, med_by_episode, paired)

OLD_SEEDS = (1581, 1582, 1583, 1584)
NEW_SEEDS = (1633, 1634, 1635, 1636, 1637, 1638, 1639, 1640)
ALL_SEEDS = OLD_SEEDS + NEW_SEEDS
N_EP = 48
NOMINAL, CONFIRM, WEAK = 31, 34, 24      # /48, 결과 전 정확 계산
NEG_MIN = 31                             # SPEC25 < ZERO 최소 (같은 문턱)
FAMILY_ALPHA = 0.05 / 17


def gates(rows, ship, spec, zero):
    ok = [r for r in rows if 'error' not in r]
    out = [('G-0 에러 0', len(rows) == len(ok), f'{len(rows)-len(ok)}'),
           ('G-1 SHIP 48 에피소드', len(ship) == N_EP, f'{len(ship)}'),
           ('G-1b SPEC25 48 에피소드', len(spec) == N_EP, f'{len(spec)}'),
           ('★G-2 두 팔 에피소드 집합 완전 일치', set(ship) == set(spec),
            f'교집합 {len(set(ship) & set(spec))}'),
           ('G-3 ZERO 48 에피소드', len(zero) == N_EP, f'{len(zero)}')]
    sh = [r for r in ok if r['arm'] == 'SHIP-NOW']
    sp = [r for r in ok if r['arm'] == 'SPEC25']
    out.append(('G-4 SHIP 시드 3개 전건', len(sh) == 3 * N_EP, f'{len(sh)}/{3*N_EP}'))
    out.append(('G-4b SPEC25 시드 3개 전건', len(sp) == 3 * N_EP, f'{len(sp)}/{3*N_EP}'))
    out.append(('★G-5 SPEC25 되읽기(loc 25 · ens 600 · jonswap)',
                bool(sp) and all(abs(r.get('loc_radius', -1) - 25.0) < 1e-9
                                 and r.get('n_ensemble') == 600
                                 and r.get('background') == 'jonswap' for r in sp), ''))
    out.append(('★G-6 SHIP 이 현재 배포값 되읽기',
                bool(sh) and all(r.get('varpro_steps') == 400
                                 and r.get('learned_init_iters') == 0 for r in sh), ''))
    # ★G-7 신규 시드가 실제로 들어왔나 (확장이 확장인지)
    got = {int(e.split('seed')[1].split('_')[0]) for e in ship}
    out.append(('★G-7 신규 시드 8개 전건 포함', set(NEW_SEEDS) <= got,
                f'{sorted(got - set(OLD_SEEDS))}'))
    return out


def label(prim, neg, new_only):
    if neg['better'] < NEG_MIN:
        return ('BASELINE-NO-SKILL-HERE',
                f'⛔기준선이 무기술이다({neg["better"]}/{neg["n"]}) ⟹ PRIMARY 무효.')
    w, n = prim['better'], prim['n']
    tail = (f" [상대중앙 {prim['rel_median_pct']:.2f}% · p={prim['p_one_sided']:.6f}"
            f" · 신규 시드만 {new_only['better']}/{new_only['n']}"
            f" · 음성대조 {neg['better']}/{neg['n']}]")
    if w >= CONFIRM:
        return ('HEADLINE-SURVIVES (CONFIRMATORY, n=48)',
                f'★n=48 에서도 {w}/{n} 로 통과한다. ⛔**"더 강해졌다" 로 쓰지 마라** — '
                '`B-11` 의 결론은 그대로이고 **크기 추정만 좁아진다**.' + tail)
    if w >= NOMINAL:
        return ('HEADLINE-SURVIVES (EXPLORATORY, n=48)', f'{w}/{n} 명목 통과, 가족단위 미달.' + tail)
    if w <= WEAK:
        return ('HEADLINE-WEAKENS-AT-48',
                f'⛔{w}/{n} — ★**`B-11`(n=16)의 통과를 표본 우연으로 의심해야 한다.** '
                '사전등록대로 **큰 표본 쪽을 채택한다.**' + tail)
    return ('UNDECIDED-AT-48',
            f'{w}/{n} — 부호로 갈리지 않는다. ⚠`B-11` 의 통과가 **표본에 민감**하다는 뜻이다.' + tail)
