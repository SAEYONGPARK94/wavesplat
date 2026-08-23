"""B-7b 단일출처 — ★**오염된 팔을 제대로 다시 잰다** (R2 토론 변호인의 "이것 하나").

## 왜
`B-7` 의 `SHIP-TRACK` 이 `'SHIP-TRACK': {}` 로 **런타임 `DEFAULT_CONFIG` 를 스냅샷**해
**당일 배포됐다 철회된 `bank_refit_every_s=5.0`** 으로 돌았다(`refit_readback` 5.0, 36/36행).
내가 그 자리를 **`B-3` 의 `FROZEN` 행으로 대체**했지만, R2 가 정확히 지적했다:
> *"그 짝은 사전등록에 없고 **배치·바이너리를 건넌다**."*
`B-3` 은 오전에, `B-7` 은 저녁에 돌았고 그 사이 `sequential_update.py` 가 **여러 번 바뀌었다**
(배포·철회·비동기 구현). ⟹ ★**같은 바이너리·같은 배치에서 두 팔을 다시 돌린다.**

## 팔 (12 레코드 × 창 {1,3,5} = 36 짝, **둘 다 신규**)
- `PROP-B7B` : `{'real_domain_gate': False, 'init_method': 'learned'}` (`B-7` 과 동일)
- `SHIP-B7B` : ★**`{'bank_refit_every_s': None}` 을 *명시*** — 빈 dict 를 쓰지 않는다.
  (실측 게이트가 bank 경로로 보내는 것은 그대로 두고, **동결만 명시**한다.)

## 판정 (결과 전 확정) — 짝 부호검정 `fc_mean`, n=36
**PRIMARY** = `SHIP-B7B` 가 `PROP-B7B` 보다 나은 창 수:
- **≥26** ⟹ ★`SHIP-BEATS-PROP (CONFIRMATORY)` · **≥24** ⟹ ⚠명목
- **19~23** ⟹ `UNDECIDED` · **≤18** ⟹ ⛔`PROP-BEATS-SHIP`
⚠**`B-7` 의 오염된 값은 33/36 이었고, 교차배치 대체값은 23/36(p=0.066, 미결)이었다.**
**이 배치가 그 둘 중 어느 쪽인지, 혹은 제3의 값인지를 정한다.**

**SECONDARY** = 두 팔 각각의 **참 제로필드 대비**(lead 1–3 / 4–14, 문턱 24 명목).
★**재현성 대조(결과 전)**: `SHIP-B7B` 의 lead 4–14 가 `B-3` 의 `FROZEN` **15/36** 과 맞는가.
어긋나면 **바이너리 변화가 결과를 움직였다**는 뜻이고 그 자체가 보고 대상이다.

★**게이트**: `SHIP-B7B` 의 `refit_readback` 이 **전건 `None`** · `refit_calls` **전건 0** ·
`PROP-B7B` 성분 >0 · 추적 스텝 >0. 가족단위 α = 0.05/15 = 0.00333.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
B2 = os.path.abspath(os.path.join(HERE, '..', 'B2_REAL_2026-08-23'))
B3 = os.path.abspath(os.path.join(HERE, '..', 'B3_REFIT_2026-08-23'))
for p in (HERE, B2, B3):
    if p not in sys.path: sys.path.insert(0, p)
from gates_b3 import RECORDS, WINDOWS, SEED, NOISE, N_PAIRS, sign_test_one_sided, key, paired  # noqa
from gates_b7 import beats_zero, LEAD_NEAR, LEAD_FAR                                           # noqa

ARMS = {'PROP-B7B': {'real_domain_gate': False, 'init_method': 'learned'},
        'SHIP-B7B': {'bank_refit_every_s': None}}          # ★빈 dict 금지 — 명시한다
ARM_ORDER = ['PROP-B7B', 'SHIP-B7B']
NOMINAL, CONFIRM = 24, 26
FROZEN_REF_FAR = 15        # B-3 의 FROZEN lead 4–14 (재현성 대조용, 원본에서 assert)
FAMILY_ALPHA = 0.05 / 15


def gates(rows):
    ok = [r for r in rows if 'error' not in r]
    out = [('G-0 런 수', len(ok) == len(ARMS) * N_PAIRS, f'{len(ok)}/{len(ARMS)*N_PAIRS}'),
           ('G-0b 에러 0', len(rows) == len(ok), f'{len(rows)-len(ok)}')]
    sh = [r for r in ok if r['cell'] == 'SHIP-B7B']
    pr = [r for r in ok if r['cell'] == 'PROP-B7B']
    out.append(('★G-1 SHIP 되읽기가 전건 None (오염 재발 방지)',
                bool(sh) and all(r.get('refit_readback') is None for r in sh),
                str(sorted({str(r.get('refit_readback')) for r in sh}))[:40]))
    out.append(('★G-2 SHIP 재적합 0회', all((r.get('refit_calls') or 0) == 0 for r in sh), ''))
    out.append(('G-3 SHIP 성분 0개 (bank 경로)', all((r.get('n_atoms') or 0) == 0 for r in sh), ''))
    out.append(('G-4 PROP 성분 >0 · 추적 >0',
                all((r.get('n_atoms') or 0) > 0 and (r.get('track_steps') or 0) > 0 for r in pr), ''))
    out.append(('G-5 36 창 전건', len({key(r) for r in ok}) == N_PAIRS, ''))
    return out


def label(p, ship_far, repro_ok):
    w, n = p['better'], p['n']
    tail = (f" [SHIP lead 4–14 참기준 {ship_far['better']}/{ship_far['n']}"
            f" · B-3 FROZEN 재현 {'일치' if repro_ok else '⛔불일치'}]")
    if w >= CONFIRM:
        return ('SHIP-BEATS-PROP (CONFIRMATORY)',
                f'배포 실측 경로가 제안 표현을 {w}/{n} 이긴다(p={p["p_one_sided"]:.6f}).' + tail)
    if w >= NOMINAL:
        return ('SHIP-BEATS-PROP (EXPLORATORY)', f'{w}/{n} 명목 통과, 가족단위 미달.' + tail)
    if w <= n - NOMINAL:
        return ('PROP-BEATS-SHIP', f'⛔제안 표현이 {n-w}/{n} 이긴다 — 예상 밖.' + tail)
    return ('UNDECIDED', f'부호로 갈리지 않는다({w}/{n}, p={p["p_one_sided"]:.4f}).' + tail)
