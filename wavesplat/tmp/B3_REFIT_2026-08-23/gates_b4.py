"""B-4 단일출처 — **전파 모형이 밥값을 하는가**: 지속(persistence) 기준선.

## 왜 이것이 오늘 가장 중요한 질문인가
`B-3` 은 *"재적합이 동결보다 낫다"* 를 보였다. 심사자의 다음 질문은 명백하다:
> *"자료를 동화하면 좋아지는 건 모든 DA 방법이 하는 말이다. **당신 방법**이 뭘 벌었나?"*

그리고 오늘 `ZERO_REF` 결함이 드러나면서 **제로필드 기준선이 무너졌다**.
예보 과제의 **표준 기준선은 지속**이다 — *"아무것도 전파하지 않고 지금 장을 그대로 둔다."*

우리 방법의 핵심 주장은 **분산관계로 전파한다**는 것이다(`ω=√(g|k|tanh(|k|d))`).
⟹ ★***`FROZEN` 이 `PERSIST` 를 못 이기면, 전파 모형이 아무것도 벌지 못하는 것이다.***
이것은 **제로필드보다 훨씬 엄한**, 그리고 **기준값 결함에 면역인**(짝 비교) 시험이다.

## 팔
- `PERSIST` (**신규 36런**): 초기화는 `FROZEN` 과 **완전히 동일**하고, 예측만
  **모든 lead 에서 `t0` 의 장을 그대로 쓴다**(전파 0). ⟹ **전파 모형만 절제한 순수 대조.**
- `FROZEN` · `REFIT2`: `results_b3.jsonl` 에서 재사용.

## 판정 (결과 전 확정) — 짝 부호검정, n=36, 창 {1,3,5}
**PRIMARY** = `FROZEN` 이 `PERSIST` 보다 나은 창의 수(`fc_mean`):
- **≥26/36** ⟹ ★`PROPAGATION-EARNS (CONFIRMATORY)` — 분산관계 전파가 실측에서 값을 번다.
- **≥24/36** ⟹ ⚠`PROPAGATION-EARNS (EXPLORATORY)`
- **19~23** ⟹ ⛔`PROPAGATION-NULL` — **전파가 지속보다 낫다는 증거가 없다.**
- **≤18/36** ⟹ ⛔⛔`PROPAGATION-HURTS` — ★**전파가 해롭다**. 이 경우 방법의 핵심 주장이 무너진다.

**SECONDARY** = `REFIT2` vs `PERSIST` (같은 문턱). ⚠**`REFIT2` 마저 지속을 못 이기면
`B-3` 의 −24% 는 "전파가 덜 틀리게 됐다" 는 뜻이고 예보 능력의 증거가 아니다.**

★**lead 별로 반드시 분해할 것** — 지속은 **짧은 lead 에서 강하고 긴 lead 에서 무너진다**.
교차점의 위치가 곧 **"우리 방법의 유효 예보 지평"** 이다. (결과 전 기재: 교차점을 보고할 것.)
⛔**단 이 교차점은 사전등록된 양이 아니다 ⟹ 서술로만 읽는다.**
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from gates_b3 import (RECORDS, WINDOWS, SEED, NOISE, FIELD, N_PAIRS,   # noqa: F401
                      sign_test_one_sided, key, paired)

ARMS = {'PERSIST': {'bank_refit_every_s': None}}      # 설정은 FROZEN 과 같다. 다른 건 예측 시각뿐.
ARM_ORDER = ['PERSIST']
REUSED = ['FROZEN', 'REFIT2']
NOMINAL, CONFIRM = 24, 26


def gates(rows):
    ok = [r for r in rows if 'error' not in r]
    new = [r for r in ok if r['cell'] in ARMS]
    out = [('G-0 신규 런 수', len(new) == N_PAIRS, f'{len(new)}/{N_PAIRS}'),
           ('G-0b 에러 0', len(rows) == len(ok), f'{len(rows)-len(ok)}')]
    for a in REUSED:
        n = sum(1 for r in ok if r['cell'] == a)
        out.append((f'G-1 재사용 팔 {a} {N_PAIRS}행', n == N_PAIRS, f'{n}/{N_PAIRS}'))
    # ★G-2 절제가 진짜인가 — PERSIST 는 lead 마다 값이 **달라야 한다**(참값이 변하므로)
    #   그러나 예측장은 lead 에 **의존하지 않아야 한다**. 러너가 예측장 해시를 남긴다.
    ph = [r.get('persist_field_hash_uniform') for r in new]
    out.append(('G-2 PERSIST 예측장이 lead 에 불변', all(ph), f'{sum(1 for x in ph if x)}/{len(ph)}'))
    # ★G-3 lead 0 은 FROZEN 과 비트동일이어야 한다 (같은 초기화, 같은 시각)
    a = {key(r): r.get('lead0') for r in new}
    b = {key(r): r.get('lead0') for r in ok if r['cell'] == 'FROZEN'}
    ks = sorted(set(a) & set(b))
    out.append(('G-3 lead 0 이 FROZEN 과 비트동일',
                bool(ks) and all(a[k] == b[k] for k in ks),
                f'{sum(1 for k in ks if a[k] == b[k])}/{len(ks)}'))
    out.append(('G-4 재적합 0회', all(r.get('refit_calls') == 0 for r in new), ''))
    return out


def label(p_fro, p_ref):
    def one(p, nm):
        w, n = p['better'], p['n']
        if w >= CONFIRM: return f'{nm} {w}/{n} p={p["p_one_sided"]:.5f} **통과(확증)**'
        if w >= NOMINAL: return f'{nm} {w}/{n} p={p["p_one_sided"]:.5f} ⚠통과(명목)'
        if w <= n - NOMINAL: return f'{nm} {w}/{n} ⛔**역전**'
        return f'{nm} {w}/{n} 널'
    w, n = p_fro['better'], p_fro['n']
    detail = one(p_fro, 'FROZEN>PERSIST') + ' · ' + one(p_ref, 'REFIT2>PERSIST')
    if w >= CONFIRM:
        return ('PROPAGATION-EARNS (CONFIRMATORY)',
                f'분산관계 전파가 지속 기준선을 이긴다. {detail}')
    if w >= NOMINAL:
        return ('PROPAGATION-EARNS (EXPLORATORY)', f'명목 통과. {detail}')
    if w <= n - NOMINAL:
        return ('PROPAGATION-HURTS',
                f'⛔⛔전파가 지속보다 **나쁘다**({w}/{n}) — 방법의 핵심 주장이 실측에서 무너진다. {detail}')
    return ('PROPAGATION-NULL',
            f'⛔전파가 지속보다 낫다는 증거가 없다({w}/{n}). {detail}')
