"""B-9b 단일출처 — ★**배포 후보 구성을 그 구성으로 잰다** (오늘 배운 것의 직접 적용).

## 왜 이 배치가 존재하는가
오늘 배포(`bank_refit_every_s=5.0`)를 **같은 날 철회**했다. 사유 한 줄:
> ***내가 근거로 든 예산 판정(`B-6c` `CONTROL-FRAME-OK`)이 배포된 구성을 덮지 않았다*** —
> 그것은 **재적합이 0회인 구성**에서 나온 판정이었다.
그리고 내가 스스로 처방을 적었다: **"배포 커밋에는 배포-후 구성으로 재실행한 게이트 출력을 첨부하라."**
⟹ ★**이 배치가 그 처방의 첫 적용이다.** 그리고 그 사이에 차단 사유(동기 정지)를
**비동기 재적합**(`bank_refit_async`, 2026-08-23, 기본 OFF)으로 구현했다.

## 팔 (12 레코드 × 창 {1,3,5} = 36 짝)
- `ASYNC5` (**신규**): `bank_refit_every_s=5.0` **+ `bank_refit_async=True`**
- `SYNC5`  (**신규, 음성 대조**): 같은 τ, **동기** — *"프레임이 실제로 깨지는가"* 를 확인한다
- 재사용: `FROZEN` (`../B3_REFIT_2026-08-23/results_b3.jsonl`)

## 판정 (결과 전 확정)
**PRIMARY (예산)** = `ASYNC5` 의 **최악 `update()` 프레임**이 프레임 간격(dt, 0.067–0.100 s) 안인가:
- **12/12 레코드** ⟹ ★`ASYNC-FRAME-OK` — **동기 정지가 제거됐다**
- 9~11 ⟹ ⚠`ASYNC-FRAME-PARTIAL` · ≤8 ⟹ ⛔`ASYNC-FRAME-FAIL`
⚠**음성 대조 필수(`G-3`)**: `SYNC5` 의 최악 프레임은 **dt 를 넘어야 한다.**
넘지 않으면 이 배치는 아무것도 증명하지 못한다(문제가 애초에 없었다는 뜻).

★**B-9b 의 PRIMARY 는 정확도다**(예산은 `B-9` 가 12/12 로 이미 통과시켰다).
**PRIMARY (정확도)** = `ASYNC5` 가 `FROZEN` 을 이기는 창 수(`fc_mean`), n=36:
- **≥26/36** ⟹ ★`ASYNC-KEEPS-GAIN (CONFIRMATORY)` — 비동기가 **정확도를 잃지 않는다**
- **≥24** ⟹ ⚠명목 · **19~23** ⟹ `UNDECIDED` · **≤18** ⟹ ⛔`ASYNC-LOSES-GAIN`
⚠비동기는 **적용이 한 프레임 이상 늦고 진행 중 kick 을 건너뛴다** ⟹ **동기보다 나쁠 수 있다.**
그 크기를 `ASYNC5` vs `SYNC5` 짝으로 함께 잰다(**결과 전 기재**, 서술 전용).

★**추가 게이트 `G-6`(결과 전)**: 페이싱이 실제로 걸렸는지 — 런 벽시계가 **≥ 12 s** 여야 한다
(14 s 지평 - 여유). 아니면 페이싱이 안 걸린 것이고 이 배치는 `B-9` 의 반복일 뿐이다.
⚠**`skipped` 가 늘어날 것으로 예상한다**(적합 1.7 s > 주기 5 s 안에 두 번 들어갈 수 있다) —
그 수를 **결과 전에 예상으로 적어 둔다**: kick 2회 중 **0~1회 skip**.

★**개입 계수기(가지 단위)**: `ASYNC5` 에서 `_bank_async_kicks>0` **그리고** `_bank_async_applied>0`
이어야 한다. 0 이면 판정은 ⛔`ASYNC-NO-INTERVENTION`(배경 적합이 안 돌았다).
⚠**단일 스레드 고정 안 함** — 비동기가 목적이므로 배경 스레드가 있어야 한다.
그래서 ⛔**이 배치의 절대 시간을 `B-6` 과 직접 비교하지 마라**(스레드 수가 다르다).
비교는 **같은 배치 안의 `ASYNC5` vs `SYNC5`** 로만 한다.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
B2 = os.path.abspath(os.path.join(HERE, '..', 'B2_REAL_2026-08-23'))
B3 = os.path.abspath(os.path.join(HERE, '..', 'B3_REFIT_2026-08-23'))
for p in (HERE, B2, B3):
    if p not in sys.path: sys.path.insert(0, p)
from gates_b3 import RECORDS, WINDOWS, SEED, NOISE, N_PAIRS, sign_test_one_sided, key, paired  # noqa

# ★B-9b: **실시간 페이싱**을 켠다 — lead 1 s 마다 벽시계 1 s.
# `B-9` 가 이 하네스의 재생 속도를 **실시간의 494배**로 실측했고, 그래서 배경 적합(1.7 s)이
# **채점이 끝난 뒤에** 착지해 정확도 절반이 측정 불가였다(상대중앙 +0.00% 가 그 증거).
# 페이싱을 켜면 적합이 **약 2 프레임 뒤에 착지**한다 = **실제 배포 조건**.
ARMS = {'ASYNC5': {'bank_refit_every_s': 5.0, 'bank_refit_async': True},
        'SYNC5':  {'bank_refit_every_s': 5.0, 'bank_refit_async': False}}
REALTIME_PACING = True      # ★이 배치의 유일한 변경점
ARM_ORDER = ['ASYNC5', 'SYNC5']
REUSED = ['FROZEN']
NOMINAL, CONFIRM = 24, 26


def gates(rows):
    ok = [r for r in rows if 'error' not in r]
    new = [r for r in ok if r['cell'] in ARMS]
    a = [r for r in ok if r['cell'] == 'ASYNC5']
    sy = [r for r in ok if r['cell'] == 'SYNC5']
    out = [('G-0 신규 런 수', len(new) == len(ARMS) * N_PAIRS, f'{len(new)}/{len(ARMS)*N_PAIRS}'),
           ('G-0b 에러 0', len(rows) == len(ok), f'{len(rows)-len(ok)}')]
    out.append(('G-1 재사용 FROZEN 36행',
                sum(1 for r in ok if r['cell'] == 'FROZEN') == N_PAIRS, ''))
    out.append(('★G-2 배경 적합 발화·적용',
                bool(a) and all((r.get('async_kicks') or 0) > 0
                                and (r.get('async_applied') or 0) > 0 for r in a),
                f"kicks {sorted({r.get('async_kicks') for r in a})} "
                f"applied {sorted({r.get('async_applied') for r in a})}"))
    # ★G-3 음성 대조 — 동기 팔은 프레임 예산을 **깨야** 한다
    out.append(('★G-3 음성대조: SYNC5 최악 프레임 > dt',
                bool(sy) and all(r['worst_update_s'] > r['dt_frame'] for r in sy),
                f"{sum(1 for r in sy if r['worst_update_s'] > r['dt_frame'])}/{len(sy)}"))
    out.append(('G-4 되읽기 일치',
                all(r.get('async_readback') == ARMS[r['cell']]['bank_refit_async'] for r in new), ''))
    out.append(('G-5 36 창 전건', len({key(r) for r in new}) == N_PAIRS, ''))
    out.append(('★G-6 실시간 페이싱이 실제로 걸렸나 (벽시계 ≥12 s)',
                all(r['wall_s'] >= 12.0 for r in new),
                f"중앙 {sorted(round(r['wall_s'],1) for r in new)[len(new)//2] if new else 0} s"))
    return out


def label(n_ok, n, acc, pair_as, neg_ok):
    if not neg_ok:
        return ('VACUOUS-NEGATIVE-CONTROL',
                '⛔동기 팔이 프레임 예산을 깨지 않았다 ⟹ 이 배치는 아무것도 증명하지 못한다.')
    if n_ok == n:
        lab = 'ASYNC-FRAME-OK'
    elif n_ok >= 9:
        lab = 'ASYNC-FRAME-PARTIAL'
    else:
        lab = 'ASYNC-FRAME-FAIL'
    w = acc['better']
    acc_lab = ('KEEPS-GAIN (CONFIRMATORY)' if w >= CONFIRM else
               'KEEPS-GAIN (EXPLORATORY)' if w >= NOMINAL else
               'LOSES-GAIN' if w <= n - NOMINAL else 'UNDECIDED')
    return (f'{lab} / {acc_lab}',
            f'예산: 비동기 최악 프레임이 dt 안 = **{n_ok}/{n}**(동기는 0/{n}). '
            f'정확도: `ASYNC5` < `FROZEN` **{w}/{acc["n"]}** {acc["rel_median_pct"]:+.2f}% '
            f'p={acc["p_one_sided"]:.6f}. 비동기 vs 동기 짝 {pair_as["better"]}/{pair_as["n"]} '
            f'{pair_as["rel_median_pct"]:+.2f}%.')
