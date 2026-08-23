"""B-6b 단일출처 — **렌더 예산의 일반형**: 점당 비용은 얼마인가.

## 왜
`B-6` 이 `FRAME-BUDGET-FAIL` 을 냈고 원인은 **렌더 하나**였다(`YS17` 0.498 s, 질의점 3416).
나는 사전등록에서 **채점 격자 전체**를 예산에 넣었는데, 배포에서 매 프레임 렌더할 점 수는
**응용이 정하는 값**이다. ⟹ 임의의 목표치를 새로 고르는 대신 ★**점당 비용 곡선**을 잰다.
그러면 *"예산이 B 초면 점 몇 개까지 되는가"* 를 **누구나 계산**할 수 있다.

## ⛔먼저 결과 전에 못 박을 사실 하나 (런 0회, 코드 연역)
`B-3b` 의 주기 축은 **이미 완결됐다**. 이 하네스의 lead 는 **0–14 s** 이고 재적합은
`frame.t - last_fit_t >= τ` 에서 발화하므로 **τ ≥ 15 는 발화 0회 = `FROZEN` 과 동일**이다
(실측 `refit_calls`: τ=1 → 14회, 2 → 7, 5 → 2, 10 → **1**).
⟹ ★***τ ∈ {1,2,5,10} 이 이 지평의 전 범위이고, τ=20 을 잴 필요가 없다.***

## 무엇을 잰다
`bank_mmse` 의 렌더(`_bank_predict_grid`)를 **질의점 수 N 을 바꿔가며** 잰다:
N ∈ {16, 64, 256, 1024, 4096, 16384}, 레코드 3개(작은/중간/큰 박스), **각 5회 반복 중앙**,
**단일 스레드 · CPU**. 앙상블 k 는 배포값 그대로.

## 판정 (결과 전 확정)
**PRIMARY** = 점당 비용의 선형성. `t(N) = a + b·N` 을 최소제곱으로 적합하고 **R² 를 본다**:
- **R² ≥ 0.99** ⟹ ★`RENDER-LINEAR` — **점당 비용 `b` 를 예산 규칙으로 발표할 수 있다.**
- **0.90 ~ 0.99** ⟹ ⚠`RENDER-MOSTLY-LINEAR`(구간을 명시해 쓴다)
- **< 0.90** ⟹ ⛔`RENDER-NONLINEAR`(단일 계수로 요약 금지)

**SECONDARY** = 0.1 s 프레임에서 감당 가능한 점 수 `N* = (0.1 − t_upd) / b`.
★**세 레코드에서 `b` 가 2배 이상 갈리면** 단일 값으로 발표하지 말고 **범위로** 적는다(결과 전 확정).
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
RECORDS = ['BS13a', 'AA14', 'YS17']      # 작은(16×16 m) · 중간(68×84) · 큰(220×240)
NS = [16, 64, 256, 1024, 4096, 16384]
N_REPEAT = 5
WIN, SEED, NOISE = 3, 42, 0.02
FRAME_BUDGET = 0.1
SPREAD_FACTOR = 2.0


def fit(ns, ts):
    import numpy as np
    A = np.column_stack([np.ones(len(ns)), np.asarray(ns, float)])
    coef, *_ = np.linalg.lstsq(A, np.asarray(ts, float), rcond=None)
    pred = A @ coef
    ss_res = float(((np.asarray(ts) - pred) ** 2).sum())
    ss_tot = float(((np.asarray(ts) - np.mean(ts)) ** 2).sum())
    return float(coef[0]), float(coef[1]), (1 - ss_res / ss_tot if ss_tot > 0 else float('nan'))


def gates(rows):
    ok = [r for r in rows if 'error' not in r]
    exp = len(RECORDS) * len(NS)
    out = [('G-0 런 수', len(ok) == exp, f'{len(ok)}/{exp}'),
           ('G-0b 에러 0', len(rows) == len(ok), f'{len(rows)-len(ok)}'),
           ('G-1 단일 스레드', all(r.get('threads') == 1 for r in ok), ''),
           ('G-2 반복 5회 전건', all(len(r.get('samples') or []) >= N_REPEAT for r in ok), ''),
           ('G-3 N 되읽기 일치', all(r['n_pts'] == r['n_req'] for r in ok), ''),
           # ★음성 대조: N 이 커지면 시간이 커져야 한다(아니면 렌더가 N 에 의존하지 않는다 = 측정 무효)
           ('G-4 시간이 N 에 단조 증가',
            all(all(x['t'] <= y['t'] * 1.5 for x, y in zip(
                sorted([r for r in ok if r['record'] == rec], key=lambda z: z['n_req']),
                sorted([r for r in ok if r['record'] == rec], key=lambda z: z['n_req'])[1:]))
                for rec in RECORDS), '')]
    return out


def label(r2, bs):
    lab = ('RENDER-LINEAR' if r2 >= 0.99 else
           'RENDER-MOSTLY-LINEAR' if r2 >= 0.90 else 'RENDER-NONLINEAR')
    spread = max(bs) / min(bs) if min(bs) > 0 else float('inf')
    how = ('단일 값' if spread < SPREAD_FACTOR else f'**범위**(레코드 간 {spread:.1f}배)')
    return (lab, f'R²={r2:.4f}. 점당 비용은 {how} 로 발표한다: '
                 f'{min(bs)*1e6:.2f} – {max(bs)*1e6:.2f} µs/점.')
