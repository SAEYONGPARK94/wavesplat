"""B-6 단일출처 — **실시간 예산**: `bank_refit_every_s=2` 를 배포해도 되는가.

## 왜
`B-3`/`B-3b`/`B-3c` 가 정확도 이득을 확증했고 **합성 회귀 위험 0 은 시험으로 고정**됐다.
남은 유일한 차단 사유가 **실시간 예산**이다. `RESULT_B3.md` 의 "1.5 → 7.0 s" 는
**장면 생성·채점 코드가 섞인 벽시계**라 예산 주장에 못 쓴다(RV-3 ⑥이 지적).

## 무엇을 잰다 — **프로세스 내부 타이머, 단일 스레드**
⚠저장소는 *"지연을 프로세스 간 비교로 쟀다"* 로 한 번 크게 물렸다(08-05, 헤드라인 사망).
⟹ **같은 프로세스 안에서 `perf_counter` 로 구간만** 잰다:
- `t_init` : `initialize()` 전체 (1회)
- `t_refit`: `_bank_fit_window` **한 번**의 시간 (재적합 단가)
- `t_upd`  : `update()` 중 **재적합이 아닌** 프레임 한 번
- `t_pred` : `predict_field()` 한 번 (질의 격자 전체)
스레드는 `torch.set_num_threads(1)` + `OMP/MKL=1` 로 고정한다.

## 판정 (결과 전 확정) — 주기 τ=2 s 기준
**PRIMARY** = `t_refit` 의 **12 레코드 중앙**이 주기 τ 에서 차지하는 비율 `t_refit / τ`:
- **< 0.25** ⟹ ★`REALTIME-OK` (여유 4배 이상)
- **0.25 ~ 0.50** ⟹ ✅`REALTIME-OK-TIGHT`
- **0.50 ~ 1.00** ⟹ ⚠`MARGINAL` (한 코어로는 되나 여유 없음)
- **≥ 1.00** ⟹ ⛔`REALTIME-FAIL` (재적합이 주기보다 오래 걸린다)

**SECONDARY** = **프레임 예산**: `t_upd + t_pred` 가 프레임 간격(레코드 dt, 0.067–0.100 s)을 넘는가.
⛔넘으면 재적합과 무관하게 **추적 자체가 실시간이 아니다** — 그 경우 PRIMARY 는 무의미하다.

★**최악 레코드도 함께 보고한다**(중앙만으로 배포 결정을 하지 않는다).
⚠**GPU 는 쓰지 않는다**(`device='cpu'`) — 배포 가정과 같다.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
B2 = os.path.abspath(os.path.join(HERE, '..', 'B2_REAL_2026-08-23'))
B3 = os.path.abspath(os.path.join(HERE, '..', 'B3_REFIT_2026-08-23'))
for p in (HERE, B2, B3):
    if p not in sys.path: sys.path.insert(0, p)
from gates_b3 import RECORDS, WINDOWS, SEED, NOISE   # noqa: F401

TAU = 2.0
WIN = 3
N_REPEAT = 3        # 레코드마다 3회 반복, 중앙을 취한다(1회 측정 금지)


def gates(rows):
    ok = [r for r in rows if 'error' not in r]
    out = [('G-0 런 수', len(ok) == len(RECORDS), f'{len(ok)}/{len(RECORDS)}'),
           ('G-0b 에러 0', len(rows) == len(ok), f'{len(rows)-len(ok)}')]
    out.append(('G-1 단일 스레드 되읽기', all(r.get('threads') == 1 for r in ok), ''))
    out.append(('G-2 재적합이 실제로 발화', all((r.get('n_refit') or 0) > 0 for r in ok),
                str(sorted({r.get('n_refit') for r in ok}))[:50]))
    out.append(('G-3 반복 3회 전건', all(len(r.get('refit_samples') or []) >= N_REPEAT for r in ok), ''))
    # ★음성 대조: 재적합 단가가 갱신 단가보다 유의하게 커야 한다(아니면 타이머가 엉뚱한 걸 쟀다)
    out.append(('G-4 t_refit > t_upd (타이머 유효성)',
                all(r['t_refit'] > r['t_upd'] for r in ok),
                f"{sum(1 for r in ok if r['t_refit'] > r['t_upd'])}/{len(ok)}"))
    out.append(('G-5 CPU 전용', all(r.get('device') == 'cpu' for r in ok), ''))
    return out


def label(ratio_med, ratio_max, frame_ok, frame_worst):
    if not frame_ok:
        return ('FRAME-BUDGET-FAIL',
                f'⛔프레임 예산부터 넘는다(최악 {frame_worst:.3f} s > dt) ⟹ 재적합과 무관하게 '
                '추적 자체가 실시간이 아니다. PRIMARY 는 무의미하다.')
    if ratio_med < 0.25:
        lab = 'REALTIME-OK'
    elif ratio_med < 0.50:
        lab = 'REALTIME-OK-TIGHT'
    elif ratio_med < 1.00:
        lab = 'MARGINAL'
    else:
        lab = 'REALTIME-FAIL'
    return (lab, f'재적합 단가 / 주기 {TAU} s = **중앙 {ratio_med:.3f}** (최악 {ratio_max:.3f}). '
                 f'프레임 예산은 통과(최악 {frame_worst:.4f} s).')
