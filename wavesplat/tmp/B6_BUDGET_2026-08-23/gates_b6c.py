"""B-6c 단일출처 — **배포 현실적 렌더 대상**으로 프레임 예산을 다시 잰다.

## ★목표치를 내가 고르지 않는다 — **코드에서 읽는다**
`B-6` 이 `FRAME-BUDGET-FAIL` 을 냈고 원인은 **채점 격자 전체(YS17 3,416점)** 였다.
사후에 목표를 내가 고르면 *"통과할 때까지 목표를 바꿨다"* 가 된다.
⟹ **배포 제어 경로가 실제로 요구하는 점 수를 코드에서 읽어 쓴다.**

`scripts/d3_d17_ctrl_closedloop.py:build_query` **독스트링 축자**:
> *"for each timestep and each foil, **4 points** (x±EPS_X for slope, t±EPS_T for w)
> at the future preview position … Returns Q (**8**·n_steps, 3)"*

⟹ ★***제어기는 프레임당 8점을 요구한다*** (수중익 2개 × 유한차분 스텐실 4점).
그리고 `scripts/d3_required_horizon.py` 독스트링: *"The FF preview requirement is 0.1–0.5 s"*,
*"the controller never touches the elevation"* — 제어기가 소비하는 것은 `alpha_wave` 이고
그것은 **8점의 유한차분**으로 얻는다. **채점 격자는 평가용이지 제어용이 아니다.**

## 판정 (결과 전 확정)
**PRIMARY** = `t_upd + t_render(8점)` 이 프레임 간격(레코드 dt, 0.067–0.100 s) 안에 들어가는가:
- **12/12 레코드 통과** ⟹ ★`CONTROL-FRAME-OK`
- **9~11** ⟹ ⚠`CONTROL-FRAME-PARTIAL`
- **≤8** ⟹ ⛔`CONTROL-FRAME-FAIL`

**SECONDARY** = 14 s 창 전체의 **점유율**(duty cycle):
`(init + n_refit·t_refit + n_frames·(t_upd + t_render8)) / 14 s`, τ=10 기준.
- **< 0.50** ⟹ ✅여유 있음 · **0.50~1.00** ⟹ ⚠빠듯 · **≥1.00** ⟹ ⛔실시간 불가

★**음성 대조 필수**: 같은 배치에서 **채점 격자 전체**도 재서 `B-6` 의 실패를 재현한다.
재현되지 않으면 이 배치의 타이머가 `B-6` 와 다른 것을 재는 것이다(⟹ `G-4`).
⚠**100 Hz(0.01 s) 예산에서도 보고한다** — 제어 루프가 그만큼 빠를 수 있다.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from gates_b6 import RECORDS, WINDOWS, SEED, NOISE, TAU, N_REPEAT   # noqa: F401

N_CONTROL_PTS = 8          # ★코드 유래: d3_d17_ctrl_closedloop.build_query (8 = 2 foils x 4)
CONTROL_SRC = 'scripts/d3_d17_ctrl_closedloop.py:build_query (Q = 8*n_steps)'
WIN = 3
FAST_BUDGET = 0.01         # 100 Hz 도 함께 보고


def gates(rows):
    ok = [r for r in rows if 'error' not in r]
    out = [('G-0 런 수', len(ok) == len(RECORDS), f'{len(ok)}/{len(RECORDS)}'),
           ('G-0b 에러 0', len(rows) == len(ok), f'{len(rows)-len(ok)}'),
           ('G-1 단일 스레드', all(r.get('threads') == 1 for r in ok), ''),
           ('G-2 제어 질의점 = 8', all(r.get('n_ctrl') == N_CONTROL_PTS for r in ok), ''),
           ('G-3 반복 전건', all(len(r.get('samples_ctrl') or []) >= N_REPEAT for r in ok), '')]
    # ★G-4 음성 대조 — 채점 격자 전체는 B-6 처럼 여전히 비싸야 한다(YS17 가 dt 를 넘어야 한다)
    ys = [r for r in ok if r['record'] == 'YS17']
    out.append(('G-4 음성대조: 채점격자 전체는 여전히 예산 초과(YS17)',
                bool(ys) and ys[0]['t_grid'] + ys[0]['t_upd'] > ys[0]['dt_frame'],
                f"{ys[0]['t_grid']:.4f}s vs dt {ys[0]['dt_frame']:.3f}s" if ys else '-'))
    return out


def label(n_pass, n, duty_med, duty_max):
    if n_pass == n:
        lab = 'CONTROL-FRAME-OK'
    elif n_pass >= 9:
        lab = 'CONTROL-FRAME-PARTIAL'
    else:
        lab = 'CONTROL-FRAME-FAIL'
    d = ('여유 있음' if duty_med < 0.50 else '빠듯' if duty_med < 1.0 else '⛔실시간 불가')
    return (lab, f'제어 경로 질의({N_CONTROL_PTS}점)로 재면 {n_pass}/{n} 레코드가 프레임 예산을 '
                 f'만족한다. 14 s 창 점유율 중앙 **{duty_med:.3f}**(최악 {duty_max:.3f}) = {d}.')
