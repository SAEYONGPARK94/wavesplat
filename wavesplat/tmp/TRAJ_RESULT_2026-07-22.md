# 궤적정책 완결 Step 1 결과 — **FAIL(clean): 궤적 우위 = oracle-파수 아티팩트** (2026-07-22)

> prereg `tmp/TRAJ_gate_prereg_2026-07-22.md`. fable red-team `tmp/TRAJ_redteam_fable_2026-07-22.md`.
> 스크립트 `tmp/traj_step1_netmmse.py`(net-MMSE)·`tmp/traj_step1b_relax.py`(RELAX 깨끗한 disambiguation).
> ⚠**내 n=2 smoke 예비판단(생존) 반증됨 — full n=24가 정정.** [[project_methods_landscape_2026-07-22]].

## 판정 = **FAIL(clean)** — 궤적 achieved 우위는 oracle 파수 의존
d74 achievability가 **oracle 파수(sea['kx'],ky 직접대입, 탐색0)** 를 쓴다는 걸 fable red-team이 적발 → 두 realistic
추정기로 재검(linear 바다, n=24):

| 정책 | oracle 파수 Δvs stat | **RELAX 추정 파수 Δvs stat** |
|---|---|---|
| CEM | **+0.0205 CI[+.014,+.027] 21/24** ✓ | **−0.0101 CI[−.044,+.024] 12/24 (동률)** |
| learned | +0.0182 CI[+.012,+.024] 22/24 ✓ | **−0.0391 CI[−.073,−.004] 7/24 (열위)** |
| greedy | +0.005 (동률) | −0.059 (열위) |
| straight-cross | −0.015 | −0.066 (열위) |
| stationary(기준) | 0.046 | **0.240 (최선)** |

- **oracle 파수**: 비근시 정책(CEM/learned)이 stationary 명확히 이김(+0.02, CI 0제외, 21~22/24) = d74 재현.
- **RELAX 추정 파수(깨끗·net-OOD 없음)**: **우위 소멸** — CEM 동률(CI 0포함), learned 열위, 단순 이동정책은 크게 열위.
- **net-MMSE(별건, `traj_step1_netmmse.py`)도 washout**(stationary 최선) — 단 이건 **net-OOD confound**(정책 기하
  ~115m vs net 훈련 ~20m 패치, geometry 실측) 있어 RELAX 결과가 더 깨끗한 근거.

## 기제
이동 궤적이 obs를 **~115m로 확산**(stationary 20m). oracle 파수면 넓은 개구가 재구성 도움(CRB 이득 실현).
그러나 **파수를 추정해야 하면** 넓고 성긴 커버리지가 **파수 추정을 어렵게** 만들어 개구 이득을 상쇄·초과 ⟹
achieved 재구성서 이동이 도움 안 됨. = **achievability gap**(CRB 이득이 realizable 추정기로 실현 안 됨).

## ⟹ 결론 (정직)
- **fable의 "궤적=유일 wall-mover ×2.5~4.4"는 CRB/oracle 성명.** achieved 재구성서 realistic 파수추정 하면
  **linear 바다서도 우위 소멸** = **B-2/M-3와 같은 oracle-의존 washout 패턴.** (Step 2 HOS는 무의미 — 쉬운 linear가
  이미 실패.)
- **단 스코프**: 이건 "RELAX/net-MMSE로 이동obs 파수추정이 안 됨" 때문 — **더 나은 이동-obs 파수추정(joint 시공간·
  synthetic-aperture 전용)** 이면 회복 여지 ARGUED. 하지만 as-is 표준 추정기론 실현 안 됨.
- **⚠ 미접촉**: fable의 **다른** movable axis = under-det 알고리즘 갭(고정기하서 net-MMSE/set-diffusion가 RELAX
  이김, FIM 0.71~0.93 vs RELAX 0.27~0.54)은 이 결과와 무관 = **여전히 더 방어가능한 학습 기회.**

## 프로세스 교훈 (규율)
- **내 n=2 smoke 낙관 예비판단(생존)이 full n=24서 반증** — [[feedback_direction_flip_discipline]] "예비 소표본
  단정 금지" 정확히 재확인. smoke를 방향으로 발표 안 하고 "full 확인 중"으로 헷지한 건 규율 준수, 그러나 반전 폭이 큼.
- fable red-team의 OOD confound 적발이 **net-MMSE FAIL 오해석("oracle 아티팩트")을 막고 깨끗한 RELAX 재검으로
  유도** → 그런데 RELAX도 FAIL ⟹ washout은 **진짜**(OOD 아님). 양방향 교정.
- 검증: 동결 prereg verdict + per-episode 일관 + 두 추정기 수렴. **fable 독립감사 권장(다음).**
