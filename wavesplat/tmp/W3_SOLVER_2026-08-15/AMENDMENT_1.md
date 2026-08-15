# W3_SOLVER — AMENDMENT 1 (2026-08-15, **판정 런 이전**)

**상태**: `results.jsonl` **부재**. 읽은 판정 행 **0**. 근거는 `smoke_results.jsonl`(2 에피소드) 뿐이고
PREREG §10 이 명시했듯 **스모크는 판정 자료가 아니다**.
`PREREG.md` 자체는 **수정하지 않았다** — md5 `9f4e9c7e7bf534253b4049b7c91cca71` 재검증함.

이 개정은 **팔·문턱·표본·통계량·방향을 하나도 바꾸지 않는다.** 셋 다 운영·해석 규율이다.

---

## A-1. 게이트 **G-10** 의 판정 대상을 "유효 행 집합" 으로 명시

**동기 (스모크 실측)**: 49잡 중 1잡이 `AcceleratorError: CUDA error: out of memory` 로 실패했다
(`SHIP-FITPHI|BF4_2comp_seed20|1593`). 원인은 `init_method='auto'` 가 부르는 **학습 제안기가 GPU 를
쓰고**, 워커 8개가 RTX 4080 Super 하나를 동시에 물기 때문이다(저장소에서 **재발**하는 GPU 병목).
러너는 실패 잡을 `done` 에 넣지 않으므로 **재실행이 자동으로 재시도**하지만, 실패 행 자체는
`results.jsonl` 에 남는다. PREREG G-10 을 문자 그대로 읽으면 **일시적 OOM 하나가 배치를 죽인다**.

**개정 (동결)**:

> **G-10** 은 **유효 행 집합** 위에서 판정한다. 유효 행 = `job_id` 별 **마지막 성공 행**.
> 통과 조건은 다음 **둘 다**이다:
> 1. 계획된 `job_id` 전부가 성공 행을 **하나 이상** 가진다(누락 0),
> 2. 재시도 끝에도 성공 행이 없는 `job_id` 가 **0** 이다.
>
> 파일에 남은 **일시 실패 행 수와 그 사유별 분포는 반드시 인쇄**한다(숨기지 않는다).
> ⚠**같은 `job_id` 의 성공 행이 2개 이상이고 `avg_nRMSE` 가 서로 다르면 `NONDETERMINISM` 으로
> 게이트를 실패시킨다** — 재시도가 결과를 흔들었다면 그건 일시 오류가 아니다.

## A-2. 실행 위상 분리 (운영, 추론 영향 없음)

GPU 를 쓰는 팔과 안 쓰는 팔을 분리해 돌린다. **수치에 영향이 없다**(동일 코드·동일 시드·동일 cfg,
바꾸는 것은 동시 실행 수뿐). 순서는 PREREG §10 의 "PRIMARY 먼저" 를 지킨다.

```bash
# Phase A -- GPU 미사용 (오라클 3팔 + 상한 + 제로 + 누출 프로브 + v2_konly 짝)
WORKERS=8 python3 tmp/W3_SOLVER_2026-08-15/run_w3.py full \
  ZERO,ORA-K-CVX,ORA-K@L3,ORA-K-NOREG,ORA-K@L4,ORA-K@L3-DUP,KONLY,KONLY-FITPHI
# Phase B -- init_method='auto' (학습 제안기 = GPU)
WORKERS=3 python3 tmp/W3_SOLVER_2026-08-15/run_w3.py full SHIP,SHIP-FITPHI
# Phase C -- 실패 잡 재시도 (같은 명령 반복; 성공 잡은 스킵된다)
WORKERS=3 python3 tmp/W3_SOLVER_2026-08-15/run_w3.py full
```

## A-3. `DEGENERATE` 대조에 대한 문장 규율 (해석만, 문턱 불변)

**동기 (스모크 실측)**: 혼잡 `BF4_30comp_seed20` 에서 `ORA-K@L3` 이 **`ZERO` 보다 나빴다**
(0.2534 vs 0.2401). PREREG §4.3 은 이미 그런 팔을 `DEGENERATE` 로 표시하도록 했지만,
**대조가 퇴화했을 때 PRIMARY 통과를 어떻게 쓸지**는 안 적었다.

**개정 (동결)**:

> PRIMARY 대조 `ORA-K@L3` 이 혼잡 층에서 `DEGENERATE` 로 판정되면, P-1 통과의 정직한 문장은
> ⛔*"λ=0 이 이긴다"* 가 **아니라**
> ✅*"λ=1e-3 은 오라클 팔을 **평평한 바다를 예측하는 것보다 못하게** 만들고, λ=0 은 그 손상을 되돌린다"*
> 이다. 판정 이름(§8)은 그대로 쓰되 **`(DEGENERATE-CONTROL)` 꼬리표를 반드시 붙인다.**

⚠**이 조항은 2 에피소드 스모크의 방향을 본 뒤에 쓰였다.** 다만 이 조항이 할 수 있는 일은
**우리 주장을 약화시키는 것뿐**이고 어떤 문턱·표본·방향도 건드리지 않는다. 그래서 적어 둔다.
