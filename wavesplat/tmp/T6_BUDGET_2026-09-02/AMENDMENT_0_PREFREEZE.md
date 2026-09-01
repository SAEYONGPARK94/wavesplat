# AMENDMENT 0 — 결과 **전** 개정 (`T6-BUDGET`)

**작성 시각의 결과 행: 0개.** 첫 러너 실행이 첫 셀에서 `AttributeError` 로 죽었고,
`results_t6_*.jsonl` 은 생성되지 않았다. 어떤 수치도 보지 않았다.

## 무엇을 고치는가

`run_t6.py` 의 `effective` 되읽기가 여섯 키를 **전부 객체 속성**으로 가정했다.
실측: `SequentialUpdater` 는 여섯 중 **넷만** 속성으로 노출한다
(`lambda_l1`·`tracking_steps`·`tracking_window`·`tracking_mode`).
`varpro_steps` 와 `learned_init_iters` 는 **`self.config` 안에만** 있다.

## 왜 그냥 `config` 에서 읽지 않는가 — 그리고 왜 그것이 약해지는가

이 저장소의 규율은 ***"`effective_config` 를 입력 dict 복사로 만들지 마라(= 의도표)"***
이고, 그 사유는 `S13b` 가 **객체 되읽기**로 `analytic_ensemble_k` 앙상블이 실제로는
런의 14%에서만 작동함을 드러낸 사건이다.

⟹ 정직한 서술: **넷은 객체 되읽기이고 둘은 `self.config` 되읽기다.**
`self.config` 는 러너가 넘긴 dict 그 자체가 아니라 **updater 가 보관한 것**이므로
러너 쪽 의도표보다는 강하지만, **객체 상태의 확인은 아니다**.
⛔`G-7` 이 그 둘에 대해 증명하는 것은 *"updater 가 그 값을 갖고 있다"* 까지이고
*"그 값이 실제로 쓰였다"* 는 **아니다**.

★대신 그 둘에 대해 **행동 증거**를 하나 추가한다: `varpro` 기록의 존재
(`initialize()` 반환의 `varpro` 키) — 이미 행에 실리고 있다. VARPRO 가 실제로
돌았는지는 그것으로 확인된다(`varpro_steps=0` 이면 `None` 이다).

## 변경 목록 (이것뿐)

1. `run_t6.py` 의 `eff` 구성: 속성 넷 + `self.config` 둘, 그리고 각 키가 어느
   출처인지 `eff_source` 로 **행에 함께 적는다**.
2. `gates_t6.py` `G-7`: 판정 값은 그대로. `eff_source` 가 기대 출처와 일치하는지
   **추가로** 확인한다(출처가 조용히 바뀌면 실패한다).
3. `FROZEN_md5.txt` 재발행.

⛔PREREG 의 설계·읽기 규칙·라벨 결정표는 **한 글자도 바꾸지 않는다.**
