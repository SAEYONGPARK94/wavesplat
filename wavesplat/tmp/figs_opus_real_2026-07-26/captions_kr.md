# 실측(Bergamasco 스테레오) 요약 그림 — Discord용 한국어 캡션

작성 2026-07-26. 모든 수치는 `tmp/wp_real_2026-07-25/wp_real.json`,
`tmp/unified_2026-07-26/{hand_amp,fallback_results}.json`에서 검증됨.
프레이밍은 `docs/HANDOFF_OPUS_2026-07-26.md` §5 + 같은 날 🛑CORRECTION 준수 —
**실측에서 학습의 역할은 정확도 우위가 아니라 붕괴 보험(collapse insurance)뿐**이고,
"3/3 수리"·"평균 개선 = 일반적 이득"은 쓰지 않는다.

---

**`results/gifs/real_collapse_repair_fields.png`**
> 실측 붕괴창 2개의 실제 파고장(GT | 해석 | 학습): 해석적 추정기가 진폭 3~5배로 폭주하는 반면 학습 prior는 폭주하지 않는다 (LJ18g w3 2.93→0.96 = 붕괴선 아래, AA14 w0 2.31→1.07 = 절반이지만 여전히 ≥1).
> 나머지 27개 정상창에서는 해석적 팔이 그대로 쓰이고 더 정확하다 — 실측에서 학습의 기여는 정확도가 아니라 붕괴 보험이다.

**`results/gifs/real_perwindow_paired.png`**
> 실측 30창 전수: 배포 추정기(해석 + GT-free 진폭 가드 τ=1.5)는 30창 중 **딱 2창**만 바꾸고, 그 둘은 해석 경로가 붕괴한 창이다.
> ⚠캐비엇 3종(전부 측정치): 평균은 0.881→0.775지만 **중앙값은 불변(0.809→0.809)** = 평균 이동은 2창짜리지 일반 이득이 아님 · 오경보 0/27이나 **τ가 같은 30창에서 정해진 in-sample** · 최종적으로 붕괴선 아래로 내려간 건 3창 중 **1창뿐**.

**`results/gifs/real_amp_signal.png`**
> GT 없이 계산되는 `amp_ratio_grid`(추정장 std / 관측 elevation rms)로 붕괴를 사전 탐지 — 심한 붕괴 2건은 τ=1.5 위, 정상 27창은 전부 아래(오경보 0/27, 단 in-sample τ).
> 경미한 붕괴 AA15a_w3(amp 0.89, nRMSE 1.12)은 **탐지 실패로 미수리** — 학습 팔이면 0.79였겠지만, 숨기지 않고 그대로 그렸다. 오차 감소 ≠ 창 수리.

**`results/gifs/real_normal27_loss.png`**
> 가드 설계를 강제한 정직한 음성: 사전등록 1차 지표(정상 27창)에서 학습 prior가 **유의하게 더 나쁨** (6/27 승, median Δ −0.057, 양측 Wilcoxon p=0.0046, rank-biserial −0.61).
> 그래서 배포 시스템은 해석적 추정기를 기본으로 돌리고 학습 팔은 붕괴 보험으로만 호출한다.

**`results/gifs/real_collapse_repair.gif`**
> 최악 붕괴창 LJ18g w3 애니메이션 — 스테레오 GT | 해석 | 학습을 리드 0–20 s에 걸쳐 비교: 해석은 진폭 폭주로 계속 발산(nowcast 2.93 → 리드 20 s 7.39), 학습은 붕괴하지 않는다(전 리드 0.81–1.37).
> 가드가 발동하는 창이며, 정상 27창에서는 해석적 팔이 쓰이고 더 정확하다 — 실측 학습 기여 = 붕괴 보험.

---

### ⛔ 이 그림들로 **말하면 안 되는 것** (§5 + 07-26 CORRECTION)
- 실측 **정확도** 학습 우위 (없음 — 정상 27창에서 오히려 p=0.005로 LOSS).
- "복잡할수록 학습이 이긴다" (합성 한정, 실측 전이 안 됨).
- mean 기반 실측 승리 주장 (붕괴창 2개가 평균을 끄는 아티팩트, 중앙값 불변).
- "붕괴창 3/3 수리" (동결 기준 <1.0으로는 2/3, 배포 체인 end-to-end로는 1/3).

### 생성 스크립트
- 재실행(붕괴창 필드, wp_real.json과 비트 동일 확인): `tmp/figs_opus_real_2026-07-26/rerun_collapse_fields.py`
- 그림 4장: `tmp/figs_opus_real_2026-07-26/make_real_figs.py`
- GIF: `tmp/figs_opus_real_2026-07-26/make_real_gif.py`
