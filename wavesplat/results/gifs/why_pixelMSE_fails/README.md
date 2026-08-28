# 왜 pixel-MSE nowcast가 틀린 지표인가 — 증거 자료 (2026-07-02)

이 GIF들의 **D 패널이 빈칸(≈mean)** 인 것은 버그가 아니라(원시 아키텍처 표현버그는 별개로 수정됨,
`docs/AMORTIZED_NET_BUG_FINDING_2026-07-02.md` 참고), **pixel-MSE 최적해가 under-determined 문제에서
prior mean으로 수축**하기 때문. 즉:
- D(pixel-MSE amortized): nRMSE 0.25 = mean, corr 0 → "정확"하나 쓸모없음(빈칸)
- B(physics RELAX): nRMSE ~0.28(mean보다 나쁨), corr 0.7 → 구조·위상 보존 → forecast·외삽에 유용

**결론: pixel-MSE로 nowcast를 채점/훈련하면 mean 붕괴가 최적. 물리의 가치(구조)는 pixel-MSE로 안 보임.**
→ 그래서 (a) 논문 headline은 forecast·외삽, (b) amortized는 구조보존 목적(RELAX distillation)으로 훈련해야 함.
이 폴더는 그 근거로 보존. distillation D로 만든 정상 GIF는 `results/gifs/` 상위에 별도 저장.
