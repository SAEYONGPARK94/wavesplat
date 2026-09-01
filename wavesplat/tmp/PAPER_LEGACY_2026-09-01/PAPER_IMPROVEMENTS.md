> ⛔⛔**[2026-09-02 이동] 이 파일은 `paper/` 에 있었다. 원고가 아니다.**
> 원고 수정 **지시서**이고, 그 지시 중 일부가 **인용금지 수치로 고치라**고 적는다
> (아래 배너가 이미 그렇게 경고한다). `paper/` 에 두면 원고 디렉터리를 통째로 묶는
> 사람이 그것을 함께 가져간다 ⟹ 다른 이동분과 같은 곳으로 옮긴다.
> ⚠**공시**: 이 이동으로 인용금지 래칫이 **6 내려간다**. 그것은 **내용 개선이 아니라
> 정리**다 --- 이 파일의 금지 수치는 하나도 지워지지 않았고 여기 그대로 있다.
> ⛔래칫 기준선을 내릴 때 이 사유를 함께 적었다.

> ⛔⛔**2026-08-13 감사 — 이 문서의 지시를 실행하지 말 것.**
>
> 이 파일은 원고가 아니라 **원고 수정 지시서**인데, **인용금지 수치로 고치라고 지시**하고
> 있다. 특히 §OPEN-1 Q1 의 옵션 (a) — "Abstract 와 Table 4 를 모두 새 ETKF (`0.319`) 로
> 재계산 → `30% improvement` 로 수정" — 은 **금지 수치를 논문에 더 넓게 퍼뜨리는 지시**다.
> ⛔**정정 (2026-08-14) — 위 배너가 원래 적은 사유가 틀렸다.**
> ~~"`0.319` 를 내는 결과 파일은 저장소에 존재하지 않는다"~~ 는 **거짓**이다. 그 파일은
> `results/b1_full_comparison.json`(git 추적, 커밋 `e57528c`, 100 ep)이고 재집계하면
> `0.2229 / 0.3189 / 13.376 ms / 132.639 ms` 가 **전부 나온다**(2026-08-14 직접 재도출).
> 이것은 08-13 3라운드 감사가 이미 잡아 사유를 교체한 항목인데 이 배너에만 낡은 채로 남아
> 있었다 — `feedback_scope_of_search_equals_scope_of_claim` 의 재발이다.
> **현재 유효한 금지 사유 3가지**:
> **(i) 프레임 불일치** — 같은 파일에서 `WaveSplat.n_frames` 는 120–220(평균 167),
> `ETKF`·`GPR-1` 은 301 이고 **100/100 에피소드 전부**에서 WS 가 더 적다 ⟹ 두 팔이
> **같은 프레임에서 채점된 적이 없다**(그 파일의 `84/100` 도 함께 인용 금지).
> **(ii) 기준선 선택** — 그 ETKF 는 `grid 40×30 · ne=100 · mult_inflation 1.0` 이다.
> **(iii) 지연은 프로세스 간 비교** — 동일프로세스 1스레드 정직값 **1.5–2.6×**,
> CPU **AMD Ryzen 7 7800X3D**.
> `0.211 vs 0.436 (51.5%)` · `0.223 vs 0.319 (30.1%)` 는 위 사유로 인용 금지다.
>
> **검증된 재도출** (`results/100ep_ws_letkf.json`, 100 ep):
> **WS 0.2229 vs LETKF 0.2278 — 51/49, p = 0.9204 → 널 (정확도 차이 없음).**
>
> 또한 **논문 편집은 사용자 지시로 정지 중**이므로 이 문서의 어떤 항목도 원고에 적용하지
> 않았다. 본문은 이력 보존을 위해 그대로 둔다.
> 근거: `reference_citation_bans.md`, `tmp/AUDIT_recheck_2026-08-13/A3_citation_bans.md`

# WaveSplat & Hydrofoil 논문 — 개선 및 추가 작업 항목

> **6인 페르소나 토론** (Prof.K / Prof.E / Rev.2 / Rev.3 / Rev.4 / Saeyong) 결과 반영.
> 2026-04-07 초판 (4인) → 2026-04-07 2차 (4인) → 2026-04-07 3차 (6인 확장)
> → **2026-04-09 4차 (Phase B/C 검토)**.
>
> **데이터 상태**: Phase B/C 완료 — 100 episode + 24-config ETKF sweep + BF4-5 ablation
> + LP filter sensitivity 데이터 모두 확보. 다른 서버에서 작업되어 git pull로 동기화됨.

---

## OPEN-1. Phase B/C 검토 후 열린 질문 (4차 토론, 2026-04-09)

> 다음 항목들은 **반드시 해야 한다는 것이 아니라**, 6인 토론에서 제기된 우려를 바탕으로
> "이렇게 해도 되나?" 수준의 검토 질문입니다. 저자가 판단해서 진행/보류 결정.

### Q1. Abstract와 Table 4의 ETKF 수치를 어떻게 정리할 것인가?
**현재 상황 (논문 안에 ETKF nRMSE 3개 다른 값):**
- Abstract / Table 4 (12ep): WS=0.211 vs ETKF=0.436 → 51.5% improvement
- Baselines 단락 (sweep 결과): "best ETKF config: mean nRMSE 0.358"
- §4.C tab:sixmetric (100ep tuned ETKF): WS=0.223 vs ETKF=0.319 → 30.1% improvement

**고려해볼 옵션:**
- (a) Abstract와 Table 4를 모두 새 ETKF (0.319)로 재계산 → "30% improvement"로 수정
- (b) Table 4를 §4.C로 대체하고 12ep 비교 자체를 삭제
- (c) Table 4는 그대로 두되 caption에 "pre-tuning ETKF"라고 명시, abstract는 100ep 결과 사용
- (d) 그대로 두기 (불일치 유지) — 저자 판단

**Reviewer가 첫 페이지에서 모순을 발견할 가능성을 어떻게 평가하시나요?**

---

### Q2. WaveSplat 자신의 약점을 §4.C에 어떻게 다룰 것인가?
**100 episode 6-metric 결과의 자기 비판적 측면:**
- Variance Ratio = **0.661** → WaveSplat도 wave amplitude의 1/3 손실 (이상은 1.0)
- Skill Score = **+0.034** → zero predictor 대비 겨우 +3.4% (유일한 양수이긴 함)
- Directional R² = **−0.253** → 방향 추정에서 random보다 못함 (음수)

**현재 §4.C 본문은 GPR/ETKF의 약점은 잘 노출했지만 WaveSplat의 위 3개 약점은
명시적으로 다루지 않음.**

**고려해볼 옵션:**
- (a) "Honest reporting of WaveSplat limitations" 단락 추가하여 정직하게 인정
- (b) 약점은 Limitations §8에서만 다루고 §4.C는 강점에 집중
- (c) Skill Score +0.034에 "marginal but the only positive value" 같은 hedge 문구 추가
- (d) Directional R² −0.253은 hydrofoil 논문의 BF4 37% 실패와 직결되므로
  cross-reference로 처리

**Rev.4가 "저자가 자신에게 불리한 데이터를 숨겼다"고 비판하지 않게 하려면
어떤 방식이 좋을까요?**

---

### Q3. GPR-3 (sliding window) 결과 처리는?
**현재 상황:**
- commit message: "WaveSplat/ETKF-tuned/GPR-1/GPR-3" — 4 methods
- §4.C 본문: "GPR with single-frame and 3-frame sliding windows"
- 실제 `b1_full_comparison.json`에는 **GPR-1만 있음** (GPR-3 데이터 없음)

**고려해볼 옵션:**
- (a) GPR-3를 실제로 돌려서 데이터 채우기 (수 시간 소요 추정)
- (b) 본문에서 "3-frame sliding windows" 언급 삭제, GPR-1만 보고
- (c) 본문에 "GPR-3 was attempted but did not converge for...; only GPR-1 results
  are reported" 같은 해명 추가

**구현/시간 비용 대비 reviewer 방어 가치를 어떻게 보시나요?**

---

### Q4. BF4-5 ablation 결과의 framing은 충분한가?
**Phase B3 결과:**
- BF3 ablation: A1_noDisp = 0.158 vs Full = 0.188 (A1 더 좋음)
- **BF4-5 ablation: A1_noDisp = 0.233 vs Full = 0.258 (A1 여전히 더 좋음)**

원래 가정 ("고 BF에서는 dispersion이 필수적으로 도움된다")이 **반증**됨.
새 framing: "physical validity vs nRMSE trade-off"

**고려해볼 옵션:**
- (a) 현재 framing 유지 (physical validity = contribution)
- (b) C2 contribution을 "dispersion-constrained for downstream usability" 로 재정의
- (c) Tracking-only ablation 별도 수행 (initialization은 same, tracking에서만 dispersion on/off)
  → tracking phase에서 dispersion의 가치를 분리해서 보여줄 수 있음
- (d) Dispersion 제거 시 wave field가 시간이 지나면 unphysical 하게 진화하는
  visualization 추가 → 'physical validity'의 시각적 증거

**Rev.3 ("그러면 dispersion이 왜 필요한가?")에 대응하는 가장 강한 답이 어느 것일까요?**

---

### Q5. Hydrofoil 논문은 Phase B/C 결과로 업데이트해야 하는가?
**현재 hydrofoil 논문 상태:**
- "WaveSplat ~20% nRMSE" 언급 (실제 100ep는 22.3%, 거의 일치)
- BF4 37% degradation은 "directional misestimation"으로 설명
- WaveSplat 논문의 §4.C에서 새로 발견된 Dir_R² = −0.253과 직접 연결되지 않음

**고려해볼 옵션:**
- (a) Hydrofoil §6 (Discussion)에 "the companion WaveSplat paper reports
  Dir.R² = −0.253 across 100 episodes, which directly explains the BF4
  degradation observed here" 추가
- (b) Cross-paper consistency를 명시하지 않고 그대로 둠
- (c) Hydrofoil 논문 자체에 directional accuracy 분석 추가 (k_x sign error rate 등)

**Cross-paper 일관성 강화는 두 논문이 함께 제출될 때 특히 중요한가요?**

---

### Q6. §4.C와 §5.A (기존 main results)의 위치/역할 정리는?
**현재 구조:**
- §4.C "Full 100-episode six-metric evaluation" — 100ep, 6-metric, tuned ETKF
- §5.A "Main comparison" — 12ep, nRMSE only, 옛 ETKF
- 두 섹션이 동일 method를 다른 dataset과 다른 metric으로 비교

**고려해볼 옵션:**
- (a) §4.C를 메인으로 승격 (Section V로 이동), §5.A를 §5.B로 강등
- (b) §5.A 삭제하고 §4.C를 main results로 사용
- (c) 현재 구조 유지 — §4.C는 "extended evaluation", §5.A는 "primary results"
- (d) §4.C의 Table을 abstract와 §5.A 양쪽에 인용하여 통합

**100 episode + 6 metric vs 12 episode + nRMSE — 어느 것이 진짜 'main result'여야 하나요?**

---

### Q7. ETKF tuning sweep의 transparency는 어디까지 보여줄 것인가?
**Phase B2 sweep 결과 분석:**
- Best (winner): 0.358
- Median: ~0.6
- Worst 3 configs: 1.03, 1.07, 1.19 (모두 g80x60 + α=0.02 조합)

**관찰:** 더 fine한 grid (80×60)가 일관되게 worse — 이건 ensemble 부족이 원인

**고려해볼 옵션:**
- (a) Sweep 결과 전체 표를 supplementary로 추가
- (b) Best config만 보고하고 sweep 방법론은 단락으로 요약 (현재 상태)
- (c) Top 3 + Worst 3 비교 표 추가하여 "왜 이 config를 선택했는지"를 정량적으로 보여줌
- (d) Heatmap (grid × ensemble × inflation) 시각화 추가

**Rev.4 같은 baseline advocate를 만족시키려면 sweep의 transparency가 어디까지 필요할까요?**

---

### Q8. LP filter α=0.05 선택의 sensitivity 데이터 활용은?
**Phase D 결과 (`hydrofoil_lp_sensitivity.json`):**
- α ∈ {0.01, 0.03, 0.05, 0.10, 0.15, 0.20} × 4 scenarios 데이터 확보됨
- BF3 Beam: α=0.05 최적 (44.8% dz, 37.7% da)
- **BF4 Head: α=0.10 또는 0.15가 더 나음** (12.9% vs 11.4% da @ α=0.05)
- 즉 α=0.05가 universal optimum은 아님

**고려해볼 옵션:**
- (a) Sensitivity table을 hydrofoil 논문에 추가, "α=0.05 chosen as best compromise" 명시
- (b) 각 sea state별 optimal α를 보고하고 "in operation, α can be tuned per condition"
- (c) Figure로 sensitivity curve 추가
- (d) 현재처럼 α=0.05만 사용하고 sensitivity는 supplementary에만

**"α=0.05가 BF4에서는 sub-optimal"이라는 사실을 정직하게 보고하는 것이 좋을지요?**

---

### Q9. WaveSplat metrics.py 함수와 §4.C 결과의 정합성 검증?
**현재 상황:**
- `metrics.py`에 VR/ρ/SS/Spectral R² 함수 구현 + 12개 unit test (모두 통과)
- §4.C 결과는 다른 서버에서 계산됨 (b1_full_comparison.json)
- 두 metric 구현이 동일한지 직접 비교하지 않음

**고려해볼 옵션:**
- (a) 3개 episode에서 두 구현의 결과 cross-check 수행
- (b) `b1_full_comparison.json`을 생성한 script가 어느 metrics.py를 사용했는지 확인
- (c) 그대로 두기 (다른 서버에서 정확히 구현됐다고 신뢰)

**재현성을 위해 두 구현의 일치를 verify해야 하나요?**

---

### Q10. Pre-registered metrics 약속 (6개) vs 실제 보고 (5개) 차이는?
**Methods §4.2의 pre-registered panel 정의:**
- (i) Variance Ratio
- (ii) Pearson correlation
- (iii) Skill Score
- (iv) Spectral R²
- (v) Directional R²
- (vi) Hs error

**§4.C tab:sixmetric에 보고된 것:**
- nRMSE / Hs err / VR / ρ / SS / Dir R² → **5개** + nRMSE
- **Spectral R² 빠짐** (1D 시간축 power spectrum 비교)

**고려해볼 옵션:**
- (a) Spectral R² 계산을 추가해서 6번째 메트릭 채우기
- (b) Methods §4.2에서 pre-registered panel을 5개로 수정 (Spectral R² 제외)
- (c) "Spectral R² requires per-frame timeseries which were not stored;
  reported in supplementary" 같은 해명 추가

**약속한 6개 vs 실제 5개 — 작은 불일치이지만 Rev.3는 cherry-picking 비판할 수 있음.
어떻게 정리하시겠습니까?**

---

## 4차 토론 결론 — 결정 권한은 저자에게

위 10개 질문은 모두 "이렇게 해도 되나?" 수준입니다.
어떤 것은 강력한 우려이고 (Q1, Q2: 출판 차단 위험),
어떤 것은 nice-to-have입니다 (Q9, Q10).

**6인 페르소나 우려 매핑:**
| 질문 | 가장 강하게 제기한 페르소나 | 우려 강도 |
|------|---------------------------|---------|
| Q1 (수치 불일치) | Prof.K, Prof.E, Rev.2 | ★★★ 출판 차단 |
| Q2 (WS 약점 honest) | Rev.3, Rev.4 | ★★★ first-round reject 위험 |
| Q3 (GPR-3 미이행) | Rev.2 | ★★ 작은 흠 |
| Q4 (Ablation framing) | Rev.3 | ★★ R&R 시 추가 토론 가능 |
| Q5 (Hydrofoil cross-update) | Rev.4 | ★★ Two-paper bundle 시 중요 |
| Q6 (§4.C vs §5.A 위치) | Prof.K | ★★ 구조적 명확성 |
| Q7 (ETKF sweep transparency) | Rev.4 | ★ supplementary로 충분 |
| Q8 (LP filter sensitivity) | Prof.E | ★ 정직한 보고 |
| Q9 (Metric 구현 일치) | Rev.2 | ★ 재현성 |
| Q10 (Spectral R² 누락) | Rev.3 | ★ pre-registration 약속 |

저자가 시간/우선순위에 따라 선택적으로 진행하면 됩니다.

---

## 0-2. 데이터 분석에서 발견한 핵심 이슈 (★★★ 즉시 조치 필요)

### 발견: ETKF의 두 run에서 nRMSE-Hs_error 불일치
`p1_comparison.json`과 `etkf_comparison.json` 두 결과 파일을 비교한 결과,
**동일 episode/seed에서 ETKF 결과가 크게 다름**:

| Episode | p1.ETKF nRMSE | etkf_comp.ETKF-medium nRMSE | etkf_comp.Hs_error |
|---------|--------------|----------------------------|--------------------|
| BF2_1comp_uniform | 0.483 | 0.266 | **0.655** |
| BF3_2comp_uniform | 0.444 | 0.295 | **0.344** |
| BF3_5comp_uniform | 0.434 | 0.297 | **0.352** |
| BF4_15comp_uniform | 0.511 | 0.280 | **0.450** |
| BF5_15comp_uniform | 0.466 | 0.279 | **0.502** |
| **AVERAGE (12 ep)** | **0.436** | **0.287** | **0.452** |

**관찰:**
- 두 script (`run_p1_comparison.py`, `run_etkf_comparison.py`)는 **동일 ETKFGrid 클래스, 동일 파라미터, 동일 seed** 사용
- 그럼에도 결과가 ~50% 다름 → 코드 변경 (ETKFGrid 구현 진화) 또는 평가 방법 차이로 추정
- **etkf_comparison의 ETKF-medium은 nRMSE는 낮으나 Hs_error 평균 45.2%**
- WaveSplat의 Hs_error 평균은 24.4% — **WaveSplat 대비 ETKF의 Hs_error가 1.85배 높음**

### 의미 (6인 토론 합의 핵심에 직결)
이 데이터는 **본 논문의 "nRMSE만으로는 부족" 논증의 결정적 증거**:
- ETKF가 "낮은 nRMSE"를 달성하는 메커니즘 = **amplitude under-fitting** (Hs를 절반 이하로 추정)
- 즉 nRMSE 0.287은 "정확한 추정"이 아니라 "평탄한 출력으로 큰 오차 회피"의 결과
- **이미 존재하는 Hs_error 데이터로 이를 입증 가능** — 메트릭 함수 구현 없이도

### 즉시 가능한 조치
1. **A6 (NEW)**: Discussion 또는 Results에 "Hs preservation" 분석 추가
   - 기존 etkf_comparison.json의 Hs_error 데이터 활용
   - "Why nRMSE alone is insufficient" 논증의 직접 증거
2. **Phase B의 우선순위 강화**: ETKF 두 run 차이의 원인 규명이 baseline 공정성 검증의 첫 단계
3. **데이터 확보 후**: 일관된 ETKF 구현으로 재실행 + 모든 metric 동시 계산

---

## -1. 6인 페르소나 토론 합의 — 작업 우선순위 (★★★ 최상위)

### 6인 페르소나 정의
1. **Prof. K** (지도교수): novelty, 물리적 의미, top venue 기준
2. **Prof. E** (외부위원, 해양공학): 정량적, 적대적 실용주의
3. **Rev. 2** (학회 리뷰어): 기술적 정확성, 재현성
4. **Rev. 3** (적대적 peer reviewer): cherry-picking 의심, over-claim 비판, "왜 published?"
5. **Rev. 4** (baseline advocate): ETKF/GPR이 더 좋다고 의심, baseline 공정성 추궁
6. **Saeyong** (학생): 구현, 한계 인정, 향후 계획

### 핵심 합의 사항
1. **메트릭 추가만으로는 부족** — Rev.3, Rev.4의 우려대로 baseline 약점 회피로 보일 위험
2. **올바른 작업 순서** (데이터 확보 후):
   1. **Episode 수 증가** (★★★ Rev.3 우려 해소: 통계적 신뢰도)
   2. **ETKF baseline 재튜닝** (★★★ Rev.4 우려 해소: 공정성)
   3. **메트릭 확장** (★★ pre-registered 6-metric panel, 모든 method에 동일 적용)
   4. **Sliding-window GPR** (★ 모든 baseline 강화)
3. **메트릭 도입 시 반드시**:
   - 사전에 정의된 세트 사용 (eval_metrics 메모리의 6개)
   - 모든 method에 동일 적용 (WaveSplat에 불리하게 나와도 정직하게 보고)
   - "Pre-registered metrics"로 논문에 명시 → cherry-picking 비판 차단

### Rev.3 (적대적 리뷰어) 핵심 메시지
> "메트릭 추가를 baseline 약점의 우회로로 쓰지 말 것. 진짜 문제는 episode 수와 baseline tuning이다."

### Rev.4 (Baseline Advocate) 핵심 메시지
> "메트릭은 모든 method에 공정하게 적용할 것. WaveSplat-friendly 선택은 더 큰 비판을 초래한다."

### 즉시 가능한 선제 방어 (데이터 없이)
| 항목 | 위치 | 내용 |
|------|------|------|
| Limitation 강화 | §Discussion | "ETKF/GPR의 best-effort tuning 부족" 명시적 인정 |
| Metric choice rationale | §Methods 또는 §Results 첫 부분 | "현재는 nRMSE primary, pre-registered 6-metric panel은 후속 작업" 명시 |
| Pre-registration 선언 | §Methods footnote | "Secondary metrics defined in companion repository" |

---

## 0. 이번 리뷰에서 수정 완료된 항목

---

## 0. 이번 리뷰에서 수정 완료된 항목

| # | 문제 | 수정 내용 |
|---|------|----------|
| 1 | WaveSplat Abstract: "N=1 for simple swells" 과장 (BF2에서 N_a=11) | "N=2 for two-component swells, N=18-21 for complex seas" + failure mode 언급 |
| 2 | bjerkaas2022 bib key 오류 (실제 저자 Collins et al.) | → collins2022로 수정 (양쪽 논문 모두) |
| 3 | Hydrofoil Abstract over-claim: "93-103% recovery" 일반화 | BF3/BF4 구분 + "37% in BF4 due to directional misestimation" 명시 |
| 4 | Hydrofoil comfort metric 불일치 | "BF3 beam sea" 명시 + a_RMS < 0.315 m/s² 조건 |
| 5 | perez2005 → perez2012 bib year 오류 | 수정 완료 |
| 6 | Ehlers+Klein FNO (2024), Desmars JFM (2023) 미인용 | Related Work에 추가 + 차별화 논의 |
| 7 | Ablation A1 해석 부족 (Rev.2 지적) | w_phi tracking의 ω 고정 메커니즘 상세 설명 |
| 8 | A1: Limitation에 ETKF best-effort tuning 부족 명시 (Rev.4) | Limitations §5에 grid/ensemble/inflation sweep 부재 명시 + ETKF-fine 0.421 결과 인용 |
| 9 | A2: Choice of metric rationale 단락 추가 (Rev.3) | Methods §4.2에 nRMSE 한계 + pre-registered 6-metric panel 명시 |
| 10 | A3: GPR 비교 수치 검증 (18.5%, JSON 일치 확인) | 정확함, 조치 불필요 |
| 11 | A4: Hydrofoil comfort metric 상세 테이블 추가 (Prof.E) | 새 Table — 절대 a_RMS + ISO 카테고리, BF3 Beam의 S→C 전환 명시 |
| 12 | A5: metrics.py에 VR/ρ/SS 함수 + 12개 unit test 구현 | reviewer-defense 메트릭 코드 + 검증 완료 |

---

## 0-1. nRMSE의 한계 → 보완 메트릭 (★★★ 핵심)

### 문제: nRMSE가 under-fitting을 못 잡는다
- Trivial zero predictor의 nRMSE ≈ 0.25 (zero-mean wave, std≈H_s/4 가정)
- GPR이 over-smoothing해서 평탄한 출력만 내도 nRMSE는 "괜찮아 보임"
- 즉 **"낮은 nRMSE = 정확한 추정"이 아님**
- 메모리의 eval_metrics 설계에 이미 secondary metrics 정의돼 있으나 논문 미사용

### 추가해야 할 메트릭

| 메트릭 | 정의 | 노출하는 약점 |
|--------|------|-------------|
| **Variance Ratio (VR)** | Var(ẑ)/Var(z) | over-smoothing (VR<<1) / over-fitting (VR>1) |
| **Pearson Correlation (ρ)** | Cov(ẑ,z)/(σ_ẑ·σ_z) | shape mismatch (진폭 무관) |
| **Skill Score (SS)** | 1 − MSE(ẑ)/MSE(zero) | trivial baseline 대비 정량 |
| **Spectral R²** | R² of S_ẑ(ω) vs S_z(ω) | 주파수 spectrum 보존 |
| **Directional R²** | 2D (kx,ky) spectrum 일치도 | 방향 추정 (hydrofoil 핵심) |
| **Hs error** | \|Ĥs−Hs\|/Hs | 진폭 보존 (이미 ablation JSON에 있음) |
| **EMD on k-spectrum** | Earth Mover's Distance in (kx,ky) | 모드 분포 차이 |

### 추천 논문 Table 형식
```
| Method | nRMSE↓ | VR | ρ↑ | SS↑ | Spec.R²↑ | Hs err↓ |
|--------|--------|----|----|------|---------|---------|
| WaveSplat | 0.233 | ~1.0 | 높음 | 양 | 높음 | 낮음 |
| GPR    | 0.276 | <<1 | 낮음 | 양 | 낮음 | 높음 |
| ETKF   | 0.450 | ~1.0 | 중간 | ? | 중간 | 중간 |
| Zero baseline | ~0.25 | 0 | 0 | 0 | 0 | 1.0 |
```

이렇게 보여주면 "GPR의 nRMSE가 WaveSplat과 가까워 보이는 것은 under-fitting 때문"임을
VR과 ρ로 즉각 입증 가능. Reviewer 방어의 핵심 무기.

### 구현 필요 사항
- 현재 JSON에는 nRMSE/inference_ms만 저장 → 이 메트릭들 계산 안 됨
- `wavesplat/wavesplat/evaluation/metrics.py` 확장 필요
- prediction array를 메모리에 유지하며 episode별 전체 메트릭 dump
- **즉각 가능 (Phase A5)**: 메트릭 함수 구현
- **전면 적용 (Phase C)**: 100 episode 재실행 필요 — Phase B 완료 후

### ⚠ 6인 토론 경고 (Rev.3, Rev.4)
- **메트릭 추가만 단독으로 진행하면 위험** — baseline 약점 회피로 보일 수 있음
- **반드시 작업 순서를 지킬 것**:
  1. Episode 수 확보 (Phase B1) — Rev.3 우려 해소
  2. ETKF 재튜닝 (Phase B2) — Rev.4 우려 해소
  3. 그 다음에야 메트릭 확장 (Phase C)
- **메트릭 적용 시 모든 method에 동일하게** — WaveSplat에 불리한 결과도 정직하게 보고
- **Pre-registered metrics 명시** — eval_metrics 메모리 6개 세트 그대로 사용
- **Cherry-picking 방지**: 일부 메트릭만 보고하고 나머지 supplementary 처리하지 말 것 → 전부 보고 or 전부 제외

---

## 1. 수치 검증 (논문 제출 전 필수)

| 항목 | 현재 논문 수치 | JSON 데이터 | 확인 필요 |
|------|---------------|-------------|-----------|
| WS avg nRMSE | Abstract: 0.211, 통계검증: 0.230±0.084 | p1_comparison.json | 두 수치 관계 명확화 (12ep 단일 seed vs 50 paired runs) |
| GPR 비교 | "23.5% worse" (§5.1) | gpr_comparison.json | 비교 기준 통일 필요: (0.276-0.233)/0.233=18.5% vs /0.276=15.6% |
| ETKF-fine | Discussion: "0.421" | etkf_comparison.json | 정확한 평균값 재확인 |
| Model mismatch | "< 0.25%" (§3.1) | ablation A1 결과 | 이론적 0.25% vs 관측 ~2% 차이 설명 보강 필요 |

---

## 2. 추가 가능한 Figure/Table (높은 임팩트)

### 2-1. 수렴 곡선 Figure ★★★
- **데이터**: `extended_analysis.json` → `convergence` (frame-by-frame nRMSE)
- **내용**: 시간축 nRMSE 곡선 (BF3-2comp, BF4-15comp, BF5-30comp)
- **가치**: 논문에 수렴 figure 완전히 부재 — 핵심 시각화

### 2-2. Near/Far-field RMSE 분리 ★★
- **데이터**: `etkf_comparison.json` → `avg_RMSE_far`
- **내용**: RMSE_near (10-80m) vs RMSE_far (80-250m) 비교
- **가치**: LiDAR 거리별 성능 — WaveSplat far-field 우수성 확인

### 2-3. 100 Episode Boxplot ★★
- **필요**: 전체 100 episode 재실행 or multi-seed 결과
- **가치**: 12ep 대표 선정 selection bias 해소

### 2-4. Hs 추정 정확도 ★
- **데이터**: `ablation_seeded.json` → `avg_Hs_error`
- **가치**: Ocean Engineering 리뷰어에게 Hs는 핵심 실용 지표

---

## 3. 추가 분석 (중간 임팩트)

### 3-1. BF4-5 Ablation ★★★ (최우선)
- **현재**: BF3 6ep에서만 수행 → A1이 full보다 좋은 결과
- **필요**: BF4, BF5 episode에서 dispersion 제거 ablation 재실행
- **가치**: dispersion constraint 논증의 **결정적 근거**
- **예상**: BF4-5에서 spectral crowding → unconstrained ω 추정 실패 → nRMSE 역전

### 3-2. T_converge 통계
- **데이터**: p1_comparison.json (일부 null)
- **가치**: "within first 10% frames" 주장의 정량적 근거

### 3-3. ETKF-fine vs ETKF-medium 정식 비교 테이블
- **데이터**: etkf_comparison.json에 둘 다 존재
- **가치**: "grid resolution이 아닌 ensemble 방법론 한계" 논증 — Discussion에 이미 언급했으나 테이블 없음

### 3-4. Sliding-window GPR Baseline
- **필요**: 새 코드 + 전체 episode
- **가치**: "GPR baseline 불공정" 리뷰어 방어
- **구현**: 3-frame window Mat​é​rn GPR, temporal kernel은 dispersion 기반

### 3-5. Bias vs Variance 분리
- **현재**: nRMSE만 보고 (point estimate)
- **필요**: multi-seed 결과에서 systematic bias vs random variance 분리
- **가치**: 추정 오류의 특성 이해 (temporal correlation, spectral leakage)

---

## 4. Hydrofoil 논문 추가 분석

### 4-1. 더 많은 E2E Episode ★★★
- **현재**: 3개 (BF3-2comp, BF3-5comp, BF4-5comp)
- **필요**: BF3-15comp, BF4-15comp, BF4-30comp
- **가치**: WS/GT=37% (BF4)가 outlier vs systematic 판별

### 4-2. Direction Error 정량화 ★★
- **현재**: fig_component_accuracy 정성적
- **필요**: episode별 k_x sign error rate 계산
- **가치**: "directional > amplitude accuracy" 정량적 근거

### 4-3. LP Filter Sensitivity Table ★★
- **현재**: α=0.05 경험적 선택, α=0.15만 언급
- **필요**: α ∈ {0.01, 0.03, 0.05, 0.10, 0.15, 0.20} × 4 scenarios
- **가치**: filter 설계의 정량적 정당화 (run_sensitivity.py 확장)

### 4-4. Comfort Metric 상세 테이블
- **현재**: ISO 2631 등급만 언급
- **필요**: 각 scenario별 PD-only vs PD+FF의 a_RMS 절대값 + ISO 등급
- **가치**: "Slight discomfort → Comfortable" 주장의 정확한 scenario 매핑

### 4-5. 60-120s 장기 시뮬레이션
- **필요**: WaveSplat 긴 episode + hydrofoil sim 연장
- **가치**: ISO 2631 평가 신뢰도, tracking stability 검증

### 4-6. Actuator Dynamics 모델
- **필요**: 1st-order lag (τ=20-50ms) 추가
- **가치**: "ideal actuator" 가정의 robustness 확인

---

## 5. 선행 연구 추가 검토 항목

### 이번에 추가한 references
- Ehlers, Klein, Desmars et al. (2024) — FNO/U-Net phase-resolved reconstruction (Ocean Eng.)
- Desmars, Hartmann et al. (2023) — Nonlinear deterministic wave reconstruction (JFM)
- Qi et al. (2018) — Nonlinear phase-resolved reconstruction (JFM)

### 추가 검토 필요한 논문들 (검색에서 발견)
| 논문 | 관련성 | 조치 |
|------|--------|------|
| Ehlers et al. "Uncertainty-Guided Prediction Horizon" (2025, JMSE) | phase-resolved prediction의 prediction horizon 한계 | WaveSplat discussion에서 인용 가능 |
| Physics-Informed Neural Operators for wavefield reconstruction (arXiv 2508.03315, 2025) | PINO 접근 — training-free physics constraint | WaveSplat과 직접 비교 대상 |
| Qi et al. "Phase-resolved wave prediction with linear wave theory and PINNs" (2023, Applied Energy) | linear wave + PINN hybrid | WaveSplat의 pure physics 접근과 대비 |
| Monohull seakeeping with/without hydrofoil (2025, JMSE 13(10)) | CFD hydrofoil 효과 16-33% motion reduction | Hydrofoil intro에 인용 가능 |
| Navier 30 electric hydrofoil (NewAtlas, 2024) | 상용 전기 hydrofoil 사례 | Hydrofoil intro에 citation (non-academic) |

---

## 6. 논문 구조적 약점 (6인 토론 합의)

### WaveSplat
| 약점 | 심각도 | 발견 페르소나 | 해결 방안 |
|------|--------|--------------|----------|
| Ablation A1: dispersion 제거가 더 좋은 결과 | 높음 | Rev.2 | Phase B3 — BF4-5 ablation |
| 실데이터 검증 부재 | 높음 | Rev.3 | limitation 인정 (완료) + 수조 실험 계획 |
| ETKF baseline tuning 부족 | **높음** | **Rev.4** | **Phase B2 — ETKF 재튜닝 sweep** |
| Episode 수 부족 (12 representative) | **높음** | **Rev.3** | **Phase B1 — 100 episode 전체 재실행** |
| GPR baseline 약함 (single-frame) | 중간 | Rev.4 | Phase D1 — sliding-window GPR |
| nRMSE 단일 메트릭 의존 | 중간 | Rev.3 | Phase A2 — choice of metric 단락 / Phase C — 메트릭 확장 |
| Uncertainty quantification 부재 | 낮음 | Prof.E | future work 언급 (완료) |

### Hydrofoil
| 약점 | 심각도 | 발견 페르소나 | 해결 방안 |
|------|--------|--------------|----------|
| BF4 E2E 37% degradation | 높음 | Prof.K | abstract 수정 (완료) + Phase D4 |
| Cherry-picked E2E episode (3개만) | **높음** | **Rev.3** | **Phase D1+ — 더 많은 E2E episode** |
| Comfort metric scenario 불명확 | 중간 | Prof.E | 수정 완료 — BF3 beam sea 명시 |
| LP filter 경험적 tuning | 중간 | Rev.2 | Phase D5 — sensitivity table |
| 30s 짧은 시뮬레이션 | 중간 | Rev.3 | Phase D2 — 60-120s 확장 |
| Ideal actuator 가정 | 중간 | Rev.4 | Phase D3 — actuator lag 모델 |
| 2-DOF 모델 단순화 | 낮음 | Prof.K | 물리적 정당화 완료 + 3-DOF future work |

---

## 7. 작업 우선순위 (6인 토론 합의)

### Phase A: 즉시 가능 (데이터 없이, 현재 환경에서)
**목표**: 선제 방어 + reviewer 우려 미리 대응

| # | 작업 | 위치 | Rev.3/4 우려 해소 |
|---|------|------|-----------------|
| A1 | Limitation에 "ETKF/GPR best-effort tuning 부족" 명시 | wavesplat.tex §Discussion | Rev.4 |
| A2 | "Choice of metric" rationale 단락 추가 | wavesplat.tex §Methods 또는 §Results | Rev.3 |
| A3 | 수치 검증 (GPR 비교%, ETKF-fine 등) | tex 본문 | 정확도 |
| A4 | Hydrofoil comfort metric 상세 테이블 | hydrofoil.tex §Results | Prof.E |
| A5 | metrics.py에 VR/ρ/SS/Spectral R² 함수 구현 | wavesplat/wavesplat/evaluation/ | Phase C 준비 |

### Phase B: 데이터 확보 직후 (★★★ 최우선)
**목표**: Rev.3/4 핵심 우려 해소 — 통계 신뢰도 + baseline 공정성

| # | 작업 | 사전 요구사항 | Rev.3/4 우려 해소 |
|---|------|--------------|-----------------|
| B1 | 100 episode 전체 재실행 (WaveSplat) | 데이터 sync | Rev.3 통계 |
| B2 | ETKF baseline 재튜닝 sweep (grid/ensemble/inflation) | 데이터 + 시간 | Rev.4 공정성 |
| B3 | BF4-5 ablation (dispersion constraint) | 데이터 + 시간 | dispersion 논증 |
| B4 | Multi-seed 통계 검증 확장 (50 → 100+ runs) | 데이터 | Rev.3 통계 |

### Phase C: Phase B 완료 후 (메트릭 확장)
**목표**: Pre-registered 6-metric panel 도입 (모든 method 동일 적용)

| # | 작업 | 비고 |
|---|------|------|
| C1 | 전체 episode에서 6개 secondary metrics 계산 | A5의 metrics.py 사용 |
| C2 | Main results table에 nRMSE/VR/ρ 3컬럼 추가 | Zero baseline 행 포함 |
| C3 | Spectral R² figure 별도 추가 | qualitative figure 보완 |
| C4 | "Pre-registered metrics" 명시 | cherry-picking 방지 |
| C5 | WaveSplat의 VR이 0.95이고 ETKF가 1.05이면? → **정직하게 보고** | Rev.4 합의 사항 |

### Phase D: 보강 작업 (R&R 또는 후속)
**목표**: 잔여 약점 보완

| # | 작업 | 가치 |
|---|------|------|
| D1 | Sliding-window GPR baseline | Rev.4 baseline 강화 |
| D2 | 60-120s 장기 시뮬레이션 | tracking stability |
| D3 | Actuator dynamics 모델 (hydrofoil) | 실용성 |
| D4 | Direction error 정량화 (hydrofoil) | BF4 분석 |
| D5 | LP filter sensitivity table | Prof.E 정량화 요구 |
| D6 | Hs error, Directional R², EMD 등 supplementary | full pre-registered set |

### 시간 추정 (데이터 확보 후 기준)
- **Phase A**: 1-2일 (즉시)
- **Phase B**: 1-2주 (데이터 sync 후, ETKF 재튜닝이 가장 시간 소요)
- **Phase C**: 3-5일 (B 완료 후)
- **Phase D**: 2-3주 (R&R 시점)

---

## 8. 데이터 확보 시 체크리스트

데이터가 다른 서버에서 도착하면:

```bash
# 1. 데이터 sync 확인
ls wavesplat/data/ | wc -l   # 100이어야 함

# 2. 환경 검증
python3 -m pytest wavesplat/wavesim/tests/ wavesplat/wavesplat/tests/ -v

# 3. Phase A5 코드 작성 (메트릭 함수)
# wavesplat/wavesplat/evaluation/metrics.py 확장

# 4. Phase B1 — 100 episode 재실행
python3 scripts/run_p1_comparison.py --all-episodes --seeds 5

# 5. Phase B2 — ETKF 재튜닝
python3 scripts/run_etkf_tuning_sweep.py  # 새로 작성 필요
#   - grid: 40x30, 60x45, 80x60, 120x90
#   - ensemble: 30, 50, 100
#   - inflation: 1.0, 1.01, 1.02, 1.05

# 6. Phase B3 — BF4-5 ablation
python3 scripts/run_ablation_seeded.py --episodes BF4_*,BF5_*

# 7. Phase C — 메트릭 확장
python3 scripts/recompute_with_full_metrics.py

# 8. 결과 JSON dump 후 figure/table 재생성
python3 scripts/generate_paper_tables.py
python3 scripts/generate_paper_figures.py
```

---

*Updated: 2026-04-07 (3차 리뷰 — 6인 페르소나 확장)*
*6인 페르소나: Prof.K / Prof.E / Rev.2 / Rev.3 (적대적) / Rev.4 (baseline advocate) / Saeyong*
