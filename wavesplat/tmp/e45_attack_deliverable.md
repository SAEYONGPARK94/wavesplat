# E-45 — 적대적 공격 보고 (fresh-context adversary)

**표적**: "cross-Tp regime prize는 실재하고, DEPLOYMENT task(단기 encounter forecast) 위에 존재하며, blind 추론은 ≈0이지만 외부 공급 θ로 회수된다: buoy-grade +0.037, operational-forecast-grade +0.028." (E-43 dose-response + E-44 encounter transfer)

**최종 판정 한 줄**: **절반 생존.** prize의 실재·deployment 전이·blind≈0·buoy-grade 회수는 최선의 공격을 견뎠다(수치는 haircut). 그러나 **ww3(operational-forecast)-grade 회수 +0.028은 BROKEN** — 아레나가 truth로 동결한 축(Tp1, spread)에 같은 grade의 오차를 정직하게 부여하면 encounter gain은 **−0.009 CI[−0.037,+0.019]** 로 소멸한다(paired drop ww3−ww3_f = +0.0512 CI[+0.0274,+0.0751], 0 배제). "58%"는 아레나 상수를 무료로 받은 값이다.

모든 수치는 raw JSON에서 재도출했고, 프로브는 `tmp/e45_attack_*.py` 3본(기존 스크립트/결과 무수정). 프로브 harness는 3중 검증: ① iid bundle이 E-43 JSON을 **bit-for-bit 재현**(+0.0639/+0.0415, 동일 rng stream), ② unimodal 프로브의 `enc10_cur`가 E-44 JSON과 **240/240 일치**, ③ 프로브 3의 orc/buoy/ww3는 E-44 저장값을 그대로 재사용.

---

## 공격선별 판정

### 1. ★Encounter task가 조작인가 (query 배치 artifact) — **SOUND**
- `track_future`(tmp/e44_probe_N2_encounter_transfer.py:54)와 `sample_obs`(scripts/d3_sa_achievability_gap.py:72)를 코드 대조: **동일한 ping-pong 운동법칙** (P0=[r_fp,r_fp], P1=[side−r_fp,side−r_fp], s=mod(U·t,2L)), 동일 기하(side=2λ, r_fp=0.09λ). 미래 트랙은 관측 법칙의 연속이다. `render_at`은 field anchor의 gt_u와 같은 관례(clean true modes).
- "미관측 영역 외삽" 가설 정량화: U·70 < L(=196.8 m), 즉 U<2.81 m/s인 바다만 미래 트랙이 공간적 신영역에 든다 — **140/957 바다**. 그 바다들의 gain +0.0555±0.0186 vs 커버된 바다 +0.0410±0.0084. corr(gain, new-territory fraction)=+0.03, corr(gain,U)=+0.05. **U 4분위 전부 양수**(Q1 +0.039 … Q4 +0.060). prize는 query 배치가 아니라 바다 전반에 퍼져 있다. 기각 실패.
- 부수 확인: prize는 partition 분리가 클수록(corr(d10, dir offset)=+0.44, far-Tp2 +0.075 vs near-Tp2 +0.008), 그리고 **2차 partition이 약할수록**(low-r2 +0.105 vs high-r2 −0.022) 크다 — "데이터가 2차 lobe를 구속 못하는 곳에서 외부 정보가 지불한다"는 기제와 정합. artifact 신호 아님.

### 2. 0–5 s vs 5–10 s 분할 — **SOUND (물리적), 단 scope를 조인다**
- 절대 skill 재도출: 5–10 s에서 **모든 arm이 예측 불능**(cur median corr +0.27, q25 음수; orc도 +0.27). 0–5 s는 cur median +0.61. 경계 artifact가 아니라 이 관측창의 predictability horizon이다. ww3 5–10 s = +0.0000 재현.
- 그러나 이는 claim을 조인다: **회수되는 가치는 전부 0–5 s lead에 있다.** 0–5 s lead가 제어 액추에이션에 충분한지는 이 arc 어디에서도 검증되지 않았다("control 관련성"은 corr-on-track까지만).

### 3. n=957의 정밀한 소효과? (분포·robust) — **WEAKENED (크기), 실재는 확인**
- 재도출: enc10 orc−cur mean +0.0431 CI[+0.0280,+0.0582], Wilcoxon 3.61e-08 — 헤드라인 수치 전부 JSON에서 재현됨.
- 그러나 **median은 mean의 절반**: orc−cur med +0.0236, buoy−cur med +0.0256, ww3−cur med +0.0152. win rate 57.7%. skew +0.65, kurt +3.0. **상위 5% |d| 바다가 총합의 40%** 를 운반. per-sea Cohen's d ≈ 0.18.
- robust 추정끼리는 합의(trimmed-10% +0.0359, Wilcoxon 유의) → "정밀하게 측정된 noise"는 아니고 실재하는 소효과. 단 서술은 "**전형적 바다 +0.02, 평균 +0.04는 tail-driven**"으로 써야 정직하다.
- corr가 제어 utility의 옳은 functional인가(phase/timing)는 N2의 사전 등록 caveat 그대로 미해결 — scope 문장에 남겨야 한다.

### 4. ★구조화(상관) 오차 — **모델된 4축에서는 SOUND / 동결 축에서 ww3 BROKEN**
- 브리핑이 지목한 "가장 유망한 선"인 **상관 오차 공격은 실패했다**: E-43의 64 바다(R=5, iid arm은 rng까지 E-43과 동일=bit-exact 재현)에서 common-mode direction, relative-geometry, (Tp2,r2) joint ±, all-axes 완전상관 — **전 구조가 iid의 ±0.006 이내** (buoy +0.064~+0.068, ww3 +0.036~+0.043). iid 가정은 {d1,d2,Tp2,r2}에 대해 load-bearing이 아니다.
- **진짜 시체는 옆에 있었다**: E-43의 bundle은 아레나가 **truth로 하드코딩한 Tp1=7.0, spread=30°, γ=3.3을 오차 0으로 공급**한다(`supplied_grid`, tmp/e43_external_regime_dose.py:85-90 — Tp1·spread·γ가 인자에 없음). 실제 buoy/WW3는 이 축들도 자기 grade의 오차로 공급한다. 같은 상대 grade를 Tp1·spread에 추가하면(field level, n=62):
  - buoy: +0.0639 → **+0.0529** (Tp1±5%) → **+0.0529** (+spread±5°) — 생존, prize의 89%→74%.
  - ww3: +0.0415 → **+0.0180 CI[−0.0103,+0.0464]** (Tp1±10%) → **+0.0058 CI[−0.0193,+0.0310]** (+spread±10°) — **통계적 0**.
- **encounter functional에서 직접 확인**(프로브 3, e44 fresh 바다 si=1000..1319, n=318, 저장된 cur/orc/buoy/ww3 재사용): buoy_f **+0.0349 CI[+0.0084,+0.0615]*** (claim +0.0371과 통계적 동일; buoy−buoy_f +0.0088 CI∋0) / **ww3_f −0.0092 CI[−0.0369,+0.0185], med −0.0162, Wilcoxon 0.34** — claim +0.0284는 CI 상단(+0.0185) 밖. **ww3 leg 사망.**

### 5. buoy/ww3 grade의 정직성 — **buoy 정직, ww3는 낙관적 + 축 목록이 불완전**
- 문헌: buoy mean-direction bias 1.4–2.8° (Spotter 비교) → d±5°는 정직. 모델 peak-direction 오차 16–17° 관측 사례 → d±15°는 그럴듯~낙관. **Tp RMSE는 1.5–2 s 보고**(군도 해역; 외해 ~1–1.5 s) → Tp±10%(=0.7 s @7 s)는 **낙관 쪽**. 즉 내 ww3_f arm은 가혹하지 않다 — 그런데도 죽었다.
- 더 근본적으로: 실제 forecast의 지배적 실패는 **partition 오류**(병합/누락/오라벨)인데 E-43/44는 regime 분류가 항상 옳다고 가정한다(아래 7-b에서 부분 해소).

### 6. 전부 시뮬레이션 — **scope 그대로, 추가로 HOS 상시지시 위반**
- 두 leg 모두 JONSWAP 합성(Tp1=7, γ=3.3, spread=30° 하드코딩, 선형 렌더링, 합성 LiDAR 샘플링, 70 s 단일 창) 안의 진술이다. 내 프로브들도 같은 아레나 안이다 — buoy leg "생존"조차 아레나-내 진술.
- **이 arc의 생존 헤드라인은 HOS 비선형에서 재현된 적이 없다**(상시지시 "검증엔 항상 HOS 포함, 헤드라인은 HOS서 재현" 위반). 실물 LiDAR 관측 0회, 실측은 stereo 유래 2 records뿐(이 arc에는 실측 leg 자체가 없음).

### 7. 통계 위생 — **두 leg는 깨끗, 주변에 3건**
- e43d `paired()`(tmp/e43d_gate.py:69-77)와 e43 자체 `paired()` 모두 **t-interval 사용, 1.96 아님** — 재도출로 CI 일치 확인. Wilcoxon 3.61e-08 재현.
- (a) **1.96 버그는 e41에 잔존**: `tmp/e41_regime_prize_ladder.py:356-357`. 단 e41은 deliverable의 leg가 아니고 e44 anchor(+0.0720, clean stats)가 prize를 독립 재측정하므로 non-load-bearing. n≥62에서 수치 차이도 무시 가능.
- (b) **e44의 사전 등록 UNI do-no-harm 컨트롤이 조용히 누락**: uni 루프가 `cur`만 채점(tmp/e44_encounter_transfer.py:161 — `for nm, g in [('cur', g_cur)]`), 240 바다가 비교 대상 없이 버려짐 = 프로토콜 위반. **내가 채워서 돌린 결과는 deliverable에 유리**: 올바른 unimodal 공급 +0.033*(buoy-grade), phantom false-bimodal 비용은 enc에서 −0.006(buoy)/−0.016(ww3) CI∋0, field ph_ww3만 −0.021* 유의. regime 오분류 리스크는 실재하나 +0.037 대비 소액.
- (c) **n=957의 결측 3은 missing-not-at-random**: si 1166/1284/1402에서 `two_stage_kp`가 17–37× 파탄(kp_est 0.002–0.005 vs true 0.082, 전부 고속 U 11.9–14.1) → 공급 grid가 k/kp bank 밖 → all-zero → NaN → pairwise 제외. E-43도 같은 기제로 **실제 n=62**(문서의 n=64 아님). cur도 그 바다들에서 파탄(fld −0.22~+0.07)이라 제외는 대략 중립이지만, **외부-θ 채널이 내부 kp 추정에 의존하며 kp가 죽으면 함께 죽는다**는 배포 취약점은 문서화돼야 한다.

---

## 결론 — 무엇이 서 있고, 어떤 문장으로 쓸 수 있는가

**서 있는 것** (최선의 공격 후):
1. cross-Tp regime prize는 field에서 +0.072, encounter(0–10 s)에서 +0.043(mean)/+0.024(median)로 실재하며 deployment functional에 전이된다 (transfer ratio 0.60; 가치의 전부가 0–5 s).
2. blind 추론 ≈0 (m2−cur +0.006 CI∋0; 본 공격의 표적 아님).
3. **buoy-grade 외부 θ 회수는 전 축 정직 오차에서도 생존**: +0.0349 CI[+0.0084,+0.0615] (claim +0.0371 대비 통계적 동일).
4. do-no-harm: 올바른 unimodal 공급은 이득(+0.033), phantom false-bimodal 해악은 소액.

**부러진 것**:
- **"operational-forecast-grade +0.028 (58%)"은 그 형태로는 거짓** — Tp1·spread를 아레나 truth로 공짜로 받은 조건부 수치. forecast-grade 오차를 그 축들에 적용하면 encounter −0.009 CI[−0.037,+0.019], field +0.006 CI[−0.019,+0.031]. 철회 또는 "d·Tp2·r2에만 오차가 있는 가상 소스" 명시로 재범위화해야 한다.

**허용 가능한 서술 (전문)**:
> "합성 JONSWAP cross-Tp 아레나(Tp1=7 s, γ=3.3, spread=30°, 선형 파, 모사 LiDAR 궤적 샘플링, 70 s 단일 창) 안에서: cross-Tp regime prize는 field +0.072, encounter forecast(0–5 s lead 지배, 0–10 s 평균 +0.043, **전형 바다 +0.02**)에서 실재하고, 이 센서로는 blind 추론이 불가하며(≈0), **모든 스펙트럼 축에 buoy-grade 오차(d±5°, Tp±5%, r2±0.05, spread±5°)를 가진 외부 소스로 +0.035가 회수된다**. operational-forecast-grade 회수는 확립되지 않았다(동결 축에 오차 부여 시 소멸). 5–10 s lead에서는 어떤 arm도 가치를 회수하지 못한다. HOS 비선형·실측 재현 없음. corr-on-track이 제어 utility를 대표하는지는 미확립."

**내가 던졌고 견딘 것**: query-배치/신영역 정량 공격, U-계층 분해, 상관 오차 5구조×2 grade(bit-exact 앵커), robust/rank 재추정, phantom-bimodal do-no-harm(누락 컨트롤 대리 실행), NaN 결측 기제 해부, 통계 재도출 전수. **buoy leg를 부수려면**: 상관 구조가 아니라 (i) HOS/실측 아레나에서의 붕괴, (ii) buoy가 partition-level 파라미터(특히 약한 2차 partition의 r2·Tp2)를 실제로 그 grade로 주는가에 대한 실증 반례, (iii) corr→제어 utility 전이 실패 — 이 셋 중 하나가 필요하다. 이번 공격 범위 밖이었다.

**프로브 파일**: `tmp/e45_attack_structured.py`(+.json/.log), `tmp/e45_attack_dnh_enc.py`, `tmp/e45_attack_enc_frozen_axes.py` — 기존 스크립트·결과 무수정.

**Sources** (error-grade 문헌 확인):
- [WAM, SWAN and WAVEWATCH III in the Finnish archipelago (Tp RMSE 1.5–2 s)](https://www.tandfonline.com/doi/full/10.1080/1755876X.2019.1633236)
- [Quantifying errors in wind and wave measurements from a compact, low-cost wave buoy (mean-direction bias 1.4–2.8°)](https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2022.966855/full)
- [Directional wave measurements from navigational buoys (peak-direction errors ~16–17°)](https://www.sciencedirect.com/science/article/pii/S0029801822024441)
