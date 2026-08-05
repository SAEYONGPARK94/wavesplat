# F-5 — Level-calibration을 §8 실측 헤드라인으로 재프레이밍 (2026-07-22)

> 담당: fable (coordinator 지시, writeup 트랙). **지시 전제 정정**: coordinator는 "RAW_SINK 포함"이라 했으나
> `results/direction3/e59v2_real_anchor.json`은 **요약 행(rows)만 저장**하고 RAW_SINK(질의점별 crb/act/dist/emp
> 원값)는 저장돼 있지 않았다(스크립트 기본값 `RAW_SINK=None`, 훅만 존재) — E60K 감사 F3와 같은 "산출물
> 미영구화" 패턴. **그래서 재인용에 그치지 않고 `tmp/e59v2_real_anchor.py::process_record`를 RAW_SINK
> 캡처 모드로 직접 재실행**(기존 코드·기존 실측 stereo 데이터 100% 재사용, 재학습·신규 시뮬 없음, 4~18초)해
> **독립 재도출**했다. 스크립트 `tmp/f5_level_calibration.py`, 원자료 `results/direction3/f5_level_calibration.json`.
> 정경 파일(`e59v2_real_anchor.json`) 무수정 확인(git status).

## 1. 독립 재도출 결과 — E59_AUDIT RESCUE 절 수치 재현

| 레코드 | 기존 인용(E59_AUDIT) | **본 재도출 (REAL)** | 판정 |
|---|---|---|---|
| AA15a | level-ρ +1.00 | **+1.000** | 일치 |
| AA14  | level-ρ +0.92 | **+0.915** | 일치(Δ0.005) |
| YS17  | level-ρ +0.89 | **+0.891** | 일치 |
| (전체) | 5~6× dynamic range | **AA15a 5.9×· AA14 6.0×· YS17 4.7×** | 일치(YS17 근접) |

**독립 스크립트 재실행(기존 코드 재사용이지만 새 프로세스, 새 rng 스트림 아님 — E-59 v2가 이미 시드 고정)으로
정확 재현.** → **THESIS_FRAMING/PAPER_SKELETON 인용값은 아티팩트 아님, MEASURED 승격 확정.**

**TWIN(합성, 상한 참고)**: 2회 실행에서 dyn_range가 달라짐(18.9~39.4× vs 7.7~31.6×) — 원인 규명:
`e59v2_real_anchor.py::process_record`의 twin 합성이 `seed=hash(rc) % 999`를 쓰는데, **Python 문자열
해시는 프로세스마다 무작위**(PYTHONHASHSEED 기본 랜덤) — 그래서 TWIN 합성 진실장 자체가 매 실행 다르다.
**REAL 트랙은 이 문제와 무관**(진짜 stereo 데이터라 해시에 안 걸림) — level-ρ·dyn_range 둘 다 기존 인용과
일치했으므로 **REAL 결과는 재현 가능·안정**. **사소한 발견(기록)**: TWIN의 "12~51×" 인용은 정밀 재현 가능한
숫자가 아니라 **자릿수 참고치로만 사용해야 함**(스크립트에 `PYTHONHASHSEED` 고정이 없음 — 원하면 1줄 수정으로
쉽게 고정 가능하나 우선순위 낮음, TWIN은 상한 참고용이지 헤드라인 아님).

## 2. §8/Abstract용 초안 문장 (재프레이밍)

### 현재 서술(순위-상관 중심, PAPER_SKELETON §8)
> "...envelope 예측 신뢰영역 vs 실제 skill(Spearman(σ_CRB,σ_actual))... 결과=PARTIAL(예측바 ρ≥0.5 미달)..."

### 제안 재구성(크기-보정 우선, 순위-상관은 보조)
> **"실측 스테레오 3개 site-day에서, CRB-앵커 envelope이 예측한 오차 크기(σ_CRB)는 실제 관측 오차의 **위치와
> 크기**를 강하게 단조 예측한다 — 예측 십분위수(decile)에 따라 실제 오차가 **4.7~6.0배** 체계적으로 증가하며
> (level-Spearman ρ=0.89~1.00, n_q=8,900~12,900 질의점/레코드), 이는 순위-상관 단독 헤드라인보다 운용자에게
> 더 직접적인 결론이다: **envelope 십분위가 낮다고 말하는 곳은 실제로 오차가 작고, 높다고 말하는 곳은 실제로
> 크다 — 6배 차이로.**"**

### 필수 병기 캐비엇 (인용 시 반드시 동반)
1. **일부는 기하다**: E59_AUDIT 원문 그대로 — "level-calibration도 일부 기하(거리)" — 순수 정보 우위 증명이
   아니라 "어디서/얼마나" 운용 질문엔 충분하다는 것. §8 rank-Δρ(+0.085, 정보 축)와 분리해서 제시할 것.
2. **동적범위 압축**: 실측(4.7~6.0×)이 합성 twin(대략 10~40× 자릿수, 정밀 재현 불가)보다 좁음 — 실측 오차
   바닥이 모델 불일치(mismatch)로 눌려 있다는 기존 해석 유지.
3. **n=3 site-day, 3 캠페인**(AA15a/AA14/YS17) — 통계적으로 얇음. 십분위-내 n_q는 크지만(8,900+) **군집은
   레코드 3개뿐**이므로 레코드-급 불확실성은 여전히 큼(E-59 blocked-CI 원칙 적용 시 레코드 수가 유효 n).
4. **스테레오 유래**(실물 이동 LiDAR 미관측) — 프로젝트 전역 캐비엇, 여기서도 유효.
5. **nowcast~짧은 lead**(0~6초, `LEADS`/`lead_fr` 그대로) — 장기 예보 외삽 미검증.

## 3. 판정
- **MEASURED (신규)**: level-calibration 결과가 독립 재도출로 확정됨(원 인용값과 소수점 2~3자리 일치) —
  이제 "재인용"이 아니라 "재확인된 측정치"로 쓸 수 있음.
- **권고**: §8/Abstract 1차 헤드라인을 rank-Spearman(0.085, 상대적으로 약하고 해석 복잡)에서
  **level-calibration(4.7~6.0×, 해석 직관적)으로 무게중심 이동**. rank-Δρ는 "정보 대 기하" 판별용 **보조
  지표**로 격하하지 말고 병기하되, NAOE 리뷰어를 겨냥한 1문장 훅으로는 level-calibration이 더 강함.
- **후속 권고(저비용, 미실행)**: `e59v2_real_anchor.py`에 `PYTHONHASHSEED` 고정 또는 `hash(rc)`를
  `zlib.crc32(rc.encode())` 같은 결정적 함수로 교체 — TWIN 상한치를 정밀 인용 가능하게 만듦(1줄 수정, 헤드라인
  영향 없음, 위생 차원).
