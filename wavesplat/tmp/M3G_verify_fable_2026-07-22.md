# M-3G 발견 독립검증 — "gate = no-op, 훈련 kp가 처음부터 beamform" (fable, 2026-07-22)

> 권위문서 `tmp/M3G_RESULT_2026-07-22.md`. red-team 대신 이 발견을 독립검증(지시대로). coordinator의
> 캐시(`tmp/bwtask_kp_beam.npy`) 미사용 — 전부 새로 계산. 코드 확인 + 8000 scene 전수 재계산(스팟체크보다
> 강한 검증, 27초 소요).

## 판정 = **CONFIRM** (핵심 발견 전부 확증), 단 결론부에 **중요 단서 1건 추가**

## 1. 코드 확인 — `d3_prior_net_data.py:132-138` (직접 열람)
```python
# kp estimate exactly as at inference time (two-stage beamform zoom)
bx, by = beamform_top(P.astype(np.float32), z.astype(np.float32), DEPTH, klim=2.0, n=120)
kp_e = max(np.hypot(bx, by), 1e-3)
bx, by = beamform_top(P.astype(np.float32), z.astype(np.float32), DEPTH, klim=3.5 * kp_e, n=120)
kp_e = max(np.hypot(bx, by), 1e-3)
```
**확인: 정확히 주장대로.** `kp_e`(→ `meta[:,0]` → `sample_loss`의 `float(kp_e)` → 세 팔 전부의 bank 격자
배치)는 처음부터 그 scene 자신의 (P,z)에서 뽑은 **2단 beamform**이지 참(oracle) 스펙트럼 피크가 아니다.
"data bwtask"가 E-B2K 원 훈련의 데이터 네임스페이스라는 것도 `d3_b2deep_train.py`(기본
`--data tmp/prior_train/bwtask`)와 E60K prereg의 "arch PriorNet+SmultHead, init v7fan, data bwtask" 서술로
재확인 — 다른 kp 소스로 몰래 바뀔 경로 없음(`d3_prior_net_taskft.py::sample_loss`가 `D['meta'][i]`를
그대로 unpack, 오버라이드 없음).

## 2. 독립 스팟체크 → **전수 재계산으로 격상** (coordinator 캐시 미사용, 새로 계산)
`tmp/prior_train/bwtask_s000~003.npz`(**8000 scene 전부**, 캐시 아님)에서 저장된 `P[i],z[i]`로부터
**독립적으로** 2단 `beamform_top`을 재실행(coordinator의 `bwtask_kp_beam.npy` 파일 열지 않음, `d18_relax_
estimator.beamform_top`만 재사용해 직접 호출):

```
total scenes = 8000
max|diff|    = 4.615e-07   (float32 반올림 수준)
median|diff| = 1.364e-09
fraction |diff|<1e-6 : 100.000%
kp 분포(meta): min 0.0063  p25 0.0471  median 0.0666  p75 0.1062  max 11.4548(이상치 존재)
count kp<0.05: 2383   <-- coordinator가 인용한 "저-kp 장파 2383개"와 정확히 일치
```
**8000개 전부 bit-exact 재현**(스팟체크가 아니라 전수 — 더 강한 증거) + coordinator가 인용한 "2383"
수치까지 독립적으로 재현(같은 kp<0.05 정의를 우연히 골랐는데 일치 — 교차검증 성격). **인덱싱 버그로 인한
우연 일치 가능성도 배제**: P,z,meta가 뒤섞여 있었다면 8000개 다양한 scene에서 이렇게 정확히 일치할 수
없음.

## 3. E-B2K eval의 kp 소스 확인 — `e60k_hos_arena.py` (코드 직접 확인)
`draw_multi`(83행) `kp = sea['kp']` → `build_sea`(158행, `tmp/e15_hos_multidraw.py`):
```python
S = sea_components(seed, Hs=float(Hs))
kp = float(np.hypot(S['kx'], S['ky'])[np.argmax(S['A'])])   # 참 스펙트럼 최대진폭 성분의 |k|
```
**확인: `sea['kp']`는 beamform이 전혀 개입하지 않는 순수 oracle** — HOS 생성 시점의 참 성분 리스트에서
진폭-최대 성분의 |k|. `net_gamma_mmse(..., kp_hint=kp, ...)`로 곧장 주입. **3자 구도 확인됨**:
- **훈련(bwtask, 8000 scene)**: kp_e = 2단 beamform (sensor) — §1,§2로 확증.
- **E-B2K eval(e60k_hos_arena)**: kp = `sea['kp']` = 참 피크 (oracle) — §3으로 확증.
- **M-3 eval(m3_sensor_kp_arena)**: kp = beamform(600점, 동일 2단 레시피, 이전 F-6/red-team서 이미 확인) —
  train과 **같은 계열**(beamform), oracle 아님.

## 4. 논리 감사 — "M-3(+0.022)가 옳은 답, +0.139는 oracle-inflated, gate는 no-op"
**핵심 3단은 타당, 결론에 단서 1건 필요.**
- **"gate = no-op" 필연적으로 참**: 훈련 kp_e가 이미 beamform이므로 "beamform으로 바꾸는" 개입 자체가
  대상이 없다 — §2가 bit-exact로 증명. 반박 불가.
- **"F1의 oracle-kp 전제가 kp에 대해 거짓"도 참** — 단 F1의 정확한 스코프(§5 E60K_RESULT F1 원문)는 "oracle
  kp/σ₀ 입력"이라 σ₀ 부분은 여전히 유효(sig0=nfrac·Hs는 훈련·평가 공통, oracle 여부 무관 — 이건 M3G_RESULT도
  안 건드림, 일관).
- **"+0.139는 eval-only oracle 아티팩트"는 강하게 지지되나, 인과의 마지막 조각(왜 하필 T가 oracle-eval서
  유리한가)은 이 검증 범위 밖** — M3G_RESULT 자체도 이 메커니즘까지는 주장 안 함(내가 과잉해석 경계).

### ★반론/구멍 (지시대로 헌팅) — 결론을 뒤집진 않으나 단서 필요
**"M-3(+0.022)가 sensor-실현의 옳은 답"이라는 문장은 살짝 과대일 수 있다.** 이유: "훈련=beamform"과
"평가(M-3)=beamform"이 **같은 함수 호출**(2단 recipe)이라는 건 맞지만, **입력 점군의 통계는 다르다**:
- 훈련 scene(`d3_prior_net_data.py`): **400점**, 그중 1/3은 `'fixed'` 모드로 **개구 지름 0.18λ**(사실상
  점-센서, 이전 red-team서 지적한 La Jument급 개구-빈곤 사례)에서 beamform.
- M-3/e60k 평가: **~600점**, return-power 로터리 + **12-bin range-stratified** 선택(실제 LiDAR 관측 기하 모사).
⟹ 둘 다 "beamform이다(oracle 아니다)"는 참이지만, **"beamform의 편향/분산 특성까지 같은 분포다"는 별개
주장이고 이 검증도 M3G_RESULT도 확인한 적 없다.** 즉 gate가 no-op라는 것(재훈련해도 안 바뀐다)과, M-3의
+0.022가 "**완전히 훈련-일치된** 최종 정답"이라는 것은 **다른 강도의 주장**이다. 전자는 이 문서로 완전
확증됐고, 후자는 **"현재 구할 수 있는 것 중 가장 훈련-일치에 가까운 추정치"** 정도로 완화해 인용하는 게
정직하다 — 완전한 훈련-평가 기하 일치를 확인하려면 400점→600점·기하 재생성 실험이 별도로 필요(안 했음,
이 검증 범위 밖, M3G_RESULT도 §"진짜 남은 경로"에서 유사한 톤으로 겸손하게 남겨둠).

## 결론
- **CONFIRM**: (1) 훈련 kp_e = 2단 beamform, 처음부터(코드+8000개 전수 재계산, bit-exact, 2383 교차재현).
  (2) E-B2K eval kp = 순수 oracle(`sea['kp']`, 코드 확인). (3) gate = 필연적 no-op(재훈련해도 kp_e 안 바뀜).
  (4) F1의 "훈련이 oracle 썼다"는 kp에 한해 거짓, 원인은 eval 쪽 oracle 주입.
- **단서(인용 시 병기 권고)**: "M-3(+0.022)가 sensor-실현의 최종/완전한 답"이라는 문장은 **"현재까지 가장
  훈련-일치된 추정치이나, 훈련-scene(400점·일부 초소형 개구) 대 평가-아레나(600점·LiDAR형 range-stratified)
  간 beamform **통계**(편향·분산) 일치는 미확인"**으로 완화 인용 권고. 결론의 실질(gate no-op, 재훈련 무의미,
  +0.139 인용 금지, B는 M-3 아레나로 재판정)은 **흔들리지 않음.**
