# `T12-ISOLATE` 사전등록 — **`RESCALE` 이득을 나르는 것이 어느 상수인가**

> **동결 시점: 결과 0행 · 러너 0줄 · 추정기 런 0회.**
> 이 문서는 `REACHABILITY.md`(같은 폴더, 런 0회 소스 조사)를 전제로 한다.
> ⛔`tmp/T11_EXTACC_2026-09-03/` 의 어떤 파일도 **수정하지 않았다**(읽기만).
>
> **선행**: `T11-EXTACC :: BAND-MISMATCH`(커밋 `7f6debc`, 게이트 17/17, 96행) —
> 제3자 참값 24블록에서 배포 팔 `SHIP` **8/24 (p=0.968) 무기술**, 좌표만 재라벨한
> `RESCALE`(`x'=x/2, t'=t/√2`) **24/24 (p=5.96e-08)**, `ORACLE` **24/24**.
> ⛔그러나 `AMENDMENT_2_PREDATA.md §6a`(**결과 0행 시점에 동결**)가 스스로 적었다:
> 재척도는 **절대 파수 상수 전부를 `s` 와 공선으로 함께** 움직인다 ⟹ **귀속이 안 된다.**
> ★그 §6a 의 상수 목록이 **결과 이전에 쓰였다**는 점이 이 배치의 축 선택을 사전등록으로
> 만든다(⚠단 §7 의 노출 공시를 함께 읽을 것).

---

## §0 PRIMARY 질문 (한 문장)

***`RESCALE` 이 얻은 기술을, 자료를 건드리지 않고 **제안 서브시스템의 절대 파수 스케일 하나만**
배포값의 절반으로 옮겨서 얻을 수 있는가.***

⛔이것은 *"우리 방법이 좋다/나쁘다"* 가 아니다. `T11` 과 같이 **귀속**이다.
⛔`RESCALE` 자체는 배포 가능한 구성이 아니다(단위를 바꾼 사고실험) — 이 배치에서도 **상한**으로만 쓴다.

---

## §1 팔 — ⛔**빈 dict `{}` 로 정의하지 않는다**

모든 팔은 `cfg = dict(DEFAULT_CONFIG)` 에서 출발해 `depth`·`seed`·`device`·`init_method='auto'`
를 **명시**하고(= `T11 run_t11.py:225-227` 규약), 그 위에 아래 **정확히 하나의** 개입을 얹는다.
개입은 **컨텍스트 매니저로 걸고 반드시 되돌린다**(§6 `G-KNOB`·`G-RESTORE`).

| # | 팔 | 개입 (배포값 → 처치값) | 도달 방식 | 역할 |
|---|---|---|---|---|
| 0a | `ZERO` | — | 창별 제로필드 | 무기술 기준 (**모든 검정의 짝**) |
| 0b | `ORACLE` | — | 참 FFT 파수에서 `A,φ` 만 LS (`T11` 개정 정의, `nmode=100`, ±ω 4열) | 정보 존재 상한 |
| 0c | `SHIP` | 없음 (출하 기본값) | — | **앵커**(`T11` 재현) · 모든 처치의 기준 |
| 0d | `RESCALE` | `x'=x/s, t'=t/√s`, **`s=2.0`** | `T11` 과 축자 동일 | ★**상한(ceiling)** · 앵커 |
| 1 | `A-MAPSCALE` | `d3_beamform_net.KLIM` 0.6→**0.3** **및** `stats['K_LIM']` 1.1→**0.55** | `KLIM` 재대입 + `KX_G,KY_G,OM_G = kgrid()` 재호출 · `LI._load_cond` 래핑 | ★**PRIMARY** |
| 2 | `A-MAPCELL` | `KLIM` 0.6→**0.3** 만 | 위 절반 | 잠금 분해 (§`REACHABILITY.md 2.2`) |
| 3 | `A-BAND` | `stats['K_LIM']` 1.1→**0.55** 만 | 위 절반 | 잠금 분해 |
| 4 | `B-DEDUP` | `cfg['learned_dedup_dk']` 0.03→**0.015** | **순수 config** | 유일한 깨끗한 단일 노브 |
| 5 | `C-MINDK` | `initializer.MIN_DK` 0.15→**0.075** | 모듈 상수 재대입(호출 시점 임포트) | 성장 분리 문턱 |
| 6 | `C-GROWGRID` | 성장 코어스 격자 `±4.0`→**`±2.0`**(`n_kx=n_ky=61` 고정) | `initializer._beamform_coarse` 래퍼 | 성장 서브시스템 스케일 |
| 7 | `D-VPLR` | VARPRO Adam `lr` 2e-3→**1e-3** | ⛔`VP_LR_DEFAULT` 재대입은 **안 먹는다** — `varpro.apply_varpro` **함수 객체**를 래퍼로 교체 | 정련 스텝 |
| 8 | `E-MAPDEPTH` | `d3_beamform_net.DEPTH` 50.0→**400.0** | `DEPTH` 재대입 + `kgrid()` 재호출 | ⚠**스케일 축 아님** — `AMENDMENT_2 §6b` 의 비대칭 교락을 닫는다 |
| 9 | `F-STEEP` | `cfg['max_steepness']` 0.15→**0.30** | 순수 config | ⚠**절제**(회수 아님) — `AMENDMENT_2 §9.1` 공시를 닫는다 |

**처치값의 규약**: 파수 차원[rad/m] 상수는 전부 **÷ s = ÷2**(`RESCALE` 이 자료의 `k` 를 ×2 하는
것과 **자료 고정 상태에서 등가**). ⛔`E-MAPDEPTH`·`F-STEEP` 은 이 규약에 **속하지 않는다**(§3).

### 1.1 ⚠팔이 **의도치 않게 함께 움직이는 것** (결과 전 공시)
| 팔 | 같이 움직이는 것 | 근거 |
|---|---|---|
| `A-BAND`·`A-MAPSCALE` | **polish Adam 물리 스텝** 0.022→0.011 rad/m (`lr·K_LIM`) | `REACHABILITY.md §2.1` |
| `A-MAPSCALE` | `A-MAPCELL` + `A-BAND` 의 **합성** | `REACHABILITY.md §2.2` |
| `C-MINDK` | 실효 억제 반경 0.2667→0.1333 (계단 함수의 우연) | `REACHABILITY.md §5.3` |
| `C-GROWGRID` | DC 블록 0.1333→0.0667 · 실효 억제 반경 0.2667→**0.2000**(의도한 0.1333 아님) · refine 격자 | `REACHABILITY.md §2.3` |
| `E-MAPDEPTH` | 조건화 맵 `OM_G` 전체 ⟹ **제안이 바뀐다** | `d3_beamform_net.py:38` |
⟹ ⛔**「단일 상수」라고 인쇄하지 마라.** 정본 표현 = **「단일 서브시스템의 절대 스케일」**.

### 1.2 ⛔이 배치가 **재지 않는 것**
`NK=64`(학습 위치임베딩에 구워짐) · 밴드 **재학습** · refine 창 단독 · 개구/픽셀 비 단독
(정보량 교락) · `max_n`(비직교) · 학습 코퍼스 분포. 사유는 `REACHABILITY.md §4`.

---

## §2 표본 — **두 층** (⚠노출 처리를 여기서 못박는다)

### 2.1 PRIMARY 층 = `T11` 의 **같은 24블록** (n = 24)
출처 = `../T11_EXTACC_2026-09-03/sample_t11.json`,
md5 **`56a0adc053b046081d538aa15b381766`**(890 B, 24항목) — `AMENDMENT_2 §8` 이 박은 값.
⛔**복사하지 않는다** — 러너·분석기가 그 경로를 **읽는다**(손복사 금지 규율).
반복 순서·시드는 `T11` 과 **축자 동일**: `SEED = 20260904`, 블록 `fi`(1-based, json 순서)에
대해 `seed = SEED + fi`. `J0 = 100` · `WIN_S = 40.0` · `DENSITY = 200` · `DEPTH = 400.0` ·
`GT_CONST = 0.5` · 채점 `4σ` · `t` 원점 0 · 좌표 재중심화 — 전부 `T11` 규약 그대로.

**★왜 재사용하는가**: 이 배치의 질문은 *"**그 24블록에서 측정된** `RESCALE` 이득이 어느 상수에서
오는가"* 다. 귀속은 **측정된 효과에 대해서만** 의미가 있고, 짝 비교(블록 내)와 상한(`RESCALE`)이
같은 블록에 있어야 성립한다. 신선 표본에서는 상한부터 다시 세워야 한다.

**⚠노출 공시 (전문)**
1. 이 24블록의 `SHIP`·`RESCALE`·`ZERO`·`ORACLE` 값은 **이미 알려져 있다**(`T11` 동결 결과).
   ⟹ 그 네 팔에 대한 이 배치의 모든 진술은 **기술통계**이며 **새 증거가 아니다.**
2. **9개 처치 팔은 이 코퍼스의 어떤 블록에서도 한 번도 실행된 적이 없다.** PRIMARY 검정의
   결과는 미지다.
3. 축 목록은 `T11 AMENDMENT_2 §6a`(**결과 0행 시점 동결**)에서 왔고, 처치값 규약(÷s)은
   기계적이다. ⟹ 축 선택은 결과에 노출되지 않았다.
4. ⚠그러나 **`s=2` 라는 선택 자체**는 `T11` 사전감사가 s∈{1,2,3} 를 본 뒤 고른 값이고,
   `RESCALE` 이 그 `s` 에서 24/24 를 냈다는 것을 나는 **안다**. ⟹ 처치값 규약이 그 `s` 를
   물려받는다. **이 배치는 그 사실을 안 상태에서 설계됐다.**
5. ⟹ **등급**: PRIMARY 는 `CONFIRMATORY`, **단 「노출된 표본 위의 확증」이라고 명시**한다.
   ⛔*"독립 확증"*·*"blind"*·*"out-of-sample"* 자구 **금지**.

### 2.2 SEC-R 층 = **held-out 보류 블록** (n = 28, ★`EXPLORATORY`)
정의(⛔손으로 목록을 적지 않는다 — **분석기가 namelist 에서 계산한다**):
```
sorted() 한 아카이브 .mat 목록의 인덱스 60–119  (= Data/mixed_data2, 60개)
  − sample_t11.json 의 24개
  − 사전감사 스모크 8개 (인덱스 60,63,65,68,71,74,77,80 = T11 gates_t11.AUDIT_SMOKE_IDX)
= 28개
```
팔은 **4개만**: `ZERO` · `SHIP` · `RESCALE` · `A-MAPSCALE`(PRIMARY 팔).
⚠**공시**: 이 여집합은 `Hs` 가 **위로 치우쳤을 것으로 예상**된다(파일명 규약이
`angle{10..60} × h{0.5..5.0}` 이라는 **추론** 하에 `Hs≥2.0` 이 25/28). ⟹ **같은 조건의
복제가 아니라 이동한 체제에서의 탐색**이다. 실제 구성은 `G-HELDOUT` 이 되읽어 기록한다.
⚠`T11` `PREREG §4` 가 `Tp` 전 파일 10.0 s 라 적었으나 그것은 **그 24개**의 관찰이다 —
`G-HELDOUT` 이 28개의 `Tp`(또는 그 대리량)를 기록한다.

### 2.3 스모크 (상시지시)
`T11 gates_t11.SMOKE_FILES` 의 **2파일**(`angle10h0.5`, `angle10h2.0` — 이미 노출된 감사 파일)
에서 **13팔 전부**를 돈다 = 22 추정기 런 ≈ 4분. ⛔프리즌 표본(24·28) **미접촉**.
스모크가 확인할 것: ①모든 게이트가 발화 가능한가 ②가장 비싼 팔의 단가 ③§6 `G-PROPID` 의
**전제**(아래 2.4) ④개입 계수기가 팔마다 0 이 아닌가.

### 2.4 ★★RNG 규약 — **팔을 더하면 앵커가 흔들릴 수 있다** (결과 전 조사, 런 0회)

`T11` 은 블록마다 `ZERO → ORACLE → SHIP → RESCALE` **네 팔**을 한 프로세스에서 돌렸다.
`T12` 는 그 사이에 9팔을 더한다 ⟹ **전역 RNG 스트림 이력이 달라질 수 있다.**

소스 조사 결과(파일:줄):
- **torch 전역**: `cd.sample()` 이 내부에서 `torch.manual_seed(seed)` 를 부른다
  (`scripts/d3_beamform_cond_diffusion.py` `sample()` 3번째 줄; `sequential_update.py:2170-2172` 주석이 축자로 확인한다) ⟹ **팔마다 제안 시점에 재시드**된다
  ⟹ 성장의 `torch.rand` 위상(`sequential_update.py:1765`)까지 **팔 내부에서 결정적**. ✅안전.
- **numpy 전역**: 이 경로에서 **전역 numpy 를 쓰는 곳은 성장 패딩 하나**뿐이다
  (`sequential_update.py:1751-1754` `np.random.randn`/`np.random.uniform`). 관측 표집은
  `np.random.default_rng(SEED+fi)`, 제안 서브샘플은 `default_rng(0)` 로 **지역 Generator** 라
  전역을 건드리지 않는다. `_seed_globals`(`:1985-1988`)는 앙상블 멤버 루프에서만 불리는데
  `learned_ensemble_k = 1` ⟹ `k_eff <= 1` 조기 반환(`:2151-2153`)이라 **실행되지 않는다**.
  ⟹ ★***성장 패딩이 한 번도 발화하지 않으면 전역 numpy 스트림은 전혀 진행하지 않고,
  팔을 몇 개 더하든 `SHIP`·`RESCALE` 은 `T11` 과 비트 동일하다.*** 발화하면 흔들린다.

**결과 전 고정**
1. 블록 내 팔 순서를 `ZERO → ORACLE → SHIP → RESCALE → (9 처치)` 로 **못박는다**
   (`T11` 의 접두사를 그대로 보존).
2. ★**`G-RNGSTREAM` 신설**: 팔마다 `np.random.get_state()` 의 md5 를 **직전/직후에 되읽어**
   행에 기록한다. 전 행에서 before==after 이면 **전역 스트림이 한 번도 진행하지 않았다**는
   기계 증명이고 `G-ANCHOR` 의 비트 실패는 **다른 원인**을 뜻한다.
3. `G-ANCHOR` 는 §6 에 쓴 대로 **2단**(BIT / 부호 재현)이고, 이 분기 역시 **지금** 등록한다.
⛔전역 스트림을 **재시드하지 않는다** — 재시드하면 그것이 곧 `T11` 과 다른 팔이 된다.

### 2.5 ★결과 전에 고정하는 **분기 하나** — GPU 재현성
`T11` 은 제안을 **cuda 에서** 돌렸다(행의 `proposal_device == "cuda"`, 24/24).
⟹ `G-PROPID`(§6)는 *"같은 시드의 제안이 비트 재현된다"* 를 전제한다.
**스모크가 같은 블록에서 `SHIP` 을 두 번 돌려 `proposal_md5` 를 비교한다.**
- 두 번이 같으면 → `G-PROPID` 는 **차단 게이트**다.
- 다르면 → `G-PROPID` 는 **비차단 기록**으로 강등되고, 그 사실과 스모크 출력을
  `AMENDMENT_1_PREDATA.md` 에 **결과 0행 시점에** 박는다.
⛔**이 분기는 지금 등록한다** — 결과를 보고 정하지 않는다(선례: `M4` 의 사후 완화 공시).

---

## §3 ⚠**「회수율」로 읽어도 되는 축과 안 되는 축**

`RESCALE` 은 자료의 `k` 를 ×2 한다. 자료를 고정한 채 등가인 것은 **[rad/m] 차원 상수의 ÷2** 뿐이다.
- **회수 축(7)**: `A-MAPSCALE`·`A-MAPCELL`·`A-BAND`·`B-DEDUP`·`C-MINDK`·`C-GROWGRID`·`D-VPLR`
  ⟹ 이들에 대해서만 `ρ`(§8.1) 를 인쇄한다.
- ⚠**회수 축 아님(2)**:
  · `E-MAPDEPTH` — `RESCALE` 은 맵 수심을 바꾸지 않는다. `k'd` 가 커져 정합필터가 **자동으로**
    심해에 가까워질 뿐이다(`AMENDMENT_2 §6b`: 정합이득 0.66→1.000). 처치는 그 **효과만** 재현한다.
  · `F-STEEP` — `ε=|k|·|w|` 는 **무차원**이라 `s` 로 스케일되지 않는다. `RESCALE` 세계에서
    클램프는 오히려 **더 세게** 문다 ⟹ ⛔**클램프 완화는 `RESCALE` 이득의 원천일 수 없다.**
    이 팔은 *"SHIP 의 실패 중 얼마가 클램프 자해인가"* 라는 **절제**다.
⟹ ⛔`E-MAPDEPTH`·`F-STEEP` 에 `ρ` 를 인쇄하지 마라. 승리해도 *"`RESCALE` 이득의 X%를 회수"* 로
쓰지 마라. 그 둘의 승리는 **다른 문장**이다(§4 의 라이더).

---

## §4 결정표 (결과 전 고정) — **이김 다리만 검정 · 음성 라벨은 이름이 「미검출」**

### 4.1 검정
```
PRIMARY   : W(A-MAPSCALE < ZERO)  >= 17     n=24, α=0.05 단측 부호검정, p(≥17|.5)=0.031957
SEC family: 나머지 8팔 각각 W(arm < ZERO)    Holm, m=8
            ★Holm 이 "적어도 하나를 기각" 하는 사건은 정확히  max_i W_i >= 19
              (첫 단계 α/8 = 0.00625, p(≥19|24,.5) = 0.003310)
            ⟹ 라벨은 첫 단계만으로 결정되고, 이후 단계는 **어느 축인지**만 정한다.
앵커      : SHIP·RESCALE 은 검정하지 않는다 — G-ANCHOR(§6) 가 T11 재현을 요구한다.
```
`PRIMARY` 팔이 SEC family 에 **중복해서 들어가지 않는다**(m=8: `A-MAPCELL`·`A-BAND`·`B-DEDUP`·
`C-MINDK`·`C-GROWGRID`·`D-VPLR`·`E-MAPDEPTH`·`F-STEEP`).

### 4.2 라벨 (4개가 4조합을 **전부** 덮는다 — `UNDECIDED` 없음)
```python
p_win   = W_MAPSCALE >= 17
any_sec = max(W_i for i in SEC8) >= 19

 p_win and not any_sec  ->  'PROPOSAL-SCALE-CARRIES'
 p_win and     any_sec  ->  'MULTIPLE-AXES-CARRY'
not p_win and any_sec   ->  'OTHER-AXIS-CARRIES'
not p_win and not any_sec -> 'NO-SINGLE-AXIS-DETECTED'
```
- ★**`NO-SINGLE-AXIS-DETECTED` 는 「미검출」이다.** ⛔*"인수분해되지 않는다"*·*"상호작용이다"*·
  *"어느 상수도 원인이 아니다"*·*"도달 불가"* 로 읽는 것을 **금지**한다.
  선례 = `M8b` 가 `GAP-NOT-FACTORIZABLE` 을 **개명**하고 *"⟹ 상호작용"* 을 철회한 사건,
  그리고 `T5-ENS` 개정문의 *"「검출 불가」이지 폐쇄가 아니다"*.
- ★**라이더(의무 병기)**: `OTHER-AXIS-CARRIES` 가 `E-MAPDEPTH` 나 `F-STEEP` **만으로** 발생하면
  라벨 옆에 반드시 적는다 — ***"이긴 축은 회수 축이 아니다(§3). 이것은 `RESCALE` 이득의 귀속이
  아니라 별개의 결함 지목이다."***
- `MULTIPLE-AXES-CARRY` 는 ⛔*"두 축이 함께 작동한다"* 가 아니라 **"둘 이상이 각자 문턱을
  넘었다"** 로만 읽는다(짝 비교가 아니다).

### 4.3 ⛔차단 조건 (라벨 인쇄 금지)
`G-*` 차단 게이트 중 하나라도 FAIL 이면 `verdict['label'] = None`, `BLOCKED`.
⟹ 인쇄 벨트: `label = f(...) if (not blocked and all(ok for _, ok, _ in GATES)) else None`
(`T11 AMENDMENT_3 §6` 이 합성 무력화로 뚫었던 구멍을 처음부터 막는다).

---

## §5 ★사전등록 예측과 검정력 — **믿음 선언이 아니라 계산**

### 5.1 등록하는 `q` (= 팔이 한 블록에서 `ZERO` 를 이길 참 확률)
```
q(A-MAPSCALE) = 0.75      ← PRIMARY
q(A-MAPCELL)  = 0.50   q(A-BAND)    = 0.55   q(B-DEDUP)    = 0.55
q(C-MINDK)    = 0.50   q(C-GROWGRID)= 0.55   q(D-VPLR)     = 0.50
q(E-MAPDEPTH) = 0.60   q(F-STEEP)   = 0.50
```
**사유(짧게, 그리고 각각 틀릴 수 있음을 명시)**
- `A-MAPSCALE` 0.75 — `AMENDMENT_2 §6a` 가 *"학습 분포 **밖**이면서 재척도가 안으로 옮기는
  유일한 절대량은 **개구/픽셀 비**"* 라 결론했고 그 비는 정확히 이 팔이 2배로 만든다
  (0.53칸 → 1.06칸, 학습 코퍼스 1.37–1.65). ⚠단 **그 진단은 1파일 관찰**에 기댄다.
- `A-MAPCELL` 0.50 — 픽셀↔원자 대응이 깨져 디코딩이 2배 틀린다(`REACHABILITY.md §2.2`).
  ★**나는 이 팔이 널일 것이라고 결과 전에 적는다.**
- `C-MINDK`·`C-GROWGRID` 0.50/0.55 — 성장 분기가 `n_gaussians >= max_n` 에서 조기 반환하고
  `T11` `SHIP` 은 24블록 중 18블록에서 원자 30개(=`max_n`)로 끝났다 ⟹ **레버가 약할 수 있다**.
- `D-VPLR` 0.50 — 배포 lr 2e-3 는 `k_p≈0.04` 의 **5%** 라 애초에 분해능 제약이 아니다.
  ★**나는 이 축이 널일 것이라고 결과 전에 적는다**(§6a 목록에 있었으나 물리적으로 무해해 보인다).
- `E-MAPDEPTH` 0.60 — 정합이득 0.66→1.00 은 실질적이지만 **제안의 위치**를 고치는지는 미지.
- `F-STEEP` 0.50 — `T11` 이 `SHIP` 19회 발화를 봤지만 방향도 크기도 측정된 적 없다.

### 5.2 라벨별 확률 (등록 예측 하, 전수 계산)
| 라벨 | 확률 |
|---|---|
| **`PROPOSAL-SCALE-CARRIES`** | **0.6986** |
| `MULTIPLE-AXES-CARRY` | 0.0676 |
| `OTHER-AXIS-CARRIES` | 0.0206 |
| `NO-SINGLE-AXIS-DETECTED` | 0.2132 |
| 합 | **1.0000** |
(`P(W≥17|24,0.75) = 0.7662` · `P(any_sec) = 1 − Π(1−P(W_i≥19|q_i)) = 0.0883`)

### 5.3 귀무(전부 `q=0.5`) 아래 오라벨률
```
PROPOSAL-SCALE-CARRIES 0.03112 · MULTIPLE-AXES-CARRY 0.00084
OTHER-AXIS-CARRIES     0.02530 · NO-SINGLE-AXIS-DETECTED 0.94274
```
`NO-SINGLE-AXIS-DETECTED` 가 0.94 인 것은 **설계대로다** — 이름이 「미검출」이기 때문.
(`T11 AMENDMENT_2 §2` 의 교훈: 옛 라벨의 죄는 확률이 높은 것이 아니라 **그 확률에 양성 주장
문구를 붙인 것**이었다.)

### 5.4 민감도 — 선언이 하중을 지지 않는다
| `q(A-MAPSCALE)` | 0.60 | 0.65 | 0.70 | **0.75** | 0.80 | 0.85 | 0.90 |
|---|---|---|---|---|---|---|---|
| `P(PRIMARY win)` | 0.192 | 0.358 | 0.565 | **0.766** | 0.911 | 0.980 | 0.998 |

| SEC 한 축의 참 `q` | 0.60 | 0.70 | 0.75 | 0.80 | 0.85 | 0.90 |
|---|---|---|---|---|---|---|
| `P(W≥19)` (그 축이 명명될 확률) | 0.040 | 0.229 | 0.422 | 0.656 | 0.861 | 0.972 |

---

## §6-A MDE — ★**병기 의무**

```
PRIMARY  α=0.05 단측 · 임계 W ≥ 17 · n=24
   검정력 0.50 → q ≥ 0.685      검정력 0.80 → q ≥ 0.760   ← ★인용 시 병기 의무
   검정력 0.90 → q ≥ 0.7955
   ⛔기각 불가 대역  W = 13..16 = 승률 54.2%~66.7%
      p = 0.4194 / 0.2706 / 0.1537 / 0.0758
      ★16/24 = 66.7% 도 p=0.0758 로 **어떤 자료에서도 기각 불가**

SEC family (Holm 첫 단계) α=0.00625 · 임계 W ≥ 19 · n=24
   검정력 0.80 → q ≥ 0.8332     ← ★음성 SEC 인용 시 병기 의무
   ⟹ 참 승률 0.70 인 축은 **77% 확률로 놓친다.**
   ⛔`NO-SINGLE-AXIS-DETECTED` 를 *"다른 축은 아무 효과가 없다"* 로 읽는 것 **금지**.

SEC-R (held-out) α=0.05 단측 · 임계 W ≥ 19 · n=28
   p(≥19|.5) = 0.04358 · 검정력 0.80 → q ≥ 0.7304
   ★등급 EXPLORATORY (체제 이동, §2.2)

층(Hs 2층, n=12) 최소 정수 10/12 (p=0.01929) · MDE q ≥ 0.8693
   ⟹ **사실상 층별 검정력 없음** ⟹ 층은 **기술통계**이고 라벨을 만들지 않는다.
```

---

## §6 차단 게이트 — ★**각 게이트가 「무엇을 어디서 되읽는가」**

⛔전부 **분석기가 직접 계산**한다. 러너가 쓴 리터럴을 분석기가 같은 리터럴과 비교하면 **항진**이다
(`T11 AMENDMENT_2 §7` 이 `G-ENV`·`G-CAP` 에서 겪은 것).

| # | 게이트 | **되읽는 대상 / 출처** | 통과 조건 |
|---|---|---|---|
| ★★`G-KNOB` | 팔 설정 **직후 살아있는 모듈·객체 상태**: `d3_beamform_net.KLIM`, `KX_G[0,0].item()`, `OM_G[0,0].item()`, `d3_beamform_net.DEPTH`, `LI._load_cond` 가 반환한 `stats['K_LIM']`, `initializer.MIN_DK`, `upd.max_steepness`, `upd.config['learned_dedup_dk']`, `info['varpro']['lr']`(`varpro.py:152` 반환), 성장 래퍼가 본 `kx_range` | 각 행에서 **정확히 그 팔의 노브만** 배포값과 다르다. 나머지 8개는 배포값과 **비트 동일** |
| ★★`G-RESTORE` | 팔 종료 **후** 같은 9개를 다시 되읽어 **배포값 복귀** 확인 | 위반 0 (⛔프로세스 전역 오염 = v1 BLOCKER-2 부류) |
| ★★`G-ANCHOR` | 이 배치의 `SHIP`·`RESCALE` nRMSE **48값** vs `../T11_EXTACC_2026-09-03/results_t11.jsonl` 의 같은 (블록, 팔) | `max|Δ| ≤ 1e-9` 이면 `BIT`. 아니면 **부호 재현 필수**: `W(RESCALE<ZERO) ≥ 17` **및** `W(SHIP<ZERO) ≤ 12`. 둘 다 실패면 **BLOCKED**(불일치 자체가 소견) |
| ★★`G-EFFECT` | 팔별 **개입 계수기**: `proposal_md5`(raw 제안) · `n_sel`(선택 원자수) · `n_grow_calls`/`n_grow_atoms`(래핑한 `model.add_gaussians`) · `vp_k_shift_median`(`varpro.py` 반환) · `vp_clamp_hits` | 각 처치 팔이 **SHIP 과 비트 동일한 행이 24/24 는 아니다** ⟹ 그러면 `NO-OP` 로 보고(널 아님) |
| ★★`G-PROPID` | 팔별 `proposal_md5` vs 같은 블록 `SHIP` 의 것 | **제안 이전 팔**(`A-MAPCELL`·`A-BAND`·`A-MAPSCALE`·`E-MAPDEPTH`) = **전건 상이**. **제안 이후 팔**(`B-DEDUP`·`C-MINDK`·`C-GROWGRID`·`D-VPLR`·`F-STEEP`) = **전건 동일**. ⚠§2.5 분기 적용 |
| ★★`G-RNGSTREAM` | 팔 **직전/직후**의 `np.random.get_state()` md5 (§2.4) | 행마다 기록. **전 행 before==after** 이면 전역 numpy 스트림 미진행 ⟹ `G-ANCHOR` 비트 기대가 구조적으로 성립 |
| ★★`G-GROWFIRE` | `n_grow_calls` 합 (SHIP 24행) | 기록. **`SHIP` 발화 행이 6 미만이면** `C-MINDK`·`C-GROWGRID` 는 결과에 관계없이 **`NOT-EXERCISED`** 로 보고(⛔「널」로 쓰지 않는다) |
| ★★`G-NOLEAK` | 붙인 `gt_field` 의 `std`·평균을 **프레임 객체에서**, `initialize()` **이후에도** 재측정 | std 0 · 값 == `GT_CONST` · 위반 0 |
| ★★`G-ATOMS` | 팔별 `proposal_md5` 24개 | 팔마다 **전건 상이**(config 캐시 누출 탐지, 9+2 팔 전부) |
| ★★`G-ARM` | `upd.arm_provenance()` | `requested='auto'` ∧ `resolved_init_method=='learned'` ∧ `real_domain_gate_fired==False` ∧ `ensemble_k == DEFAULT_LEARNED_ENSEMBLE_K`(임포트) — **전건** |
| ★`G-SHIPPIN` | `run.DEFAULT_CONFIG` 와 `sequential_update` 의 단일출처 상수를 **임포트해** 대조 (`LAMBDA_L1_DEFAULT`·`INIT_METHOD_DEFAULT`·`VARPRO_STEPS_DEFAULT`·`LEARNED_INIT_ITERS_DEFAULT`·`AUTO_SWITCH_THRESH_DEFAULT`·`K_SNAP_DK_DEFAULT`·`DEFAULT_LEARNED_ENSEMBLE_K`·`M_OBS_CAP`·`MIN_DK`·`VP_LR_DEFAULT`·`KLIM`·`NK`·`DEPTH`·`K_LIM`) | 전부 사전등록 핀과 일치 (⛔손복사 0) |
| ★`G-SAMPLE` | **분석기가 `sample_t11.json` 을 직접 읽고** 아카이브 namelist 를 **직접 연다** | 24행 == 그 목록 · 실존 24/24 · 사전감사 스모크 8 과 교집합 **0** |
| ★`G-HELDOUT` | 분석기가 namelist 에서 **여집합을 계산** | 28개 · `sample_t11` 과 교집합 0 · 스모크 8 과 교집합 0 · 구성(각도·Hs) 기록 |
| ★`G-N` | 행 수 | PRIMARY 층 팔별 **24행 · 총 312행**(13팔) · SEC-R 팔별 **28행 · 총 112행**(4팔). ★문턱 17 은 **n=24 전용**, 19 는 **n=28 전용** |
| ★`G-FINITE` | 모든 행의 `nrmse` | 전건 `isfinite` (⛔`NaN < x` = False ⟹ 조용한 패배) |
| ★`G-TORIGIN` | 관측 배열에서 `float(tt.min())` | 전건 0.0 |
| ★`G-DENSITY` | 만들어진 프레임에서 되읽은 `n_offered`·`n_frames`·`dt`·`tq`·`e_shape` | `n_offered == 200*40` · `n_frames == 40` · `|tq − 39·dt| ≤ 1e-9` · `e_shape[0] ≥ J0+40` |
| ★`G-RESCALE` | `x_used/x_orig`·`t_used/t_orig` 의 중앙값 | `a = 1/2` · `|b − √a|/√a ≤ 1e-6` (ω=√(gk) 보존) |
| ★`G-SIZE` | `os.path.getsize(ZIP)` **분석기가 직접** | `== 12828148812` |
| `G-ENV` | `torch.get_num_threads()` 되읽기 · **행의 `seed` 집합** 대조(`{SEED+1..SEED+24}`) · 임포트 경로 md5 **항목별 재계산** · 체크포인트 **sha256 2개** | 위반 0 |
| `G-FREEZE` | §7 의 측정면/분석면 3중 대조 | 위반 0 |
| `G-BLOCK` | `blocked_by()` 자가진단(합성 입력으로 **긴 이름 우선** 매칭 확인) | ⛔`T11` 이 `blocked_by()` 가 항상 빈 리스트를 내는 버그를 겪었다 — 자가진단 필수 |
| `G-COV` | 게이트가 실제로 자료를 봤다(`rows > 0`) | 차단 목록 안 |

⚠**구조적으로 참인 것을 이름에 표기**: `G-RESCALE`·`G-TORIGIN`·`G-NOLEAK` 의 일부 항은
러너 소스를 고치지 않는 한 실패할 수 없다(**변화 탐지기**). `T11 AMENDMENT_3 §10` 의 관례대로
그렇게 **이름에 적어** `gates_pass` 를 오독하지 않게 한다.

---

## §7 동결 — **측정면 / 분석면**

| 면 | 파일 | 대조 |
|---|---|---|
| **측정면** | `consts_t12.py`(★신설 — 상수 단일출처) · `run_t12.py` · **임포트 경로 12** · **체크포인트 2** · (참조) `../T11_EXTACC_2026-09-03/sample_t11.json` · `PREREG.md` · `REACHABILITY.md` | 현재 == `meta['code_md5']` == **`FROZEN_md5.txt`(결과 0행 동결)** 의 **3중 대조** |
| **분석면** | `gates_t12.py` · `analyze_t12.py` · 사후 개정문 | 현재 == `FROZEN_md5_ANALYSIS.txt` |

★**`T11` 대비 개선**: `T11` 은 러너가 `gates_t11.py` 를 상수 단일출처로 임포트했는데 개정문 3 이
그 파일을 고쳐 **측정면/분석면 경계가 그 파일 안에서 갈렸다**. `T12` 는 **상수(`consts_t12.py`)와
판정(`gates_t12.py`)을 파일로 분리**해 그 모호성을 없앤다.

**임포트 경로 12** (⛔`T11` 은 8이었고 `initializer.py` 가 **빠져 있었다** — 이 배치는 `MIN_DK` 를
움직이므로 필수):
```
wavesplat/optimizer/sequential_update.py   wavesplat/optimizer/learned_init.py
wavesplat/optimizer/initializer.py  ★NEW   wavesplat/optimizer/varpro.py
wavesplat/model/spectral_gaussian.py       wavesplat/model/dispersion.py
wavesplat/model/tnse.py             ★NEW   wavesplat/run.py
scripts/d3_beamform_net.py                 scripts/d3_beamform_cond_diffusion.py
scripts/d3_setdiff_prior.py         ★NEW   scripts/d3_setdiff_posterior_b2.py  ★NEW
```
**체크포인트 2**: `results/direction3/beamform_cond_diffusion_mixed_v1.pt`
(sha256 **핀** = `636c72a797ee6d0b725449293d5cd230dbd64c8eb03e345da4561b0c265239d9`,
`learned_init.py:60`) · `results/direction3/setdiff_prior.pt`(sha256 = 러너가 **런 시작 전에**
계산해 `meta` 에 기록, 이후 대조).
⚠md5 는 **런 시작 전과 종료 후 둘 다** 잰다(`T11` 초판이 끝에만 재서 도중 편집이 「그 런의
상태」로 기록될 뻔했다).

---

## §8 기술통계 (⛔라벨을 만들지 않는다)

### 8.1 회수율 `ρ` — **회수 축에만**
```
ρ_T(b) = ( nRMSE_SHIP(b) − nRMSE_T(b) ) / ( nRMSE_SHIP(b) − nRMSE_RESCALE(b) )
```
`T11` 동결 재집계(런 0회): 분모는 **24/24 에서 양수**, 최소 0.01639, 중앙 0.04021
⟹ **정의역이 안전하다**. 중앙값과 사분위를 보고한다.
⛔`ρ` 를 팔 간에 **더하지 마라**(합이 1 이 될 이유가 없다 — 선례: `\BOneRel`·`\BSixteenRel` 을
산술 결합해 *"절제가 이득의 100%를 지운다"* 로 읽힌 `P8` MAJOR).
⛔`E-MAPDEPTH`·`F-STEEP` 에는 인쇄하지 않는다(§3).

### 8.2 ★매개 열 (**추가 런 0회**, 행에 실린 되읽기만)
팔·블록마다: `kp_true`(표적 FFT 최대 파워 빈의 `|k|`, ORACLE 이 이미 계산) ·
`kmed_raw`(raw 제안 원자 `|k|` 중앙) · `kmed_sel`(선택 원자) · `kmed_final`(설치·VARPRO 후) ·
`n_raw`/`n_sel`/`n_atoms_installed` · `ratio = kmed_sel / kp_true`.
★`T11` 사전감사가 인용한 *"학습 제안의 원자가 참값의 **16배** 파수"* 를 **팔별로 재현**한다.
⛔**이것은 기술통계다** — *"기제를 지목했다"* 로 쓰지 마라(방향·크기 미검정).

### 8.3 층 (`Hs < 2.0` / `≥ 2.0`, 12/12) · 각도 6클러스터
전부 **기술통계**. `n=12` 는 MDE `q ≥ 0.869`, 각도 `n=4` 는 만장일치도 `p=0.0625 > 0.05`
⟹ ⛔**클러스터 주장 불가**.

### 8.4 다중비교 공시
PRIMARY 1 + SEC family 8(Holm) = **보정 대상 9개**. `SEC-R`·층·각도·`ρ`·매개열은
**무보정 기술통계**이며 라벨을 만들지 않는다. 이 문장을 `verdict['disclosures']` 에 **기계로** 넣는다.

---

## §9 ⛔범위와 금지 (`verdict['disclosures']` 에 기계로 삽입)

1. ⛔**`RESCALE` 을 실제 성능으로 인용 금지** — 단위를 바꾼 사고실험이지 배포 구성이 아니다.
2. ⛔**`ORACLE` 은 추정기가 아니다**(참 파수를 받는다) — 정보 존재의 상한.
3. ⛔**처치 팔은 전부 배포 구성이 아니다.** 어떤 팔이 이겨도 **배포 권고가 아니다** —
   배포는 별도 사전등록이다(선례: `S-ENS` §5 가 결과 전에 *"필요조건 하나"* 로 고정).
4. ⛔**「단일 상수」 금지** — §1.1 의 동반 이동 때문에 정본 표현은 **「단일 서브시스템의 절대
   스케일」**이다.
5. ⛔**`E2` 와 대각선 비교 금지 · 저들 논문 수치와 비교 금지 · solver 미명시**
   (`T11 PREREG §6` 의 의무 공시 — `T11 AMENDMENT_3 §1` 이 그것을 **삭제**로 집행한 사고를
   재발시키지 않기 위해 여기 축자로 옮긴다).
6. ⚠**예산 비대칭 공시**: 우리 코퍼스는 프레임당 관측을 캡(400)으로 잘라 쓰고, 이 코퍼스는
   자르지 않는다 ⟹ *"같은 관측 수"* 라도 **뽑은 모집단이 다르다**(`T11 NOTE §6.1`).
7. ⚠**`Tp` 가 (PRIMARY 층에서) 전 파일 10.0 s** ⟹ 이 배치는 **한 주기 조건**의 진술이다.
8. ⚠**완전 blind 아님 · 노출 표본**(§2.1 전문) ⟹ *"독립 확증"*·*"blind"*·*"out-of-sample"* 금지.
9. ⚠**조건화 빔포머 `DEPTH=50` 하드코딩**(`d3_beamform_net.py:30`)은 config `depth=400` 이
   닿지 않는 곳이다. 이 배치는 `E-MAPDEPTH` 로 **재기만** 하고 배포 기본값을 바꾸지 않는다.
10. ⛔**성장 발화가 희소하면 `C-*` 두 축은 `NOT-EXERCISED`** 이지 널이 아니다(§6 `G-GROWFIRE`).
11. ⛔**`D-VPLR` 이 널이면 「no-op」과 구별**해야 한다 — `info['lr']` 되읽기와
    `vp_k_shift_median` 이 그 구별의 근거다(`REACHABILITY.md §5` 항목 5: 기본 인자 바인딩 때문에
    순진한 개입은 조용히 no-op 이 된다).
12. ⛔**`proposal_device`**: `propose_atoms` 는 device 를 config 에서 받지 않고 스스로
    `cuda if available` 로 고른다 ⟹ 행의 되읽기를 믿고 리터럴을 믿지 마라. ⛔`wall_s` 를
    팔 간 비교에 쓰지 마라(BLAS 스레드가 팔마다 다르게 쓰인다).

---

## §10 비용 · 실행 순서

**단가**(`T11` 실측 재집계, 런 0회): `SHIP` 벽시계 **중앙 8.95 s · 최대 12.6 s**,
`RESCALE` 중앙 10.15 s, `ORACLE` 0.2 s, `ZERO` 0 s. ⟹ 추정기 1런 ≈ **10 s**.

| 층 | 블록 | 팔 | 추정기 런 | 행 | 예상 시간 |
|---|---|---|---|---|---|
| 스모크 | 2 (노출 파일) | 13 | 22 (+재현성 2) | 26 | **≈ 4분** |
| PRIMARY | 24 | 13 | **264** | 312 | **≈ 46분** |
| SEC-R | 28 | 4 | **84** | 112 | **≈ 15분** |
| **합** | | | **372** | **424** | **≈ 65분** |
(+ 아카이브 읽기 24+28 파일. `T11` 은 24블록 2추정기팔에 총 8분이었고 그중 추정기가 ~8분
⟹ I/O 는 블록당 1–2 s 수준.)

**실행 순서**(⛔이 순서를 지킨다 — 2026-09-04 규율):
```
① PREREG.md + REACHABILITY.md 동결 (지금 = 결과 0행, 러너 0줄)
② consts_t12.py  (상수 단일출처, 측정면)
③ analyze_t12.py + gates_t12.py  ← ★러너보다 **먼저**
④ run_t12.py
⑤ FROZEN_md5.txt 생성 (결과 0행 시점)
⑥ 스모크 2파일 × 13팔 + §2.4·§2.5 분기 판정 → 필요하면 AMENDMENT_1_PREDATA.md
⑦ PRIMARY 24블록 → 분석기 → verdict
⑧ SEC-R 28블록 → 분석기 (EXPLORATORY 별도 절)
```
⚠**메모리**: `T11` 이 `predict_field` 에 평탄 격자를 넘겨 2.68억 점 → WSL 24 GB OOM 으로
**두 번 죽었다**. `T12` 러너는 `predict_field` 에 **축 벡터**를 넘긴다(`T11 run_t11.py:239-243`
규약 그대로). 그리고 배치는 `systemd-run --user --scope -q -p MemoryMax=… -p MemorySwapMax=0`
안에서 돈다.
⚠**CPU 직렬** — 다른 무거운 작업과 겹치지 않게 한다.

---

## §11 이 사전등록이 상속한 규율 (출처)

| 규율 | 출처 | 이 문서의 이행 |
|---|---|---|
| 이김 다리만 검정 · 음성 라벨은 이름이 「미검출」 | `T11 AMENDMENT_2 §1` | §4.2 |
| 예측 `q` 를 고정하고 **라벨 확률을 계산해 표로** | `T11 AMENDMENT_2 §2`(v1 폐기 사유) | §5.1–5.4 |
| MDE 를 결과 전에 계산 · **병기 의무** · 기각 불가 대역 | `T11 AMENDMENT_2 §3` | §6-A |
| 게이트는 **되읽기** — 러너 리터럴 대조는 항진 | `T11 AMENDMENT_2 §7` · `AMENDMENT_3 §2` | §6 열 2 |
| 동결을 **측정면/분석면**으로 분할 | `T11 AMENDMENT_3 §2` | §7 |
| 팔 정체는 `arm_provenance()` 되읽기 | `feedback` 2026-09-03 (`E1-EXTBASE` 사고) | §6 `G-ARM` |
| ⛔`{}` 빈 dict 으로 팔을 정의하지 마라 | `B-7 gates_b7.py:46` BLOCKER | §1 |
| 개입 기대값은 **단일출처** · 원자 카운터로 런 0회 확인 | `W4` 4중 기재 사고 · `W3` 설계 | §6 `G-EFFECT`·`G-GROWFIRE` |
| 효과를 **그것이 들어가는 양**이 아니라 **판정**에서 재라 | `B-2` 수심 17→30 · `T11 §9.1` 클램프 예측 반증 | §3 · `F-STEEP` 팔 |
| 「도달 불가」와 「지렛대 부족」을 섞지 마라 | `LEARNED_AUDIT_2026-08-30` | §6 `G-GROWFIRE` · `REACHABILITY.md §4` |
| 손복사 기준 상수 금지 | 「손복사 상수 사냥꾼」 | §6 `G-SHIPPIN` · §2.1(표본을 읽는다) |
| 사후 완화는 **결과 전에 분기를 등록**해서 피한다 | `M4 G-15` 공시 | §2.4 · §2.5 |
| 라벨 인쇄를 `blocked_by` **하나**에 매달지 마라 | `T11 AMENDMENT_3 §6` | §4.3 |
| 재작성은 기능을 조용히 떨어뜨린다(N−1) | `T11 AMENDMENT_3 §1`(하루 세 번) | §9-5 를 축자로 이식 |
