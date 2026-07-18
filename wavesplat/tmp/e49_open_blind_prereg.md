# E-49 — OPEN-ARENA BLIND (arena Y): does the blind≈0 negative generalize? (PRE-REGISTERED)

> Origin: E-48 페르소나 토론(2인)의 **killer-class 조건 ②**. 검사(prosecution)가 "blind≈0의
> 전수열거는 regime-퇴화 아레나 *안*에서만 돌았다 → '구성적 참'과 분리 안 됨"이라 주장.
> 코드 검증(2026-07-18): 모든 blind 실행(e42/e43b/e43c/e43e/e46)이 `build_sea_crosstp` =
> **arena X (shape 고정: Tp1=7·gamma=3.3·spread=30)** 에서만 돌았음. `e42_blind_recovery.py`
> SCOPE 주석: *"arena Y (open gamma/spread/Tp1 = misspecified) ... **Y is a separate run.**"*
> → **미실행 확정.** 이 스크립트가 그 arena Y run이다.

## 프레이밍 정정 (코드 근거)
- arena X는 **이미 bimodal cross-Tp** (`build_sea_crosstp`: r2~U(.2,.5)·d1·d2·Tp2 변동).
  oracle가 +0.104 회수 = **회수할 구조 실재.** 따라서 blind≈0은 "회수할 게 없어서"(구성적 참)가
  **아니라** 진짜 벽 (옹호 a fortiori). 단일모드 퇴화 아레나(`build_sea_c`, 3.55e-15)는 **conditioning
  실험(E-29/30)의 아레나로 별개**였음 — 검사가 둘을 융합했음.
- 그러므로 arena Y가 검증하는 것은 "구조 있냐"가 아니라 **"음성이 shape-오지정(현실적·broadband)
  아레나서도 버티느냐"**. arena Y = 인컴번트 net의 **훈련분포**(`d3_prior_net_data`: Tp~U(5,11)·
  gamma~U(1.5,6)·spread~U(15,45)·crossing aug) → net에 최유리·현실에 최근접. blind는 여기서
  **최불리**(고정-shape 사전). 이 비대칭이 곧 검정: net 홈그라운드서도 blind가 못 이기면 강한 음성.

## 질문
open-shape (misspecified) 아레나에서:
1. **prize가 존재하는가** — 참 파티션(shape 정확) 공급이 배포가능 net(cur)을 이기는가?
2. **blind가 그걸 회수하는가** — 고정-shape 전수열거(full_blind_grid)가 cur을 이기는가?

## 아레나 Y-open (frozen)
`build_sea_open(si)` = `build_sea_crosstp` 사본 + shape 개방 (per-sea 1회 추첨, 양 lobe 공통):
- `Tp1 ~ U(5,11)`, `gamma_J ~ U(1.5,6)`, `spread ~ U(15,45)`  (net 훈련분포)
- `Tp2 = clip(Tp1 * U(0.6,1.6), 4, 12)`, `d2 = d1 + sign*U(60,150)`, `r2 ~ U(0.2,0.5)`,
  `Hs ~ U(1.5,3)`, `U ~ U(0,1.5)*cp`, `nfrac ~ U(0.02,0.09)`, 100+100 comp, NO bound waves.
- **FRESH block** si=600..695 (n=96). 어떤 이전 실험도 안 쓴 구간. rng offset `29000+31*si`,
  jonswap seeds `26000+si`·`27000+si` (모든 기존 base와 분리). GT 1200 held-out px, corr metric.
- unimodal do-no-harm: seeds `3900+j` (n=24), 같은 open shape, r2=0.
- **S에 저장**: gamma, spread, Tp1, Tp2, d1, d2, r2 (oracle_reg·ens1 구성에 필요).

## 아레나 (frozen)
동일 solver `solve_with_grid` (SMULT=3, blur 0, GT-peek 없음), 동일 관측기하.

## 팔 (frozen)
| 팔 | 정의 | 역할 |
|---|---|---|
| `cur_grid` | net_gamma_map → solve_with_grid | **배포가능 baseline (arena Y서 in-distribution)** |
| `cur_dep`  | incumbent_field corr | 배포 파이프라인 |
| `ens1`     | lobe_grid(kp,[(Tp1,gamma,1)],[(mu_hat,spread,1)]) — **참 shape** 1-lobe @ 빔포밍방향 | 파티션 없이 우세방향만 |
| `oracle_reg` | (1-r2)·lobe(Tp1,gamma,d1,spread) + r2·lobe(Tp2,gamma,d2,spread) — **참 파티션+참 shape** | **prize 천장 (공급가치)** |
| `FULL`     | full_blind_grid (고정 gamma3.3/spread30, 952,560 가설, GT-free) | **핵심 blind 팔** |

## 사전등록 대조 + 판정 (frozen)
paired t-CI (동일 `paired()` 함수). 주 대조:
- **PRIZE(info, thesis-primary)**: `oracle_reg - ens1` (참-shape 파티션 지식이 참-shape 무파티션을 이기는가)
- **PRIZE(deploy, 문맥)**: `oracle_reg - cur_grid` (in-dist net 대비 잔여 배포 prize)
- **NET_ALREADY(문맥)**: `cur_grid - ens1` (in-dist net이 파티션을 이미 부분 포착하는가; E-29 const−net −0.0424)
- **BLIND**: `FULL - cur_grid`, `FULL - ens1`, `FULL - cur_dep`

> **★라벨 정정 (실행 전, 독립감사 반영 — smoke 데이터 아닌 E-29 사전지식 근거)**: arena Y는 net
> 훈련분포라 cur이 강한 in-dist baseline. `oracle_reg−cur`만으로 PRIZE를 키하면 "net이 이미 회수"를
> "prize 소멸/regime-specific"으로 **오분류**(E-29와 모순). ⟹ **PRIZE primary = `oracle_reg−ens1`**
> (정보가치)로 freeze; `oracle_reg−cur`는 배포-잔여 문맥. bar(+0.03·CI·win60)·아레나·팔·시드 전부 불변.

**판정 규칙 (frozen, n=96):**
1. **PRIZE_INFO**: `oracle_reg - ens1 ≥ +0.03` AND 95% CI excl 0. (파티션 정보가치 실재)
2. **BLIND_NULL (neg 일반화)**: `FULL - cur_grid` **및** `FULL - ens1` **둘 다 CI 상한 < +0.03**.
3. **주 확정 = PRIZE_INFO AND BLIND_NULL**: "open/현실 아레나서도 파티션 정보는 실재하나 (이 고정-shape)
   blind는 회수 못 함" = 음성 척추가 arena-X 특이 아님. E-48 killer-class ② 해소.
   (부기: `net_already`=`cur−ens1`>0 CI excl0 AND `oracle_reg−cur` CI∋0 이면 "net이 이미 회수 →
   net 대비 잔여 배포 prize 없음; 파티션 정보 자체는 실재" 로 명기 — regime-specific 아님.)
4. **BLIND_BREAKS**: `FULL - cur_grid` **또는** `FULL - ens1` 중 하나라도 `≥ +0.03` AND CI excl 0
   AND win ≥ 60% → blind가 arena Y서 회수 → 음성이 arena-X 특이 → **척추 재고.**
5. **INCONCLUSIVE-NOPRIZE**: `oracle_reg−ens1` CI∋0 → 참-shape 파티션조차 open서 가치 없음
   = 채널 가치 자체가 regime-specific. (이때만 "regime-specific" 라벨이 옳음.)
6. **INCONCLUSIVE/partial**: 그 외 (예: 분산 팽창으로 rule2 CI 미발화) → 라벨 강제 말고 CI 그대로 보고.
7. **do-no-harm** (unimodal n=24, **약한 floor·가드레일 전용**): `FULL - cur_grid ≥ -0.01`.

**R2 편향 명기**: FULL은 고정 gamma3.3/spread30, ens1/oracle_reg는 **참 shape** → FULL은 shape-불리 =
음성 확정 쪽 보수 편향 (거짓 BREAK 방지). 회의론 반론("blind를 wrong-shape로 불구화")은 리포트서 선제 인정.

## 스코프 캐비엇 (정직 명기)
- **FULL은 고정-shape 전수열거**(gamma3.3/spread30). arena Y서 이건 **shape-misspecified**이므로,
  "no learned blind can recover"의 **완전 정보-폐쇄가 아니라** "이 고정-shape blind 추정기가
  open서 일반화되나"의 **robustness 검정**이다. 더 강한 폐쇄(shape-marginalized blind, 또는
  5-dim θ CRB)는 여전히 미납 (E-48 keystone ③). 이 실험은 그 전 단계.
- n=96 (arc 교훈: n<96 헤드라인 금지). 발견블록 없음(전부 fresh, si=600..695).
- 소표본·단일블록 위 단정 금지. 결과는 CI와 함께, 사전등록 bar로만 판정.

## 비용/출력
~38 s/sea × (96+24) ≈ 75 min. GPU (RTX 4080S). Out: `results/direction3/e49_open_blind.json`.
독립 에이전트 사전검토(forking-paths·팔 공정성·scope) 후 full run.

---

# 결과 (n=96, 2026-07-18) + 이중검증 + arena-X disambiguation

## 원시 대조 (n=96, arena Y-open)
```
oracle_reg − ens1   +0.1244* CI[+0.087,+0.162] win 77/95
oracle_reg − cur    +0.0690* CI[+0.045,+0.093] win 74/96   ← prize over deployable net
cur_grid   − ens1   +0.0580* CI[+0.028,+0.088] win 58/95   ← net이 단일로브 이미 이김
FULL       − cur    +0.0125  CI[−0.012,+0.037] win 49/96   ← blind이 배포 net 못 이김 (n.s.)
FULL       − cur_dep +0.0239 CI[−0.002,+0.050] win 52/96   ← 배포 파이프라인 대비 한계선(p0.07, rule 미발화)
FULL       − ens1   +0.0709* CI[+0.035,+0.107] win 59/95   ← auto-BREAK 발화
do-no-harm uni F−cur +0.009  CI[−0.025,+0.044] (OK)
```
auto-verdict = **BREAKS** (frozen rule4가 FULL−ens1서 발화). **그러나 이중검증이 이를 뒤집음.**
(사전등록 대조 `FULL−cur_dep`은 위에 포함 — marginal p0.07, 어떤 rule도 미발화, 결론 불변.)

## 검증 1 — 독립 재도출 (원자료서 재계산)
- 6/6 대조 **5자리까지 정확 재현** — 산술 아티팩트 없음.
- **★분해**: `FULL−ens1(+0.071) = FULL−cur(+0.013 n.s.) + cur−ens1(+0.058)`. (이 등식 자체는 평균
  선형성에 의한 **항등식(tautology)** — 증거력은 등식이 아니라 **82/18 분해 비율**에 있음.) FULL이 ens1을
  이기는 것의 **82%가 cur−ens1 상속**(net이 단일로브 이김) = **blind 고유 아님.** blind의 배포-baseline
  대비 고유 회수 = **+0.013 (n.s.) ≈ 0.**
- **FULL−cur = 견고한 NULL**: median +0.004, win 51%, 평균 +0.012는 **117% 꼬리산**(상위5% 제거 시 음수).
- FULL−ens1 평균 +0.071도 **median +0.034로 ~2배 과장**(꼬리 40%).

## 검증 2 — arena-X disambiguation (★결정타, GPU 불필요, 기존 json에 **이미 저장**된 값)
**정직 정정(fable 검토)**: ens1 arm은 arena X서 **미측정이 아니었다** — `e43e`(si400-447 n=48)·`e42`(si100-163
n=64)의 rows에 ens1/FULL/cur이 저장돼 있었고, `FULL−ens1 = +0.055 (p=0.008)`은 E-49 실행 **전부터 디스크에
존재**했다. ⟹ (i) disambiguation은 신규 측정이 아니라 **기존 기록 확인**이라 더 견고. (ii) **그러나 rule 4에
FULL−ens1 break 기준을 freeze한 것은 설계 결함** — 그 아티팩트성이 이미 arena-X 저장 데이터로 예견 가능했는데
**사전-감사도 나도 prereg 시점에 ens1의 arena-X 거동을 확인 안 했다.** 아래 표는 그 저장값 직접 계산:
```
                       arena X (e43e/e42)      arena Y (e49)
cur_grid − ens1        +0.062* / +0.060*       +0.058*      ← 사실상 동일
FULL     − ens1        +0.055*                 +0.071*      ← 사실상 동일
FULL     − cur         −0.007 (n.s.)           +0.012 (n.s.)← 사실상 동일
oracle   − cur         +0.054*                 +0.069*      ← 사실상 동일
oracle   − ens1        +0.116*                 +0.124*      ← 사실상 동일
```
**arena X와 arena Y가 모든 대조서 동일.** FULL−ens1 "break"(+0.055)는 blind≈0이 확립된 **arena X서도 이미 존재.**
⟹ **BREAK은 arena-Y regime 효과가 아니라 arena-불변 약-baseline 아티팩트.** 검사의 "arena-X 특이" 우려는
**직접 측정으로 반증됨.**

## ★정정된 판정 (이중검증 후)
- **주 확정 (재프레임)**: **배포-음성이 open/현실 아레나로 일반화됨** — `FULL−cur = +0.012 (Y, n.s., median +0.004)`
  가 `−0.007 (X, n.s.)`를 재현. blind(고정-shape)는 배포 net을 **어느 아레나서도 못 이김.** ⟹ **E-48 killer-② 해소:
  음성은 arena-X 특이가 아니다.**
- **prize 확증·arena-불변**: `oracle−cur ≈ +0.05~0.07` (두 아레나). 외부-공급 채널 가치 실재.
- **auto-BREAK 기각**: frozen rule상 정당 발화했으나 **약-baseline(ens1) confound**(82% cur−ens1 상속, arena-불변).
  **척추 승격 금지.** — 코디의 예비해석 중 "blind가 실정보 회수"는 **과대였고 철회**(고유분 +0.013 n.s.).
- **캐비엇 불변**: (1) FULL은 **고정-shape** → 완전 정보-폐쇄 아님(shape-marginalized blind / **5-dim CRB 미납**).
  (2) prize는 **참-파티션 공급** 기준이고, 배포엔 부이급 입력 필요한데 **현존 계기 부재**(E-46: 실부이 ~21.9°).
- **방법론 교훈 (자화자찬 아님 — 자기가 판 함정)**: auto-label "BREAKS" + 코디 초기 독법 둘 다 break 쪽으로
  기울었고, 사후검증이 약-baseline 아티팩트로 정정한 건 맞다. **그러나 이 함정은 내가 판 것이다** — rule 4에
  FULL−ens1 break 기준을 넣은 것 자체가 결함이었고, 그 아티팩트성은 prereg 시점에 e43e 저장 데이터를 열었으면
  보였다(사전-감사도 나도 안 봄). 정직한 교훈 = "**새 test 통계를 사전등록할 땐 그것이 기존 아레나서 어떻게
  거동하는지 먼저 확인하라.**" 이중검증의 값은 함정 회피가 아니라 **내가 판 함정을 실행 후에라도 잡은 것.**
- **잔여 캐비엇 (fable 각주)**: (i) 표적 sea 1개서 `cur=−0.007`(net 파국 실패)·`FULL=0.593≈oracle` — open
  아레나에 **net이 파국 실패하고 blind가 오라클급 회수하는 sea가 실존**(n=1, median 서사는 체계효과 부재를 옳게
  말하나, net 실패모드의 존재는 배포-리스크 각주). (ii) "no blind method can"은 이 데이터가 **불지지** — 주어는
  "이 고정-shape 952,560-가설 전수열거"로 한정 필수(shape-marginalized blind·CRB 미검).
