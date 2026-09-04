# `T12-ISOLATE` 도달성 조사 — **`RESCALE` 교락의 여섯 상수를 하나씩 뗄 수 있는가**

> ⛔**런 0회.** 이 문서는 **소스 읽기 + `T11` 동결 결과 파일 재집계**만으로 쓰였다.
> 추정기·제안망·확산 체크포인트를 **한 번도 실행하지 않았다.**
>
> **열람 범위 = 주장 범위.** 아래 판정의 근거는 전부 이 파일들의 **실제로 연 구간**이다:
> `wavesplat/optimizer/initializer.py`(`:40-100`,`:170-200`,`:290-380`,`:495-570`) ·
> `wavesplat/optimizer/learned_init.py`(전문) ·
> `wavesplat/optimizer/sequential_update.py`(`:450-530`,`:890-1180`,`:1555-1790`) ·
> `wavesplat/optimizer/varpro.py`(`:25-45`,`:109-160`,`:195-235`) ·
> `wavesplat/model/spectral_gaussian.py`(`:215-235`) · `wavesplat/run.py`(`:25-120`) ·
> `scripts/d3_beamform_net.py`(`:1-80`) · `scripts/d3_beamform_cond_diffusion.py`(부분) ·
> `scripts/d3_setdiff_prior.py`(`:20-90`) · `scripts/d3_setdiff_posterior_b2.py`(`polish_atoms`).
> ⛔12 GB zip 은 **열지 않았다**(namelist 도 이번엔 안 열었다 — `T11` 의 `sample_t11.json`
> 과 `NOTE.md §8` 의 기록을 인용한다).

---

## §0 배치 팔이 실제로 도는 경로 (이것이 도달성의 분모다)

`T11` 의 `SHIP` 은 `DEFAULT_CONFIG` + `init_method='auto'` 이고, `AUTO_SWITCH_THRESH_DEFAULT = 0`
이므로 라우터가 **항상 `learned`** 로 간다(`T11` `G-ARM` 이 24/24 되읽기로 확인:
`resolved_init_method == 'learned'` · `real_domain_gate_fired == false`).
⟹ 실행되는 순서는 다음 하나뿐이다:

```
make_updater(cfg)                                   sequential_update.py:2254
 └ _auto_select_init                                sequential_update.py:1069
    └ propose_atoms  (확산 제안 + polish)            learned_init.py:124
       ├ _load_cond → stats = torch.load(PRIOR_CKPT)['stats']      learned_init.py:113
       ├ cd._bmap_from_obs → d3_beamform_net.beamform  (조건화 맵)  learned_init.py:151
       ├ cd.sample (확산 역과정)                                    learned_init.py:152
       └ cd.polish_atoms(..., iters=80)                             learned_init.py:158
          └ denorm_atoms(z, stats)  kx = z·stats['K_LIM']           d3_setdiff_prior.py:69
    └ pool_and_select(energy_thresh, dedup_dk, max_n)  learned_init.py:359
       (raw 원자는 sequential_update.py:1122 가 cfg['learned_atoms'] 에 캐시한다)
 └ _initialize_learned                               sequential_update.py:951
    ├ learned_init_params → pool_and_select(재실행, 캐시 사용)      learned_init.py:359
    ├ ls_fit_wphi (ridge 0.5)                                       learned_init.py:386
    ├ 결합 적합 루프  learned_init_iters = 0 ⟹ **0회**              sequential_update.py:978
    ├ _adaptive_prune_grow (adaptive_n=True)                        sequential_update.py:1555
    │   └ 성장 분기: `n_gaussians >= max_n(30)` 이면 **즉시 반환**   sequential_update.py:1632
    │   └ _beamform_coarse(±4.0, n_kx=61)                           sequential_update.py:1692-1694
    │   └ DC 블록 zero (±1셀 = ±0.1333)                             sequential_update.py:1699
    │   └ MIN_DK 억제 박스                                          sequential_update.py:1687,1721
    │   └ _beamform_refine(delta=0.15, n_fine=31)                   sequential_update.py:1712-1713
    └ apply_varpro(steps=400)  ⟹ lr 은 **기본 인자**                sequential_update.py:1004-1006
 └ clamp_steepness(max_steepness) — 위 여러 지점                    spectral_gaussian.py:219
```

★**중요**: `auto_guard` 기본값이 `False`(`sequential_update.py:1153`)라
`initializer.beamform_peak_ratio` 는 **호출되지 않는다**. 그리고 `learned_peak_anchor` 기본
`False`(`learned_init.py:363`)라 `_beamform_peak_locations` 도 **호출되지 않는다**.
⟹ ★***이 경로에서 `MIN_DK` 가 쓰이는 곳은 `_adaptive_prune_grow` 성장 분기 하나뿐이다.***

---

## §1 도달성 표

`✅CONFIG` = 순수 config 노브 · `✅PATCH` = 패키지 무수정 몽키패치로 도달 ·
`⚠LOCKED` = 다른 상수와 잠겨 단독 처치 불가 · `⛔UNREACHABLE` = 이 배치에서 못 잼

| # | 후보 | 유일 정의 (파일:줄) | 값 | 계산에 들어가는 지점 | 도달성 | 같이 움직이는 것 |
|---|---|---|---|---|---|---|
| **A1** | 사전 학습 밴드 | `results/direction3/setdiff_prior.pt` 의 `stats['K_LIM']`, 적용 = `scripts/d3_setdiff_prior.py:69` | **1.1** | `denorm_atoms` 가 정규화 원자 `z∈[-1,1]` 를 `k = z·K_LIM` 로 물리화. `learned_init.py:158`(polish 안) 과 `:159`(최종 원자) 둘 다 | **✅PATCH** — `LI._load_cond` 를 래핑해 `stats` 사본의 `K_LIM` 만 바꾼다(`T11` 의 `ObsProbe` 가 이미 같은 방식으로 `_load_cond` 를 래핑한다). ⛔**재학습 불필요** — 밴드는 **선형 denorm 스칼라 하나**다 | ★**polish 물리 스텝**(§2.1) · ★**A2 와 학습가중치로 잠겨 있다**(§2.2) |
| **A2** | 조건화 맵 격자칸 | `scripts/d3_beamform_net.py:31` `KLIM = 0.6`, 격자 생성 `:35-43` | 칸 `2·0.6/63` = **0.019048** | `beamform()` 이 모듈 전역 `KX_G,KY_G,OM_G` 를 **호출 시점**에 읽는다(`:47-49`) | **✅PATCH** — `d3_beamform_net.KLIM = 0.3` 후 `KX_G,KY_G,OM_G = kgrid()` 재대입. `cd` 가 `:33` 에서 임포트한 것은 **함수 객체**라 새 전역을 본다 | ★**개구/픽셀 비**(F1 과 같은 자유도) · ★A1 과 잠김 |
| **A2′** | 조건화 맵 격자점수 | 같은 줄 `NK = 64` | 64 | 같음 | ⛔**UNREACHABLE** — `CondEncoder` 가 64→32→16 로 줄이고 **학습된 위치임베딩 `pos` 가 `(1,16·16,d)`** 로 고정돼 있다(`d3_beamform_cond_diffusion.py` `CondEncoder.__init__`). NK 를 바꾸면 **텐서 모양이 안 맞는다** ⟹ 재학습 | — |
| **A3** | 조건화 맵의 **수심** | `scripts/d3_beamform_net.py:30` `DEPTH = 50.0` | 50 m | `kgrid()` 가 `OM_G = √(g·k·tanh(k·DEPTH))` 로 정합필터의 ω 를 만든다(`:38`). `learned_init.py:151` 은 depth 인자를 **넘기지 않는다** ⟹ config `depth=400` 이 여기 **안 닿는다** | **✅PATCH** — `DEPTH` 재대입 + `kgrid()` 재호출 | 없음(단독) — ★단 **스케일 축이 아니다**(§3) |
| **B1** | `learned_dedup_dk` | `learned_init.py:344` `config.get('learned_dedup_dk', 0.03)` | **0.03** | `pool_and_select(:184)` 이 `|Δkx|<ddk ∧ |Δky|<ddk` 인 원자를 버린다. **두 번** 쓰인다 — 라우터(`sequential_update.py:1116`)와 설치(`learned_init.py:359`) | **✅✅CONFIG** — 저장소에서 **유일하게 완전히 깨끗한 단일 노브** | 없음. ⚠단 **라우터와 설치가 같은 값을 봐야 한다**(둘 다 같은 `cfg` 를 읽으므로 자동) |
| **B2** | `learned_energy_thresh` | `learned_init.py:343` | 0.03 | 같은 함수, `en > thresh·en.max()` | ✅CONFIG | — (⚠**상대량이라 스케일 무관** ⟹ 이 배치의 축이 아니다) |
| **C1** | `MIN_DK` | `initializer.py:63` | **0.15** | 이 경로에서는 **성장 분기 한 곳**(`sequential_update.py:1683` 이 **호출 시점 임포트**, `:1687` 선택, `:1721` 억제 박스 `int(MIN_DK/dkx)+1` 셀) | **✅PATCH** — `initializer.MIN_DK = 0.075`. 호출 시점 임포트라 몽키패치가 먹는다 | 억제 **셀 수**가 `dkx`(C2)와 결합 — `(int(MIN_DK/dkx)+1)·dkx` 라 **양자화**된다(§2.3) |
| **C2** | 성장 빔포머 격자 · **DC 블록 0.133** | `sequential_update.py:1693-1694` `kx_range=(-4.0,4.0), n_kx=61` ⟹ `dkx = 8/60 = 0.13333`; DC 제로 `:1699` `[cy-1:cy+2, cx-1:cx+2]` = **|kx|,|ky| ≤ 0.1333 삭제** | 0.1333 | 성장 분기만 | **✅PATCH(간접)** — 리터럴은 하드코딩이지만 `_adaptive_prune_grow` 가 `_beamform_coarse` 를 **호출 시점에 임포트**한다(`:1682-1684`) ⟹ `initializer._beamform_coarse` 를 **래퍼로 갈아끼워** `kx_range/ky_range` 를 강제하면 도달한다. DC 블록과 억제 박스는 **반환된 축에서 파생**되므로 자동 추종 | ★**DC 블록·억제 박스·refine 창이 전부 `dkx` 파생** ⟹ 이 셋은 **독립 처치가 아니다**(§2.3) |
| **C3** | 성장 refine 창 | `sequential_update.py:1712-1713` `delta=0.15, n_fine=31` | 0.15 / 0.01 | 성장 분기만 | ⛔**UNREACHABLE(단독)** — `_beamform_refine` 도 호출 시점 임포트라 래핑은 되지만 **C2 와 같은 래퍼 지점**이라 분리해서 의미가 없다. ⟹ **C2 처치에 동봉**하고 공시 | C2 |
| **D1** | VARPRO Adam lr | `varpro.py:34` `VP_LR_DEFAULT = 2e-3`, 시그니처 `varpro.py:197` | **2e-3** | `sequential_update.py:1005` 이 `steps=` 만 넘기고 **`lr` 은 안 넘긴다** ⟹ 기본 인자 | **✅PATCH** — ⛔★**`varpro.VP_LR_DEFAULT = 1e-3` 은 안 먹는다**: 파이썬 기본 인자는 **`def` 실행 시점에 평가**돼 이미 바인딩됐다. `sequential_update.py:1004` 가 **모듈을 호출 시점에 임포트**하므로 `varpro.apply_varpro` **함수 객체 자체**를 래퍼로 갈아끼워야 한다 | 없음. ★되읽기 쉬움 — `apply_varpro` 반환의 `info['lr']`(`varpro.py:152`)가 행에 실린다 |
| **D2** | polish Adam lr | `scripts/d3_setdiff_posterior_b2.py` `polish_atoms(..., lr=0.02)` | 0.02 (**정규화 z 단위**) | `learned_init.py:158` 이 `iters` 만 넘긴다 ⟹ 기본 인자. **물리 스텝 = `lr·K_LIM` = 0.022 rad/m** | ⚠**LOCKED** — 도달은 가능하나(`cd.polish_atoms` 교체) **A1 을 바꾸면 물리 스텝이 자동으로 같은 비율로 바뀐다** ⟹ A1 단독 처치는 실은 **A1+D2 이중 처치**다(§2.1) | A1 |
| **E1** | `max_steepness` | `sequential_update.py:525` `config.get('max_steepness', 0.15)`, 적용 `spectral_gaussian.py:219-230` (`ε = |k|·|w|`) | 0.15 | `:900`, `:984`, `:1545`, `:1781` + `varpro.py:246` | ✅CONFIG | — ⚠**무차원량이라 `s` 로 스케일되지 않는다**(§3) |
| **F1** | 개구/픽셀 비 | 자료의 관측 발자국 ÷ A2 의 칸 | `AMENDMENT_2 §6a` 인용: SHIP **0.53칸** → RESCALE 1.06 · 학습 코퍼스 1.37–1.65 | 조건화 맵의 PSF 폭 | ⚠**A2 와 같은 자유도** — 비를 바꾸는 방법이 둘뿐이고 하나는 A2(칸 크기), 다른 하나는 **관측 발자국 크롭**이다. 크롭은 **정보량과 교락**(관측 개수·공간 커버리지가 같이 줄어든다) ⟹ **단일요인 아님** | A2 / 정보량 |
| **G1** | `max_n` | `run.py DEFAULT_CONFIG` `max_n: 30` | 30 | `pool_and_select` 캡(`learned_init.py:342`) **그리고** 성장 조기반환(`sequential_update.py:1632`) | ✅CONFIG — ⛔**단 처치하지 않는다**: `B-2f` 선례대로 `max_n` 은 **선택과 성장을 동시에** 움직여 비직교다(`sequential_update.py:1626-1631` 주석이 축자로 경고) | 선택 + 성장 |

---

## §2 결합 (⟹ 「독립 처치」가 **아닌** 것들)

### 2.1 A1(밴드) 과 D2(polish 스텝)
`polish_atoms` 는 **정규화 좌표 `z` 에서** Adam 을 돌리고(`lr=0.02`), 손실은
`denorm_atoms(z, stats)` 로 물리화한 뒤 계산한다. ⟹ **물리 파수의 한 스텝 = `lr·K_LIM`**.
`K_LIM` 을 1.1→0.55 로 내리면 물리 스텝도 0.022→0.011 로 **자동으로 반감**한다.
⟹ ★***"학습 밴드만 옮긴다" 는 처치는 존재하지 않는다*** — 항상 polish 해상도가 동반된다.
⛔**떼려면** `lr` 을 0.04 로 올려 물리 스텝을 고정해야 하고, 그러면 처치가 둘이 된다.
**이 배치는 떼지 않고 공시한다**(A1 처치 = 밴드 + polish 스텝).

### 2.2 A1(밴드) 과 A2(맵 칸) — ★**학습된 가중치가 둘을 잠근다**
망은 물리 파수를 **모른다**. 학습이 만든 대응은 *"Bmap 의 픽셀 `p` ↔ 정규화 원자 `z`"* 하나이고,
학습 시 그 대응을 고정한 것이 **`KLIM=0.6`(픽셀↔물리 k) 과 `K_LIM=1.1`(z↔물리 k) 의 쌍**이다.
- `KLIM` 만 반감 ⟹ 같은 물리 봉우리가 **두 배 바깥 픽셀**로 가고, 망은 그에 해당하는 `z` 를
  뱉으며, `K_LIM=1.1` 이 그것을 **2배 큰 물리 k** 로 되돌린다 ⟹ **디코딩이 깨진다.**
- `K_LIM` 만 반감 ⟹ 픽셀 대응은 그대로인데 출력만 절반 ⟹ **원자가 통째로 아래로 밀린다.**
- 둘 다 반감 ⟹ *"제안 서브시스템이 `s=2` 로 재척도된 세계를 본다"* = **정합**.
⟹ ★그래서 이 배치는 셋을 **모두** 팔로 둔다(`A-MAPCELL`·`A-BAND`·`A-MAPSCALE`).
`A-MAPSCALE` 은 앞의 둘의 **합성**이므로 넷(SHIP 포함)이 2×2 를 이룬다.
⛔**단 2×2 를 "상호작용" 으로 쓰지 마라** — `M8b` 개정에서 저장소가 그 자구를 철회했다.

### 2.3 C1(MIN_DK) 과 C2(성장 격자) — 양자화 결합
억제 반경은 `(int(MIN_DK/dkx)+1)·dkx` 다.
| `dkx` | `MIN_DK=0.15` 셀 | 실효 반경 |
|---|---|---|
| 0.13333 (배포) | `int(1.125)+1 = 2` | **0.2667** |
| 0.06667 (C2 처치) | `int(2.25)+1 = 3` | **0.2000** |
⟹ **C2 를 처치하면 억제 반경도 0.2667→0.2 로 따라 움직인다**(의도 밖).
그리고 **DC 블록**(±1셀)은 정의상 `dkx` 그 자체다 ⟹ C2 처치는 **DC 블록 + 억제 반경 + refine
격자(C3)** 를 한꺼번에 옮긴다. ⛔**단일 상수 처치가 아니다** — *"성장 서브시스템의 절대 스케일"*
하나의 처치로 읽어야 한다.

### 2.4 A2(맵 칸) 과 F1(개구/픽셀 비)
`F1 = PSF 폭 ÷ 맵 칸` 이고 PSF 폭은 자료(관측 발자국)가 정한다. 발자국을 고정하면
**F1 을 움직이는 유일한 노브가 A2** 다 ⟹ ★***F1 은 독립 축이 아니다.*** 자료 쪽으로 떼려면
관측을 공간적으로 크롭해야 하는데 그러면 정보량이 함께 준다.
⟹ **이 배치에서 F1 은 A2 의 해석**으로만 등장하고 별도 팔을 만들지 않는다.

### 2.5 ⚠성장 분기가 **애초에 안 돌 수 있다**
`sequential_update.py:1632` 가 `n_gaussians >= max_n(30)` 이면 성장 전체를 조기 반환한다.
`T11` 동결 결과 재집계(런 0회): `SHIP` 의 `n_atoms_installed` 는
`[7,8,9,18,24,26,30×18]` — **24블록 중 18블록이 정확히 30 = `max_n`** 이다.
⚠이 되읽기는 **prune+grow 이후** 값이라 *"성장이 안 돌았다"* 를 증명하지 **못한다**
(가지치기로 내려갔다가 도로 30까지 자랐을 수도 있다).
⟹ ★**C1·C2 축은 「성장이 실제로 발화한 행」에서만 해석 가능**하다.
⟹ 러너가 `model.add_gaussians` 를 래핑해 **호출 수와 추가 원자 수를 행에 되읽어야** 하고,
발화가 희소하면 그 축의 결과는 **널이 아니라 `NOT-EXERCISED`** 로 보고해야 한다
(저장소 규율: *"도달 불가"* 와 *"지렛대 부족"* 을 섞지 마라).

---

## §3 ⚠**스케일 축이 아닌 것** (「`RESCALE` 의 몇 %를 회수하는가」로 읽으면 안 되는 것)

`RESCALE`(`x'=x/s, t'=t/√s`, `s=2`)은 자료의 파수를 **2배**로 만든다. 자료를 고정한 채 같은
효과를 내려면 **파수 차원[rad/m]을 가진 상수를 절반**으로 내려야 한다: A1·A2·B1·C1·C2·D1.
그러나 —

- **A3(맵 수심 50 m)**: `RESCALE` 은 맵 수심을 바꾸지 않는다. 대신 `k'd` 가 커져 정합필터가
  **자동으로 심해에 가까워진다**(`AMENDMENT_2 §6b`: SHIP `k=0.02` 에서 Δω/ω 12.7% ·
  정합이득 0.66 → RESCALE `k'=0.06` 에서 0.25% · 1.000). ⟹ A3 처치(50→400)는
  **그 효과만 골라 재현**하는 것이지 스케일 처치가 아니다. ★그러나 `RESCALE` 이득의
  **일부를 실제로 나를 수 있는** 축이고, 그것이 이 배치가 A3 를 넣는 이유다.
- **E1(`max_steepness` 0.15)**: `ε = |k|·|w|` 는 **무차원**이다. `RESCALE` 세계에서는 같은
  물리파의 `ε` 가 **2배**가 되어 클램프가 **더 세게** 문다 ⟹ ⛔***클램프 완화는 `RESCALE`
  이득의 원천이 될 수 없다.*** E1 은 **절제**(*"SHIP 의 실패 중 얼마가 클램프 자해인가"*)이지
  회수 축이 아니다. ⚠`T11` `G-CLAMP` 가 `SHIP` **19회** · `RESCALE` **5회**를 기록했고
  `AMENDMENT_2 §9.1` 이 *"⛔이것을 성능 차의 원인으로 지목하지 마라"* 라 못박았다 —
  이 배치가 그 공시를 **닫는다**.

---

## §4 ⛔도달 불가 목록 (사유 포함)

| 후보 | 사유 | 이 배치의 대리 처치 |
|---|---|---|
| **A2′ `NK=64`** | `CondEncoder.pos` 가 `(1, 16·16, d)` 학습 파라미터 ⟹ 입력 해상도가 **가중치에 구워졌다**. 바꾸면 텐서 모양 불일치 | 없음 — `KLIM`(A2)이 같은 자유도(칸 크기)를 덮는다 |
| **사전 학습 밴드를 「재학습으로」 옮기기** | 16,000 스펙트럼 재생성 + 확산 재학습 ⟹ 이 배치 예산 밖. ⛔그리고 **필요 없다**: 밴드는 `stats['K_LIM']` **스칼라 하나**의 선형 denorm 이라 A1 패치가 같은 일을 한다 | `A-BAND`(§2.1 공시 동반) |
| **C3 refine 창 단독** | C2 와 같은 함수 래핑 지점 · 코어스 봉우리가 정하는 중심에 종속 | `C-GROWGRID` 에 동봉 + 공시 |
| **F1 개구/픽셀 비 단독** | 자료 속성 × A2. 자료 쪽으로 떼면 정보량과 교락 | `A-MAPCELL`(칸만 움직이는 팔)이 비를 **정확히 2배**로 만든다 |
| **`max_n=30`** | 선택 캡과 성장 조기반환을 동시에 움직인다(코드 주석이 축자로 경고, `:1626-1631`) | 처치하지 않음. 대신 성장 발화 계수기를 기록 |
| **학습 코퍼스 분포**(`Tp∼U(5,11)`, `Hs∼U(0.8,3.2)`, `d=50`, `M=32`, 방향 ±50°, 퍼짐 12–40°; `d3_beamform_cond_diffusion.py` `make_batch`) | 재학습 필요 | 없음 — **한계 절 문장**으로 남긴다 |

---

## §5 ★조사 중 새로 나온 것 (`AMENDMENT_2 §6a` 목록에 **없던** 것)

1. **polish Adam lr 0.02 (z 단위 ⟹ 0.022 rad/m 물리 스텝)** — 목록에 없다. 자료의 `k_p ≈ 0.04`
   대비 **55%** 다. A1 과 잠겨 있어(§2.1) 항상 함께 움직인다. ⟹ **`§6a` 목록은 6개가 아니라
   최소 7개였다.**
2. **성장 refine 창 `delta=0.15 / n_fine=31`**(`:1712-1713`) — 목록에 없다. `MIN_DK` 와 같은
   0.15 이고 `k_p` 의 **3.75배**다.
3. **억제 반경의 양자화 결합**(§2.3) — 실효 억제 반경은 `MIN_DK` 가 아니라
   `(int(MIN_DK/dkx)+1)·dkx` 라는 **계단 함수**다:
   | 처치 | `MIN_DK` | `dkx` | 셀 | 실효 반경 |
   |---|---|---|---|---|
   | 배포 | 0.15 | 0.13333 | 2 | **0.26667** |
   | `C-MINDK` | 0.075 | 0.13333 | 1 | **0.13333** (우연히 정확히 절반) |
   | `C-GROWGRID` | 0.15 | 0.06667 | 3 | **0.20000** (의도한 절반이 **아니다**) |
   ⟹ ★**두 처치 중 어느 쪽도 「하나만」 반으로 만들지 못한다**. `C-MINDK` 가 우연히 정확히
   절반이 된 것은 **계단의 우연**이지 설계가 아니다 — 문턱이 조금만 달랐으면 아니었다.
   ⟹ 처치 후 **실효 반경(셀 수 × `dkx`)을 행에 되읽어 기록**해야 한다.
4. **`d3_beamform_net.DEPTH=50` 이 `kgrid()` 안에서 `OM_G` 에 구워진다**(`:38`) ⟹ config 의
   `depth` 는 **정의상** 조건화 맵에 못 닿는다. `AMENDMENT_2 §6b` 가 이미 공시했으나
   **줄번호가 `:31` 로 적혀 있었고 실제는 `:30`** 이다(그 개정문 자신이 `⚠줄번호 드리프트` 로
   기록해 두었다 — 여기서 재확인).
5. **`VP_LR_DEFAULT` 는 모듈 상수 재대입으로 안 바뀐다**(§1 D1). 이것은 **저장소의 다른
   「손복사 상수」 결함과 다른 부류**다 — 단일출처가 맞는데 **바인딩 시점** 때문에 개입이
   조용히 no-op 이 된다. ⟹ ★**개입 계수기 없이 이 축을 처치하면 「널」로 오보고된다.**

---

## §6 요약

- **처치 가능(단독 또는 「서브시스템 단위」로 정의 가능)**: **8축**
  `A-MAPCELL`(A2) · `A-BAND`(A1+D2) · `A-MAPSCALE`(A1+A2+D2) · `B-DEDUP`(B1) ·
  `C-MINDK`(C1) · `C-GROWGRID`(C2+C3, +억제반경 동반) · `D-VPLR`(D1) · `E-MAPDEPTH`(A3) ·
  `F-STEEP`(E1, ★절제이지 회수 아님)  ⟹ 러너 팔로는 **9개**
- **도달 불가**: **6건** — `NK=64`(학습 가중치에 구움) · 밴드 재학습(예산; 그러나 **불필요**) ·
  C3 단독 · F1 단독(정보량 교락) · `max_n`(비직교) · 학습 코퍼스 분포(재학습)
- **⛔단일요인이라고 부를 수 없는 것**: A1↔D2 · A1↔A2 · C1↔C2↔C3 · A2↔F1
