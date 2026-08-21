# `FAIR` 개입 실험 — 원자료 (32런)

**왜 이 디렉터리가 있나.** 2026-08-13 3라운드 회귀검증(`../R3_regression.md §M-6`)이
*"판정을 뒤집은 32런 개입 실험의 원자료가 저장소에 없다 — `p=0.804 / p=1.000` 은 **재도출 불가**"*
를 MAJOR 로 잡았다. 실제로 실험은 스크래치패드에서 monkeypatch 로 돌았고 저장소 파일은 건드리지 않았다
(`../V1_refute_nodisp.md §7` 이 그렇게 자백한다). 저장소의 상시 지시
(`feedback_repo_tmp_no_scratchpad.md` · `A1_s13_rederive.md:333` *"집계 스크립트를 파일로 남길 것"*)를
V1 자신이 지키지 않은 것이다. **이 디렉터리는 그 원자료·스크립트·집계기를 복원한 것이다. 새 런은 0회.**

이 32런은 [`../V1_refute_nodisp.md §7`](../V1_refute_nodisp.md) 의 근거이고, 그 결과로
`tmp/S13_oos_2026-08-12/RESULT.md` 의 **기여 #2b `ALIVE`·"복권" 이 `UNDECIDED` 로 철회**됐다.

---

## 1. 무엇을 돌렸나

**질문.** S13 의 `SHIP_NODISP` 팔(62/64, +70.2%/+777.8%)은 **분산관계 하드 제약의 절제**인가,
아니면 **ω 를 아무도 초기화하지 않아서 난수에서 출발하는 고장난 팔**인가?

`use_dispersion=False` 일 때 `omega_free` 는 `spectral_gaussian.py:63-64` 에서
`torch.randn(n) * 0.5` 로만 생성되고, 그 값을 채우는 코드 경로가 저장소에 없다
(초기화 공급자 `initializer.initialize_from_frames` / `learned_init.learned_init_params` 도,
설치 지점 `sequential_update.py:543-553` / `:630-638` 도 **kx, ky, w, phi 4개만** 넘긴다).

**개입.** `SequentialUpdater._build_optimizer` 를 감싸서, 초기화기가 kx/ky 를 설치한 **직후 1회만**
`omega_free.data ← dispersion_omega(√(kx²+ky²), depth)` 로 채운다. 그 외 모든 것이 동일하고
`randn` 호출 자체는 그대로 두므로 **RNG 스트림이 `NODISP` 와 비트 동일**하다.

| 팔 | 정의 | 출처 |
|---|---|---|
| `SHIP` | 출하 구성 (`use_dispersion=True`) | S13 `results.jsonl` 기록값 |
| `NODISP` | `use_dispersion=False`, ω 는 **난수** 출발 | S13 `results.jsonl` 기록값 |
| **`FAIR`** | `use_dispersion=False`, ω 를 **분산관계 값에서 출발만** 시킴 (이후 완전 자유, tracking 에서도 자유) | **여기 `fair.py` 가 새로 돌린 32런** |

**표본.** S13 의 128 에피소드를 층별로 정렬해 4개씩 건너뛴 **층당 16개 = 32런**
(`fair.py`: `eps[::4][:16]`). 실측 커버리지 — 8 구성 `BF4/BF5 × {2,5,15,30}comp` × 시드 `{32,36,48,52,56,60}`.
데이터는 `data_s9_sparse_2026-08-11/` · `data_s8_crowded_2026-08-10/` · `data_s9_crowded_2026-08-11/`.
`git_head` = `5f0846c`. 시드 소모 **0개**(기존 에피소드 재사용), 생성 **0회**.

---

## 2. 재현 명령

```bash
cd /home/vexit/wavesplat/wavesplat

# (a) 집계만 재도출 — 런 0회, 즉시 (아래 §3 표가 그대로 나온다)
python3 tmp/AUDIT_recheck_2026-08-13/fair_ablation/analyze_fair.py

# (b) 32런 전체 재실행 — CPU 약 25분. 이어달리기 가능(이미 있는 ep 는 건너뛴다)
#     ⚠ fair.py:24 의 OUT 경로가 스크래치패드로 하드코딩돼 있다. 재실행 전 이 디렉터리로 바꿀 것:
#     OUT='/home/vexit/wavesplat/wavesplat/tmp/AUDIT_recheck_2026-08-13/fair_ablation/fair.jsonl'
python3 tmp/AUDIT_recheck_2026-08-13/fair_ablation/fair.py
```

| 파일 | md5 | 내용 |
|---|---|---|
| `fair.py` | `af8277fbf03cd8cc5e02db4813cef1ea` | 개입 러너 (monkeypatch + 32런). **원본 그대로 보존** — 경로 하드코딩 포함 |
| `fair.jsonl` | `e60717a891c1ce96672fc64963a5666f` | 32행 원자료. 행당 `ep · strat · {fair,ship,nod}_{init,fin}` |
| `fair.log` | `dcb4aa3904819d93fbc6a890113d4b46` | 러너 stdout (에피소드별 벽시계 시간 포함) |
| `analyze_fair.py` | — | 집계기 (**2026-08-13 신규**, R3 §M-6 처방) |

---

## 3. 결과 (`analyze_fair.py` 출력, `V1 §7` 과 전건 일치)

| 층 | 대조 | med(rel) | 패/n | 부호검정 p |
|---|---|---|---|---|
| 희소 | `NODISP` vs `SHIP` | **+721.4%** | 15/16 | **5.19e-04** |
| 희소 | **`FAIR` vs `SHIP`** | **+7.3%** | 9/16 | **0.804 (널)** |
| 희소 | `NODISP` vs `FAIR` | +358.5% | 15/16 | 5.19e-04 |
| 혼잡 | `NODISP` vs `SHIP` | **+62.2%** | 16/16 | **3.05e-05** |
| 혼잡 | **`FAIR` vs `SHIP`** | **−0.7%** | 8/16 | **1.000 (널)** |
| 혼잡 | `NODISP` vs `FAIR` | +54.8% | 16/16 | 3.05e-05 |

`init_nrmse` 중앙 — 희소 `SHIP` 0.0121 / `FAIR` **0.0299** / `NODISP` 0.2416 ·
혼잡 `SHIP` 0.1546 / `FAIR` **0.1486** / `NODISP` 0.2755.

**해석.** 분산관계를 **하드 제약으로 강제**하는 것과, ω 를 **자유 파라미터로 두되 분산관계 값에서
출발**시키는 것은 정확도에서 **구별되지 않는다**. ⟹ S13 이 잰 `62/64 · +70.2% / +777.8%` 는
**전부 `omega_free` 의 난수 출발점**이 만든 것이고, **분산 제약의 이득이 아니다**.

★단 **분산관계가 쓸모없다는 뜻은 아니다** — `FAIR` 팔조차 ω 를 분산관계에서 **받아** 출발한다.
측정된 것은 *"분산관계는 ω 를 알려주는 데서 값을 하고, 하드 제약으로 묶는 데서는 값을 안 한다"* 이다.

---

## 4. 한계 (인용 전에 읽을 것)

- ⚠**사전등록 없음 = 탐색적.** n=16/층. 확증하려면 n=64/층으로 사전등록할 것.
- ⚠**보수적 편향**: adaptive grow 로 나중에 추가되는 성분은 `spectral_gaussian.py:196` 상
  여전히 **난수 ω** 를 받는다 ⟹ `FAIR` 은 "이상적 공정 팔" 보다 **불리하다** ⟹ 위 널은 보수적이다.
  확증 배치에서는 grow 성분의 ω 도 분산관계로 시딩해 이 편향을 없앨 것.
- ⚠**`fair.py` 는 저장소 파일을 수정하지 않는 in-memory monkeypatch** 다. 따라서 이 결과는
  "저장소에 그런 옵션이 있다" 가 아니라 "**한 줄 시딩이 가능하고 그러면 효과가 사라진다**" 를 보인다.
- 비트 재현 확인(V1 §7): `BF4_15comp_seed32` `SHIP` 재실행 = `0.11077267540773998`
  (S13 기록값과 마지막 자리까지 일치) ⟹ 하네스는 결정적이다.
- **제로필드 게이트**: `NODISP` 는 혼잡에서 제로필드에 **25/64 밖에 못 이긴다**(V1 §3b)
  ⟹ 그 팔은 "더 나쁜 추정기" 가 아니라 **추정기가 아니다**. 어떤 형태의 #2b 확증이든
  이 게이트를 사전에 걸어야 한다(같은 게이트로 2026-04 `ablation_bf45.json` 도 런 0회로 탈락한다).
