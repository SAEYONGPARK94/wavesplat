# `T13-DEPTHFIX` 도달성 — 처치가 **실제로 조건화 맵에 닿는가** (코드 근거 · 런 0회)

작성 2026-09-04 · HEAD `7f6debc` · ⛔추정기 실행 0회 · 체크포인트 로드 1회(CPU, 프로브용)
⛔이 문서는 **설계 근거**다. 러너·게이트·분석기는 여기 없다.

> ★결론 먼저: **`d3_beamform_net.DEPTH` 만 바꾸면 처치는 0회 발화한다**(실측: Bmap 비트 불변).
> `KX_G, KY_G, OM_G = kgrid()` 를 **다시 호출**해야 닿고, 되돌리면 **비트 동일**로 복귀한다.
> 모듈 리로드는 **불필요하고 비권장**이다(§3.3).

---

## 1. 결함 지도 (파일:줄 — 전부 이번에 직접 열었다)

| 위치 | 내용 |
|---|---|
| `scripts/d3_beamform_net.py:30` | `G = 9.81; DEPTH = 50.0` — 모듈 상수 |
| `:35-40 kgrid()` | `om = √(G·k·tanh(k·DEPTH))` — **호출 시점의 `DEPTH` 전역**을 읽는다 |
| `:43` | `KX_G, KY_G, OM_G = kgrid()` — **모듈 로드 시 1회** 고정 |
| `:46-55 beamform(P, z)` | `:49` 가 `KX_G`·`KY_G`·`OM_G` 를 **호출 시점의 모듈 전역**에서 읽는다. `depth` 인자 **없음** |
| `scripts/d3_beamform_cond_diffusion.py:33` | `from d3_beamform_net import beamform, KLIM, NK, M_OBS, DEPTH` — **같은 함수 객체**를 가져온다(`cd.DEPTH` 는 값 복사 = 50.0, 학습에서만 쓰임 `:96,108`) |
| `:174-176 _bmap_from_obs(Ps, z)` | `beamform(...)` 호출. `depth` 인자 **없음** |
| `wavesplat/optimizer/learned_init.py:151` | `Bmap = cd._bmap_from_obs(Ps, zf)` — 배포 학습 경로의 **유일한** 맵 계산 지점. `propose_atoms(frames, depth, …)` 의 `depth`(`:124`)는 `:158 polish_atoms` 로만 간다 |
| `learned_init.py:98-120 _load_cond` | `sys.path.insert(0, <repo>/scripts)` 후 **bare name** `import d3_beamform_cond_diffusion as cd`. `_CACHE` 에 **모듈 객체**를 저장 |
| `learned_init.py:322` | `ls_fit_wphi` → `fit_amplitudes_phases(…, depth, …)` — config 수심 도달 ✅ |
| `scripts/d3_setdiff_posterior.py:52` | `loglik_atoms` 가 `ω = √(G·k·tanh(k·depth))` 를 **인자 depth** 로 계산 — polish 는 config 수심 도달 ✅ |
| `wavesplat/optimizer/sequential_update.py:951-966` | `_initialize_learned` → `learned_init_params(frames, self.depth, self.config, seed)` |
| `sequential_update.py:1097-1122` | `auto` 라우터가 `propose_atoms(frames, self.depth, …)` 를 부르고 `cfg['learned_atoms']` 에 캐시 |

⟹ **config `depth` 가 닿는 곳**: polish(`:158`) · LS 적합(`:322`) · 추정기 분산관계. **닿지 않는 곳**: 조건화 맵 `OM_G` **하나**.
⟹ **`OM_G` 를 읽는 곳**: `beamform` `:49` **하나**. 학습 경로에서 `beamform` 을 부르는 곳: `_bmap_from_obs` **하나**(`learned_init.py:151`).
프로브(§2)가 `polish_atoms`·`sample`·`propose_atoms` 소스에 `OM_G`/`KX_G` 참조 **0** 임을 확인했다.

### 1.1 학습 코퍼스도 d=50 이었다 — **배포 체크포인트의 실제 레시피에서** 확인
과제 지시는 `d3_beamform_cond_diffusion.py:96,108` 을 인용했으나, 배포 체크포인트는 그 스크립트가 아니라
`_mixed_v1` 재학습(`learned_init.py:45-62`)이다. 그 레시피(디스크 사본)를 열었다:
- `tmp/mixed_retrain_2026-07-26/gen_mixed.py:54` `DEPTH = 50.0` · `:304` `jonswap_components(…, depth=DEPTH, …)` ·
  `:344` `elevation(…, DEPTH, …)` · `:382` `Bmap = cd.beamform(…)` (= `OM_G`@d=50)
- `tmp/M7_SUPERVISION_2026-08-28/train_m7.py:121` 이 `('DEPTH', gm.DEPTH, cd.DEPTH, …)` 로 **두 상수의 일치를 검사**한다.
⟹ ✅**학습 시점에는 자료(d=50)와 맵(d=50)이 자기정합**이었다. 과제 진술 확인.
⚠단 **부채 H2**: `sha256sum -c tmp/mixed_retrain_2026-07-26/FROZEN.sha256` → **3/3 FAILED**(지금 재확인)
⟹ v1 을 만든 코드의 **바이트가 디스크에 없다**. 디스크 사본이 d=50 이라는 것은 강한 정황이지 증명이 아니다.
(`tmp/_audit/H2_CHECKPOINT_PROVENANCE_2026-09-02.md`)

---

## 2. 실행 프로브 (런 0회 — 체크포인트만 로드, 합성 관측 400점, CPU)

`CUDA_VISIBLE_DEVICES=""` · `systemd-run --user --scope -p MemoryMax=4G` 안에서 실행. 스크립트 요지:
```python
cd, model, stats, abar, betas, denorm = LI._load_cond(LI.DEFAULT_CKPT, 'cpu')   # 추정기와 같은 경로
bn = sys.modules['d3_beamform_net']
B0 = cd._bmap_from_obs(P, z)                       # 출하 맵 (d=50)
bn.DEPTH = 400.0;               B1 = cd._bmap_from_obs(P, z)      # 처치 A: DEPTH 만
bn.KX_G, bn.KY_G, bn.OM_G = bn.kgrid();  B2 = cd._bmap_from_obs(P, z)   # 처치 B: 재계산
bn.DEPTH = 50.0; bn.KX_G, bn.KY_G, bn.OM_G = bn.kgrid();  B3 = cd._bmap_from_obs(P, z)   # 되돌림
```
출력(축자):
```
cwd after import (side effect): /home/vexit/wavesplat/wavesplat
cd is sys.modules[cd]: True
cd.beamform is bn.beamform: True
cd.beamform.__globals__ is bn.__dict__: True
bn.DEPTH 50.0 cd.DEPTH 50.0 DEVICE cpu cpu
OM_G@import md5 b66c0cb8a2fc436a2d6df01cd4c7451b
DEPTH only -> Bmap changed? False | OM_G md5 same? True          ← ⛔처치 A = 0회 발화
kgrid() recompute -> Bmap changed? True | OM_G md5 8c0f70a67f342fa561975c99057df24d | max|dB| 0.282523512840271
analyzer-side recompute d=400 allclose to readback: True | d=50 allclose to import value? (should be False now) False
cells @ (32,33) k~.019 : d50 0.5174301266670227 d400 0.5435504913330078 readback 0.5435504913330078
revert -> bit-identical to shipped Bmap? True | md5 back? True   ← ✅가역 (팔 간 토글 가능)
OM_G/KX_G referenced in polish/sample/propose_atoms source: False
cache returns same module: True
propose_atoms looks up cd._bmap_from_obs at call time: True     ← ✅래퍼로 가로챌 수 있다
```
⚠위 md5 는 **CPU float32** 값이다. CUDA 에서 `sqrt`/`tanh` 의 마지막 ulp 가 다를 수 있으므로
**분석기는 md5 등식을 장치 간에 쓰지 말고 `allclose(atol=1e-5)` 로 공식 재계산과 대조**해야 한다(PREREG `G-MAP`).

---

## 3. 그래서 처치는 어떻게 넣어야 하는가

### 3.1 ✅채택 — 모듈 속성 교체 + `kgrid()` 재호출 (컨텍스트 매니저, 런마다 진입/이탈)
```python
# 지침 스케치(⛔러너 아님). bn = sys.modules['d3_beamform_net'] — 반드시 bare name 으로 잡을 것.
class MapDepth:
    def __init__(self, depth): self.depth = float(depth)
    def __enter__(self):
        bn = sys.modules['d3_beamform_net']
        self._saved = (bn.DEPTH, bn.KX_G, bn.KY_G, bn.OM_G)
        bn.DEPTH = self.depth
        bn.KX_G, bn.KY_G, bn.OM_G = bn.kgrid()        # ★이 줄이 없으면 0회 발화 (§2)
    def __exit__(self, *a):
        bn = sys.modules['d3_beamform_net']
        bn.DEPTH, bn.KX_G, bn.KY_G, bn.OM_G = self._saved   # 되돌림 = 비트 동일 (§2)
```
- `FIX` 팔의 `upd.initialize(...)` 전체를 `with MapDepth(upd.depth):` 안에서 돈다 — 제안(`propose_atoms`)은
  `initialize()` 안의 `_auto_select`(`sequential_update.py:1097-1122`)에서 불리므로 그 안에 있어야 한다.
- 수심 값은 **리터럴이 아니라 `upd.depth` 되읽기**(= config 400 이 실제로 객체에 들어간 값). 자료마다 수심이 다르면
  이 형태가 그대로 「처치 2(자료의 실제 수심)」의 일반형이다 — 이 코퍼스는 단일 수심이라 처치 1 과 일치한다.
- `kgrid()` 는 `bn.DEVICE` 에 텐서를 만든다(`:36`) ⟹ 실제 런(제안 device = cuda, T11 `proposal_device` 24/24 `cuda`)에서도
  `P`(`cd.DEVICE`) 와 같은 장치가 된다. ⛔직접 `torch.tensor(..)` 로 CPU 에 `OM_G` 를 만들면 **장치 불일치**로 죽는다.

### 3.2 ★되읽기 프로브 (게이트 `G-MAP` 의 재료 — 러너 리터럴 금지)
`cd._bmap_from_obs` 를 감싸 **호출 시점**에 다음을 기록한다(`propose_atoms` 가 `cd._bmap_from_obs` 를 **호출 시점에 이름 조회**하므로
`_CACHE` 에 모듈이 들어 있어도 가로채진다 — §2 마지막 줄):
- `bn.OM_G` 의 md5 **와** 4개 셀 값(`[32,33]`·`[32,34]`·`[33,33]`·`[40,40]`, k≈0.019/0.038/0.027/0.215) **와** 전체 4096 float32(팔당 1회 `meta` 에)
- `n_bmap_calls`(런당 **정확히 1** 이어야 한다 — `ensemble_k=1`, `nsamp=4` 는 한 맵에서 뽑는다)
- 생성된 `Bmap` 텐서의 md5
분석기는 `OM_G` 를 **공식으로 독립 재계산**(`√(g·k·tanh(k·d))`, `linspace(-0.6,0.6,64)`, `g` 는 `dispersion.G` 임포트)해
`SHIP` 행 ⟹ `d=50` 과 `allclose` **이고** `d=depth_config` 와 **불일치**, `FIX` 행 ⟹ 그 반대 — 를 요구한다.
러너가 어떤 값을 넣었든 **소비자(`beamform`)가 실제로 본 전역**을 기록하므로 항진이 아니다: 모듈 정체가 어긋나면
(§3.3) 래퍼가 안 걸려 `None` 이 남거나 d=50 해시가 남아 **FAIL** 한다.

### 3.3 ⛔비채택
| 방법 | 왜 안 되는가 |
|---|---|
| `bn.DEPTH = 400` 만 | **0회 발화** — §2 실측. `:43` 이 로드 시 1회만 돈다 |
| `importlib.reload(d3_beamform_net)` | 같은 `__dict__` 를 재실행하므로 기술적으로는 닿지만 `:18-19` 의 `sys.path.insert`·**`os.chdir`** 와 무거운 임포트(`d6`·`d15`·`d23`·`d40`)를 재실행하고, `cd.DEPTH`(값 복사)는 그대로 50 이라 **두 상수가 갈린다**. 비권장 |
| `scripts.d3_beamform_net` 처럼 **패키지 경로로 임포트**해 패치 | `sys.modules` 에 **다른 모듈 객체**가 생겨 `cd.beamform.__globals__` 에 닿지 않는다 = 0회 발화. `_load_cond` 가 쓰는 **bare name** 과 같은 이름으로 잡아야 한다(§2 `cd is sys.modules[...]: True`) |
| `config['learned_atoms']` 에 러너가 계산한 제안을 넣기 | 도달은 하나 **하네스가 제안을 계산**하는 팔이 되고 `G-ATOMS` 의 `assert 'learned_atoms' not in cfg` 규율과 충돌. 처치 범위가 「맵 한 텐서」보다 커진다 |
| 패키지 수정(`beamform(P, z, depth)`) | ⛔설계 단계 금지 + **출하 팔이 아니게 된다**. §5 에 미래 형태만 적는다 |

### 3.4 임포트 순서·부작용
- 러너는 `learned_init._load_cond` 와 **같은 방식**으로 먼저 임포트한다: `sys.path.insert(0, <repo>/scripts)` → `import d3_beamform_net as bn` →
  `import d3_beamform_cond_diffusion as cd`. 이후 `_load_cond` 는 `sys.modules` 의 같은 객체를 받는다(§2).
- `d3_beamform_net.py:19` 가 **`os.chdir(<repo>)`** 한다 ⟹ 러너는 절대경로만 쓴다(T11 러너가 그렇다).
- 첫 `_load_cond` 이전에 패치하려면 위 임포트가 먼저 있어야 한다. `propose_atoms` 는 `device` 를 **스스로** 고른다(`learned_init.py:133-134`,
  T11 공시) ⟹ `SHIP`·`FIX` 가 같은 프로세스에서 돌면 같은 장치다. 게이트 `G-DEVICE` 가 팔 간 동일을 되읽는다.

---

## 4. 앵커 가능성 — T11 동결 바이트가 **지금도** 패키지와 일치한다
T11 `FROZEN_md5.txt`(결과 0행 동결)의 임포트 8파일 + ckpt 를 현재 디스크와 대조(2026-09-04):
```
39d7c33c…  wavesplat/optimizer/sequential_update.py   ✅
5ebcf5a3…  wavesplat/optimizer/learned_init.py         ✅
039e2273…  wavesplat/optimizer/varpro.py               ✅
3e4a6fe1…  wavesplat/model/spectral_gaussian.py        ✅
88fe53a8…  wavesplat/model/dispersion.py               ✅
d0c4422e…  wavesplat/run.py                            ✅
2401025a…  scripts/d3_beamform_net.py                  ✅
4926acd6…  scripts/d3_beamform_cond_diffusion.py       ✅
636c72a7…  beamform_cond_diffusion_mixed_v1.pt (sha256) ✅
```
`results_t11.jsonl` md5 `28616470c636e780529b91fad48a2fe4`(커밋 `7f6debc`) · `sample_t11.json` md5 `56a0adc053b046081d538aa15b381766`.
⟹ `SHIP`·`RESCALE` 를 같은 시드·같은 순서로 다시 돌리면 T11 값과 **비트 동일**해야 한다 = `G-ANCHOR` 의 근거.
⚠단 T11 제안은 **cuda** 에서 돌았고 GPU 비트결정성은 T11 이 검사하지 않았다 ⟹ PREREG §10 스모크의 `G-REPEAT`(같은 파일 2회)가
**먼저** 반복 재현성을 세운다. 그것이 실패하면 앵커 실패는 「드리프트」가 아니라 「GPU 비결정성」이고, 그 경우의 처방은 사전에 적어 둔다(PREREG §8).

시드 규약(앵커 재현에 필수): T11 `run_t11.py:289 enumerate(files, 1)` · `:297 rng = default_rng(G.SEED + fi)` · `:327 seed=G.SEED + fi`,
`files = sample_t11.json` **순서**. ⟹ T13 은 T11 24파일에 **같은 `fi`(= `sample_t11.json` 의 1-based 위치)** 를 써야 한다.

---

## 5. (⛔이 배치 밖) 패키지 수리의 형태와 그 함의
- 형태: `beamform(P, z, om_grid=None)`(None 이면 현행 전역) · `_bmap_from_obs(Ps, z, depth=None)` · `propose_atoms` 가 `depth` 를 넘김 ·
  학습 스크립트는 `DEPTH` 를 명시 전달. 회귀 시험: (i) d=50 config 의 맵이 **오늘과 비트 동일** (ii) d=400 config 의 맵이 공식과 일치·d=50 과 불일치.
- ★함의: 자국 합성 코퍼스 **110 에피소드 전부 `depth=50`**(h5 실측) · `configs/default.yaml:39`·`estimation.yaml:7` 도 50
  ⟹ 그 수리는 **모든 기존 합성 판정에 비트 무연산**이다(맵이 이미 정합). 영향받는 것은 **외부 코퍼스 배치(T11·T13)뿐**.
- ⛔그래도 배포 결정은 **별개**다: T13 결과 + `test_shipping_defaults` 핀 갱신 + 배포-후 게이트 출력 첨부(08-23 교훈)가 필요하다.
