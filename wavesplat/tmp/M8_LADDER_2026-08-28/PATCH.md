# M-8 패키지 패치 명세 — **가산 2키, 기본값 보존형** (⛔아직 적용하지 않았다)

> 사다리의 R2–R5 는 이 패치 없이는 **도달 불가**다(`PREREG §1.2` A2·A3).
> ⛔**본 배치를 돌리기 전에 적용 + 회귀 통과 + 동결**이어야 한다.
> 선례: `k_snap_dk`(B-16) · `bank_refit_every_s`(D) · `cap_mode`(M5b) — 셋 다 기본값-꺼짐.

---

## 1. `wavesplat/optimizer/bank_mmse.py :: fit_bank_mmse`

시그니처에 **뒤에** 두 인자를 붙인다(위치인자 순서 불변):

```python
def fit_bank_mmse(P, z, depth, sigma0, kp_hint=None, dk_c=None, val_frac=0.25,
                  seed=0, max_em=6, band=(0.35, 2.5),
                  dk_frac=None, k_set=None):
```

후보집합 구성부(`:152-158`)만 갈아끼운다. **그 외 한 줄도 건드리지 않는다** —
val 분할·beamform zoom·γ 사전·pexp/smult 격자·SBL-EM·최종 solve 는 전부 축자 유지:

```python
    if dk_c is None:
        dk_c = kp / 10.0 if dk_frac is None else float(dk_frac) * kp
    if k_set is not None:
        # 격자 대신 주어진 후보. dk_c 는 의미를 잃으므로 NaN 으로 적어 둔다(⛔0.0 금지 —
        # 0.0 은 "간격이 0" 처럼 읽힌다). 나머지 파이프라인은 kxb/kyb 만 본다.
        kxb = np.asarray(k_set[0], float).ravel()
        kyb = np.asarray(k_set[1], float).ravel()
        if len(kxb) == 0 or len(kxb) != len(kyb):
            raise ValueError('k_set must be a non-empty (kx, ky) pair of equal length')
        dk_c = float('nan')
    else:
        b_lo, b_hi = band
        ax = np.arange(-b_hi * kp, b_hi * kp + dk_c / 2, dk_c)
        KX, KY = np.meshgrid(ax, ax)
        kk = np.hypot(KX, KY)
        sel = (kk > b_lo * kp) & (kk < b_hi * kp)
        kxb, kyb = KX[sel].astype(float), KY[sel].astype(float)
```

⚠**기본값 경로 확인**: `dk_frac=None, k_set=None` 이면 `dk_c = kp/10.0` 로 종전과 **같은 식**이고,
격자 생성 블록도 축자 그대로다 ⟹ **비트 동일**.

## 2. `wavesplat/optimizer/sequential_update.py :: _bank_fit_window`

```python
        self._bank = [fit_bank_mmse(P, z, self.depth, sigma0, kp_hint=None,
                                    seed=s,
                                    dk_frac=self.config.get('bank_dk_frac'),
                                    k_set=self.config.get('bank_k_set'))
                      for s in self._bank_seeds]
```

★그리고 **조용한 무시를 막는다** — `_initialize_inner` 의 라우팅 직전(또는
`SequentialUpdater.__init__`)에:

```python
        if (config.get('bank_dk_frac') is not None
                or config.get('bank_k_set') is not None) and \
                str(config.get('real_estimator', 'bank_mmse')) != 'bank_mmse':
            raise ValueError("bank_dk_frac / bank_k_set require real_estimator='bank_mmse'")
```

⛔*"설정을 넣었다 ≠ 설정이 먹었다"* — 이 저장소가 반복해서 대가를 치른 부류다.
(추가 안전장치는 `PREREG §9 G-INT` 가 **객체 되읽기**로 건다.)

## 3. `wavesplat/run.py :: DEFAULT_CONFIG`

⛔**추가하지 않는다.** 두 키는 **연구 전용**이고 배포 표면에 올리지 않는다
(`learned_merge_frac` 선례: 노출되지 않은 채로 `config=` 경로로만 도달).
⟹ `DEFAULT_CONFIG` 불변 ⟹ `gen_deploy_constants.py`·논문 매크로 **무영향**.

## 4. 동결 전 회귀 — `regress_bank_mmse.py` (추정기 런 0회, 네 겹)

| 겹 | 무엇을 |
|---|---|
| **L1** | 실측 12 레코드 × 시드 3 에서 **기본 인자** 적합의 `(kxb, kyb, om, gam, w_pred)` 가 패치 전 소스(git blob)의 결과와 **비트 동일** 36/36 |
| **L2** | `dk_frac=0.1` 이 기본과 **비트 동일**(부동소수 경로까지 같은 식인지) |
| **L3** | `k_set=(kx,ky)` 주입이 **손으로 만든 동일 적합**과 비트 동일 — 즉 주입 경로가 γ·EM·solve 를 바꾸지 않는다 |
| **L4** | ★**처치가 실제로 다르다**: `bank_B` 가 `frac` 을 따라 `1904–1918 → 472 → 118 → 30` 으로 움직인다(12/12) |

⚠**L1 의 기준선은 git blob 이어야 한다**(현재 working tree 가 아니라). `M5b` 의
`scene_pre_source()` 선례를 따른다.

## 5. 부작용 점검 (런 0회)

* `wavesplat/tests/test_bank_mmse_port.py` — 연구 함수 대비 비트 일치 핀. **계속 통과해야 한다.**
* `_bank_refit_async_kick` 의 `type(self)._bank_fit_window(shim, P, z)` 경로도 같은 config 를 본다 ⟹ 일관.
* `BankMMSEFit.dk_c` 가 NaN 이 될 수 있다 ⟹ `verbose` 인쇄(`_initialize_bank:1145`)의
  `round(f.dk_c, 4)` 는 NaN 을 그대로 찍는다(예외 아님). 결과 dict 에는 `dk_c` 가 없다(확인함).
* 다른 배치 재실행 시 검사기가 `bank_mmse.py`·`sequential_update.py` `⚠md5 변경` 을 찍는다 —
  **의도된 신호**이고, 붙일 문장 = *"기본값 경로는 비트 동일하고 증거는 `regress_bank_mmse.py` L1·L2 다."*
