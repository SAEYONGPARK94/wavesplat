# M4 개정 사전등록 1 — 게이트 두 개의 **명세 결함** 수리 (2026-08-28)

> **지위**: 이 문서는 `PREREG.md` 의 **개정**이다. `PREREG.md` 는 한 글자도 고치지 않는다.
> 둘이 어긋나면 **더 늦은 문서(이것)가 어느 항목을 덮는지 §2 가 축자로 적는다.**
> 진단서 = `AMENDMENT_1.md`(269행). ⛔진단서는 **결함의 기술**이고, 이 문서가 **처방의 동결**이다.
>
> **작성 시점의 상태(숨기지 않는다)**: 배치는 이미 돌았고 집계도 끝났다 —
> `PRIMARY CAPACITY-HURTS 12/12` · `SECONDARY SYN-CAPACITY-HURTS 48/48` · 게이트 **46/48**.
> ⟹ ★**나는 방향을 본 뒤에 게이트를 고치고 있다.** §1 이 그 긴장을 회계한다.
> ⛔여기서 판정 코드를 **적용하지 않는다**(§5 가 집행 순서다). 이 문서가 먼저 동결된다.

---

## 0. 한 줄

**게이트 `G-13 real SHIP-K-INL` 은 자료가 무엇이든 통과할 수 없었고(항등적 실패),
`G-15` 는 사전등록의 의도(*"캐시가 값을 바꾸지 않았다"*)를 **끝단 nowcast** 로 구현해
추정기의 전역 RNG 결합을 함께 재고 있었다.** 이 개정은 전자를 **수리**하고 후자를
**원자 수준으로 내리면서, 끝단 동일성 주장 자체를 철회**한다.

---

## 1. ★긴장의 회계 — 무엇이 사후 자유도이고 무엇이 아닌가

### 1.1 사후 자유도가 **아닌** 것 (결과와 무관하게 항등적)

| 사실 | 왜 결과 독립인가 | 확인 방법(런 0회) |
|---|---|---|
| `G-13 real SHIP-K-INL` 은 **어떤 자료로도** 통과 불가 | 세는 집합이 `real = [… and not r.get('pc')]`(`gates_m4.py:480`)인데 그 팔의 행은 러너가 **예외 없이** `pc=True` 로 찍는다(`run_m4.py:224,256`) ⟹ `n≡0`, `want=12` | 코드 두 줄. 자료를 보지 않아도 증명된다 |
| 그 게이트의 출력은 **어떤 통계에도 들어가지 않는다** | `median_of_seeds:297` 이 `r.get('pc')` 행을 거른다 ⟹ `paired`·`skill`·`holm`·용량-반응표가 그 12행을 **보지 않는다** | `gates_m4.py:293-300` |
| 기대값 `12` 는 **동결 시점**에 정해져 있었다 | `len(RECORDS) × len(INLINE_SEEDS)` = 12×1, `PREREG §4.2` | 동결본 `PREREG.md` |
| `G-15` 가 재던 양이 **원자가 아니라 끝단**이라는 것 | `repr(inl[k]['nowcast']) == repr(base[k]['nowcast'])`(`gates_m4.py:521-526`) | 코드 |
| 설계 전제(*"캐시 = 인라인"*)는 **게이트보다 강하게** 확인됐다 | 진단서 §3.1 이 원자 배열을 직접 12/12 비교, `maxΔ = 0.000e+00`, 전역 RNG 를 12345 draw 흔든 뒤에도 동일 | `AMENDMENT_1.md §3.1` |

⟹ **A-1 과 A-3 은 사후 자유도가 아니다.** A-1 은 항등식을 고치는 것이고(고쳐도 어떤 수도
움직이지 않는다), A-3 은 *"게이트가 만족 가능한가"* 를 시험하는 장치일 뿐 판정을 만들지 않는다.

### 1.2 사후 자유도인 것 — ★**이 절을 지우지 마라**

1. ★**A-2 는 사후 결정이다.** `G-15` 를 **끝단 → 원자**로 내리는 결정을 나는
   *"끝단 형태가 10/12 로 실패했다"* 를 **본 뒤에** 한다. 방향은 **배치의 게이트가
   통과하는 쪽**이다. 기제(전역 RNG · 성장 가지 12/12 대응)가 독립적으로 측정됐다는 것은
   *결정의 정당화*이지 *결정의 맹검*이 아니다. ⛔**둘을 같은 것으로 쓰지 마라.**
2. ★**"VOID 하지 않는다" 도 결과를 본 뒤의 선택이다.** VOID 는 유일하게 기계적으로
   맹검인 처분이고, 나는 그것을 고르지 않았다(사유 = 진단서 §5.2: 840행 무손상, 결함은
   게이트 명세 두 곳). **선택했다는 사실 자체를 기록한다.**
3. ★**나는 `PRIMARY` 를 이미 안다**(12/12, `rel_median +1527.8%`). 그러므로 이 개정이
   PRIMARY 를 건드리지 않는다는 것을 **논증이 아니라 게이트로** 요구한다(§3 `G-A1`).
4. ⚠**새 게이트를 결과 뒤에 추가하는 것 자체**가 자유도다. 그래서 이 개정이 추가하는
   게이트는 **전부 더 엄격한 방향**이고, 그중 **둘은 새로 실패할 수 있다**
   (`G-13pc-repro`·`G-A1`). 완화되는 것은 `G-15` **하나뿐**이고 그것이 §1.2.1 이다.

### 1.3 ★반대 논거 — 그리고 내가 답할 수 없는 것

| 반대 논거 | 내 답 | 남는 것 |
|---|---|---|
| **(가)** *"방향을 보고 게이트를 고쳤다. 반대 결과였어도 같은 수리를 썼겠는가는 검증 불가다."* | A-1 은 코드만으로 증명되므로 이 반론이 닿지 않는다. **A-2 에는 닿는다.** | ⛔**A-2 에 대해서는 반박하지 않는다.** 그래서 A-2 의 산출은 *"게이트 통과"* 가 아니라 **좁아진 주장**이다(§4). |
| **(나)** *"실패하는 시험을 통과하는 시험으로 갈아치웠다."* | 부분적으로 **사실이다**. 끝단 동일성은 더 강한 성질이고 **거짓으로 측정됐다**(10/12 불일치). 우리는 그것을 **주장하지 않는다**. | 보상은 하나뿐 — ★**실패한 측정을 판정문에 영구히 남긴다**(`G-15x`, 판정 아님). 지우고 대체하지 않는다. |
| **(다)** *"진짜 수리는 게이트가 아니라 추정기다."* | **동의한다.** `sequential_update.py:1617` 이 성장 위상을 **전역 torch RNG** 에서 뽑는 한 `(원자, 시드)` 의 순수함수가 아니다. 전용 `Generator` 로 바꾸면 강한 시험이 **가능해진다**. | 그것은 **배포 거동 변경**이라 별도 사전등록이다(진단서 §5.4). ⟹ 오늘 우리는 **더 약한 측정을 골랐다**고 기록한다. |
| **(라)** *"게이트가 하나 더 필요하다: 팔별 행 수를 필터 위에서 세지 말고 **작업 목록과 대조**하라."* | 옳다. 그것이 `G-13all` 인데 **이 개정에 넣지 않는다** — 결과를 본 뒤의 **재설계**이고, A-1 이면 결함은 닫힌다. | 다음 사전등록 항목으로 **남긴다**(§7). |
| **(마)** *"양성대조 12런을 다시 돌리면 오늘의 작업트리로 도는데, 그 사이 임포트 경로 파일이 움직였다."* | **사실이다**(§6). 그래서 재실행은 **재현 게이트를 동반**한다(`G-13pc-repro`). | 실패하면 ⛔**패치 금지**·보고(§7). |

---

## 2. `PREREG.md` 중 이 개정이 덮는 문장 (축자)

| PREREG 위치 | 원문 | 이 개정 이후 |
|---|---|---|
| §4.2 대조 2 | *"`SHIP-K-INL` … ★**비트동일 12/12 요구**(`G-15`) = *"캐시가 값을 바꾸지 않았다"* 의 증명."* | **원자 배열 비트동일 12/12 요구**로 바꾼다. **끝단(nowcast) 동일성은 요구하지 않고 주장하지도 않는다**(측정 결과 10/12 불일치, 사유 = 전역 RNG). |
| §8 [1] | *"런0 게이트를 실제 파일에 대고 돌린다 — ★`M2` 의 `G-13` … 재발하지 않도록"* | ★**런타임 게이트에도 같은 시험을 건다**(A-3). 종전 자가진단은 `gates_runfree()` 와 0행 가드만 봤다. |
| §8 변형 하네스 | *"② 일관된 세계 → **48/48 PASS**"* | ⛔**그 하네스는 러너가 아니라 게이트를 모형했다**(§3 A-3 머리말). 그 문장은 *"가짜 행이 게이트의 가정과 일관됐다"* 로만 읽어야 한다. |

⛔**문턱·팔·표본·라벨·PRIMARY·해석 규칙(`READING`)·`DIRECTION`·`DISCLOSURE` 는 불변이다.**

---

## 3. 개정 항목 — 적용할 코드(축자)

### A-1 `G-13` 분리 + `G-13pc` 신설 (사후 자유도 **아님**)

`gates_m4.py:112` 를

```python
REAL_ARMS = LADDER + ['SHIP-K-LSC', 'K-MAX-LSC', 'K-8-NOGROW', 'SHIP-K-INL']
```

로 두는 대신 **집합을 스탬프대로 가른다**(⛔`REAL_ARMS` 의 유일한 사용처는 `:483` 이다):

```python
REAL_ARMS = LADDER + ['SHIP-K-LSC', 'K-MAX-LSC', 'K-8-NOGROW']   # ⛔pc 팔은 여기 없다
PC_ARMS = ['SHIP-K-INL']          # ★러너가 `pc=True` 로 찍는 팔 (run_m4.py:224,256)
```

`gates_m4.gates_run()` 의 `G-13` 블록(`:479-487`)을

```python
    real = [r for r in ok if r.get('stratum') == 'real' and not r.get('pc')]
    syn = [r for r in ok if r.get('stratum') == 'syn' and not r.get('pc')]
    pc = [r for r in ok if r.get('pc')]        # ★양성대조는 **다른 집합**이다
    # G-13 팔별 행 수 — ⛔`real` 은 pc 행을 배제하므로 인라인 팔을 여기서 세면 항등 실패다
    for a in REAL_ARMS:
        want = len(RECORDS) * len(EST_SEEDS)
        n = sum(1 for r in real if r['arm'] == a)
        out.append((f'G-13 real {a}', n == want, f'{n}/{want}'))
    # ★G-13pc — 세는 집합이 러너의 스탬프와 **같은 쪽**이다
    for a in PC_ARMS:
        want = len(RECORDS) * len(INLINE_SEEDS)
        n = sum(1 for r in pc if r['arm'] == a)
        out.append((f'G-13pc {a}', n == want, f'{n}/{want}'))
```

로 바꾼다. **기대값은 종전과 같은 산술**(`12×3` · `12×1`)이고 손복사 상수는 0개다.

### A-2 `G-15` 를 **원자 md5·최대편차**로 내린다 (★사후 자유도 — §1.2.1)

**러너**(`run_m4.py`) — 인라인 제안을 **포착**해 행에 증거를 남긴다.
현재는 `atoms_md5=None`(`:146`)이라 원자 대조가 **불가능**하다.

```python
_PROPOSE = [0]
_LAST_PROPOSAL = [None]        # ★인라인 제안 원자 (양성대조 A-2)


def _install_propose_counter():
    import wavesplat.optimizer.learned_init as LI
    if getattr(LI.propose_atoms, '_m4_count2', False):
        return
    real = LI.propose_atoms

    def counting(*a, **kw):
        _PROPOSE[0] += 1
        r = real(*a, **kw)
        try:                    # ⛔포착 실패가 추정을 죽이면 안 된다
            _LAST_PROPOSAL[0] = dict(kx=np.asarray(r[0], float).tolist(),
                                     ky=np.asarray(r[1], float).tolist(),
                                     energy=np.asarray(r[2], float).tolist())
        except Exception:       # noqa: BLE001
            _LAST_PROPOSAL[0] = None
        return r

    counting._m4_count2 = True
    LI.propose_atoms = counting
```

`run_real()` 에서 `_PROPOSE[0] = 0` 옆에 `_LAST_PROPOSAL[0] = None` 을 넣고,
**추정과 채점이 끝난 뒤**(⟹ 추정에 영향 0) 반환 dict 에 다음을 합친다:

```python
    cmp_inline = {}
    if k['inline'] and _LAST_PROPOSAL[0] is not None:
        cached = json.load(open(os.path.join(ATOMS_REAL,
                                             f'{record}__s{seed}.json')))['atoms']
        ia = _LAST_PROPOSAL[0]
        d = (max(float(np.max(np.abs(np.asarray(ia[f], float)
                                     - np.asarray(cached[f], float))))
                 for f in ('kx', 'ky', 'energy'))
             if len(cached['kx']) == len(ia['kx']) else None)
        cmp_inline = dict(atoms_md5_inline=_md5(ia), atoms_md5_cached=_md5(cached),
                          atoms_maxabs_diff=d, n_atoms_proposed=len(ia['kx']),
                          n_atoms_cached=len(cached['kx']))
    return dict(..., **cmp_inline)
```

⚠**두 md5 는 같은 함수**(`gates_b22.atoms_md5`)로 낸다 — 규약이 다르면 대조가 성립하지 않는다
(`run_m4._md5` 가 이미 그 함수를 쓴다). ⚠**캐시 파일은 추정 뒤에 읽는다**: 인라인 팔이
캐시를 *쓴* 것이 아니라 *대조한* 것임을 순서로 보장한다(`G-17` 은 종전대로 확산 호출을 센다).

**게이트**(`gates_m4.gates_run`) — `:520-526` 을 바꾼다:

```python
    # ★G-15 캐시 양성대조 — **원자 배열** 비교 (PREREG:175 의 의도)
    inl = [r for r in ok if r['arm'] in PC_ARMS and r.get('pc')]
    want = len(RECORDS) * len(INLINE_SEEDS)
    same = [r for r in inl
            if r.get('atoms_maxabs_diff') == 0.0
            and r.get('atoms_md5_inline') is not None
            and r['atoms_md5_inline'] == r.get('atoms_md5_cached')
            and r.get('n_atoms_proposed') == r.get('n_atoms_cached')]
    out.append(('G-15 캐시 원자 == 인라인 제안 원자 (비트동일)',
                len(inl) == want and len(same) == want,
                f'{len(same)}/{len(inl)} (기대 {want})'))
    # ★G-15x 공시(판정 아님) — **끝단** nowcast 는 비트동일하지 **않다**.
    #   추정기 성장 가지가 전역 torch RNG 를 쓴다(sequential_update.py:1617) ⟹
    #   인라인 팔은 확산 체인을 소모한 상태로 그 지점에 도착한다. 이 수를 지우지 마라.
    basem = {(r['unit'], r['est_seed']): r for r in ok
             if r['arm'] == BASE and not r.get('pc')}
    same_e = [r for r in inl
              if (r['unit'], r['est_seed']) in basem
              and repr(r.get(FIELD_REAL))
              == repr(basem[(r['unit'], r['est_seed'])].get(FIELD_REAL))]
    out.append(('G-15x 공시: 끝단 nowcast 가 기준선과 동일한 셀 (판정 아님)',
                True, f'{len(same_e)}/{len(inl)} — ⛔이 배치는 끝단 동일성을 주장하지 않는다'))
```

**신설 `G-13pc-repro`** — 재실행이 개정 전 12행을 **비트 재현**하는가(§6 의 동결 이탈 때문에
반드시 필요하다):

```python
PC_FROZEN = os.path.join(HERE, 'frozen_pc_pre_amend1.jsonl')   # 개정 전 pc 12행

    # ★G-13pc-repro — 개정 전 행과 비트동일 (⛔실패 시 패치 금지, §7)
    old = {}
    if os.path.exists(PC_FROZEN):
        for line in open(PC_FROZEN):
            if line.strip():
                r0 = json.loads(line)
                old[(r0['arm'], r0['unit'], r0['est_seed'])] = r0
    rep = [r for r in inl
           if (r['arm'], r['unit'], r['est_seed']) in old
           and repr(r.get(FIELD_REAL))
           == repr(old[(r['arm'], r['unit'], r['est_seed'])].get(FIELD_REAL))]
    out.append(('G-13pc-repro 재실행이 개정 전 pc 행을 비트 재현',
                bool(old) and len(rep) == want, f'{len(rep)}/{len(old)} (기대 {want})'))
```

**신설 `G-A1` (판정 불변)** — 개정이 PRIMARY 를 움직이지 않았음을 **논증이 아니라 비트로**:

```python
PRE_AMEND_VERDICT = os.path.join(HERE, 'verdict_m4_pre_amend1.json')
AMEND_INVARIANT_KEYS = ('n', 'better', 'tie', 'rel_median_pct', 'p_one_sided')


def gate_amend_invariance(label, primary, label_syn, syn_primary):
    """★개정 전/후 판정이 **비트 동일**한가. 개정이 통계를 건드렸다면 여기서 죽는다."""
    if not os.path.exists(PRE_AMEND_VERDICT):
        return ('G-A1 개정 전 판정과 비트 동일', False, '개정 전 verdict 사본이 없다')
    v0 = json.load(open(PRE_AMEND_VERDICT))
    bad = [k for k in AMEND_INVARIANT_KEYS
           if repr(v0['primary'].get(k)) != repr(primary.get(k))]
    bad += [f'syn:{k}' for k in AMEND_INVARIANT_KEYS
            if repr((v0['syn'].get(PRIMARY_ARM) or {}).get(k))
            != repr((syn_primary or {}).get(k))]
    ok = (not bad and v0['label'] == label and v0['label_secondary'] == label_syn)
    return ('G-A1 개정 전 판정과 비트 동일', ok,
            f"{v0['label']}/{v0['label_secondary']} vs {label}/{label_syn} · 이탈 {bad}")
```

`analyze_m4.py` 는 `gates` 목록 끝에 한 줄을 더한다(라벨 계산 뒤):

```python
    gates.append(list(G.gate_amend_invariance(lab, prim, slab, syn.get(G.PRIMARY_ARM))))
```

### A-3 ★자가진단을 **런타임 게이트 만족가능성**까지 올린다 (M2 교훈의 런타임판)

★**왜 종전 하네스가 못 잡았나 — 이것이 이 개정에서 가장 중요한 문장이다.**
`PREREG §8` 은 *"가짜 행 840개 … 일관된 세계 → 48/48 PASS"* 라 적었다. 그런데 오늘 자료에서
`G-13 real SHIP-K-INL` 은 **항등적으로 0/12** 다. 두 사실이 동시에 참이려면 그 가짜 행에는
***`pc` 스탬프가 없는*** `SHIP-K-INL` 행 12개가 있었어야 한다 ⟹
★***그 하네스는 러너가 아니라 게이트를 모형했다.*** 손으로 적은 세계는 **의도표**이고,
이 저장소가 `effective_config` 에서 이미 대가를 치른 부류다.

⟹ **처방은 "합성 행 시험을 추가하라" 가 아니다**(그것은 이미 있었고 통과했다).
⟹ ★***"합성 행을 러너의 작업 목록과 스탬프에서 만들어라"*** 다.

**단일출처화**(`gates_m4.py`) — 러너가 이것을 부르고, 자가진단도 이것을 부른다:

```python
def expected_jobs(stratum):
    """★러너의 작업 목록 **단일출처**. `run_m4.jobs()` 가 이 함수를 부른다."""
    if stratum == 'real':
        return [(a, rec, s) for a in REAL_ARMS for rec in RECORDS for s in EST_SEEDS]
    if stratum == 'syn':
        eps = sorted(e['episode'] for e in syn_episodes())
        return [(a, e, s) for a in SYN_ARMS if a != SYN_IMPORTED_BASE
                for e in eps for s in EST_SEEDS]
    if stratum == 'pc':
        return [(a, rec, s) for a in PC_ARMS for rec in RECORDS for s in INLINE_SEEDS]
    raise ValueError(stratum)


def row_identity(stratum, arm, unit, seed):
    """★행의 **식별 스탬프** 단일출처 — `pc` 는 여기서만 붙는다.
    ⛔러너와 게이트가 각자 알고 있으면 그것이 `G-13` 을 낳은 결함이다."""
    row = dict(stratum=('real' if stratum == 'pc' else stratum), arm=arm,
               unit=unit, est_seed=int(seed))
    if stratum == 'pc':
        row['pc'] = True
    return row
```

`run_m4.jobs(stratum)` 은 **`return G.expected_jobs(stratum)`** 으로 줄이고
(⚠현행과 **순서·구성이 같음**을 확인했다: real 11팔×12×3=396 · syn 3팔×48×3=432 · pc 12),
`main()` 의 `if stratum == 'pc': row['pc'] = True` 는
**`row.update(G.row_identity(stratum, arm, unit, s))`** 로 바꾼다.

**합성 세계**(`gates_m4.synthetic_rows()`) — 위 두 함수로 **정상 세계**를 만든다.
게이트가 읽는 필드를 일관되게 채우고, `pc` 행의 `nowcast` 는 `PC_FROZEN` 에서 읽는다
(⟹ `G-13pc-repro` 도 합성 세계에서 시험된다. **면제 게이트 0개**).

**자가진단 추가 항목**(`selftest()`):

```python
    # [11] ★런타임 게이트 만족가능성 — 정상 세계에서 **전 게이트가 통과**해야 한다
    srows = synthetic_rows()
    gr = gates_run(srows)
    chk(all(okv for _, okv, _ in gr),
        '합성 정상 세계에서 FAIL: ' + str([n for n, o, m in gr if not o]))
    # [12] ★그 시험이 공허하지 않은가 — 팔 하나를 지우면 **반드시** 누가 FAIL 해야 한다
    for a in ('SHIP-K-INL', 'K-MAX', 'K-8', 'SHIP-K'):
        g2 = gates_run([r for r in srows if r['arm'] != a])
        chk(any(not okv for _, okv, _ in g2), f'{a} 를 지웠는데 전 게이트 통과 = 공허')
```

★**[12] 는 저장소 방벽 `tmp/_audit/check_gate_blindspots.py` 와 정확히 같은 시험이다**
(삭제 민감도). 하나는 배치 안에서 **사전등록 시점**에, 하나는 저장소 전체에서 **사후**에 돈다.

---

## 4. ★이 개정 이후 무엇을 주장할 수 있는가 (결과 전 고정)

- ✅**주장한다**: *"캐시한 원자는 인라인 제안 원자와 **비트 동일**하다(12/12)."*
  ⟹ 사다리 팔들이 **같은 원자 풀**을 나눠 썼다는 설계 전제는 지탱된다(`G-19` 와 함께).
- ⛔**주장하지 않는다**: *"인라인 팔과 캐시 팔이 **같은 추정치**를 낸다."* — **거짓이다**
  (10/12 불일치, `G-15x` 가 그 수를 판정문에 남긴다). 사유는 추정기의 전역 RNG 결합이고
  그것은 **이 배치의 처치가 아니다**.
- ⛔**주장하지 않는다**: *"개정으로 배치가 인용 가능해졌다."* — 인용 가능 여부는
  게이트 전건 통과 + 짝 verdict 의 문제이고, **그 판단은 이 문서가 아니라 개정 실행 뒤의
  `verdict_m4.json` 과 사람이 한다.** 이 문서는 **처방만** 동결한다.
- ⚠**병기 의무**: `CAPACITY-HURTS` 를 인용할 때 *"양성대조는 원자 수준에서만 성립하고
  끝단 동일성은 성립하지 않는다(전역 RNG)"* 를 같이 적는다.

---

## 5. 집행 순서 (⛔이 순서를 바꾸지 마라)

```bash
cd /home/vexit/wavesplat/wavesplat

# 1) 개정 **전** 상태를 증거로 고정한다 (⛔덮어쓰기 금지)
cp tmp/M4_CAPACITY_2026-08-28/verdict_m4.json \
   tmp/M4_CAPACITY_2026-08-28/verdict_m4_pre_amend1.json
mv tmp/M4_CAPACITY_2026-08-28/results_m4_9.jsonl \
   tmp/M4_CAPACITY_2026-08-28/frozen_pc_pre_amend1.jsonl
#    ⚠이름이 `results_m4*.jsonl` 밖이어야 분석기 glob 에 두 번 들어가지 않는다.
#    ⚠러너의 재개 로직(`done` 집합)이 옛 파일을 보면 **0런**이 된다 — 그래서 옮긴다.

# 2) 이 문서와 1)의 산출물 md5 를 `FROZEN_md5_amend1.txt` 에 기록 (§8)

# 3) 판정 코드에 A-1·A-2·A-3 적용 → 자가진단 (⛔실패하면 §7)
python3 tmp/M4_CAPACITY_2026-08-28/gates_m4.py --selftest

# 4) 양성대조 **12런만** 재실행 (CPU · 약 100 s · ⛔GPU 금지 · ⛔real/syn 재실행 금지)
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 \
  python3 tmp/M4_CAPACITY_2026-08-28/run_m4.py pc

# 5) 재집계 (⛔4)보다 먼저 돌리지 마라 — pc 행 0개로 G-13pc 가 실패한다)
python3 tmp/M4_CAPACITY_2026-08-28/analyze_m4.py

# 6) 방벽
python3 tmp/_audit/check_batch_artifacts.py tmp/M4_CAPACITY_2026-08-28
python3 tmp/_audit/check_gate_refs.py       tmp/M4_CAPACITY_2026-08-28
python3 tmp/_audit/check_gate_blindspots.py tmp/M4_CAPACITY_2026-08-28
```

⛔**하지 말 것**: 실측 396런·합성 432런 재실행(진단서 §3.1·§3.2 가 순서 무관을 직접 보였다) ·
`probe_m4.py` 재실행(원자 풀이 움직이면 `FROZEN_md5_phaseA.txt` 가 깨진다) ·
GPU(다른 세션이 `M1` 생성 중) · 문턱·팔·표본·라벨 변경.

---

## 6. ⚠공시 — 동시 세션이 **임포트 경로 파일**을 바꿨다 (동결 이탈, 개정 전 발견)

`tmp/B2_REAL_2026-08-23/scene_station.py` 가 **2026-08-28 14:53** 에 수정됐다.
M4 의 828행은 **14:16–14:27** 에 났고 동결은 **12:50** 이다.

| | md5 |
|---|---|
| `FROZEN_md5.txt:34` (동결 시점 = 행이 난 시점) | `18183f898008d239db002a3634f19686` |
| 현재 작업트리 | `68dcd2073c4ec60897645faa29b16172` |

내용은 `cap_mode` 인자 추가(`M5` 진단의 처방)이고 **기본값 `'uniform'` 은 종전과 비트 동일**
이라고 그 diff 가 **주장**한다. ⟹ ★***주장을 판정에서 재는 것이 이 저장소의 규율이다***
(`B-2` 수심 교훈). **`G-13pc-repro` 가 정확히 그 시험**이 된다 — 재실행 12행이 개정 전 12행을
비트 재현하면 그 주장이 M4 경로에서 확인되고, 실패하면 **그 편집이 M4 의 장면을 바꿨다**는
발견이다(⟹ §7).

⛔**다른 세션의 파일을 되돌리지 않는다.** ⚠`check_batch_artifacts` 는 이미 이 이탈을
`★자기동결` 열에 찍고 있다 — **재동결로 지우지 마라**(그러면 열이 `OK` 로 돌아가지만
사실은 지워지지 않는다).

---

## 7. 실패 시 규칙 (결과 전에 고정)

- **자가진단 [11] 이 실패하면** ⟹ ⛔**게이트를 완화하지 마라.** 합성 세계가 러너의 출력과
  다른 것이거나(⟹ `synthetic_rows` 를 고친다), 게이트가 여전히 만족 불가인 것이다
  (⟹ **그 사실을 보고한다**). 어느 쪽인지 **적어라**.
- **`G-13pc-repro` 가 실패하면** ⟹ ⛔**패치 금지.** §6 의 편집이 장면을 바꿨다는 뜻이고,
  그때는 M4 의 828행이 **현재 트리에서 재현되지 않는다**. 이 개정을 중단하고 보고한다.
- **`G-A1` 이 실패하면** ⟹ ⛔**개정이 통계를 건드렸다.** 즉시 되돌린다(개정 전 verdict 사본이
  §5-1 에 있다).
- **개정 후에도 FAIL 이 남으면** ⟹ ⛔**두 번째 개정을 쓰지 마라.** 남은 FAIL 을 **그대로 실은 채**
  보고한다. ★*"통과할 때까지 개정한다"* 는 `B-5` 가 이미 이름 붙인 위험
  (*"질 때까지 조율"*)의 거울상이다.
- **다음 사전등록으로 미룬 것**: `G-13all`(행 식별자 다중집합 == 작업 목록) ·
  추정기 RNG 위생(전용 `Generator`) · `apg_dn` 순증감이 성장 발화를 숨기는 문제(진단서 §3.4) ·
  `learned_init.py:129-130` 독스트링 정정.

---

## 8. 동결

이 문서가 **먼저** 동결되고, 그 다음에 코드가 바뀐다.

```bash
cd /home/vexit/wavesplat/wavesplat/tmp/M4_CAPACITY_2026-08-28
{ echo '# M4 AMENDMENT 1 동결 (2026-08-28). ⛔개정 코드 적용 **전** 상태.'
  echo '# 개정 전 판정·양성대조 행·판정 코드의 md5. 개정 후 md5 는 아래 표와 다르다.'
  md5sum AMENDMENT_1.md AMENDMENT_1_PREREG.md PREREG.md \
          verdict_m4_pre_amend1.json frozen_pc_pre_amend1.jsonl \
          gates_m4.py run_m4.py analyze_m4.py probe_m4.py
} > FROZEN_md5_amend1.txt
```

**개정 전 판정 코드 md5**(개정 후 이 셋은 **바뀐다** — 바뀌는 것이 개정이다):

| 파일 | 개정 전 md5 |
|---|---|
| `gates_m4.py` | `40ede67930173114cf4547237205ab19` |
| `run_m4.py` | `c65216c6eaf2e363148552bc64e3fa62` |
| `analyze_m4.py` | `1f7975088cbd3776a35dbe175f1acd2c` |
| `probe_m4.py` (⛔불변이어야 한다) | `8eeca60956eced7b7b1bc7717a928359` |
| `results_m4_9.jsonl` → `frozen_pc_pre_amend1.jsonl` | `cf295e7e7449ec298520cbc361729df6` |

★**`probe_m4.py` 와 `atoms_real/` 의 md5 가 움직이면 이 개정은 무효다** — 원자 풀은
`FROZEN_md5_phaseA.txt` 에 Phase B 결과 0행 시점으로 묶여 있고, A-2 의 대조는 **그 파일들**을
기준으로 한다.
