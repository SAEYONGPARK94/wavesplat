"""M3b 동결 분석기 — ★**결과 파일보다 먼저 쓰였다.** 판정은 전부 `gates_m3b.py` 가 낸다.

이 파일이 하는 일은 셋뿐이다: (a) 행을 모으고 (b) 게이트를 **전부** 돌리고
(c) `gates_m3b` 의 라벨·읽기·분해를 **그대로 옮겨 적는다.**
⛔문턱·팔·읽기 규칙을 여기서 **정의하지 않는다**(단일출처는 `gates_m3b.py`).
⛔작업 개수를 손으로 적지 않는다 — `run_m3b.expected_jobs()/expect_counts()` 가 모형이다.

## ★결과 전에 선언하는 것 (이 파일에만 있는 것)
1. **0행 가드** — 유효 행 0이면 `verdict` 을 **쓰지 않고** `exit 1`.
2. **차단 게이트**(⟹ `BATCH-VOID`, 셀·주장 해석 금지) = `PREREG.md` §6 의 셋:
   `G-PCA`·`G-PCA2`(PC-A 비트불변) · `G-8`(PC-C 실효 수심) · `G-9`(PC-D ZERO 누출).
   ⚠`G-PCB`(PC-B)는 **차단이 아니다**(`gates_m3b.READING_PCB`).
3. **불완전 게이트**(⟹ `INCOMPLETE`, 주장 해석 금지) = `G-5`(에러 0) · `G-6a…e`(행수) ·
   `G-7`(에피소드 집합 일치). 표본이 안 찬 상태의 라벨은 라벨이 아니다.
4. **`S1-SHELL` 의 안/바깥 정의**(등록 부차, 방향은 `gates_m3b.SECONDARY` 가 선언):
   셸이 **9칸(홀수)** 이므로 가운데 `30_50` 을 **어느 절반에도 넣지 않고 따로 보고**한다.
   안쪽 = `0_5 … 20_30`(4칸) · 바깥 = `50_75 … 150_inf`(4칸). ⛔승격 금지.
   ★**커버리지 0 금지** — 어떤 셸의 짝이 0개면 계수기를 `0` 이 아니라 **`None`** 으로 적는다.
5. ⛔`shell_frac` 은 **팔 간 비교하지 않는다**(정의가 다르다 — `gates_m3b` 주석).

    python3 tmp/M3b_DEPTH_2026-08-29/analyze_m3b.py
"""
import glob
import json
import math
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gates_m3b as G                                          # noqa: E402
import run_m3b as R                                            # noqa: E402  ★작업 모형

BLOCKING = ('G-PCA', 'G-PCA2', 'G-8', 'G-9')
INCOMPLETE = ('G-5', 'G-6a', 'G-6b', 'G-6c', 'G-6d', 'G-6e', 'G-7')

SHELL_INNER = G.SHELL_BARE[:4]          # 0_5 · 5_10 · 10_20 · 20_30
SHELL_MID = G.SHELL_BARE[4]             # 30_50 — 9칸의 가운데. 어느 절반에도 넣지 않는다.
SHELL_OUTER = G.SHELL_BARE[5:]          # 50_75 · 75_100 · 100_150 · 150_inf


# ── 셸 보조 ──────────────────────────────────────────────────────────────────
def _shell_key(arm, bare):
    """ship 은 접두어 없음(`0_5`), spec 은 `obs:` 접두어. 경계는 동일."""
    return bare if G.ARMS[arm]['family'] == 'ship' else G.SHELL_MAP[bare]


def _shell_field(arm):
    return ('cap_avg_shell_rmse' if G.ARMS[arm]['family'] == 'ship'
            else 'cap_shell_rmse')


def shell_med(rows, arm, bare):
    """에피소드별 median-of-3 (해당 셸). 유한값만. 없으면 그 에피소드는 빠진다."""
    fld, key = _shell_field(arm), _shell_key(arm, bare)
    d = {}
    for r in rows:
        if 'error' in r or r.get('arm') != arm:
            continue
        sh = r.get(fld) or {}
        v = sh.get(key)
        if v is None:
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(v):
            d.setdefault(r['episode'], []).append(v)
    return {e: st.median(v) for e, v in d.items()}


def _spearman(x, y):
    """순위 Pearson. n<3 이거나 분산 0 이면 None."""
    n = len(x)
    if n < 3:
        return None

    def rank(v):
        order = sorted(range(n), key=lambda i: v[i])
        rk = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for t in range(i, j + 1):
                rk[order[t]] = avg
            i = j + 1
        return rk
    rx, ry = rank(x), rank(y)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return (num / (dx * dy)) if dx > 0 and dy > 0 else None


def shell_profile(rows, claim):
    """주장 하나의 셸별 상대 이득(%). 이득 = **−상대차** (낮을수록 좋으므로 부호를 뒤집는다)."""
    a, b = G.CLAIMS[claim]['a'], G.CLAIMS[claim]['b']
    prof, gains, idx = {}, [], []
    for i, bare in enumerate(G.SHELL_BARE):
        ma, mb = shell_med(rows, a, bare), shell_med(rows, b, bare)
        pr = G.paired(ma, mb)
        rel = pr['rel_median_pct']
        # ★커버리지 0 ⟹ 계수기를 0 이 아니라 None (필터 항진·0-게이트 금지)
        prof[bare] = dict(n=(pr['n'] or None), better=(pr['better'] if pr['n'] else None),
                          rel_median_pct=rel, gain_pct=(None if rel is None else -rel))
        if pr['n'] and rel is not None:
            gains.append(-rel)
            idx.append(i)

    def _med(keys):
        v = [prof[k]['gain_pct'] for k in keys if prof[k]['gain_pct'] is not None]
        return (st.median(v) if v else None)
    return dict(per_shell=prof,
                rho_shellindex_gain=(_spearman(idx, gains) if len(gains) >= 3 else None),
                inner_median_gain=_med(SHELL_INNER), mid_gain=_med([SHELL_MID]),
                outer_median_gain=_med(SHELL_OUTER),
                inner_shells=list(SHELL_INNER), mid_shell=SHELL_MID,
                outer_shells=list(SHELL_OUTER),
                direction_declared='바깥 절반 > 안쪽 절반 (수심은 예보 연산자에 들어간다)',
                promotion='⛔승격 금지 — 등록 부차. 어긋나면 어긋난 대로 보고한다.')


# ── 행 수집 ──────────────────────────────────────────────────────────────────
rows = []
for p in sorted(glob.glob(os.path.join(HERE, 'results_m3b_*.jsonl'))):
    if 'smoke' in os.path.basename(p):
        continue                              # ⛔스모크 산출물은 판정에 넣지 않는다
    with open(p) as fh:
        for line in fh:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
ok = [r for r in rows if 'error' not in r]
# ★0행 가드 — ⛔결과 0행에서 verdict 을 쓰지 않는다.
if not ok:
    print(f'⛔결과 0행({len(rows)}행 중 유효 0) — 러너를 먼저 돌려라. verdict 을 쓰지 않는다.')
    sys.exit(1)

# ── 게이트 ───────────────────────────────────────────────────────────────────
expect = R.expect_counts(R.expected_jobs())          # ★러너가 모형(손으로 적지 않는다)
gates = list(G.gates_runfree()) + list(G.gates_data()) + list(G.gates_run(rows, expect=expect))

pca = {}
if os.path.exists(R.PCA_PATH):
    with open(R.PCA_PATH) as fh:
        pca = json.load(fh)
gates += list(G.gates_pca(pca))

pcb = R.build_pcb(ok)
gates += list(G.gates_pcb(pcb))

by_id = {}
for n_, q_, d_ in gates:
    by_id[G.gate_id(n_)] = bool(q_)
blocked = [g for g in BLOCKING if not by_id.get(g, False)]
incomplete = [g for g in INCOMPLETE if not by_id.get(g, False)]

# ── 집계 · 라벨 ──────────────────────────────────────────────────────────────
agg = G.aggregate(ok)
labels, claims = {}, {}
for c in G.CLAIM_ORDER:
    pr = agg['claims'][c]
    lab, msg = G.claim_label(pr)
    labels[c] = lab
    claims[c] = dict(label=lab, label_msg=msg, paired=pr,
                     a=G.CLAIMS[c]['a'], b=G.CLAIMS[c]['b'],
                     question=G.CLAIMS[c]['q'], single_factor=G.CLAIMS[c]['clean'],
                     reading=G.READING[c][lab])

skill = dict(gate_text=G.SKILL_GATE, neg_min=G.NEG_MIN, sz_min=G.SZ_MIN,
             spec_rt12_vs_zero=agg['neg_rt12'], spec_rt50_vs_zero=agg['neg_rt50'],
             ckpt_rt12_vs_zero=agg['sz_rt12'], ckpt_rt50_vs_zero=agg['sz_rt50'])
skill['comparator_has_skill'] = (agg['neg_rt12']['better'] or 0) >= G.NEG_MIN
skill['treatment_has_skill'] = (agg['sz_rt12']['better'] or 0) >= G.SZ_MIN
skill['still_no_skill'] = not (skill['comparator_has_skill'] or skill['treatment_has_skill'])
if not skill['comparator_has_skill']:
    claims['C3-SHALLOW-COMPARE']['reading'] = (
        '⛔**기술 게이트 미달 ⟹ `C3` 를 읽지 않는다**(결과 전 선언, `SKILL_GATE`). '
        + claims['C3-SHALLOW-COMPARE']['reading'])

dec_key, dec_text = G.decompose(labels)

secondary = {'S1-SHELL': dict(
    declared=G.SECONDARY['S1-SHELL']['text'],
    C1=shell_profile(ok, 'C1-BASELINE-DEPTH'),
    C2=shell_profile(ok, 'C2-CKPT50-DEPTH'))}
exploratory = {'E1-COND': dict(declared=G.EXPLORATORY['E1-COND']['text'],
                               paired=agg['e1'],
                               a=G.EXPLORATORY['E1-COND']['a'],
                               b=G.EXPLORATORY['E1-COND']['b'])}

label = (f'DEPTH-ENFORCED :: C1={labels["C1-BASELINE-DEPTH"]} · '
         f'C2={labels["C2-CKPT50-DEPTH"]} · C3={labels["C3-SHALLOW-COMPARE"]}')
if incomplete:
    label = f'INCOMPLETE (게이트 {",".join(incomplete)} 미충족 — ⛔주장 해석 금지)'
if blocked:
    label = (f'BATCH-VOID (차단 게이트 {",".join(blocked)} 실패 — '
             f'⛔주장 해석 금지, `PREREG.md` §6)')

v = dict(batch='M3b', cell=G.CELL, n_episodes=G.N_EP, n_per_cell=G.N_PER_CELL,
         thresholds=G.threshold_table(), seeds_consumed=list(G.SEEDS_CONSUMED),
         expect=expect, n_rows=len(rows), n_ok=len(ok),
         gates=[[g[0], bool(g[1]), str(g[2])] for g in gates],
         gates_pass=sum(1 for g in gates if g[1]), gates_total=len(gates),
         blocking_failed=blocked, incomplete_failed=incomplete,
         claims=claims, skill_gate=skill,
         decomposition={'key': list(dec_key), 'text': dec_text},
         secondary=secondary, exploratory=exploratory,
         medians={a: (st.median(agg['med'][a].values()) if agg['med'][a] else None)
                  for a in G.ARM_ORDER},
         zero_median=(st.median(agg['zero'].values()) if agg['zero'] else None),
         pca={'n_rows': len(pca.get('rows', []))},
         pcb={'n_rows': len(pcb['rows']), 'n_frozen': len(pcb['frozen']),
              'reading': G.READING_PCB},
         limits=dict(L1=G.ARMS['CKPT50-RT12']['note'], void=G.VOID_IS_INDEPENDENT,
                     fix_policy=G.FIX_POLICY, runpy_policy=G.RUNPY_POLICY),
         label=label)
with open(os.path.join(HERE, 'verdict_m3b.json'), 'w') as fh:
    fh.write(json.dumps(v, indent=1, ensure_ascii=False))

# ── 출력 ─────────────────────────────────────────────────────────────────────
print('=== 게이트 ===')
for n_, q_, d_ in gates:
    print(f'  [{"PASS" if q_ else "FAIL"}] {n_}: {d_}')
print(f'\n게이트 {v["gates_pass"]}/{v["gates_total"]} · '
      f'차단 실패 {blocked or "없음"} · 불완전 {incomplete or "없음"}')

print(f'\n=== 주장 (n={G.N_PER_CELL}, median-of-3, 명목 {G.NOMINAL} · '
      f'확증 {G.CONFIRM} · 역전 ≤{G.WEAK}) ===')
for c in G.CLAIM_ORDER:
    d = claims[c]
    pr = d['paired']
    print(f"  {c:20s} {d['a']:16s} < {d['b']:16s}  {d['label']:22s} "
          f"{pr['better']}/{pr['n']} {(pr['rel_median_pct'] or 0):+7.2f}% "
          f"p={(pr['p_one_sided'] or 1):.6f}")
print(f"\n기술 게이트: SPEC25-RT12<ZERO {skill['spec_rt12_vs_zero']['better']}"
      f"/{skill['spec_rt12_vs_zero']['n']} (문턱 {G.NEG_MIN}) · "
      f"CKPT50-RT12<ZERO {skill['ckpt_rt12_vs_zero']['better']}"
      f"/{skill['ckpt_rt12_vs_zero']['n']} (문턱 {G.SZ_MIN}) · "
      f"RT50 쪽 {skill['spec_rt50_vs_zero']['better']}/{skill['spec_rt50_vs_zero']['n']} · "
      f"{skill['ckpt_rt50_vs_zero']['better']}/{skill['ckpt_rt50_vs_zero']['n']}")
print(f'\n분해 {dec_key}: {dec_text}')
for c in G.CLAIM_ORDER:
    print(f'\n[{c}] {claims[c]["reading"]}')
print('\n=== S1-SHELL (등록 부차 · ⛔승격 금지) ===')
for c in ('C1', 'C2'):
    s = secondary['S1-SHELL'][c]
    print(f"  {c}: ρ(셸,이득)={s['rho_shellindex_gain']} · "
          f"안쪽중앙 {s['inner_median_gain']} · 가운데({SHELL_MID}) {s['mid_gain']} · "
          f"바깥중앙 {s['outer_median_gain']}")
e1 = exploratory['E1-COND']['paired']
print(f"\n=== E1-COND (탐색적 · ⛔승격 금지) === {e1['better']}/{e1['n']} "
      f"{(e1['rel_median_pct'] or 0):+.2f}%")
print(f'\n라벨: {label}')
