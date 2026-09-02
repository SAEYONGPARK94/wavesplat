#!/usr/bin/env python3
"""`paper/figures/fig_raydrop.pdf` --- **센서 반환 모형(raydrop) 교락**의 그림.
⛔런 0회: `tmp/B15_RAYDROP_2026-08-24/` 의 **동결 행 재집계** + **동결 verdict** 뿐이다.

⛔이 스크립트가 존재하는 이유(2026-09-02):

논문 주석이 적은 대로, 철회된 100ep 코퍼스에서 나온 `fig5_raydrop` 이 아카이브로 가고
`B-15` 는 **표 행 하나**(`tab:earns`)로만 남았다. 표 행이 인쇄하는 것은
`n=20 · 17/20 · -42.6% · p=0.0013 · cmp 20/20` 다섯 개고, 그것으로는 **볼 수 없는 것이
셋** 있다:

  ① **짝 구조** --- `uniform` 과 `slope` 는 **같은 20 개 파랑장**이다
     (`gt_wave_field` max|Δ|=0.0, 게이트 `G-1`). 표 행은 `slope` 열만 인쇄하므로
     독자는 이것이 **짝 개입**이라는 것을 알 수 없다.
  ② **비교자가 무너졌는가** --- 크기가 기하 가중에서 무가중으로 가며 **커진다**.
     저장소가 반복해서 대가를 치른 `F1` 부류(*"비교자가 무너져서 생긴 margin 을
     기술로 읽는 것"*)의 정확한 모양이다. ★**답은 그림에만 있다**:
     비교자는 무너지기는커녕 **20/20 에피소드에서 절대 오차가 좋아진다.**
  ③ **짝별 분산** --- 승수는 16 → 17 로 **하나** 움직이는데, 짝별 상대 margin 은
     -80% 에서 +58% 까지 흔들린다. 표의 중앙값 하나가 그것을 숨긴다.

★그러므로 이 그림은 `fig_results` 와 **겹치지 않는다**. `fig_results` 패널 (a) 는
`B-15` 의 **`slope` 승수 17/20 과 그 비교자 20/20 만** 그린다
(`gen_fig_results.py:ROWS` 의 `by_raydrop.slope.*`). 여기서 그리는 것은
**짝·절대 수준·제로필드·짝별 분산**이고 그 넷은 어느 그림에도 없다.

★규약은 표와 같다: **손으로 적은 수 0개.** 값은 동결 verdict 에서 읽거나 동결 행에서
재집계하고, **재집계가 verdict 과 어긋나면 죽는다**(`gen_abs_table.py` 의 교차 검증 규약).

⛔이 그림이 **주장하지 않는 것**(`RESULT_B15.md` 가 결과 전 사전등록대로 적은 그대로):
  1. ⛔*"`uniform` 이 옳다"* 가 아니다. **모형이 틀린 것과 결론이 바뀌는 것은 다르다.**
  2. ⛔`B-12`(n=48, HOS 참값)의 판정을 **대체하지 않는다** --- 이 배치는 **선형 참값**이다.
  3. ⛔크기 -42.61% 를 `B-12` 의 -40.19% 나 `B-1` 의 -38.58% 와 **직접 비교하지 마라**
     (표본·참값·센서 모형이 다르다). ⟹ 캡션이 이것을 병기해야 한다.
  4. ⚠개입은 발화했으나(`G-2`) **작다** --- 반환 입사각 중앙 75.37° → 77.63°.
     ⟹ *"반환 모형을 크게 바꿔도 무관"* 으로 확대 해석하지 마라. 그래서 그 두 각을
     **그림 안에 인쇄한다**(캡션에만 두면 그림만 보는 독자가 개입의 크기를 모른다).
     ⚠★`RESULT_B15.md` 는 이 작음을 *"평온 해면 폴백 때문에 개입이 **부분적**일 수
     있다"* 로 설명했다. **그 설명은 틀렸다**(아래 발견): 폴백은 평온할 때만이 아니라
     **항상** 걸리므로 개입은 부분적이지 않고 **완전하다** --- 다만 그 개입이
     *"가중을 끄는 것"* 이지 *"정반사로 바꾸는 것"* 이 아닐 뿐이다.
     ⟹ 크기 경고는 **유지**되고 사유만 바뀐다.

⚠사전등록의 SECONDARY 는 **승수 차**(16-17 = -1 ⟹ `RAYDROP-DOES-NOT-CHANGE-HEADLINE`)
이지 **크기 차가 아니다.** 패널 (b) 가 크기를 그리므로 캡션이 그 구분을 적어야 한다
--- 크기의 이동은 `RESULT_B15.md` 가 적은 **서술**이고 사전등록된 대조가 아니다.

⛔⛔★★★**이 그림은 `B-15` 의 이름을 따라 적으면 안 된다 (2026-09-02 발견).**

배치의 라벨은 `HEADLINE-SURVIVES-**CORRECT**-SENSOR` 이고 `RESULT_B15.md` 는 `slope` 를
*"물리적으로 옳은 센서 모형"* 이라 부른다. **그 물리 전제는 그 뒤에 저장소가 스스로
반증했다.** 축자, `wavesim/raydrop_threshold.py:9-11`:

> *"raydrop_slope's specular gate **never fires at this geometry** (required facet
>  slope r/h = 3..40 vs wave-slope sd ~0.08) and **falls back to constant-P**."*

`raydrop_slope.apply_dropout:115-120` 이 그 폴백이다(정반사 후보가 0개면 균등 Bernoulli).
⟹ **`slope` 는 정반사 모형이 아니라 「무가중」 모형**이고, `uniform` 이 「기하 가중」
(`P ∝ R_fresnel(θ)·cosθ/d²`)이다. 논문 §Method 는 **이미 그렇게 고쳐 적었다** ---
*"The implementation consequently always takes its unweighted fallback path … The
contrast is therefore a test of sensitivity to the spatial weighting of returns,
not of diffuse versus specular optics."*

⟹ ★**측정은 무손상이다**(같은 20 파랑장의 순수 짝 개입, 승수·p·크기 전부 그대로).
   바뀌는 것은 **그 축의 이름**뿐이다.
⟹ ⛔**그러므로 그림에 *"physically correct"* 를 인쇄하지 않는다.** 그렇게 적으면
   논문에서 가장 많이 읽히는 표면에 **철회된 주장을 되살리는 것**이고, 그것이 이
   저장소가 반복해서 당한 「낡은 문장이 정정을 되감는다」 부류다.
⟹ ★동결 verdict 의 **라벨 문자열은 그대로 둔다**(동결물이다). 그림이 그 이름을
   **따라 적지 않을** 뿐이다. 아래 `NAME` 이 그 분리를 코드로 만든다.
"""
import glob
import hashlib
import json
import os
import re
import statistics as st
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
BATCH = os.path.join(ROOT, 'tmp', 'B15_RAYDROP_2026-08-24')

# ── ★단일출처: 문턱·시드·부호검정은 배치의 게이트 모듈에서 **import** 한다.
#   ⛔손으로 옮겨 적으면 배치가 바뀔 때 그림이 조용히 낡는다(저장소의 「손복사 기준
#   상수」 결함 부류). `analyze_b15.py` 와 같은 경로를 쓴다.
if BATCH not in sys.path:
    sys.path.insert(0, BATCH)
import gates_b15 as G                                            # noqa: E402

# ── ★동결 확인. ⛔"동결한 적이 없다" 가 "동결 뒤에 움직였다" 보다 나쁘다는 것이
#   이 저장소의 교훈이다 ⟹ 그림을 그리기 전에 md5 를 대조하고, 어긋나면 죽는다.
for _line in open(os.path.join(BATCH, 'FROZEN_md5.txt')):
    _md5, _name = _line.split()
    _got = hashlib.md5(open(os.path.join(BATCH, _name), 'rb').read()).hexdigest()
    if _got != _md5:
        raise SystemExit(f'⛔동결 파일이 움직였다: {_name} ({_got} != {_md5})')

V = json.load(open(os.path.join(BATCH, 'verdict_b15.json')))

# ── ★가정을 전부 assert 한다. 하나라도 바뀌면 이 그림은 다른 것을 그리는 셈이다.
assert V['batch'] == 'B-15', V['batch']
assert V['gates_pass'] == V['gates_total'] == 6, (V['gates_pass'], V['gates_total'])
assert all(g[1] for g in V['gates']), '게이트가 하나라도 FAIL 이면 그림을 그리지 않는다'
assert V['label'] == 'HEADLINE-SURVIVES-CORRECT-SENSOR', V['label']
assert V['secondary_label'] == 'RAYDROP-DOES-NOT-CHANGE-HEADLINE', V['secondary_label']
assert V['neg_control_ok'] is True, '음성대조가 깨지면 PRIMARY 를 읽지 않는다 ⟹ 그림도 없다'
assert tuple(V['by_raydrop']) == G.RAYDROPS == ('uniform', 'slope'), tuple(V['by_raydrop'])
assert V['n_rows'] == V['n_ok'], '오류 행이 있다 (G-5)'

UNI, SLO = G.RAYDROPS                     # 'uniform' · 'slope' --- 저장소의 **역사적 이름**
OURS, CMP = 'SHIP-NOW', 'SPEC25'          # ⚠팔 이름은 행에서 다시 확인한다(아래 assert)

# ★★그림에 인쇄할 이름은 **논문 §Method 의 서술**이지 배치 라벨이 아니다(위 독스트링).
#   ⛔`RESULT_B15.md` 의 *"물리적으로 옳은"* 을 여기 옮겨 적지 마라.
NAME = {UNI: 'geometry-weighted', SLO: 'unweighted'}

# ★그 이름의 근거를 **소스에서** 고정한다. 정반사 원뿔이 넓어지면 게이트가 발화할 수
#   있고 그러면 `slope` 는 더 이상 「무가중」이 아니다 ⟹ 그때는 이 그림이 죽어야 한다.
#   ⚠순수 numpy 모듈이고 **호출하지 않는다**(서명만 읽는다) --- 시뮬레이터를 돌리지 않는다.
import inspect                                                   # noqa: E402
sys.path.insert(0, ROOT)
import wavesim.raydrop_slope as RS                                # noqa: E402
_ALPHA = inspect.signature(RS.apply_dropout).parameters['alpha_threshold_deg'].default
assert _ALPHA == 5.0, (
    f'⛔정반사 원뿔 반각이 바뀌었다({_ALPHA}°) — `slope` 가 아직 constant-P 폴백인지 '
    '다시 재라. 그림의 축 이름이 「무가중」이라 적고 있다')


# =============================================================================
# §1  동결 행 재집계 --- `analyze_b15.py` 와 **같은 규약**(median-over-seeds)
# =============================================================================
def load_rows():
    rows = []
    for p in sorted(glob.glob(os.path.join(BATCH, 'results_b15_*.jsonl'))):
        rows += [json.loads(l) for l in open(p)]
    return rows


def med_by_episode(rows, arm):
    """추정기/필터 시드에 대한 median → 에피소드 단위 값 (`B-11` 규약, `analyze_b15.py` 축자).

    ⚠`zero_nRMSE` 는 **비교자 행에만** 기록돼 있다(우리 팔 행은 `null`) ---
    제로필드 채점을 비교자 러너가 한 번만 돌렸기 때문이다. ⟹ 두 팔의 제로 기준은
    **같은 값**이고, 그것이 짝 비교를 성립시킨다.
    """
    by = {}
    for r in rows:
        if r['arm'] == arm:
            by.setdefault(r['episode'], []).append(r)
    out = {}
    for ep, rs in by.items():
        zs = [x['zero_nRMSE'] for x in rs if x.get('zero_nRMSE') is not None]
        out[ep] = dict(nrmse=st.median([x['avg_nRMSE'] for x in rs]),
                       zero=st.median(zs) if zs else None,
                       n_seed=len(rs), raydrop=rs[0]['raydrop'], pair=rs[0]['pair'],
                       inc=rs[0]['mean_inc_deg'], gt_hash=rs[0]['gt_hash'])
    return out


ROWS = load_rows()
assert len(ROWS) == V['n_rows'], f"행 수가 verdict 과 다르다: {len(ROWS)} != {V['n_rows']}"
assert {r['arm'] for r in ROWS} == {OURS, CMP}, {r['arm'] for r in ROWS}

# ★비교자가 **무엇인지**도 assert 한다 --- 범례에 'LETKF' 라 적을 근거는 행에 있다.
#   ⛔팔 이름만 보고 성격을 적으면 팔이 바뀔 때 범례가 조용히 거짓이 된다.
for r in ROWS:
    if r['arm'] == CMP:
        assert (r['loc_radius'] == G.SPEC['loc'] and r['n_ensemble'] == G.SPEC['n_ensemble']
                and r['background'] == G.SPEC['background']), f'⛔비교자 구성이 다르다: {r}'

OU = med_by_episode(ROWS, OURS)
CM = med_by_episode(ROWS, CMP)
assert len(OU) == len(CM) == len(G.RAYDROPS) * G.N_PAIR == 40, (len(OU), len(CM))

# ⚠★운영 사고의 흔적을 **조용히 넘기지 않는다**: 첫 6샤드를 멀티스레드로 띄웠다가
#   스래싱으로 죽였고(`RESULT_B15.md` 「방법론 기록」), 그때 완주한 셀 **하나**가
#   `results_b15_4.jsonl` 에 남아 재실행분과 겹친다 ⟹ 그 셀만 median-of-**4** 다.
#   동결 verdict 도 같은 4행을 썼으므로 값은 일치하지만, **아는 채로** 통과시킨다.
_odd = {ep: d['n_seed'] for d in (OU, CM) for ep, d in d.items() if d['n_seed'] != len(G.EST_SEEDS)}
assert _odd == {'episode_BF4_15comp_seed2_uniform': 4}, (
    f'⛔시드 수가 예상과 다르다: {_odd} — median-of-k 의 k 가 바뀌었나?')

# ★`G-1` 을 그림 쪽에서도 다시 확인한다: 짝의 참 파랑장이 같아야 x-y 산점도가 뜻을 가진다.
PAIRS = sorted({d['pair'] for d in OU.values()})
assert len(PAIRS) == G.N_PAIR == 20, len(PAIRS)
for p in PAIRS:
    a, b = OU[f'episode_{p}_{UNI}'], OU[f'episode_{p}_{SLO}']
    assert abs(a['gt_hash'] - b['gt_hash']) <= 1e-6, f'⛔짝의 참 파랑장이 다르다: {p}'

# ★제로필드 기준이 **짝 안에서 비트 동일**함을 확인한다.
#   ⟹ 패널 (a) 에서 제로 점들이 대각선 위에 정확히 놓이는 것은 **성질이지 우연이 아니다**.
#   ⚠그리고 `G-6` 대로 **상수가 아니다**(에피소드마다 다르다) --- 실측 쪽에서 상수
#   `ZERO_REF=1.0` 을 썼다가 전 계수를 철회한 일이 있어 이 구분이 중요하다.
ZERO = {}
for p in PAIRS:
    zu, zs = CM[f'episode_{p}_{UNI}']['zero'], CM[f'episode_{p}_{SLO}']['zero']
    assert zu is not None and zs is not None, f'⛔제로필드 기준이 없다: {p}'
    assert zu == zs, f'⛔짝의 제로 기준이 다르다: {p} ({zu} != {zs})'
    ZERO[p] = zu
assert len({round(z, 6) for z in ZERO.values()}) > 1, '⛔제로 기준이 상수다 (G-6 위반)'


def agg(rd):
    """`analyze_b15.py` 를 그대로 다시 밟는다 --- 값이 갈리면 아래에서 죽는다."""
    eps = sorted(e for e, d in OU.items() if d['raydrop'] == rd)
    w = sum(1 for e in eps if OU[e]['nrmse'] < CM[e]['nrmse'])
    neg = sum(1 for e in eps if CM[e]['nrmse'] < CM[e]['zero'])
    ours_neg = sum(1 for e in eps if OU[e]['nrmse'] < CM[e]['zero'])
    rel = st.median([(OU[e]['nrmse'] - CM[e]['nrmse']) / CM[e]['nrmse'] * 100 for e in eps])
    return dict(n=len(eps), ship_beats_spec=w, spec_beats_zero=neg, ours_beats_zero=ours_neg,
                rel_median_pct=rel,
                p_one_sided=G.sign_test_one_sided(w, len(eps)),
                median_ship=st.median([OU[e]['nrmse'] for e in eps]),
                median_spec=st.median([CM[e]['nrmse'] for e in eps]),
                mean_inc_deg=st.median([OU[e]['inc'] for e in eps]))


A = {rd: agg(rd) for rd in G.RAYDROPS}

# ── ★★교차 검증: 재집계가 **동결 verdict 과 어긋나면 죽는다.**
#   ⛔침묵시키지 마라 --- 어긋난다는 것은 둘 중 하나의 집계가 틀렸다는 뜻이고,
#   그림이 표와 다른 수를 인쇄하게 된다(`gen_abs_table.py` 와 같은 규약).
for rd in G.RAYDROPS:
    v, a = V['by_raydrop'][rd], A[rd]
    for k in ('n', 'ship_beats_spec', 'spec_beats_zero'):
        if int(v[k]) != int(a[k]):
            raise SystemExit(f'⛔교차 검증 실패 {rd}.{k}: verdict {v[k]} vs 재집계 {a[k]}')
    for k in ('rel_median_pct', 'median_ship', 'median_spec', 'mean_inc_deg', 'p_one_sided'):
        if abs(float(v[k]) - float(a[k])) > 1e-12:
            raise SystemExit(f'⛔교차 검증 실패 {rd}.{k}: verdict {v[k]} vs 재집계 {a[k]}')
    # ★음성대조는 사전등록의 **무효화 조건**이다 ⟹ 문턱까지 다시 밟는다.
    if a['spec_beats_zero'] < G.NEG_MIN:
        raise SystemExit(f'⛔{rd} 음성대조 미달 {a["spec_beats_zero"]}/{a["n"]} — 그림 금지')
assert A[SLO]['ship_beats_spec'] >= G.CONFIRM, '라벨과 승수가 어긋난다'

# 짝별 값 (패널이 쓰는 전부)
PX = [OU[f'episode_{p}_{UNI}']['nrmse'] for p in PAIRS]      # 우리 팔, uniform
PY = [OU[f'episode_{p}_{SLO}']['nrmse'] for p in PAIRS]      # 우리 팔, slope
CX = [CM[f'episode_{p}_{UNI}']['nrmse'] for p in PAIRS]      # 비교자, uniform
CY = [CM[f'episode_{p}_{SLO}']['nrmse'] for p in PAIRS]      # 비교자, slope
ZX = [ZERO[p] for p in PAIRS]                                # 제로필드 (짝 안에서 동일)
REL = {rd: [(OU[f'episode_{p}_{rd}']['nrmse'] - CM[f'episode_{p}_{rd}']['nrmse'])
            / CM[f'episode_{p}_{rd}']['nrmse'] * 100 for p in PAIRS] for rd in G.RAYDROPS}

# ★F1 부류의 핵심 수: **비교자가 물리적으로 옳은 센서에서 좋아진 에피소드 수.**
#   margin 이 커진 것이 *비교자가 무너져서* 인지 아닌지가 여기서 갈린다.
OURS_BETTER_AT_SLOPE = sum(1 for a, b in zip(PX, PY) if b < a)
CMP_BETTER_AT_SLOPE = sum(1 for a, b in zip(CX, CY) if b < a)

INK, MUTE, ACC, WARN = '#1a1a1a', '#8a8a8a', '#0b5394', '#a33'

# ⚠matplotlib 은 `usetex` 없이 돈다 ⟹ **LaTeX 이스케이프를 쓰면 그대로 인쇄된다.**
#   초판이 `[\%]` 와 `vs.\ ` 를 그대로 찍었다. 코드 이름은 mathtext 등폭으로 낸다.
TT = lambda s: f'$\\mathtt{{{s}}}$'                              # noqa: E731

fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.4, 3.55),
                               gridspec_kw={'width_ratios': [1.0, 1.0]})

# =============================================================================
# 패널 (a) --- 짝 산점도. **개입 전후의 절대 오차**, 그리고 제로필드.
#   ★대각선 아래 = 무가중 반환에서 더 정확. 제로필드는 참 파랑장이 같으므로
#     **정의상 대각선 위**에 놓인다 --- 그 자체가 `G-1` 의 눈으로 보는 증거다.
#   ⛔주석에도 *"물리적으로 옳은 센서"* 를 쓰지 마라 --- 낡은 지시 주석이 정정을
#     되감는 것이 이 저장소의 재발 공장이다(「N곳 전부」의 N 에는 주석이 포함된다).
#   ⚠입사각이 왜 75.37°→77.63° 로 **올라가는지** 는 여기서 주장하지 않는다:
#     기하 가중은 `R·cosθ`(높은 입사각을 선호)와 `1/d²`(가까운=낮은 입사각을 선호)의
#     **경쟁**이라 부호가 자명하지 않다. 그림은 `G-2` 가 잰 값을 인쇄만 한다.
# =============================================================================
# ⚠여백 1.06 배로는 통계 상자의 마지막 줄이 **제로필드 사각형과 겹쳤다**(대각선의
#   맨 끝이 오른쪽 위 모서리이므로 그 구석은 비지 않는다). ⟹ 위쪽 여백을 벌린다.
#   ★두 축의 한계를 같게 유지해야 대각선이 45° 로 읽힌다 --- 한쪽만 늘리지 마라.
lo, hi = 0.0, max(PX + PY + CX + CY + ZX) * 1.26
axA.plot([lo, hi], [lo, hi], color=MUTE, lw=0.9, ls=':', zorder=1)
axA.plot(ZX, ZX, 's', ms=4.0, mfc='none', mec=MUTE, mew=0.9, zorder=2,
         label='zero field (unchanged)')
axA.plot(CX, CY, 'o', ms=4.6, mfc='none', mec=WARN, mew=1.1, zorder=3,
         label='comparator (LETKF)')
axA.plot(PX, PY, 'o', ms=4.6, color=ACC, zorder=4, label='our arm')
axA.set_xlim(lo, hi)
axA.set_ylim(lo, hi)
axA.set_aspect('equal', adjustable='box')
axA.set_xlabel(f'nRMSE, {NAME[UNI]} returns ({TT(UNI)})', fontsize=7.0)
axA.set_ylabel(f'nRMSE, {NAME[SLO]}\nreturns ({TT(SLO)})', fontsize=7.0)
axA.tick_params(labelsize=7)
for sp in ('top', 'right'):
    axA.spines[sp].set_visible(False)
# ★★음성대조를 **그림 안에** 인쇄한다. 이것이 없으면 패널 (b) 의 커진 margin 이
#   *"비교자가 무너져서"* 로 읽힐 수 있다 --- 저장소가 반복해서 당한 `F1` 부류다.
# ⛔초판은 여기에 더해 대각선 옆에 *"better under the correct sensor"* 를 45° 로
#   눕혀 적었는데 **비교자 점 무리를 관통**했다. ⟹ 지우고, 그 뜻을 이 상자의
#   **첫 줄**이 지게 한다 --- 대각선 아래 점의 수가 곧 그 문장이므로 중복이었다.
axA.text(0.008, hi * 0.995,
         f'below the diagonal, i.e. more accurate with {NAME[SLO]} returns\n'
         f'    our arm {OURS_BETTER_AT_SLOPE}/{G.N_PAIR}'
         f'    comparator {CMP_BETTER_AT_SLOPE}/{G.N_PAIR}\n'
         f'beats its own zero field   ({TT(UNI)} $\\rightarrow$ {TT(SLO)})\n'
         f"    our arm {A[UNI]['ours_beats_zero']}/{G.N_PAIR} $\\rightarrow$ "
         f"{A[SLO]['ours_beats_zero']}/{G.N_PAIR}"
         f"    comparator {A[UNI]['spec_beats_zero']}/{G.N_PAIR} $\\rightarrow$ "
         f"{A[SLO]['spec_beats_zero']}/{G.N_PAIR}",
         fontsize=6.2, color=INK, ha='left', va='top', linespacing=1.55)
axA.legend(fontsize=6.5, loc='lower right', frameon=False, handletextpad=0.3,
           borderaxespad=0.3, labelspacing=0.35)
# ⛔손으로 적은 수 0개 --- 짝 수도 `gates_b15.N_PAIR` 에서 온다.
axA.set_title(f'(a) the same {G.N_PAIR} wave fields, two return models',
              fontsize=8, loc='left')

# =============================================================================
# 패널 (b) --- 짝별 상대 margin. **중앙값 하나가 숨기는 분산**을 보인다.
#   ⚠사전등록된 SECONDARY 는 **승수 차**(-1)이지 크기 차가 아니다 ⟹ 승수를 함께 인쇄한다.
# =============================================================================
for u, s in zip(REL[UNI], REL[SLO]):
    # ⚠`slope` 에서 지는 짝만 색을 준다 --- 3 개가 어디로 갔는지 세어 볼 수 있게.
    c, z = (WARN, 3) if s > 0 else (MUTE, 2)
    axB.plot([0, 1], [u, s], '-', color=c, lw=0.7, alpha=0.75, zorder=z)
    axB.plot([0, 1], [u, s], 'o', ms=2.6, color=c, alpha=0.9, zorder=z)
axB.axhline(0, color=INK, lw=0.9, ls=':', zorder=1)
for x, rd in ((0, UNI), (1, SLO)):
    m = A[rd]['rel_median_pct']
    axB.plot([x - 0.16, x + 0.16], [m, m], '-', color=ACC, lw=2.2, zorder=5)
    # ⛔막대 **아래**에 적었더니 라벨이 짝 점들 위에 앉았다(중앙값 근처가 가장 빽빽하다).
    #   흰 상자로 덮으면 자료를 가린다 ⟹ 자료가 없는 **바깥 여백**으로 내보내고,
    #   같은 높이에 두어 막대와 눈으로 이어지게 한다.
    axB.text(-0.42 if x == 0 else 1.42, m, f'median\n{m:+.1f}%',
             fontsize=6.6, color=ACC, weight='bold', linespacing=1.3,
             ha='left' if x == 0 else 'right', va='center')
    # ⛔초판은 이 상자를 y=68 에 두었는데 `slope` 열의 최상단 짝이 +57.8% 라 **겹쳤다**.
    #   ⟹ 축 위쪽에 자리를 만들고(ylim 96) 데이터 위로 올린다.
    axB.text(x, 94, f"{A[rd]['ship_beats_spec']}/{A[rd]['n']} won\n"
                    f"$p={A[rd]['p_one_sided']:.4f}$",
             fontsize=6.8, color=INK, ha='center', va='top', linespacing=1.4)
axB.text(1.44, 4.0, 'comparator is\nbetter above\nthis line', fontsize=6.2, color=MUTE,
         style='italic', ha='right', va='bottom')
axB.set_xlim(-0.44, 1.44)
axB.set_ylim(-92, 96)
axB.set_yticks([-80, -60, -40, -20, 0, 20, 40, 60])
axB.set_xticks([0, 1])
axB.set_xticklabels([f'{TT(UNI)}\n({NAME[UNI]})', f'{TT(SLO)}\n({NAME[SLO]})'],
                    fontsize=7.0)
axB.set_ylabel('per-episode relative difference,\nour arm vs. comparator  [%]',
               fontsize=7.0)
axB.tick_params(axis='y', labelsize=7)
axB.tick_params(axis='x', length=0)
for sp in ('top', 'right'):
    axB.spines[sp].set_visible(False)
# ★개입의 크기를 그림 안에 남긴다 --- 캡션에만 두면 그림만 보는 독자가 모른다.
#   `slope` 에는 평온 해면 폴백이 있어 개입은 발화하되(`G-2`) **작다**.
# ⚠mathtext `^\circ` 는 수와 기호 사이에 빈칸을 넣는다 ⟹ 유니코드로 직접 쓴다.
axB.set_xlabel('median incidence angle of returned points: '
               f"{A[UNI]['mean_inc_deg']:.2f}° → "
               f"{A[SLO]['mean_inc_deg']:.2f}°",
               fontsize=6.6, color=MUTE, labelpad=6)
# ★★제목의 '하나' 도 손으로 적지 않는다 --- 사전등록의 SECONDARY 는 **승수 차**이고
#   그 값이 `RAYDROP-DOES-NOT-CHANGE-HEADLINE` 라벨을 만든다.
DWIN = A[UNI]['ship_beats_spec'] - A[SLO]['ship_beats_spec']
assert abs(DWIN) <= G.SEC_SAME, f'⛔승수 차 {DWIN} 는 더 이상 "바뀌지 않는다" 가 아니다'
assert V['secondary_text'].strip().endswith(str(DWIN)), (V['secondary_text'], DWIN)
axB.set_title(f'(b) the win count moves by {abs(DWIN)}; the episodes do not',
              fontsize=8, loc='left')

fig.tight_layout(pad=0.6, w_pad=1.6)
out = os.path.join(ROOT, 'paper', 'figures', 'fig_raydrop.pdf')
fig.savefig(out, bbox_inches='tight', pad_inches=0.02)

# =============================================================================
# §2  사이드카 --- 이 그림이 **인쇄하는 모든 수**의 목록.
#   ⛔인용금지 래칫(`tmp/STATUS_2026-08-28/audit_paper_bans.py:83`)이 `paper/**/*.pdf` 를
#   `pdftotext` 로 훑는다 ⟹ 그림 안의 수도 검사 대상이다. 값 단위 면제에 쓰라고
#   `<파일>.values.json` 규약(`gen_abs_table.py:182`)으로 남긴다.
#   ★손으로 목록을 적지 않는다.
#   ⛔초판은 **캔버스 객체**에서 긁었는데 matplotlib 이 축 밖 눈금 라벨을 그대로
#     들고 있어 **그려지지도 않은 `0.40` 이 목록에 들어갔다**. 면제 목록이 실제보다
#     넓어지는 것은 *"인쇄되지 않은 금지 수를 면제"* 하는 쪽의 오류다.
#   ⟹ **래칫이 읽는 바로 그 경로**(`pdftotext`)로 뽑는다.
# =============================================================================
import subprocess                                                # noqa: E402
_txt = subprocess.run(['pdftotext', '-q', out, '-'], capture_output=True,
                      text=True, timeout=120).stdout
# ⛔추출이 조용히 비면 **빈 면제 목록**이 나가고 래칫은 통과한다 --- 저장소가
#   `paper/` 쪽에서 이미 당한 부류다(PDF 추출 침묵 가드). ⟹ 죽인다.
if not _txt.strip():
    raise SystemExit('⛔pdftotext 가 아무것도 뽑지 못했다 — 사이드카를 쓰지 않는다')
vals = sorted({m.group(0) for m in re.finditer(r'\d+(?:\.\d+)?', _txt)})
# ★자기검사: 이 그림이 **판정의 수를 실제로 인쇄했는지** 확인한다.
#   ⛔글자가 아니라 곡선으로 조판되면(폰트 미임베드 등) 위 목록이 조용히 반쪽이 된다.
for _need in (f"{A[SLO]['ship_beats_spec']}/{A[SLO]['n']}",
              f"{A[SLO]['p_one_sided']:.4f}",
              f"{A[SLO]['rel_median_pct']:.1f}"[1:],
              f"{A[SLO]['mean_inc_deg']:.2f}"):
    if _need not in _txt:
        raise SystemExit(f'⛔그림이 판정 수 {_need} 를 인쇄하지 않았다(또는 추출 실패)')
side = os.path.join(ROOT, 'paper', 'figures', 'fig_raydrop.values.json')
json.dump({'source': 'scripts/gen_fig_raydrop.py',
           'note': ('동결 verdict + 동결 행 재집계 산출값(런 0회). 인용금지 래칫이 '
                    '값 단위 면제에 쓴다. ⛔면제는 "이 그림이 재생성으로 같은 수를 '
                    '인쇄" 할 때만 유효하다.'),
           'batch': 'tmp/B15_RAYDROP_2026-08-24',
           'verdict': 'verdict_b15.json',
           'label': V['label'], 'secondary_label': V['secondary_label'],
           'values': vals}, open(side, 'w'), indent=1, ensure_ascii=False)

print(f'{"":10s} {"ours<cmp":>9s} {"p":>9s} {"rel med":>9s} '
      f'{"cmp<zero":>9s} {"ours<zero":>10s} {"inc":>8s}')
for rd in G.RAYDROPS:
    a = A[rd]
    print(f"{rd:10s} {a['ship_beats_spec']:6d}/{a['n']:<2d} {a['p_one_sided']:9.4g} "
          f"{a['rel_median_pct']:+8.2f}% {a['spec_beats_zero']:6d}/{a['n']:<2d} "
          f"{a['ours_beats_zero']:7d}/{a['n']:<2d} {a['mean_inc_deg']:7.2f}°")
print(f'\n비교자가 {NAME[SLO]} 반환에서 좋아진 짝: {CMP_BETTER_AT_SLOPE}/{G.N_PAIR} '
      f'(우리 팔 {OURS_BETTER_AT_SLOPE}/{G.N_PAIR})'
      '  ⟹ 커진 margin 은 비교자 붕괴가 아니다')
print(f"PRIMARY   {V['label']} — {V['label_text']}")
print(f"SECONDARY {V['secondary_label']} — {V['secondary_text']}")
print(f'\n✅wrote {out}\n✅wrote {side}  ({len(vals)} 개 인쇄값)')
