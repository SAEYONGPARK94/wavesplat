"""S7 채점기 — `PREREG.md` + `AMENDMENT_1.md` 를 그대로 집행한다.

★**결과를 읽기 전에 md5 동결한다.** 동결 후에는 수정하지 않는다.

집행하는 규약 (AMENDMENT_1 §A9):
  에피소드당 3시드 **중앙** → 짝 Δ(같은 에피소드) → **에피소드별 Δ 부호**가 검정 통계량(n=32)
  판정 = (Δ 중앙 < 0) AND (Holm-2 조정 p ≤ 0.05), 유효 n = 32 − 동점수   [A10]
  판정 규약은 **절대**, 상대는 의무 병기, 불일치는 결과로 보고            [A9-6]

⛔이 파일은 판정 규칙을 담고 결과를 담지 않는다. 실행은 결과 생성 후.
"""
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np
from scipy.stats import binom, binomtest

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get('S7_OUT', os.path.join(HERE, 'results.jsonl'))
MDEJ = os.path.join(HERE, 'MDE_prespec_s7.json')

# ── 사전 확정 상수 (동결) ─────────────────────────────────────────────────────
ALPHA = 0.05
N_EP = 32                       # 레짐당
HARM_SIGN_K = 21                # A4 — 희소 안전 게이트, **무보정**
D2_REF = {'P1': -0.02583, 'P2': -0.04435}     # A20 크기 재현 참조 (S6, 선택된 값)
D2_FRAC = 0.50
MDE_SIGN80_C = 0.7234           # A6 — 문턱 23/32 에서 검정력 80% 에 필요한 부호율의 z 환산
NEAR_SHELLS = ('0_5', '5_10', '10_20')        # A2 — 지지집합 근처
SHELL_KEYS = ('0_5', '5_10', '10_20', '20_30', '30_50', '50_75', '75_100',
              '100_150', '150_inf')
ALPHA_LEAD = '0.2'              # S2 동결 규약의 대표 리드
POWER_PRIOR = {0.8125: 0.9374, 0.75: 0.7367, 0.70: 0.4951, 0.65: 0.2685}   # 사전값 (A6)

PRIMARY = [('P1', 'AUTO-NT-L1-1e4', 'AUTO-NT'),      # λ 단독
           ('P2', 'AUTO-NT-L1-1e4', 'AUTO')]         # 묶음 vs 출하 기본값 (노브 2개)
SINGLE = ('S1', 'AUTO-L1-1e4', 'AUTO')               # A17 — 출하 yaml 한 줄
SECOND = [('D3e4', 'AUTO-NT-L1-3e4', 'AUTO-NT'),     # 용량반응
          ('NT', 'AUTO-NT', 'AUTO'),                 # 추적기 off 단독 (6번째 재현)
          ('DOSE', 'AUTO-NT-L1-1e4', 'AUTO-NT-L1-3e4')]

REQUIRED = [
    'I1  워커 누출 프로브 (AUTO-NT vs AUTO-NT-DUP)',
    'I2  셀당 cfg 유일 + 실측 lambda_l1 / tracking_lr',
    'I3  1152/1152 · 에러 0',
    'I4  regime 카운트 crowded 576 / sparse 576  [A1]',
    'I5  실행 전 게이트 G-1~G-5 재수록  [A5]',
    'I6  대조표 (절대·상대·부호·동점·원p·Holm·mde_agg·mde_sign80)  [A6/A9]',
    'I7  시드별 부호수 + mean-of-3',
    'I8  PRIMARY 판정 문자열 + 접미 라벨',
    'I9  희소 게이트 판정 (HARM-SIGN / HARM-TAIL / BENEFIT / NONINFERIOR)  [A4]',
    'I10 묶음 분해 — 에피소드별 (중앙끼리 더해지지 않는다)  [A18]',
    'I11 사전 검정력표 재수록 (사후 재계산 금지)  [A3]',
    'I12 S6 대비 수준 병기 (짝 아님 · 풀링 금지)',
    'I13 블록별 부호수·Δ중앙 + CONDITION-SPECIFIC / HETEROGENEOUS  [A8]',
    'I14 지지집합 안/밖 + 9셸 + SUPPORT-TRADEOFF  [A2]',
    'I15 라우트 분포 + ensemble_k + 층화  [A11]',
    'I16 기제 계측 (window_resid / decorr / shrink) + ΔN  [A16]',
    'I17 alpha_wave (SECONDARY, 판정 미사용)  [A21]',
    'I18 수준 대 짝Δ CONVENTION-CONFLICT  [A19]',
    'I19 D2-S7 크기 재현 (SIGN-ONLY / SIGN+MAGNITUDE)  [A20]',
    'I20 S2_ALPHA on/off 비트동일 스팟체크  [A21]',
    'I21 DEPLOY-SINGLE 족 판정 (PRIMARY 에 기여하지 않음)  [A17]',
]
EMITTED = set()

OUT = []


def p(s=''):
    print(s, flush=True)
    OUT.append(s)


def emit(tag):
    EMITTED.add(tag.split()[0])


# ── 자료 ─────────────────────────────────────────────────────────────────────
def regime(ep):
    if ep.startswith('data_s7_crowded'):
        return 'crowded'
    if ep.startswith('data_s7_sparse'):
        return 'sparse'
    raise ValueError(f'알 수 없는 경로: {ep}')


def block(ep):
    m = re.search(r'episode_(BF\d)_(\d+)comp', ep)
    return f'{m.group(1)}_{m.group(2)}c'


def getm(rec, metric):
    """중첩 지표(shell/alpha) 포함 추출."""
    if metric.startswith('shell:'):
        return (rec.get('shell_rmse') or {}).get(metric.split(':', 1)[1], np.nan)
    if metric == 'alpha':
        return (rec.get('alpha_episode') or {}).get(ALPHA_LEAD, np.nan)
    v = rec.get(metric, np.nan)
    return np.nan if v is None else v


def load():
    recs = [json.loads(l) for l in open(RES) if l.strip()]
    good = [r for r in recs if 'error' not in r]
    return recs, good


def per_ep(good, metric):
    """(cell, episode) -> 3시드 중앙. 시드별 값도 같이 돌려준다."""
    raw = defaultdict(dict)
    for r in good:
        v = getm(r, metric)
        if v is not None and np.isfinite(v):
            raw[(r['cell'], r['episode'])][r['seed']] = float(v)
    med = {k: float(np.median(list(v.values()))) for k, v in raw.items() if v}
    return med, raw


def contrast(med, eps, t, b, rg):
    """짝 Δ. 절대·상대 둘 다. 판정은 절대(A9-6)."""
    P = [(e, med[(t, e)] - med[(b, e)], med[(b, e)])
         for e in eps if regime(e) == rg and (t, e) in med and (b, e) in med]
    d = np.array([x[1] for x in P], float)
    base = np.array([x[2] for x in P], float)
    rel = d / base
    ties = int((d == 0).sum())
    n_eff = len(d) - ties
    imp = int((d < 0).sum())
    wor = int((d > 0).sum())
    pv = binomtest(imp, n_eff, 0.5).pvalue if n_eff else float('nan')
    pv_h = binomtest(wor, n_eff, 0.5).pvalue if n_eff else float('nan')
    return dict(eps=[x[0] for x in P], d=d, base=base, rel=rel, n=len(d), ties=ties,
                n_eff=n_eff, imp=imp, wor=wor, p=pv, p_harm=pv_h,
                med=float(np.median(d)) if len(d) else np.nan,
                med_rel=float(np.median(rel)) if len(d) else np.nan,
                mean_rel=float(np.mean(rel)) if len(d) else np.nan)


def holm(pvals):
    """Holm-Bonferroni 조정 p (단조 보정 포함)."""
    idx = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(idx):
        run = max(run, min(1.0, (m - rank) * pvals[i]))
        adj[i] = run
    return adj


def sign_line(c, name, mde=None, mde80=None):
    s = (f'  {name:34s} 개선 {c["imp"]}/{c["n"]} 동점 {c["ties"]} '
         f'| Δ중앙 {c["med"]:+.5f} ({c["med_rel"] * 100:+.1f}%) '
         f'| mean상대 {c["mean_rel"] * 100:+.1f}% | 원p {c["p"]:.5f}')
    if mde is not None:
        s += f' | |Δ|/mde_agg {abs(c["med"]) / mde:.2f}'
    if mde80 is not None:
        s += f' | |Δ|/mde_sign80 {abs(c["med"]) / mde80:.2f}'
    return s


# ── 본체 ─────────────────────────────────────────────────────────────────────
def main():
    if not os.path.exists(RES):
        sys.exit(f'없음: {RES}')
    recs, good = load()
    eps = sorted({r['episode'] for r in good})
    mde = json.load(open(MDEJ))['by_regime']

    p('=' * 92)
    p('S7 채점 — PREREG.md + AMENDMENT_1.md 집행')
    p(f'  results md5 = {hashlib.md5(open(RES, "rb").read()).hexdigest()}')
    p('=' * 92)

    # ---- I3 완주 / I4 regime 카운트 --------------------------------------------
    p('\n### I3 완주 · 에러')
    err = [r for r in recs if 'error' in r]
    cells = sorted({r['cell'] for r in good})
    p(f'  레코드 {len(recs)} | 성공 {len(good)} | 에러 {len(err)} | 셀 {len(cells)}: {cells}')
    if err:
        p(f'  ⛔ 에러 예: {err[0].get("error", "")[:160]}')
    emit('I3')

    p('\n### I4 regime 카운트  [A1 — 상속 regime() 오라벨 방어]')
    cnt = Counter((r['cell'], r['regime']) for r in good)
    bad = [(c, rg, n) for (c, rg), n in sorted(cnt.items()) if n != 96]
    tot = Counter(r['regime'] for r in good)
    p(f'  총 crowded {tot.get("crowded", 0)} / sparse {tot.get("sparse", 0)} '
      f'(기대 576 / 576)')
    p(f'  셀×레짐 96 아님: {"0  ✅" if not bad else f"⛔ {bad}"}')
    # 경로에서 재도출해 레코드 라벨과 대조 (이중 확인)
    mism = [r['job_id'] for r in good if r['regime'] != regime(r['episode'])]
    p(f'  레코드 라벨 vs 경로 재도출 불일치: {len(mism)}  {"✅" if not mism else "⛔"}')
    emit('I4')

    # ---- I2 cfg 유일 + 실측 노브 -----------------------------------------------
    p('\n### I2 셀별 실측 노브 (의도값과 일치하는가)')
    for c in cells:
        cfgs = {json.dumps({k: v for k, v in (r.get('cfg') or {}).items()}, sort_keys=True)
                for r in good if r['cell'] == c}
        one = json.loads(next(iter(cfgs)))
        p(f'  {c:16s} cfg유일 {len(cfgs)}종 | lambda_l1={one.get("lambda_l1", "기본(1e-3)")} '
          f'| tracking_lr={one.get("tracking_lr", "기본(ON)")} '
          f'| init={one.get("init_method")} | ens_k={one.get("analytic_ensemble_k")}')
    emit('I2')

    # ---- I1 워커 누출 -----------------------------------------------------------
    p('\n### I1 워커 누출 프로브 (AUTO-NT vs AUTO-NT-DUP)')
    med_n, raw_n = per_ep(good, 'nRMSE')
    dup = [(e, med_n[('AUTO-NT', e)] - med_n[('AUTO-NT-DUP', e)]) for e in eps
           if ('AUTO-NT', e) in med_n and ('AUTO-NT-DUP', e) in med_n]
    mx = max((abs(x[1]) for x in dup), default=float('nan'))
    p(f'  n={len(dup)}  max|Δ| = {mx:.2e}  {"✅ 비트 일치" if mx == 0 else "⛔ 불일치"}')
    emit('I1')

    # ---- I11 사전 검정력 (사후 재계산 금지) ------------------------------------
    p('\n### I11 사전 검정력표 (동결값 — ⛔사후 재계산 금지)  [A3]')
    for r_, pw in POWER_PRIOR.items():
        p(f'  참 부호율 {r_:.4f} → P(≥23/32) = {pw:.4f}')
    p('  ⚠분기 사전확률 (S6 기준): DEPLOY 0.937 · BUNDLE-ONLY 0.063 · KNOB-REAL 0.0008 · NULL 0.0001')
    p('  ⟹ ★이 배치가 실제로 시험하는 것은 PRIMARY-1(λ 단독) 뿐이다. P-2 통과는 신규 정보가 아니다.')
    emit('I11')

    # ---- PRIMARY -----------------------------------------------------------------
    p('\n' + '=' * 92)
    p('### I6 / I8  PRIMARY (혼잡 32ep, Holm-2)   판정 = Δ중앙<0 AND Holm-2 p ≤ 0.05  [A10]')
    prim = {}
    for tag, t, b in PRIMARY:
        prim[tag] = contrast(med_n, eps, t, b, 'crowded')
    adj = holm([prim[t]['p'] for t, _, _ in PRIMARY])
    M = mde['crowded']
    # mde_sign80 은 S7 관측 σ_Δ 로 재산출 (사전값도 병기)
    verdict_pass = {}
    for i, (tag, t, b) in enumerate(PRIMARY):
        c = prim[tag]
        s80 = MDE_SIGN80_C * 1.4826 * np.median(np.abs(c['d'] - np.median(c['d'])))
        c['holm'] = float(adj[i])
        c['mde_sign80'] = float(s80)
        p(sign_line(c, f'{tag}: {t} − {b}', M['mde_agg'], s80))
        p(f'  {"":34s} Holm-2 조정 p = {adj[i]:.5f}  '
          f'{"✅ 통과" if (c["med"] < 0 and adj[i] <= ALPHA) else "미달"}'
          f'   (mde_sign80 사전 0.01879 / 관측 {s80:.5f})')
        verdict_pass[tag] = bool(c['med'] < 0 and adj[i] <= ALPHA)
    p('  ⛔"관측/mde_agg ≥ 1" 을 "탐지 가능" 으로 읽지 않는다 — S6 반례: 1.79배인데 13/16 미달  [A6]')
    emit('I6')

    # 반대 방향 (HARMFUL) — 같은 Holm 족의 스텝을 소비한다  [A15]
    harmful = {t: bool(prim[t]['med'] > 0 and adj[i] <= ALPHA)
               for i, (t, _, _) in enumerate(PRIMARY)}

    # ---- I19 크기 재현 D2-S7 -----------------------------------------------------
    p('\n### I19 D2-S7 크기 재현  [A20]')
    mag = {}
    for tag, _, _ in PRIMARY:
        ref = D2_REF[tag]
        frac = abs(prim[tag]['med']) / abs(ref)
        mag[tag] = 'SIGN+MAGNITUDE' if frac >= D2_FRAC else 'SIGN-ONLY'
        p(f'  {tag}: S6 참조 {ref:+.5f} → S7 {prim[tag]["med"]:+.5f} = {frac * 100:.1f}%  ⟹ {mag[tag]}')
    p('  ⚠참조값은 3수준 중 argmax|Δ| 로 **선택된 값**이라 상향 편의가 있다 (승자의 저주, A13)')
    p('  ⛔SIGN-ONLY 이면 S6 수치 인용 중단 (S1 이 −0.0701 에 한 처분과 동일)')
    emit('I19')

    # ---- I13 블록 분해 -------------------------------------------------------------
    p('\n### I13 블록별 (BF × 성분수) — 조건 단위 자유도 4, 조건 라벨 바닥 p = 0.125  [A8]')
    suffix = defaultdict(list)
    for tag, _, _ in PRIMARY:
        c = prim[tag]
        bl = defaultdict(list)
        for e, d in zip(c['eps'], c['d']):
            bl[block(e)].append(d)
        meds = {k: float(np.median(v)) for k, v in bl.items()}
        line = '  '.join(f'{k} {int((np.array(v) < 0).sum())}/{len(v)} ({np.median(v):+.4f})'
                         for k, v in sorted(bl.items()))
        n_fav = sum(1 for k, v in bl.items() if (np.array(v) < 0).sum() > len(v) / 2)
        ratio = max(map(abs, meds.values())) / max(min(map(abs, meds.values())), 1e-12)
        p(f'  {tag}: {line}')
        p(f'      우세 블록 {n_fav}/4 | 블록 간 |Δ중앙| 비 {ratio:.1f}배')
        if n_fav < 3:
            suffix[tag].append('CONDITION-SPECIFIC')
        if ratio > 5:
            suffix[tag].append('HETEROGENEOUS')
    emit('I13')

    # ---- I14 지지집합 안/밖 + 셸 ---------------------------------------------------
    p('\n### I14 지지집합 안/밖 + 9셸  [A2 — 전역 nRMSE 가 못 보는 거래]')
    for tag, t, b in PRIMARY:
        p(f'  --- {tag}: {t} − {b} (혼잡) ---')
        for key, lab in (('RMSE_obs_r30', '관측 ≤30m'), ('RMSE_unobs_r30', '미관측 >30m'),
                         ('RMSE_obs_r5', '관측 ≤5m'), ('RMSE_unobs_r5', '미관측 >5m')):
            m2, _ = per_ep(good, key)
            c2 = contrast(m2, eps, t, b, 'crowded')
            p(sign_line(c2, f'    {lab}'))
        near_bad = []
        for sk in SHELL_KEYS:
            m2, _ = per_ep(good, f'shell:{sk}')
            c2 = contrast(m2, eps, t, b, 'crowded')
            flag = ''
            if sk in NEAR_SHELLS and c2['wor'] >= HARM_SIGN_K:
                flag = '  ⛔악화'
                near_bad.append(sk)
            p(f'    셸 {sk:9s} 개선 {c2["imp"]}/{c2["n"]}  Δ중앙 {c2["med"]:+.5f}{flag}')
        if near_bad:
            suffix[tag].append('SUPPORT-TRADEOFF')
            p(f'    ⟹ ★SUPPORT-TRADEOFF ({near_bad}) — ⛔"전역적으로 좋아진다"·"배포 개선" 금지')
    p('  ⚠E-D(`0_5` −0.1417 … `30_50` +0.0177 … `100_150` +0.0577)와 **방향만** 대조. ⛔풀링 금지')
    emit('I14')

    # ---- I8 판정 -------------------------------------------------------------------
    p('\n' + '=' * 92)
    p('### I8 PRIMARY 판정')
    d3 = contrast(med_n, eps, 'AUTO-NT-L1-3e4', 'AUTO-NT', 'crowded')
    d3_pass = bool(d3['med'] < 0 and d3['p'] <= ALPHA)
    if any(harmful.values()):
        base_v = 'HARMFUL'
    elif verdict_pass['P1'] and verdict_pass['P2']:
        base_v = 'DEPLOY-CANDIDATE'
    elif verdict_pass['P1']:
        base_v = 'KNOB-REAL / DEPLOY-UNSUPPORTED'
    elif verdict_pass['P2']:
        base_v = 'BUNDLE-ONLY'
    elif d3_pass:
        base_v = 'SECONDARY-ONLY'
    else:
        base_v = 'NULL'
    sfx = sorted(set(suffix['P1']) | set(suffix['P2']) | {mag['P1'], mag['P2']})
    p(f'  ★판정 = **{base_v}**' + (f'  /{"  /".join(sfx)}' if sfx else ''))
    p(f'     P-1(λ 단독) {"통과" if verdict_pass["P1"] else "미달"} · '
      f'P-2(묶음 vs 출하) {"통과" if verdict_pass["P2"] else "미달"} · '
      f'SECONDARY 3e-4 {"통과" if d3_pass else "미달"}')
    if base_v == 'BUNDLE-ONLY':
        p('  ⛔이 분기는 **새 정보가 아니다** — "추적기-off 재현(6번째) + λ 단독 미지지" 라고 적는다  [A3]')
    if base_v == 'SECONDARY-ONLY':
        p('  ⛔3e-4 를 배포 후보로 부르지 않는다 — "재사전등록 필요"  [A13]')
    p('  ⛔PRIMARY-2 통과를 λ 축의 증거로 인용 금지  [A3]')
    p('  ⛔PRIMARY-2 에 단일 원인 이름 금지 (추적기 + λ 2노브)')
    emit('I8')

    # ---- I9 희소 게이트 ------------------------------------------------------------
    p('\n' + '=' * 92)
    p('### I9 희소 게이트 (독립 검정족, ⚠**무보정** — 위음성이 위험한 오류)  [A4]')
    Ms = mde['sparse']
    sparse_verdicts = {}
    for tag, t, b in PRIMARY:
        c = contrast(med_n, eps, t, b, 'sparse')
        p(sign_line(c, f'{tag}(희소): {t} − {b}', Ms['mde_agg']))
        labels = []
        if c['wor'] >= HARM_SIGN_K and c['p_harm'] <= 0.1102:
            labels.append('HARM-SIGN')
        # HARM-TAIL (a)-(d)
        tail = []
        if int((c['rel'] > 0.5).sum()) >= 6:
            tail.append(f'(a) 상대악화>+50% {int((c["rel"] > 0.5).sum())}/{c["n"]}')
        if c['rel'].max() > 2.0:
            tail.append(f'(b) 최대 상대악화 {c["rel"].max() * 100:+.1f}%')
        if np.sign(c['mean_rel']) != np.sign(c['med_rel']):
            tail.append(f'(c) mean {c["mean_rel"] * 100:+.1f}% vs 중앙 {c["med_rel"] * 100:+.1f}% 부호 반대')
        bl = defaultdict(list)
        for e, d in zip(c['eps'], c['d']):
            bl[block(e)].append(d)
        allbad = [k for k, v in bl.items() if (np.array(v) > 0).all()]
        if allbad:
            tail.append(f'(d) 전부 악화 블록 {allbad}')
        if tail:
            labels.append('HARM-TAIL')
        if c['imp'] >= 23 and c['p'] <= ALPHA:
            labels.append('SPARSE-BENEFIT')
        if c['med'] < Ms['mde_agg'] and not tail:
            labels.append(f'SPARSE-NONINFERIOR(δ={Ms["mde_agg"]:.5f})')
        if not labels:
            labels = ['SPARSE-UNDECIDED']
        sparse_verdicts[tag] = labels
        p(f'    → {" + ".join(labels)}')
        for tt in tail:
            p(f'      꼬리: {tt}')
        for b_, v in sorted(bl.items()):
            p(f'      블록 {b_}: 개선 {int((np.array(v) < 0).sum())}/{len(v)}  '
              f'Δ중앙 {np.median(v):+.5f}')
    p('  검정력(사전): 참 피해율 0.6875→0.72 · 0.70→0.77 · 0.75→0.92 (문턱 21/32)')
    p('  ⛔"무해하다" 금지. 미검출이면 **검출 가능했던 최소 피해를 반드시 병기**한다.')
    if any('HARM' in x for v in sparse_verdicts.values() for x in v):
        p('  ⛔★희소 피해 검출 ⟹ 혼잡이 통과해도 "배포 개선" 금지. '
          '허용 = "혼잡 전용 이득 + 희소 피해"')
    emit('I9')

    # ---- I21 DEPLOY-SINGLE ---------------------------------------------------------
    p('\n### I21 DEPLOY-SINGLE 족 — 출하 yaml **한 줄**(`lambda_l1`) 만 바꾼 팔  [A17]')
    tag, t, b = SINGLE
    cs = {rg: contrast(med_n, eps, t, b, rg) for rg in ('crowded', 'sparse')}
    adjs = holm([cs['crowded']['p'], cs['sparse']['p']])
    for i, rg in enumerate(('crowded', 'sparse')):
        c = cs[rg]
        p(sign_line(c, f'{t} − {b} ({rg})', mde[rg]['mde_agg']))
        ok = c['med'] < 0 and adjs[i] <= ALPHA
        p(f'  {"":34s} Holm-2(자체족) p = {adjs[i]:.5f}  '
          f'→ {"SINGLE-KNOB-SUPPORTED" if ok else "SINGLE-KNOB-UNSUPPORTED"}')
    p('  ⛔이 족의 통과는 PRIMARY 판정에 기여하지 않는다 (§1 판정표 불변)')
    emit('I21')

    # ---- I10 묶음 분해 (에피소드별) --------------------------------------------------
    p('\n### I10 묶음 분해 — **에피소드별**  [A18 — 중앙끼리는 더해지지 않는다]')
    a = contrast(med_n, eps, 'AUTO-NT', 'AUTO', 'crowded')            # 추적기 몫
    b2 = contrast(med_n, eps, 'AUTO-NT-L1-1e4', 'AUTO-NT', 'crowded')  # λ 몫
    tot_c = prim['P2']
    p(sign_line(a, '  추적기 off 몫 (AUTO-NT − AUTO)'))
    p(sign_line(b2, '  λ 몫 (L1-1e4 − AUTO-NT)'))
    p(sign_line(tot_c, '  합계 (L1-1e4 − AUTO)'))
    pere = np.array([x + y for x, y in zip(a['d'], b2['d'])]) - tot_c['d']
    p(f'  에피소드별 항등식 잔차 max|.| = {np.max(np.abs(pere)):.2e} (0 이어야 정상)')
    p(f'  ⚠중앙 합 {a["med"] + b2["med"]:+.5f} ≠ 합계 중앙 {tot_c["med"]:+.5f} — **정상**이다')
    p('  ⛔순차 분해이지 직교 분해가 아니다 (S5 H6 제한 승계)')
    p('  ✅`AUTO-NT − AUTO` 는 S3·S5·S6 에 이은 **여섯 번째 재현 시험**')
    emit('I10')

    # ---- I15 라우트 ----------------------------------------------------------------
    p('\n### I15 `auto` 라우트 분포 + ensemble_k  [A11 — "혼잡 전용" vs "라우트 전용"]')
    for rg in ('crowded', 'sparse'):
        rc = Counter((r.get('auto_selected'), r.get('ensemble_k'))
                     for r in good if r['regime'] == rg)
        p(f'  {rg}: {dict(rc)}')
    routes = {r['episode']: r.get('auto_selected') for r in good if r['cell'] == 'AUTO'}
    for tag, t, b in PRIMARY:
        c = prim[tag]
        strat = defaultdict(list)
        for e, d in zip(c['eps'], c['d']):
            strat[routes.get(e)].append(d)
        p(f'  {tag} 라우트별: ' + '  '.join(
            f'{k}: {int((np.array(v) < 0).sum())}/{len(v)} ({np.median(v):+.5f})'
            for k, v in sorted(strat.items(), key=lambda x: str(x[0]))))
    p('  ⛔"혼잡 전용" 대신 **"(혼잡 ∧ 실현 라우트 X) 전용"** 으로 표기')
    emit('I15')

    # ---- I16 기제 + ΔN --------------------------------------------------------------
    p('\n### I16 기제 계측 + 용량(N) 교란  [A16]')
    for key in ('window_resid', 'all_resid', 'decorr_mean', 'shrink_mean', 'N'):
        m2, _ = per_ep(good, key)
        for tag, t, b in PRIMARY:
            c2 = contrast(m2, eps, t, b, 'crowded')
            p(sign_line(c2, f'  {key} · {tag}(혼잡)'))
    mN, _ = per_ep(good, 'N')
    dN = contrast(mN, eps, 'AUTO-NT-L1-1e4', 'AUTO-NT', 'crowded')
    if dN['med'] != 0 or dN['ties'] < dN['n']:
        p('  ⛔★ΔN ≠ 0 ⟹ "정규화 효과" 라는 이름을 쓰지 않고 **"정규화+용량 묶음"** 으로 적는다')
    p('  ⛔`window_resid` 는 **무가중**이며 학습 목적함수가 아니다 (S5 §5 승계)')
    p('  ⛔기제 지표는 판정에 쓰지 않는다')
    emit('I16')

    # ---- I17 alpha_wave --------------------------------------------------------------
    p('\n### I17 alpha_wave (SECONDARY · 보정 없음 · 판정 미사용)  [A21]')
    ma, _ = per_ep(good, 'alpha')
    if ma:
        for rg in ('crowded', 'sparse'):
            for tag, t, b in PRIMARY:
                c2 = contrast(ma, eps, t, b, rg)
                p(sign_line(c2, f'  α_wave(lead {ALPHA_LEAD}) · {tag}({rg})'))
    else:
        p('  ⛔alpha_legs 미기록 — S2_ALPHA 게이트를 확인하라')
    p('  ⛔"제어 성능이 좋아진다" 금지. 허용 = "동일 스텐실에서 α_wave 도 같은/반대 방향"')
    p('  ⛔제어점(x≈4 m) ≠ 관측 중심(x≈20.8 m). ⛔α_wave MDE 사전 동결본 없음')
    emit('I17')

    # ---- I18 수준 vs 짝Δ --------------------------------------------------------------
    p('\n### I18 수준(중앙) 대 짝 Δ — CONVENTION-CONFLICT 점검  [A19]')
    for rg in ('crowded', 'sparse'):
        lv = {c: float(np.median([med_n[(c, e)] for e in eps
                                  if regime(e) == rg and (c, e) in med_n])) for c in cells}
        p(f'  {rg} 수준: ' + '  '.join(f'{k} {v:.5f}' for k, v in sorted(lv.items())))
        for tag, t, b in PRIMARY:
            c = contrast(med_n, eps, t, b, rg)
            if np.sign(lv[t] - lv[b]) != np.sign(c['med']):
                p(f'    ⛔CONVENTION-CONFLICT {tag}: 수준차 {lv[t] - lv[b]:+.5f} vs '
                  f'짝Δ중앙 {c["med"]:+.5f} — **판정은 짝 Δ 규약**')
    p('  ⛔수준표를 판정 근거로 인용 금지 (S6 희소에서 실제로 갈렸다)')
    emit('I18')

    # ---- I7 시드별 --------------------------------------------------------------------
    p('\n### I7 시드별 부호수 + mean-of-3 (규약 산물 여부)')
    seeds = sorted({r['seed'] for r in good})
    for tag, t, b in PRIMARY:
        line = []
        for s in seeds:
            m1 = {(r['cell'], r['episode']): r['nRMSE'] for r in good if r['seed'] == s}
            c2 = contrast(m1, eps, t, b, 'crowded')
            line.append(f'{s}: {c2["imp"]}/{c2["n"]}')
        mm = {}
        for r in good:
            mm.setdefault((r['cell'], r['episode']), []).append(r['nRMSE'])
        mm = {k: float(np.mean(v)) for k, v in mm.items()}
        c3 = contrast(mm, eps, t, b, 'crowded')
        p(f'  {tag}: ' + '  '.join(line) + f'  | mean-of-3: {c3["imp"]}/{c3["n"]}')
    emit('I7')

    # ---- I12 S6 대비 수준 ---------------------------------------------------------------
    p('\n### I12 S6 대비 수준 (⛔짝 아님 — 다른 에피소드. 풀링·짝검정 금지)')
    s6p = os.path.join(HERE, '..', 'S6_l1_sweep_2026-08-08', 'results.jsonl')
    if os.path.exists(s6p):
        s6 = [json.loads(l) for l in open(s6p) if l.strip()]
        s6 = [r for r in s6 if 'error' not in r]
        s6m = defaultdict(list)
        for r in s6:
            s6m[(r['cell'], r['episode'])].append(r['nRMSE'])
        s6m = {k: float(np.median(v)) for k, v in s6m.items()}
        s6eps = sorted({e for _, e in s6m})

        def s6lev(cell, rg):
            v = [s6m[(cell, e)] for e in s6eps
                 if (cell, e) in s6m
                 and (('15comp' in e or '30comp' in e) == (rg == 'crowded'))]
            return float(np.median(v)) if v else float('nan')
        for rg in ('crowded', 'sparse'):
            for c in ('AUTO', 'AUTO-NT', 'AUTO-NT-L1-1e4'):
                s7v = [med_n[(c, e)] for e in eps if regime(e) == rg and (c, e) in med_n]
                p(f'  {rg:8s} {c:16s} S6 {s6lev(c, rg):.5f} → S7 {np.median(s7v):.5f}')
        p('  ⚠블록 난이도가 크게 다르면 **사전 검정력표는 이 블록에 대해 무효**다  [CON C-10c]')
    emit('I12')

    # ---- I5 / I20 외부 산출물 --------------------------------------------------------
    p('\n### I5 실행 전 게이트 재수록 · I20 S2_ALPHA 비트동일')
    for f, tag in (('gate.txt', 'I5'), ('alpha_bitcheck.txt', 'I20')):
        fp = os.path.join(HERE, f)
        if os.path.exists(fp):
            body = open(fp).read().strip().splitlines()
            p(f'  --- {f} (마지막 6줄) ---')
            for line in body[-6:]:
                p(f'    {line}')
            emit(tag)
        else:
            p(f'  ⛔ {f} 없음')

    # ---- §5 자체 채점 (하드코딩 금지) --------------------------------------------------
    p('\n' + '=' * 92)
    missing = [x for x in REQUIRED if x.split()[0] not in EMITTED]
    p(f'### §5 자체 채점: {len(EMITTED)}/{len(REQUIRED)} 항 출력')
    if missing:
        p('⛔ §5 위반 — 이 배치의 무결성 주장은 인용 불가. 누락:')
        for m in missing:
            p(f'   - {m}')
    else:
        p('✅ §5 전항 출력')
    p('=' * 92)

    open(os.path.join(HERE, 'analysis.txt'), 'w').write('\n'.join(OUT) + '\n')


if __name__ == '__main__':
    main()
