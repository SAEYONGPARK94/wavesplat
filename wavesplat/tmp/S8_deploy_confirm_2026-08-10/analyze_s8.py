"""S8 채점기 — `PREREG.md` + `AMENDMENT_1.md` 를 그대로 집행한다.

★**결과를 읽기 전에 md5 동결한다.** 동결 후 수정하지 않는다.

집행 규약 (A15 로 S7 A9 전항 승계):
  에피소드당 3시드 **중앙** → 짝 Δ(같은 에피소드) → **에피소드별 Δ 부호**가 검정 통계량(n=32)
  판정 규약 = **절대**, 상대는 의무 병기(= 에피소드별 비의 **중앙**, ⛔중앙들의 비 아님)
  `FIELD-NRMSE-CONFIRMED`(두 레짐 동시) = **IUT** ⟹ 각 레짐 **원 p ≤ 0.05**
  개별 주장(`CROWDED-ONLY`/`SPARSE-ONLY`) = **Holm-2**                        [A11]
"""
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np
from scipy.stats import binomtest

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get('S8_OUT', os.path.join(HERE, 'results.jsonl'))
MDEJ = os.path.join(HERE, 'MDE_prespec_s8.json')

# ── 동결 상수 ────────────────────────────────────────────────────────────────
ALPHA = 0.05
HARM_K = 21                      # 안전/셸 게이트 문턱 (무보정)
BEST_K = 23                      # /NOT-BEST · /CONTROL-NULL 문턱
D2_REF_CROWDED = -0.019153       # A12 — 혼잡에만 적용
D2_FRAC = 0.50
SHELLS = ('0_5', '5_10', '10_20', '20_30', '30_50', '50_75', '75_100', '100_150', '150_inf')
NEAR = ('0_5', '5_10', '10_20')
ALPHA_LEAD = '0.2'
ALPHA_DELTA = {'sparse': 0.0356}                       # A21 비열등 마진 (희소만)
TREAT, BASE = 'AUTO-L1-1e4', 'AUTO'
NULL_CELL = 'AUTO-SEED-NULL'
POWER_PRIOR = {'crowded': (0.8125, 0.9374, (0.636, 0.928)),
               'sparse': (0.71875, 0.5886, (0.533, 0.863))}
BRANCH_PRIOR = ('BOTH 0.552 / C-ONLY 0.386 / S-ONLY 0.037 / NEITHER 0.026 (관측 부호율)',
                'BOTH 0.198 / C-ONLY 0.539 / S-ONLY 0.071 / NEITHER 0.193 (승자의 저주 0.75/0.65)',
                'ρ=0.5 코퓰러: S-ONLY 0.012 · NEITHER 0.050')
GATE_MDE80 = {'crowded': {'0_5': 9.2, '5_10': 10.7, '10_20': 7.8},
              'sparse': {'0_5': 20.3, '5_10': 15.8, '10_20': 15.8}}

REQUIRED = [
    'I1  워커 누출 (AUTO vs AUTO-DUP2)',
    'I2  셀당 cfg 유일 + 실측 lambda_l1 / tracking_lr / loci gate',
    'I3  1920/1920 · 에러 0',
    'I4  regime 카운트 + 경로 재도출 대조',
    'I5  실행 전 게이트 G-1~G-6 재수록',
    'I6  PRIMARY 대조표 (절대·상대·부호·동점·원p·Holm·mde_agg·mde_sign80)',
    'I7  판정 + 접미 라벨 (IUT / Holm-2 병기)',
    'I8  9셸 + 근거리 게이트 + all_resid + MDE80 표  [A4/A5]',
    'I9  안전 게이트 + 무처치 널 대조 (TAIL-UNCALIBRATED)  [A2]',
    'I10 D2-S8 크기 재현 (혼잡만)  [A12]',
    'I11 블록별 부호수 + CONDITION-SPECIFIC / HETEROGENEOUS  [A1]',
    'I12 전 셀 PRIMARY 지표 순위표 + /NOT-BEST  [A3]',
    'I13 α_wave 수준 + 희소 비열등성 + /CONTROL-NULL  [A21]',
    'I14 λ 용량-반응 (출하 팔) + SECONDARY-ONLY  [A10]',
    'I15 2-loci 양방향 분리  [A9]',
    'I16 tracking_lr 용량-반응 (DOSE-MONOTONE / SWITCH / UNDECIDED)  [A14]',
    'I17 프레임 실패 꼬리 + /TAIL-FRAME  [A20]',
    'I18 기제 계측 + ΔN + 계산량 구조  [A7/A16]',
    'I19 라우트 불변 점검  [A19]',
    'I20 수준 대 짝Δ CONVENTION-CONFLICT',
    'I21 시드별 + mean-of-3',
    'I22 사전 검정력·분기확률 재수록 (⛔사후 재계산 금지)',
    'I23 S7 대비 수준 (⛔짝 아님 · 풀링/p결합 금지)',
]
EMITTED, OUT = set(), []


def p(s=''):
    print(s, flush=True)
    OUT.append(s)


def emit(t):
    EMITTED.add(t)


def regime(ep):
    if ep.startswith('data_s8_crowded'):
        return 'crowded'
    if ep.startswith('data_s8_sparse'):
        return 'sparse'
    raise ValueError(ep)


def block(ep):
    m = re.search(r'episode_(BF\d)_(\d+)comp', ep)
    return f'{m.group(1)}_{m.group(2)}c'


def getm(rec, k):
    if k.startswith('shell:'):
        return (rec.get('shell_rmse') or {}).get(k.split(':', 1)[1], np.nan)
    if k == 'alpha':
        return (rec.get('alpha_episode') or {}).get(ALPHA_LEAD, np.nan)
    v = rec.get(k, np.nan)
    return np.nan if v is None else v


def per_ep(good, k):
    raw = defaultdict(list)
    for r in good:
        v = getm(r, k)
        if v is not None and np.isfinite(v):
            raw[(r['cell'], r['episode'])].append(float(v))
    return {kk: float(np.median(v)) for kk, v in raw.items()}


def C(med, eps, t, b, rg):
    e = [x for x in eps if regime(x) == rg and (t, x) in med and (b, x) in med]
    d = np.array([med[(t, x)] - med[(b, x)] for x in e])
    base = np.array([med[(b, x)] for x in e])
    rel = d / base
    ties = int((d == 0).sum())
    n_eff = len(d) - ties
    imp, wor = int((d < 0).sum()), int((d > 0).sum())
    return dict(eps=e, d=d, base=base, rel=rel, n=len(d), ties=ties, n_eff=n_eff,
                imp=imp, wor=wor,
                p=binomtest(imp, n_eff, .5).pvalue if n_eff else np.nan,
                p_h=binomtest(wor, n_eff, .5).pvalue if n_eff else np.nan,
                med=float(np.median(d)) if len(d) else np.nan,
                med_rel=float(np.median(rel)) if len(d) else np.nan,
                mean_rel=float(np.mean(rel)) if len(d) else np.nan)


def holm(pv):
    idx, m, adj, run = np.argsort(pv), len(pv), np.empty(len(pv)), 0.0
    for rank, i in enumerate(idx):
        run = max(run, min(1.0, (m - rank) * pv[i]))
        adj[i] = run
    return adj


def L(c, name, extra=''):
    return (f'  {name:40s} 개선 {c["imp"]}/{c["n"]} 동점 {c["ties"]} | '
            f'Δ중앙 {c["med"]:+.5f} ({c["med_rel"] * 100:+.1f}%) | '
            f'mean상대 {c["mean_rel"] * 100:+.1f}% | 원p {c["p"]:.5f}{extra}')


def main():
    recs = [json.loads(l) for l in open(RES) if l.strip()]
    good = [r for r in recs if 'error' not in r]
    eps = sorted({r['episode'] for r in good})
    cells = sorted({r['cell'] for r in good})
    M = json.load(open(MDEJ))['by_regime']
    mn = per_ep(good, 'nRMSE')
    sfx = defaultdict(set)

    p('=' * 94)
    p('S8 채점 — PREREG.md + AMENDMENT_1.md 집행')
    p(f'  results md5 = {hashlib.md5(open(RES, "rb").read()).hexdigest()}')
    p('=' * 94)

    p('\n### I3 완주')
    err = [r for r in recs if 'error' in r]
    p(f'  레코드 {len(recs)} | 성공 {len(good)} | 에러 {len(err)} | 셀 {len(cells)}: {cells}')
    if err:
        p(f'  ⛔ {err[0].get("error", "")[:160]}')
    emit('I3')

    p('\n### I4 regime 카운트')
    tot = Counter(r['regime'] for r in good)
    bad = [(c, rg, n) for (c, rg), n in sorted(Counter(
        (r['cell'], r['regime']) for r in good).items()) if n != 96]
    mism = [r['job_id'] for r in good if r['regime'] != regime(r['episode'])]
    p(f'  crowded {tot.get("crowded", 0)} / sparse {tot.get("sparse", 0)} (기대 960/960) | '
      f'셀×레짐≠96: {len(bad)} | 경로 재도출 불일치: {len(mism)}')
    emit('I4')

    p('\n### I2 셀별 실측 노브')
    for c in cells:
        cf = {json.dumps({k: v for k, v in (r.get('cfg') or {}).items()}, sort_keys=True)
              for r in good if r['cell'] == c}
        one = json.loads(next(iter(cf)))
        g = {r.get('loci_gate', '') for r in good if r['cell'] == c}
        sd = sorted({r['seed'] for r in good if r['cell'] == c})
        p(f'  {c:22s} cfg {len(cf)}종 | l1={one.get("lambda_l1", "기본1e-3")} | '
          f'trk_lr={one.get("tracking_lr", "기본0.005")} | gate={sorted(g)} | seeds={sd}')
    emit('I2')

    p('\n### I1 워커 누출 (AUTO vs AUTO-DUP2)')
    dd = [abs(mn[('AUTO', e)] - mn[('AUTO-DUP2', e)]) for e in eps
          if ('AUTO-DUP2', e) in mn]
    p(f'  n={len(dd)}  max|Δ| = {max(dd, default=float("nan")):.2e}  '
      f'{"✅ 비트 일치" if dd and max(dd) == 0 else "⛔"}')
    emit('I1')

    p('\n### I22 사전 검정력·분기확률 (동결값, ⛔사후 재계산 금지)')
    for rg, (r_, pw, ci) in POWER_PRIOR.items():
        p(f'  {rg}: 참 부호율 {r_:.5f} → P(≥23/32)={pw:.4f}  CP95={ci}')
    for b in BRANCH_PRIOR:
        p(f'  분기: {b}')
    p('  ⛔"가장 흔한 결말조차 55%" 를 설계 정당화로 쓰지 않는다 (A13)')
    p('  ⛔"S7·S8 에서 두 번 독립 재현" 금지 — S7=발견/제안, S8=1차 확증(n=1). ⛔p 결합도 금지')
    emit('I22')

    # ── PRIMARY ───────────────────────────────────────────────────────────────
    p('\n' + '=' * 94)
    p('### I6/I7  PRIMARY = `AUTO-L1-1e4 − AUTO`  (IUT: 각 레짐 원 p ≤ 0.05 / 개별: Holm-2)')
    prim = {rg: C(mn, eps, TREAT, BASE, rg) for rg in ('crowded', 'sparse')}
    adj = holm([prim['crowded']['p'], prim['sparse']['p']])
    iut, hol = {}, {}
    for i, rg in enumerate(('crowded', 'sparse')):
        c = prim[rg]
        s80 = 0.7234 * 1.4826 * np.median(np.abs(c['d'] - np.median(c['d'])))
        iut[rg] = bool(c['med'] < 0 and c['p'] <= ALPHA)
        hol[rg] = bool(c['med'] < 0 and adj[i] <= ALPHA)
        p(L(c, f'PRIMARY-{rg[0].upper()} ({rg})',
            f' | |Δ|/mde_agg {abs(c["med"]) / M[rg]["mde_agg"]:.2f}'))
        p(f'  {"":40s} IUT(원p) {"통과" if iut[rg] else "미달"} · '
          f'Holm-2 {adj[i]:.5f} {"통과" if hol[rg] else "미달"} · '
          f'mde_sign80 사전 {M[rg]["mde_sign80"]:.5f} / 관측 {s80:.5f}')
    p('  ⛔"관측/mde_agg ≥ 1" 을 "탐지 가능"으로 읽지 않는다 (S6 반례 1.79배·미달)')
    p('  ⚠유효 n<32 이면 IUT 와 Holm 이 갈릴 수 있다 — 그 경우 두 판정을 병기한다 (A11)')
    emit('I6')

    # 셸 · all_resid 게이트
    p('\n### I8 9셸 + 근거리 게이트 + all_resid  [A4/A5]')
    worse_shell = defaultdict(list)
    for rg in ('crowded', 'sparse'):
        p(f'  --- {rg} ---')
        for k, lab in (('all_resid', 'all_resid(무가중 센서잔차)'),
                       ('RMSE_obs_r5', '관측 ≤5m'), ('RMSE_obs_r30', '관측 ≤30m'),
                       ('RMSE_unobs_r30', '미관측 >30m')):
            c2 = C(per_ep(good, k), eps, TREAT, BASE, rg)
            flag = '  ⛔악화' if (k == 'all_resid' and c2['wor'] >= HARM_K) else ''
            if flag:
                worse_shell[rg].append('all_resid')
            p(L(c2, f'    {lab}') + flag)
        for sk in SHELLS:
            c2 = C(per_ep(good, f'shell:{sk}'), eps, TREAT, BASE, rg)
            bad = c2['wor'] >= HARM_K
            if bad:
                worse_shell[rg].append(sk)
            mde = GATE_MDE80[rg].get(sk)
            p(f'    셸 {sk:9s} 개선 {c2["imp"]}/{c2["n"]} 악화 {c2["wor"]}/{c2["n"]}  '
              f'Δ중앙 {c2["med"]:+.5f} ({c2["med_rel"] * 100:+.1f}%)'
              + (f'  [MDE80 +{mde}%]' if mde else '') + ('  ⛔악화 셸' if bad else ''))
        near_bad = [s for s in worse_shell[rg] if s in NEAR]
        if near_bad:
            sfx[rg].add('SUPPORT-TRADEOFF')
        p(f'    ⟹ 악화 셸 {worse_shell[rg] or "0개"} · 근거리 악화 {near_bad or "없음"}')
    p('  ⛔"근거리 무손상" 금지 → 허용 = "위 MDE80 이상의 근거리 손상은 검출되지 않았다"')
    p('  ⛔위양성: 널에서 셸당 0.055 · 근거리3셸 0.156 · 9셸 0.400 ⟹ 미발화는 무해의 증거가 아니다')
    p('  ⛔`avg_obs_loss` 는 λ·Σ|w| 를 포함 ⟹ 관측 적합 근거로 인용 금지 (A5)')
    emit('I8')

    # 안전 게이트 + 무처치 널
    p('\n### I9 안전 게이트 + 무처치 널 대조  [A2]')
    for rg in ('crowded', 'sparse'):
        c = prim[rg]
        nul = C(mn, eps, NULL_CELL, BASE, rg)
        mde_ep = M[rg]['mde_ep']

        def tail(cc):
            g = np.abs(cc['d']) > mde_ep                       # ★절대 크기 가드 (A2)
            rel_g = cc['rel'][g]
            out = {}
            out['a'] = int((rel_g > 0.5).sum())
            out['b'] = float(rel_g.max()) if rel_g.size else float('nan')
            out['c'] = np.sign(cc['mean_rel']) != np.sign(cc['med_rel'])
            bl = defaultdict(list)
            for e, x in zip(cc['eps'], cc['d']):
                bl[block(e)].append(x)
            out['d'] = [k for k, v in bl.items() if (np.array(v) > 0).all()]
            out['n_guard'] = int(g.sum())
            return out

        t_, n_ = tail(c), tail(nul)
        fired = {'a': t_['a'] >= 6, 'b': (t_['b'] > 2.0), 'c': bool(t_['c']), 'd': bool(t_['d'])}
        nullf = {'a': n_['a'] >= 6, 'b': (n_['b'] > 2.0), 'c': bool(n_['c']), 'd': bool(n_['d'])}
        labels = []
        if c['wor'] >= HARM_K and c['p_h'] <= 0.1102:
            labels.append('HARM-SIGN')
        live = [k for k in 'abcd' if fired[k] and not nullf[k]]
        uncal = [k for k in 'abcd' if fired[k] and nullf[k]]
        if live:
            labels.append(f'HARM-TAIL({",".join(live)})')
        if uncal:
            labels.append(f'TAIL-UNCALIBRATED({",".join(uncal)})')
        if c['imp'] >= 23 and c['p'] <= ALPHA:
            labels.append('BENEFIT')
        if c['med'] < M[rg]['mde_agg'] and not live:
            labels.append(f'NONINFERIOR(δ={M[rg]["mde_agg"]:.5f})')
        p(f'  {rg}: 처치 (a){t_["a"]} (b){t_["b"] * 100:+.1f}% (c){t_["c"]} (d){t_["d"]} '
          f'[가드통과 {t_["n_guard"]}/32]')
        p(f'  {"":8s} 널   (a){n_["a"]} (b){n_["b"] * 100:+.1f}% (c){n_["c"]} (d){n_["d"]} '
          f'[가드통과 {n_["n_guard"]}/32]  ({NULL_CELL} − {BASE}, 처치=0)')
        p(f'    → {" + ".join(labels) or "UNDECIDED"}')
        if any('HARM-TAIL' in x or x == 'HARM-SIGN' for x in labels):
            sfx[rg].add('TAIL-RISK')
    p('  ⛔널에서도 발화하는 기준은 판정력이 없다 — "피해 검출"·"무해" 둘 다 금지 (A2)')
    p('  ⛔"무해하다" 금지. 미검출이면 검출 가능했던 최소 피해를 병기한다.')
    emit('I9')

    # D2 (혼잡만)
    p('\n### I10 D2-S8 크기 재현 (⚠혼잡만 — 희소 참조는 검출바닥의 0.43배)  [A12]')
    frac = abs(prim['crowded']['med']) / abs(D2_REF_CROWDED)
    mag = 'SIGN+MAGNITUDE' if frac >= D2_FRAC else 'SIGN-ONLY'
    sfx['crowded'].add(mag)
    p(f'  혼잡: S7 참조 {D2_REF_CROWDED:+.6f} → S8 {prim["crowded"]["med"]:+.6f} '
      f'= {frac * 100:.1f}%  ⟹ {mag}')
    p(f'  희소: S8 {prim["sparse"]["med"]:+.6f} '
      f'(mde_sign80 {M["sparse"]["mde_sign80"]:.5f}) — ⛔크기 재현 평가 불가, 라벨 없음')
    p('  ⛔희소 백분율의 헤드라인 사용 금지 (절대 Δ·mde_sign80 병기 의무)')
    emit('I10')

    # 블록
    p('\n### I11 블록별 (조건 자유도 4, 조건 라벨 바닥 p 0.125)  [A1]')
    for rg in ('crowded', 'sparse'):
        c = prim[rg]
        bl = defaultdict(list)
        for e, x in zip(c['eps'], c['d']):
            bl[block(e)].append(x)
        meds = {k: float(np.median(v)) for k, v in bl.items()}
        nfav = sum(1 for k, v in bl.items() if (np.array(v) < 0).sum() > len(v) / 2)
        ratio = max(map(abs, meds.values())) / max(min(map(abs, meds.values())), 1e-12)
        p(f'  {rg}: ' + '  '.join(f'{k} {int((np.array(v) < 0).sum())}/{len(v)} '
                                  f'({np.median(v):+.5f})' for k, v in sorted(bl.items())))
        p(f'      우세 {nfav}/4 · 비 {ratio:.1f}배')
        if nfav < 3:
            sfx[rg].add('CONDITION-SPECIFIC')
        if ratio > 5:
            sfx[rg].add('HETEROGENEOUS')
    p('  ⛔"BF·성분수 전반에 유효" 금지 — α=0.05 원리적 불가. 범위 = BF4–BF5 · 2/5/15/30 성분')
    emit('I11')

    # 순위표 + NOT-BEST
    p('\n### I12 전 셀 PRIMARY 지표 순위 + /NOT-BEST  [A3]')
    for rg in ('crowded', 'sparse'):
        lv = sorted(((c, float(np.median([mn[(c, e)] for e in eps
                                          if regime(e) == rg and (c, e) in mn])))
                     for c in cells), key=lambda x: x[1])
        p(f'  {rg} 수준 순위: ' + '  '.join(f'{k}={v:.5f}' for k, v in lv))
        beat = []
        for c in cells:
            if c in (TREAT, 'AUTO-DUP2', NULL_CELL):
                continue
            cc = C(mn, eps, c, TREAT, rg)
            if cc['imp'] >= BEST_K and cc['p'] <= ALPHA:
                beat.append(f'{c}({cc["imp"]}/32, {cc["med"]:+.5f})')
        if beat:
            sfx[rg].add('NOT-BEST')
        p(f'    짝 Δ 로 처치를 이기는 셀(≥{BEST_K}/32): {beat or "없음"}')
    p('  ⛔수준표를 우열 근거로 쓰지 않는다 — 판정은 짝 Δ (S7 에서 실제로 갈렸다)')
    emit('I12')

    # alpha
    p('\n### I13 α_wave (SECONDARY, 판정 미사용) + /CONTROL-NULL  [A21]')
    ma = per_ep(good, 'alpha')
    for rg in ('crowded', 'sparse'):
        lv = {c: float(np.median([ma[(c, e)] for e in eps
                                  if regime(e) == rg and (c, e) in ma]))
              for c in (BASE, TREAT)}
        c2 = C(ma, eps, TREAT, BASE, rg)
        p(L(c2, f'  α_wave({rg})') + f' | 수준 {lv[BASE]:.4f}→{lv[TREAT]:.4f} '
          f'(널 대비 기술 {100 * (1 - lv[BASE]):.0f}%→{100 * (1 - lv[TREAT]):.0f}%)')
        if c2['imp'] < BEST_K:
            sfx[rg].add('CONTROL-NULL')
        if rg in ALPHA_DELTA:
            ni = c2['med'] < ALPHA_DELTA[rg]
            p(f'    희소 비열등(δ={ALPHA_DELTA[rg]}): {"성립" if ni else "미성립"}')
        if np.sign(lv[TREAT] - lv[BASE]) != np.sign(c2['med']):
            p('    ⛔CONVENTION-CONFLICT(alpha) — 수준과 짝 Δ 부호가 다르다')
    p('  ⛔"제어가 좋아진다" 금지. 혼잡은 δ 가 수준의 12% 라 **비열등성 주장 금지**.')
    p('  ⛔제어점 x≈4 m ≠ 관측 중심 x≈20.8 m. ⛔α MDE 사전 동결본 없음(δ 는 S7 산포에서 산출)')
    emit('I13')

    # ── 판정 ──────────────────────────────────────────────────────────────────
    p('\n' + '=' * 94)
    p('### I7 판정')
    harmful = any(prim[rg]['med'] > 0 and prim[rg]['p'] <= ALPHA for rg in prim)
    dose = {k: C(mn, eps, k, BASE, 'crowded') for k in ('AUTO-L1-3e4', 'AUTO-L1-0')}
    dadj = holm([dose['AUTO-L1-3e4']['p'], dose['AUTO-L1-0']['p']])
    dose_pass = any(dose[k]['med'] < 0 and dadj[i] <= ALPHA
                    for i, k in enumerate(('AUTO-L1-3e4', 'AUTO-L1-0')))
    if harmful:
        v = 'HARMFUL'
    elif iut['crowded'] and iut['sparse']:
        v = 'FIELD-NRMSE-CONFIRMED' + ('' if not worse_shell['crowded'] + worse_shell['sparse']
                                       else ' (⚠악화 셸 있음)')
    elif hol['crowded']:
        v = 'CROWDED-ONLY'
    elif hol['sparse']:
        v = 'SPARSE-ONLY'
    elif dose_pass:
        v = 'SECONDARY-ONLY'
    else:
        v = 'NULL'
    allsfx = sorted(sfx['crowded'] | sfx['sparse'])
    p(f'  ★판정 = **{v}**' + (f'  /{"  /".join(allsfx)}' if allsfx else ''))
    p(f'     혼잡 IUT {"통과" if iut["crowded"] else "미달"} / Holm {"통과" if hol["crowded"] else "미달"}'
      f'  ·  희소 IUT {"통과" if iut["sparse"] else "미달"} / Holm {"통과" if hol["sparse"] else "미달"}')
    p(f'     접미 출처: 혼잡 {sorted(sfx["crowded"])} · 희소 {sorted(sfx["sparse"])}')
    if v == 'SECONDARY-ONLY':
        p('  ⛔허용 문장은 오직 "사전 지정 수준 미달, 탐색적 수준에서만 관측 — 재사전등록 필요"  [A10]')
    p('  ⛔"배포 확증" 무한정 표현 금지 — 라벨은 **지표 범위(field nRMSE)** 를 담는다  [A3]')
    p('  ⛔"출하 yaml 한 줄" 금지 — 기본값은 estimation.yaml:12 **와** run.py:26 두 곳 + scripts 7곳  [A6]')
    p('  ⛔이 배치의 양성으로 **실측 경로**에 대해 아무 말도 하지 않는다  [A6]')
    p(f'  ⚠HARMFUL 은 완결성 목적 라벨이다 (사전확률 <1e-6)  [A16]')
    emit('I7')

    # ── SECONDARY ─────────────────────────────────────────────────────────────
    p('\n### I14 λ 용량-반응 (출하 팔, 자체 Holm-2, 탐색적)  [A10]')
    for i, k in enumerate(('AUTO-L1-3e4', 'AUTO-L1-0')):
        p(L(dose[k], f'  {k} − AUTO (혼잡)') + f' | Holm-2 {dadj[i]:.5f}')
        p(L(C(mn, eps, k, BASE, 'sparse'), f'  {k} − AUTO (희소)'))
    lvl = {c: float(np.median([mn[(c, e)] for e in eps if regime(e) == 'crowded']))
           for c in ('AUTO', 'AUTO-L1-3e4', TREAT, 'AUTO-L1-0')}
    p(f'  용량 곡선(혼잡 수준): 1e-3 {lvl["AUTO"]:.5f} → 3e-4 {lvl["AUTO-L1-3e4"]:.5f} → '
      f'1e-4 {lvl[TREAT]:.5f} → 0 {lvl["AUTO-L1-0"]:.5f}')
    p('  ⛔"λ 는 낮을수록 좋다" 금지 (S6 비단조). 단조성은 보고용이며 판정 규칙이 아니다.')
    emit('I14')

    p('\n### I15 2-loci 양방향 분리 (탐색적)  [A9]')
    for k, lab in (('AUTO-L1-1e4-P1ONLY', 'Phase-1 만 λ=1e-4 (추적 출하값)'),
                   ('AUTO-L1-1e4-TRKONLY', '추적만 λ=1e-4 (Phase-1 출하값)')):
        for rg in ('crowded', 'sparse'):
            p(L(C(mn, eps, k, BASE, rg), f'  {lab[:26]} ({rg})'))
    both = C(mn, eps, TREAT, BASE, 'crowded')
    p1 = C(mn, eps, 'AUTO-L1-1e4-P1ONLY', BASE, 'crowded')
    tk = C(mn, eps, 'AUTO-L1-1e4-TRKONLY', BASE, 'crowded')
    sep = abs(p1['med'] - tk['med'])
    if sep < M['crowded']['mde_sign80']:
        sfx['crowded'].add('TWO-LOCI')
        p(f'  ⚠두 방향 차 {sep:.5f} < mde_sign80 {M["crowded"]["mde_sign80"]:.5f} ⟹ **탐지 한계 아래**'
          f' — `/TWO-LOCI`, "어디서 일하는지 미해결"')
    p(f'  (참고) 합계 {both["med"]:+.5f} · P1ONLY {p1["med"]:+.5f} · TRKONLY {tk["med"]:+.5f}'
      ' — ⛔세 수는 합쳐지지 않는다(에피소드별 분해, A15/A18)')
    p('  ⛔"몫/기여" 단어 금지 — 허용 = "한쪽을 출하값에 고정했을 때의 조건부 차이"')
    emit('I15')

    p('\n### I16 tracking_lr 용량-반응 (`0_5` 셸)  [A14]')
    m05 = per_ep(good, 'shell:0_5')
    c0 = C(m05, eps, 'AUTO-NT', BASE, 'crowded')
    cq = C(m05, eps, 'AUTO-TRK-QTR', BASE, 'crowded')
    p(L(c0, '  lr=0 (AUTO-NT) − AUTO'))
    p(L(cq, '  lr=0.25× (AUTO-TRK-QTR) − AUTO'))
    common = [e for e in c0['eps'] if e in cq['eps']]
    a0 = np.array([abs(c0['d'][c0['eps'].index(e)]) for e in common])
    aq = np.array([abs(cq['d'][cq['eps'].index(e)]) for e in common])
    mono = int((a0 >= aq).sum())
    sw = int((aq <= 0.25 * a0).sum())
    if mono >= BEST_K and c0['wor'] >= HARM_K and cq['wor'] >= HARM_K:
        dv = 'DOSE-MONOTONE'
    elif sw >= BEST_K:
        dv = 'DOSE-SWITCH'
    else:
        dv = 'DOSE-UNDECIDED'
    p(f'  |Δ(0)| ≥ |Δ(0.25×)|: {mono}/{len(common)} · |Δ(0.25×)| ≤ 0.25|Δ(0)|: {sw}/{len(common)}'
      f'  ⟹ **{dv}**')
    p('  ⛔기제 이름이 아니다. ⛔`DOSE-*` 를 배포 판단에 쓰지 않는다.')
    p('  ✅`AUTO-NT − AUTO`(nRMSE)는 S3·S5·S6·S7 에 이은 **일곱 번째 재현**:')
    for rg in ('crowded', 'sparse'):
        p(L(C(mn, eps, 'AUTO-NT', BASE, rg), f'    AUTO-NT − AUTO ({rg})'))
    emit('I16')

    p('\n### I17 프레임 실패 꼬리  [A20]')
    for k in ('nRMSE_p95', 'nRMSE_max', 'frac_frames_nRMSE_gt_0p5'):
        mm = per_ep(good, k)
        for rg in ('crowded', 'sparse'):
            c2 = C(mm, eps, TREAT, BASE, rg)
            cn = C(mm, eps, NULL_CELL, BASE, rg)
            flag = ''
            if k == 'nRMSE_max' and c2['wor'] >= HARM_K and cn['wor'] < HARM_K:
                sfx[rg].add('TAIL-FRAME')
                flag = '  ⛔TAIL-FRAME'
            p(L(c2, f'  {k} ({rg})') + f' | 널 악화 {cn["wor"]}/{cn["n"]}' + flag)
    emit('I17')

    p('\n### I18 기제 + ΔN + 계산량 구조  [A7/A16]')
    for k in ('window_resid', 'decorr_mean', 'shrink_mean', 'N', 'init_iters'):
        mm = per_ep(good, k)
        if not mm:
            p(f'  {k}: 미기록')
            continue
        for rg in ('crowded',):
            p(L(C(mm, eps, TREAT, BASE, rg), f'  {k} ({rg})'))
    dN = C(per_ep(good, 'N'), eps, TREAT, BASE, 'crowded')
    p(f'  ΔN: 중앙 {dN["med"]:+.2f}, 동점 {dN["ties"]}/{dN["n"]}, 최대|ΔN| {np.abs(dN["d"]).max():.0f}')
    p('  ⛔ΔN 중앙≠0 이면 "정규화 효과" 대신 "정규화+용량 묶음". ⛔`window_resid` 는 무가중.')
    p('  허용(구조): 처치는 그래디언트 스텝 수를 바꾸지 않는다 — analytic 900 / learned 500 /'
      ' 추적 10·프레임 고정, 가변항은 grow 리핏 +50 과 N. ⛔벽시계·`elapsed_ms` 인용 금지.')
    emit('I18')

    p('\n### I19 라우트 불변  [A19]')
    rt = {(r['cell'], r['episode'], r['seed']): (r.get('auto_selected'), r.get('ensemble_k'))
          for r in good}
    sd = sorted({r['seed'] for r in good if r['cell'] == BASE})
    same = sum(1 for e in eps for s in sd
               if (TREAT, e, s) in rt and rt[(BASE, e, s)] == rt[(TREAT, e, s)])
    p(f'  AUTO vs {TREAT} 라우트·ensemble_k 동일: {same}/{len(eps) * len(sd)}')
    p(f'  분포: ' + str({rg: dict(Counter((r.get('auto_selected'), r.get('ensemble_k'))
                                          for r in good if r['regime'] == rg))
                        for rg in ('crowded', 'sparse')}))
    p('  ⟹ 전부 동일하면 대조는 **라우트 내부**("라우트 불변"). 하나라도 다르면 "(레짐 ∧ 라우트)" 로 표기')
    emit('I19')

    p('\n### I20 수준 대 짝Δ CONVENTION-CONFLICT')
    for rg in ('crowded', 'sparse'):
        lt = float(np.median([mn[(TREAT, e)] for e in eps if regime(e) == rg]))
        lb = float(np.median([mn[(BASE, e)] for e in eps if regime(e) == rg]))
        if np.sign(lt - lb) != np.sign(prim[rg]['med']):
            p(f'  ⛔{rg}: 수준차 {lt - lb:+.5f} vs 짝Δ {prim[rg]["med"]:+.5f} — 판정은 짝 Δ')
        else:
            p(f'  {rg}: 수준차 {lt - lb:+.5f} · 짝Δ {prim[rg]["med"]:+.5f} (부호 일치)')
    emit('I20')

    p('\n### I21 시드별 + mean-of-3')
    for rg in ('crowded', 'sparse'):
        line = []
        for s in sd:
            m1 = {(r['cell'], r['episode']): r['nRMSE'] for r in good if r['seed'] == s}
            line.append(f'{s}: {C(m1, eps, TREAT, BASE, rg)["imp"]}/32')
        mmean = defaultdict(list)
        for r in good:
            mmean[(r['cell'], r['episode'])].append(r['nRMSE'])
        mmean = {k: float(np.mean(v)) for k, v in mmean.items()}
        p(f'  {rg}: ' + '  '.join(line) +
          f'  | mean-of-3: {C(mmean, eps, TREAT, BASE, rg)["imp"]}/32')
    emit('I21')

    p('\n### I23 S7 대비 수준 (⛔짝 아님 · 풀링/p결합 금지)')
    s7 = os.path.join(HERE, '..', 'S7_confirm_n32_2026-08-10', 'results.jsonl')
    if os.path.exists(s7):
        r7 = [json.loads(l) for l in open(s7) if l.strip()]
        r7 = [x for x in r7 if 'error' not in x]
        m7 = defaultdict(list)
        for x in r7:
            m7[(x['cell'], x['episode'])].append(x['nRMSE'])
        m7 = {k: float(np.median(v)) for k, v in m7.items()}
        for rg, pref in (('crowded', 'data_s7_crowded'), ('sparse', 'data_s7_sparse')):
            for c in (BASE, TREAT):
                v7 = [v for (cc, e), v in m7.items() if cc == c and e.startswith(pref)]
                v8 = [mn[(c, e)] for e in eps if regime(e) == rg and (c, e) in mn]
                p(f'  {rg:8s} {c:14s} S7 {np.median(v7):.5f} → S8 {np.median(v8):.5f}')
        p('  ⚠블록 난이도가 크게 다르면 사전 검정력표는 이 블록에 대해 무효다')
    emit('I23')

    p('\n### I5 실행 전 게이트 재수록')
    g = os.path.join(HERE, 'gate.txt')
    if os.path.exists(g):
        for line in open(g).read().strip().splitlines()[-8:]:
            p(f'    {line}')
        emit('I5')
    else:
        p('  ⛔ gate.txt 없음')

    p('\n' + '=' * 94)
    miss = [x for x in REQUIRED if x.split()[0] not in EMITTED]
    p(f'### §8 자체 채점: {len(EMITTED)}/{len(REQUIRED)} 항')
    if miss:
        p('⛔ §8 위반 — 무결성 주장 인용 불가. 누락:')
        for m in miss:
            p(f'   - {m}')
    else:
        p('✅ §8 전항 출력')
    p('=' * 94)
    open(os.path.join(HERE, 'analysis.txt'), 'w').write('\n'.join(OUT) + '\n')


if __name__ == '__main__':
    main()
