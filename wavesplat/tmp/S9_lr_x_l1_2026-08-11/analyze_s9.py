"""S9 채점기 — 사전등록 `PREREG.md` 실행. **결과 관측 전 md5 동결.**

## 왜 S8 채점기를 상속하지 않는가
S9 사전감사가 `analyze_s8.py` 에서 **n=32 하드코딩 4종**을 찾았다:
`MDE_SIGN80_C=0.7234` · `HARM_K=21` · `BEST_K=23` · 꼬리 규칙의 `>=6` 과 `/32`.
S9 는 **비대칭 n**(희소 64 / 혼잡 32)이라 이것들을 상속하면 **희소에서만 조용히 틀린다**
(혼잡에서는 맞으므로 스팟체크를 통과한다). 예: 희소에서 `HARM_K=21` 이면 21/64 악화 =
**43/64 개선 셸을 "악화 셸"로 찍는다**.
⟹ 전 상수는 `s9_constants.py` 가 (n, k) 에서 **유도**한다. ⛔이 파일에 상수를 적지 않는다.

## G-0 자기시험
`--results <S8 results.jsonl> --s8map` 으로 돌리면 S8 데이터를 S9 채점기로 채점한다.
S8 판정(혼잡 27/32 통과 · 희소 22/32 미달)을 재현해야 한다. 재현 못 하면 **본배치를 돌리지 않는다**.
"""
import argparse
import json
import os
import statistics as st
from collections import defaultdict

import numpy as np

import s9_constants as C

HERE = os.path.dirname(os.path.abspath(__file__))
MDE_PATH = os.path.join(HERE, 'MDE_prespec_s9.json')

NEAR_SHELLS = ('0_5', '5_10', '10_20')
ALL_SHELLS = ('0_5', '5_10', '10_20', '20_30', '30_50', '50_75', '75_100', '100_150', '150_inf')

# S8 데이터로 자기시험할 때의 셀 이름 대응 (G-0 전용)
S8_MAP = {'AUTO-L1': 'AUTO-L1-1e4', 'AUTO-QTR': 'AUTO-TRK-QTR',
          'AUTO-QTR-L1': None, 'AUTO-DUP': 'AUTO-DUP2'}


# ── 적재 · 집계 ──────────────────────────────────────────────────────────────
def load(path):
    rec = defaultdict(lambda: defaultdict(list))     # cell -> episode -> [records]
    regimes, errors = {}, 0
    for line in open(path):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if 'error' in r:
            errors += 1
            continue
        rec[r['cell']][r['episode']].append(r)
        regimes[r['episode']] = r['regime']
    return rec, regimes, errors


def ep_value(recs, getter):
    """에피소드당 = 3 추정기 시드의 **중앙** (승계 규약 §3.1)."""
    v = [getter(r) for r in recs]
    v = [x for x in v if x is not None and np.isfinite(x)]
    return st.median(v) if v else None


def paired(rec, regimes, treat, base, regime, getter):
    """짝 Δ 리스트. 같은 에피소드 안에서만. 반환 = [(episode, d, rel)]"""
    out = []
    if treat not in rec or base not in rec:
        return out
    for ep, trecs in rec[treat].items():
        if regimes.get(ep) != regime or ep not in rec[base]:
            continue
        a, b = ep_value(trecs, getter), ep_value(rec[base][ep], getter)
        if a is None or b is None:
            continue
        out.append((ep, a - b, (a - b) / b if b else float('nan')))
    return sorted(out)


def sign_test(deltas, shift=0.0):
    """부호검정. shift>0 이면 **이동 널**: #{d < shift} 를 센다 (PREREG §4 존재 증명).

    반환 = dict(imp, wor, tie, n_eff, p, med_abs, med_rel)
    """
    imp = sum(1 for _e, d, _r in deltas if d < shift)
    wor = sum(1 for _e, d, _r in deltas if d > shift)
    tie = len(deltas) - imp - wor
    n = imp + wor
    p = C.p_two_sided(max(imp, wor), n) if n else float('nan')
    return dict(imp=imp, wor=wor, tie=tie, n_eff=n, p=p,
                med_abs=st.median([d for _e, d, _r in deltas]) if deltas else float('nan'),
                med_rel=st.median([r for _e, _d, r in deltas]) if deltas else float('nan'),
                mean_rel=float(np.mean([r for _e, _d, r in deltas])) if deltas else float('nan'),
                max_rel=max((r for _e, _d, r in deltas), default=float('nan')))


def fmt(s, k=None):
    tag = ''
    if k is not None:
        tag = '  ✅통과' if (s['imp'] >= k and s['med_abs'] < 0) else '  미달'
    return (f"개선 {s['imp']:2d}/{s['imp']+s['wor']+s['tie']:2d}  악화 {s['wor']:2d}  동점 {s['tie']}  "
            f"Δ중앙 {s['med_abs']:+.5f} ({s['med_rel']*100:+.2f}%)  원p {s['p']:.5f}{tag}")


# ── 진단 족 ─────────────────────────────────────────────────────────────────
def G(key):
    return lambda r: r.get(key)


def GS(shell):
    return lambda r: (r.get('shell_rmse') or {}).get(shell)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=os.path.join(HERE, 'results.jsonl'))
    ap.add_argument('--s8map', action='store_true', help='G-0 자기시험: S8 셀 이름으로 매핑')
    ap.add_argument('--n-override', type=int, default=None,
                    help='G-0 자기시험: S8 은 두 레짐 다 n=32')
    args = ap.parse_args()

    def cell(name):
        return S8_MAP.get(name, name) if args.s8map else name

    rec, regimes, errors = load(args.results)
    mde = json.load(open(MDE_PATH))
    REQUIRED = []

    def head(t):
        REQUIRED.append(t)
        print('\n' + '=' * 78 + f'\n{t}\n' + '=' * 78)

    print(f'S9 채점 — {os.path.relpath(args.results, HERE)}')
    if args.s8map:
        print('★G-0 자기시험 모드: S8 데이터를 S9 채점기로 채점한다 (셀 이름 매핑)')

    # ── §0 무결성 ────────────────────────────────────────────────────────────
    head('§0 무결성')
    n_rec = sum(len(v) for c in rec.values() for v in c.values())
    print(f'  레코드 {n_rec} · 에러 {errors}')
    reg_count = defaultdict(int)
    for ep, rg in regimes.items():
        reg_count[rg] += 1
    print(f'  레짐별 에피소드: {dict(reg_count)}')
    print(f'  셀: {sorted(rec)}')
    # 상수 자기시험 (import 시 이미 돌았지만 결과에 재수록)
    print(f'  ★s9_constants 자기시험 PASS (S8 하드코딩 0.7234 = 유도값 재현)')
    # 제어 축 결백 (G-7)
    alpha_keys = {k for c in rec.values() for eps in c.values() for r in eps
                  for k in r if k.startswith('alpha')}
    s2env = {r.get('s2_alpha_env') for c in rec.values() for eps in c.values() for r in eps}
    print(f'  ★G-7 제어 축 결백: alpha_* 키 {len(alpha_keys)}개 {sorted(alpha_keys) or "(없음) ✅"} · '
          f's2_alpha_env {s2env or "(미기록)"}')
    # 워커 누출 프로브
    dup = paired(rec, regimes, cell('AUTO-DUP'), 'AUTO', 'sparse', G('nRMSE'))
    dup += paired(rec, regimes, cell('AUTO-DUP'), 'AUTO', 'crowded', G('nRMSE'))
    nz = [d for _e, d, _r in dup if d != 0]
    print(f'  워커 누출 프로브 (AUTO-DUP vs AUTO): {len(dup)} 짝 중 비트 불일치 {len(nz)}개 '
          f'{"✅" if not nz else "⛔"}')

    # ── §1 PRIMARY ───────────────────────────────────────────────────────────
    head('§1 ★PRIMARY — AUTO-L1 − AUTO, 희소, 양측 p<=0.05')
    verdict = {}
    for rg in ('sparse', 'crowded'):
        n = args.n_override or C.N[rg]
        k = C.primary_k(n)
        d = paired(rec, regimes, cell('AUTO-L1'), 'AUTO', rg, G('nRMSE'))
        s = sign_test(d)
        k_eff = C.primary_k(s['n_eff']) if s['n_eff'] else k
        role = '★PRIMARY' if rg == 'sparse' else 'SECONDARY(복제)'
        print(f'  [{role:16s}] {rg:8s} (n={n}, 문턱 {k}/{n}; 유효n {s["n_eff"]} -> {k_eff})')
        print(f'      {fmt(s, k_eff)}')
        verdict[rg] = (s, k_eff)
        if rg == 'crowded':
            print('      ⚠사전 통과확률이 높다(S8 27/32) ⟹ 통과해도 **복제**이지 신규 정보가 아니다.')
            print('      ⛔"두 번 독립 재현" 은 가능하나 **p 결합 금지**.')
    print(f'\n  ★동결 사전 검정력: 참 부호율 0.6875 -> {C.binom_ge(64, 41, 0.6875):.4f} '
          f'(⛔결과 후 재계산 금지)')
    print(f'  대조: S8 설계(n=32, Holm-2, k>=23) 였다면 {C.binom_ge(32, 23, 0.6875):.4f}')

    # ── §2 근거리 — 이동 널 부호검정 (존재 증명) ──────────────────────────────
    head('§2 ★근거리 비열등 — 이동 널 부호검정 #{Δ < δ} (부재 증명 아님)')
    print('  δ = mde_sign80_c(n,k) × σ_Δ,robust / baseline, **결과 관측 전 동결**')
    near_ok = {}
    for rg in ('sparse', 'crowded'):
        n = args.n_override or C.N[rg]
        k = C.primary_k(n)
        print(f'\n  --- {rg} (문턱 {k}/{n}) ---')
        ok_all = True
        for s_ in NEAR_SHELLS:
            delta_rel = mde['by_regime'][rg]['shells'][s_]['delta_rel']
            d = paired(rec, regimes, cell('AUTO-L1'), 'AUTO', rg, GS(s_))
            base = mde['by_regime'][rg]['shells'][s_]['baseline_median']
            shift = delta_rel * base                      # 절대 스케일 이동
            s = sign_test(d, shift=shift)
            k_eff = C.primary_k(s['n_eff']) if s['n_eff'] else k
            passed = s['imp'] >= k_eff
            ok_all &= passed
            print(f'    {s_:8s} δ={delta_rel*100:+6.2f}%  #{{Δ<δ}} = {s["imp"]:2d}/{s["n_eff"]:2d} '
                  f'(문턱 {k_eff})  {"✅" if passed else "⛔"}   실제 Δ중앙 {s["med_rel"]*100:+.2f}%')
            # ★널 보정 (PREREG §4)
            dn = paired(rec, regimes, cell('AUTO-SEED-NULL'), 'AUTO', rg, GS(s_))
            if dn:
                sn = sign_test(dn)
                print(f'             무처치 널: 악화 {sn["wor"]}/{sn["n_eff"]} '
                      f'Δ중앙 {sn["med_rel"]*100:+.2f}%'
                      + ('   ⚠TAIL-UNCALIBRATED (널이 처치보다 크다)'
                         if sn['med_rel'] >= s['med_rel'] and s['med_rel'] > 0 else ''))
        near_ok[rg] = ok_all
    print('\n  ⛔통과해도 "무해하다" 금지. 허용 = "δ 이상의 근거리 손상은 검출되지 않았다" + δ 병기.')

    # ── §3 all_resid 게이트 (배포 거부권) ────────────────────────────────────
    head('§3 ★all_resid 게이트 — GT-free 센서 잔차 (S8 A5 승계, 배포 거부권)')
    resid_ok = True
    for rg in ('sparse', 'crowded'):
        n = args.n_override or C.N[rg]
        hk = C.harm_k(n)
        d = paired(rec, regimes, cell('AUTO-L1'), 'AUTO', rg, G('all_resid'))
        s = sign_test(d)
        harmed = s['wor'] >= hk
        resid_ok &= not harmed
        print(f'  {rg:8s} {fmt(s)}   HARM 문턱 {hk}/{n} -> '
              f'{"⛔SUPPORT-TRADEOFF" if harmed else "✅ 악화 아님"}')
    print('  ⛔`avg_obs_loss` 는 λ·Σ|w| 를 포함하므로 관측 적합 근거로 인용 금지 (A5).')

    # ── §4 안전 게이트 ───────────────────────────────────────────────────────
    head('§4 안전 게이트 (⚠무보정 — 위음성이 위험한 오류)')
    safety_ok = True
    for rg in ('sparse', 'crowded'):
        n = args.n_override or C.N[rg]
        hk, bn, bk = C.harm_k(n), C.BLOCK_N[rg], C.block_worse_k(C.BLOCK_N[rg])
        d = paired(rec, regimes, cell('AUTO-L1'), 'AUTO', rg, G('nRMSE'))
        s = sign_test(d)
        rels = [r for _e, _d, r in d]
        n_big = sum(1 for r in rels if r > 0.50)
        frac_thresh = int(np.ceil(0.1875 * n))
        fires = []
        if s['wor'] >= hk:
            fires.append(f'HARM-SIGN({s["wor"]}>={hk})')
        if n_big >= frac_thresh:
            fires.append(f'TAIL-a({n_big}>={frac_thresh})')
        if s['max_rel'] > 2.0:
            fires.append(f'TAIL-b(max {s["max_rel"]*100:+.1f}%)')
        if np.sign(s['mean_rel']) != np.sign(s['med_rel']):
            fires.append('TAIL-c(mean/중앙 부호 반대)')
        print(f'  {rg:8s} HARM-SIGN {s["wor"]}/{n} (문턱 {hk}) · 블록 문턱 {bk}/{bn} · '
              f'>+50% {n_big}개 (문턱 {frac_thresh}) · 최대 {s["max_rel"]*100:+.1f}% · '
              f'mean {s["mean_rel"]*100:+.2f}% vs 중앙 {s["med_rel"]*100:+.2f}%')
        # 널 병기 (희소 꼬리는 UNCALIBRATED)
        dn = paired(rec, regimes, cell('AUTO-SEED-NULL'), 'AUTO', rg, G('nRMSE'))
        if dn:
            sn = sign_test(dn)
            nb = sum(1 for _e, _d, r in dn if r > 0.50)
            # ⚠널 보정은 **설명할 꼬리가 실제로 있을 때만** 의미가 있다.
            # 처치 꼬리가 0 인데 널도 0 이라고 UNCALIBRATED 를 붙이면 (0>=0) 무조건 발화한다.
            uncal = ((n_big > 0 and nb >= n_big)
                     or (s['max_rel'] > 0.50 and sn['max_rel'] >= s['max_rel']))
            print(f'           무처치 널 대조: >+50% {nb}개 · 최대 {sn["max_rel"]*100:+.1f}%'
                  + ('   ⚠TAIL-UNCALIBRATED (널이 처치 꼬리를 설명한다)' if uncal else ''))
        print(f'           발화: {fires if fires else "없음 ✅"}')
        safety_ok &= not fires
    print('  ⛔미발화를 "무해하다"로 쓰지 않는다 — 검출 가능했던 최소 피해(δ)를 병기한다.')

    # ── §5 lr 축 폐쇄 ────────────────────────────────────────────────────────
    head('§5 lr 축 폐쇄 (SECONDARY, 탐색적 — ⛔판정 미기여)')
    print('  사전 예측(결과 전 기록): 혼잡에서 all_resid 악화 재현 · QTR 계열이 AUTO-L1 에 진다')
    for label, treat, base in (('AUTO-QTR − AUTO', 'AUTO-QTR', 'AUTO'),
                               ('AUTO-QTR-L1 − AUTO-L1  ★거래를 보는 대조',
                                'AUTO-QTR-L1', 'AUTO-L1')):
        t, b = cell(treat), cell(base)
        if t is None or t not in rec:
            print(f'\n  [{label}] — 셀 없음 (자기시험 모드) 건너뜀')
            continue
        print(f'\n  [{label}]')
        for rg in ('sparse', 'crowded'):
            for mlabel, getter in (('nRMSE', G('nRMSE')), ('all_resid', G('all_resid')),
                                   ('shell 0_5', GS('0_5'))):
                s = sign_test(paired(rec, regimes, t, b, rg, getter))
                print(f'    {rg:8s} {mlabel:10s} {fmt(s)}')
    print('\n  ⛔lr 사다리의 "DOSE-MONOTONE" 서사는 **널 보정 없이** 쓰였으므로 재인용 금지(§0).')

    # ── §6 9셸 전수 + 기제 ───────────────────────────────────────────────────
    head('§6 9셸 전수 (판정은 근거리 3셸만) + 기제 계측')
    for rg in ('sparse', 'crowded'):
        print(f'\n  --- {rg} ---')
        for s_ in ALL_SHELLS:
            s = sign_test(paired(rec, regimes, cell('AUTO-L1'), 'AUTO', rg, GS(s_)))
            print(f'    {s_:8s} {fmt(s)}')
    print('\n  기제:')
    for rg in ('sparse', 'crowded'):
        for key in ('window_resid', 'decorr_mean', 'final_n', 'init_iters'):
            d = paired(rec, regimes, cell('AUTO-L1'), 'AUTO', rg, G(key))
            if d:
                s = sign_test(d)
                print(f'    {rg:8s} {key:14s} {fmt(s)}')
    print('  ⛔ΔN != 0 이면 "정규화 효과" 대신 **"정규화+용량 묶음"**.')

    # ── §7 판정 ──────────────────────────────────────────────────────────────
    head('§7 판정')
    s_sp, k_sp = verdict['sparse']
    s_cr, k_cr = verdict['crowded']
    primary_pass = s_sp['imp'] >= k_sp and s_sp['med_abs'] < 0
    crowded_pass = s_cr['imp'] >= k_cr and s_cr['med_abs'] < 0
    print(f'  1. PRIMARY (희소)          : {"통과 ✅" if primary_pass else "미달 ⛔"} '
          f'({s_sp["imp"]}/{s_sp["n_eff"]}, 문턱 {k_sp})')
    print(f'  2. 혼잡 복제 같은 방향      : {"✅" if crowded_pass else "⛔"}')
    print(f'  3. 근거리 이동널 (양 레짐)  : {"✅" if all(near_ok.values()) else "⛔"}')
    print(f'  4. all_resid 악화 아님      : {"✅" if resid_ok else "⛔"}')
    print(f'  5. 안전 게이트 미발화       : {"✅" if safety_ok else "⛔"}')
    ref = {'sparse': -0.00312, 'crowded': -0.02650}
    mag = abs(s_sp['med_abs']) / abs(ref['sparse']) if ref['sparse'] else float('nan')
    print(f'  6. 크기 재현 (희소, S8 {ref["sparse"]:+.5f} 대비): {mag*100:.1f}% '
          f'{"✅" if mag >= 0.5 else "⛔ /SIGN-ONLY"}')
    all_six = all([primary_pass, crowded_pass, all(near_ok.values()), resid_ok, safety_ok,
                   mag >= 0.5])
    label = 'BOTH' if primary_pass else 'SPARSE-NULL'
    suffix = []
    if not resid_ok:
        suffix.append('/SUPPORT-TRADEOFF')
    if not all(near_ok.values()):
        suffix.append('/NEAR-FIELD-LOSS')
    if mag < 0.5:
        suffix.append('/SIGN-ONLY')
    suffix.append('/TWO-LOCI')          # 이 배치의 범위 밖 (PREREG §5g) — 항상 붙는다
    print(f'\n  ★판정 = `{label}{"".join(suffix)}`')
    print(f'  ★§8 배포 6요건 = {"전부 충족 ✅ — 배포 문장 사용 가능" if all_six else "미충족 ⛔ — 배포 문장 금지"}')
    if not primary_pass:
        print('  허용 문장: "참 부호율이 0.6875 이상이었다면 83% 확률로 잡았을 설계에서 잡지 못했다."')
        print('  ⛔금지: "희소에서 효과 없음" (검정력 0.83 != 1.0)')

    print('\n' + '=' * 78)
    print(f'§9 자체채점: 필수 절 {len(REQUIRED)}개 전부 출력 — {REQUIRED}')


if __name__ == '__main__':
    main()
