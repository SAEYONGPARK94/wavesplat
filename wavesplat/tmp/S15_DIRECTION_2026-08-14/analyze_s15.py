#!/usr/bin/env python3
"""S15_DIRECTION 채점기 — `PREREG.md` (md5 `FROZEN_md5.txt`) 를 그대로 구현.

    python3 tmp/S15_DIRECTION_2026-08-14/analyze_s15.py [results.jsonl]

⛔**결과를 본 뒤에 이 파일을 수정하지 않는다.** 수정이 필요하면 그 사실과 이유를 `RESULT.md` 에
적는다. (사전등록 §10)

출력 순서 = PREREG 절 순서: 무결성 -> PRIMARY -> 판정 -> SECONDARY -> 기술보고.
"""
import json
import os
import sys
from math import comb

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = os.path.join(HERE, 'results.jsonl')

# PREREG §4 축소 n 임계표 (단측 alpha=0.025 == 양측 alpha=0.05 정확 이항)
CRIT = {10: 9, 11: 10, 12: 10, 13: 11, 14: 12, 15: 12, 16: 13}
MIN_N = 10
ALPHA1, ALPHA2 = 0.025, 0.05
L3, L4 = 1e-3, 1e-4

PRIMARY_ARMS = ('ORA-WPHI-NT@L3', 'ORA-WPHI-0STEP@L3')
L4_ARMS = ('ORA-WPHI-NT@L4', 'ORA-WPHI-0STEP@L4')


def sf(k, n, p=0.5):
    return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def load(path):
    """jsonl 을 읽고 `(episode, arm, est_seed)` 로 **중복을 제거**한다 (last-write-wins).

    왜: 러너를 실수로 두 번 띄우면 같은 셀이 두 줄 들어가고, 그러면 median-of-3 이
    조용히 median-of-6 이 된다. 재개 로직은 append 전에만 작동하므로 채점기에서 한 번 더 막는다.
    제거된 줄 수는 §9 에 인쇄한다.
    """
    keep, errs, n_raw = {}, [], 0
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        n_raw += 1
        if 'error' in r:
            errs.append(r)
            continue
        keep[(r['episode'], r['arm'], r.get('est_seed'))] = r
    rows = list(keep.values())
    return rows, errs, n_raw - len(errs) - len(rows)


def med_by_cell(rows, metric):
    """(episode, arm) -> 추정기 시드 중앙값 (PREREG §8.1 median-of-3)."""
    acc = {}
    for r in rows:
        v = r.get(metric)
        if v is None or not np.isfinite(v):
            continue
        acc.setdefault((r['episode'], r['arm']), []).append(float(v))
    return ({k: float(np.median(v)) for k, v in acc.items()},
            {k: (len(v), float(np.max(v) - np.min(v))) for k, v in acc.items()})


def meta(rows):
    m = {}
    for r in rows:
        m[r['episode']] = dict(direction=r['direction'], stratum=r['stratum'],
                               bf=r['bf'], ncomp=r['ncomp'], ep_seed=r['ep_seed'],
                               pair=r.get('pair'))
    return m


def paired(cells, info, arm_a, arm_b, direction, stratum):
    """짝 Δ = (arm_a - arm_b), 같은 에피소드 안. PREREG §8.2."""
    out = []
    for ep, mi in sorted(info.items()):
        if mi['direction'] != direction or mi['stratum'] != stratum:
            continue
        a, b = cells.get((ep, arm_a)), cells.get((ep, arm_b))
        if a is None or b is None:
            continue
        out.append((ep, mi, a, b, a - b))
    return out


def sign_test(deltas, one_sided=True, label=''):
    """정확 부호검정. 동점 제외 후 §4 축소표에서 임계값을 읽는다."""
    d = [x for x in deltas if x != 0.0]
    ties = len(deltas) - len(d)
    n = len(d)
    pos = sum(1 for x in d if x > 0)
    if n < MIN_N:
        return dict(n=n, ties=ties, pos=pos, crit=None, p=None,
                    verdict='UNDECIDED-TIES', label=label)
    crit = CRIT[min(n, 16)]
    p1 = sf(pos, n)
    p = p1 if one_sided else min(1.0, 2 * min(p1, sf(n - pos, n)))
    return dict(n=n, ties=ties, pos=pos, crit=crit, p=p,
                pass_=bool(pos >= crit), label=label)


def sizes(pairs):
    """PREREG §8.6 크기 3규약 의무 병기 (g = arm_a, b = arm_b)."""
    g = np.array([p[2] for p in pairs], float)
    b = np.array([p[3] for p in pairs], float)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel = (g - b) / b
    return dict(med_rel=100 * float(np.median(rel)),
                rel_med=100 * float((np.median(g) - np.median(b)) / np.median(b)),
                rel_mean=100 * float((np.mean(g) - np.mean(b)) / np.mean(b)),
                abs_med=float(np.median(g - b)),
                med_g=float(np.median(g)), med_b=float(np.median(b)))


def pfmt(st):
    """p 를 안전하게 인쇄한다. n<10 이면 p 가 None 이라 포맷하면 죽는다 (스모크에서 실측)."""
    return f'{st["p"]:.3e}' if st.get('p') is not None else 'n<10 UNDECIDED-TIES'


def show(tag, pairs, st, sz):
    if st['crit'] is None:
        print(f'  {tag:<46s} n={st["n"]} ties={st["ties"]}  -> {st["verdict"]}')
        return
    print(f'  {tag:<46s} {st["pos"]:2d}/{st["n"]:<2d} (문턱 {st["crit"]}) '
          f'p={st["p"]:.3e} ties={st["ties"]}  '
          f'{"PASS" if st["pass_"] else "fail"}')
    print(f'  {"":46s} 크기 3규약: med(rel)={sz["med_rel"]:+.1f}%  '
          f'rel(med)={sz["rel_med"]:+.1f}%  rel(mean)={sz["rel_mean"]:+.1f}%  '
          f'| abs med Δ={sz["abs_med"]:+.6f}')


def holm(tests, alpha=0.05):
    live = [(k, v) for k, v in tests.items() if v.get('p') is not None]
    live.sort(key=lambda kv: kv[1]['p'])
    m = len(live)
    out, prev = {}, False
    for i, (k, v) in enumerate(live):
        thr = alpha / (m - i)
        rej = (v['p'] <= thr) and not prev
        if not rej:
            prev = True
        out[k] = dict(p=v['p'], holm_thr=thr, reject=rej)
    return out


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    rows, errs, n_dup = load(path)
    info = meta(rows)
    print('=' * 92)
    print(f'S15_DIRECTION 채점 — {path}')
    print('=' * 92)

    # ── §9 무결성 ───────────────────────────────────────────────────────────
    print('\n[§9 무결성]')
    print(f'  행 {len(rows)} (에러 {len(errs)}, 중복 제거 {n_dup})')
    if n_dup:
        print('  ⚠중복 행이 있었다 — 러너가 두 번 돈 흔적. 원인을 RESULT.md 에 적을 것.')
    if errs:
        for e in errs[:10]:
            print(f'    ! {e.get("arm")} {e.get("episode")} {e.get("error")}')
    digests = {r.get('code_md5') for r in rows}
    imports = {r.get('import_md5') for r in rows}
    print(f'  code_md5 {len(digests)}종 {sorted(d[:8] for d in digests if d)}')
    print(f'  import_md5 {len(imports)}종 {sorted(d[:8] for d in imports if d)}')
    if len(digests) != 1 or len(imports) != 1:
        print('  ⚠지문이 하나가 아니다 — 배치 중 코드가 바뀌었다. 원인을 RESULT.md 에 적을 것.')
    lam = {}
    for r in rows:
        lam.setdefault(r['arm'], set()).add(
            (r.get('lambda_l1'),
             (r.get('effective_config') or {}).get('lambda_l1')))
    print('  팔별 (의도 lambda_l1, 되읽은 lambda_l1):')
    for a in sorted(lam):
        print(f'    {a:<19s} {sorted(str(x) for x in lam[a])}')
    # 결정론 프로브 (T-5)
    cells_n, spread_n = med_by_cell(rows, 'avg_nRMSE')
    dup = [(ep, abs(cells_n[(ep, 'AUTO-NT@L4')] - cells_n[(ep, 'AUTO-NT-DUP@L4')]))
           for ep in info
           if (ep, 'AUTO-NT@L4') in cells_n and (ep, 'AUTO-NT-DUP@L4') in cells_n]
    if dup:
        print(f'  T-5 AUTO-NT vs DUP: {len(dup)} 짝, worst |Δ| = '
              f'{max(d for _, d in dup):.3e} (0 이어야 환경 게이트 누출 없음)')
    z = [r for r in rows if r['arm'] == 'ZERO' and r.get('zero_gap') is not None]
    if z:
        print(f'  G-9 ZERO 채점기 vs 닫힌형 worst |Δ| = '
              f'{max(float(r["zero_gap"]) for r in z):.3e}')
    o0 = [r for r in rows if r['arm'].startswith('ORA-WPHI-0STEP')]
    if o0:
        print(f'  G-4 0STEP avg_nRMSE worst = '
              f'{max(abs(float(r["avg_nRMSE"])) for r in o0):.3e} (파라미터 전달 무결성)')
    sh = [r['shell_native_max_abs'] for r in rows
          if r.get('shell_native_max_abs') is not None]
    if sh:
        print(f'  G-8b 셸 헬퍼 vs run.py 원본 worst |Δ| = {max(sh):.3e}')

    cells, spread = med_by_cell(rows, 'avg_obs_loss')

    # 시드 산포 (PREREG §8.1 의 퇴화 확인 + §8.7 MDE)
    print('\n[시드 산포 / MDE — 배치 내 산출, 사전 동결 아님]')
    for a in sorted({k[1] for k in spread}):
        s = [v[1] for k, v in spread.items() if k[1] == a]
        ns = [v[0] for k, v in spread.items() if k[1] == a]
        if s:
            print(f'  {a:<19s} 시드수 {min(ns)}-{max(ns)}  obs_loss 시드범위: '
                  f'median {np.median(s):.3e}  max {np.max(s):.3e}'
                  + ('   ⚠산포 0 = median-of-3 퇴화' if np.max(s) == 0 else ''))
    z0 = [v[1] for k, v in spread.items() if k[1] == 'ORA-WPHI-0STEP@L3']
    mde = (float(np.median(z0)) if z0 else float('nan'))
    print(f'  MDE 대용(0STEP@L3 시드범위 중앙) = {mde:.3e}'
          + ('   ⛔산포 0 ⟹ |Δ|/MDE 인용 금지 (PREREG §8.7)' if mde == 0 else ''))

    # ── §4 PRIMARY ──────────────────────────────────────────────────────────
    print('\n[§4 PRIMARY — dir180 · 혼잡 · lambda=1e-3 · avg_obs_loss]')
    P = paired(cells, info, PRIMARY_ARMS[0], PRIMARY_ARMS[1], 180, 'crowded')
    st_p = sign_test([p[4] for p in P], one_sided=True, label='PRIMARY')
    sz_p = sizes(P) if P else None
    if P:
        show('PRIMARY dir180 NT>0STEP (벽이 있는가)', P, st_p, sz_p)
    else:
        print('  자료 없음')

    print('\n[N-1 필요조건 — dir0 · 혼잡 · lambda=1e-3]')
    N1 = paired(cells, info, PRIMARY_ARMS[0], PRIMARY_ARMS[1], 0, 'crowded')
    st_n1 = sign_test([p[4] for p in N1], one_sided=True, label='N-1')
    if N1:
        show('N-1 dir0 NT>0STEP (벽이 dir0 에 있는가)', N1, st_n1, sizes(N1))
    else:
        print('  자료 없음')

    # ── §5 판정 ─────────────────────────────────────────────────────────────
    print('\n[§5 판정]')
    if not P or not N1:
        verdict = 'INCOMPLETE'
    elif st_p.get('crit') is None or st_n1.get('crit') is None:
        verdict = 'UNDECIDED-TIES'
    else:
        p_ok, n_ok = st_p['pass_'], st_n1['pass_']
        verdict = ('WALL-DIRECTION-INVARIANT' if (p_ok and n_ok) else
                   'WALL-IS-GEOMETRY' if (not p_ok and n_ok) else
                   'ANOMALY' if (p_ok and not n_ok) else 'INCONCLUSIVE-NO-WALL')
    print(f'  ★판정 = **{verdict}**')
    print({'WALL-DIRECTION-INVARIANT':
           '  벽이 개구 상대 전파방향에 불변. 척추는 이 반증을 생존한다.\n'
           '  ⛔증명 아님 — 조우주파수 반증가설은 여전히 UNTESTED (PREREG §0.1/§5.1).',
           'WALL-IS-GEOMETRY':
           '  ★사망조건 발동 — "위상 벽은 following-seas 기하의 인공물". 새 척추를 철회한다.\n'
           '  ⛔"효과 0" 으로 읽지 말 것: 배제되는 것은 "S4 만큼 큰 벽이 dir180 에도 남는다".',
           'ANOMALY': '  조사 대상. 어떤 헤드라인도 쓰지 않는다.',
           'INCONCLUSIVE-NO-WALL': '  G-0 이 통과했는데도 벽이 없다 ⟹ 코퍼스/집계 문제. 판정 없음.',
           'UNDECIDED-TIES': '  동점 제외 후 n<10. 판정 없음.',
           'INCOMPLETE': '  자료 부족.'}[verdict])

    # ── §6 SECONDARY ────────────────────────────────────────────────────────
    print('\n[§6 SECONDARY — Holm-4, alpha=0.05]')
    tests = {}

    # S-1 방향 대조 (양측): gap180 - gap0, 같은 (셀, ep_seed)
    gp = {}
    for direction, lst in ((180, P), (0, N1)):
        for ep, mi, a, b, d in lst:
            gp.setdefault((mi['bf'], mi['ncomp'], mi['ep_seed']), {})[direction] = d
    s1d = [v[180] - v[0] for v in gp.values() if 0 in v and 180 in v]
    tests['S-1 방향대조(양측) gap180-gap0'] = sign_test(s1d, one_sided=False, label='S-1')
    if s1d:
        st = tests['S-1 방향대조(양측) gap180-gap0']
        rel = [100 * (v[180] - v[0]) / v[0] for v in gp.values()
               if 0 in v and 180 in v and v[0] != 0]
        print(f'  S-1 방향대조: {st["pos"]}/{st["n"]} 양수 (문턱 {st.get("crit")}) '
              f'p={pfmt(st)}  gap 변화 med(rel)={np.median(rel):+.1f}% '
              f'abs med={np.median(s1d):+.6f}')

    # S-2 lambda=1e-4 반복 (단측), S-3 희소 (단측)
    for tag, arms, direction, stratum in (
            ('S-2 L4 dir180 혼잡', L4_ARMS, 180, 'crowded'),
            ('S-2 L4 dir0   혼잡', L4_ARMS, 0, 'crowded'),
            ('S-3 L3 dir180 희소', PRIMARY_ARMS, 180, 'sparse'),
            ('S-3 L3 dir0   희소', PRIMARY_ARMS, 0, 'sparse'),
            ('S-3 L4 dir180 희소', L4_ARMS, 180, 'sparse'),
            ('S-3 L4 dir0   희소', L4_ARMS, 0, 'sparse')):
        pr = paired(cells, info, arms[0], arms[1], direction, stratum)
        if not pr:
            continue
        st = sign_test([p[4] for p in pr], one_sided=True, label=tag)
        tests[tag] = st
        show(tag, pr, st, sizes(pr))

    # S-4 SHIP vs LETKF_PLUS (단측, avg_nRMSE) + 필요조건
    print('\n  S-4 SHIP vs LETKF_PLUS (⛔크기 인용 금지 — 기여 ⑥ 4중 사망)')
    for direction in (180, 0):
        sl = paired(cells_n, info, 'SHIP', 'LETKF_PLUS', direction, 'crowded')
        sz_ = paired(cells_n, info, 'SHIP', 'ZERO', direction, 'crowded')
        lz = paired(cells_n, info, 'LETKF_PLUS', 'ZERO', direction, 'crowded')
        if not sl:
            continue
        st_sl = sign_test([-p[4] for p in sl], one_sided=True, label=f'S-4 dir{direction}')
        st_sz = sign_test([-p[4] for p in sz_], one_sided=True)
        st_lz = sign_test([-p[4] for p in lz], one_sided=True)
        deg = not (st_sz.get('pass_') and st_lz.get('pass_'))
        tests[f'S-4 dir{direction} SHIP<LETKF'] = st_sl
        print(f'    dir{direction:<3d} SHIP<LETKF {st_sl["pos"]}/{st_sl["n"]} '
              f'(문턱 {st_sl.get("crit")}) p={pfmt(st_sl)} | '
              f'SHIP<ZERO {st_sz["pos"]}/{st_sz["n"]} · LETKF<ZERO '
              f'{st_lz["pos"]}/{st_lz["n"]}'
              + ('   ⟹ INCONCLUSIVE-DEGENERATE' if deg else ''))

    print('\n  [Holm-4 보정]')
    for k, v in sorted(holm(tests).items(), key=lambda kv: kv[1]['p']):
        print(f'    {k:<40s} p={v["p"]:.3e} <= {v["holm_thr"]:.4f} ? '
              f'{"REJECT H0" if v["reject"] else "-"}')

    # ── 기술 보고 ───────────────────────────────────────────────────────────
    print('\n[T-1 window_resid — S4 §3 의 이분법 (탐색적, 검정 없음)]')
    wc, _ = med_by_cell(rows, 'window_resid')
    for direction in (0, 180):
        for arm in (PRIMARY_ARMS[0], PRIMARY_ARMS[1], 'AUTO-NT@L4'):
            v = [wc[(ep, arm)] for ep in info
                 if (ep, arm) in wc and info[ep]['direction'] == direction
                 and info[ep]['stratum'] == 'crowded']
            if v:
                print(f'  dir{direction:<3d} {arm:<19s} window_resid 중앙 = '
                      f'{np.median(v):.5f} m  (n={len(v)})')

    print('\n[T-2 셸(obs:) / 고리(ship:) — ⚠서로 다른 양이다]')
    for direction in (0, 180):
        for arm in (PRIMARY_ARMS[0], 'AUTO-NT@L4', 'SHIP', 'LETKF_PLUS', 'ZERO'):
            acc = {}
            for r in rows:
                if r['arm'] != arm or r['direction'] != direction:
                    continue
                if r['stratum'] != 'crowded':
                    continue
                for k, v in {**(r.get('shell_rmse') or {}),
                             **(r.get('ship_ring_rmse') or {})}.items():
                    if v is not None and np.isfinite(v):
                        acc.setdefault(k, []).append(v)
            if not acc:
                continue
            def _lo(kv):                    # '100_150' 이 '10_20' 앞에 오지 않게 수치 정렬
                return float(kv[0].split(':')[1].split('_')[0])
            obs = ' '.join(f'{k.split(":")[1]}={np.median(v):.3f}'
                           for k, v in sorted(acc.items(), key=_lo)
                           if k.startswith('obs:'))
            shp = ' '.join(f'{k.split(":")[1]}={np.median(v):.3f}'
                           for k, v in sorted(acc.items(), key=_lo)
                           if k.startswith('ship:'))
            print(f'  dir{direction:<3d} {arm:<13s} obs:  {obs}')
            print(f'  {"":18s} ship: {shp}')

    print('\n[T-3 라우터 (⚠체크포인트는 direction_deg~U(-50,50) 에서만 학습됐다)]')
    for direction in (0, 180):
        for arm in ('AUTO-NT@L4', 'SHIP'):
            sel = [r.get('auto_selected') for r in rows
                   if r['arm'] == arm and r['direction'] == direction]
            ns = [r.get('auto_n_selected') for r in rows
                  if r['arm'] == arm and r['direction'] == direction
                  and r.get('auto_n_selected') is not None]
            if sel:
                cnt = {s: sel.count(s) for s in set(sel)}
                print(f'  dir{direction:<3d} {arm:<13s} route {cnt}  '
                      f'n_selected 중앙 {np.median(ns) if ns else float("nan")}')

    print('\n[T-4 ORA-K-NT@L4 — k 오라클 단독]')
    for direction in (0, 180):
        pr = paired(cells, info, 'ORA-K-NT@L4', 'AUTO-NT@L4', direction, 'crowded')
        if pr:
            st = sign_test([p[4] for p in pr], one_sided=False)
            print(f'  dir{direction:<3d} ORA-K-NT − AUTO-NT (obs_loss) '
                  f'{st["pos"]}/{st["n"]} p={pfmt(st)} '
                  f'abs med Δ={np.median([p[4] for p in pr]):+.6f}')

    print('\n[T-6 셀별 분해 (기술통계, 검정 없음)]')
    for direction in (0, 180):
        for bf in ('BF4', 'BF5'):
            for nc in (15, 30, 2, 5):
                d = [p[4] for p in paired(cells, info, PRIMARY_ARMS[0], PRIMARY_ARMS[1],
                                          direction,
                                          'crowded' if nc in (15, 30) else 'sparse')
                     if p[1]['bf'] == bf and p[1]['ncomp'] == nc]
                if d:
                    print(f'  dir{direction:<3d} {bf}_{nc:2d}comp  gap 중앙 = '
                          f'{np.median(d):+.6f}  부호+ {sum(1 for x in d if x>0)}/{len(d)}')

    print('\n' + '=' * 92)
    print(f'★판정 = {verdict}')
    print('=' * 92)


if __name__ == '__main__':
    main()
