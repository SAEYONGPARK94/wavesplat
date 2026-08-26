#!/usr/bin/env python3
r"""T2 재집계기 — **논문에 새로 들어가는 두 수의 단일출처** (2026-08-27).

⛔**새 런 0회.** 기존 배치의 **동결 원자료**에서만 재도출한다.
⛔**기존 판정 라벨·문턱·게이트·결과 파일을 하나도 건드리지 않는다.**

## 왜 이 파일이 필요한가
2026-08-26 심사 토론이 두 가지를 지적했고, 둘 다 **논문에 숫자를 넣거나 빼는 일**이다.
이 저장소는 *"손으로 복사한 기준 상수"* 부류로 반복해서 대가를 치렀으므로
(⟹ `tmp/_audit/find_handcopied_refs.py`), **원고에 들어가는 수는 코드가 낳아야 한다.**

### 축 A — (d) 같은 규칙을 두 문단에 다르게 적용한다
`thesis_main.tex` 의 **혼잡 표현절제**(`B-16`) 문단은 용량--반응의 크기 $-26.28\%$ 를
표에 싣는데, 50줄 뒤 **희소**(`B-19`) 문단은 *구조가 완전히 같은* $-24.62\%$ 를
*"무기술 팔 대비 크기는 비교량이 아니다"* 라며 스스로 금지한다.
`B-19` 는 자기 배치 안에 참 제로필드 열이 있어 그 판정을 할 수 있었고
(`neg_kgrid2_vs_zero` = 2/16), **`B-16` 에는 그 열이 없다**(`zero_nRMSE` 전 행 `null`).
⟹ 여기서 그 열을 **같은 48 에피소드의 `B-12` 동결 행**에서 가져와 채운다.

⚠**정직성**: 이것은 `B-16` 사전등록에 없던 **사후 재집계**다. 그러나 방향이
***주장을 빼는 쪽***이다 — 새 주장을 사지 않고, 이미 실린 크기 하나를 **철회**시킨다.
★그리고 그 대체는 공짜가 아니다: **양성대조**로 `B-16` 의 `SHIP-NOW` 행이 `B-12` 의
`SHIP-NOW` 행과 **비트 동일**함을 먼저 확인한다 (= 두 배치가 같은 채점 위에 있다).

### 축 B — 크기 인용 규율
같은 `SPEC25` 열(`S16` 384행)에 **우리 팔만 갈아 끼우면** 마진이 크게 움직이는데,
그 팔들끼리는 짝 부호검정에서 **널**이다. ⟹ 원고가 인용하는 소수 둘째 자리는
**표본이 아니라 시드 추첨을 인용**하고 있다. 이 파일이 그 스프레드를 낳는다.

산출: `tmp/_audit/verdict_t2_recheck.json`
  (⚠파일명이 `verdict*` 인 이유 = `tmp/_audit/check_thesis_numbers.py` 가 원고의 수를
   `tmp/**/verdict*.json` 에서 찾는다. 여기 없으면 새 수가 *"대응 기록 없음"* 으로 뜬다.)

사용:  python3 tmp/_audit/recheck_t2_2026-08-27.py
종료코드 1 = 재도출이 기존 동결 판정(`verdict_b16.json`)과 어긋남 = **사람이 볼 일**.
"""
import io, json, os, statistics as st, sys
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
TMP = os.path.abspath(os.path.join(HERE, '..'))

B16 = [os.path.join(TMP, 'B16_KGRID_2026-08-24', f'results_b16_{i}.jsonl') for i in range(4)]
B12 = os.path.join(TMP, 'B12_HOS_EXTEND_2026-08-23', 'results_b12.jsonl')
S16 = os.path.join(TMP, 'S16_BASELINE_2026-08-14', 'results.jsonl')
W16 = os.path.join(TMP, 'W16_REBASELINE_2026-08-22', 'results.jsonl')
W15 = os.path.join(TMP, 'W15_DEPLOY_VERIFY_2026-08-22', 'results.jsonl')
S13 = os.path.join(TMP, 'S13_oos_2026-08-12', 'results.jsonl')
S13B = os.path.join(TMP, 'S13b_seed_robust_2026-08-13', 'results_s13b.jsonl')
FIELD = 'avg_nRMSE'


def sign_p(w, n):
    """단측 부호검정 P(X >= w | p=0.5). ⚠`B-19`·`B-16` 분석기와 같은 규약."""
    return sum(comb(n, k) for k in range(w, n + 1)) / 2 ** n if n else None


def load(*paths):
    rows = []
    for p in paths:
        with open(p, encoding='utf-8') as fh:
            rows += [json.loads(l) for l in fh if l.strip()]
    return rows


def arm_of(r):
    """⚠배치마다 팔 키가 다르다 — `W16`/`W15` 는 `cell`, 나머지는 `arm`.
    ★이것 자체가 저장소의 결함 부류(*'같은 것을 두 이름으로'*)이고 `gates_b1.py` 가
    이미 같은 흡수를 한다. **여기서도 같은 방식으로 흡수**한다."""
    return r.get('cell') if r.get('cell') is not None else r.get('arm')


def med_by_episode(rows, arms, stratum=None):
    d = {}
    for r in rows:
        if 'error' in r or arm_of(r) not in arms:
            continue
        if stratum is not None and r.get('stratum') != stratum:
            continue
        v = r.get(FIELD)
        if v is not None:
            d.setdefault(r['episode'], []).append(v)
    return {e: st.median(v) for e, v in d.items()}


def paired(a, b):
    ks = sorted(set(a) & set(b))
    w = sum(1 for k in ks if a[k] < b[k])
    t = sum(1 for k in ks if a[k] == b[k])
    return {'n': len(ks), 'better': w, 'n_tie': t,
            'rel_median_pct': st.median([(a[k] - b[k]) / b[k] * 100 for k in ks]) if ks else None,
            'p_one_sided': sign_p(w, len(ks))}


def main():
    out, gates = {}, []

    # ================= 축 A — B-16 혼잡 절제의 음성 대조 =================
    b16, b12 = load(*B16), load(B12)
    arms16 = {a: med_by_episode(b16, {a}) for a in ('SHIP-NOW', 'KGRID1', 'KGRID2')}

    # ★양성대조 — 두 배치가 같은 채점 위에 있는가 (비트 동일)
    def by_seed(rows, arm):
        d = {}
        for r in rows:
            if arm_of(r) == arm and r.get(FIELD) is not None:
                d.setdefault(r['episode'], {})[r['est_seed']] = r[FIELD]
        return d
    s16r, s12r = by_seed(b16, 'SHIP-NOW'), by_seed(b12, 'SHIP-NOW')
    n_cmp = sum(len(v) for v in s16r.values())
    n_bad = sum(1 for e in s16r for s in s16r[e] if s12r.get(e, {}).get(s) != s16r[e][s])
    gates.append(('★A-0 양성대조: B-16 `SHIP-NOW` 가 B-12 동결 행과 비트 동일',
                  n_bad == 0 and n_cmp == 144, f'{n_cmp - n_bad}/{n_cmp} 일치 · 불일치 {n_bad}'))

    # 참 제로필드 — B-12 가 같은 에피소드에 대해 기록한 값 (에피소드 안 불변이어야 한다)
    zsets = {}
    for r in b12:
        if r.get('zero_nRMSE') is not None:
            zsets.setdefault(r['episode'], set()).add(round(r['zero_nRMSE'], 12))
    zero = {e: next(iter(v)) for e, v in zsets.items()}
    gates.append(('A-1 참 제로필드가 에피소드 안에서 불변 · 48 에피소드',
                  len(zero) == 48 and all(len(v) == 1 for v in zsets.values()),
                  f'{len(zero)} 에피소드 · 값 개수 {sorted({len(v) for v in zsets.values()})}'))
    gates.append(('A-2 에피소드 집합 완전 일치 (B-16 팔 3개 ∩ 제로필드)',
                  all(set(v) == set(zero) for v in arms16.values()),
                  f'교집합 {len(set(arms16["SHIP-NOW"]) & set(zero))} / 합집합 '
                  f'{len(set(arms16["SHIP-NOW"]) | set(zero))}'))

    out['b16_vs_true_zero'] = {a: paired(arms16[a], zero) for a in arms16}
    out['b16_median_zero_nRMSE'] = st.median(list(zero.values()))

    # 동결 판정 재현 — ⛔여기서 어긋나면 사람이 볼 일이다
    out['b16_primary_rederived'] = paired(arms16['SHIP-NOW'], arms16['KGRID1'])
    out['b16_dose_rederived'] = paired(arms16['KGRID1'], arms16['KGRID2'])
    frozen = json.load(open(os.path.join(TMP, 'B16_KGRID_2026-08-24', 'verdict_b16.json'),
                            encoding='utf-8'))
    ok = all(abs(out[k]['rel_median_pct'] - frozen[j]['rel_median_pct']) < 1e-9
             and out[k]['better'] == frozen[j]['better']
             for k, j in (('b16_primary_rederived', 'primary'), ('b16_dose_rederived', 'dose')))
    gates.append(('★A-3 동결 `verdict_b16.json` 재현 (primary·dose)', ok,
                  f'primary {out["b16_primary_rederived"]["better"]}/48 '
                  f'{out["b16_primary_rederived"]["rel_median_pct"]:.2f}% · '
                  f'dose {out["b16_dose_rederived"]["better"]}/48 '
                  f'{out["b16_dose_rederived"]["rel_median_pct"]:.2f}%'))

    # ================= 축 B — 크기 인용 스프레드 =================
    base = med_by_episode(load(S16), {'SPEC25'}, stratum='crowded')
    ours = {
        'S13 계열 (median-of-3, 2026-08-12~13)':
            med_by_episode(load(S13, S13B), {'SHIP', 'SHIP@43', 'SHIP@44'}, stratum='crowded'),
        'W15 PRE (배포 전, 시드 1626-1628)':
            med_by_episode(load(W15), {'PRE'}, stratum='crowded'),
        'W15 POST (배포 후, 시드 1626-1628)':
            med_by_episode(load(W15), {'POST'}, stratum='crowded'),
        'W16 = B-1 헤드라인 (배포 후, 시드 1629-1631)':
            med_by_episode(load(W16), {'SHIP'}, stratum='crowded'),
    }
    gates.append(('B-0 네 팔이 모두 SPEC25 와 같은 64 혼잡 에피소드',
                  all(set(v) == set(base) and len(v) == 64 for v in ours.values()),
                  ' · '.join(f'{k.split()[0]}:{len(set(v) & set(base))}' for k, v in ours.items())))
    out['vs_spec25'] = {k: paired(v, base) for k, v in ours.items()}
    K = list(ours)
    out['arm_vs_arm'] = {f'{K[2]}  <  {K[1]}': paired(ours[K[2]], ours[K[1]]),
                         f'{K[3]}  <  {K[2]}': paired(ours[K[3]], ours[K[2]]),
                         f'{K[3]}  <  {K[1]}': paired(ours[K[3]], ours[K[1]])}
    sp = [v['rel_median_pct'] for v in out['vs_spec25'].values()]
    out['spread_pp'] = {'min': min(sp), 'max': max(sp), 'range_pp': max(sp) - min(sp),
                        'seed_only_pp': abs(out['vs_spec25'][K[3]]['rel_median_pct']
                                            - out['vs_spec25'][K[2]]['rel_median_pct'])}
    # ★시드만 다른 두 팔이 정말 같은 구성인가 (배포 상수 되읽기)
    cfgs = {}
    for c in ('POST',):
        rs = [r for r in load(W15) if r['cell'] == c]
        cfgs['W15 POST'] = {'varpro_steps': sorted({r.get('varpro_steps') for r in rs}),
                            'init_iters': sorted({r.get('init_iters') for r in rs}),
                            'ensemble_k': sorted({r.get('ensemble_k') for r in rs}),
                            'seeds': sorted({r['seed'] for r in rs})}
    rs = load(W16)
    cfgs['W16'] = {'varpro_steps': sorted({r.get('varpro_steps') for r in rs}),
                   'init_iters': sorted({r.get('init_iters') for r in rs}),
                   'ensemble_k': sorted({r.get('ensemble_k') for r in rs}),
                   'seeds': sorted({r['seed'] for r in rs})}
    same = all(cfgs['W15 POST'][k] == cfgs['W16'][k]
               for k in ('varpro_steps', 'init_iters', 'ensemble_k'))
    diff_seed = cfgs['W15 POST']['seeds'] != cfgs['W16']['seeds']
    gates.append(('★B-1 W15 POST 와 W16 은 배포 상수 동일 · 추정기 시드만 다르다',
                  same and diff_seed, json.dumps(cfgs, ensure_ascii=False)))
    out['deploy_readback'] = cfgs

    # ================= 축 C — 기여 ⑧ 6-지표 패널 =================
    PANEL = ('variance_ratio', 'skill_score', 'dir_r2', 'hs_error', 'corr', 'rho')
    scored = {}
    for tag, paths in (('B-1/W16', [W16]), ('B-1/S16', [S16]), ('B-12', [B12]),
                       ('B-15', [os.path.join(TMP, 'B15_RAYDROP_2026-08-24',
                                              f'results_b15_{i}.jsonl') for i in range(6)]),
                       ('B-16', B16),
                       ('B-19', [os.path.join(TMP, 'B19_KGRID_SPARSE_2026-08-25',
                                              f'results_b19_{i}.jsonl') for i in range(4)])):
        paths = [p for p in paths if os.path.exists(p)]
        keys = set()
        for r in load(*paths)[:500]:
            keys |= set(r.keys())
        scored[tag] = sorted(k for k in keys if any(t in k.lower() for t in PANEL))
    out['panel_metric_keys_in_citable_batches'] = scored
    gates.append(('C-0 인용 가능 08월 배치에 6-지표 패널 채점 열이 0개',
                  all(v == [] for v in scored.values()),
                  json.dumps(scored, ensure_ascii=False)))

    # ================= 축 D — ★원고의 "유래표" 를 기계로 검사한다 =================
    # 반올림 인용으로 바꾸면 `check_thesis_numbers.py` 의 퍼센트 대조가 그 수를 못 잡는다
    # (정규식이 소수점을 요구한다). ⟹ 원고는 정확값을 **주석 유래표** 한 곳에 남기는데,
    # ★그 표 자체가 손복사다 = 이 저장소가 반복해서 물린 결함 부류.
    # ⟹ 여기서 표를 **파싱해서 출처와 대조**한다. 실제로 이 게이트가 작성 직후 4행을 잡았다.
    tex = os.path.abspath(os.path.join(TMP, '..', 'thesis', 'thesis_main.tex'))
    rows, bad = [], []
    if os.path.exists(tex):
        import re
        want = re.compile(r'^%\s+약\s+(-?\d+)%\s+\((?P<tag>[^)]+)\)\s*(?P<exact>-?\d+\.\d+)\s')
        for ln in io.open(tex, encoding='utf-8'):
            m = want.match(ln.rstrip('\n'))
            if m:
                rows.append((m.group('tag').strip(), float(m.group('exact')), int(m.group(1))))
        # 출처의 실제 값 (⛔손복사 금지 — 판정 파일과 이 배치의 재집계에서 읽는다)
        def vj(rel, *path):
            n = json.load(open(os.path.join(TMP, rel), encoding='utf-8'))
            for k in path:
                n = n[k]
            return n['rel_median_pct'] if isinstance(n, dict) and 'rel_median_pct' in n else n
        truth = {
            'B-1': vj('B1_BASELINE_2026-08-22/verdict_b1.json', 'crowded'),
            'B-12': vj('B12_HOS_EXTEND_2026-08-23/verdict_b12.json', 'ship_vs_spec'),
            'B-11b': vj('B11_HOS_BASELINE_2026-08-23/verdict_b11b.json', 'ship_vs_spec'),
            'S13': out['vs_spec25'][K[0]]['rel_median_pct'],
            'W15 PRE': out['vs_spec25'][K[1]]['rel_median_pct'],
            'W15 POST': out['vs_spec25'][K[2]]['rel_median_pct'],
        }
        b15 = json.load(open(os.path.join(TMP, 'B15_RAYDROP_2026-08-24', 'verdict_b15.json'),
                             encoding='utf-8'))['by_raydrop']
        w16 = json.load(open(os.path.join(TMP, 'W16_REBASELINE_2026-08-22', 'verdict_w16.json'),
                             encoding='utf-8'))['results']['crowded']['rel_median_pct']
        for tag, exact, rounded in rows:
            if tag == 'B-15':
                t = min((b15[k]['rel_median_pct'] for k in b15),
                        key=lambda v: abs(v - exact))          # slope/uniform 중 가까운 쪽
            elif tag == 'B-1' and abs(exact - w16) < abs(exact - truth['B-1']):
                t = w16                                        # 두 번째 B-1 행 = 무기술 대비
            else:
                t = truth.get(tag)
            if t is None:
                bad.append(f'{tag}: 출처 미정의')
            elif abs(t - exact) > 5e-4:
                bad.append(f'{tag}: 원고 {exact} vs 출처 {t:.4f}')
            elif round(abs(t)) != abs(rounded):
                bad.append(f'{tag}: 반올림 {rounded} 가 {t:.4f} 와 안 맞는다')
    gates.append(('★D-0 원고 주석 유래표의 정확값·반올림이 출처와 일치',
                  bool(rows) and not bad,
                  f'{len(rows)} 행 검사 · 이탈 {bad if bad else 0}'))
    out['thesis_provenance_rows'] = len(rows)

    out['gates'] = [[g[0], bool(g[1]), str(g[2])] for g in gates]
    out['gates_pass'] = sum(1 for g in gates if g[1])
    out['gates_total'] = len(gates)
    out['batch'] = 'T2-RECHECK'
    out['runs'] = 0
    out['source_results'] = [os.path.relpath(p, os.path.abspath(os.path.join(TMP, '..')))
                             for p in (B16 + [B12, S16, W16, W15, S13, S13B])]
    out['_honesty'] = ('사후 재집계다 — `B-16` 사전등록에 음성 대조가 없었다. '
                       '방향이 **주장을 빼는 쪽**이라 허용하되, 그 사실을 원고에 적는다.')

    with open(os.path.join(HERE, 'verdict_t2_recheck.json'), 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)

    print('=== 게이트 ===')
    for n_, q_, d_ in gates:
        print(f'  [{"PASS" if q_ else "FAIL"}] {n_}\n         {d_}')
    print(f'\n=== 축 A — B-16 혼잡 절제 팔의 참 제로필드 대비 (제로필드 = B-12 동결 열) ===')
    for a, p in out['b16_vs_true_zero'].items():
        print(f'  {a:10s} < 참 제로필드 : {p["better"]:2d}/{p["n"]}  '
              f'상대중앙 {p["rel_median_pct"]:+7.2f}%  단측 p={p["p_one_sided"]:.3g}')
    print(f'  (참 제로필드 nRMSE 중앙 = {out["b16_median_zero_nRMSE"]:.4f})')
    print('\n=== 축 B — 같은 SPEC25 열, 우리 팔만 교체 (혼잡 64) ===')
    for k, p in out['vs_spec25'].items():
        print(f'  {k:44s} {p["better"]:2d}/{p["n"]}  {p["rel_median_pct"]:+7.2f}%  '
              f'p={p["p_one_sided"]:.3g}')
    print('  --- 그런데 팔끼리는 ---')
    for k, p in out['arm_vs_arm'].items():
        print(f'  {k}\n      {p["better"]}/{p["n"]} (동점 {p["n_tie"]}) '
              f'{p["rel_median_pct"]:+.4f}%  p={p["p_one_sided"]:.3g}')
    print(f'  ⟹ 스프레드 {out["spread_pp"]["range_pp"]:.2f} pp · '
          f'★시드만 재추첨한 두 팔 사이 {out["spread_pp"]["seed_only_pp"]:.2f} pp')
    print('\n=== 축 C — 6-지표 패널 채점 열 ===')
    for k, v in scored.items():
        print(f'  {k:10s} {v}')
    bad = [g[0] for g in gates if not g[1]]
    print(f'\n게이트 {out["gates_pass"]}/{out["gates_total"]}' + (f' ⛔FAIL: {bad}' if bad else ''))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
