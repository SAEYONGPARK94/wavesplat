"""T3-NOW 분석기 — ⛔**판정은 `gates_t3.analyze()` 가 한다.** 여기는 입출력뿐이다.

    python3 tmp/T3_NOWCAST_2026-08-30/analyze_t3.py            # -> verdict_t3now.json
    python3 tmp/T3_NOWCAST_2026-08-30/analyze_t3.py smoke      # 스모크 행으로 게이트만

⛔게이트가 하나라도 실패하면 `label` 은 `None` 이고, **그 상태로 닫는다**(선례 `M-8`).
결과를 보고 게이트를 고치지 않는다.

⚠**스모크 모드는 판정이 아니다.** 셀이 1개이므로 **표본 게이트는 당연히 실패한다**
(`G-0`·`G-0c`·`G-1`·`G-MIRROR`·`G-ANCHOR`·`G-PC` 등이 `N_PAIRS=36` 을 요구한다).
⟹ 스모크 모드는 게이트 발화 목록에 더해 ★**표본 크기와 무관한 셀 단위 불변식**을 따로
찍는다(거울 비트 · 앵커 비트 · 멤버 규약 · VARPRO · 장치 스탬프 · 채점 기준).
그 진단의 기대값은 전부 `gates_t3.expected_*()` 에서 오고 **판정에 쓰이지 않는다**.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
for _p in (HERE, ROOT, os.path.join(ROOT, 'tmp', 'B2_REAL_2026-08-23'),
           os.path.join(ROOT, 'tmp', 'B3_REFIT_2026-08-23')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(ROOT)

import gates_t3 as G                                                # noqa: E402

RESULTS = os.path.join(HERE, 'results_t3.jsonl')
SMOKE = os.path.join(HERE, 'smoke_t3.jsonl')
VERDICT = os.path.join(HERE, 'verdict_t3now.json')


def load(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as fh:
        for line in fh:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


# ── ★스모크 전용 진단 (⛔판정 아님 · 표본 크기 무관한 셀 단위 불변식만) ──────
def smoke_checks(rows):
    """(이름, PASS?, 상세) 목록. 기대값은 전부 `gates_t3` 에서 받는다."""
    ok = [r for r in rows if 'error' not in r]
    by = {r['arm']: r for r in ok}
    out = []

    def A(n, p, d=''):
        out.append((n, bool(p), str(d)))

    A('에러 0', len(rows) == len(ok) and len(rows) > 0, f'{len(rows) - len(ok)} 에러')
    A('팔 전건', set(by) == set(G.ARM_ORDER), str(sorted(by)))

    for arm in G.ARM_ORDER:
        r = by.get(arm)
        if r is None:
            A(f'{arm} 행 존재', False, '없음')
            continue
        nm = G.expected_members(arm)
        A(f'{arm} 멤버 수/오프셋', r['n_members'] == nm
          and tuple(r['member_offsets']) == tuple(G.ARMS[arm]['MEM']),
          f'{r["n_members"]} {r["member_offsets"]}')
        vp = G.expected_vp_steps(arm)
        A(f'{arm} vp_steps == {vp}', list(r['vp_steps_executed']) == [vp] * nm
          and all(bool(x) == bool(vp) for x in r['vp_fired']),
          f'{r["vp_steps_executed"]} fired={r["vp_fired"]}')
        bm = G.expected_bank_members(arm)
        A(f'{arm} bank_members == {bm}', list(r['ens_k_readback']) == [bm] * nm,
          str(r['ens_k_readback']))
        want = [G.member_overrides(arm, o) for o in G.ARMS[arm]['MEM']]
        A(f'{arm} 멤버 시드 규약',
          list(r['seed_readback']) == [w['seed'] for w in want]
          and list(r['diffusion_seed_readback'])
          == [w.get('learned_diffusion_seed') for w in want],
          f'seed={r["seed_readback"]} dseed={r["diffusion_seed_readback"]}')
        if nm > 1:
            nd = len(set(r['field_md5_members']))
            if arm in G.PROP_ARMS:
                # 확산 재표집 ⟹ **셀마다** 전건 distinct 여야 한다(`G-MIRROR-b` PROP 가지)
                A(f'{arm} 멤버 장이 서로 다르다(공허 아님)', nd == nm,
                  str([h[:8] for h in r['field_md5_members']]))
            else:
                # ⚠bank 멤버는 val-split 시드만 다르다 ⟹ **셀 단위로는 퇴화가 정상**이다
                #   (36셀 중 9셀 완전퇴화 · 11셀 부분퇴화 = `AMENDMENT_0_PREFREEZE §1.1`).
                #   `G-MIRROR-b` 는 배치 전체에서 「전-distinct 셀 ≥1」만 요구한다.
                A(f'⚠{arm} 멤버 distinct(공시 — 셀 단위 퇴화는 정상)', True,
                  f'{nd}/{nm} '
                  f'{[h[:8] for h in r["field_md5_members"]]} '
                  f'{"⚠이 셀은 퇴화 ⟹ 거울이 이 셀에서는 공허하다" if nd == 1 else ""}')
        path = G.ARMS[arm]['PATH']
        A(f'{arm} 경로 되읽기 == {path}',
          all(x == ('bank_mmse' if path == 'bank' else 'learned')
              for x in r['init_method_readback'])
          and all((x == 0) if path == 'bank' else (x > 0) for x in r['n_atoms'])
          and all(x is (path == 'bank') for x in r['real_gate']),
          f'{r["init_method_readback"]} N={r["n_atoms"]} gate={r["real_gate"]}')
        if arm in G.ANCHORS:
            A(f'★{arm} 앵커 비트 일치', r['anchor_bitmatch'] is True,
              str(r['anchor_bitmatch']))
        else:
            A(f'{arm} 앵커 없음(None)', r['anchor_bitmatch'] is None,
              str(r['anchor_bitmatch']))

    # ★★거울 — 표본 크기와 무관한 **셀 단위** 항등식
    a, b = by.get('SHIP-K1M3'), by.get('SHIP-K3')
    if a and b:
        A('★★G-MIRROR(셀) 하네스 median == 패키지 k=3 (해시)',
          a['field_md5_med'] is not None
          and a['field_md5_med'] == b['field_md5_members'][0],
          f'{str(a["field_md5_med"])[:12]} vs {b["field_md5_members"][0][:12]}')
        A('★★G-MIRROR(셀) 값 비트 일치', a['e_med3'] == b['e_m0'],
          f'{a["e_med3"]!r} vs {b["e_m0"]!r}')

    # 채점 기준 · 장치 · 승인
    bad = [(r['record'], r['win']) for r in ok
           if r['zero_lead0'] != G.zero_lead0(r['record'], r['win'])]
    A('G-Z(셀) 참 제로필드 == zero_ref.py', not bad, f'{len(bad)} 불일치')
    stamp = sorted({(r['device'], r['cuda_available'], str(r['omp_threads']))
                    for r in ok})
    A('G-7(셀) 장치/스레드 단일 스탬프', len(stamp) == 1, str(stamp))
    A('G-7b/c 핀·승인', all(r['deploy_pin_ok'] is True and r['approved'] is True
                          for r in ok), '')
    A('행 계약 35키 전건',
      not [k for r in ok for k in G.ROW_CONTRACT if k not in r], '')
    return out


def main():
    smoke = len(sys.argv) > 1 and sys.argv[1] == 'smoke'
    path = SMOKE if smoke else RESULTS
    rows = load(path)
    print(f'{os.path.basename(path)}: {len(rows)} 행'
          f'{" ⚠스모크(판정 아님)" if smoke else ""}')

    if smoke:
        print('\n=== 게이트 발화 (⚠셀 1개 ⟹ 표본 게이트는 당연히 FAIL) ===')
        for n, p, d in G.gates(rows):
            print(f'  [{"PASS" if p else "FAIL"}] {n}: {d}')
        print('\n=== ★셀 단위 불변식 (표본 크기 무관 — 여기서 FAIL 이면 진짜 결함) ===')
        for n, p, d in smoke_checks(rows):
            print(f'  [{"PASS" if p else "FAIL"}] {n}: {d}')
        print('\n=== 행 ===')
        for r in rows:
            if 'error' in r:
                print(f'  ⛔ERROR {r["arm"]}: {r["error"]}')
                continue
            med = '—' if r['e_med3'] is None else f'{r["e_med3"]:.5f}'
            spr = '—' if r['spread_rms'] is None else f'{r["spread_rms"]:.5f}'
            print(f'  {r["arm"]:12s} m0={r["e_m0"]:.5f} med3={med} spread={spr} '
                  f'zero={r["zero_lead0"]:.5f} vp={r["vp_steps_executed"]} '
                  f'N={r["n_atoms"]} anchor={r["anchor_bitmatch"]} '
                  f'{r["wall_s"]:.1f}s = {r["wall_s"] / r["n_members"]:.1f}s/런')
        return

    v = G.analyze(rows)
    json.dump(v, open(VERDICT, 'w'), indent=1, ensure_ascii=False)

    print('\n=== 게이트 ===')
    for g in v['gates']:
        print(f'  [{"PASS" if g["pass"] else "FAIL"}] {g["name"]}: {g["detail"]}')
    npass = sum(1 for g in v['gates'] if g['pass'])
    print(f'  ⟹ {npass}/{len(v["gates"])}')

    print('\n=== 판정 ===')
    t, m = v['thresholds'], v['mde']
    print(f"  문턱 n={t['n']} NOMINAL={t['NOMINAL']} HOLM={t['HOLM_MIN_WINS']} "
          f"NOT_EARN={t['NOT_EARN']} · MDE 가족 q80={m['q80']} "
          f"(⛔단일검정 {v['mde_single']['q80']} 로 바꿔 적지 마라)")
    for cid, key in (('C1 PRIMARY', 'primary'), ('C2', 'c2'), ('C3', 'c3')):
        d = v[key]
        extra = (f" 비={d['ratio_median']:.4f}" if d.get('ratio_median') is not None
                 else (f" rel={d['rel_median_pct']}%"
                       if d.get('rel_median_pct') is not None else ''))
        p = ('—' if d['p_one_sided'] is None else f"{d['p_one_sided']:.5f}")
        print(f"  {cid:11s} {d['arm']} < {d['ref']}: {d['better']}/{d['n']} "
              f"p={p}{extra}  Holm={v['holm_survivors'][cid.split()[0]]}")

    print('\n=== ⚠탐색적 (가족 밖 · ⛔승격 금지) ===')
    e = v['exploratory']
    i = e['interaction']
    print(f"  E1 상호작용 g_V400<g_V0: {i['v400_gains_more']}/{i['n']} "
          f"(g_V400={i['g_v400_median']}, g_V0={i['g_v0_median']})")
    b = e['bank_ensemble']
    print(f"  E2 bank 앙상블 SHIP-K3<SHIP-K1M3: {b['better']}/{b['n']} "
          f"rel={b['rel_median_pct']}%")
    for arm, s in e['spread'].items():
        print(f"  E3 {arm:12s} spread={s['spread_median']} e_m0={s['e_m0_median']} "
              f"R={s['R_median']}")

    print('\n=== 서술 (참 제로필드 대비) ===')
    for k, d in v['descriptive']['vs_zero'].items():
        print(f"  {k:22s} {d['better']}/{d['n']} 비={d['ratio_median']}")
    print(f"  wall_s 중앙={v['descriptive']['wall_s_median']}")

    if v['blocked_by']:
        print(f'\n⛔라벨 없이 닫힘 — 실패 게이트 {len(v["blocked_by"])}건: '
              f'{v["blocked_by"]}')
    else:
        for lab, why in ((v['label'], v['why']), (v['label_c2'], v['why_c2']),
                         (v['label_c3'], v['why_c3'])):
            print(f'\n★{lab}\n   {why}')
    print(f'\n방향: {v["direction"]}')
    print(f'-> {VERDICT}')


if __name__ == '__main__':
    main()
