"""T-4 분석기 — ⛔**판정 규칙은 전부 `gates_t4.py`.** 여기는 입출력과 집계뿐이다.

    python3 tmp/T4_GEOMETRY_2026-08-30/analyze_t4.py            # -> verdict_t4.json
    python3 tmp/T4_GEOMETRY_2026-08-30/analyze_t4.py smoke      # 스모크 행으로 게이트만

⛔차단 게이트가 하나라도 실패하면 `label` 은 `None` 이고 **그 상태로 닫는다**(선례 `M-8`).
  결과를 보고 게이트를 고치지 않는다.
⛔이 파일에 문턱·라벨·해석 문구가 없다 — `gates_t4.thresholds/label/fold/READING` 을 부른다.
⚠`LEADFIELD-VOID` 는 **판정 거부가 아니라 무효 라벨**이다(`G-CMP` 는 차단 게이트지만
  실패 시 라벨이 `VOID` 로 나온다) ⟹ `blocked_by` 에서 `G-CMP` 를 뺀 나머지가 판정을 막는다.
"""
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
for _p in (HERE, ROOT,
           os.path.join(ROOT, 'tmp', 'B11_HOS_BASELINE_2026-08-23'),
           os.path.join(ROOT, 'tmp', 'B12_HOS_EXTEND_2026-08-23'),
           os.path.join(ROOT, 'tmp', 'S16_BASELINE_2026-08-14')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(ROOT)

import gates_t4 as G                                                # noqa: E402

SMOKE = os.path.join(HERE, 'smoke_t4.jsonl')
VERDICT = os.path.join(HERE, 'verdict_t4.json')


def _load(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


def safe_gates(rows):
    """`gates_t4.gates(rows)` 를 부르되 **크래시를 FAIL 로 강등**한다.

    ⚠2026-08-30 스모크가 잡은 것: `G-CMP` 가 `thresholds(nc)` 를 부르는데 그 함수는
    `nc >= 5` 에서만 정의된다(단측 부호검정 바닥 p = 2^-n ≤ 0.05 ⟺ n ≥ 5) ⟹ 짝이 5개
    미만이면 `StopIteration` **으로 죽는다**. ⛔이것은 `M-10 m-2` 와 같은 부류
    (*"게이트가 FAIL 이 아니라 크래시한다"*)다.
    ★**본런(n=48)에는 영향이 없다** — 스모크(에피소드 2)와 대량 에러 상황에서만 난다.
    ⛔그래도 분석기가 조용히 죽게 두지 않는다: 크래시를 **차단 FAIL 한 줄**로 만들고
      그 상태로 닫는다(⛔`gates_t4.py` 는 수정하지 않는다 — 판정 기계는 동결물이다).
    """
    try:
        return list(G.gates(rows)), None
    except Exception as exc:                                        # noqa: BLE001
        eps = len({r.get('episode') for r in rows if 'error' not in r})
        det = ('%r — 표본이 작아 `thresholds(n)` 이 정의되지 않는다(에피소드 %d, 필요 ≥5). '
               '⚠본런 n=%d 에서는 나지 않는다.' % (exc, eps, G.N_EP))
        return [('G-CRASH gates_t4.gates() 가 FAIL 이 아니라 크래시했다', False, det, True)], exc


def _cmp(rows, a_arm, a_get, b_arm, b_get):
    a = G.med_by_episode(rows, a_arm, a_get)
    b = G.med_by_episode(rows, b_arm, b_get)
    w, n, rel = G.paired_wins(a, b)
    return dict(better=w, n=n,
                rel_median_pct=(None if rel is None else round(100.0 * rel, 3)),
                p_one_sided=(G.sign_test_one_sided(w, n) if n else None))


def analyze(rows):
    good = [r for r in rows if 'error' not in r]
    gl, crash = safe_gates(rows)
    gates = [dict(name=n, **{'pass': bool(o)}, detail=d, blocking=bool(b))
             for n, o, d, b in gl]
    blocked = [g['name'] for g in gates
               if g['blocking'] and not g['pass'] and not g['name'].startswith('G-CMP')]
    cmp_ok = all(g['pass'] for g in gates if g['name'].startswith('G-CMP'))

    L0, LP = 0.0, G.PRIMARY_LEAD_TP
    gp = lambda L: G.lead_getter('lead_rmse', L)          # noqa: E731
    gz = lambda L: G.lead_getter('lead_zero', L)          # noqa: E731

    primary = _cmp(good, 'SHIP', gp(LP), 'SPEC25', gp(LP))
    comparator = _cmp(good, 'SPEC25', gp(LP), 'SPEC25', gz(LP))
    existence = _cmp(good, 'SHIP', gp(LP), 'SHIP', gz(LP))
    band = _cmp(good, 'SHIP', G.band_getter('lead_rmse'),
                'SPEC25', G.band_getter('lead_rmse'))

    # SECONDARY-B `DECAY`: 에피소드마다 r_L = SHIP_L / SPEC25_L 을 만들고 r_{1.0} > r_{0.0}.
    # ⛔두 rung 의 p 를 비교하지 않는다 — 이것은 **한 개의 짝 검정**이다.
    ratios = {}
    for L in (L0, LP):
        s = G.med_by_episode(good, 'SHIP', gp(L))
        b = G.med_by_episode(good, 'SPEC25', gp(L))
        ratios[L] = {e: s[e] / b[e] for e in (set(s) & set(b)) if b[e]}
    ks = sorted(set(ratios[L0]) & set(ratios[LP]))
    wd = sum(1 for e in ks if ratios[LP][e] > ratios[L0][e])
    decay = dict(more=wd, n=len(ks),
                 p_one_sided=(G.sign_test_one_sided(wd, len(ks)) if ks else None),
                 r_lead0_median=(round(st.median([ratios[L0][e] for e in ks]), 5)
                                 if ks else None),
                 r_primary_median=(round(st.median([ratios[LP][e] for e in ks]), 5)
                                   if ks else None))

    # 사다리 — ⚠**서술 전용, 검정 아님**(`PREREG §10-9`: rung 순위 지목 금지)
    ladder = {}
    for L in G.LEADS_TP:
        ladder[repr(float(L))] = dict(
            ship_vs_spec=_cmp(good, 'SHIP', gp(L), 'SPEC25', gp(L)),
            ship_vs_zero=_cmp(good, 'SHIP', gp(L), 'SHIP', gz(L)),
            spec_vs_zero=_cmp(good, 'SPEC25', gp(L), 'SPEC25', gz(L)),
            ship_median=(round(st.median(list(
                G.med_by_episode(good, 'SHIP', gp(L)).values())), 5)
                if G.med_by_episode(good, 'SHIP', gp(L)) else None),
            spec_median=(round(st.median(list(
                G.med_by_episode(good, 'SPEC25', gp(L)).values())), 5)
                if G.med_by_episode(good, 'SPEC25', gp(L)) else None),
            zero_median=(round(st.median(list(
                G.med_by_episode(good, 'SPEC25', gz(L)).values())), 5)
                if G.med_by_episode(good, 'SPEC25', gz(L)) else None))

    n = primary['n'] or G.N_EP
    thr = G.thresholds(n)
    lab = None if blocked else G.label(primary['better'], n, cmp_ok)
    return dict(
        batch='T4-GEOM', axis='LEADTIME', prefix='lead:',
        n_rows=len(rows), n_good=len(good),
        screen=dict(go=G.screen_go(),
                    no_go={nm: list(f) for nm, gg, f, c in G.screen() if not gg}),
        gates=gates, blocked_by=blocked, comparator_ok=bool(cmp_ok),
        thresholds=thr, mde=dict(q80=round(thr['q80'], 4), n=thr['n']),
        primary_lead_tp=G.PRIMARY_LEAD_TP, leads_tp=list(G.LEADS_TP),
        band_lead_tp=list(G.BAND_LEAD_TP),
        primary=primary, comparator=comparator, existence=existence,
        band_sensitivity=band, secondary_b_decay=decay, ladder=ladder,
        label=lab, fold=(G.fold(lab) if lab else None),
        reading=(G.READING.get(lab) if lab else None),
        disclosures=[
            '⚠blind 아님 — 결과 전 예측은 `LEADFIELD-HOLDS` 였다(PREREG §6-1).',
            '⚠접두어 `lead:` — `obs:`(공간, M-10) · `ship:`(배-거리)와 **다른 양**이다.',
            '⚠PRIMARY rung 1.0 Tp 는 실측 밴드(0.327–7.152 Tp)의 **아래쪽**이다.',
            '⚠혼잡 층 한정(ncomp ≥ 15) · 같은 48 에피소드를 보는 여섯 번째.',
            '⚠동결 재현은 비트가 아니라 rel Δ ≤ 1e-6 (G-14 가 비트 일치 수를 보고).',
            '⚠재현 통화 = 스레드 고정 · GPU 허용(M-10 AMENDMENT_1) ⟹ prior_device 병기.',
            '⛔사다리는 서술 전용 — rung 순위 지목 금지(F1 교훈).',
            '⛔기제·인과 지목 금지(재채점은 기제를 재지 않는다).'])


def main():
    smoke = len(sys.argv) > 1 and sys.argv[1] == 'smoke'
    if smoke:
        rows = _load(SMOKE) if os.path.exists(SMOKE) else []
        print('smoke_t4.jsonl: %d 행  ⚠스모크(판정 아님)' % len(rows))
        print('\n=== 게이트 발화 확인 (⚠표본 게이트는 스모크에서 실패가 정상) ===')
        gl, crash = safe_gates(rows)
        for nm, ok, det, blk in gl:
            print('  [%s]%s %s: %s' % ('PASS' if ok else 'FAIL',
                                       '' if blk else '(범위축소)', nm, det))
        good = [r for r in rows if 'error' not in r]
        print('\n=== 게이트 발화 근거 (⚠행 필드 그대로 — ⛔여기서 판정하지 않는다) ===')
        ev = [('G-1 표본', '팔별 행수 %r · 에피소드 %d'
               % ({a: sum(1 for r in good if r.get('arm') == a) for a in G.ARM_ORDER},
                  len({r['episode'] for r in good}))),
              ('G-2 에러', '%d' % sum(1 for r in rows if 'error' in r)),
              ('G-3 되읽기', '%r' % [(r['obj_init_method'], r['obj_tracking_mode'],
                                      r['cap_varpro_steps'], r['arm_cfg_nonempty'])
                                     for r in good if r['arm'] == 'SHIP']),
              ('G-4 동결 relΔ', '%r' % ['%.1e' % G._rel(float(r['avg_nRMSE']), f)
                                        for r in good
                                        for f in [G.load_frozen_b12().get(
                                            (r['arm'], r['episode'], r['seed']))]
                                        if f is not None]),
              ('G-5 창·원점', '%r' % sorted({(r['n_scored'], r['t_first'], r['t_last'],
                                             r['n_origins']) for r in good})),
              ('G-6 음성대조', '%d/%d True' % (sum(1 for r in good
                                                  if r['neg_lead0_bitmatch'] is True),
                                              len(good))),
              ('G-7 양성대조', '%r' % ['%.1e' % r['pos_compose_relerr'] for r in good
                                       if r['pos_compose_relerr'] is not None]),
              ('G-8 rung', '%r' % sorted({tuple(sorted(r['lead_rmse'])) for r in good})),
              ('G-9 lead 프레임', '%r' % sorted({tuple(sorted(r['lead_frames'].items()))
                                                for r in good})),
              ('G-10 스탬프', '%r' % sorted({(r['device'], r['cuda_available'],
                                             r['omp_threads'], r['code_md5'][:8])
                                            for r in good})),
              ('G-12 개입 계수', '%r' % sorted({(r['arm'], r['cap_forecast_steps'],
                                                r['cap_predict_future']) for r in good})),
              ('G-13 원점 집합', '%r' % sorted({(r['n_origins'], r['origin_first'],
                                                r['origin_last'], r['origin_stride_s'])
                                               for r in good})),
              ('G-15 mesh', '%r' % sorted({r['mesh_verdict'] for r in good})),
              ('G-16 층별 원점', '%r' % sorted({(r['episode'].split('_')[0],
                                                r['n_origins']) for r in good})),
              ('합성 전용', '%r' % sorted({(r['real_gate'], r['synthetic_only']['ok'])
                                          for r in good}))]
        for nm, det in ev:
            print('  %-16s %s' % (nm, det))
        print('\n=== 행 ===')
        for r in rows:
            if 'error' in r:
                print('  ⛔ERROR %-6s %-38s s%s %s'
                      % (r.get('arm'), str(r.get('episode'))[:38], r.get('seed'),
                         str(r.get('error'))[:120]))
                continue
            print('  %-6s %-30s s%-3s  L0=%.5f L0.2=%.5f L0.5=%.5f L1=%.5f | Z1=%.5f'
                  % (r['arm'], r['episode'][:30], r['seed'],
                     r['lead_rmse'][repr(0.0)], r['lead_rmse'][repr(0.2)],
                     r['lead_rmse'][repr(0.5)], r['lead_rmse'][repr(1.0)],
                     r['lead_zero'][repr(1.0)]))
            print('         org=%d[%d..%d] lf=%s neg=%s pos=%s fc=%d pf=%d '
                  'prior=%s mesh=%s %.1fs'
                  % (r['n_origins'], r['origin_first'], r['origin_last'],
                     r['lead_frames'], r['neg_lead0_bitmatch'],
                     ('%.2e' % r['pos_compose_relerr'])
                     if r['pos_compose_relerr'] is not None else '-',
                     r['cap_forecast_steps'], r['cap_predict_future'],
                     r['prior_device'], r['mesh_verdict'], r['wall_s']))
        return

    rows = G.load_rows()
    if not rows:
        # ⛔0행 가드 — `M-4` 는 결과 0행에서 분석기를 돌려 `label`·`gates` 키를 갖춘
        #   `verdict_*.json` 을 남겼고, 검사기가 그것을 「인용 가능」으로 찍었다.
        #   ⟹ 여기서는 **파일을 쓰지 않고** 죽는다.
        print('⛔results_t4_*.jsonl 이 0행이다 — verdict 을 쓰지 않는다(0행 가드, 선례 M-4).')
        sys.exit(2)
    v = analyze(rows)
    json.dump(v, open(VERDICT, 'w'), indent=1, ensure_ascii=False)

    print('results_t4_*.jsonl: %d 행 (유효 %d)' % (v['n_rows'], v['n_good']))
    print('\n=== 게이트 ===')
    for g in v['gates']:
        print('  [%s]%s %s: %s' % ('PASS' if g['pass'] else 'FAIL',
                                   '' if g['blocking'] else '(범위축소)',
                                   g['name'], g['detail']))
    print('  ⟹ %d/%d' % (sum(1 for g in v['gates'] if g['pass']), len(v['gates'])))

    t, m = v['thresholds'], v['mde']
    print('\n=== 판정 ===')
    print('  문턱 n=%d CONFIRM=%d NOMINAL=%d NOT_EARN=%d · MDE q80=%s'
          % (t['n'], t['CONFIRM'], t['NOMINAL'], t['NOT_EARN'], m['q80']))
    p = v['primary']
    print('  PRIMARY-A  SHIP<SPEC25 @ lead %.1f Tp : %d/%d  rel=%s%%  p=%s'
          % (v['primary_lead_tp'], p['better'], p['n'], p['rel_median_pct'],
             p['p_one_sided']))
    c = v['comparator']
    print('  G-CMP      SPEC25<ZERO               : %d/%d  p=%s'
          % (c['better'], c['n'], c['p_one_sided']))
    e = v['existence']
    print('  S-1        SHIP<ZERO (서술)          : %d/%d' % (e['better'], e['n']))
    b = v['band_sensitivity']
    print('  민감도     밴드 RMS(0.5,1.0)         : %d/%d  rel=%s%%'
          % (b['better'], b['n'], b['rel_median_pct']))
    d = v['secondary_b_decay']
    print('  SEC-B      DECAY r_1.0 > r_0.0       : %d/%d  p=%s  (r0=%s r1=%s)'
          % (d['more'], d['n'], d['p_one_sided'], d['r_lead0_median'],
             d['r_primary_median']))
    print('\n  사다리 (⚠서술 전용 · ⛔rung 순위 지목 금지):')
    for k, row in v['ladder'].items():
        print('    lead %-4s  SHIP %s  SPEC25 %s  ZERO %s | SHIP<SPEC25 %d/%d'
              % (k, row['ship_median'], row['spec_median'], row['zero_median'],
                 row['ship_vs_spec']['better'], row['ship_vs_spec']['n']))

    if v['blocked_by']:
        print('\n⛔라벨 없이 닫힘 — 실패 차단 게이트 %d건: %s'
              % (len(v['blocked_by']), v['blocked_by']))
    else:
        print('\n★%s  (접기 = %s)\n   %s' % (v['label'], v['fold'], v['reading']))
    print('\n공시:')
    for s in v['disclosures']:
        print('  ' + s)
    print('\n-> %s' % VERDICT)


if __name__ == '__main__':
    main()
