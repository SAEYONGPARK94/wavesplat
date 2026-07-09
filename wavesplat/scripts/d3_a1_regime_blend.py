"""A1 — regime-adaptive blend of existing encounter-forecast configs.

Recommendation #1 of docs/RMSE_SUMMARY_and_OPTIONS_2026-07-08.md: no new
experiment, just pick the best config per regime. The catch is that "pick the
best" on the same five records that report the score is selection-on-test, so
the blend is scored under leave-one-record-out cross-validation: the switch
rule is fitted on the other records and applied to the held-out one.

The rule is deployable — it reads only the forecast lead and the sea-state
regime, never the ground truth:

    lead <= L*  ->  config_short
    lead >  L*  ->  config_long

with (config_short, config_long, L*) fitted per regime. Regime (wind sea vs
storm) is a sea-state-level split; here it is taken from Hs, which the
estimator recovers from the LiDAR return itself.

Storm has a single record (LJ18g), so its rule cannot be cross-validated;
its numbers are in-sample and flagged as such.

Writes results/direction3/a1_regime_blend.json + docs/A1_REGIME_BLEND.md.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

RES = 'results/direction3'
W = '20.0'
WIND = ['YS17', 'AA15a', 'AA14', 'AA15b']
STORM = ['LJ18g']
HS_STORM_THRESH = 4.0          # m; wind-sea records sit at 1.3-2.0, LJ18g 10.9

CONFIGS = {
    'hand': '',
    'net': '_netv3',
    'bw': '_bw',
    'net+bw': '_bw_netv3',
    'bwx': '_bwx',
    'v8bw+bw': '_bw_netv8bw',
}
BANDS = [(0, 2, '0-2s'), (2, 10, '2-10s'), (0, 10, '0-10s'), (10, 20, '10-20s')]
# 0-2s and 2-10s are disjoint: the control loop consumes 0-2s, situational
# awareness the longer leads, so a single pooled 0-10s number hides which of
# the two a config actually serves.


def load_record(rec):
    """-> leads, gt[win, lead], {cfg: pred[win, lead]}, Hs."""
    base = json.load(open(f'{RES}/falcon_real_operator_{rec}.json'))
    leads = np.asarray(base['leads'], float)
    nl = len(leads)
    gt = np.array([np.asarray(w['gt'], float)[:nl] for w in base['windows']])
    hs = float(np.mean([w['Hs'] for w in base['windows']]))

    preds = {}
    for name, suf in CONFIGS.items():
        p = f'{RES}/falcon_real_operator_{rec}{suf}.json'
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        pr = np.array([np.asarray(w['by_W'][W]['pred'], float)[:nl]
                       for w in d['windows']])
        if pr.shape != gt.shape:
            continue
        preds[name] = pr
    return leads, gt, preds, hs


def mean_or_none(vals):
    """Mean over the entries that exist; None when a record supplies none.

    Records differ in horizon (AA15a/AA14/AA15b stop at 14 s) and AA14 has
    gaps in its stereo GT, so some (config, band) cells are genuinely absent.
    """
    ok = [v for v in vals if v is not None]
    return float(np.mean(ok)) if ok else None


def band_nrmse(pred, gt, leads, lo, hi):
    """Per-record band nRMSE, normalized by the band-pooled GT std.

    Matches the aggregation used by RMSE_SUMMARY_and_OPTIONS_2026-07-08.md
    (verified to reproduce that table exactly).
    """
    m = (leads >= lo) & (leads <= hi)
    p, g = pred[:, m].ravel(), gt[:, m].ravel()
    ok = np.isfinite(p) & np.isfinite(g)
    p, g = p[ok], g[ok]
    if len(g) < 4 or np.std(g) == 0:
        return None
    return float(np.sqrt(np.mean((p - g) ** 2)) / np.std(g))


def blend(preds, leads, short, long, lstar):
    """Deployable switch: config_short for lead<=L*, config_long beyond."""
    out = np.where((leads <= lstar)[None, :], preds[short], preds[long])
    return out


def fit_fixed(recs, data, objective_band):
    """Best single config on `recs` — the honest baseline for the blend.

    Comparing the blend against a config that was picked after looking at the
    test records (as the summary table's `net+bw` was) would credit the blend
    with someone else's hindsight.
    """
    lo, hi, _ = objective_band
    names = [c for c in CONFIGS if all(c in data[r][2] for r in recs)]
    best = None
    for n in names:
        vals = [band_nrmse(data[r][2][n], data[r][1], data[r][0], lo, hi)
                for r in recs]
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        sc = float(np.mean(vals))
        if best is None or sc < best[0]:
            best = (sc, n)
    return best


def fit_rule(recs, data, objective_band):
    """Grid-search (short, long, L*) minimizing mean band nRMSE over recs."""
    lo, hi, _ = objective_band
    leads = data[recs[0]][0]
    cand_l = [float(x) for x in leads if 0 <= x <= 10]
    names = [c for c in CONFIGS if all(c in data[r][2] for r in recs)]
    best = None
    for s in names:
        for lg in names:
            for lstar in cand_l:
                if s == lg and lstar != cand_l[0]:
                    continue          # degenerate duplicates
                vals = []
                for r in recs:
                    ld, gt, preds, _ = data[r]
                    b = band_nrmse(blend(preds, ld, s, lg, lstar), gt, ld,
                                   lo, hi)
                    if b is None:
                        break
                    vals.append(b)
                else:
                    sc = float(np.mean(vals))
                    if best is None or sc < best[0]:
                        best = (sc, s, lg, lstar)
    return best


def oracle_per_lead(preds, gt, leads, lo, hi):
    """Lower bound: per-lead argmin over configs, chosen with the GT.

    Pools squared errors sample-by-sample and normalizes with the same
    band-pooled GT std as band_nrmse, so the result is directly comparable
    to (and cannot exceed) any fixed or blended config. Weighting leads
    equally instead would break the bound whenever leads carry unequal
    numbers of valid GT samples, as in AA14.
    """
    names = list(preds)
    m = (leads >= lo) & (leads <= hi)
    sse, n = 0.0, 0
    for li in np.where(m)[0]:
        g = gt[:, li]
        best = None
        for nm in names:
            e = (preds[nm][:, li] - g) ** 2
            e = e[np.isfinite(e)]
            if len(e) == 0:
                continue
            s = float(np.sum(e))
            if best is None or s < best[0]:
                best = (s, len(e))
        if best is not None:
            sse += best[0]
            n += best[1]
    if n == 0:
        return None
    gb = gt[:, m].ravel()
    gb = gb[np.isfinite(gb)]
    if len(gb) < 4 or np.std(gb) == 0:
        return None
    return float(np.sqrt(sse / n) / np.std(gb))


def main():
    recs = WIND + STORM
    data = {r: load_record(r) for r in recs}

    report = {'W': W, 'objective_band': '0-10s',
              'aggregation': 'mean of per-record band nRMSE',
              'records': {r: {'Hs': data[r][3]} for r in recs},
              'regime_split': {'rule': f'Hs > {HS_STORM_THRESH} m -> storm',
                               'wind': WIND, 'storm': STORM}}

    # ---- reference: every fixed config, per regime ----
    fixed = {}
    for name in CONFIGS:
        ent = {}
        for lo, hi, bl in BANDS:
            wv = [band_nrmse(data[r][2][name], data[r][1], data[r][0], lo, hi)
                  for r in WIND if name in data[r][2]]
            sv = [band_nrmse(data[r][2][name], data[r][1], data[r][0], lo, hi)
                  for r in STORM if name in data[r][2]]
            ent[bl] = {'wind': mean_or_none(wv), 'storm': mean_or_none(sv)}
        fixed[name] = ent
    report['fixed_configs'] = fixed

    # ---- A1 wind regime: leave-one-record-out ----
    obj = (0, 10, '0-10s')
    loro = {}
    for held in WIND:
        train = [r for r in WIND if r != held]
        sc, s, lg, lstar = fit_rule(train, data, obj)
        fsc, fname = fit_fixed(train, data, obj)
        ld, gt, preds, _ = data[held]
        ent = {'fitted_on': train, 'rule': {'short': s, 'long': lg,
                                            'L_star': lstar},
               'train_score_0-10s': sc, 'test': {},
               'fixed_baseline': {'config': fname, 'train_score': fsc,
                                  'test': {}}}
        for lo, hi, bl in BANDS:
            ent['test'][bl] = band_nrmse(blend(preds, ld, s, lg, lstar),
                                         gt, ld, lo, hi)
            ent['fixed_baseline']['test'][bl] = band_nrmse(
                preds[fname], gt, ld, lo, hi)
        # held-out reference points
        ent['test_ref'] = {
            bl: {n: band_nrmse(preds[n], gt, ld, lo, hi) for n in preds}
            for lo, hi, bl in BANDS}
        ent['test_oracle_0-10s'] = oracle_per_lead(preds, gt, ld, 0, 10)
        loro[held] = ent
    report['wind_loro'] = loro

    # LORO summary: honest held-out mean
    for lo, hi, bl in BANDS:
        report.setdefault('wind_loro_mean', {})[bl] = mean_or_none(
            [loro[r]['test'][bl] for r in WIND])
        report.setdefault('wind_loro_fixed_mean', {})[bl] = mean_or_none(
            [loro[r]['fixed_baseline']['test'][bl] for r in WIND])
    report['wind_loro_oracle_0-10s'] = mean_or_none(
        [loro[r]['test_oracle_0-10s'] for r in WIND])

    # ---- per-band specialists under the same LORO protocol ----
    # The pooled 0-10s objective above is dominated by long leads, so it can
    # only ever reward the long-lead specialist. Scoring each disjoint band
    # separately asks the question the control problem actually poses.
    spec = {}
    for lo, hi, bl in [(0, 2, '0-2s'), (2, 10, '2-10s')]:
        picks, tests = {}, []
        for held in WIND:
            train = [r for r in WIND if r != held]
            _, fname = fit_fixed(train, data, (lo, hi, bl))
            ld, gt, preds, _ = data[held]
            picks[held] = fname
            tests.append(band_nrmse(preds[fname], gt, ld, lo, hi))
        spec[bl] = {'loro_picks': picks, 'heldout_mean': mean_or_none(tests)}
    report['band_specialists'] = spec

    # ---- A1 as actually motivated: switch at the control horizon (2 s) ----
    # L* is not fitted here; it is the control requirement boundary. Only the
    # long-lead config is chosen, by LORO on 2-10s.
    nat = {'L_star': 2.0, 'short': 'net', 'note':
           'L* fixed at the control horizon, not fitted', 'folds': {}}
    for held in WIND:
        train = [r for r in WIND if r != held]
        _, lg = fit_fixed(train, data, (2, 10, '2-10s'))
        ld, gt, preds, _ = data[held]
        pb = blend(preds, ld, 'net', lg, 2.0)
        nat['folds'][held] = {
            'long': lg,
            '0-2s': band_nrmse(pb, gt, ld, 0, 2),
            '2-10s': band_nrmse(pb, gt, ld, 2, 10),
            '0-10s': band_nrmse(pb, gt, ld, 0, 10)}
    for bl in ['0-2s', '2-10s', '0-10s']:
        nat[f'heldout_mean_{bl}'] = mean_or_none(
            [nat['folds'][r][bl] for r in WIND])
    report['a1_control_horizon'] = nat

    # ---- is the "v8bw is the wind-sea specialist" reversal real? ----
    # Four records is thin. Cluster-bootstrap the windows within each record
    # and re-average, so the CI reflects window-level noise on the claim.
    def paired_boot(a, b, lo, hi, n_boot=2000, seed=0):
        rng = np.random.default_rng(seed)
        d = []
        for _ in range(n_boot):
            va, vb = [], []
            for r in WIND:
                ld, gt, preds, _ = data[r]
                sel = rng.integers(0, gt.shape[0], gt.shape[0])
                xa = band_nrmse(preds[a][sel], gt[sel], ld, lo, hi)
                xb = band_nrmse(preds[b][sel], gt[sel], ld, lo, hi)
                if xa is None or xb is None:
                    break
                va.append(xa)
                vb.append(xb)
            else:
                d.append(np.mean(va) - np.mean(vb))
        d = np.asarray(d)
        return {'delta_median': float(np.median(d)),
                'ci95': [float(np.percentile(d, 2.5)),
                         float(np.percentile(d, 97.5))],
                'p_positive': float(np.mean(d > 0)), 'n_boot': len(d)}

    report['reversal_test'] = {
        'claim': 'v8bw+bw beats net+bw on wind 2-10s (delta > 0 favors v8bw)',
        'wind_2-10s': paired_boot('net+bw', 'v8bw+bw', 2, 10),
        'wind_0-2s_net_vs_v8bw': paired_boot('v8bw+bw', 'net', 0, 2),
    }

    # ---- storm regime: in-sample only (n=1) ----
    sc, s, lg, lstar = fit_rule(STORM, data, obj)
    ld, gt, preds, _ = data['LJ18g']
    storm = {'fitted_rule': {'short': s, 'long': lg, 'L_star': lstar},
             'fitted_is_in_sample': True, 'n_records': 1,
             'fitted_bands': {bl: band_nrmse(blend(preds, ld, s, lg, lstar),
                                             gt, ld, lo, hi)
                              for lo, hi, bl in BANDS},
             # Deployed storm rule: plain `net` at every lead. A switch point
             # fitted on the one storm record we own would be reporting its
             # own training score.
             'deployed_rule': 'net at all leads',
             'deployed_bands': {bl: band_nrmse(preds['net'], gt, ld, lo, hi)
                                for lo, hi, bl in BANDS}}
    report['storm'] = storm

    os.makedirs(RES, exist_ok=True)
    with open(f'{RES}/a1_regime_blend.json', 'w') as f:
        json.dump(report, f, indent=1)

    # ---- console ----
    print('=== A1 영역-적응 블렌드 (W=20, 목적=0-10s) ===\n')
    print('고정 구성 참조 (레코드평균 nRMSE):')
    print(f"{'config':<10} {'풍파 0-2s':>10} {'풍파 0-10s':>11} "
          f"{'폭풍 0-2s':>10} {'폭풍 0-10s':>11}")
    for n, e in fixed.items():
        def f(x):
            return f'{x:>10.3f}' if x is not None else f'{"--":>10}'
        print(f"{n:<10} {f(e['0-2s']['wind'])} {f(e['0-10s']['wind']):>11} "
              f"{f(e['0-2s']['storm'])} {f(e['0-10s']['storm']):>11}")

    print('\n풍파 LORO (규칙은 나머지 3레코드로 적합, held-out에서 평가):')
    for r in WIND:
        e = loro[r]
        ru = e['rule']
        ref = {k: v for k, v in e['test_ref']['0-10s'].items()
               if v is not None}
        best_fixed = min(ref, key=ref.get)
        fb = e['fixed_baseline']
        print(f"  {r:<6} rule: lead<={ru['L_star']:.0f}s -> {ru['short']}, "
              f"else {ru['long']}  | LORO 고정 baseline: {fb['config']}")
        print(f"         0-10s: blend={e['test']['0-10s']:.3f} "
              f"고정={fb['test']['0-10s']:.3f} "
              f"oracle={e['test_oracle_0-10s']:.3f}  | "
              f"0-2s: blend={e['test']['0-2s']:.3f} "
              f"고정={fb['test']['0-2s']:.3f}")
        print(f"         (사후지식 참조: 최적고정={best_fixed} "
              f"{ref[best_fixed]:.3f})")

    lm, lf = report['wind_loro_mean'], report['wind_loro_fixed_mean']
    print("\n  ★ 풍파 held-out 평균 (blend·고정 둘 다 LORO로 선택 = 공정):")
    print(f"      A1 blend       : 0-2s={lm['0-2s']:.3f}  "
          f"0-10s={lm['0-10s']:.3f}")
    print(f"      최적 단일 고정 : 0-2s={lf['0-2s']:.3f}  "
          f"0-10s={lf['0-10s']:.3f}")
    print(f"      oracle 하한    :             "
          f"0-10s={report['wind_loro_oracle_0-10s']:.3f}")
    print("\n  (아래는 테스트를 보고 고른 구성 — baseline 아님, 참조용)")
    print(f"      고정 net+bw    : 0-2s={fixed['net+bw']['0-2s']['wind']:.3f}"
          f"  0-10s={fixed['net+bw']['0-10s']['wind']:.3f}")
    print(f"      고정 net       : 0-2s={fixed['net']['0-2s']['wind']:.3f}"
          f"  0-10s={fixed['net']['0-10s']['wind']:.3f}")

    print("\n  대역별 전문가 (각 대역을 목적으로 LORO 선택한 단일 고정 구성):")
    for bl, e in spec.items():
        picks = ', '.join(f'{k}->{v}' for k, v in e['loro_picks'].items())
        print(f"      {bl:<6} held-out 평균={e['heldout_mean']:.3f}   ({picks})")

    nat = report['a1_control_horizon']
    longs = {v['long'] for v in nat['folds'].values()}
    print("\n  ★ A1 (제어지평 L*=2s 고정, 장리드 구성만 LORO 선택):")
    print(f"      장리드 선택={'/'.join(sorted(longs))}")
    print(f"      held-out  0-2s={nat['heldout_mean_0-2s']:.3f}  "
          f"2-10s={nat['heldout_mean_2-10s']:.3f}  "
          f"0-10s={nat['heldout_mean_0-10s']:.3f}")

    rt = report['reversal_test']
    print("\n  역전 검정 (윈도우 cluster bootstrap, 2000회):")
    for k, lbl in [('wind_2-10s', 'v8bw+bw < net+bw @풍파 2-10s'),
                   ('wind_0-2s_net_vs_v8bw', 'net < v8bw+bw @풍파 0-2s')]:
        e = rt[k]
        print(f"      {lbl:<32} Δ={e['delta_median']:+.3f} "
              f"[{e['ci95'][0]:+.3f}, {e['ci95'][1]:+.3f}]  "
              f"P(Δ>0)={e['p_positive']:.3f}")

    print("\n폭풍 LJ18g (n=1 — 교차검증 불가):")
    db = storm['deployed_bands']
    print(f"  배치 규칙: {storm['deployed_rule']}  "
          f"0-2s={db['0-2s']:.3f}  2-10s={db['2-10s']:.3f}  "
          f"0-10s={db['0-10s']:.3f}")
    fb2, fr = storm['fitted_bands'], storm['fitted_rule']
    print(f"  (참고) 자기자신에 적합한 규칙 lead<={fr['L_star']:.0f}s->"
          f"{fr['short']} else {fr['long']}: 0-10s={fb2['0-10s']:.3f} "
          f"— in-sample, 채택 안 함")
    print(f"  ※ v8bw를 폭풍에 쓰면 0-2s={fixed['v8bw+bw']['0-2s']['storm']:.3f}"
          f" (OOD 붕괴) — regime 라우팅이 막아주는 실패")
    print(f"\n-> {RES}/a1_regime_blend.json")


if __name__ == '__main__':
    sys.exit(main())
