"""NEUTRAL — 명제 I 판정가능성 심문. 기존 원자료 재분석만 (새 런 0건).

입력 (전부 기존 파일, 읽기 전용):
  tmp/s1s2_2026-08-05/results.jsonl              (2026-08-05, 오라클 배치)
  tmp/wp_integration_2026-07-25/arm_runs.jsonl   (2026-07-25, learned/learned_konly/v2_konly)
  tmp/wp_crowded_2026-07-25/{ws,cond}_runs.jsonl (2026-07-25, -45% 헤드라인)
  tmp/cdip_support_2026-08-03/cdip_support_raw.json (100ep 확산)

출력: NEUTRAL_reanalysis.json
사용: python3 tmp/waveprior_debate_2026-08-05/NEUTRAL_reanalysis.py   (repo root = wavesplat/)
"""
import json, re, statistics as st

EPS = ['episode_BF4_15comp_seed0_uniform.h5', 'episode_BF4_30comp_seed0_uniform.h5',
       'episode_BF5_15comp_seed0_uniform.h5', 'episode_BF5_30comp_seed0_uniform.h5']
SH = [e[8:-3].replace('comp_seed0_uniform', 'c') for e in EPS]
CROWDED = [f'episode_BF{bf}_{nc}comp_seed{s}_{a}.h5'
           for bf in (4, 5) for nc in (15, 30) for s in range(5) for a in ('uniform', 'slope')]
# S1S2 RESULT.md §4.1 의 동결 LETKF 참조값 (seed 42 단일시드, 재실행 없음)
LETKF = dict(zip(EPS, [0.2282, 0.2347, 0.2298, 0.2651]))

J = lambda p: [json.loads(l) for l in open(p)]
today = J('tmp/s1s2_2026-08-05/results.jsonl')
wpi = J('tmp/wp_integration_2026-07-25/arm_runs.jsonl')
ws = J('tmp/wp_crowded_2026-07-25/ws_runs.jsonl')
cond = J('tmp/wp_crowded_2026-07-25/cond_runs.jsonl')
OUT = {}

# ---------------------------------------------------------------- Q2-a 비트동일
FIELDS = [('nRMSE', 'avg_nRMSE'), ('Variance_Ratio', 'avg_Variance_Ratio'),
          ('Pearson_rho', 'avg_Pearson_rho'), ('Skill_Score', 'avg_Skill_Score'),
          ('Directional_R2', 'avg_Directional_R2'), ('Hs_error', 'avg_Hs_error'),
          ('N', 'final_n'), ('init_nrmse', 'init_nrmse')]
bit = {}
for e in EPS:
    t = [r for r in today if r['arm'] == 'S1-A0' and r['episode'] == e and r['seed'] == 42][0]
    w = [r for r in wpi if r['arm'] == 'v2_konly' and r['episode'] == e and r['est_seed'] == 42][0]
    bit[e] = {a: (repr(t[a]) == repr(w[b])) for a, b in FIELDS}
OUT['Q2a_bit_identity_control_arm'] = {
    'pairs_checked': f'{len(EPS)} episodes x {len(FIELDS)} metrics, lane est_seed=42',
    'all_exact': all(all(v.values()) for v in bit.values()),
    'per_episode': {e[8:-3]: v for e, v in bit.items()}}

# ---------------------------------------------------------------- Q2-b 부분집합 대표성
def med_wpi(arm, eps):
    return {e: st.median([r['avg_nRMSE'] for r in wpi
                          if r['arm'] == arm and r['episode'] == e and r['err'] is None])
            for e in eps}
v2_40, le_40, lk_40 = med_wpi('v2_konly', CROWDED), med_wpi('learned', CROWDED), med_wpi('learned_konly', CROWDED)
gain40 = {e: v2_40[e] - le_40[e] for e in CROWDED}
ranked = sorted(CROWDED, key=lambda e: -gain40[e])
OUT['Q2b_subset_representativeness'] = {
    'all40': {'v2_konly_mean': st.mean(v2_40.values() if 0 else [v2_40[e] for e in CROWDED]),
              'learned_mean': st.mean([le_40[e] for e in CROWDED]),
              'learned_konly_mean': st.mean([lk_40[e] for e in CROWDED]),
              'paired_gain_median': st.median([gain40[e] for e in CROWDED]),
              'wins': f"{sum(1 for e in CROWDED if gain40[e] > 0)}/40"},
    'the4': {'v2_konly_mean': st.mean([v2_40[e] for e in EPS]),
             'learned_mean': st.mean([le_40[e] for e in EPS]),
             'paired_gain_median': st.median([gain40[e] for e in EPS]),
             'wins': f"{sum(1 for e in EPS if gain40[e] > 0)}/4"},
    'rank_of_the4_within_40_by_learned_gain': {
        e[8:-3]: {'gain': round(gain40[e], 4), 'rank': ranked.index(e) + 1} for e in EPS},
    'gain40_quartiles': [round(x, 4) for x in st.quantiles([gain40[e] for e in CROWDED], n=4)]}

# ---------------------------------------------------------------- Q2-c 팔별 수준값
def med_today(arm, mode):
    return {e: st.median([r['nRMSE'] for r in today if r['arm'] == arm and r['episode'] == e
                          and (mode == 'med' or r['seed'] == 42)]) for e in EPS}
def med_wpi_mode(arm, mode):
    return {e: st.median([r['avg_nRMSE'] for r in wpi if r['arm'] == arm and r['episode'] == e
                          and r['err'] is None and (mode == 'med' or r['est_seed'] == 42)]) for e in EPS}
levels = {}
for mode in ('s42', 'med'):
    d = {'C_0725_v2_konly': med_wpi_mode('v2_konly', mode), 'C_today_S1A0': med_today('S1-A0', mode),
         'learned': med_wpi_mode('learned', mode), 'learned_konly': med_wpi_mode('learned_konly', mode),
         'S2_T1_oracle_place': med_today('S2-T1', mode), 'S2_T1f_oracle_place_freeze': med_today('S2-T1f', mode),
         'X_T1f_notrack': med_today('X-T1f-notrack', mode), 'LETKF_frozen': LETKF}
    levels[mode] = {k: {'per_ep': {e[8:-3]: round(v[e], 4) for e in EPS},
                        'mean4': round(st.mean([v[e] for e in EPS]), 4)} for k, v in d.items()}
    levels[mode]['_raw'] = d
OUT['Q2c_levels'] = {m: {k: v for k, v in levels[m].items() if k != '_raw'} for m in levels}

# ---------------------------------------------------------------- Q3 회수율 불안정성
rec = {}
for mode in ('s42', 'med'):
    d = levels[mode]['_raw']
    C_l, C_o = d['C_0725_v2_konly'], d['C_today_S1A0']
    for cname, ceil in [('vs_S2_T1_placement_only', d['S2_T1_oracle_place']),
                        ('vs_S2_T1f_placement_plus_freeze', d['S2_T1f_oracle_place_freeze']),
                        ('vs_X_T1f_notrack', d['X_T1f_notrack'])]:
        per = {}
        for e in EPS:
            gl, go = C_l[e] - d['learned'][e], C_o[e] - ceil[e]
            per[e[8:-3]] = round(gl / go * 100, 1) if go > 0 else None
        vals = [v for v in per.values() if v is not None]
        rec[f'{mode}|{cname}'] = {
            'per_episode_pct': per, 'median_pct': round(st.median(vals), 1),
            'pooled_pct': round((st.mean([C_l[e] for e in EPS]) - st.mean([d['learned'][e] for e in EPS])) /
                                (st.mean([C_o[e] for e in EPS]) - st.mean([ceil[e] for e in EPS])) * 100, 1)}
med_all = [v['median_pct'] for v in rec.values()]
per_all = [x for v in rec.values() for x in v['per_episode_pct'].values() if x is not None]
OUT['Q3_recovery_rate_instability'] = {
    'variants': rec,
    'median_pct_range_over_6_defensible_specs': [min(med_all), max(med_all)],
    'per_episode_pct_range': [min(per_all), max(per_all)],
    'note': '분자/분모 모두 측정값. 흔드는 것은 (a)시드 프로토콜 (b)어느 오라클 팔을 천장으로 쓰느냐 뿐.'}

# ---------------------------------------------------------------- Q3-b 오라클이 상한인가
beat = {}
for e in EPS:
    L = [r['avg_nRMSE'] for r in wpi if r['arm'] == 'learned' and r['episode'] == e and r['est_seed'] == 42][0]
    Lall = [r['avg_nRMSE'] for r in wpi if r['arm'] == 'learned' and r['episode'] == e]
    T1 = [r['nRMSE'] for r in today if r['arm'] == 'S2-T1' and r['episode'] == e]
    T1f = [r['nRMSE'] for r in today if r['arm'] == 'S2-T1f' and r['episode'] == e]
    beat[e[8:-3]] = {'learned_s42': round(L, 4), 'T1_s42': round(T1[[r['seed'] for r in today if r['arm'] == 'S2-T1' and r['episode'] == e].index(42)], 4),
                     'learned_beats_T1_at_s42': L < [r['nRMSE'] for r in today if r['arm'] == 'S2-T1' and r['episode'] == e and r['seed'] == 42][0],
                     'learned_envelope': [round(min(Lall), 4), round(max(Lall), 4)],
                     'T1_envelope': [round(min(T1), 4), round(max(T1), 4)],
                     'envelopes_overlap_T1': min(Lall) < max(T1),
                     'T1f_envelope': [round(min(T1f), 4), round(max(T1f), 4)],
                     'envelopes_overlap_T1f': min(Lall) < max(T1f)}
OUT['Q3b_is_oracle_an_upper_bound'] = beat

# ---------------------------------------------------------------- Q2-d -45% 헤드라인 재현 + 계량 정체
wsb, wsa, cdm = {}, {}, {}
for e in CROWDED:
    w = [r for r in ws if r['episode'] == e and r['err'] is None]
    c = [r['cond_polish_nrmse_Hs'] for r in cond if r['episode'] == e and r['err'] is None]
    if w:
        wsb[e] = st.median([min(r['init_nrmse'], r['avg_nRMSE']) for r in w])
        wsa[e] = st.median([r['avg_nRMSE'] for r in w])
    if c:
        cdm[e] = st.median(c)
com = [e for e in CROWDED if e in wsb and e in cdm]
a0i = {e: st.median([r['init_nrmse'] for r in today if r['arm'] == 'S1-A0' and r['episode'] == e]) for e in EPS}
t1i = {e: st.median([r['init_nrmse'] for r in today if r['arm'] == 'S2-T1f' and r['episode'] == e]) for e in EPS}
OUT['Q2d_headline_reproduction_and_metric_identity'] = {
    'reproduced_on': len(com),
    'ws_best_min_init_tracked_mean': round(st.mean([wsb[e] for e in com]), 4),
    'ws_tracked_mean': round(st.mean([wsa[e] for e in com]), 4),
    'cond_mean': round(st.mean([cdm[e] for e in com]), 4),
    'pct': round(100 * (st.mean([cdm[e] for e in com]) - st.mean([wsb[e] for e in com])) / st.mean([wsb[e] for e in com]), 1),
    'wins': f"{sum(1 for e in com if cdm[e] < wsb[e])}/{len(com)}",
    'metric_identity': {
        'cond_arm': 'cond_polish_nrmse_Hs = 3 nowcast 프레임 각각에서 새 관측창으로 재적합(BMA-of-4 + 80-iter polish)한 필드 nRMSE 의 평균. 추적기 미경유.',
        'ws_arm': 'min(init_nrmse, avg_nRMSE): init 은 init 창 1프레임 스냅샷, avg 는 추적 전 프레임 시간평균. 추적기 경유.',
        'consequence': '배치 품질과 재적합 주기가 교락. 오늘 §4.4 가 잰 추적기 2.6배 인플레가 이 교락의 크기 척도.'},
    'snapshot_level_on_the4': {
        'A0_init_mean4': round(st.mean([a0i[e] for e in EPS]), 4),
        'T1f_init_mean4': round(st.mean([t1i[e] for e in EPS]), 4),
        'cond_mean4': round(st.mean([cdm[e] for e in EPS]), 4),
        'cond_recovers_pct_of_oracle_snapshot_gain': round(
            100 * (st.mean([a0i[e] for e in EPS]) - st.mean([cdm[e] for e in EPS])) /
            (st.mean([a0i[e] for e in EPS]) - st.mean([t1i[e] for e in EPS])), 1)}}

# ---------------------------------------------------------------- Q4 층화 상속
sup = json.load(open('tmp/cdip_support_2026-08-03/cdip_support_raw.json'))['synthetic']
rows = [(int(re.search(r'_(\d+)comp_', s['episode']).group(1)), s['spread_total_deg']) for s in sup]
byn = {}
for n, sp in rows:
    byn.setdefault(n, []).append(sp)
def best_thr(cut):
    best = (None, 0.0)
    for t in sorted(set(round(sp, 4) for _, sp in rows)):
        acc = sum((sp >= t) == (n >= cut) for n, sp in rows) / len(rows)
        if acc > best[1]:
            best = (t, acc)
    return best
crowd = [sp for n, sp in rows if n >= 15]
spar = [sp for n, sp in rows if n < 15]
OUT['Q4_stratification_inheritance'] = {
    'spread_by_ncomp': {n: {'n': len(v), 'min': round(min(v), 3), 'med': round(st.median(v), 3),
                            'max': round(max(v), 3)} for n, v in sorted(byn.items())},
    'identity_Nge5': {'best_threshold_deg': best_thr(5)[0], 'accuracy': best_thr(5)[1]},
    'identity_Nge15_WavePrior_crowded_cut': {'best_threshold_deg': best_thr(15)[0], 'accuracy': best_thr(15)[1]},
    'overlap_5comp_vs_15comp': {'5comp_med': round(st.median(byn[5]), 3), '15comp_med': round(st.median(byn[15]), 3),
                                '5comp_range': [round(min(byn[5]), 3), round(max(byn[5]), 3)],
                                '15comp_range': [round(min(byn[15]), 3), round(max(byn[15]), 3)]},
    'crowded_vs_sparse_spread_separable': max(spar) < min(crowd),
    'N_eff_gamma3p3_from_depth3': {'1': 1.0, '4': 2.38, '5': 2.56, '6': 1.97, '7': 1.62, '8': 1.87,
                                   '9': 2.97, '10': 3.64, '12': 2.72, '15': 4.56, '30': 7.82},
    'source_N_eff': 'tmp/depth3_ceiling_2026-08-05/CON_VERDICT.md + CON_followups.json'}

# ---------------------------------------------------------------- 부수: 검정력/분산
lanes = {}
for e in EPS:
    ds = []
    for lane in (42, 1, 2, 3, 4):
        c = [r['avg_nRMSE'] for r in wpi if r['arm'] == 'v2_konly' and r['episode'] == e and r['est_seed'] == lane][0]
        l = [r['avg_nRMSE'] for r in wpi if r['arm'] == 'learned' and r['episode'] == e and r['est_seed'] == lane][0]
        ds.append(round(c - l, 4))
    lanes[e[8:-3]] = {'deltas': ds, 'median': round(st.median(ds), 4), 'wins': f"{sum(1 for d in ds if d > 0)}/5"}
allp = [d for v in lanes.values() for d in v['deltas']]
rngs = {}
for e in EPS:
    f = lambda rows, key, pred: (lambda v: round(max(v) - min(v), 4))([r[key] for r in rows if pred(r)])
    rngs[e[8:-3]] = {
        'v2_konly_5lane': f(wpi, 'avg_nRMSE', lambda r: r['arm'] == 'v2_konly' and r['episode'] == e),
        'learned_5lane': f(wpi, 'avg_nRMSE', lambda r: r['arm'] == 'learned' and r['episode'] == e),
        'S1A0_3seed': f(today, 'nRMSE', lambda r: r['arm'] == 'S1-A0' and r['episode'] == e),
        'S2T1f_3seed': f(today, 'nRMSE', lambda r: r['arm'] == 'S2-T1f' and r['episode'] == e)}
OUT['Q_power_and_variance'] = {
    'per_lane_paired_delta_on_the4': lanes,
    'pooled20': {'median': round(st.median(allp), 4), 'wins': f"{sum(1 for d in allp if d > 0)}/20",
                 'sd': round(st.stdev(allp), 4), 'effect_over_sd': round(st.median(allp) / st.stdev(allp), 2)},
    'seed_range_per_cell': rngs,
    's1s2_prereg_noise_floor_seed_range_median': 0.0379}

json.dump(OUT, open('tmp/waveprior_debate_2026-08-05/NEUTRAL_reanalysis.json', 'w'),
          indent=1, ensure_ascii=False, default=float)
print(json.dumps(OUT, indent=1, ensure_ascii=False, default=float))
