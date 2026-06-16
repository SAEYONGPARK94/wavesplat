#!/usr/bin/env python3
"""
Direction A 최종 검증 스크립트 (P1-1 ~ P3-6)
- P1-1: GP vs PINO 통계적 유의성 (paired t-test + Wilcoxon)
- P1-2: 해상상태별 GP vs PINO 분해
- P1-3: WaveSplat VR=0.914 이상치 원인 분석
- P2-4: Oracle GP 30-ep 재현 확인
- P2-5: Per-episode catastrophic failure 목록
- P3-6: Ablation 완결성 (ms5/reg_amp/ms5+reg_amp)
"""
import json
import numpy as np
from scipy import stats
from collections import defaultdict

RESULTS_DIR = 'wavesplat/results'

def load_json(name):
    with open(f'{RESULTS_DIR}/{name}') as f:
        return json.load(f)

def extract_per_ep_metric(data, metric='nRMSE'):
    """Extract ordered per-episode metric values."""
    eps = sorted(data['per_episode'].keys())
    return eps, [data['per_episode'][e][metric] for e in eps]

def sea_state_from_name(name):
    """e.g. 'BF4_15comp_seed2_uniform' -> 'BF4_15c'"""
    parts = name.replace('.h5', '').replace('episode_', '').split('_')
    bf = parts[0]
    comp = parts[1].replace('comp', 'c')
    return f"{bf}_{comp}"

# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("Direction A 최종 검증")
print("=" * 80)

gp_full = load_json('gp_disp_full_eval.json')
pino = load_json('pino_s4_30ep_eval.json')

gp_data = gp_full['ws_ms5_reg_amp']
oracle_data = gp_full['oracle']
ws_data = gp_full['wavesplat']
ms5_data = gp_full['ws_multistart']
reg_data = gp_full['ws_reg_amp']

# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 80)
print("P1-1: GP vs PINO 통계적 유의성 검정")
print("─" * 80)

gp_eps, gp_nrmse = extract_per_ep_metric(gp_data, 'nRMSE')
pino_eps, pino_nrmse = extract_per_ep_metric(pino, 'nRMSE')

# Match episodes
common_eps = sorted(set(gp_eps) & set(pino_eps))
gp_matched = [gp_data['per_episode'][e]['nRMSE'] for e in common_eps]
pino_matched = [pino['per_episode'][e]['nRMSE'] for e in common_eps]
diff = [g - p for g, p in zip(gp_matched, pino_matched)]

print(f"공통 에피소드: {len(common_eps)}")
print(f"GP mean nRMSE:   {np.mean(gp_matched):.4f} ± {np.std(gp_matched):.4f}")
print(f"PINO mean nRMSE: {np.mean(pino_matched):.4f} ± {np.std(pino_matched):.4f}")
print(f"Mean diff (GP-PINO): {np.mean(diff):.4f} ± {np.std(diff):.4f}")

# Paired t-test
t_stat, p_ttest = stats.ttest_rel(gp_matched, pino_matched)
print(f"\nPaired t-test: t={t_stat:.3f}, p={p_ttest:.4f}")

# Wilcoxon signed-rank
try:
    w_stat, p_wilcoxon = stats.wilcoxon(gp_matched, pino_matched)
    print(f"Wilcoxon signed-rank: W={w_stat:.0f}, p={p_wilcoxon:.4f}")
except Exception as e:
    print(f"Wilcoxon: {e}")

# Win/loss count
gp_wins = sum(1 for d in diff if d < 0)
pino_wins = sum(1 for d in diff if d > 0)
ties = sum(1 for d in diff if d == 0)
print(f"\nGP wins: {gp_wins}/{len(diff)}, PINO wins: {pino_wins}/{len(diff)}, ties: {ties}")

# Effect size (Cohen's d for paired)
d_cohen = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0
print(f"Cohen's d (paired): {d_cohen:.3f}")

if p_ttest < 0.05:
    winner = "GP" if np.mean(diff) < 0 else "PINO"
    print(f"→ 결론: {winner}가 유의하게 우수 (p={p_ttest:.4f} < 0.05)")
else:
    print(f"→ 결론: 유의한 차이 없음 (p={p_ttest:.4f} >= 0.05)")

# Also test other metrics
for metric in ['VR', 'Pearson', 'Skill']:
    gp_m = [gp_data['per_episode'][e][metric] for e in common_eps]
    pino_m = [pino['per_episode'][e][metric] for e in common_eps]
    t, p = stats.ttest_rel(gp_m, pino_m)
    gp_mean = np.mean(gp_m)
    pino_mean = np.mean(pino_m)
    print(f"  {metric}: GP={gp_mean:.4f} vs PINO={pino_mean:.4f}, t={t:.2f}, p={p:.4f}"
          f" {'*' if p < 0.05 else ''}")


# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 80)
print("P1-2: 해상상태별 GP vs PINO 분해 비교")
print("─" * 80)

# Group by sea state
by_ss = defaultdict(lambda: {'gp': [], 'pino': []})
for e in common_eps:
    ss = sea_state_from_name(e)
    by_ss[ss]['gp'].append(gp_data['per_episode'][e]['nRMSE'])
    by_ss[ss]['pino'].append(pino['per_episode'][e]['nRMSE'])

print(f"{'Sea State':<14} {'GP nRMSE':>10} {'PINO nRMSE':>12} {'Δ(GP-PINO)':>12} {'GP wins':>8}")
print("-" * 60)
for ss in sorted(by_ss.keys()):
    gp_m = np.mean(by_ss[ss]['gp'])
    pi_m = np.mean(by_ss[ss]['pino'])
    delta = gp_m - pi_m
    gp_w = sum(1 for g, p in zip(by_ss[ss]['gp'], by_ss[ss]['pino']) if g < p)
    n = len(by_ss[ss]['gp'])
    print(f"{ss:<14} {gp_m:>10.4f} {pi_m:>12.4f} {delta:>+12.4f} {gp_w:>5}/{n}")

# Complexity analysis: simple (5c) vs complex (15c, 30c)
simple_gp = [gp_data['per_episode'][e]['nRMSE'] for e in common_eps if '5comp' in e]
simple_pi = [pino['per_episode'][e]['nRMSE'] for e in common_eps if '5comp' in e]
complex_gp = [gp_data['per_episode'][e]['nRMSE'] for e in common_eps if '15comp' in e or '30comp' in e]
complex_pi = [pino['per_episode'][e]['nRMSE'] for e in common_eps if '15comp' in e or '30comp' in e]

print(f"\n단순 해상 (5comp, n={len(simple_gp)}): GP={np.mean(simple_gp):.4f} vs PINO={np.mean(simple_pi):.4f}"
      f" Δ={np.mean(simple_gp)-np.mean(simple_pi):+.4f}")
print(f"복잡 해상 (15-30c, n={len(complex_gp)}): GP={np.mean(complex_gp):.4f} vs PINO={np.mean(complex_pi):.4f}"
      f" Δ={np.mean(complex_gp)-np.mean(complex_pi):+.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 80)
print("P1-3: WaveSplat VR=0.914 이상치 원인 분석")
print("─" * 80)

ws_eps, ws_vr = extract_per_ep_metric(ws_data, 'VR')
ws_nrmse_list = [ws_data['per_episode'][e]['nRMSE'] for e in ws_eps]

vr_over1 = [(e, ws_data['per_episode'][e]['VR'], ws_data['per_episode'][e]['nRMSE'])
            for e in ws_eps if ws_data['per_episode'][e]['VR'] > 1.0]

print(f"WaveSplat 전체 평균 VR: {np.mean(ws_vr):.4f}")
print(f"VR > 1.0 에피소드 수: {len(vr_over1)}/{len(ws_eps)}")
print(f"VR median: {np.median(ws_vr):.4f}")
print(f"VR 25th-75th percentile: [{np.percentile(ws_vr, 25):.3f}, {np.percentile(ws_vr, 75):.3f}]")

if vr_over1:
    print(f"\nVR > 1.0 에피소드 (진폭 과대추정):")
    for e, vr, nr in sorted(vr_over1, key=lambda x: -x[1])[:10]:
        print(f"  {e:<40} VR={vr:.3f} nRMSE={nr:.3f}")

# Compare with GP VR
gp_vr = [gp_data['per_episode'][e]['VR'] for e in ws_eps if e in gp_data['per_episode']]
ws_vr_matched = [ws_data['per_episode'][e]['VR'] for e in ws_eps if e in gp_data['per_episode']]
print(f"\nVR 비교 (matched episodes):")
print(f"  WaveSplat: mean={np.mean(ws_vr_matched):.3f}, >1.0 count={sum(1 for v in ws_vr_matched if v>1.0)}")
print(f"  GP K_disp: mean={np.mean(gp_vr):.3f}, >1.0 count={sum(1 for v in gp_vr if v>1.0)}")

# Check if VR>1 correlates with sea state
vr_by_ss = defaultdict(list)
for e in ws_eps:
    ss = sea_state_from_name(e)
    vr_by_ss[ss].append(ws_data['per_episode'][e]['VR'])

print(f"\n해상상태별 WaveSplat VR:")
for ss in sorted(vr_by_ss.keys()):
    vals = vr_by_ss[ss]
    print(f"  {ss:<14} mean={np.mean(vals):.3f} >1.0={sum(1 for v in vals if v>1.0)}/{len(vals)}")


# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 80)
print("P2-4: Oracle GP 30-ep 결과 확인")
print("─" * 80)

oracle_eps, oracle_nrmse = extract_per_ep_metric(oracle_data, 'nRMSE')
print(f"Oracle 에피소드 수: {len(oracle_eps)}")
print(f"Oracle nRMSE: {np.mean(oracle_nrmse):.4f} ± {np.std(oracle_nrmse):.4f}")

oracle_metrics = {}
for metric in ['nRMSE', 'VR', 'Pearson', 'Skill']:
    vals = [oracle_data['per_episode'][e][metric] for e in oracle_eps]
    oracle_metrics[metric] = np.mean(vals)
    print(f"  {metric}: {np.mean(vals):.4f}")

# Verify against reported value
reported_nrmse = 0.126
actual_nrmse = np.mean(oracle_nrmse)
print(f"\n보고된 oracle nRMSE: {reported_nrmse}")
print(f"JSON에서 계산한 값:  {actual_nrmse:.4f}")
if abs(actual_nrmse - reported_nrmse) < 0.001:
    print("→ 일치 ✓")
else:
    print(f"→ 불일치! 차이={actual_nrmse - reported_nrmse:+.4f}")

# Check if all 30 episodes present
expected_eps = set()
for seed in range(5):
    for ss in ['BF3_5comp', 'BF4_5comp', 'BF4_15comp', 'BF4_30comp', 'BF5_15comp', 'BF5_30comp']:
        expected_eps.add(f"{ss}_seed{seed}_uniform")

actual_eps = set(oracle_eps)
missing = expected_eps - actual_eps
extra = actual_eps - expected_eps
print(f"\n기대 에피소드: {len(expected_eps)}, 실제: {len(actual_eps)}")
if missing:
    print(f"누락: {missing}")
if extra:
    print(f"추가: {extra}")
if not missing and not extra:
    print("→ 30 에피소드 완전 일치 ✓")


# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 80)
print("P2-5: Per-episode catastrophic failure 목록 (nRMSE > 0.4)")
print("─" * 80)

THRESHOLD = 0.4

methods = {
    'GP ws_ms5_reg_amp': gp_data,
    'WavePINO S4': pino,
    'WaveSplat': ws_data,
    'Oracle': oracle_data,
}

for name, data in methods.items():
    eps_list = sorted(data['per_episode'].keys())
    failures = [(e, data['per_episode'][e]['nRMSE'])
                for e in eps_list if data['per_episode'][e]['nRMSE'] > THRESHOLD]
    vals = [data['per_episode'][e]['nRMSE'] for e in eps_list]
    print(f"\n{name} (n={len(eps_list)}, mean={np.mean(vals):.4f}):")
    if failures:
        print(f"  Catastrophic failures (nRMSE > {THRESHOLD}): {len(failures)}")
        for e, nr in sorted(failures, key=lambda x: -x[1]):
            print(f"    {e:<40} nRMSE={nr:.4f}")
    else:
        print(f"  Catastrophic failures: 0")
    # Also report max and worst-case
    worst_e = max(eps_list, key=lambda e: data['per_episode'][e]['nRMSE'])
    print(f"  Worst: {worst_e} nRMSE={data['per_episode'][worst_e]['nRMSE']:.4f}")

# Robustness: std and max
print(f"\nRobustness 비교:")
print(f"{'Method':<25} {'Mean':>7} {'Std':>7} {'Max':>7} {'Min':>7} {'Max-Mean':>9}")
for name, data in methods.items():
    eps_list = sorted(data['per_episode'].keys())
    vals = [data['per_episode'][e]['nRMSE'] for e in eps_list]
    print(f"{name:<25} {np.mean(vals):>7.4f} {np.std(vals):>7.4f} {np.max(vals):>7.4f} "
          f"{np.min(vals):>7.4f} {np.max(vals)-np.mean(vals):>9.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 80)
print("P3-6: Ablation 완결성 (ms5 / reg_amp / ms5+reg_amp)")
print("─" * 80)

ablation_methods = {
    'WaveSplat (baseline)': ws_data,
    'ws_multistart (ms5)': ms5_data,
    'ws_reg_amp': reg_data,
    'ws_ms5_reg_amp': gp_data,
    'Oracle': oracle_data,
}

# Verify all have 30 episodes
print("에피소드 수 확인:")
for name, data in ablation_methods.items():
    n = len(data['per_episode'])
    print(f"  {name:<25}: {n} episodes {'✓' if n == 30 else '✗ MISSING'}")

print(f"\n{'Method':<25} {'nRMSE':>8} {'VR':>7} {'Pearson':>9} {'Skill':>8}")
print("-" * 62)
for name, data in ablation_methods.items():
    eps_list = sorted(data['per_episode'].keys())
    avg = {m: np.mean([data['per_episode'][e][m] for e in eps_list])
           for m in ['nRMSE', 'VR', 'Pearson', 'Skill']}
    print(f"{name:<25} {avg['nRMSE']:>8.4f} {avg['VR']:>7.3f} {avg['Pearson']:>9.3f} {avg['Skill']:>8.3f}")

# Ablation deltas
print(f"\nAblation deltas (nRMSE 감소 = 개선):")
ws_mean = np.mean([ws_data['per_episode'][e]['nRMSE'] for e in sorted(ws_data['per_episode'].keys())])
for name, data in ablation_methods.items():
    if name == 'WaveSplat (baseline)' or name == 'Oracle':
        continue
    eps_list = sorted(data['per_episode'].keys())
    avg_nr = np.mean([data['per_episode'][e]['nRMSE'] for e in eps_list])
    delta_pct = (avg_nr - ws_mean) / ws_mean * 100
    print(f"  {name:<25}: {avg_nr:.4f} ({delta_pct:+.1f}% vs WaveSplat)")

# Check synergy: ms5+reg_amp < min(ms5, reg_amp)?
ms5_mean = np.mean([ms5_data['per_episode'][e]['nRMSE'] for e in sorted(ms5_data['per_episode'].keys())])
reg_mean = np.mean([reg_data['per_episode'][e]['nRMSE'] for e in sorted(reg_data['per_episode'].keys())])
combo_mean = np.mean([gp_data['per_episode'][e]['nRMSE'] for e in sorted(gp_data['per_episode'].keys())])
print(f"\n시너지 분석:")
print(f"  ms5={ms5_mean:.4f}, reg_amp={reg_mean:.4f}, ms5+reg_amp={combo_mean:.4f}")
print(f"  min(ms5, reg_amp)={min(ms5_mean, reg_mean):.4f}")
synergy = min(ms5_mean, reg_mean) - combo_mean
print(f"  시너지 = min - combo = {synergy:+.4f} ({'양의 시너지 ✓' if synergy > 0 else '시너지 없음'})")

# Per-episode ablation: how often does ms5+reg_amp beat both components?
common_abl = sorted(set(ms5_data['per_episode'].keys()) &
                    set(reg_data['per_episode'].keys()) &
                    set(gp_data['per_episode'].keys()))
combo_best = 0
for e in common_abl:
    c = gp_data['per_episode'][e]['nRMSE']
    m = ms5_data['per_episode'][e]['nRMSE']
    r = reg_data['per_episode'][e]['nRMSE']
    if c <= min(m, r):
        combo_best += 1
print(f"  ms5+reg_amp가 ms5와 reg_amp 모두 이긴 에피소드: {combo_best}/{len(common_abl)}")


# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("전체 검증 결과 요약")
print("=" * 80)

results = {}

# Save structured results
results['P1_1_statistical_test'] = {
    'n_episodes': len(common_eps),
    'gp_mean': float(np.mean(gp_matched)),
    'pino_mean': float(np.mean(pino_matched)),
    'paired_ttest': {'t': float(t_stat), 'p': float(p_ttest)},
    'gp_wins': gp_wins,
    'pino_wins': pino_wins,
    'cohens_d': float(d_cohen),
}

results['P1_2_by_sea_state'] = {}
for ss in sorted(by_ss.keys()):
    results['P1_2_by_sea_state'][ss] = {
        'gp': float(np.mean(by_ss[ss]['gp'])),
        'pino': float(np.mean(by_ss[ss]['pino'])),
        'delta': float(np.mean(by_ss[ss]['gp']) - np.mean(by_ss[ss]['pino'])),
    }

results['P1_3_wavesplat_vr'] = {
    'mean_vr': float(np.mean(ws_vr)),
    'median_vr': float(np.median(ws_vr)),
    'vr_above_1_count': len(vr_over1),
    'total_episodes': len(ws_eps),
}

results['P2_4_oracle_verification'] = {
    'n_episodes': len(oracle_eps),
    'nRMSE': float(np.mean(oracle_nrmse)),
    'reported': reported_nrmse,
    'match': bool(abs(actual_nrmse - reported_nrmse) < 0.001),
    'all_30_present': bool(len(missing) == 0 and len(extra) == 0),
}

results['P2_5_catastrophic'] = {}
for name, data in methods.items():
    eps_list = sorted(data['per_episode'].keys())
    vals = [data['per_episode'][e]['nRMSE'] for e in eps_list]
    failures = [e for e in eps_list if data['per_episode'][e]['nRMSE'] > THRESHOLD]
    results['P2_5_catastrophic'][name] = {
        'n_failures': len(failures),
        'max_nrmse': float(np.max(vals)),
        'std_nrmse': float(np.std(vals)),
    }

results['P3_6_ablation'] = {
    'synergy': float(synergy),
    'combo_best_count': combo_best,
    'combo_best_total': len(common_abl),
    'all_30_episodes': bool(all(len(d['per_episode']) == 30 for d in ablation_methods.values())),
}

with open(f'{RESULTS_DIR}/final_verification.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n결과 저장: {RESULTS_DIR}/final_verification.json")
