"""CON-1: era comparability + direct 4-episode overlap between WavePrior (07-25) and oracle (08-05).
No new wave runs. Pure re-analysis of existing jsonl."""
import json, statistics as st, os
HERE = '/home/vexit/wavesplat/wavesplat/tmp'
EPS = ['episode_BF4_15comp_seed0_uniform.h5','episode_BF4_30comp_seed0_uniform.h5',
       'episode_BF5_15comp_seed0_uniform.h5','episode_BF5_30comp_seed0_uniform.h5']

# --- today's oracle raw ---
s1s2 = {}
for l in open(f'{HERE}/s1s2_2026-08-05/results.jsonl'):
    d = json.loads(l)
    s1s2.setdefault((d['arm'], d['episode']), {})[d['seed']] = d

# --- WavePrior integration raw ---
wp = {}
for l in open(f'{HERE}/wp_integration_2026-07-25/arm_runs.jsonl'):
    d = json.loads(l)
    if d.get('err'): continue
    wp.setdefault((d['arm'], d['episode']), {})[(d['est_seed'], d['diff_seed'])] = d

out = {}

# 1. seed-42 lane identity: wp v2_konly (07-25) vs today's S1-A0
print('=== 1. ERA CHECK: v2_konly seed42 (2026-07-25 wp_integration) vs S1-A0 seed42 (2026-08-05) ===')
rows = []
for ep in EPS:
    a = wp[('v2_konly', ep)][(42, 0)]
    b = s1s2[('S1-A0', ep)][42]
    rows.append(dict(ep=ep, wp0725=a['avg_nRMSE'], s1s2_0805=b['nRMSE'],
                     d=a['avg_nRMSE']-b['nRMSE'],
                     wp_init=a['init_nrmse'], s_init=b['init_nrmse'],
                     wp_finaln=a['final_n'], s_finaln=b['N']))
    print(f"  {ep[8:-3]:28s} 07-25={a['avg_nRMSE']:.10f}  08-05={b['nRMSE']:.10f}  d={a['avg_nRMSE']-b['nRMSE']:+.2e}  "
          f"init {a['init_nrmse']:.6f}/{b['init_nrmse']:.6f}  N {a['final_n']}/{b['N']}")
out['era_seed42'] = rows

# 2. medians on the 4 overlap episodes
def med_wp(arm, ep, key='avg_nRMSE'):
    return st.median([v[key] for v in wp[(arm, ep)].values()])
def med_s(arm, ep, key='nRMSE'):
    return st.median([v[key] for v in s1s2[(arm, ep)].values()])

print('\n=== 2. SAME 4 EPISODES: learned vs oracle ===')
hdr = f"{'episode':28s} {'v2(07-25)':>10s} {'v2(08-05)':>10s} {'learned':>9s} {'lrn_konly':>9s} {'orcl T1f':>9s} {'orcl+notrk':>10s} {'LETKF':>8s}"
print(hdr)
LETKF = {'episode_BF4_15comp_seed0_uniform.h5':0.2282,'episode_BF4_30comp_seed0_uniform.h5':0.2347,
         'episode_BF5_15comp_seed0_uniform.h5':0.2298,'episode_BF5_30comp_seed0_uniform.h5':0.2651}
tab = []
for ep in EPS:
    r = dict(ep=ep,
             v2_0725=med_wp('v2_konly', ep), v2_0805=med_s('S1-A0', ep),
             learned=med_wp('learned', ep), learned_konly=med_wp('learned_konly', ep),
             oracle=med_s('S2-T1f', ep), oracle_notrack=med_s('X-T1f-notrack', ep),
             letkf=LETKF[ep])
    tab.append(r)
    print(f"{ep[8:-3]:28s} {r['v2_0725']:10.4f} {r['v2_0805']:10.4f} {r['learned']:9.4f} "
          f"{r['learned_konly']:9.4f} {r['oracle']:9.4f} {r['oracle_notrack']:10.4f} {r['letkf']:8.4f}")
m = lambda k: st.mean([r[k] for r in tab])
print(f"{'MEAN4':28s} {m('v2_0725'):10.4f} {m('v2_0805'):10.4f} {m('learned'):9.4f} "
      f"{m('learned_konly'):9.4f} {m('oracle'):9.4f} {m('oracle_notrack'):10.4f} {m('letkf'):8.4f}")
out['overlap_table'] = tab

# 3. recovery fraction under 3 different denominators
print('\n=== 3. "RECOVERY FRACTION" under different (defensible) denominators ===')
for r in tab:
    base = r['v2_0805']          # today's current arm, same seeds set? (median-of-3 vs median-of-5)
    base25 = r['v2_0725']
    gain = base25 - r['learned']
    for name, ceil in [('oracle_tracked', r['oracle']), ('oracle_notrack', r['oracle_notrack'])]:
        head = base25 - ceil
        print(f"  {r['ep'][8:-3]:26s} vs {name:15s}: learned gain {gain:+.4f} / headroom {head:+.4f} "
              f"= {100*gain/head if head!=0 else float('nan'):6.1f}%")
g25 = m('v2_0725')
print(f"  MEAN4: learned recovers {100*(g25-m('learned'))/(g25-m('oracle')):.1f}% of oracle-TRACKED headroom, "
      f"{100*(g25-m('learned'))/(g25-m('oracle_notrack')):.1f}% of oracle-NOTRACK headroom")

json.dump(out, open(f'{HERE}/waveprior_debate_2026-08-05/CON_era_and_overlap.json','w'), indent=1)
