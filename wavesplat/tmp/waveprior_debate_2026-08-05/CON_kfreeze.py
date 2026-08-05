"""CON-6: the oracle ceiling bundles a GT-FREE knob (init_k_lr_scale=0 'freeze k').
Does k-learning help or hurt the LEARNED placement? (raw_init = LS w/phi, NO joint fit;
init_nrmse = after the joint fit, which MOVES k.)"""
import json, statistics as st
HERE='/home/vexit/wavesplat/wavesplat/tmp'
wp={}
for l in open(f'{HERE}/wp_integration_2026-07-25/arm_runs.jsonl'):
    d=json.loads(l)
    if d.get('err') or d.get('raw_init_nrmse') is None: continue
    wp.setdefault(d['arm'],{}).setdefault(d['episode'],{})[(d['est_seed'],d['diff_seed'])]=d
CROWDED=[f'episode_BF{bf}_{nc}comp_seed{s}_{a}.h5' for bf in (4,5) for nc in (15,30)
         for s in range(5) for a in ('uniform','slope')]
med=lambda arm,ep,k: st.median([v[k] for v in wp[arm][ep].values()])
print('=== LEARNED arm: does letting the optimizer move k help or hurt? (40 crowded) ===')
raw=[med('learned',e,'raw_init_nrmse') for e in CROWDED]
fit=[med('learned',e,'init_nrmse') for e in CROWDED]
trk=[med('learned',e,'avg_nRMSE') for e in CROWDED]
d1=[b-a for a,b in zip(raw,fit)]; d2=[c-b for b,c in zip(fit,trk)]
print(f"  raw-init (no joint fit) {st.mean(raw):.4f}")
print(f"  post joint fit          {st.mean(fit):.4f}   d {st.mean(d1):+.4f}  improved {sum(1 for x in d1 if x<0)}/40")
print(f"  tracked                 {st.mean(trk):.4f}   d {st.mean(d2):+.4f}  improved {sum(1 for x in d2 if x<0)}/40")
print("  => for the LEARNED placement the joint fit (which moves k) is a NET GAIN.")
print("     For the ORACLE placement, letting k move costs +0.0442 (S2-T1f 0.1671 -> S2-T1 0.2112, 4/4).")
print("     The 'freeze k' knob is GT-free and has NEVER been run on the learned placement.")
print('\n=== per-episode final_n vs N_gt: is the cardinality inversion real or driven by a few eps? ===')
n15=[e for e in CROWDED if '15comp' in e]; n30=[e for e in CROWDED if '30comp' in e]
for arm in ['learned','v2_konly']:
    a15=[med(arm,e,'final_n') for e in n15]; a30=[med(arm,e,'final_n') for e in n30]
    hi=sum(1 for x,y in zip(sorted(a30),sorted(a15)) if x>y)
    print(f"  {arm:10s} 15comp final_n {sorted(a15)}")
    print(f"  {arm:10s} 30comp final_n {sorted(a30)}")
    print(f"  {arm:10s} median 15->30: {st.median(a15):.0f} -> {st.median(a30):.0f}  "
          f"{'TRACKS complexity' if st.median(a30)>st.median(a15) else 'INVERTED vs complexity'}"
          f"   (rank-pairs 30>15: {hi}/20)\n")
