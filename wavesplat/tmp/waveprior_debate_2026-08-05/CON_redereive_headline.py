"""CON-5: independent re-derivation of the WavePrior integration headline + the placement ablation,
straight from arm_runs.jsonl (dual-verification directive). No new runs."""
import json, statistics as st, itertools, math
HERE='/home/vexit/wavesplat/wavesplat/tmp'
wp={}
for l in open(f'{HERE}/wp_integration_2026-07-25/arm_runs.jsonl'):
    d=json.loads(l)
    if d.get('err'): continue
    wp.setdefault(d['arm'],{}).setdefault(d['episode'],{})[(d['est_seed'],d['diff_seed'])]=d
CROWDED=[f'episode_BF{bf}_{nc}comp_seed{s}_{a}.h5' for bf in (4,5) for nc in (15,30)
         for s in range(5) for a in ('uniform','slope')]
def med(arm,ep,k='avg_nRMSE'): return st.median([v[k] for v in wp[arm][ep].values()])
def sign_p(w,n):   # two-sided exact sign test
    return min(1.0, 2*sum(math.comb(n,i) for i in range(w,n+1))/2**n)

print('=== 40 crowded, median-of-5, tracked avg_nRMSE ===')
for a in ['v2_konly','learned','learned_konly']:
    print(f"  {a:14s} mean {st.mean([med(a,e) for e in CROWDED]):.4f}")
for A,B in [('learned','v2_konly'),('learned_konly','v2_konly'),('learned_konly','learned')]:
    dl=[med(A,e)-med(B,e) for e in CROWDED]
    w=sum(1 for x in dl if x<0)
    print(f"  {A:14s} vs {B:14s}: medD {st.median(dl):+.4f}  A-better {w}/40  two-sided sign p {sign_p(max(w,40-w),40):.2e}")

print('\n=== the record\'s own mechanism claim, re-derived: "the win is kx/ky, not LS w/phi" ===')
dl=[med('learned_konly',e)-med('learned',e) for e in CROWDED]
print(f"  learned_konly - learned: medD {st.median(dl):+.4f}  konly-better {sum(1 for x in dl if x<0)}/40 "
      f"-> the LS w/phi contributes {'NOTHING measurable' if abs(st.median(dl))<0.01 else 'something'}")
print("  (learned_konly = learned kx/ky + v2-style w=0.01/random phi)")

print('\n=== but WHICH episodes carry the gain? (the 4 oracle episodes vs the rest) ===')
ORACLE4=['episode_BF4_15comp_seed0_uniform.h5','episode_BF4_30comp_seed0_uniform.h5',
         'episode_BF5_15comp_seed0_uniform.h5','episode_BF5_30comp_seed0_uniform.h5']
for tag,eps in [('the 4 oracle eps',ORACLE4),('other 36 crowded',[e for e in CROWDED if e not in ORACLE4]),
                ('uniform 20',[e for e in CROWDED if 'uniform' in e]),('slope 20',[e for e in CROWDED if 'slope' in e])]:
    dl=[med('learned',e)-med('v2_konly',e) for e in eps]
    print(f"  {tag:18s} n={len(eps):2d}  learned {st.mean([med('learned',e) for e in eps]):.4f} "
          f"v2 {st.mean([med('v2_konly',e) for e in eps]):.4f}  medD {st.median(dl):+.4f}  wins {sum(1 for x in dl if x<0)}/{len(eps)}")
print("  => the oracle's 4 episodes are all seed0_uniform, the WEAKEST slice for the learned arm.")

print('\n=== 5-metric panel on the 40 crowded (re-derived) ===')
for k in ['avg_nRMSE','avg_Variance_Ratio','avg_Pearson_rho','avg_Skill_Score','avg_Directional_R2','avg_Hs_error']:
    a=st.mean([med('learned',e,k) for e in CROWDED]); b=st.mean([med('v2_konly',e,k) for e in CROWDED])
    print(f"  {k:22s} learned {a:+.4f}  v2 {b:+.4f}  d {a-b:+.4f}")

print('\n=== final_n: does the learned arm change CARDINALITY as well as coordinates? ===')
for a in ['learned','v2_konly','learned_konly']:
    fs=[med(a,e,'final_n') for e in CROWDED]
    print(f"  {a:14s} final_n median {st.median(fs):.0f}  range {min(fs):.0f}-{max(fs):.0f}")
n15=[e for e in CROWDED if '15comp' in e]; n30=[e for e in CROWDED if '30comp' in e]
for a in ['learned','v2_konly']:
    print(f"  {a:14s} 15comp median final_n {st.median([med(a,e,'final_n') for e in n15]):.0f} (N_gt 15) | "
          f"30comp {st.median([med(a,e,'final_n') for e in n30]):.0f} (N_gt 30)")
