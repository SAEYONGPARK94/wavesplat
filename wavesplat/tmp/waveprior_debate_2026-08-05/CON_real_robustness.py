"""CON-4: does the recorded 'real-data contribution = collapse robustness' survive re-derivation?
Raw: tmp/wp_real_2026-07-25/wp_real.json (30 windows, 5 records, hand bank vs learned proposal)."""
import json, statistics as st
d=json.load(open('/home/vexit/wavesplat/wavesplat/tmp/wp_real_2026-07-25/wp_real.json'))
R=d['results']
print(f"n windows {len(R)}  records {sorted(set(r['record'] for r in R))}")
coll=[r for r in R if r['collapse']]; norm=[r for r in R if not r['collapse']]
print(f"collapse-flagged {len(coll)}  normal {len(norm)}\n")

def blk(rows,tag):
    h=[r['hand_now'] for r in rows]; l=[r['learned_now'] for r in rows]
    dl=[a-b for a,b in zip(l,h)]
    w=sum(1 for x in dl if x<0)
    print(f"{tag:22s} n={len(rows):2d}  hand med {st.median(h):.3f} mean {st.mean(h):.3f} MAX {max(h):.3f}  |  "
          f"learned med {st.median(l):.3f} mean {st.mean(l):.3f} MAX {max(l):.3f}  |  "
          f"medD {st.median(dl):+.3f}  learned wins {w}/{len(rows)}")
blk(R,'ALL 30'); blk(norm,'NORMAL 27'); blk(coll,'COLLAPSE 3')
print('\ncollapse windows:')
for r in coll:
    print(f"  {r['record']}_w{r['win']}: hand {r['hand_now']:.3f} -> learned {r['learned_now']:.3f} "
          f"(sd {r['learned_now_sd']:.3f}, seeds {[round(x,3) for x in r['learned_now_seeds']]})")

print('\n--- ROBUSTNESS AUDIT: is "learned never collapses" a distributional fact or a 3-point one? ---')
ho=sorted([r['hand_now'] for r in R]); lo=sorted([r['learned_now'] for r in R])
for q,lab in [(-1,'max'),(-2,'2nd worst'),(-3,'3rd worst'),(-4,'4th worst')]:
    print(f"  {lab:10s}: hand {ho[q]:.3f}   learned {lo[q]:.3f}")
print(f"  windows with nRMSE>1.0 : hand {sum(1 for x in ho if x>1)}/30   learned {sum(1 for x in lo if x>1)}/30")
print(f"  windows with nRMSE>0.9 : hand {sum(1 for x in ho if x>0.9)}/30   learned {sum(1 for x in lo if x>0.9)}/30")
print(f"  std across windows     : hand {st.pstdev(ho):.3f}   learned {st.pstdev(lo):.3f}")
print(f"  IQR                    : hand {ho[22]-ho[7]:.3f}   learned {lo[22]-lo[7]:.3f}")

print('\n--- record-unit sign test on the 27 normal windows (the prereg unit problem) ---')
import collections
byrec=collections.defaultdict(list)
for r in norm: byrec[r['record']].append(r['learned_now']-r['hand_now'])
for k,v in sorted(byrec.items()):
    print(f"  {k:8s} n={len(v)}  medD {st.median(v):+.3f}  {'learned better' if st.median(v)<0 else 'hand better'}")
nrec=len(byrec); nl=sum(1 for v in byrec.values() if st.median(v)<0)
print(f"  records where learned better: {nl}/{nrec}  (two-sided sign-test floor p = {2*0.5**nrec:.4f})")

print('\n--- lead (forecast) axis, same windows ---')
for tag,rows in [('ALL 30',R),('NORMAL 27',norm),('COLLAPSE 3',coll)]:
    h=[r['hand_lead'] for r in rows]; l=[r['learned_lead'] for r in rows]
    dl=[a-b for a,b in zip(l,h)]
    print(f"  {tag:11s} hand_lead med {st.median(h):.3f} MAX {max(h):.3f} | learned_lead med {st.median(l):.3f} MAX {max(l):.3f} | medD {st.median(dl):+.3f} wins {sum(1 for x in dl if x<0)}/{len(rows)}")

print('\n--- zero-predictor reference: nRMSE of a flat sea ~ 0.25 on synthetic; on real these are Hs-normalised too')
print(f"  fraction of real windows where BOTH arms exceed 0.9: {sum(1 for r in R if r['hand_now']>0.9 and r['learned_now']>0.9)}/30")
