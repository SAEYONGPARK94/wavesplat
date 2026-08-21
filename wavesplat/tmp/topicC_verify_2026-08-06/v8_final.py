import json, statistics as st, math
R1=[json.loads(l) for l in open('tmp/rx1_tracker_2026-08-05/results2.jsonl')]
R2=[json.loads(l) for l in open('tmp/rx2_init_2026-08-05/results.jsonl')]
EPS=sorted(set(r['episode'] for r in R1))
print("=== wall_s medians ===")
for lbl,R,cells in [('rx1',R1,['D-BASE','D-NOTRACK','X-BASE','X-NOTRACK']),('rx2',R2,['AUTO','G1P1','X-BASE','ORACLE'])]:
    for c in cells:
        w=[r['wall_s'] for r in R if r['cell']==c]
        print(f"  {lbl} {c:10s} wall_s median={st.median(w):.1f}s  n={len(w)}")
tot=sum(r['wall_s'] for r in R1)
print(f"  rx1 total wall_s across 180 runs = {tot/3600:.2f} CPU-h")
dep=[r['wall_s'] for r in R1 if r['cell'].startswith('D-')]
print(f"  rx1 deploy-cell wall_s median = {st.median(dep):.1f}s (n={len(dep)})")

print()
print("=== NEUTRAL: sqrt(m) frame trajectory ===")
def sm(rho,v): return math.sqrt(max(0.0,(1-rho**2)+(math.sqrt(max(v,0))-rho)**2))
for c in ['X-BASE','X-NOTRACK','D-BASE','D-NOTRACK','K4']:
    rows=[r for r in R1 if r['cell']==c]
    nF=len(rows[0]['rho_frames'])
    def at(i): return st.mean([sm(r['rho_frames'][i],r['vr_frames'][i]) for r in rows])
    print(f"  {c:10s} nF={nF} f1={at(0):.3f} f40={at(min(39,nF-1)):.3f} f120={at(min(119,nF-1)):.3f} f_last={at(nF-1):.3f}")
# diff D-NOTRACK - D-BASE in m by frame block
db={r['job_id'].split('|',1)[1]:r for r in R1 if r['cell']=='D-BASE'}
dn={r['job_id'].split('|',1)[1]:r for r in R1 if r['cell']=='D-NOTRACK'}
def mval(rho,v): return (1-rho**2)+(math.sqrt(max(v,0))-rho)**2
blocks=[(0,30),(30,60),(60,90),(90,120)]
print("  Delta m (D-NOTRACK - D-BASE) by frame block:")
for a,b in blocks:
    d=[st.mean([mval(dn[k]['rho_frames'][i],dn[k]['vr_frames'][i])-mval(db[k]['rho_frames'][i],db[k]['vr_frames'][i]) for i in range(a,b)]) for k in db]
    print(f"    frames {a+1}-{b}: {st.mean(d):+.4f}")
print("  X-BASE vs X-NOTRACK mean sqrt(m) difference:", end=' ')
xb={r['job_id'].split('|',1)[1]:r for r in R1 if r['cell']=='X-BASE'}
xn={r['job_id'].split('|',1)[1]:r for r in R1 if r['cell']=='X-NOTRACK'}
print(f"{st.mean([st.mean([sm(xb[k]['rho_frames'][i],xb[k]['vr_frames'][i]) for i in range(160)])-st.mean([sm(xn[k]['rho_frames'][i],xn[k]['vr_frames'][i]) for i in range(160)]) for k in xb]):+.4f}")

print()
print("=== cell-level decorr/shrink/rho/sqrt(v) (median of 12) ===")
for c in ['D-BASE','D-NOTRACK']:
    rows=[r for r in R1 if r['cell']==c]
    print(f"  {c:10s} decorr={st.median([r['decorr_framemean'] for r in rows]):.4f} shrink={st.median([r['shrink_framemean'] for r in rows]):.4f} "
          f"rho={st.median([st.mean(r['rho_frames']) for r in rows]):.4f} sqrtv={st.median([math.sqrt(st.mean(r['vr_frames'])) for r in rows]):.4f} "
          f"m={st.median([r['m_framemean'] for r in rows]):.4f} nRMSE={st.median([r['nRMSE'] for r in rows]):.4f}")
    d=st.median([r['decorr_framemean'] for r in rows]); m=st.median([r['m_framemean'] for r in rows]); n=st.median([r['nRMSE'] for r in rows])
    print(f"      sqrt(decorr/m)={math.sqrt(d/m):.4f} -> rescaled nRMSE {n:.4f} -> {n*math.sqrt(d/m):.4f}   decorr/m share={100*d/m:.1f}%")

print()
print("=== rx2 decorr share of residual (CON '93.8-96.3%') ===")
for c in ['X-BASE','AUTO','G1P1','ORACLE']:
    rows=[r for r in R2 if r['cell']==c]
    sh=[r['decorr_mean']/r['m_mean'] for r in rows]
    print(f"  {c:9s} decorr/m median={100*st.median(sh):.1f}%  range {100*min(sh):.1f}-{100*max(sh):.1f}%")
