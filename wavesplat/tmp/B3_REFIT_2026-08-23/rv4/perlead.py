import json, os, statistics as st
from math import comb
HERE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def sgn(w,n): return sum(comb(n,k) for k in range(w,n+1))/2**n
rows=[json.loads(l) for l in open(HERE+'/results_b4.jsonl')]
for f in ('results_b3.jsonl','results_b3b.jsonl'):
    rows+= [json.loads(l) for l in open(HERE+'/'+f)]
ok=[r for r in rows if 'error' not in r]
zero=json.load(open(HERE+'/true_zero_perlead.json'))
cells=sorted({r['cell'] for r in ok})
print('cells:',cells, {c:sum(1 for r in ok if r['cell']==c) for c in cells})
K=lambda r:(r['record'],r['win'])
per={}
for r in ok:
    per.setdefault(r['cell'],{})[K(r)]=r.get('per_lead') or {}
leads=[float(x) for x in sorted(per['FROZEN'][('AA14',1)], key=float)]
keys=sorted(per['FROZEN'])
print('\n=== A. per-lead: FROZEN vs PERSIST vs REFIT2, and vs TRUE ZERO ===')
print(f"{'lead':>5} {'medZERO':>8} {'medPER':>8} {'medFRO':>8} {'medRE2':>8} {'F>P':>6} {'R>P':>6} {'F>0':>6} {'pF>0':>9} {'R2>0':>6}")
rowsout=[]
for l in leads:
    s=str(l)
    z ={k: zero[f'{k[0]}|{k[1]}'][s] for k in keys}
    P ={k: per['PERSIST'][k][s] for k in keys}
    F ={k: per['FROZEN'][k][s] for k in keys}
    R ={k: per['REFIT2'][k][s] for k in keys}
    fp=sum(1 for k in keys if F[k]<P[k]); rp=sum(1 for k in keys if R[k]<P[k])
    fz=sum(1 for k in keys if F[k]<z[k]); rz=sum(1 for k in keys if R[k]<z[k])
    pz=sum(1 for k in keys if P[k]<z[k])
    rowsout.append((l,fz,pz,rz,fp,rp))
    print(f"{l:5.0f} {st.median(z.values()):8.3f} {st.median(P.values()):8.3f} {st.median(F.values()):8.3f} {st.median(R.values()):8.3f} {fp:3d}/36 {rp:3d}/36 {fz:3d}/36 {sgn(fz,36):9.5f} {rz:3d}/36")
print('\nPERSIST beats true-zero per lead:', [(int(l),pz) for l,fz,pz,rz,fp,rp in rowsout])
print('\n=== B. Holm across the 15 per-lead tests (FROZEN vs true zero) ===')
ps=sorted([(sgn(fz,36),int(l),fz) for l,fz,pz,rz,fp,rp in rowsout])
m=len(ps); rej=True
for i,(p,l,w) in enumerate(ps):
    a=0.05/(m-i)
    if p>a: rej=False
    print(f"  lead {l:2d}  {w:2d}/36  p={p:.6f}  Holm-alpha={a:.5f}  {'PASS' if (p<=a and rej) else 'FAIL'}")
print('\n=== C. crossover audit ===')
for l,fz,pz,rz,fp,rp in rowsout:
    print(f"  lead {l:4.1f}: FROZEN better than PERSIST in {fp}/36  (analyzer fires at >18 and l>0)")
