import json, os, statistics as st
from math import comb
H=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def sgn(w,n): return sum(comb(n,k) for k in range(w,n+1))/2**n
rows=[json.loads(l) for l in open(H+'/results_b4.jsonl')]
for f in ('results_b3.jsonl','results_b3b.jsonl'): rows+=[json.loads(l) for l in open(H+'/'+f)]
ok=[r for r in rows if 'error' not in r]
tp=json.load(open(H+'/rv4/true_persist.json'))
K=lambda r:(r['record'],r['win'])
per={}
for r in ok: per.setdefault(r['cell'],{})[K(r)]=r['per_lead']
keys=sorted(per['FROZEN']); leads=[float(x) for x in sorted(per['FROZEN'][keys[0]],key=float)]
T=lambda k,l,f: tp[f'{k[0]}|{k[1]}']['per_lead'][str(l)][f]
def fc(getter, lo=2.0, hi=14.0):
    return {k: st.mean([getter(k,l) for l in leads if lo<=l<=hi]) for k in keys}
def cmp(A,B,nm):
    w=sum(1 for k in keys if A[k]<B[k]); t=sum(1 for k in keys if A[k]==B[k])
    rel=st.median([(A[k]-B[k])/B[k]*100 for k in keys])
    print(f"  {nm:34s} {w:2d}/36 (ties {t})  rel_med {rel:+7.2f}%  p={sgn(w,36):.3g}")
    return w
print('=== ★1. 사전등록 PRIMARY 를 "아무것도 예측하지 않기" 가 더 잘 통과한다 ===')
Z=fc(lambda k,l: T(k,l,'zero')); P=fc(lambda k,l: per['PERSIST'][k][str(l)])
F=fc(lambda k,l: per['FROZEN'][k][str(l)]); R=fc(lambda k,l: per['REFIT2'][k][str(l)])
cmp(F,P,'FROZEN  < PERSIST (사전등록)')
cmp(Z,P,'★ZERO-FIELD < PERSIST (음성대조)')
cmp(R,P,'REFIT2  < PERSIST (SECONDARY)')
print('\n=== ★2. 레코드 단위(12) 접기 ===')
for nm,A in (('FROZEN<PERSIST',F),('ZERO<PERSIST',Z),('REFIT2<PERSIST',R)):
    recs=sorted({k[0] for k in keys})
    w=sum(1 for rc in recs if all(A[k]<P[k] for k in keys if k[0]==rc))
    print(f"  {nm:18s} 전창승 레코드 {w}/12  p={sgn(w,12):.4g}")
print('\n=== ★3. 지평 제한: 어디까지가 진짜 기술인가 (vs 참 제로필드) ===')
for lo,hi in ((0,3),(1,3),(2,3),(4,14),(5,14)):
    A=fc(lambda k,l: per['FROZEN'][k][str(l)],lo,hi); B=fc(lambda k,l: T(k,l,'zero'),lo,hi)
    C=fc(lambda k,l: per['PERSIST'][k][str(l)],lo,hi)
    print(f"  lead {lo}-{hi}: FROZEN<ZERO {sum(1 for k in keys if A[k]<B[k]):2d}/36 "
          f"p={sgn(sum(1 for k in keys if A[k]<B[k]),36):.3g} | "
          f"FROZEN<PERSIST {sum(1 for k in keys if A[k]<C[k]):2d}/36 | "
          f"ZERO<PERSIST {sum(1 for k in keys if B[k]<C[k]):2d}/36")
print('\n=== ★4. B-3b "기술수명 3-4 s" 다중성: Holm over 75 검정(5팔 x 15 lead) ===')
tests=[]
for arm in ('FROZEN','REFIT10','REFIT5','REFIT2','REFIT1'):
    for l in leads:
        w=sum(1 for k in keys if per[arm][k][str(l)]<T(k,l,'zero'))
        tests.append((sgn(w,36),arm,int(l),w))
tests.sort(); m=len(tests); alive=True; res={}
for i,(p,a,l,w) in enumerate(tests):
    al=0.05/(m-i)
    if p>al: alive=False
    res[(a,l)]= (p<=al and alive)
fr=[l for l in map(int,leads) if res[('FROZEN',l)]]
print(f"  m={m}. FROZEN 이 Holm-75 를 통과하는 lead = {fr}  (첫 실패 lead = {min(set(map(int,leads))-set(fr))})")
for a in ('REFIT2','REFIT1','REFIT5','REFIT10'):
    print(f"  {a}: 통과 lead {len([l for l in map(int,leads) if res[(a,l)]])}/15")
print('\n=== ★5. G-2 강한 대조 대체 — PERSIST 예측장 불변성의 근거 ===')
print('  refit_calls:', sorted({r['refit_calls'] for r in ok if r['cell']=='PERSIST'}),
      '| init_fit_calls:', sorted({r['init_fit_calls'] for r in ok if r['cell']=='PERSIST'}),
      '| hash_uniform:', sum(1 for r in ok if r['cell']=='PERSIST' and r['persist_field_hash_uniform']),'/36')
print('  lead0 PERSIST==FROZEN:', sum(1 for k in keys if per['PERSIST'][k]['0.0']==per['FROZEN'][k]['0.0']),'/36')

print('\n=== ★6. 독립성 사다리 (RV-3 처방) — 레코드/사이트 접기, 두 규약 ===')
recs=sorted({k[0] for k in keys})
site=lambda r: r[:2] if r[:2] in ('AA','BS','LJ','YS') else r[:2]
sites=sorted({site(r) for r in recs})
def ladder(A,B,nm):
    w_all=sum(1 for rc in recs if all(A[k]<B[k] for k in keys if k[0]==rc))
    w_mean=sum(1 for rc in recs if st.mean([A[k] for k in keys if k[0]==rc])<st.mean([B[k] for k in keys if k[0]==rc]))
    s_mean=sum(1 for s in sites if st.mean([A[k] for k in keys if site(k[0])==s])<st.mean([B[k] for k in keys if site(k[0])==s]))
    print(f"  {nm:20s} 창 {sum(1 for k in keys if A[k]<B[k])}/36 p={sgn(sum(1 for k in keys if A[k]<B[k]),36):.3g} | "
          f"레코드(전창승) {w_all}/12 p={sgn(w_all,12):.4g} | 레코드(평균) {w_mean}/12 p={sgn(w_mean,12):.4g} | "
          f"사이트(평균) {s_mean}/4 p={sgn(s_mean,4):.4g}")
ladder(F,P,'FROZEN<PERSIST'); ladder(Z,P,'ZERO<PERSIST'); ladder(R,P,'REFIT2<PERSIST')
ladder(F,Z,'FROZEN<ZERO'); ladder(R,Z,'REFIT2<ZERO')
print('\n=== ★7. 다중성 가족 — B-4 는 같은 12 레코드의 몇 번째인가 ===')
import glob,os
print(' ', sorted(os.path.basename(p) for p in glob.glob(H+'/../B2_REAL_2026-08-23/results*.jsonl')+glob.glob(H+'/results*.jsonl')))
print('  gates_b3.FAMILY_ALPHA = 0.05/7 = %.5f ; B-4 는 이 분모를 갱신하지 않았다'%(0.05/7))
