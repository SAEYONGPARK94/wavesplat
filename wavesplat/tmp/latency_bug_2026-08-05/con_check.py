"""CON verification: (a) universal relative bias prediction, (b) arm symmetry, (c) ratio survival."""
import json, re, numpy as np
STEP = 1263.9/1000.0   # MY measured mean step (1251.6, 1276.1 ms)
PERIOD = 32.52         # MY measured inter-event gap
PRED = STEP/PERIOD
print(f"[MEASURED-BY-ME] step={STEP*1000:.1f} ms  period={PERIOD:.2f} s")
print(f"[DERIVED] universal expected relative bias on ANY mean-of-time.time()-deltas = {PRED*100:.3f}%")
print(f"          -> correction factor x{1/(1-PRED):.4f}  (arm-independent, N-independent, m-independent)\n")

NF = {'BF2':220,'BF3':190,'BF4':160,'BF5':120}
def sea(e): return re.search(r'BF\d', e).group(0)

# --- WS arms: reconstruct clean by ADDING BACK detected steps (not by dropping) ---
for path in ['tmp/rerun_2026-07-25/batchC_ws_phasefix.jsonl',
             'tmp/rerun_2026-07-24/ep100_runs.jsonl']:
    rows=[json.loads(l) for l in open(path)]
    rows=[r for r in rows if isinstance(r.get('elapsed_ms'),(int,float))]
    G={}
    for r in rows: G.setdefault(r['episode'],[]).append(r)
    raw=[]; rep=[]; nev=0; T=0.0
    for ep,rs in G.items():
        n=NF[sea(ep)]; sh=1220.0/n
        med=np.median([r['elapsed_ms'] for r in rs])
        for r in rs:
            v=r['elapsed_ms']; k=int(round(max(0.0,(med-v)/sh)))
            if (med-v) < 0.5*sh: k=0
            nev+=k; raw.append(v); rep.append(v+k*sh)
            T += (v+k*sh)*n/1000.0     # timed seconds in that run
    raw=np.array(raw); rep=np.array(rep)
    exp_ev = T/PERIOD
    print(f"{path.split('/')[-1]}")
    print(f"   raw_mean={raw.mean():.3f}  repaired_mean={rep.mean():.3f}  "
          f"bias={(raw.mean()-rep.mean())/rep.mean()*100:+.2f}%   (predicted {-PRED*100:+.2f}%)")
    print(f"   events detected={nev}  expected(Poisson)={exp_ev:.1f}  "
          f"+-{np.sqrt(exp_ev):.1f}  timed_wall={T:.0f}s")
