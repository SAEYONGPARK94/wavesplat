"""E-28 red-team audit: re-derive all headline numbers + power/TOST analysis."""
import json
import numpy as np
from scipy import stats

def paired(d, label, margins=(0.02, 0.03, 0.05)):
    d = np.asarray(d, float)
    n = len(d)
    m = d.mean()
    sd = d.std(ddof=1)
    se = sd / np.sqrt(n)
    tc95 = stats.t.ppf(0.975, n - 1)
    tc90 = stats.t.ppf(0.95, n - 1)
    ci95 = (m - tc95 * se, m + tc95 * se)
    ci90 = (m - tc90 * se, m + tc90 * se)
    tstat = m / se if se > 0 else np.nan
    p = 2 * stats.t.sf(abs(tstat), n - 1)
    win = int((d > 0).sum())
    # MDE at 80% power, two-sided alpha=.05, given observed sd
    mde = (tc95 + stats.t.ppf(0.80, n - 1)) * se
    print(f"\n[{label}] n={n} mean={m:+.4f} sd={sd:.4f} se={se:.4f} "
          f"t={tstat:+.2f} p={p:.3f} win={win}/{n}")
    print(f"  95% CI [{ci95[0]:+.4f},{ci95[1]:+.4f}]  "
          f"90% CI [{ci90[0]:+.4f},{ci90[1]:+.4f}]  MDE(80%)={mde:.4f}")
    for dm in margins:
        # TOST: equivalent within +-dm iff 90% CI inside [-dm, +dm]
        eq = ci90[0] > -dm and ci90[1] < dm
        print(f"  TOST margin +-{dm}: {'EQUIVALENT' if eq else 'NOT established'}")
    # one-sided: what gain is excluded at 95%?
    print(f"  gains > {m + tc90 * se:+.4f} excluded (one-sided 95%)")
    return m, se, ci95, ci90

print("=" * 70)
print("A design (lambda_gate 0.02), results/direction3/e28_gennet_loro.json")
J = json.load(open('results/direction3/e28_gennet_loro.json'))
pooled = J['pooled']; extrap = J['extrap']
d_pool = [r['gn'] - r['cur'] for r in pooled]
d_ext = [r['gn'] - r['cur'] for r in extrap]
det = [r['det'] for r in pooled]
order = np.argsort(det)
k = len(pooled) // 3
under = [pooled[i] for i in order[:k]]
well = [pooled[i] for i in order[-k:]]
print("per-sea gn-cur:", np.round(d_pool, 4).tolist())
print("det order (sea idx low->high):", order.tolist())
paired(d_pool, "A pooled gn-cur")
paired([r['gn'] - r['cur'] for r in under], "A under-det tercile gn-cur (n=3!)")
paired(d_ext, "A extrap gn-cur")
# counter-evidence: does a learned prior help on low-det seas at all?
for name, rows in [('under', under), ('well', well), ('pooled', pooled)]:
    cc = np.mean([r['cur'] - r['cl'] for r in rows])
    gg = np.mean([r['gn'] - r['cl'] for r in rows])
    print(f"  {name}: cur-classical {cc:+.4f} | gn-classical {gg:+.4f} "
          f"(challenger recovers {100*gg/cc if cc else np.nan:.0f}% of incumbent edge)")
# writeup check
print("\nwriteup vs rederived: pooled -0.015 ->", round(np.mean(d_pool), 3),
      "| win 4/10 ->", int(np.sum(np.array(d_pool) > 0)),
      "| under -0.028 ->", round(np.mean([r['gn'] - r['cur'] for r in under]), 3),
      "| extrap -0.027 ->", round(np.mean(d_ext), 3))
print("d_well(gn-cl) json", round(J['d_well'], 3),
      "rederived", round(np.mean([r['gn'] - r['cl'] for r in well]), 3))
print("gate_corr json", round(J['gate_corr'], 3))
ap = [r['ap'] for r in pooled]
print("alpha_prior mean/std:", round(np.mean(ap), 3), round(np.std(ap), 4))

print("\n" + "=" * 70)
print("A design lambda_gate 0.10 (e28_gennet_loro_lg10.json)")
J10 = json.load(open('results/direction3/e28_gennet_loro_lg10.json'))
d10 = [r['gn'] - r['cur'] for r in J10['pooled']]
paired(d10, "A lg10 pooled gn-cur")
det10 = [r['det'] for r in J10['pooled']]
o10 = np.argsort(det10)
u10 = [J10['pooled'][i] for i in o10[:3]]
paired([r['gn'] - r['cur'] for r in u10], "A lg10 under tercile (n=3)")
print("writeup: pooled -0.036 win 2/10, under -0.045, gate_corr -0.21")
print("rederived:", round(np.mean(d10), 3), int(np.sum(np.array(d10) > 0)),
      round(np.mean([r['gn'] - r['cur'] for r in u10]), 3),
      round(J10['gate_corr'], 3))
# same eval seas in both runs?
same = all(abs(a['cur'] - b['cur']) < 1e-12 and abs(a['cl'] - b['cl']) < 1e-12
           for a, b in zip(J['pooled'], J10['pooled']))
print("lg02 and lg10 evaluated on IDENTICAL seas (cl,cur bit-equal)?", same)

print("\n" + "=" * 70)
print("B design (e28_loro.json): back out se from stored 90% t-CI, n=24")
B = json.load(open('results/direction3/e28_loro.json'))
tc90_23 = stats.t.ppf(0.95, 23)
tc95_23 = stats.t.ppf(0.975, 23)
for name, r in B.items():
    se = (r['ci'][1] - r['ci'][0]) / (2 * tc90_23)
    m = r['d']
    ci95 = (m - tc95_23 * se, m + tc95_23 * se)
    mde = (tc95_23 + stats.t.ppf(0.80, 23)) * se
    p_sign = 2 * stats.binom.cdf(min(r['win'], 24 - r['win']), 24, 0.5)
    print(f"\n[B {name}] d={m:+.4f} se={se:.4f} 90%CI[{r['ci'][0]:+.4f},"
          f"{r['ci'][1]:+.4f}] 95%CI[{ci95[0]:+.4f},{ci95[1]:+.4f}] "
          f"win {r['win']}/24 (sign-test p={p_sign:.4f}) MDE(80%)={mde:.4f}")
    for dm in (0.02, 0.03, 0.05):
        lo90 = m - tc90_23 * se; hi90 = m + tc90_23 * se
        eq = lo90 > -dm and hi90 < dm
        print(f"  TOST +-{dm}: {'EQUIVALENT' if eq else 'NOT established'}")
    print(f"  vs_hand {r['vs_hand']:+.4f}  film {r['film']:.3f} cur {r['cur']:.3f} "
          f"hand {r['hand']:.3f}")

print("\n" + "=" * 70)
print("A eval-sea hidden metadata: replicate build_sea rng draws (U, nfrac, bw)")
def hashabs(s):
    return sum(ord(ch) for ch in s) % 97
REG = {'spread': dict(hs=(1.5, 3.0), U=(0.0, 1.5), nf=(0.02, 0.09), bw=(0.0, 0.2)),
       'extrap': dict(hs=(4.0, 5.0), U=(1.5, 1.5), nf=(0.04, 0.08), bw=(0.3, 0.5))}
print(f"{'sea':>10s} {'Hs':>6s} {'U/cp':>6s} {'nfrac':>6s} {'in-train U<=0.43?':>18s}")
for regime, ns in [('spread', 10), ('extrap', 5)]:
    R = REG[regime]
    for si in range(ns):
        rng = np.random.default_rng(7000 + 31 * si + hashabs(regime))
        Hs = rng.uniform(*R['hs'])
        mdir = rng.uniform(0, 360)          # direction (consumed before U)
        U = rng.uniform(*R['U'])
        nfrac = rng.uniform(*R['nf'])
        print(f"{regime}{si:>4d} {Hs:6.2f} {U:6.2f} {nfrac:6.3f} "
              f"{'IN-DIST' if U <= 0.43 else 'held-out':>18s}")
