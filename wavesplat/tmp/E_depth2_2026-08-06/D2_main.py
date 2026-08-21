"""Depth-2 for topic E: mechanism / confound / power / null-control.

PRE-SPECIFIED CONVENTIONS (fixed before any number was computed):
  unit           = episode
  lane -> episode: MEDIAN of the 5 (est_seed,diff_seed) lanes  [PRIMARY]
                   MEAN as sensitivity only
  episode -> set : PAIRED delta, MEDIAN  [PRIMARY]; MEAN as sensitivity
  test           = paired TWO-SIDED sign test (ties dropped) [PRIMARY]
                   two-sided Wilcoxon as sensitivity
  direction      = negative delta means learned/auto is better
  null           = flat-sea (zero predictor) per episode, from D2_nulls.json
Run from /home/vexit/wavesplat/wavesplat.
"""
import json, itertools, math
import numpy as np
from scipy.stats import binomtest, wilcoxon, spearmanr

JUL = 'tmp/wp_integration_2026-07-25/arm_runs.jsonl'
HO_AUTO = 'tmp/heldout40_2026-07-26/arm_runs_auto_live.jsonl'
HO_V2 = 'tmp/heldout40_2026-07-26/arm_runs_v2_control.jsonl'
NULLS = json.load(open('tmp/E_depth2_2026-08-06/D2_nulls.json'))

CROWDED = [f'episode_BF{bf}_{nc}comp_seed{s}_{arm}.h5'
           for bf in (4, 5) for nc in (15, 30) for s in range(5)
           for arm in ('uniform', 'slope')]
SPARSE = ['episode_BF2_1comp_seed0_uniform.h5', 'episode_BF2_1comp_seed0_slope.h5',
          'episode_BF3_1comp_seed0_uniform.h5', 'episode_BF3_2comp_seed0_uniform.h5',
          'episode_BF3_2comp_seed0_slope.h5', 'episode_BF3_5comp_seed0_uniform.h5',
          'episode_BF3_5comp_seed1_uniform.h5', 'episode_BF3_5comp_seed0_slope.h5',
          'episode_BF4_1comp_seed0_uniform.h5', 'episode_BF4_5comp_seed0_uniform.h5',
          'episode_BF4_5comp_seed0_slope.h5', 'episode_BF4_5comp_seed1_uniform.h5']


def load(path):
    out = []
    for line in open(path):
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def lanes(rows, arm=None, field='avg_nRMSE'):
    """-> {episode: {(est,diff): value}}"""
    d = {}
    for r in rows:
        if arm is not None and r.get('arm') != arm:
            continue
        v = r.get(field)
        if v is None:
            continue
        d.setdefault(r['episode'], {})[(r['est_seed'], r['diff_seed'])] = float(v)
    return d


def agg(lane_map, how='median'):
    f = np.median if how == 'median' else np.mean
    return {e: float(f(list(v.values()))) for e, v in lane_map.items()}


def signtest(deltas, direction='two-sided'):
    d = [x for x in deltas if abs(x) > 0]
    neg = sum(1 for x in d if x < 0)
    if not d:
        return dict(n=0, wins=0, p=float('nan'))
    p = binomtest(neg, len(d), 0.5, alternative=direction).pvalue
    return dict(n=len(d), wins=neg, p=float(p))


def mde(sd, n, alpha=0.05, power=0.80):
    return (1.959963985 + 0.8416212336) * sd / math.sqrt(n)


def summarize(tag, deltas, base=None):
    a = np.array(deltas, float)
    st = signtest(a)
    w2 = wilcoxon(a, alternative='two-sided').pvalue if len(a) > 5 else float('nan')
    w1 = wilcoxon(a, alternative='less').pvalue if len(a) > 5 else float('nan')
    rel_med = 100 * np.median(a) / base if base else float('nan')
    print(f"  {tag:38s} n={len(a):3d} medΔ={np.median(a):+.4f} meanΔ={np.mean(a):+.4f} "
          f"rel={rel_med:+6.1f}% wins={st['wins']}/{st['n']} signp={st['p']:.3g} "
          f"wilcox2={w2:.3g} wilcox1={w1:.3g} sd={a.std(ddof=1):.4f} "
          f"MDE80={mde(a.std(ddof=1), len(a)):.4f}")
    return dict(n=len(a), med=float(np.median(a)), mean=float(np.mean(a)),
                wins=st['wins'], nz=st['n'], p=st['p'], sd=float(a.std(ddof=1)))


jul = load(JUL)
print("=" * 100)
print("SECTION 1 — NULL CONTROL (flat-sea zero predictor), July wp_integration")
print("=" * 100)
for how in ('median', 'mean'):
    v2 = agg(lanes(jul, 'v2_konly'), how)
    lr = agg(lanes(jul, 'learned'), how)
    for name, eps in (('CROWDED', CROWDED), ('SPARSE', SPARSE)):
        nl = {e: NULLS['data_finemesh_2026-07-23/' + e]['null_avg_nRMSE'] for e in eps}
        print(f"[lane {how}] {name}: null med={np.median(list(nl.values())):.4f}")
        for arm, m in (('v2_konly', v2), ('learned', lr)):
            d = [m[e] - nl[e] for e in eps]
            print(f"     arm={arm:9s} armMed={np.median([m[e] for e in eps]):.4f} "
                  f"ratio_to_null={np.median([m[e]/nl[e] for e in eps]):.3f}", end=' ')
            summarize(f"{arm} minus NULL", d)
print()

print("=" * 100)
print("SECTION 2 — STRATUM CONFOUND: is the crowded win a stratum effect?")
print("=" * 100)
v2 = agg(lanes(jul, 'v2_konly'))
lr = agg(lanes(jul, 'learned'))
delt = {e: lr[e] - v2[e] for e in CROWDED}


def stratum(e):
    p = e.replace('episode_', '').replace('.h5', '').split('_')
    return p[0], p[1], p[3]          # BF, ncomp, raydrop


print("  per-stratum (BF x ncomp):")
for bf in ('BF4', 'BF5'):
    for nc in ('15comp', '30comp'):
        eps = [e for e in CROWDED if stratum(e)[0] == bf and stratum(e)[1] == nc]
        summarize(f"{bf}/{nc}", [delt[e] for e in eps],
                  base=np.median([v2[e] for e in eps]))
print("  per-stratum (raydrop):")
for rd in ('uniform', 'slope'):
    eps = [e for e in CROWDED if stratum(e)[2] == rd]
    summarize(f"raydrop={rd}", [delt[e] for e in eps], base=np.median([v2[e] for e in eps]))
print("  leave-one-stratum-out (drop the named stratum, keep the rest):")
for bf in ('BF4', 'BF5'):
    for nc in ('15comp', '30comp'):
        eps = [e for e in CROWDED if not (stratum(e)[0] == bf and stratum(e)[1] == nc)]
        summarize(f"drop {bf}/{nc}", [delt[e] for e in eps],
                  base=np.median([v2[e] for e in eps]))
print()

print("=" * 100)
print("SECTION 3 — ALTERNATIVE MECHANISMS (regression to the mean / N / null level)")
print("=" * 100)
v2i = agg(lanes(jul, 'v2_konly', 'init_nrmse'))
lri = agg(lanes(jul, 'learned', 'init_nrmse'))
v2n = agg(lanes(jul, 'v2_konly', 'final_n'))
lrn = agg(lanes(jul, 'learned', 'final_n'))
v2r = agg(lanes(jul, 'v2_konly', 'raw_init_nrmse'))
lrr = agg(lanes(jul, 'learned', 'raw_init_nrmse'))
eps = CROWDED
d_avg = np.array([delt[e] for e in eps])
d_init = np.array([lri[e] - v2i[e] for e in eps])
d_n = np.array([lrn[e] - v2n[e] for e in eps])
base_v2 = np.array([v2[e] for e in eps])
nullv = np.array([NULLS['data_finemesh_2026-07-23/' + e]['null_avg_nRMSE'] for e in eps])
hs = np.array([NULLS['data_finemesh_2026-07-23/' + e]['Hs'] for e in eps])
print(f"  crowded Δinit_nrmse : med {np.median(d_init):+.4f}  "
      f"Δavg_nRMSE med {np.median(d_avg):+.4f}  "
      f"ratio(init share) {np.median(d_init)/np.median(d_avg):.2f}")
print(f"  crowded Δfinal_n    : med {np.median(d_n):+.1f}  "
      f"v2 med {np.median([v2n[e] for e in eps]):.1f} learned med {np.median([lrn[e] for e in eps]):.1f}")
for nm, x in (('v2 baseline level', base_v2), ('flat-sea null level', nullv),
              ('Hs', hs), ('Δfinal_n', d_n), ('Δinit_nrmse', d_init)):
    rho, p = spearmanr(x, d_avg)
    print(f"  Spearman(Δavg_nRMSE, {nm:20s}) = {rho:+.3f}  p={p:.4f}")
# does the gain survive normalizing each episode by its own null?
d_rel = np.array([(lr[e] - v2[e]) / NULLS['data_finemesh_2026-07-23/' + e]['null_avg_nRMSE']
                  for e in eps])
summarize("Δ normalized by episode null", d_rel)
# partial check: split by whether v2 beat its own null
hi = [e for e in eps if v2[e] > NULLS['data_finemesh_2026-07-23/' + e]['null_avg_nRMSE']]
lo = [e for e in eps if v2[e] <= NULLS['data_finemesh_2026-07-23/' + e]['null_avg_nRMSE']]
print(f"  episodes where v2 is WORSE than flat sea: {len(hi)}/40 ; better: {len(lo)}/40")
if hi:
    summarize("Δ | v2 worse than null", [delt[e] for e in hi])
if lo:
    summarize("Δ | v2 better than null", [delt[e] for e in lo])
print()

print("=" * 100)
print("SECTION 4 — COLLAPSE-LANE DOMINANCE (does one bad lane drive the arm means?)")
print("=" * 100)
lv2, llr = lanes(jul, 'v2_konly'), lanes(jul, 'learned')
for name, eps2 in (('CROWDED', CROWDED), ('SPARSE', SPARSE)):
    for arm, L in (('v2_konly', lv2), ('learned', llr)):
        ncol = 0
        tot = 0
        for e in eps2:
            vals = np.array(list(L[e].values()))
            m = np.median(vals)
            ncol += int(np.sum(vals > 3 * m))
            tot += len(vals)
        print(f"  {name:8s} {arm:9s} collapse lanes (>3x episode median): {ncol}/{tot}")
    # drop-collapse recomputation on the paired delta
    dd = []
    for e in eps2:
        a = np.array(list(lv2[e].values()))
        b = np.array(list(llr[e].values()))
        a2 = a[a <= 3 * np.median(a)]
        b2 = b[b <= 3 * np.median(b)]
        dd.append(np.median(b2) - np.median(a2))
    summarize(f"{name} Δ after dropping collapse lanes", dd,
              base=np.median([v2[e] if name == 'CROWDED' else v2[e] for e in eps2]))
    # worst-lane (max) convention
    dd2 = [max(llr[e].values()) - max(lv2[e].values()) for e in eps2]
    summarize(f"{name} Δ of worst lane (max)", dd2)
    dd3 = [min(llr[e].values()) - min(lv2[e].values()) for e in eps2]
    summarize(f"{name} Δ of best lane (min)", dd3)
print()

print("=" * 100)
print("SECTION 5 — HELD-OUT 40 AUTO: intent-to-treat vs conditioning on the router")
print("=" * 100)
ha, hv = load(HO_AUTO), load(HO_V2)
la = lanes(ha, None)
lv = lanes(hv, None)
eps_ho = sorted(set(la) & set(lv))
route = {}
for r in ha:
    route.setdefault(r['episode'], []).append(r.get('auto_selected'))
dom = {e: max(set(v), key=v.count) for e, v in route.items()}
aM = agg(la)
vM = agg(lv)
d_all = [aM[e] - vM[e] for e in eps_ho]
print(f"  episodes {len(eps_ho)}, dominant route learned={sum(1 for e in eps_ho if dom[e]=='learned')}"
      f" v2_konly={sum(1 for e in eps_ho if dom[e]=='v2_konly')}")
summarize("ITT: all episodes (auto - v2)", d_all, base=np.median([vM[e] for e in eps_ho]))
for r_ in ('learned', 'v2_konly'):
    sub = [e for e in eps_ho if dom[e] == r_]
    summarize(f"conditioned on route={r_}", [aM[e] - vM[e] for e in sub],
              base=np.median([vM[e] for e in sub]))
# lane-level ITT (paired per lane), and per-lane routing
dl = []
for e in eps_ho:
    for k in la[e]:
        if k in lv[e]:
            dl.append(la[e][k] - lv[e][k])
summarize("ITT lane-level (n=200 lanes)", dl)
# null control on held-out 40
nlho = {e: NULLS['data_heldout40_2026-07-26/' + e]['null_avg_nRMSE'] for e in eps_ho}
for nm, m in (('auto', aM), ('v2_konly', vM)):
    summarize(f"{nm} minus flat-sea null (all 40)", [m[e] - nlho[e] for e in eps_ho])
for r_ in ('learned', 'v2_konly'):
    sub = [e for e in eps_ho if dom[e] == r_]
    for nm, m in (('auto', aM), ('v2_konly', vM)):
        summarize(f"{nm} minus null | route={r_}", [m[e] - nlho[e] for e in sub])
print()

print("=" * 100)
print("SECTION 6 — CONVENTION GRID for the two headline contrasts")
print("=" * 100)
for name, eps2, A, B in (('JUL crowded (learned-v2)', CROWDED, 'learned', 'v2_konly'),
                         ('JUL sparse  (learned-v2)', SPARSE, 'learned', 'v2_konly')):
    for lane_how in ('median', 'mean'):
        a = agg(lanes(jul, A), lane_how)
        b = agg(lanes(jul, B), lane_how)
        d = [a[e] - b[e] for e in eps2]
        st = signtest(d)
        st1 = signtest(d, 'less')
        w2 = wilcoxon(d, alternative='two-sided').pvalue
        w1 = wilcoxon(d, alternative='less').pvalue
        print(f"  {name} lane={lane_how:6s} medΔ={np.median(d):+.4f} meanΔ={np.mean(d):+.4f} "
              f"sign {st['wins']}/{st['n']} p2={st['p']:.3g} p1={st1['p']:.3g} "
              f"wil2={w2:.3g} wil1={w1:.3g}")
# unpaired sensitivity (Mann-Whitney) for crowded
from scipy.stats import mannwhitneyu
a = [lr[e] for e in CROWDED]
b = [v2[e] for e in CROWDED]
print(f"  UNPAIRED crowded Mann-Whitney p={mannwhitneyu(a, b, alternative='two-sided').pvalue:.3g} "
      f"(paired sign p={signtest([x-y for x, y in zip(a, b)])['p']:.3g})")
a = [lr[e] for e in SPARSE]
b = [v2[e] for e in SPARSE]
print(f"  UNPAIRED sparse  Mann-Whitney p={mannwhitneyu(a, b, alternative='two-sided').pvalue:.3g} "
      f"(paired sign p={signtest([x-y for x, y in zip(a, b)])['p']:.3g})")
print()
print("floor p: n=40 two-sided 2/2**40=%.3g ; n=18 %.4g ; n=12 %.4g ; n=4 %.4g"
      % (2 / 2**40, 2 / 2**18, 2 / 2**12, 2 / 2**4))
