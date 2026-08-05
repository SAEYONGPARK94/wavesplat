"""Auditor follow-ups: (a) matched-axis oracle recovery, (b) C5 mechanism control,
(c) C3/S2 catastrophe decomposition, (d) aggregation-convention sweep, (e) stratification.
Existing raw only. No new wave runs."""
import json, re
import numpy as np
from scipy import stats
from collections import defaultdict

T = '/home/vexit/wavesplat/wavesplat/tmp/'
def comp(ep): return int(re.search(r'_(\d+)comp', ep).group(1))
OUT = {}

# ================= (a) matched-axis oracle recovery =================
s = defaultdict(lambda: defaultdict(list))
for l in open(T+'s1s2_2026-08-05/results.jsonl'):
    r = json.loads(l); s[r['arm']][r['episode']].append(r['nRMSE'])
EPS = sorted(s['S1-A0'])
i = defaultdict(lambda: defaultdict(list))
for l in open(T+'wp_integration_2026-07-25/arm_runs.jsonl'):
    r = json.loads(l)
    if not r.get('err'): i[r['episode']][r['arm']].append(r)
c = defaultdict(list)
for l in open(T+'wp_crowded_2026-07-25/cond_runs.jsonl'):
    r = json.loads(l)
    if r['cond_polish_nrmse_Hs'] is not None: c[r['episode']].append(r['cond_polish_nrmse_Hs'])
LETKF = {'episode_BF4_15comp_seed0_uniform.h5':0.2282,'episode_BF4_30comp_seed0_uniform.h5':0.2347,
         'episode_BF5_15comp_seed0_uniform.h5':0.2298,'episode_BF5_30comp_seed0_uniform.h5':0.2651}
S  = lambda a, e: float(np.median(s[a][e]))
W  = lambda a, e, k='avg_nRMSE': float(np.median([r[k] for r in i[e][a]]))
M  = lambda f: float(np.mean([f(e) for e in EPS]))

print('='*100)
print('(a) FOUR POINTS ON THE SAME AXIS  -- overlap = %d episodes, ALL crowded, ALL seed0_uniform' % len(EPS))
print('='*100)
print('\n--- AXIS 1: TRACKED (sequential filter, avg over tracking frames) ---')
rowsT = [('beamform init  (v2_konly, 07-25)', lambda e: W('v2_konly', e)),
         ('beamform init  (S1-A0,   08-05)', lambda e: S('S1-A0', e)),
         ('learned-init   (WP learned)',      lambda e: W('learned', e)),
         ('learned-init   (WP learned_konly)',lambda e: W('learned_konly', e)),
         ('LETKF',                            lambda e: LETKF[e]),
         ('ORACLE (kx,ky) frozen  S2-T1f',    lambda e: S('S2-T1f', e))]
print(' %-36s %8s %8s %8s %8s | %8s' % ('arm', *[e[8:-13].replace('comp_see','c') for e in EPS], 'MEAN4'))
for lab, f in rowsT:
    print(' %-36s %8.4f %8.4f %8.4f %8.4f | %8.4f' % (lab, *[f(e) for e in EPS], M(f)))
baseT, ceilT = M(lambda e: W('v2_konly', e)), M(lambda e: S('S2-T1f', e))
print('\n headroom (beamform->oracle) = %.4f' % (baseT-ceilT))
for lab, f in [('WP learned', lambda e: W('learned', e)), ('WP learned_konly', lambda e: W('learned_konly', e)),
               ('LETKF', lambda e: LETKF[e])]:
    print('   %-20s recovers %5.1f%% of oracle headroom' % (lab, 100*(baseT-M(f))/(baseT-ceilT)))

print('\n--- AXIS 2: UNTRACKED / batch nowcast (no sequential filter) ---')
rowsN = [('beamform, no tracker  X-A0-notrack', lambda e: S('X-A0-notrack', e)),
         ('learned proposal+polish (WP cond)',  lambda e: float(np.median(c[e]))),
         ('ORACLE (kx,ky) no tracker X-T1f-nt', lambda e: S('X-T1f-notrack', e))]
for lab, f in rowsN:
    print(' %-36s %8.4f %8.4f %8.4f %8.4f | %8.4f' % (lab, *[f(e) for e in EPS], M(f)))
baseN = M(lambda e: S('X-A0-notrack', e)); ceilN = M(lambda e: S('X-T1f-notrack', e))
condN = M(lambda e: float(np.median(c[e])))
print('\n headroom (beamform->oracle, untracked) = %.4f' % (baseN-ceilN))
print('   %-20s recovers %5.1f%% of oracle headroom' % ('WP cond', 100*(baseN-condN)/(baseN-ceilN)))
print('\n *** AXIS MISMATCH: handoff script compared WP cond (UNTRACKED) to S2-T1f (TRACKED)')
print('     -> reported 117.2%% ("beats the oracle"). On the matched untracked axis it is %.1f%%.' %
      (100*(baseN-condN)/(baseN-ceilN)))
print('\n--- TRACKER COST, conditional on k quality (same 4 eps) ---')
print('   with beamform k : notrack %.4f -> tracked %.4f  (%+.1f%%)' % (baseN, M(lambda e: S('S1-A0', e)), 100*(M(lambda e: S('S1-A0',e))/baseN-1)))
print('   with ORACLE   k : notrack %.4f -> tracked %.4f  (%+.1f%%)' % (ceilN, ceilT, 100*(ceilT/ceilN-1)))
OUT['axis'] = dict(tracked=dict(base=baseT, ceil=ceilT, learned=M(lambda e: W('learned', e)),
                                learned_konly=M(lambda e: W('learned_konly', e)),
                                letkf=M(lambda e: LETKF[e]),
                                recovery_pct=100*(baseT-M(lambda e: W('learned', e)))/(baseT-ceilT)),
                   untracked=dict(base=baseN, ceil=ceilN, cond=condN,
                                  recovery_pct=100*(baseN-condN)/(baseN-ceilN)),
                   n_overlap_episodes=len(EPS), overlap_scope='all crowded, all seed0_uniform')

# ================= (b) C5 mechanism control =================
print('\n' + '='*100); print('(b) C5 MECHANISM: is "proposal fails when sparse (n_eff)" a controlled claim?'); print('='*100)
R = json.load(open(T+'wp_real_2026-07-25/wp_real.json'))['results']
norm = [r for r in R if not r['collapse']]
ne = np.array([r['n_eff'] for r in norm]); h = np.array([r['hand_now'] for r in norm])
lr = np.array([r['learned_now'] for r in norm]); D = h - lr
print(' n_eff vs hand_now   Spearman rho=%+.3f p=%.4f   <- baseline DIFFICULTY also tracks n_eff' % stats.spearmanr(ne, h))
print(' n_eff vs learned_now Spearman rho=%+.3f p=%.4f' % stats.spearmanr(ne, lr))
print(' n_eff vs D(hand-lrn) Spearman rho=%+.3f p=%.4f   (D>0 = learned better)' % stats.spearmanr(ne, D))
print(' n_eff vs LOG-RATIO log(lrn/hand) rho=%+.3f p=%.4f  <- scale-free version' % stats.spearmanr(ne, np.log(lr/h)))
rec = np.array([r['record'] for r in norm])
print('\n WITHIN-RECORD control (record absorbs site/sea-state):')
allw = []
for u in sorted(set(rec)):
    m = rec == u
    if m.sum() >= 4:
        rho, p = stats.spearmanr(ne[m], D[m]); allw.append(rho)
        print('   %-6s n=%d  rho(n_eff,D)=%+.3f p=%.3f   n_eff range %.1f-%.1f' % (u, m.sum(), rho, p, ne[m].min(), ne[m].max()))
print('   mean within-record rho = %+.3f  (pooled rho was %+.3f)' % (np.mean(allw), stats.spearmanr(ne, D)[0]))
print('\n *** n_eff spans only %.1f-%.1f across ALL 27 windows; the synthetic "sparse" regime that' % (ne.min(), ne.max()))
print('     the proposal was said to fail on is n_comp 1-2. No real window is anywhere near it.')
OUT['c5_mech'] = dict(rho_neff_D=stats.spearmanr(ne, D)[0], p=stats.spearmanr(ne, D)[1],
                      rho_neff_hand=stats.spearmanr(ne, h)[0], within_record_mean_rho=float(np.mean(allw)),
                      neff_range=[float(ne.min()), float(ne.max())])

# ================= (c) C3 / S2 catastrophe decomposition =================
print('\n' + '='*100); print('(c) C3 held-out: what actually drives the S1/S2 FAILs?'); print('='*100)
def hl(f):
    o = defaultdict(dict)
    for l in open(T+'heldout40_2026-07-26/'+f):
        r = json.loads(l); o[r['episode']][(r['est_seed'], r['diff_seed'])] = r['avg_nRMSE']
    return o
hL, hV, hA = hl('arm_runs.jsonl'), hl('arm_runs_v2_control.jsonl'), hl('arm_runs_auto_live.jsonl')
lanes = sorted(hL[list(hL)[0]]); m = lambda o, e: float(np.median([o[e][k] for k in lanes]))
sp = [e for e in hL if comp(e) in (1, 2)]
blow = [e for e in sp if m(hA, e) > 0.10]
print(' sparse n=%d ; auto/guard blowups(>0.10): %d -> %s' % (len(sp), len(blow), [e[8:-3] for e in blow]))
print(' all blowups are BF2_1comp? %s' % all('BF2_1comp' in e for e in blow))
keep = [e for e in sp if e not in blow]
print(' S2 bar = pure_v2_mean + 0.005')
print('   WITH blowups   : guard %.5f vs bar %.5f -> %s' % (np.mean([m(hA,e) for e in sp]), np.mean([m(hV,e) for e in sp])+0.005,
      'PASS' if np.mean([m(hA,e) for e in sp]) <= np.mean([m(hV,e) for e in sp])+0.005 else 'FAIL'))
print('   WITHOUT blowups: guard %.5f vs bar %.5f -> %s' % (np.mean([m(hA,e) for e in keep]), np.mean([m(hV,e) for e in keep])+0.005,
      'PASS' if np.mean([m(hA,e) for e in keep]) <= np.mean([m(hV,e) for e in keep])+0.005 else 'FAIL'))
d = np.array([m(hA, e) - m(hV, e) for e in sp])
print('   paired on sparse: medianD %+.5f  guard-better %d/%d  Wilcoxon2 p=%.3f' %
      (np.median(d), int((d < 0).sum()), len(d), stats.wilcoxon(d)[1]))
dall = np.array([m(hA, e) - m(hV, e) for e in hL])
print('\n S1 (guard vs pure_v2, all 40): meanD %+.5f -> bootstrap CI [%.4f, %.4f] straddles 0 => FAIL' % (dall.mean(), -0.02502, 0.01691))
print('   SAME DATA, median/rank: Wilcoxon p(less)=0.00225 ; on the %d episodes the router acted, %d win / %d loss' %
      (int((dall != 0).sum()), int((dall < 0).sum()), int((dall > 0).sum())))
print('   *** S1 FAIL is a mean-vs-median disagreement, not an absence of effect.')
OUT['c3'] = dict(sparse_blowups=[e for e in blow], all_bf2_1comp=all('BF2_1comp' in e for e in blow),
                 s2_with=float(np.mean([m(hA,e) for e in sp])), s2_without=float(np.mean([m(hA,e) for e in keep])),
                 s1_meanD=float(dall.mean()), s1_wins=int((dall<0).sum()), s1_losses=int((dall>0).sum()))

# ================= (d) aggregation-convention sweep =================
print('\n' + '='*100); print('(d) AGGREGATION CONVENTION: median-of-differences vs difference-of-medians vs means'); print('='*100)
print(' %-34s %11s %11s %11s %11s %6s' % ('conclusion', 'med(pair d)', 'mean(pair d)', 'med(A)-med(B)', 'mean(A)-mean(B)', 'FLIP?'))
def sweep(lab, A, B):
    A, B = np.asarray(A), np.asarray(B); d = A - B
    v = [np.median(d), d.mean(), np.median(A)-np.median(B), A.mean()-B.mean()]
    fl = 'YES' if len(set(np.sign(np.round(v, 6)))) > 1 else '-'
    print(' %-34s %+11.5f %+11.5f %+11.5f %+11.5f %6s' % (lab, *v, fl))
    return dict(med_pair=v[0], mean_pair=v[1], med_diff=v[2], mean_diff=v[3], flip=fl == 'YES')
agg = {}
cg = defaultdict(list); wi = defaultdict(list); wt = defaultdict(list)
for l in open(T+'wp_crowded_2026-07-25/cond_runs.jsonl'):
    r = json.loads(l)
    if r['cond_polish_nrmse_Hs'] is not None: cg[r['episode']].append(r['cond_polish_nrmse_Hs'])
for l in open(T+'wp_crowded_2026-07-25/ws_runs.jsonl'):
    r = json.loads(l); wi[r['episode']].append(r['init_nrmse']); wt[r['episode']].append(r['avg_nRMSE'])
prim = sorted([e for e in cg if comp(e) in (15, 30) and e in wi])
agg['C1'] = sweep('C1 cond vs ws_best (crowded40)', [np.median(cg[e]) for e in prim],
                  [min(np.median(wi[e]), np.median(wt[e])) for e in prim])
g = defaultdict(lambda: defaultdict(list))
for l in open(T+'wp_integration_2026-07-25/arm_runs.jsonl'):
    r = json.loads(l); g[r['episode']][r['arm']].append(r['avg_nRMSE'])
pr2 = sorted([e for e in g if comp(e) in (15, 30)])
agg['C2'] = sweep('C2 learned vs v2 (crowded40)', [np.median(g[e]['learned']) for e in pr2], [np.median(g[e]['v2_konly']) for e in pr2])
cr = [e for e in hL if comp(e) in (15, 30)]
agg['C3'] = sweep('C3 heldout learned vs v2 (crowd16)', [m(hL, e) for e in cr], [m(hV, e) for e in cr])
ft = defaultdict(list)
for l in open(T+'wp_e2e_2026-07-25/ft_cond_runs.jsonl'):
    r = json.loads(l)
    if r['cond_polish_nrmse_Hs'] is not None: ft[r['episode']].append(r['cond_polish_nrmse_Hs'])
p4 = [e for e in prim if e in ft]
agg['C4'] = sweep('C4 e2e-ft vs current (crowded40)', [np.median(ft[e]) for e in p4], [np.median(cg[e]) for e in p4])
agg['C5a'] = sweep('C5 real learned vs hand (norm27)', lr, h)
Ra = R; ha = np.array([r['hand_now'] for r in Ra]); la = np.array([r['learned_now'] for r in Ra])
agg['C5b'] = sweep('C5 real learned vs hand (ALL 30)', la, ha)
mm = re.findall(r'hand ([\d.]+)\s+fanTF ([\d.]+)\s+oldTF ([\d.]+)', open(T+'G1_waveprior_power/eval_fan_prereg.log').read())
agg['C6'] = sweep('C6 G1 fanTF vs hand (n=30)', [float(b) for a, b, _ in mm], [float(a) for a, b, _ in mm])
LN = [(42,0),(1,1),(2,2),(3,3),(4,4)]
v2 = defaultdict(dict); eps = set()
for l in open(T+'rerun_2026-07-25/batchC_ws_phasefix.jsonl'):
    dd = json.loads(l); v2[dd['episode']][dd['seed']] = dd['nRMSE']; eps.add(dd['episode'])
lo = defaultdict(dict)
for src in (T+'wp_integration_2026-07-25/arm_runs.jsonl', T+'unified_2026-07-26/arm_runs_extra.jsonl'):
    for l in open(src):
        dd = json.loads(l)
        if dd.get('arm') == 'learned' and dd.get('err') is None: lo[dd['episode']][(dd['est_seed'], dd['diff_seed'])] = dd['avg_nRMSE']
SPe = sorted([e for e in eps if comp(e) in (1, 2) and e in lo])
agg['C7'] = sweep('C7 pure-learned vs v2 (sparse40)', [np.median([lo[e][k] for k in LN]) for e in SPe],
                  [np.median([v2[e][E] for E, _ in LN]) for e in SPe])
OUT['aggregation'] = agg

# ================= (e) stratification inheritance =================
print('\n' + '='*100); print('(e) STRATIFICATION: every conclusion below is defined by an n_comp cut'); print('='*100)
NEFF = json.load(open(T+'depth3_ceiling_2026-08-05/CON_followups.json'))
print(' N_eff source keys:', list(NEFF)[:8])
print('\n within-"crowded": 15comp vs 30comp')
for nc in (15, 30):
    ee = [e for e in prim if comp(e) == nc]
    d1 = np.array([np.median(cg[e]) - min(np.median(wi[e]), np.median(wt[e])) for e in ee])
    ee2 = [e for e in pr2 if comp(e) == nc]
    d2 = np.array([np.median(g[e]['learned']) - np.median(g[e]['v2_konly']) for e in ee2])
    print('   %2dcomp  C1 medD %+.4f   C2 medD %+.4f' % (nc, np.median(d1), np.median(d2)))
print('\n within-"sparse": 1comp vs 2comp')
for nc in (1, 2):
    ee = [e for e in SPe if comp(e) == nc]
    if not ee: continue
    L = np.array([np.median([lo[e][k] for k in LN]) for e in ee]); V = np.array([np.median([v2[e][E] for E, _ in LN]) for e in ee])
    print('   %dcomp n=%2d  learned-better %2d/%2d  medD %+.5f  meanD %+.5f  blowups %d' %
          (nc, len(ee), int((L-V < 0).sum()), len(ee), np.median(L-V), (L-V).mean(), int((L > 0.1).sum())))

json.dump(OUT, open(T+'waveprior_audit_2026-08-05/audit_followups.json', 'w'), indent=1, default=float)
print('\nwrote audit_followups.json')
