#!/usr/bin/env python3
"""PRO analysis: put beamform / WavePrior-learned / oracle / LETKF on one axis.

Reuses existing raw JSONL only. No model runs.
Bridge: wp_integration `v2_konly` arm is BIT-IDENTICAL (20/20 lanes, 0.0e0) to
batchC_ws_phasefix, which is BIT-IDENTICAL to today's S1-A0 = S2-C0.
So the three runs share a common anchor arm; pairing is not "different runs paired".
"""
import json, statistics as st
from pathlib import Path

R = Path('/home/vexit/wavesplat/wavesplat')
OUT = R / 'tmp/waveprior_debate_2026-08-05'

EPS = ['episode_BF4_15comp_seed0_uniform.h5', 'episode_BF4_30comp_seed0_uniform.h5',
       'episode_BF5_15comp_seed0_uniform.h5', 'episode_BF5_30comp_seed0_uniform.h5']
SHORT = {e: e[8:-12].replace('comp_seed', 'c') for e in EPS}

wp = [json.loads(l) for l in open(R / 'tmp/wp_integration_2026-07-25/arm_runs.jsonl')]
s2 = [json.loads(l) for l in open(R / 'tmp/s1s2_2026-08-05/results.jsonl')]
bl = [json.loads(l) for l in open(R / 'tmp/rerun_2026-07-25/batchA_baselines.jsonl')]

LETKF = {r['episode']: r['nRMSE'] for r in bl if r['method'] == 'LETKF'}
ETKF = {r['episode']: r['nRMSE'] for r in bl if r['method'] == 'ETKF'}


def wp_get(ep, arm, key='avg_nRMSE', seed=None):
    v = [r[key] for r in wp if r['episode'] == ep and r['arm'] == arm
         and not r.get('err') and (seed is None or r['est_seed'] == seed)]
    return v


def s2_get(ep, arm, key='nRMSE', seed=None):
    v = [r[key] for r in s2 if r['episode'] == ep and r['arm'] == arm
         and (seed is None or r['seed'] == seed)]
    return v


# ---------------- Protocol A: seed 42 only (fully matched lane) ----------------
# ---------------- Protocol B: each run's own median protocol ------------------
def build(protocol):
    tab = {}
    for ep in EPS:
        if protocol == 'seed42':
            g = lambda a: (wp_get(ep, a, seed=42)[0] if wp_get(ep, a, seed=42) else None)
            h = lambda a: (s2_get(ep, a, seed=42)[0] if s2_get(ep, a, seed=42) else None)
            gi = lambda a: (wp_get(ep, a, 'init_nrmse', seed=42)[0] if wp_get(ep, a, 'init_nrmse', seed=42) else None)
            hi = lambda a: (s2_get(ep, a, 'init_nrmse', seed=42)[0] if s2_get(ep, a, 'init_nrmse', seed=42) else None)
        else:
            g = lambda a: (st.median(wp_get(ep, a)) if wp_get(ep, a) else None)
            h = lambda a: (st.median(s2_get(ep, a)) if s2_get(ep, a) else None)
            gi = lambda a: (st.median(wp_get(ep, a, 'init_nrmse')) if wp_get(ep, a, 'init_nrmse') else None)
            hi = lambda a: (st.median(s2_get(ep, a, 'init_nrmse')) if s2_get(ep, a, 'init_nrmse') else None)
        tab[ep] = dict(
            beamform=g('v2_konly'), beamform_s2=h('S1-A0'),
            learned=g('learned'), learned_konly=g('learned_konly'),
            oracle=h('S2-T1f'), oracle_ktrain=h('S2-T1'),
            oracle_notrack=h('X-T1f-notrack'), beamform_notrack=h('X-A0-notrack'),
            ngt_only=h('S2-C1'),
            i_beamform=gi('v2_konly'), i_learned=gi('learned'),
            i_learned_konly=gi('learned_konly'), i_oracle=hi('S2-T1f'),
            letkf=LETKF[ep], etkf=ETKF[ep])
    return tab


def show(tab, name):
    print(f'\n{"="*100}\n### PROTOCOL {name}\n{"="*100}')
    cols = ['beamform', 'learned', 'learned_konly', 'oracle', 'oracle_notrack', 'letkf', 'etkf']
    hdr = f'{"episode":14s}' + ''.join(f'{c:>16s}' for c in cols)
    print(hdr)
    for ep in EPS:
        t = tab[ep]
        print(f'{SHORT[ep]:14s}' + ''.join(
            (f'{t[c]:16.4f}' if t[c] is not None else f'{"-":>16s}') for c in cols))
    print(f'{"MEAN4":14s}' + ''.join(
        f'{st.mean([tab[e][c] for e in EPS]):16.4f}' for c in cols))

    # consistency: beamform via wp arm vs via s1s2 arm
    d = [abs(tab[e]['beamform'] - tab[e]['beamform_s2']) for e in EPS]
    print(f'\n[anchor check] max |wp v2_konly - s1s2 S1-A0| = {max(d):.3e}'
          f'  ({"BIT-IDENTICAL" if max(d) == 0 else "protocol-differs"})')

    # ---- recovery of the oracle-placement ceiling ----
    print(f'\n--- RECOVERY RATIO  r = (beamform - learned)/(beamform - oracle) ---')
    print('  (placement-only ceiling: oracle = S2-T1f, identical shipped tracker on top)')
    print(f'{"episode":14s}{"BF":>10s}{"LRN":>10s}{"ORC":>10s}{"gain_lrn":>11s}'
          f'{"gain_orc":>11s}{"recovery":>11s}{"LETKF":>10s}')
    rec, rec_k = [], []
    for ep in EPS:
        t = tab[ep]
        gl, go = t['beamform'] - t['learned'], t['beamform'] - t['oracle']
        r = gl / go if go > 0 else float('nan')
        rec.append(r)
        glk = t['beamform'] - t['learned_konly']
        rec_k.append(glk / go if go > 0 else float('nan'))
        print(f'{SHORT[ep]:14s}{t["beamform"]:10.4f}{t["learned"]:10.4f}{t["oracle"]:10.4f}'
              f'{gl:11.4f}{go:11.4f}{r*100:10.1f}%{t["letkf"]:10.4f}')
    pooled = ((st.mean([tab[e]['beamform'] for e in EPS]) - st.mean([tab[e]['learned'] for e in EPS]))
              / (st.mean([tab[e]['beamform'] for e in EPS]) - st.mean([tab[e]['oracle'] for e in EPS])))
    pooled_k = ((st.mean([tab[e]['beamform'] for e in EPS]) - st.mean([tab[e]['learned_konly'] for e in EPS]))
                / (st.mean([tab[e]['beamform'] for e in EPS]) - st.mean([tab[e]['oracle'] for e in EPS])))
    print(f'\n  per-episode recovery (learned)       : {[f"{x*100:.0f}%" for x in rec]}'
          f'  median {st.median(rec)*100:.1f}%')
    print(f'  per-episode recovery (learned_konly) : {[f"{x*100:.0f}%" for x in rec_k]}'
          f'  median {st.median(rec_k)*100:.1f}%')
    print(f'  POOLED recovery (mean4 based)        : learned {pooled*100:.1f}%'
          f'   learned_konly {pooled_k*100:.1f}%')

    # ---- vs the full ceiling (placement + tracker off) ----
    print(f'\n--- vs FULL ceiling (oracle placement + tracker OFF = X-T1f-notrack) ---')
    b = st.mean([tab[e]['beamform'] for e in EPS])
    l = st.mean([tab[e]['learned'] for e in EPS])
    o = st.mean([tab[e]['oracle'] for e in EPS])
    on = st.mean([tab[e]['oracle_notrack'] for e in EPS])
    print(f'  beamform {b:.4f} -> learned {l:.4f} -> oracle(track) {o:.4f} -> oracle(notrack) {on:.4f}')
    print(f'  learned recovers {(b-l)/(b-on)*100:.1f}% of the FULL headroom (b-on = {b-on:.4f})')
    print(f'  residual after learned  : {l-on:.4f}  = {(l-on)/(b-on)*100:.1f}% of headroom')
    print(f'    of which tracker tax  : {o-on:.4f}  = {(o-on)/(l-on)*100:.1f}% of that residual')
    print(f'    of which placement    : {l-o:.4f}  = {(l-o)/(l-on)*100:.1f}% of that residual')

    # ---- placement quality (init_nrmse) ----
    print(f'\n--- PLACEMENT QUALITY at end of init (init_nrmse) ---')
    print(f'{"episode":14s}{"BF":>10s}{"LRN":>10s}{"LRN_k":>10s}{"ORC":>10s}{"init_recov":>12s}')
    ir = []
    for ep in EPS:
        t = tab[ep]
        g = (t['i_beamform'] - t['i_learned']) / (t['i_beamform'] - t['i_oracle'])
        ir.append(g)
        print(f'{SHORT[ep]:14s}{t["i_beamform"]:10.4f}{t["i_learned"]:10.4f}'
              f'{t["i_learned_konly"]:10.4f}{t["i_oracle"]:10.4f}{g*100:11.1f}%')
    print(f'{"MEAN4":14s}' + ''.join(f'{st.mean([tab[e][c] for e in EPS]):10.4f}'
          for c in ['i_beamform', 'i_learned', 'i_learned_konly', 'i_oracle']))
    print(f'  init-level recovery: per-ep {[f"{x*100:.0f}%" for x in ir]} median {st.median(ir)*100:.1f}%')
    return tab, rec, rec_k


for proto in ['seed42', 'own_median']:
    show(build(proto), proto)

# ---------------- correlation: init placement quality vs tracked error ----------
print(f'\n{"="*100}\n### MECHANISM: does init placement quality predict tracked error? (all 5 wp lanes)\n{"="*100}')
import math
pairs = []
for ep in EPS:
    for arm in ['v2_konly', 'learned', 'learned_konly']:
        for r in wp:
            if r['episode'] == ep and r['arm'] == arm and not r.get('err'):
                pairs.append((r['init_nrmse'], r['avg_nRMSE'], arm, SHORT[ep]))


def pearson(x, y):
    n = len(x); mx, my = st.mean(x), st.mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
    return num / den


xs = [p[0] for p in pairs]; ys = [p[1] for p in pairs]
print(f'  n={len(pairs)} runs (4 ep x 3 arms x 5 lanes), rho(init_nrmse, tracked nRMSE) = {pearson(xs, ys):+.3f}')
for ep in EPS:
    sub = [p for p in pairs if p[3] == SHORT[ep]]
    print(f'    within {SHORT[ep]:12s} n={len(sub):2d}  rho = {pearson([p[0] for p in sub], [p[1] for p in sub]):+.3f}')
# add oracle points from s1s2 (seed42 lane, comparable)
allp = list(pairs)
for ep in EPS:
    for arm in ['S1-A0', 'S2-T1f', 'S2-T1']:
        for r in s2:
            if r['episode'] == ep and r['arm'] == arm:
                allp.append((r['init_nrmse'], r['nRMSE'], arm, SHORT[ep]))
print(f'  + oracle arms folded in: n={len(allp)}, rho = '
      f'{pearson([p[0] for p in allp], [p[1] for p in allp]):+.3f}')
