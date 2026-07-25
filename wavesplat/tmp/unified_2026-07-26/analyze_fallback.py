"""UNIFIED Part-C: score the FROZEN collapse-fallback arm on all 30 real windows.

FROZEN detector (PREREG.md): fire iff amp_ratio_grid > 1.5.
FROZEN fallback: hand everywhere; if detector fires -> use learned_now for that window.
Sources: tmp/unified_2026-07-26/hand_amp.json (hand re-score, bit-identical control) +
learned_now from wp_real.json (carried in hand_amp.json). Writes fallback_results.json.
"""
import json
import numpy as np
from scipy.stats import wilcoxon

HAND_AMP = 'tmp/unified_2026-07-26/hand_amp.json'
TAU = 1.5   # FROZEN amplitude-ratio threshold


def main():
    d = json.load(open(HAND_AMP))
    rows = d['rows']
    assert abs(d['hand_now_maxdiff']) < 1e-9, 'hand bit-identity control failed'

    coll = [r for r in rows if r['collapse']]
    norm = [r for r in rows if not r['collapse']]
    assert len(coll) == 3 and len(norm) == 27

    def fires(r):
        return r['amp_ratio_grid'] > TAU

    def fb_value(r):
        return r['learned_now'] if fires(r) else r['hand_now']

    # (i) detection
    fired_coll = [r for r in coll if fires(r)]
    detection = dict(n_collapse=len(coll), n_fired=len(fired_coll),
                     fired=[f"{r['record']}_w{r['win']}" for r in fired_coll],
                     missed=[f"{r['record']}_w{r['win']}" for r in coll if not fires(r)],
                     pass_bar=len(fired_coll) >= 2)
    # per collapse window: hand vs fallback
    coll_detail = []
    for r in coll:
        coll_detail.append(dict(window=f"{r['record']}_w{r['win']}", hand=r['hand_now'],
                                learned=r['learned_now'], fires=fires(r),
                                fallback=fb_value(r), amp=r['amp_ratio_grid'],
                                repair=r['hand_now'] - fb_value(r)))

    # (ii) specificity / no-harm on 27 normal
    false_fires = [r for r in norm if fires(r)]
    hand_norm = [r['hand_now'] for r in norm]
    fb_norm = [fb_value(r) for r in norm]
    dn = np.array(fb_norm) - np.array(hand_norm)
    try:
        p_norm = float(wilcoxon(fb_norm, hand_norm).pvalue) if np.any(dn != 0) else 1.0
    except Exception:
        p_norm = 1.0
    specificity = dict(n_normal=len(norm), n_false_fire=len(false_fires),
                       false_fire_rate=len(false_fires) / len(norm),
                       false_fired=[f"{r['record']}_w{r['win']}" for r in false_fires],
                       hand_mean=float(np.mean(hand_norm)), fb_mean=float(np.mean(fb_norm)),
                       mean_delta=float(dn.mean()), p_paired=p_norm,
                       noharm_pass=bool(p_norm >= 0.05 or dn.mean() <= 0))

    # (iii) all-30 aggregate
    hand_all = [r['hand_now'] for r in rows]
    fb_all = [fb_value(r) for r in rows]
    aggregate = dict(n=len(rows), hand_mean=float(np.mean(hand_all)),
                     fallback_mean=float(np.mean(fb_all)),
                     hand_median=float(np.median(hand_all)),
                     fallback_median=float(np.median(fb_all)),
                     learned_all_mean=float(np.mean([r['learned_now'] for r in rows])))

    decision = bool(detection['pass_bar'] and specificity['noharm_pass'])
    out = dict(tau=TAU, detection=detection, coll_detail=coll_detail,
               specificity=specificity, aggregate=aggregate, decision=decision,
               all_amp_sorted=sorted([(round(r['amp_ratio_grid'], 3),
                                       f"{r['record']}_w{r['win']}", r['collapse'])
                                      for r in rows], reverse=True))
    json.dump(out, open('tmp/unified_2026-07-26/fallback_results.json', 'w'), indent=1)

    print(f"FROZEN detector: amp_ratio_grid > {TAU}\n")
    print("(i) DETECTION:", detection)
    for c in coll_detail:
        print(f"    {c['window']:12s} hand={c['hand']:.3f} learned={c['learned']:.3f}"
              f" amp={c['amp']:.3f} fires={c['fires']} -> fallback={c['fallback']:.3f}"
              f" (repair {c['repair']:+.3f})")
    print("\n(ii) SPECIFICITY/NO-HARM (27 normal):", specificity)
    print("\n(iii) AGGREGATE (all 30):", aggregate)
    print(f"\nDECISION (detection>=2/3 AND no-harm) = {decision}")


if __name__ == '__main__':
    main()
