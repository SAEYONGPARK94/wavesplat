"""Visual verification GIFs: GT | WaveSplat(analytic prior) | WavePrior(learned net)
using the CURRENT best learned net (2nd-lobe-fixed). Renders a few cases so we can SEE
whether the estimation is sane — especially real AA15b (the real cross-Tp bimodal record).
"""
import d3_compare_gifs as cg

cg.NET_CKPT = 'results/direction3/prior_net_2ndlobeTF.pt'   # current best learned prior
CASES = [('linear', None, 0), ('hos', None, 0), ('real', 'YS17', 3), ('real', 'AA15b', 3)]

for kind, rec, w in CASES:
    print(f"[{kind} {rec or ''}] building scene...", flush=True)
    sc = cg.scene_real(rec, win=w) if kind == 'real' else cg.scene_sim(kind, seed=w)
    sc['n_obs'] = len(sc['z'])
    sol = cg.solve(sc)
    met = cg.metrics(sc, sol)
    path = f"results/gifs/verify_{kind}_{rec or 'synth'}.gif"
    cg.render(sc, sol, met, path)
    ws = met['WaveSplat']['map_nrmse_nowcast']; wp = met['WavePrior']['map_nrmse_nowcast']
    print(f"  -> {path}  nowcast nRMSE  WaveSplat {ws:.3f} / WavePrior(learned) {wp:.3f}", flush=True)
