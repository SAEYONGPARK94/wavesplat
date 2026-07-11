"""GIF: is the estimate actually good? Truth vs hand vs bw-net on a real
window, animated over forecast lead.

Picks (by default) the most-rescued bad window from gate_promotion_eval.json
(max l1s2bw_s0 - hand among hand<0.5), rebuilds that window verbatim, and
renders truth / hand / net predicted fields on the record grid for leads
0..10 s, ship encounter point marked, per-lead track corr annotated.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter

from d3_falcon_real_operator import RECORDS
from d3_sa_multirecord import MRecord
from d3_prior_net_eval import get_net, net_gamma_mmse, DEVICE
from d3_gate_promotion_eval import window_data
from d3_gpu_mmse import mmse_gpu

OUT = 'results/direction3/gifs_rescue'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--record', default=None)
    ap.add_argument('--wi', type=int, default=None)
    a = ap.parse_args()
    rows = json.load(open('results/direction3/gate_promotion_eval.json'))
    if a.record is None:
        bad = [r for r in rows if r['hand'] < 0.5]
        r = max(bad, key=lambda r: r['bw_s0_head'] - r['hand'])
    else:
        r = next(x for x in rows if x['record'] == a.record
                 and x['wi'] == a.wi)
    name, wi = r['record'], r['wi']
    print(f"window {name} w{wi}: hand {r['hand']:+.2f} "
          f"net+head {r['bw_s0_head']:+.2f}")

    cfg = RECORDS[name]
    rec = MRecord(cfg['path'], stride=cfg['stride'])
    nfw = int(round(40.0 / rec.dt))
    n_lead = int(round(cfg['lead_max'] / rec.dt))
    i0s = np.linspace(nfw + 1, rec.T_all - n_lead - 2, 12).astype(int)
    P, z, Pq, gt_q, leads, Hs, kp, i0 = window_data(
        rec, cfg, int(i0s[wi]), nfw, n_lead, wi)
    Zfut = rec.window(int(i0s[wi]), n_lead + 1)
    t0 = float(rec.t[int(i0s[wi])])
    sig0 = 0.05 * Hs
    net = get_net('results/direction3/prior_net_l1s2bw.pt')
    from d3_l1_s2head_train import SmultHead
    head = SmultHead().to(DEVICE)
    head.load_state_dict(torch.load(
        'results/direction3/prior_net_l1s2bw.pt',
        map_location=DEVICE)['head'])
    head.eval()

    # grid queries per lead (stride grid for speed)
    st = 2
    gx, gy = rec.gx[::st], rec.gy[::st]
    Xg, Yg = np.meshgrid(gx, gy)
    show_leads = [l_ for l_ in leads if l_ <= 10.0]
    pred_h, pred_n = [], []
    for l_ in show_leads:
        Q = np.column_stack([Xg.ravel(), Yg.ravel(),
                             np.full(Xg.size, t0 + l_)])
        with torch.no_grad():
            ph = mmse_gpu(torch.tensor(P, device=DEVICE,
                                       dtype=torch.float32),
                          torch.tensor(z, device=DEVICE,
                                       dtype=torch.float32),
                          cfg['depth'],
                          torch.tensor(Q, device=DEVICE,
                                       dtype=torch.float32),
                          sig0, kp_hint=kp).cpu().numpy()
        pn = net_gamma_mmse(net, P, z, cfg['depth'], Q, sig0,
                            seed=int(i0s[wi]), kp_hint=kp, use_em=False,
                            smult_head=head)
        pred_h.append(ph.reshape(Xg.shape))
        pred_n.append(np.asarray(pn).reshape(Xg.shape))
        print(f"lead {l_:4.1f}s rendered", flush=True)

    vmax = np.nanpercentile(np.abs(Zfut), 97)
    os.makedirs(OUT, exist_ok=True)
    path = f"{OUT}/rescue_{name}_w{wi}.gif"
    fig, axs = plt.subplots(1, 3, figsize=(12.2, 4.2), dpi=100)
    wr = PillowWriter(fps=2)
    with wr.saving(fig, path, dpi=100):
        for li, l_ in enumerate(show_leads):
            Zt = Zfut[int(round(l_ / rec.dt))]
            for ax in axs:
                ax.clear()
            panels = [(Zt[::st, ::st], f'truth (stereo)'),
                      (pred_h[li], f"hand  corr {r['hand']:+.2f}"),
                      (pred_n[li],
                       f"net+s2-head  corr {r['bw_s0_head']:+.2f}")]
            for ax, (Z, ttl) in zip(axs, panels):
                ax.pcolormesh(gx, gy, Z, cmap='RdBu_r',
                              vmin=-vmax, vmax=vmax, shading='auto')
                sp = Pq[li][:2]
                ax.plot(sp[0], sp[1], marker='*', ms=14, mec='k',
                        mfc='yellow')
                ax.set_title(ttl, fontsize=10)
                ax.set_aspect('equal')
                ax.set_xticks([])
                ax.set_yticks([])
            fig.suptitle(f"{name} window {wi} — lead {l_:.0f} s "
                         f"(★ = ship encounter point)", fontsize=11)
            wr.grab_frame()
    print(f"wrote {path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
