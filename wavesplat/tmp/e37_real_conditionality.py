"""Does "the incumbent is a constant prior + a direction read-off" hold on REAL data?

e30_conditionality (synthetic arena, n=16) found that replacing the net's per-sea
gamma with its OWN across-sea mean map (rotated to mu_hat) costs only
    -0.0129 +- 0.0242 (t=-0.53, p=0.604, 6/16)  -- indistinguishable,
i.e. the beamform map's other ~2600 pixels buy nothing and the whole learned
conditioning is a direction read-off. That is the load-bearing claim behind
declaring the prior-net direction CLOSED.

But the project's LEARNING HEADLINE came from REAL data, on a DIFFERENT task:
forward-fan encounter forecasting (leads 0-10 s along the ship track), where the
net beats the hand prior. If the net were "just a constant prior + direction",
a constant prior must reproduce that win. If it does NOT, the closure is an
artifact of the synthetic field-reconstruction arena and does NOT transfer.

Arms (identical windows / solver / kp_hint / sigma0 as d3_netfan_eval):
  net    the incumbent l1s2bw                     (reproduces 'this' in l1s2bw_real_eval.json)
  const  the net's OWN mean log-gamma map, aligned to mu_hat=0 across all windows,
         rotated back to each window's mu_hat   -> ZERO per-window info but direction
  noalign the same mean map WITHOUT rotation    -> control: direction read-off value
  hand   the classical arm (from val_gate_probe.json, as d3_netfan_eval uses)
"""
import os, sys, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')

from d3_falcon_real_operator import RECORDS, bilinear, resolve_direction
from d3_sa_multirecord import MRecord
from d3_prior_net_train import PriorNet, NGRID, KREL
from d3_prior_net_eval import net_gamma_mmse, DEVICE
from d3_sa_nomp import beamform_power, split_masks
from d18_relax_estimator import beamform_top, omega
from realdata_analysis import directional_spread
from e30_conditionality_probe import rotate_map

G = 9.81
W_MAX = 40.0      # fan construction span (d3_netfan_eval main)
W = 20.0          # final obs mask (d3_netfan_eval module-level W) -- NOT the same!
CKPT = 'results/direction3/prior_net_l1s2bw.pt'
OUT = 'results/direction3/e37_real_conditionality.json'


class ConstNet(torch.nn.Module):
    """Returns a fixed log10-gamma map, rotated to self.mu. Ignores its input."""
    def __init__(self, mean_aligned):
        super().__init__()
        self.mean_aligned = mean_aligned
        self.mu = 0.0

    def forward(self, X):
        M = rotate_map(self.mean_aligned, self.mu)
        return torch.tensor(M[None], dtype=torch.float32, device=DEVICE)


def get_net():
    ck = torch.load(CKPT, map_location=DEVICE)
    net = PriorNet().to(DEVICE)
    net.load_state_dict(ck['model'])
    net.eval()
    return net


def make_X(P, z, depth, kp):
    m1, m2 = split_masks(P, mode='legs')
    _, _, p1 = beamform_power(P[m1], z[m1], depth, KREL * kp, NGRID)
    _, _, p2 = beamform_power(P[m2], z[m2], depth, KREL * kp, NGRID)
    p1 = (p1 / max(p1.max(), 1e-12)).reshape(NGRID, NGRID)
    p2 = (p2 / max(p2.max(), 1e-12)).reshape(NGRID, NGRID)
    pm = np.minimum(p1, p2)
    return np.stack([p1, p2, pm,
                     np.log10(p1 + 1e-6) / 6 + 1, np.log10(p2 + 1e-6) / 6 + 1,
                     np.log10(pm + 1e-6) / 6 + 1]).astype(np.float32)


def build_windows():
    """Mirrors d3_netfan_eval.main() window construction exactly."""
    out = []
    for name in ('YS17', 'AA15a'):
        cfg = RECORDS[name]
        rec = MRecord(cfg['path'], stride=cfg['stride'])
        depth, U = cfg['depth'], cfg['U']
        leads = np.arange(0.0, cfg['lead_max'] + 1e-9, 1.0)
        nfw = int(round(W_MAX / rec.dt))
        n_lead = int(round(cfg['lead_max'] / rec.dt))
        cx = 0.5 * (rec.gx.min() + rec.gx.max())
        cy = 0.5 * (rec.gy.min() + rec.gy.max())
        Xg, Yg = np.meshgrid(rec.gx, rec.gy)
        i0s = np.linspace(nfw + 1, rec.T_all - n_lead - 2, 12).astype(int)
        for wi, i0 in enumerate(i0s):
            i0 = int(i0)
            rng = np.random.default_rng(700 + wi)
            Zmem = rec.window(i0 - nfw + 1, nfw)
            Zfut = rec.window(i0, n_lead + 1)
            t0 = float(rec.t[i0])
            Hs = 4.0 * float(np.nanstd(Zmem))
            spread, lam_p, pdir_ax, kp = directional_spread(
                np.nan_to_num(Zmem[-1]), rec.dx, rec.dy)
            om_p = np.sqrt(G * kp * np.tanh(kp * depth))
            pdir, _ = resolve_direction(Zmem, rec, kp, pdir_ax, om_p / kp)
            h = -np.array([np.cos(np.radians(pdir)), np.sin(np.radians(pdir))])
            T_tr = W_MAX + cfg['lead_max']

            def ship_pos(t, h=h, cx=cx, cy=cy, t0=t0, T_tr=T_tr, U=U):
                return np.array([cx, cy]) + U * (t - (t0 - W_MAX) - T_tr / 2) * h

            r0, r1 = cfg['r_fan']
            P_all, z_all = [], []
            for j in range(nfw):
                t_j = float(rec.t[i0 - nfw + 1 + j])
                sp = ship_pos(t_j)
                RX, RY = Xg - sp[0], Yg - sp[1]
                R = np.hypot(RX, RY)
                ca = (RX * h[0] + RY * h[1]) / np.maximum(R, 1e-9)
                m = ((R >= r0) & (R <= r1) & (ca >= 0.5) & np.isfinite(Zmem[j]))
                idx = np.flatnonzero(m)
                if len(idx) == 0:
                    continue
                pick = rng.choice(idx, min(12, len(idx)), replace=False)
                iy, ix = np.unravel_index(pick, Zmem[j].shape)
                P_all.append(np.column_stack([rec.gx[ix], rec.gy[iy],
                                              np.full(len(pick), t_j)]))
                z_all.append(Zmem[j][iy, ix] + rng.normal(0, 0.02, len(pick)))
            P_all = np.vstack(P_all)
            z_all = np.concatenate(z_all)
            m = P_all[:, 2] >= t0 - W
            P, z = P_all[m], z_all[m]
            if len(z) > 600:
                ii = np.random.default_rng(0).choice(len(z), 600, False)
                P, z = P[ii], z[ii]
            Pq = np.array([[*ship_pos(t0 + l_), t0 + l_] for l_ in leads])
            gt_q = np.array([bilinear(Zfut[int(round(l_ / rec.dt))],
                                      rec.gx, rec.gy, px, py)
                             for (px, py, _t), l_ in zip(Pq, leads)])
            out.append(dict(record=name, wi=wi, P=P, z=z, Pq=Pq, gt_q=gt_q,
                            depth=depth, kp=float(kp), Hs=float(Hs),
                            leads=leads, seed=i0))
    return out


def score(net_obj, w):
    fq = net_gamma_mmse(net_obj, w['P'], w['z'], w['depth'], w['Pq'],
                        0.05 * w['Hs'], seed=w['seed'], kp_hint=w['kp'],
                        use_em=False)
    v = np.isfinite(w['gt_q']) & (w['leads'] <= 10.0)
    return float(np.corrcoef(fq[v], w['gt_q'][v])[0, 1])


def main():
    net = get_net()
    hand = {(c['record'], c['wi']): c['realized'] for c in json.load(
        open('results/direction3/val_gate_probe.json'))['cells']}
    wins = build_windows()
    print(f"실측 창 {len(wins)}개 구성 완료", flush=True)

    # per-window net gamma map + mu_hat
    maps, mus = [], []
    for w in wins:
        X = make_X(w['P'], w['z'], w['depth'], w['kp'])
        with torch.no_grad():
            Ylog = net(torch.tensor(X[None], device=DEVICE))[0].cpu().numpy()
        bx, by = beamform_top(w['P'].astype(np.float32),
                              w['z'].astype(np.float32), w['depth'],
                              klim=3.5 * w['kp'], n=120)
        maps.append(Ylog)
        mus.append(float(np.arctan2(by, bx)))
    aligned = np.stack([rotate_map(maps[i], -mus[i]) for i in range(len(wins))])
    mean_aligned = aligned.mean(0)
    print(f"평균 정렬맵 완성 (n={len(wins)}), mu_hat "
          f"{np.rad2deg(np.array(mus)).round(0)}", flush=True)

    cnet = ConstNet(mean_aligned)
    rows = []
    for i, w in enumerate(wins):
        s_net = score(net, w)
        cnet.mu = mus[i]
        s_const = score(cnet, w)
        cnet.mu = 0.0
        s_noal = score(cnet, w)
        rows.append(dict(record=w['record'], wi=w['wi'], net=s_net,
                         const=s_const, noalign=s_noal,
                         hand=hand.get((w['record'], w['wi']))))
        r = rows[-1]
        print(f"  {w['record']:6s} win{w['wi']:2d}  net {s_net:+.3f}  "
              f"const {s_const:+.3f}  noalign {s_noal:+.3f}  "
              f"hand {r['hand']:+.3f}", flush=True)

    from scipy import stats
    a = lambda k: np.array([r[k] for r in rows], float)
    net_, con, noa, hnd = a('net'), a('const'), a('noalign'), a('hand')
    print(f"\n=== 실측 conditionality (n={len(rows)}창, 2레코드) ===")
    print(f"  net    {net_.mean():+.4f}")
    print(f"  const  {con.mean():+.4f}   (net의 자기 평균맵 + 방향 회전)")
    print(f"  noalign{noa.mean():+.4f}   (회전 없음 = 방향 읽기 제거)")
    print(f"  hand   {hnd.mean():+.4f}")
    d = con - net_
    t, p = stats.ttest_rel(con, net_)
    se = d.std(ddof=1) / np.sqrt(len(d))
    print(f"\n  ★const − net = {d.mean():+.4f} ± {se:.4f}  t={t:+.2f} p={p:.3f}  "
          f"승 {int((con > net_).sum())}/{len(d)}")
    print(f"     -> {'구분 불가 => 합성서의 폐쇄가 실측에도 전이' if p > 0.1 else '★유의한 차이 => net은 constant가 아니다. 폐쇄는 아레나 특유'}")
    print(f"  방향 읽기의 가치 = {con.mean() - noa.mean():+.4f}")
    t2, p2 = stats.ttest_rel(net_, hnd)
    print(f"  net − hand = {(net_-hnd).mean():+.4f}  t={t2:+.2f} p={p2:.3f}  "
          f"승 {int((net_>hnd).sum())}/{len(net_)}")
    t3, p3 = stats.ttest_rel(con, hnd)
    print(f"  ★const − hand = {(con-hnd).mean():+.4f}  t={t3:+.2f} p={p3:.3f}  "
          f"승 {int((con>hnd).sum())}/{len(con)}")
    print(f"     -> constant prior가 실측 헤드라인(net>hand)을 "
          f"{'재현함' if (con-hnd).mean() > 0.5*(net_-hnd).mean() else '재현 못 함'}")
    json.dump(rows, open(OUT, 'w'), indent=1)
    print(f"saved -> {OUT}")


if __name__ == '__main__':
    main()
