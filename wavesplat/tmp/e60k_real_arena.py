"""E-B2K real leg -- 6 site-day clusters, 3 arms x 3 seeds.

Pre-registration: tmp/E60K_PREREG_v3_FROZEN_2026-07-19.md sections 2 and 2.1.

HONESTY LABEL (frozen, prereg section 1): this leg is NOT independent
confirmation. T and L reuse 2026-07-12 checkpoints, so the five existing
records' deltas were already known when the bar was set. The leg is
    "posterior direction-consistency summary of 5 known records
     + ONE prospective test (BS11a)".
BS11a is included unconditionally, whatever its sign. Its blocker is frozen:
Delta(T-L) <= -0.05 on BS11a blocks any PASS (max PARTIAL).

BS11a geometry is fixed by ONE formula chosen before the result existed --
the AA records' dimensionless ratios transplanted:
    r_fan = (0.127 lam, 0.635 lam),  U = 0.121 c_p,  lead_max = 3.1 Tp
with lam = 11.5 m, c_p = 4.24 m/s, Tp = 2.6 s for BS11a  ->
    r_fan = (1.46, 7.30) m,  U = 0.51 m/s,  lead_max = 8.1 s.
"""
import json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from d3_sa_multirecord import MRecord, RECORDS as SA_RECORDS   # noqa: E402
from d3_gate_promotion_eval import window_data, RECS           # noqa: E402
from d3_falcon_real_operator import RECORDS                    # noqa: E402
from d3_prior_net_eval import get_net, net_gamma_mmse, DEVICE  # noqa: E402
from d3_l1_s2head_train import SmultHead                       # noqa: E402
from d3_gpu_mmse import mmse_gpu                               # noqa: E402

OUT = 'results/direction3/e60k_real_arena.json'
REF = 'results/direction3/gate_promotion_eval.json'
ARMS = {f'{a}_s{s}': f'results/direction3/b2deep_{a}_s{s}.pt'
        for a in ('T', 'L', 'N') for s in (0, 1, 2)}

# FROZEN prospective record (pinned by FILENAME -- the BS11a/BS11b label swap
# was corrected 2026-07-19; the in-range file, patch/lambda 1.22, is the Oct-4
# one). site-day = BS_2011-10-04, a new cluster (BS11c/d would collapse into it).
BS11A = dict(path='data/real_bergamasco/BS_2011/Surfaces_20111004_113800_short.nc',
             depth=30.0, stride=1, r_fan=(1.46, 7.30), U=0.51, lead_max=8.1)
SITE_DAY = {'YS17': 'YS_2017-05-13', 'AA15a': 'AA_2015-03-05',
            'AA14': 'AA_2014-03-27', 'AA15b': 'AA_2015-05-15',
            'LJ18g': 'LJ_2018-01-03', 'BS11a': 'BS_2011-10-04'}


def load_model(ck):
    net = get_net(ck)
    head = SmultHead().to(DEVICE)
    head.load_state_dict(torch.load(ck, map_location=DEVICE)['head'])
    head.eval()
    return net, head


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    models = {k: load_model(v) for k, v in ARMS.items()}
    ref = {(r['record'], r['wi']): r for r in json.load(open(REF))}
    assert BS11A['path'] == SA_RECORDS['BS11a']['path'], \
        'BS11a filename does not match the house record table'
    print(f'models {sorted(models)}\nBS11a pinned: '
          f'{os.path.basename(BS11A["path"])}', flush=True)

    cfgs = {n: RECORDS[n] for n in RECS}
    cfgs['BS11a'] = BS11A
    rows = json.load(open(OUT))['rows'] if os.path.exists(OUT) else []
    done = {(r['record'], r['wi']) for r in rows}

    for name, cfg in cfgs.items():
        if only and name != only:
            continue
        rec = MRecord(cfg['path'], stride=cfg['stride'])
        nfw = int(round(40.0 / rec.dt))
        n_lead = int(round(cfg['lead_max'] / rec.dt))
        i0s = np.linspace(nfw + 1, rec.T_all - n_lead - 2, 12).astype(int)
        nk = 0
        for wi, i0 in enumerate(i0s):
            if (name, wi) in done:
                nk += 1
                continue
            t0 = time.time()
            try:
                P, z, Pq, gt_q, leads, Hs, kp, _ = window_data(
                    rec, cfg, int(i0), nfw, n_lead, wi)
            except Exception as e:
                print(f'  [{name} w{wi}] window_data failed: {repr(e)[:70]}')
                continue
            v = np.isfinite(gt_q) & (leads <= 10.0)
            if v.sum() < 5 or len(z) < 20:
                print(f'  [{name} w{wi}] skipped (n_obs {len(z)}, '
                      f'n_query {int(v.sum())})')
                continue
            sig0 = 0.05 * Hs
            out = {'record': name, 'site_day': SITE_DAY[name], 'wi': wi,
                   'n_obs': int(len(z)), 'Hs': float(Hs), 'kp': float(kp)}
            r = ref.get((name, wi))
            if r is not None:                    # frozen hand (5 known records)
                out['hand'] = r['hand']
                out['hand_source'] = 'frozen_ref'
            else:                                # BS11a: computed identically
                with torch.no_grad():
                    ph = mmse_gpu(
                        torch.tensor(P, device=DEVICE, dtype=torch.float32),
                        torch.tensor(z, device=DEVICE, dtype=torch.float32),
                        cfg['depth'],
                        torch.tensor(Pq, device=DEVICE, dtype=torch.float32),
                        sig0, kp_hint=kp).cpu().numpy()
                out['hand'] = float(np.corrcoef(ph[v], gt_q[v])[0, 1])
                out['hand_source'] = 'fresh_mmse_gpu'
            for mk, (net, head) in models.items():
                fq = net_gamma_mmse(net, P, z, cfg['depth'], Pq, sig0,
                                    seed=int(i0), kp_hint=kp, use_em=False,
                                    smult_head=head)
                out[mk] = float(np.corrcoef(fq[v], gt_q[v])[0, 1])
            rows.append(out)
            nk += 1
            tl = np.mean([out[f'T_s{s}'] - out[f'L_s{s}'] for s in (0, 1, 2)])
            tn = np.mean([out[f'T_s{s}'] - out[f'N_s{s}'] for s in (0, 1, 2)])
            print(f'[{name} w{wi}] hand {out["hand"]:+.3f}  T-L {tl:+.4f}  '
                  f'T-N {tn:+.4f}  n_obs {out["n_obs"]:4d} '
                  f'[{time.time() - t0:.0f}s]', flush=True)
        print(f'  -> {name}: {nk}/12 windows', flush=True)
        json.dump({'config': dict(records=list(cfgs), arms=sorted(ARMS),
                                  bs11a=BS11A,
                                  prereg='tmp/E60K_PREREG_v3_FROZEN_2026-07-19.md'),
                   'rows': rows}, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({len(rows)} rows)')


if __name__ == '__main__':
    main()
