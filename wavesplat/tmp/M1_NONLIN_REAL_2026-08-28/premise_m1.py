"""M1 전제 — **이 배치가 실측한다**. 사전등록의 모든 수가 이 파일 출력에서 온다(손복사 0).

    python3 tmp/M1_NONLIN_REAL_2026-08-28/premise_m1.py   ->  premise_m1.json
"""
import collections
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import gates_m1 as G                                           # noqa: E402
from wavesim import mesh_adequacy                              # noqa: E402

OUT = os.path.join(HERE, 'premise_m1.json')
G_ACC = 9.81


def p1_nyquist():
    """P-1 나이퀴스트 — **격자를 올린 사유를 산술로**."""
    eps, _ = G.select()
    seeds = sorted({(e['ncomp'], e['seed']) for e in eps})
    kx = ky = kk = 0.0
    lam_min = 1e9
    for nc, s in seeds:
        c = G.components(nc, s)
        kx = max(kx, max(abs(x.kx) for x in c))
        ky = max(ky, max(abs(x.ky) for x in c))
        k = max(np.hypot(x.kx, x.ky) for x in c)
        kk = max(kk, k)
        lam_min = min(lam_min, 2 * np.pi / k)
    wp = 2 * np.pi / G.TP
    return dict(Tp=G.TP, omega_p=wp, omega_max=3 * wp,
                k_peak=float(wp ** 2 / G_ACC), lambda_peak=float(2 * np.pi / (wp ** 2 / G_ACC)),
                k_max_theory=G.k_max_theory(),
                lambda_min_theory=float(2 * np.pi / G.k_max_theory()),
                k_max_observed=float(kk), kx_max_observed=float(kx),
                ky_max_observed=float(ky), lambda_min_observed=float(lam_min),
                nyquist_B12=G.nyquist(G.B12_HOS_NX, G.B12_HOS_NY),
                nyquist_M1=G.nyquist(G.HOS_NX, G.HOS_NY),
                shortfall_B12=float(G.k_max_theory() / G.nyquist(G.B12_HOS_NX, G.B12_HOS_NY)[0]),
                mission_option_a=dict(grid=[384, 256], L=[600.0, 400.0],
                                      nyquist=G.nyquist(384, 256, 600.0, 400.0),
                                      sufficient=bool(min(G.nyquist(384, 256, 600.0, 400.0))
                                                      >= G.k_max_theory())),
                mission_option_b=dict(grid=[192, 128], L=[300.0, 200.0],
                                      nyquist=G.nyquist(192, 128, 300.0, 200.0),
                                      sufficient=bool(min(G.nyquist(192, 128, 300.0, 200.0))
                                                      >= G.k_max_theory())),
                chosen=dict(grid=[G.HOS_NX, G.HOS_NY], L=[G.HOS_LX, G.HOS_LY],
                            nyquist=G.nyquist(G.HOS_NX, G.HOS_NY),
                            sufficient=bool(min(G.nyquist(G.HOS_NX, G.HOS_NY))
                                            >= G.k_max_theory()),
                            half_bin=G.HALF_BIN))


def p2_mesh():
    """P-2 메시 — M=3 화면이 `B-26` 의 M=1 화면과 **비트 등가**임을 실측."""
    d_m1, d_b26 = G.mesh_d(G.cfg_m1(False)), G.mesh_d(G.G26._cfg())
    rows = {}
    for tag, nx, ny, m in (('B-26 (200x167, M=1)', G.G26.MESH_NX, G.G26.MESH_NY, None),
                           ('B-26 격자에 M=3', G.G26.MESH_NX, G.G26.MESH_NY, G.HOS_M),
                           ('M1 (598x499, M=3)', G.MESH_NX, G.MESH_NY, G.HOS_M)):
        cfg = G.cfg_m1(False)
        cfg.mesh.nx, cfg.mesh.ny = nx, ny
        v, u = [], []
        for nc in G.NCOMPS:
            for s in list(G.SEED_POOL)[:G.K_SEEDS + 4]:
                info = mesh_adequacy.check(G.components(nc, s), cfg, hos_order=m)
                v.append(info['mesh_verdict'])
                u.append(info['var_under_resolved'])
        rows[tag] = dict(mesh=[nx, ny], d=G.mesh_d(cfg), hos_order=m,
                         verdicts=dict(collections.Counter(v)),
                         var_under_median=float(np.median(u)), var_under_max=float(max(u)))
    eps, rej = G.select()
    sel = collections.Counter(e['mesh_verdict'] for e in eps if e['gt_arm'] == 'LIN')
    return dict(d_M1=d_m1, d_B26=d_b26, three_d_M1=3 * d_m1,
                exact_equivalence=bool(abs(3 * d_m1 - d_b26) < 1e-12),
                n_tri_M1=2 * (G.MESH_NX - 1) * (G.MESH_NY - 1),
                n_tri_B26=2 * (G.G26.MESH_NX - 1) * (G.G26.MESH_NY - 1),
                sweep=rows, selected_verdicts=dict(sel),
                rejected=[(r['ncomp'], r['seed'], r['mesh_verdict'], round(r['var_under'], 4))
                          for r in rej])


def p3_steepness():
    """P-3 가파름 — `SS-REAL` 의 k·Hs 와 `B-11`/`B-12` 코퍼스의 k·Hs."""
    def khs(Hs, Tp):
        return float((2 * np.pi / Tp) ** 2 / G_ACC * Hs)
    return dict(SS_REAL=dict(Hs=G.HS, Tp=G.TP, k_Hs=khs(G.HS, G.TP)),
                BF4=dict(Hs=1.2, Tp=7.0, k_Hs=khs(1.2, 7.0)),
                BF5=dict(Hs=2.0, Tp=9.0, k_Hs=khs(2.0, 9.0)),
                ratio_SSREAL_over_BF4=khs(G.HS, G.TP) / khs(1.2, 7.0),
                note='실측 12 레코드의 k·Hs 중앙 0.2445 는 B-26 premise 에서 왔다(재계산 아님)')


def p4_cost():
    """P-4 비용 — ⛔산술이 아니라 **실측**. `probe_cost.jsonl` 을 그대로 싣는다."""
    p = os.path.join(HERE, 'probe_cost.jsonl')
    probes = [json.loads(l) for l in open(p)] if os.path.exists(p) else []
    man = G._load_manifest()
    real = collections.defaultdict(list)
    for v in man.values():
        if v.get('status') == 'ok' and v.get('elapsed_s'):
            real[v['gt_arm']].append(v['elapsed_s'])
    return dict(probes=probes,
                measured_full_episode={k: dict(n=len(v), median_s=float(np.median(v)))
                                       for k, v in real.items()},
                plan=dict(n_pairs=G.N_EP, serial_only=True,
                          note='생성 직렬(GPU 직렬화, B-26 실측 4-way 소득 0) · 추정만 8 샤드'))


def main():
    out = dict(batch='M1_NONLIN_REAL_2026-08-28',
               P1_nyquist=p1_nyquist(), P2_mesh=p2_mesh(),
               P3_steepness=p3_steepness(), P4_cost=p4_cost(),
               frozen_seeds={k: list(v) for k, v in G.FROZEN_SEEDS.items()},
               seeds_consumed=list(G.SEEDS_CONSUMED),
               thresholds=G.threshold_table())
    json.dump(out, open(OUT, 'w'), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1)[:4000])


if __name__ == '__main__':
    main()
