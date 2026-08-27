"""B-26 전제 확인 — ★**사전등록의 모든 수를 코드가 낸다**(손복사 0).

⛔이것은 **결과가 아니라 전제**다. 추정기는 한 번도 돌지 않는다.
산출: `premise_b26.json`.  ⚠`results*.jsonl` 로 이름 짓지 않는다 — 판정 자료가 아니다.

    python3 tmp/B26_SEASTATE_2026-08-27/premise_b26.py

전제 4건:
 P-1 BF2/BF3 **혼잡** 에피소드가 저장소에 몇 개 있는가 (전수)
 P-2 실측 12 레코드의 해상상태 **재계산** (`scene_station.py` 규약 축자)
 P-3 `B-12` HOS 생성기가 짧은 주기를 담을 수 있는가 (스펙트럼 나이퀴스트 · 메시 판정)
 P-4 배포 사전의 **학습 Tp 분포**에서 실측 중앙이 어디인가
"""
import glob
import json
import os
import re
import sys
import warnings

warnings.filterwarnings('ignore')
import numpy as np                                             # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
REPO = os.path.abspath(os.path.join(ROOT, '..'))
for _p in (HERE, ROOT, os.path.join(ROOT, 'scripts'),
           os.path.join(ROOT, 'tmp', 'B2_REAL_2026-08-23')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(ROOT)

import gates_b26 as G                                          # noqa: E402

G_ACC = 9.81
W_MAX = 40.0            # scene_station.py:82  W_max = 40.0
LEAD_MAX = 14.0         # scene_station.py:85  lm = 14.0
N_WINDOWS = 6           # scene_station.py:94  np.linspace(lo, hi, 6)


# ── P-1 ──────────────────────────────────────────────────────────────────────
def p1_corpus_census():
    cnt, crowded = {}, []
    for p in glob.glob(os.path.join(REPO, '**', 'episode_BF*.h5'), recursive=True):
        m = re.match(r'episode_(BF\d)_(\d+)comp_', os.path.basename(p))
        if not m:
            continue
        bf, nc = m.group(1), int(m.group(2))
        cnt[f'{bf}_{nc}comp'] = cnt.get(f'{bf}_{nc}comp', 0) + 1
        if bf in ('BF2', 'BF3') and nc >= 15:
            crowded.append(p)
    return dict(census=dict(sorted(cnt.items())),
                bf23_crowded_count=len(crowded), bf23_crowded=crowded[:5],
                bf2_total=sum(v for k, v in cnt.items() if k.startswith('BF2')),
                bf3_total=sum(v for k, v in cnt.items() if k.startswith('BF3')))


# ── P-2 ──────────────────────────────────────────────────────────────────────
def p2_real_seastate():
    """`scene_station.scene_station()` 의 규약 축자: 같은 12 레코드 · 같은 창 · 같은 Hs 정의."""
    from d3_sa_multirecord import MRecord
    from realdata_analysis import directional_spread, peak_period
    from scene_station import RECORDS_12

    out = {}
    for name, cfg in RECORDS_12.items():
        rec = MRecord(os.path.join(ROOT, cfg['path']), stride=cfg['stride'])
        depth = cfg['depth']
        nfw = int(round(W_MAX / rec.dt))
        n_lead = int(round(LEAD_MAX / rec.dt))
        if rec.T_all < nfw + n_lead + 4:
            n_lead = max(2, rec.T_all - nfw - 4)
        lo, hi = nfw + 1, rec.T_all - n_lead - 2
        idxs = np.linspace(lo, hi, N_WINDOWS).astype(int)
        per = []
        for win in range(N_WINDOWS):
            i0 = int(idxs[win])
            Zmem = rec.window(i0 - nfw + 1, nfw)
            Hs = 4.0 * float(np.nanstd(Zmem))                  # scene_station.py:100
            _, lam_p, _, kp = directional_spread(np.nan_to_num(Zmem[-1]),
                                                 rec.dx, rec.dy)  # :101
            w = float(np.sqrt(G_ACC * kp * np.tanh(np.clip(kp * depth, 0, 20)))) if kp > 0 else np.nan
            tp_t, _ = peak_period(Zmem, rec.t[i0 - nfw + 1:i0 + 1])
            per.append(dict(win=win, Hs=Hs, kp=float(kp), lam_p=float(lam_p),
                            Tp_disp=float(2 * np.pi / w) if w > 0 else float('nan'),
                            Tp_temporal=float(tp_t),
                            Hs_over_lam=float(Hs / lam_p), k_Hs=float(kp * Hs)))
        med = {k: float(np.median([q[k] for q in per]))
               for k in ('Hs', 'Tp_disp', 'Tp_temporal', 'lam_p', 'Hs_over_lam', 'k_Hs')}
        out[name] = dict(depth=depth, box_x=float(rec.gx.max() - rec.gx.min()),
                         box_y=float(rec.gy.max() - rec.gy.min()),
                         dt=float(rec.dt), T_all=int(rec.T_all), median=med, windows=per)
    agg = {k: float(np.median([v['median'][k] for v in out.values()]))
           for k in ('Hs', 'Tp_disp', 'Tp_temporal', 'lam_p', 'Hs_over_lam', 'k_Hs')}
    rng = {k: [float(min(v['median'][k] for v in out.values())),
               float(max(v['median'][k] for v in out.values()))]
           for k in ('Hs', 'Tp_disp', 'Tp_temporal', 'lam_p')}
    return dict(records=out, median_over_records=agg, range_over_records=rng,
                n_records=len(out),
                # ⚠LJ18a 는 Hs 8.68 m · Hs/λ 0.215 = 물리적으로 불가 ⟹ 클램프 인공물 의심.
                # ⛔사후배제하지 않는다. 참고용으로 뺀 값도 같이 낸다.
                median_without_LJ18a={
                    k: float(np.median([v['median'][k] for n, v in out.items()
                                        if n != 'LJ18a']))
                    for k in ('Hs', 'Tp_disp', 'Tp_temporal', 'lam_p', 'Hs_over_lam')})


# ── P-3 ──────────────────────────────────────────────────────────────────────
def p3_hos_capacity():
    """`gen_hos` 의 HOS 격자가 각 해상상태를 담는가. ⛔`gen_hos` 를 import 만 하고 안 돌린다."""
    from wavesim import mesh_adequacy
    from wavesim.wave_generator import jonswap_components
    sys.path.insert(0, os.path.join(ROOT, 'tmp', 'HOSGATE_2026-08-13'))
    import gen_hos as GH
    kxn = np.pi * GH.HOS_NX / GH.HOS_LX
    kyn = np.pi * GH.HOS_NY / GH.HOS_LY
    cfg = G._cfg()
    SS = dict(BF2=(0.3, 4.0), BF3=(0.6, 5.5), BF4=(1.2, 7.0), BF5=(2.0, 9.0),
              **{c: (v['Hs'], v['Tp']) for c, v in G.CORNERS.items()})
    out = {}
    for name, (hs, tp) in SS.items():
        for nc in G.NCOMPS:
            al, ae, verd_lin, verd_hos, vu = [], [], [], [], []
            for seed in G.SEED_POOL[:12]:
                comp = jonswap_components(Hs=hs, Tp=tp, n_components=nc, depth=50.0,
                                          seed=seed, random_amplitudes=True,
                                          random_frequencies=True,
                                          stream_tag=f'{name}_{nc}')
                A = np.array([c.A for c in comp])
                kx = np.abs([c.kx for c in comp])
                ky = np.abs([c.ky for c in comp])
                m = (kx > kxn) | (ky > kyn)
                al.append(int(m.sum()))
                ae.append(float((A[m] ** 2).sum() / (A ** 2).sum()))
                verd_lin.append(mesh_adequacy.check(comp, cfg, hos_order=None)['mesh_verdict'])
                i3 = mesh_adequacy.check(comp, cfg, hos_order=GH.HOS_M)
                verd_hos.append(i3['mesh_verdict'])
                vu.append(float(i3['var_under_resolved']))
            from collections import Counter
            out[f'{name}|{nc}'] = dict(
                Hs=hs, Tp=tp, aliased_components=[min(al), max(al)],
                aliased_energy_frac=[float(min(ae)), float(max(ae))],
                mesh_linear=dict(Counter(verd_lin)), mesh_hos_M3=dict(Counter(verd_hos)),
                var_under_hos_median=float(np.median(vu)))
    return dict(hos_nyquist_kx=float(kxn), hos_nyquist_ky=float(kyn),
                hos_grid=[GH.HOS_NX, GH.HOS_NY], hos_L=[GH.HOS_LX, GH.HOS_LY],
                hos_M=GH.HOS_M, mesh=[G.MESH_NX, G.MESH_NY], cells=out)


def main():
    r = dict(batch='B-26', kind='premise (NOT results)',
             P1_corpus=p1_corpus_census(),
             P2_real_seastate=p2_real_seastate(),
             P3_hos_capacity=p3_hos_capacity(),
             P4_training_tp=G.train_seastate_facts(),
             corners=G.CORNERS, gen_params=G.gen_params(),
             thresholds=G.threshold_table())
    json.dump(r, open(os.path.join(HERE, 'premise_b26.json'), 'w'),
              indent=1, ensure_ascii=False)
    p2 = r['P2_real_seastate']
    print('=== P-1 BF2/BF3 혼잡 에피소드 ===')
    print(f'  ncomp>=15 인 BF2/BF3: **{r["P1_corpus"]["bf23_crowded_count"]}개** '
          f'(BF2 총 {r["P1_corpus"]["bf2_total"]} · BF3 총 {r["P1_corpus"]["bf3_total"]})')
    print('=== P-2 실측 12 레코드 해상상태 (재계산) ===')
    for k, v in p2['median_over_records'].items():
        print(f'  {k:12s} 중앙 {v:8.4f}   범위 {p2["range_over_records"].get(k)}')
    print('=== P-3 HOS 수용력 ===')
    for k, v in r['P3_hos_capacity']['cells'].items():
        if k.endswith('|15'):
            print(f'  {k:16s} alias {v["aliased_components"]} '
                  f'E {100*v["aliased_energy_frac"][1]:5.2f}% · lin {v["mesh_linear"]} '
                  f'· HOS-M3 {v["mesh_hos_M3"]}')
    print('=== P-4 학습 Tp 분포 ===')
    print(f'  {r["P4_training_tp"]}')


if __name__ == '__main__':
    main()
