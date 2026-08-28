"""M7 전제 측정 (premise, NOT results) — 학습 0회 · 추정 0런 · GPU 0회.

이 파일이 내는 수는 전부 **저장소의 동결 자료/코드**에서 온다. 손복사 상수 0건.
  P1 표현 용량 — SS-REAL 처럼 짧은 바다가 32-원자 코드(K_LIM=1.1)와 빔포밍 조건자(|k|<=0.6)에
      담기는가. ★"필터 항진" 점검: HARD 분포가 `gen_mixed.REPR_MIN_FRAC` 에 구조적으로 걸리면
      그 팔은 비어버린다.
  P2 실측 12 레코드 — B-26 `premise_b26.p2_real_seastate()` 와 **같은 규약**(scene_station 12
      레코드 · 같은 창 · 같은 Hs 정의)으로 재측정하되 B-26 이 버린 **방향 확산**을 함께 낸다.
  P3 지지집합 — 배포 학습 혼합(`gen_mixed._draw_HsTp`)이 실측 (Hs,Tp) 에 얼마나 질량을 주는가.
  P4 비용 — 동결 로그에서 읽는다(새 실행 0회).

사용: python3 tmp/M7_SUPERVISION_2026-08-28/premise_m7.py [--part all|1|2|3|4]
"""
import argparse
import json
import os
import sys
import statistics as st

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
for p in (ROOT, os.path.join(ROOT, 'scripts'), os.path.join(ROOT, 'tmp/mixed_retrain_2026-07-26'),
          os.path.join(ROOT, 'tmp/B2_REAL_2026-08-23')):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(ROOT)

OUT = os.path.join(ROOT, 'tmp/M7_SUPERVISION_2026-08-28/premise_m7.json')
B26_PREMISE = os.path.join(ROOT, 'tmp/B26_SEASTATE_2026-08-27/premise_b26.json')
G_ACC = 9.81
N_SCENES = 400          # P1 표본 (장면당 ~0.01 s)
N_MC = 200000           # P3 몬테카를로


# ── P1 ───────────────────────────────────────────────────────────────────────
def hard_anchors():
    """HARD (Hs,Tp) 앵커 = **B-26 동결 premise 의 레코드별 중앙값 12쌍** (손복사 0)."""
    d = json.load(open(B26_PREMISE))
    recs = d['P2_real_seastate']['records']
    return [(k, float(v['median']['Hs']), float(v['median']['Tp_disp'])) for k, v in sorted(recs.items())]


def p1_capacity():
    import gen_mixed as gm
    from wavesim.wave_generator import wavenumber_from_omega
    anchors = hard_anchors()
    anc = [(h, t) for _, h, t in anchors]

    def draw_hard(rng):
        """HARD: 절반은 실측 12 레코드 중앙값 앵커(±15%, 배포와 같은 규약), 절반은 현행 혼합."""
        if rng.uniform() < gm.ANCHOR_FRAC:
            Hs, Tp = anc[rng.integers(len(anc))]
            j = gm.ANCHOR_JITTER
            return (float(Hs * rng.uniform(1 - j, 1 + j)), float(Tp * rng.uniform(1 - j, 1 + j)))
        return _draw_base(rng)

    _draw_base = gm._draw_HsTp
    bank = gm.GeometryBank(verbose=False)
    out = {}
    for name, draw in (('CURRENT', _draw_base), ('HARD', draw_hard)):
        gm._draw_HsTp = draw
        rng = np.random.default_rng(12345)
        rf, kHs, band06, tps, hss, ntrue = [], [], [], [], [], []
        n_reject = 0
        for _ in range(N_SCENES):
            s = gm.sample_scene_ok(rng, bank)
            if s is None:
                n_reject += 1
                continue
            rf.append(s['repr_frac'])
            if s['repr_frac'] < gm.REPR_MIN_FRAC:
                n_reject += 1          # 20회 재시도 후에도 못 넘은 장면
            tps.append(s['Tp']); hss.append(s['Hs']); ntrue.append(s['n_true'])
            om_p = 2 * np.pi / s['Tp']
            k_p = wavenumber_from_omega(float(om_p), gm.DEPTH)
            kHs.append(k_p * s['Hs'])
            k = np.hypot(s['tkx'], s['tky']); e = s['tA'] ** 2
            band06.append(float(e[k <= 0.6].sum() / max(e.sum(), 1e-18)))
        q = lambda v, p: float(np.quantile(v, p))
        out[name] = dict(
            n=len(rf), n_reject_after_20=n_reject,
            repr_frac=dict(median=q(rf, .5), p10=q(rf, .1), min=float(np.min(rf)),
                           frac_below_min=float(np.mean(np.array(rf) < gm.REPR_MIN_FRAC))),
            target_energy_in_beamform_band_0p6=dict(median=q(band06, .5), p10=q(band06, .1)),
            Tp=dict(median=q(tps, .5), p10=q(tps, .1), min=float(np.min(tps))),
            Hs=dict(median=q(hss, .5), p90=q(hss, .9), max=float(np.max(hss))),
            k_Hs=dict(median=q(kHs, .5), p90=q(kHs, .9)),
            n_true=dict(median=q(ntrue, .5)))
    gm._draw_HsTp = _draw_base
    out['anchors_hard'] = [dict(record=r, Hs=h, Tp=t) for r, h, t in anchors]
    out['K_LIM_HARD'] = gm.K_LIM_HARD
    out['REPR_MIN_FRAC'] = gm.REPR_MIN_FRAC
    out['beamform_KLIM'] = __import__('d3_beamform_net').KLIM
    return out


# ── P2 ───────────────────────────────────────────────────────────────────────
def p2_real_spread():
    """B-26 `p2_real_seastate` 와 같은 규약. ★차이 = 버려졌던 `spread` 를 기록한다."""
    from d3_sa_multirecord import MRecord
    from realdata_analysis import directional_spread, peak_period
    from scene_station import RECORDS_12
    W_MAX, LEAD_MAX, N_WINDOWS = 40.0, 14.0, 6     # scene_station.py:82,85,94 = premise_b26.py:36-38
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
            Hs = 4.0 * float(np.nanstd(Zmem))
            spread, lam_p, pdir, kp = directional_spread(np.nan_to_num(Zmem[-1]), rec.dx, rec.dy)
            tp_t, _ = peak_period(Zmem, rec.t[i0 - nfw + 1:i0 + 1])
            per.append(dict(win=win, Hs=Hs, kp=float(kp), lam_p=float(lam_p),
                            spread_deg=float(spread), peak_dir_deg=float(pdir),
                            k_Hs=float(kp * Hs), Tp_temporal=float(tp_t)))
        out[name] = {k: float(np.median([q[k] for q in per]))
                     for k in ('Hs', 'kp', 'lam_p', 'spread_deg', 'k_Hs', 'Tp_temporal')}
        out[name]['windows'] = per
    agg = {k: float(np.median([v[k] for v in out.values()]))
           for k in ('Hs', 'kp', 'lam_p', 'spread_deg', 'k_Hs')}
    return dict(records=out, median_over_records=agg,
                spread_range=[float(min(v['spread_deg'] for v in out.values())),
                              float(max(v['spread_deg'] for v in out.values()))])


# ── P3 ───────────────────────────────────────────────────────────────────────
def p3_support():
    import gen_mixed as gm
    from wavesim.wave_generator import wavenumber_from_omega
    b26 = json.load(open(B26_PREMISE))
    real = b26['P2_real_seastate']['median_over_records']
    rng = np.random.default_rng(7)
    tp = np.empty(N_MC); hs = np.empty(N_MC)
    for i in range(N_MC):
        hs[i], tp[i] = gm._draw_HsTp(rng)
    kp = np.array([wavenumber_from_omega(float(2 * np.pi / t), gm.DEPTH) for t in tp[:20000]])
    kHs = kp * hs[:20000]
    return dict(real_median_Tp_disp=real['Tp_disp'], real_median_Hs=real['Hs'],
                real_median_k_Hs=real['k_Hs'],
                p_Tp_le_real=float(np.mean(tp <= real['Tp_disp'])),
                p_Hs_le_real=float(np.mean(hs <= real['Hs'])),
                p_both=float(np.mean((tp <= real['Tp_disp']) & (hs <= real['Hs']))),
                p_kHs_ge_real=float(np.mean(kHs >= real['k_Hs'])),
                Tp_box=list(gm.TP_BOX), Hs_box=list(gm.HS_BOX),
                anchors=[list(a) for a in gm.ANCHORS], anchor_frac=gm.ANCHOR_FRAC,
                anchor_jitter=gm.ANCHOR_JITTER, n_mc=int(N_MC))


# ── P4 ───────────────────────────────────────────────────────────────────────
def p4_cost():
    def curve_sec(p):
        d = json.load(open(os.path.join(ROOT, p)))
        c = d['curve'][-1]
        return dict(path=p, steps=c['step'] + 1, sec=c['sec'],
                    ms_per_step=1000.0 * c['sec'] / (c['step'] + 1))
    def jsonl_wall(paths):
        w = []
        for p in paths:
            for line in open(os.path.join(ROOT, p)):
                r = json.loads(line)
                if isinstance(r.get('wall_s'), (int, float)):
                    w.append(r['wall_s'])
        return dict(n=len(w), median_s=st.median(w), total_h=sum(w) / 3600.0)
    import glob
    return dict(
        train_mixed_v1=curve_sec('tmp/mixed_retrain_2026-07-26/train_curve.json'),
        train_mixed_v3=curve_sec('tmp/mixed_retrain_2026-07-26/train_curve_v3.json'),
        finetune_field_A=curve_sec('tmp/wp_e2e_2026-07-25/train_curve_A.json'),
        finetune_field_B=curve_sec('tmp/wp_e2e_2026-07-25/train_curve_B.json'),
        synth_run_B16=jsonl_wall(sorted(glob.glob('tmp/B16_KGRID_2026-08-24/results_b16_*.jsonl'))),
        real_run_B2=jsonl_wall(['tmp/B2_REAL_2026-08-23/results.jsonl']))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', default='all')
    a = ap.parse_args()
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    res['batch'] = 'M7_SUPERVISION_2026-08-28'
    res['kind'] = 'premise (NOT results)'
    if a.part in ('all', '4'):
        res['P4_cost'] = p4_cost(); print('P4 ok')
    if a.part in ('all', '3'):
        res['P3_support'] = p3_support(); print('P3 ok')
    if a.part in ('all', '1'):
        res['P1_capacity'] = p1_capacity(); print('P1 ok')
    if a.part in ('all', '2'):
        res['P2_real'] = p2_real_spread(); print('P2 ok')
    json.dump(res, open(OUT, 'w'), indent=1)
    print('wrote', OUT)
