"""★일치 증명 — 수정 후 배포 경로 == 연구 경로 (AA15a w3 / AA15a w5 / YS17 w3).

세 층위로 측정한다 (전부 실행값, 문서 인용 없음):
  L0 부분 인자   : TSHIFT 만 / FLOAT32 만 이 각각 비트 수준에서 무엇을 하는가
  L1 프로토콜 동일: 배포(미캡 스트림, float64 프레임) vs 연구 fit_bank_mmse(sc P,z)
                   -> 예측면 max|Δ| 가 0 이어야 한다
  L2 배포 실물   : FrameData float32 (real_data.py:305 의 실제 규약) 일 때의 잔차
  L3 API 표면    : upd.predict_field == median(...).astype(float32)
"""
import os, sys, json, warnings
warnings.filterwarnings('ignore')
import numpy as np, torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.path.insert(0, os.path.join(ROOT, 'tmp', 'bankmmse_port_2026-07-29'))
os.chdir(ROOT)

import d3_compare_gifs as cg
import make_gif_bank_deployed as M
from wavesplat.optimizer.bank_mmse import fit_bank_mmse

SEEDS = [42, 1042, 2042]
WINDOWS = [('AA15a', 3), ('AA15a', 5), ('YS17', 3)]
OUT = 'tmp/P1_real_path_2026-08-11/reconcile_proof.json'


def research_fits(sc):
    """The published research call: fit on scene_real's capped 600 obs, float64."""
    P, z = np.asarray(sc['P'], float), np.asarray(sc['z'], float)
    s0 = 0.05 * 4.0 * float(np.std(z))
    return [fit_bank_mmse(P, z, float(sc['depth']), s0, kp_hint=None, seed=s)
            for s in SEEDS]


def med(fits, Q):
    return np.median(np.stack([f.render(Q) for f in fits], 0), 0)


results = {}
for rec, win in WINDOWS:
    key = f'{rec}_w{win}'
    print(f'\n########## {key} ##########', flush=True)
    sc = cg.scene_real(rec, win=win)
    sc_full = M.scene_uncapped(rec, win)
    n_full = len(sc_full['z'])
    ii = np.random.default_rng(0).choice(n_full, 600, replace=False)
    cap_ok = bool(np.array_equal(sc_full['P'][ii], sc['P'])
                  and np.array_equal(sc_full['z'][ii], sc['z']))
    print(f'  미캡 {n_full}점; rng(0) 캡이 scene_real 의 600점을 재현: {cap_ok}')

    gx, gy = np.asarray(sc['gxq'], float), np.asarray(sc['gyq'], float)
    Xq, Yq = np.meshgrid(gx, gy)
    leads = [0.0, 1.0, 7.0, float(sc['leads'][-1])]
    t0a = float(sc['Pq'][0, 2])

    R = research_fits(sc)
    ent = {'n_full': n_full, 'cap_reproduces_scene_real': cap_ok,
           'research': {'kp': [f.kp for f in R], 'pexp': [f.pexp for f in R],
                        'smult': [f.smult for f in R], 'vr': [f.vr for f in R]}}

    # ---- L1: 배포(미캡 + float64 프레임) vs 연구 -------------------------
    upd64, t0d64, _, _ = M.estimate(sc, seed=42, sc_obs=sc_full,
                                    dtype=torch.float64)
    ts = t0a - t0d64          # 배포 래퍼가 뺀 상수 시간원점

    # ---- L0: TSHIFT 단독 효과 (비트 수준) --------------------------------
    P0, z0 = np.asarray(sc['P'], float), np.asarray(sc['z'], float)
    s0 = 0.05 * 4.0 * float(np.std(z0))
    Pt = P0.copy(); Pt[:, 2] -= ts
    Rt = [fit_bank_mmse(Pt, z0, float(sc['depth']), s0, kp_hint=None, seed=s)
          for s in SEEDS]
    Q_abs = np.column_stack([Xq.ravel(), Yq.ravel(), np.full(Xq.size, t0a + 7.0)])
    Q_rel = Q_abs.copy(); Q_rel[:, 2] -= ts
    d_ts = float(np.abs(med(R, Q_abs) - med(Rt, Q_rel)).max())
    print(f'  L0 TSHIFT 단독 max|Δ| = {d_ts:.3e}  (t_shift={ts:.4f})')
    ent['L0_tshift_maxabs'] = d_ts
    ent['t_shift'] = ts
    b64 = upd64._primary._bank
    print(f'  배포(f64) kp={[round(f.kp,4) for f in b64]} '
          f'pexp={[f.pexp for f in b64]} smult={[f.smult for f in b64]} '
          f'n_obs={[f.n_obs for f in b64]}')
    l1 = {}
    for L in leads:
        Qa = np.column_stack([Xq.ravel(), Yq.ravel(), np.full(Xq.size, t0a + L)])
        Qd = Qa.copy(); Qd[:, 2] = t0d64 + L
        d = float(np.abs(med(R, Qa) - med(b64, Qd)).max())
        eq = bool(np.array_equal(med(R, Qa), med(b64, Qd)))
        l1[str(L)] = {'maxabs': d, 'array_equal': eq}
        print(f'    L1 lead {L:5.1f}: max|Δ| = {d:.3e}   array_equal={eq}')
    ent['L1_deployed_f64_vs_research'] = l1

    # ---- L1b: 시간원점까지 맞추면 비트 동일한가 --------------------------
    l1b = {}
    for L in leads:
        Qd = np.column_stack([Xq.ravel(), Yq.ravel(),
                              np.full(Xq.size, t0d64 + L)])
        a_, b_ = med(Rt, Qd), med(b64, Qd)
        l1b[str(L)] = {'maxabs': float(np.abs(a_ - b_).max()),
                       'array_equal': bool(np.array_equal(a_, b_))}
        print(f'    L1b lead {L:5.1f} (동일 시간원점): '
              f"max|Δ| = {l1b[str(L)]['maxabs']:.3e}   "
              f"array_equal={l1b[str(L)]['array_equal']}")
    ent['L1b_same_time_origin'] = l1b

    # ---- L2: 배포 실물 (FrameData float32) ------------------------------
    upd32, t0d32, _, _ = M.estimate(sc, seed=42, sc_obs=sc_full)
    b32 = upd32._primary._bank
    l2 = {}
    for L in leads:
        Qa = np.column_stack([Xq.ravel(), Yq.ravel(), np.full(Xq.size, t0a + L)])
        Qd = Qa.copy(); Qd[:, 2] = t0d32 + L
        d = float(np.abs(med(R, Qa) - med(b32, Qd)).max())
        l2[str(L)] = d
    print(f'    L2 float32 프레임 잔차 max|Δ| = '
          f'{max(l2.values()):.3e} (관측 양자화)')
    ent['L2_deployed_f32_vs_research_maxabs'] = l2
    ent['deployed_f32'] = {'pexp': [f.pexp for f in b32],
                           'smult': [f.smult for f in b32],
                           'vr': [f.vr for f in b32]}

    # ---- L3: predict_field API 표면 -------------------------------------
    L = leads[-1]
    Qd = np.column_stack([Xq.ravel(), Yq.ravel(), np.full(Xq.size, t0d64 + L)])
    pf = upd64.predict_field(torch.tensor(gx), torch.tensor(gy),
                             t0d64 + L).cpu().numpy()
    ref = med(b64, Qd).reshape(len(gy), len(gx))
    ent['L3_predict_field_equals_median_f32'] = bool(
        np.array_equal(pf, ref.astype(np.float32)))
    ent['L3_predict_field_vs_f64_maxabs'] = float(np.abs(pf - ref).max())
    print(f'    L3 predict_field == median.astype(f32): '
          f"{ent['L3_predict_field_equals_median_f32']}  "
          f"(f64 대비 {ent['L3_predict_field_vs_f64_maxabs']:.3e} = dtype 규약)")

    # ---- 참고: 수정 전 배포(캡된 600점을 먹임) --------------------------
    updbad, t0bad, _, _ = M.estimate(sc, seed=42)          # 옛 호출 = 이중 캡
    bb = updbad._primary._bank
    Qa = np.column_stack([Xq.ravel(), Yq.ravel(),
                          np.full(Xq.size, t0a + leads[-1])])
    Qd = Qa.copy(); Qd[:, 2] = t0bad + leads[-1]
    g = sc['maps_gt'][-1]; mm = np.isfinite(g)
    sg = float(np.std(sc['maps_gt'][np.isfinite(sc['maps_gt'])]))
    nr_res = float(np.sqrt(np.mean(
        (med(R, Qa).reshape(g.shape)[mm] - g[mm]) ** 2)) / sg)
    nr_fix = float(np.sqrt(np.mean(
        (med(b32, Qd).reshape(g.shape)[mm] - g[mm]) ** 2)) / sg)
    nr_bad = float(np.sqrt(np.mean(
        (med(bb, Qd).reshape(g.shape)[mm] - g[mm]) ** 2)) / sg)
    print(f'  [참고] lead {leads[-1]:.0f} nRMSE  연구 {nr_res:.4f} | '
          f'수정후배포 {nr_fix:.4f} | 수정전배포 {nr_bad:.4f}   '
          f'(수정전 smult={[f.smult for f in bb]})')
    ent['ref_nrmse_last_lead'] = {'research': nr_res, 'deployed_fixed': nr_fix,
                                  'deployed_before_fix': nr_bad,
                                  'lead': leads[-1]}
    results[key] = ent
    json.dump(results, open(OUT, 'w'), indent=1)

print(f'\nwrote {OUT}')
worst = max(v['maxabs'] for e in results.values()
            for v in e['L1_deployed_f64_vs_research'].values())
worstb = max(v['maxabs'] for e in results.values()
             for v in e['L1b_same_time_origin'].values())
alleq = all(v['array_equal'] for e in results.values()
            for v in e['L1b_same_time_origin'].values())
print(f'★ L1  (배포 t_shift 유지) 전체 최대|Δ| = {worst:.3e}   <1e-12: {worst < 1e-12}')
print(f'★ L1b (시간원점 일치)     전체 최대|Δ| = {worstb:.3e}  '
      f'array_equal 전부: {alleq}')
