"""M-5 전제 확인 — ★**사전등록의 모든 수를 코드가 낸다**(손복사 0).

⛔이것은 **결과가 아니라 전제**다. 추정기는 한 번도 돌지 않는다. GPU 0회 · 생성 0회.
산출 = `premise_m5.json`.  ⚠`results*.jsonl` 로 이름 짓지 않는다 — 판정 자료가 아니다.

    python3 tmp/M5_WINDOW_2026-08-28/premise_m5.py

전제 6건:
 P-1 ★**수준 허용성** — 각 (record, win, level) 에서 cap 이전 관측 후보가 `m_cap` 이상인가.
     ⛔하나라도 미달이면 그 수준은 **개수 교락**을 만든다 ⟹ 동결 **전에** 수준을 고친다.
 P-2 ★**미공시 불일치의 크기** — 실측 하네스의 `W=40 s` 가 레코드별 몇 `Tp` 인가.
 P-3 규약 상수 되읽기 (배포 `tp_factor` · 학습 `TOBS_FACTOR` · `M_OBS` 3중 일치).
 P-4 ★**MDE/검정력** — 부호검정 n=12(문턱 10)·n=36(문턱 24)이 무엇을 못 재는가.
 P-5 참 제로필드 기준 (⛔`ZERO_REF=1.0` 을 쓰지 않기 위해; **부수 서술 전용**).
 P-6 비용 — `B-2` 동결 행의 벽시계에서 456런의 CPU 예산을 추정한다.
"""
import json
import math
import os
import sys
import warnings

warnings.filterwarnings('ignore')
import numpy as np                                              # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
for _p in (HERE, ROOT, os.path.join(ROOT, 'scripts'),
           os.path.join(ROOT, 'tmp', 'B2_REAL_2026-08-23'),
           os.path.join(ROOT, 'tmp', 'B3_REFIT_2026-08-23')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(ROOT)

import gates_m5 as G                                            # noqa: E402

G_ACC = G.G_ACC
W_MAX = 40.0            # scene_station.py:82  `W_max = 40.0` (메모리 적재 길이, 고정)
LEAD_MAX = 14.0         # scene_station.py:85
N_WINDOWS_GRID = 6      # scene_station.py:94  np.linspace(lo, hi, 6)


def _tp_from_kp(kp, depth):
    if not (kp > 0):
        return float('nan')
    w = math.sqrt(G_ACC * kp * math.tanh(min(kp * depth, 20.0)))
    return 2 * math.pi / w if w > 0 else float('nan')


def level_seconds(level, tp):
    """`gates_m5.LEVELS` 의 한 수준이 이 (record,win) 에서 몇 초인가."""
    spec = G.LEVELS[level]
    if spec['kind'] == 'sec':
        return float(spec['value'])
    if spec['kind'] == 'tp':
        return float(spec['value']) * float(tp)
    return float('inf')


def p1_p2_p5():
    from d3_sa_multirecord import MRecord
    from realdata_analysis import directional_spread
    from scene_station import RECORDS_12

    recs, viol = {}, []
    for name in G.RECORDS:
        cfg = RECORDS_12[name]
        depth = float(cfg['depth'])
        rec = MRecord(os.path.join(ROOT, cfg['path']), stride=cfg['stride'])
        nfw = int(round(W_MAX / rec.dt))
        n_lead = int(round(LEAD_MAX / rec.dt))
        if rec.T_all < nfw + n_lead + 4:
            n_lead = max(2, (rec.T_all - nfw - 4))
        lo, hi = nfw + 1, rec.T_all - n_lead - 2
        idxs = np.linspace(lo, hi, N_WINDOWS_GRID).astype(int)

        per = {}
        for win in G.WINDOWS:
            i0 = int(idxs[win])
            Zmem = rec.window(i0 - nfw + 1, nfw)
            sg = float(np.nanstd(Zmem))
            _, lam_p, _, kp = directional_spread(np.nan_to_num(Zmem[-1]),
                                                 rec.dx, rec.dy)
            tp = _tp_from_kp(float(kp), depth)
            # 프레임별 유효 점 수 → 프레임당 실제 추출 수 min(m_scan, valid)
            picks = np.minimum(np.isfinite(Zmem).reshape(nfw, -1).sum(1), G.M_SCAN)
            lv = {}
            for level in G.LEVEL_ORDER:
                Ws = level_seconds(level, tp)
                nf = min(nfw, int(math.floor(Ws / rec.dt)) + 1)
                pool = int(picks[nfw - nf:].sum())
                adm = pool >= G.M_CAP
                lv[level] = dict(W_s=Ws, W_over_Tp=Ws / tp, n_frames=nf,
                                 pool=pool, admissible=bool(adm))
                if not adm:
                    viol.append((name, win, level, pool))
            # P-5 참 제로필드 (lead 0) — ⛔부수 서술 전용
            Zfut = rec.window(i0, 1)
            sq = max(1, int(round(4.0 / rec.dx)))
            g0 = np.asarray(Zfut[0][::sq, ::sq], float)
            mm = np.isfinite(g0)
            zero_ref = float(np.sqrt(np.mean(g0[mm] ** 2)) / sg) if mm.any() else None
            per[win] = dict(kp=float(kp), lam_p=float(lam_p), Tp=tp, sigma=sg,
                            Hs=4.0 * sg, zero_ref_true=zero_ref,
                            min_picks_per_frame=int(picks.min()),
                            levels=lv)
        recs[name] = dict(depth=depth, dt=float(rec.dt), T_all=int(rec.T_all),
                          box_x=float(rec.gx.max() - rec.gx.min()),
                          box_y=float(rec.gy.max() - rec.gy.min()),
                          windows=per)
        print(f'  {name:7s} dt={rec.dt:.4f} '
              + ' '.join(f'w{w}:Tp={per[w]["Tp"]:.2f},40/Tp={W_MAX / per[w]["Tp"]:.1f}'
                         for w in G.WINDOWS), flush=True)

    anch = [recs[n]['windows'][G.PRIMARY_WIN]['levels']['ANCHOR']['W_over_Tp']
            for n in G.RECORDS]
    allw = [recs[n]['windows'][w]['levels']['ANCHOR']['W_over_Tp']
            for n in G.RECORDS for w in G.WINDOWS]
    p2 = dict(anchor_W_over_Tp_primary_win=dict(
                  median=float(np.median(anch)), min=float(min(anch)),
                  max=float(max(anch)), n=len(anch)),
              anchor_W_over_Tp_all_windows=dict(
                  median=float(np.median(allw)), min=float(min(allw)),
                  max=float(max(allw)), n=len(allw)),
              train_support=list(G.TOBS_FACTOR_TRAIN),
              deploy_factor=G.TP_FACTOR_DEPLOY,
              # ★"몇 배 밖인가" = 앵커 중앙 / 학습 상한
              median_factor_outside_train_support=float(
                  np.median(anch) / G.TOBS_FACTOR_TRAIN[1]),
              n_records_outside_support=int(sum(
                  1 for x in anch if x > G.TOBS_FACTOR_TRAIN[1])))
    return recs, viol, p2


def p3_constants():
    return dict(scene_defaults=G.SCENE_DEFAULTS,
                W_anchor=G.W_ANCHOR, m_scan=G.M_SCAN, m_cap=G.M_CAP,
                deploy_tp_factor=G.TP_FACTOR_DEPLOY,
                train_tobs_factor=list(G.TOBS_FACTOR_TRAIN),
                m_obs_train=G.M_OBS_TRAIN, m_obs_cond=G.M_OBS_COND,
                m_obs_cap_learned=G.M_OBS_CAP_LEARNED,
                windows=list(G.WINDOWS), records=list(G.RECORDS),
                seed=G.SEED, noise=G.NOISE, n_runs=G.N_RUNS)


def p4_power():
    def power(n, thr, q):
        return sum(math.comb(n, k) * q ** k * (1 - q) ** (n - k)
                   for k in range(thr, n + 1))
    qs = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    return dict(
        primary=dict(n=G.N_PRIMARY, threshold=G.THRESHOLD_12,
                     alpha=G.sign_test_one_sided(G.THRESHOLD_12, G.N_PRIMARY),
                     floor_p=G.sign_test_one_sided(G.N_PRIMARY, G.N_PRIMARY),
                     power={str(q): power(G.N_PRIMARY, G.THRESHOLD_12, q) for q in qs},
                     mde_q_at_power80=next((q for q in qs
                                            if power(G.N_PRIMARY, G.THRESHOLD_12, q)
                                            >= 0.80), None)),
        secondary=dict(n=G.N_SECONDARY, threshold=G.THRESHOLD_36,
                       alpha=G.sign_test_one_sided(G.THRESHOLD_36, G.N_SECONDARY),
                       power={str(q): power(G.N_SECONDARY, G.THRESHOLD_36, q)
                              for q in qs},
                       mde_q_at_power80=next((q for q in qs
                                              if power(G.N_SECONDARY, G.THRESHOLD_36, q)
                                              >= 0.80), None),
                       caveat='⛔n=36 은 한 레코드의 창 3개를 겹쳐 센다 = 가짜복제. '
                              'PRIMARY 로 쓰지 않는다.'))


def p6_cost():
    rows = G._load_jsonl(G.FROZEN_B2)
    per = {}
    for arm in G.ARM_ORDER:
        v = [r['wall_s'] for r in rows if r.get('cell') == arm and 'error' not in r]
        per[arm] = float(np.median(v)) if v else None
    n_arm = G.N_RUNS / len(G.ARM_ORDER)
    tot = sum(per[a] * n_arm for a in G.ARM_ORDER if per[a])
    return dict(median_wall_s_from_B2=per, n_runs=G.N_RUNS,
                serial_minutes=tot / 60.0,
                shard6_minutes=tot / 60.0 / 6, gpu_runs=0, episodes_generated=0)


def main():
    print('P-1/P-2/P-5 (레코드 적재 — 추정기 0회) ...', flush=True)
    recs, viol, p2 = p1_p2_p5()
    out = dict(batch=G.BATCH, kind='premise (결과 아님)',
               P1_level_admissibility=dict(records=recs, violations=viol,
                                           all_admissible=not viol),
               P2_window_mismatch=p2, P3_constants=p3_constants(),
               P4_power=p4_power(), P6_cost=p6_cost())
    with open(os.path.join(HERE, 'premise_m5.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    print('\n=== P-1 허용성 ===')
    print(f'  위반 {len(viol)} 건: {viol[:8]}')
    print('=== P-2 창 불일치 (primary win) ===')
    print('  ' + json.dumps(p2['anchor_W_over_Tp_primary_win']))
    print(f'  학습 지지집합 {p2["train_support"]}×Tp · 배포 {p2["deploy_factor"]}×Tp')
    print(f'  중앙 {p2["median_factor_outside_train_support"]:.2f}배 밖 · '
          f'{p2["n_records_outside_support"]}/{len(G.RECORDS)} 레코드가 밖')
    print('=== P-4 MDE ===')
    p4 = out['P4_power']
    print(f'  n=12 문턱 {p4["primary"]["threshold"]} α={p4["primary"]["alpha"]:.4f} '
          f'MDE(q@80%)={p4["primary"]["mde_q_at_power80"]}')
    print(f'  n=36 문턱 {p4["secondary"]["threshold"]} α={p4["secondary"]["alpha"]:.4f} '
          f'MDE(q@80%)={p4["secondary"]["mde_q_at_power80"]}')
    print('=== P-6 비용 ===')
    print('  ' + json.dumps(out['P6_cost']))


if __name__ == '__main__':
    main()
