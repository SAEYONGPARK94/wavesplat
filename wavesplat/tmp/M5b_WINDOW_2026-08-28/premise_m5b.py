"""M-5b 전제 확인 — ★**사전등록의 모든 수를 코드가 낸다**(손복사 0).

⛔이것은 **결과가 아니라 전제**다. 추정기는 한 번도 돌지 않는다. GPU 0회 · 생성 0회.
산출 = `premise_m5b.json`.  ⚠`results*.jsonl` 로 이름 짓지 않는다 — 판정 자료가 아니다.

    python3 tmp/M5b_WINDOW_2026-08-28/premise_m5b.py

전제 8건 (P-1…P-6 은 `M5` 계승, ★**P-7·P-8 이 신설**):
 P-1 ★**수준 허용성** — 각 (record, win, level) 에서 cap 이전 후보가 `m_cap` 이상인가.
 P-2 ★**미공시 불일치의 크기** — 실측 하네스의 `W=40 s` 가 레코드별 몇 `Tp` 인가.
 P-3 규약 상수 되읽기 (배포 `tp_factor` · 학습 `TOBS_FACTOR` · `M_OBS` 3중 일치 · `cap_mode`).
 P-4 ★**MDE/검정력** — 부호검정 n=12(문턱 10)·n=36(문턱 24)이 무엇을 **못** 재는가.
 P-5 참 제로필드 기준 (⛔`ZERO_REF=1.0` 을 쓰지 않기 위해; **부수 서술 전용**).
 P-6 비용 — `B-2` 동결 행의 벽시계에서 456런의 CPU 예산을 추정한다.
 ★P-7 **기본값 비트동일** — `scene_station.py` 수정 후에도 **기본 경로**(`cap_mode='uniform'`)가
     `B-2` 동결 `obs_hash` 를 **12/12 비트재현**하는가. (= 호출자 30곳이 무해함을 증명. `G-5b1`)
 ★P-8 **pool·끝시각 동결** — **수정 전 소스**(git blob, md5 대조)와 수정 후 소스가
     `m_cap` 을 껐을 때 **비트동일한 pool** 을 내는가, 그 pool 의 `t_end_true` 는 무엇인가,
     그리고 그 셀의 **최신 프레임이 비어있지 않은가**. (= `G-1b`·`G-1c`·`G-5b2` 의 앵커)

★P-9 는 없다 — 밀도 공변은 **런타임 행**에 싣고 판정문에 붙인다(`G-10`).
   (전제에서도 P-1 의 `pool`/`n_frames` 로 수준별 밀도를 읽을 수 있다.)
"""
import hashlib
import importlib.util
import json
import math
import os
import sys
import warnings

warnings.filterwarnings('ignore')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')               # ⛔GPU 금지
import numpy as np                                              # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
for _p in (HERE, ROOT, os.path.join(ROOT, 'scripts'),
           os.path.join(ROOT, 'tmp', 'B2_REAL_2026-08-23'),
           os.path.join(ROOT, 'tmp', 'B3_REFIT_2026-08-23')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(ROOT)

import gates_m5b as G                                           # noqa: E402

G_ACC = G.G_ACC
W_MAX = 40.0            # scene_station.py  `W_max = 40.0` (메모리 적재 길이, 고정)
LEAD_MAX = 14.0         # scene_station.py
N_WINDOWS_GRID = 6      # scene_station.py  np.linspace(lo, hi, 6)


def _tp_from_kp(kp, depth):
    if not (kp > 0):
        return float('nan')
    w = math.sqrt(G_ACC * kp * math.tanh(min(kp * depth, 20.0)))
    return 2 * math.pi / w if w > 0 else float('nan')


def level_seconds(level, tp):
    """`gates_m5b.LEVELS` 의 한 수준이 이 (record,win) 에서 몇 초인가."""
    spec = G.LEVELS[level]
    if spec['kind'] == 'sec':
        return float(spec['value'])
    if spec['kind'] == 'tp':
        return float(spec['value']) * float(tp)
    return float('inf')


def load_pre_module():
    """★`scene_station.py` 의 **수정 전** 판본을 git blob 에서 꺼내 모듈로 올린다.

    ⚠파일 깊이가 같은 곳(`tmp/<batch>/`)에 써야 그 모듈의 `ROOT` 계산이 맞는다.
    ⛔저장소에 **두 번째 사본을 남기지 않는다** — 임포트 후 지운다(같은 것이 N곳 금지)."""
    src = G.scene_pre_source()
    path = os.path.join(HERE, '_scene_station_pre_tmp.py')
    with open(path, 'w') as fh:
        fh.write(src)
    try:
        spec = importlib.util.spec_from_file_location('scene_station_pre_m5b', path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        os.remove(path)
    assert 'cap_mode' not in src, 'blob 핀이 틀렸다 (수정 전 소스에 cap_mode 가 있다)'
    return mod


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
            picks = np.minimum(np.isfinite(Zmem).reshape(nfw, -1).sum(1), G.M_SCAN)
            lv = {}
            for level in G.LEVEL_ORDER:
                Ws = level_seconds(level, tp)
                nf = min(nfw, int(math.floor(Ws / rec.dt)) + 1)
                pool = int(picks[nfw - nf:].sum())
                adm = pool >= G.M_CAP
                lv[level] = dict(W_s=Ws, W_over_Tp=Ws / tp, n_frames=nf,
                                 pool=pool, admissible=bool(adm),
                                 # ★밀도 공변 (결과 전 공시) — 고정 600점이 깔리는 밀도
                                 dens_per_s_if_capped=G.M_CAP / Ws if Ws > 0 else None)
                if not adm:
                    viol.append((name, win, level, pool))
            Zfut = rec.window(i0, 1)
            sq = max(1, int(round(4.0 / rec.dx)))
            g0 = np.asarray(Zfut[0][::sq, ::sq], float)
            mm = np.isfinite(g0)
            zero_ref = float(np.sqrt(np.mean(g0[mm] ** 2)) / sg) if mm.any() else None
            per[win] = dict(kp=float(kp), lam_p=float(lam_p), Tp=tp, sigma=sg,
                            Hs=4.0 * sg, zero_ref_true=zero_ref,
                            min_picks_per_frame=int(picks.min()),
                            last_frame_picks=int(picks[-1]),
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
              median_factor_outside_train_support=float(
                  np.median(anch) / G.TOBS_FACTOR_TRAIN[1]),
              n_records_outside_support=int(sum(
                  1 for x in anch if x > G.TOBS_FACTOR_TRAIN[1])))
    return recs, viol, p2


def p7_default_bitidentical():
    """★수정 **후** 모듈의 **기본 경로**가 `B-2` 동결 장면을 비트재현하는가.

    ⛔이것이 `scene_station.py` 를 30여 호출자와 공유하면서 고칠 수 있는 유일한 근거다."""
    from scene_station import scene_station                      # 수정 후 모듈
    ref = G.b2_scene_ref()
    per, n_eq = {}, 0
    for rec in G.RECORDS:
        sc = scene_station(rec, win=G.FROZEN_B2_WIN)             # ★전부 기본값
        h = G.hash_obs(sc['P'], sc['z'])
        eq = bool(ref.get(rec) == h)
        n_eq += int(eq)
        per[rec] = dict(b2=ref.get(rec), mine=h, equal=eq)
        print(f'  P7 {rec:7s} b2={ref.get(rec)} mine={h} {"OK" if eq else "⛔"}',
              flush=True)
    return dict(n_ref=len(ref), n_checked=len(G.RECORDS), n_equal=n_eq,
                cap_mode_default=G.SCENE_DEFAULTS['cap_mode'], per_record=per)


def p8_pool_and_end():
    """★`m_cap` 을 끈 **cap 이전 pool** 을 수정 전/후 소스에서 각각 만들어 대조하고,
    그 pool 의 `t_end_true` 와 **최신 프레임 비어있지 않음**을 동결한다.

    · 수정 전/후가 같다 ⟹ 바뀐 것은 **cap 분기 하나뿐**이다(`G-5b2` 의 근거).
    · `t_end_true == sc['t0']` ⟹ 그 셀의 **최신 프레임에 관측이 있다**
      ⟹ `G-1b` 의 행 내부 앵커(`t0_abs`)가 유효하다."""
    from scene_station import scene_station as post
    pre_mod = load_pre_module()
    cells, n_eq, n_ne = {}, 0, 0
    for rec in G.RECORDS:
        cells[rec] = {}
        for w in G.WINDOWS:
            a = pre_mod.scene_station(rec, win=w, W=G.W_ANCHOR, m_cap=G.POOL_NO_CAP)
            b = post(rec, win=w, W=G.W_ANCHOR, m_cap=G.POOL_NO_CAP)
            ha = G.hash_obs(a['P'], a['z'])
            hb = G.hash_obs(b['P'], b['z'])
            t_end = float(np.asarray(a['P'], float)[:, 2].max())
            t0 = float(a['t0'])
            same = bool(ha == hb and len(a['z']) == len(b['z']))
            nonempty = bool(abs(t_end - t0) <= G.T_EQ_TOL)
            n_eq += int(same); n_ne += int(nonempty)
            cells[rec][str(w)] = dict(
                pool_hash=ha, pool_n=int(len(a['z'])),
                t_end_true=t_end, t0=t0, dt=float(a['dt']),
                pre_post_pool_equal=same, last_frame_nonempty=nonempty,
                pool_hash_post=hb)
        print(f'  P8 {rec:7s} ' + ' '.join(
            f'w{w}:n={cells[rec][str(w)]["pool_n"]},'
            f'{"=" if cells[rec][str(w)]["pre_post_pool_equal"] else "⛔"}'
            f'{"·" if cells[rec][str(w)]["last_frame_nonempty"] else "⛔empty"}'
            for w in G.WINDOWS), flush=True)
    n = len(G.RECORDS) * len(G.WINDOWS)
    return dict(scene_pre_blob=G.SCENE_PRE_BLOB, scene_pre_md5=G.SCENE_PRE_MD5,
                scene_post_md5=hashlib.md5(open(os.path.join(
                    ROOT, 'tmp', 'B2_REAL_2026-08-23', 'scene_station.py'),
                    'rb').read()).hexdigest(),
                n_cells=n, n_pre_post_equal=n_eq, n_last_frame_nonempty=n_ne,
                cells=cells)


def p3_constants():
    return dict(scene_defaults={k: v for k, v in G.SCENE_DEFAULTS.items()},
                cap_mode_batch=G.CAP_MODE,
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
                shard6_minutes=tot / 60.0 / 6, gpu_runs=0, episodes_generated=0,
                caveat='⚠`M5b` 는 `W>=W_ANCHOR` 수준에서 cap 이전 pool 장면을 한 번 더 '
                       '만든다(G-5b2) ⟹ (record,win) 당 캐시 1건 추가. 실측 벽시계는 '
                       '이보다 조금 길다.')


def main():
    print('P-7 기본값 비트동일 (추정기 0회) ...', flush=True)
    p7 = p7_default_bitidentical()
    print('P-8 pool·끝시각 동결 (수정 전 소스 vs 수정 후) ...', flush=True)
    p8 = p8_pool_and_end()
    print('P-1/P-2/P-5 (레코드 적재 — 추정기 0회) ...', flush=True)
    recs, viol, p2 = p1_p2_p5()
    # ★**중복 구현 교차검사** — `p1_p2_p5()` 는 `scene_station` 의 창 선택 논리
    #   (`nfw`·`n_lead`·`linspace(lo,hi,6)[win]`·프레임별 `min(m_scan, valid)`)를
    #   **두 번째로 구현**한다. 저장소가 반복해서 대가를 치른 결함 부류가 정확히 그것이므로
    #   (*"같은 기능이 두 파일에 있고 한쪽만 계측"*), P8 이 **진짜 빌더**로 만든 pool 크기와
    #   대조한다. 어긋나면 전제가 실패한다(⛔조용히 넘어가지 않는다).
    xchk = []
    for rec, per in recs.items():
        for w, v in per['windows'].items():
            a = int(v['levels']['ANCHOR']['pool'])
            b = int(p8['cells'][rec][str(w)]['pool_n'])
            if a != b:
                xchk.append([rec, w, a, b])
    print(f'=== ★P1↔P8 교차검사 (중복 구현) === 불일치 {len(xchk)}/'
          f'{len(G.RECORDS) * len(G.WINDOWS)} {xchk[:4]}')

    out = dict(batch=G.BATCH, kind='premise (결과 아님)',
               P1_level_admissibility=dict(records=recs, violations=viol,
                                           all_admissible=not viol),
               P2_window_mismatch=p2, P3_constants=p3_constants(),
               P4_power=p4_power(), P6_cost=p6_cost(),
               P7_default_bitidentical=p7, P8_pool_and_end=p8,
               P1_P8_crosscheck=dict(
                   n_cells=len(G.RECORDS) * len(G.WINDOWS), mismatches=xchk,
                   why='P1 의 창 선택 논리는 scene_station 의 두 번째 구현이다. '
                       'P8 은 진짜 빌더가 낸 pool 이다. 둘이 같아야 P1 을 믿을 수 있다.'))
    with open(os.path.join(HERE, 'premise_m5b.json'), 'w') as fh:
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
    print('=== ★P-7 기본값 비트동일 ===')
    print(f'  {p7["n_equal"]}/{p7["n_checked"]} (ref {p7["n_ref"]}) '
          f'· cap_mode 기본값={p7["cap_mode_default"]!r}')
    print('=== ★P-8 pool·끝시각 ===')
    print(f'  수정 전/후 pool 비트동일 {p8["n_pre_post_equal"]}/{p8["n_cells"]} · '
          f'최신프레임 비어있지않음 {p8["n_last_frame_nonempty"]}/{p8["n_cells"]}')
    print('=== P-6 비용 ===')
    print('  ' + json.dumps(out['P6_cost']))
    bad = (viol or xchk or p7['n_equal'] != p7['n_checked']
           or p8['n_pre_post_equal'] != p8['n_cells']
           or p8['n_last_frame_nonempty'] != p8['n_cells'])
    print('\n⛔전제 위반이 있다 — 동결 전에 고쳐라.' if bad
          else '\n✅전제 8건 + P1↔P8 교차검사 전부 통과.')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
