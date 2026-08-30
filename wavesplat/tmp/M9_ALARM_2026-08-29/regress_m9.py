#!/usr/bin/env python3
"""M-9 회귀 실행기 — 판정 규약은 전부 `gates_m9.py`(단일출처). 이 파일은 실행만 한다.

    python3 tmp/M9_ALARM_2026-08-29/regress_m9.py selftest   # 게이트 자가진단 (런 0회, <1 s)
    python3 tmp/M9_ALARM_2026-08-29/regress_m9.py rg1        # ★M6 동결 360행 재생 (런 0회)
    python3 tmp/M9_ALARM_2026-08-29/regress_m9.py rg0 pre    # 라이브 4셀, 수리 **전** 스냅샷 (~30 s)
    python3 tmp/M9_ALARM_2026-08-29/regress_m9.py rg0 post   # 수리 **후** 재실행 + pre 와 비트 대조

rg1 은 제품 술어(`noop_alarm_due`)가 있으면 post 모드(발화 열 = 제품 임포트),
없으면 pre 모드(계약 유도식으로 동결 예상표만 재확인)로 **자동 판별**하고 모드를 출력한다.
rg0 의 4셀은 gates_m9.RG0_* 동결 정의를 따른다 — LAND 셀은 해당 lead 의 update 직전에
배경 스레드를 join 해 착지/kick 을 **결정적으로** 만든다(수면 아님·경합 없음).

exit 0 = 해당 단계 전건 통과. 산출물: rg1_report.json / rg0_pre.json / rg0_post.json.
"""
import os

# ★스레드 고정 — 규율(2026-08-30): 러너가 스스로, import 보다 먼저 (본보기 run_m3.py:19-21)
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'

import hashlib     # noqa: E402
import json        # noqa: E402
import sys         # noqa: E402
import time        # noqa: E402
import warnings    # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gates_m9 as G                                       # noqa: E402

_md5 = lambda p: hashlib.md5(open(p, 'rb').read()).hexdigest()   # noqa: E731


def _print_gates(gates):
    ok_all = True
    for x in gates:
        mark = {True: 'PASS', False: '⛔FAIL', None: '—보류'}[x['ok']]
        print(f"  [{mark:>6}] {x['name']}  {x['note']}")
        ok_all &= x['ok'] is not False
    return ok_all


def cmd_selftest():
    print(G.selftest())
    return 0


def cmd_rg1():
    rows, md5 = G.load_rows()
    pred, src = G.import_product_predicate()
    rep = G.rg1_run(rows, md5, predicate=pred, predicate_src=src)
    rep['md5_sequential_update'] = _md5(G.SEQ_UPD_PATH)
    rep['ts'] = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"RG-1 재생 — 모드 {rep['mode']} (술어: {src})")
    ok = _print_gates(rep['gates'])
    lab, why = G.label_final(rep)
    print(f'label(rg1 단독): {lab} — {why}')
    json.dump(rep, open(G.RG1_REPORT, 'w'), ensure_ascii=False, indent=1)
    print('→', G.RG1_REPORT)
    return 0 if ok else 1


def _rg0_run_cells():
    """동결 4셀 실행 — 러너 골격은 run_m6.run() 을 따르고(예측→채점→관측 투입 순서 불변)
    페이싱 대신 gates_m9.RG0_JOIN_BEFORE_LEAD 의 join 지점만 다르다."""
    os.chdir(G.ROOT)
    import numpy as np
    import torch
    from scene_station import scene_station                # B2 (gates_m9 가 path 삽입)
    import make_gif_bank_deployed as MG                    # bankmmse_port
    from wavesplat.data_loader import FrameData
    from wavesplat.optimizer.sequential_update import make_updater
    from wavesplat.run import DEFAULT_CONFIG, set_seed

    G6 = G.G6
    sc = scene_station(G.RG0_RECORD, win=G.RG0_WIN, noise=G6.NOISE,
                       with_future_obs=True)
    t_shift = float(np.asarray(sc['P'])[:, 2].min())
    t0 = float(sc['Pq'][0, 2]) - t_shift
    gxq, gyq = np.asarray(sc['gxq']), np.asarray(sc['gyq'])
    tx = torch.tensor(gxq, dtype=torch.float32)
    ty = torch.tensor(gyq, dtype=torch.float32)
    Pf = np.asarray(sc['P_fut'], float)
    zf = np.asarray(sc['z_fut'], float)

    cells = {}
    for cell in G.RG0_CELLS:
        t0w = time.time()
        frames = MG.frames_from_scene(sc, t_shift, dtype=torch.float32)
        cfg = dict(DEFAULT_CONFIG)
        cfg.update(depth=float(sc['depth']), seed=G6.SEED, device='cpu')
        cfg.update(G6.BASE_PINS)                 # 2026-08-22 배포 3키 명시 핀
        cfg.update(G.RG0_CELL_CFG[cell])         # ⛔빈 dict 팔 금지 — 셀이 2키 명시
        join_lead = G.RG0_JOIN_BEFORE_LEAD.get(cell)
        set_seed(G6.SEED)
        upd = make_updater(cfg)
        per_lead, field_md5s, applied_after = {}, {}, {}
        with warnings.catch_warnings(record=True) as wlog:
            warnings.simplefilter('always')
            upd.initialize(frames,
                           x_range=(float(gxq.min()), float(gxq.max())),
                           y_range=(float(gyq.min()), float(gyq.max())),
                           verbose=False)
            for li, l_ in enumerate(sc['leads']):
                gmap = np.asarray(sc['maps_gt'][li], float)
                mm = np.isfinite(gmap)
                if mm.sum() > 0:                 # ① 예측·채점 (run_m6 과 동일 순서)
                    Z = upd.predict_field(tx, ty, t0 + float(l_)) \
                           .cpu().numpy().reshape(len(gyq), len(gxq))
                    per_lead[str(float(l_))] = float(
                        np.sqrt(np.mean((Z[mm] - gmap[mm]) ** 2)) / sc['sg'])
                    field_md5s[str(float(l_))] = hashlib.md5(
                        Z.astype(np.float64).tobytes()).hexdigest()
                if li == 0:
                    continue
                tt = t0 + float(l_) + t_shift
                m = np.abs(Pf[:, 2] - tt) < 1e-6
                if m.sum() == 0:
                    continue
                if join_lead is not None and float(l_) == join_lead:
                    fut = getattr(upd, '_bank_async', None)   # ★결정적 착지: 수면 아님
                    if fut is not None:
                        fut['thread'].join(timeout=300)
                        assert not fut['thread'].is_alive(), 'join 실패 — 셀 무효'
                nq = int(m.sum())                # ② 그 시각 관측 투입
                upd.update(FrameData(t=float(l_) + t0,
                                     x_obs=torch.tensor(Pf[m, 0], dtype=torch.float32),
                                     y_obs=torch.tensor(Pf[m, 1], dtype=torch.float32),
                                     z_obs=torch.tensor(zf[m], dtype=torch.float32),
                                     confidence=torch.ones(nq, dtype=torch.float32),
                                     incidence_angles=torch.zeros(nq, dtype=torch.float32)))
                applied_after[str(float(l_))] = int(getattr(upd, '_bank_async_applied', 0))
            health = (upd.bank_refit_health()
                      if hasattr(upd, 'bank_refit_health') else {})
        fired_n = sum(1 for w in wlog if issubclass(w.category, RuntimeWarning)
                      and G.ALARM_WARN_TOKEN in str(w.message))
        first_apply = next((float(k) for k in sorted(applied_after, key=float)
                            if applied_after[k] > 0), None)
        fut = getattr(upd, '_bank_async', None)              # 다음 셀과의 경합 차단
        if fut is not None:
            fut['thread'].join(timeout=300)
        cells[cell] = dict(
            per_lead=per_lead, field_md5s=field_md5s, applied_after=applied_after,
            counters=dict(kicks=int(getattr(upd, '_bank_async_kicks', 0)),
                          skipped=int(getattr(upd, '_bank_async_skipped', 0)),
                          applied=int(getattr(upd, '_bank_async_applied', 0))),
            first_apply=first_apply, fired_n=int(fired_n),
            health={k: (bool(v) if isinstance(v, bool) else v)
                    for k, v in health.items()},
            health_keys=sorted(health), wall_s=time.time() - t0w)
        print(f"  {cell:<13} fired={fired_n} kicks={cells[cell]['counters']['kicks']} "
              f"skip={cells[cell]['counters']['skipped']} "
              f"applied={cells[cell]['counters']['applied']} "
              f"first_apply={first_apply} {cells[cell]['wall_s']:.1f}s")
    meta = dict(record=G.RG0_RECORD, win=G.RG0_WIN, t0=t0, seed=G6.SEED,
                noise=G6.NOISE, base_pins=G6.BASE_PINS,
                md5_sequential_update=_md5(G.SEQ_UPD_PATH),
                threads_pinned={v: os.environ[v] for v in
                                ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                                 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')},
                loadavg1=os.getloadavg()[0],
                ts=time.strftime('%Y-%m-%d %H:%M:%S'))
    return dict(cells=cells, meta=meta)


def cmd_rg0(tag):
    assert tag in ('pre', 'post'), "rg0 pre|post"
    print(f'RG-0 라이브 4셀 ({tag}) — 레코드 {G.RG0_RECORD}/w{G.RG0_WIN}')
    rep = _rg0_run_cells()
    pre_rep = None
    if tag == 'post':
        if os.path.exists(G.RG0_REPORT['pre']):
            pre_rep = json.load(open(G.RG0_REPORT['pre']))
        else:
            print('⚠rg0_pre.json 부재 — R-3 비트 대조 불가(보류로 기록)')
    rep['gates'] = G.rg0_gates(tag, rep, pre_rep)
    ok = _print_gates(rep['gates'])
    json.dump(rep, open(G.RG0_REPORT[tag], 'w'), ensure_ascii=False, indent=1)
    print('→', G.RG0_REPORT[tag])
    return 0 if ok else 1


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ('selftest', 'rg1', 'rg0'):
        print(__doc__)
        return 2
    if sys.argv[1] == 'selftest':
        return cmd_selftest()
    if sys.argv[1] == 'rg1':
        return cmd_rg1()
    return cmd_rg0(sys.argv[2] if len(sys.argv) > 2 else '')


if __name__ == '__main__':
    sys.exit(main())
