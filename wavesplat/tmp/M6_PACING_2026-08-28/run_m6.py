"""M-6 러너 — 재생 배속 r 의 용량-반응. 프로토콜·채점은 B-9b 와 글자 그대로 같고
(lead ℓ 마다 ① 예측·채점 → ② 그 시각 관측 투입 — 절대 뒤집지 마라),
**유일한 변경 = 페이싱 목표가 `t_run0 + ℓ/r`** 이고 계측 4종이 추가된다:
  · 매 update() 직후 `_bank_async_applied` 되읽기  → `first_apply_lead`, `applied_pre_wait`
    (★B-9 의 교훈: 최종 `bank_refit_wait` **이전** 스냅샷이어야 "채점 중 적용"을 잰다 —
     B-9 는 wait 뒤에 계수기를 읽어 적용 시점이 채점 안인지 밖인지 행에서 구분되지 않았다)
  · `_bank_fit_window` 호출별 벽시계(배경 스레드 포함)     → `t_fit_median_s`
  · :1197-1208 RuntimeWarning 포집                          → `noop_warning_fired`
  · 구성 **객체 되읽기**(의도표 아님) + 임포트 경로 md5    → `readback`, `md5_*`

⛔실행 규율(PREREG §6): **직렬 1프로세스**. 페이싱 셀은 벽시계에 민감하다 — 병렬화하면
t_fit 이 경합으로 늘어나 무릎이 이동한다. G-6 이 그것을 잡지만, 애초에 하지 마라.
GPU 0회(bank 경로는 numpy CPU). M1 생성이 GPU 를 점유 중이어도 무관하나 CPU 부하는 loadavg1 로 행에 남긴다.
"""
import hashlib, json, os, subprocess, sys, time, warnings
warnings.filterwarnings('ignore')
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
for p in (HERE, ROOT, os.path.join(ROOT, 'scripts'),
          os.path.join(ROOT, 'tmp', 'B2_REAL_2026-08-23'),
          os.path.join(ROOT, 'tmp', 'B3_REFIT_2026-08-23'),
          os.path.join(ROOT, 'tmp', 'bankmmse_port_2026-07-29')):
    sys.path.insert(0, p)
os.chdir(ROOT)
import gates_m6 as G                        # noqa: E402
from scene_station import scene_station     # noqa: E402
import make_gif_bank_deployed as M          # noqa: E402

OUT = os.path.join(HERE, 'results_m6.jsonl')
_md5 = lambda p: hashlib.md5(open(p, 'rb').read()).hexdigest()


def run(record, win, arm):
    from wavesplat.data_loader import FrameData
    from wavesplat.optimizer.sequential_update import make_updater
    import wavesplat.optimizer.sequential_update as SU
    from wavesplat.run import DEFAULT_CONFIG, set_seed
    pace = G.PACE_OF[arm]
    t0w = time.time()
    sc = scene_station(record, win=win, noise=G.NOISE, with_future_obs=True)
    t_shift = float(np.asarray(sc['P'])[:, 2].min())
    t0 = float(sc['Pq'][0, 2]) - t_shift
    frames = M.frames_from_scene(sc, t_shift, dtype=torch.float32)
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(depth=float(sc['depth']), seed=G.SEED, device='cpu')
    cfg.update(G.BASE_PINS)                 # 2026-08-22 배포 3키 명시 핀
    cfg.update(G.ARMS[arm])

    fit_walls = []                          # ★가지 단위 계수기 + 벽시계 (배경 스레드 호출 포함)
    orig = SU.SequentialUpdater._bank_fit_window
    def wrapped(self, *a, **kw):
        _t = time.perf_counter()
        try:
            return orig(self, *a, **kw)
        finally:
            fit_walls.append(time.perf_counter() - _t)
    SU.SequentialUpdater._bank_fit_window = wrapped
    try:
        set_seed(G.SEED)
        upd = make_updater(cfg)
        with warnings.catch_warnings(record=True) as wlog:
            warnings.simplefilter('always')
            upd.initialize(frames,
                           x_range=(float(sc['gxq'].min()), float(sc['gxq'].max())),
                           y_range=(float(sc['gyq'].min()), float(sc['gyq'].max())),
                           verbose=False)
            n_init = len(fit_walls)
            gxq, gyq = np.asarray(sc['gxq']), np.asarray(sc['gyq'])
            tx = torch.tensor(gxq, dtype=torch.float32)
            ty = torch.tensor(gyq, dtype=torch.float32)
            Pf = np.asarray(sc['P_fut'], float); zf = np.asarray(sc['z_fut'], float)
            per_lead, upd_times = {}, []
            applied_after = {}              # lead → 그 update() 직후 누적 applied
            _t_run0 = time.perf_counter()
            for li, l_ in enumerate(sc['leads']):
                g = np.asarray(sc['maps_gt'][li], float); mm = np.isfinite(g)
                if mm.sum() > 0:            # ① 예측·채점
                    Z = upd.predict_field(tx, ty, t0 + float(l_)) \
                           .cpu().numpy().reshape(len(gyq), len(gxq))
                    per_lead[float(l_)] = float(
                        np.sqrt(np.mean((Z[mm] - g[mm]) ** 2)) / sc['sg'])
                if li == 0:
                    continue
                tt = t0 + float(l_) + t_shift
                m = np.abs(Pf[:, 2] - tt) < 1e-6
                if m.sum() == 0:
                    continue
                nq = int(m.sum())
                if pace != float('inf'):    # ★페이싱: 이 lead 의 벽시계 시작점 = t_run0 + ℓ/r
                    _slack = (_t_run0 + float(l_) / pace) - time.perf_counter()
                    if _slack > 0:
                        time.sleep(_slack)
                _t_up = time.perf_counter()  # ② 그 시각 관측 투입
                upd.update(FrameData(t=float(l_) + t0,
                                     x_obs=torch.tensor(Pf[m, 0], dtype=torch.float32),
                                     y_obs=torch.tensor(Pf[m, 1], dtype=torch.float32),
                                     z_obs=torch.tensor(zf[m], dtype=torch.float32),
                                     confidence=torch.ones(nq, dtype=torch.float32),
                                     incidence_angles=torch.zeros(nq, dtype=torch.float32)))
                upd_times.append(time.perf_counter() - _t_up)
                applied_after[float(l_)] = int(getattr(upd, '_bank_async_applied', 0))
            # ★최종 wait **이전** 스냅샷 = "채점 중" 상태 (B-9 의 계측 구멍을 막는다)
            health_pre = (upd.bank_refit_health()
                          if hasattr(upd, 'bank_refit_health') else {})
            applied_pre_wait = int(getattr(upd, '_bank_async_applied', 0))
            upd.bank_refit_wait(timeout=180)
        noop_warn = any(issubclass(w.category, RuntimeWarning)
                        and 'bank_refit_async is ON' in str(w.message) for w in wlog)
        first_apply = next((l_ for l_ in sorted(applied_after)
                            if applied_after[l_] > 0), None)
        fc = [v for k, v in per_lead.items() if k >= 2.0]
        late = [v for k, v in per_lead.items() if k >= 8.0]
        tf = [t for t in fit_walls[n_init:]]
        rb = {k: upd.config.get(k) for k in
              ('bank_refit_every_s', 'bank_refit_async', *G.BASE_PINS)}
        rb['_bank_refit_every_s_attr'] = getattr(upd, '_bank_refit_every_s', 'MISSING')
        return dict(cell=arm, pace=(None if pace == float('inf') else pace),
                    record=record, win=win,
                    lead0=per_lead.get(0.0),
                    fc_mean=float(np.mean(fc)) if fc else None,
                    late_mean=float(np.mean(late)) if late else None,
                    per_lead={str(k): v for k, v in sorted(per_lead.items())},
                    refit_calls=len(fit_walls) - n_init, init_fit_calls=n_init,
                    t_fit_median_s=(float(np.median(tf)) if tf else None),
                    t_fit_all_s=[round(t, 3) for t in tf],
                    worst_update_s=max(upd_times) if upd_times else float('nan'),
                    median_update_s=float(np.median(upd_times)) if upd_times else float('nan'),
                    n_update=len(upd_times),
                    async_kicks=int(getattr(upd, '_bank_async_kicks', 0)),
                    async_applied_final=int(getattr(upd, '_bank_async_applied', 0)),
                    async_skipped=int(getattr(upd, '_bank_async_skipped', 0)),
                    applied_pre_wait=applied_pre_wait,
                    applied_after_lead={str(k): v for k, v in sorted(applied_after.items())},
                    first_apply_lead=first_apply,
                    health_pre_wait=health_pre,
                    noop_warning_fired=bool(noop_warn),
                    readback=rb,
                    loadavg1=os.getloadavg()[0],
                    md5_sequential_update=_md5('wavesplat/optimizer/sequential_update.py'),
                    md5_bank_mmse=_md5('wavesplat/optimizer/bank_mmse.py'),
                    git_head=subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                                            capture_output=True, text=True).stdout.strip(),
                    n_fut_obs=int(len(zf)), sigma=float(sc['sg']),
                    wall_s=time.time() - t0w)
    finally:
        SU.SequentialUpdater._bank_fit_window = orig


def main():
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            try:
                r = json.loads(l); done.add((r['cell'], r['record'], r['win']))
            except Exception:
                pass
    # ★배속을 바깥 고리로 돌지 않는다 — (record, win) 바깥 고리 = 부하 드리프트가
    #   특정 배속에 몰리는 교락을 줄인다(그래도 loadavg1·t_fit 이 행마다 남는다).
    for record in G.RECORDS:
        for win in G.WINDOWS:
            for arm in G.ARM_ORDER:
                if (arm, record, win) in done:
                    continue
                try:
                    row = run(record, win, arm)
                except Exception as e:      # noqa: BLE001
                    row = dict(cell=arm, record=record, win=win, error=repr(e))
                with open(OUT, 'a') as f:
                    f.write(json.dumps(row) + '\n')
                print(row.get('cell'), record, win,
                      row.get('fc_mean'), row.get('first_apply_lead'),
                      row.get('error', ''), flush=True)


if __name__ == '__main__':
    main()
