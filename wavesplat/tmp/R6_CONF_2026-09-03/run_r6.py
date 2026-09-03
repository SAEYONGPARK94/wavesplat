"""`R6-CONF` 러너 — **신뢰가중 입력만** 개입한다. ⛔추정 체인은 `R1b`/`B-2` 축자.

★신뢰가중은 **GT 없이** 만든다(`PREREG §1`): 평면 근사 법선 `(0,0,1)` + **관측된 z**.
⛔참 경사 `Zx,Zy` 를 쓰면 GT 누출이다 — 이 파일은 `maps_gt` 를 신뢰가중 계산에 쓰지 않는다.
"""
import hashlib, inspect, json, os, sys, time, warnings
warnings.filterwarnings('ignore')
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
          'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[v] = '1'
import numpy as np                                                  # noqa: E402
import torch                                                        # noqa: E402
torch.set_num_threads(1)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
B2D = os.path.join(ROOT, 'tmp', 'B2_REAL_2026-08-23')
for p in (HERE, B2D, ROOT, os.path.join(ROOT, 'scripts'),
          os.path.join(ROOT, 'tmp', 'bankmmse_port_2026-07-29')):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(ROOT)
import gates_r6 as G                                                # noqa: E402
from scene_station import scene_station                             # noqa: E402
from wavesplat.model.confidence import observation_weights          # noqa: E402

OUT = os.path.join(HERE, 'results_r6.jsonl')
GEO, GEO_SHA = G.measured_geometry()
D_REF = G.d_ref_default()
_MD5 = [os.path.join(ROOT, 'wavesplat', 'optimizer', 'sequential_update.py'),
        os.path.join(ROOT, 'wavesplat', 'run.py'),
        os.path.join(ROOT, 'wavesplat', 'model', 'confidence.py'),
        os.path.join(B2D, 'scene_station.py'), os.path.join(B2D, 'gates_b2.py'),
        os.path.join(HERE, 'gates_r6.py'), os.path.join(HERE, 'run_r6.py')]


def code_md5():
    return {os.path.basename(p): hashlib.md5(open(p, 'rb').read()).hexdigest()[:12] for p in _MD5}


def geo_conf(P, z, g):
    """★GT-free 신뢰가중. `z` 는 **관측값**이고 법선은 평면 근사 `(0,0,1)` 이다."""
    dx = g['x_s'] - P[:, 0]
    dy = g['y_s'] - P[:, 1]
    dz = g['h'] - z
    r = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    cos_t = np.clip(dz / np.maximum(r, 1e-9), 0.0, 1.0)
    w = observation_weights(torch.tensor(np.arccos(cos_t), dtype=torch.float32),
                            torch.tensor(r, dtype=torch.float32), d_ref=D_REF)
    return w.numpy(), r, cos_t


def frames_with_conf(sc, t_shift, conf):
    """`make_gif_bank_deployed.frames_from_scene` 축자 — **`confidence` 만** 갈아 끼운다."""
    from wavesplat.data_loader import FrameData
    P = np.asarray(sc['P'], float); z = np.asarray(sc['z'], float)
    out = []
    for tt in np.unique(P[:, 2]):
        m = P[:, 2] == tt
        n = int(m.sum())
        out.append(FrameData(t=float(tt - t_shift),
                             x_obs=torch.tensor(P[m, 0], dtype=torch.float32),
                             y_obs=torch.tensor(P[m, 1], dtype=torch.float32),
                             z_obs=torch.tensor(z[m], dtype=torch.float32),
                             confidence=(torch.ones(n, dtype=torch.float32) if conf is None
                                         else torch.tensor(conf[m], dtype=torch.float32)),
                             incidence_angles=torch.zeros(n, dtype=torch.float32)))
    return out


def frozen_anchor():
    """★앵커 = 동결 `R1b` **judge 패스의 `SHIP` 옛 규약 곡선**(현행 수심).

    ⛔`B-2` 가 아니다 — `R1b` 가 그쪽이 **수심 17** 임을 확립했다.
    """
    out = {}
    for l in open(G.R1B_ROWS):
        if l.strip():
            r = json.loads(l)
            if r.get('pass_') == 'judge' and r.get('cell') == 'SHIP':
                out[('SHIP', r['record'])] = dict(curve=r['curve_old'])
    return out


def run(record, arm, frozen):
    from wavesplat.optimizer.sequential_update import make_updater
    from wavesplat.run import DEFAULT_CONFIG, set_seed
    t0w = time.time()
    sc = scene_station(record, win=G.WIN)
    P = np.asarray(sc['P'], float); z = np.asarray(sc['z'], float)
    t_shift = float(P[:, 2].min())
    t0 = float(sc['Pq'][0, 2]) - t_shift
    conf = cstat = None
    if G.geo_of(arm):
        w, r, ct = geo_conf(P, z, GEO[record])
        # ★탐침 `CVAR` — 미클램프 conf 를 **자기 최대로** 나눈다(PREREG §4).
        #   ⛔배포 후보가 아니다. 기능은 *"어느 경로가 이 필드를 읽는가"* 의 계측기다.
        raw = ct / np.maximum(r / D_REF, 1e-12)
        conf = (raw / raw.max()).astype(np.float32)
        w = conf
        cstat = dict(conf_med=float(np.median(w)), conf_min=float(w.min()),
                     conf_max=float(w.max()),
                     raw_med=float(np.median(raw)), raw_min=float(raw.min()),
                     frac_saturated=float(np.mean(raw >= 1.0)),
                     r_med=float(np.median(r)), cos_med=float(np.median(ct)))
    frames = frames_with_conf(sc, t_shift, conf)
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(depth=float(sc['depth']), seed=G.SEED, device='cpu')
    cfg.update(G.BASE_ARMS[G.base_of(arm)])
    set_seed(G.SEED)
    upd = make_updater(cfg)
    res = upd.initialize(frames, x_range=(float(sc['gxq'].min()), float(sc['gxq'].max())),
                         y_range=(float(sc['gyq'].min()), float(sc['gyq'].max())),
                         verbose=False)
    gxq, gyq = np.asarray(sc['gxq']), np.asarray(sc['gyq'])
    curve, zcurve = [], []
    for li, l_ in enumerate(sc['leads']):
        g = np.asarray(sc['maps_gt'][li], float)
        mm = np.isfinite(g)
        if not mm.any():
            curve.append(float('nan')); zcurve.append(float('nan')); continue
        Z = upd.predict_field(torch.tensor(gxq, dtype=torch.float32),
                              torch.tensor(gyq, dtype=torch.float32),
                              t0 + float(l_)).cpu().numpy().reshape(len(gyq), len(gxq))
        curve.append(float(np.sqrt(np.mean((Z[mm] - g[mm]) ** 2)) / sc['sg']))
        zcurve.append(float(np.sqrt(np.mean(g[mm] ** 2)) / sc['sg']))
    fz = frozen.get(('SHIP', record))
    rel = None
    if arm == 'SHIP-C1' and fz:
        a = np.asarray(fz['curve'], float); b = np.asarray(curve, float)
        m2 = np.isfinite(a) & np.isfinite(b)
        rel = (float(np.max(np.abs(b[m2] - a[m2]) / np.maximum(np.abs(a[m2]), 1e-12)))
               if m2.any() else 1.0)
    eff = getattr(upd, 'config', {}) or {}
    src = inspect.getsource(geo_conf)
    return dict(cell=arm, record=record, base=G.base_of(arm), geo=G.geo_of(arm),
                nowcast=curve[0], curve=curve, zero_ref=zcurve[0],
                beats_zero=bool(curve[0] < zcurve[0]),
                conf_stats=cstat, d_ref=D_REF, geom_sha=GEO_SHA,
                geom=(GEO[record] if G.geo_of(arm) else None),
                nogt_ok=not any(t in src for t in ('maps_gt', 'Zx', 'Zy', 'gt_')),
                shipping={k: eff.get(k) for k in G.SHIPPING_KEYS},
                anchor_relerr=rel, seed=G.SEED, threads=torch.get_num_threads(),
                n_atoms=int(getattr(getattr(upd, 'model', None), 'n_gaussians', 0) or 0),
                real_gate=getattr(upd, '_auto_real_gate', None),
                pid=os.getpid(), code_md5=code_md5(), wall_s=round(time.time() - t0w, 2))


def main():
    frozen = frozen_anchor()
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            if l.strip():
                r = json.loads(l); done.add((r['cell'], r['record']))
    with open(OUT, 'a') as fh:
        for rec in G.RECORDS:
            for arm in G.ARM_ORDER:
                if (arm, rec) in done:
                    continue
                r = run(rec, arm, frozen)
                fh.write(json.dumps(r, ensure_ascii=False) + '\n'); fh.flush()
                cs = r['conf_stats'] or {}
                print(f"  {arm:10s} {rec:7s} nowcast={r['nowcast']:.4f} zero={r['zero_ref']:.4f} "
                      f"conf_med={cs.get('conf_med', '-')} 앵커={r['anchor_relerr']} "
                      f"{r['wall_s']:.1f}s", flush=True)
    print('✅ 완료'); return 0


if __name__ == '__main__':
    sys.exit(main())
