"""`E1-EXTBASE` 러너 — 같은 장면에서 **우리 팔**과 **제3자 팔**을 나란히 돌린다.

⛔`external/tfenv/bin/python` 으로 실행한다(TF 2.17 + torch CPU 가 그 환경에만 있다).
⛔우리 코드·저들 코드 **둘 다 무수정**. 추정 체인은 `R1B/run_r1b.py` 를 축자로 옮긴다.
"""
import hashlib
import json
import os
import sys
import time
import warnings

warnings.filterwarnings('ignore')
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '1'
os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import numpy as np                                                   # noqa: E402
import torch                                                         # noqa: E402
torch.set_num_threads(1)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
B2DIR = os.path.join(ROOT, 'tmp', 'B2_REAL_2026-08-23')
for _p in (HERE, B2DIR, ROOT, os.path.join(ROOT, 'scripts'),
           os.path.join(ROOT, 'tmp', 'bankmmse_port_2026-07-29')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(ROOT)
import gates_e1 as G                                                 # noqa: E402
import extarm_e1 as EX                                               # noqa: E402
import scene_station as SS                                           # noqa: E402
from scene_station import scene_station                              # noqa: E402
from d3_sa_multirecord import MRecord, _unpack                       # noqa: E402
import make_gif_bank_deployed as M                                   # noqa: E402

OUT = os.path.join(HERE, 'results_e1.jsonl')
META = os.path.join(HERE, 'meta_e1.json')


def raw_window(rec, i0, n):
    s = rec.stride
    Z = (_unpack(rec.zd, np.s_[i0:i0 + n, ::s, ::s]) if s > 1
         else _unpack(rec.zd, np.s_[i0:i0 + n]))
    Z[:, ~rec.valid] = np.nan
    return Z


def geometry(record):
    cfg = SS.RECORDS_12[record]
    rec = MRecord(os.path.join(ROOT, cfg['path']), stride=cfg['stride'])
    nfw = int(round(40.0 / rec.dt))
    n_lead = int(round(14.0 / rec.dt))
    if rec.T_all < nfw + n_lead + 4:
        n_lead = max(2, (rec.T_all - nfw - 4))
    lo, hi = nfw + 1, rec.T_all - n_lead - 2
    i0 = int(np.linspace(lo, hi, 6).astype(int)[G.WIN])
    sq = max(1, int(round(4.0 / rec.dx)))
    return rec, nfw, n_lead, i0, sq, cfg['depth']


def positive_control(model):
    """★`G-POS` — 아는 파도장 600점을 배치 안에서 다시 통과시킨다."""
    n, L = G.CNN_N, 90.0
    x = np.linspace(-L / 2, L / 2, n)
    X, Y = np.meshgrid(x, x)
    rng = np.random.default_rng(7)
    comps = []
    for _ in range(5):
        lam = rng.uniform(18.0, 45.0); th = np.radians(rng.uniform(70, 110))
        k = 2 * np.pi / lam
        comps.append((rng.uniform(0.10, 0.35), k * np.cos(th), k * np.sin(th),
                      np.sqrt(9.8 * k), rng.uniform(0, 2 * np.pi)))
    fld = lambda t: sum(a * np.cos(kx * X + ky * Y - w * t + p)                # noqa: E731
                        for a, kx, ky, w, p in comps).astype(np.float32)
    dt = 0.2
    frames = [fld(-dt), fld(0.0), fld(dt)]
    _, Zd, _, _ = EX.predict(model, frames, x, x, 600, dt, np.inf,
                             G.SAMPLE_SEED, n=n)
    t = frames[1]
    return float(np.sqrt(np.mean((Zd - t) ** 2)) / np.std(t))


def kgrid_check():
    """★`G-KGRID` — 내 파수 격자가 저들 `config256.mat` 과 **원소별로** 같은가.

    ⚠결과 0행에서 명세를 고쳤다: 초판은 내가 **손으로 적은 범위 두 값**과 비교했는데,
    그 값은 저들 파일에서 읽은 게 아니라 **인쇄된 4자리를 보고 지어낸 것**이었다
    (상대오차 8.0e−07 로 실패). ⟹ ★***내가 모르는 수를 문턱으로 적었다*** —
    `H3b` 의 `G-IMPORT` 와 같은 부류다. 이제 **저들 배포물을 직접 읽어 전체 배열을 대조**한다.
    """
    from scipy.io import loadmat
    d = loadmat(os.path.join(HERE, 'vendor_config256.mat'))
    KXv, KYv = np.asarray(d['KX_ab']), np.asarray(d['KY_ab'])
    # 저들 박스: XX -45..45 · YY -115..-25 ⟹ 두 변 모두 90 m, N=256
    KX, KY = EX.kgrid(256, 90.0 / 255.0, 90.0 / 255.0)
    sc = max(np.abs(KXv).max(), np.abs(KYv).max())
    rel = float(max(np.max(np.abs(KX - KXv)), np.max(np.abs(KY - KYv))) / sc)
    return bool(rel < G.KGRID_TOL), rel


def run(record, model):
    from wavesplat.optimizer.sequential_update import make_updater
    from wavesplat.run import DEFAULT_CONFIG, set_seed
    t0w = time.time()
    rec, nfw, n_lead, i0, sq, depth = geometry(record)
    sc = scene_station(record, win=G.WIN)
    gxq, gyq = np.asarray(sc['gxq']), np.asarray(sc['gyq'])

    mu_mem = np.nanmean(raw_window(rec, i0 - nfw + 1, nfw), axis=0)
    mu_fut = np.nanmean(raw_window(rec, i0, n_lead + 1), axis=0)
    delta_q = (mu_fut - mu_mem)[::sq, ::sq]
    g = np.asarray(sc['maps_gt'][0], float) + delta_q                # ★R1b 수리 규약
    fin = np.isfinite(g)
    sg = sc['sg']
    score = lambda Z: float(np.sqrt(np.mean((Z[fin] - g[fin]) ** 2)) / sg)     # noqa: E731
    zero = float(np.sqrt(np.mean(g[fin] ** 2)) / sg)

    # ── 우리 팔
    t_shift = float(np.asarray(sc['P'])[:, 2].min())
    t0 = float(sc['Pq'][0, 2]) - t_shift
    frames_t = M.frames_from_scene(sc, t_shift, dtype=torch.float32)
    cfg = dict(DEFAULT_CONFIG); cfg.update(depth=depth, seed=G.SEED, device='cpu')
    cfg.update(G.SHIP_ARM)
    set_seed(G.SEED)
    upd = make_updater(cfg)
    upd.initialize(frames_t, x_range=(float(gxq.min()), float(gxq.max())),
                   y_range=(float(gyq.min()), float(gyq.max())), verbose=False)
    Zs = upd.predict_field(torch.tensor(gxq, dtype=torch.float32),
                           torch.tensor(gyq, dtype=torch.float32),
                           t0).cpu().numpy().reshape(len(gyq), len(gxq))
    eff = getattr(upd, 'config', {}) or {}
    arm_ok = ({k: eff.get(k) for k in G.SHIPPING_KEYS} == G.shipping_expect())

    # ── 제3자 팔 (사다리 + PRIMARY 칸)
    Zf3 = raw_window(rec, i0 - 1, 3)[:, ::sq, ::sq] * 0 + np.nan     # 자리만 잡는다
    full = raw_window(rec, i0 - 1, 3)
    frames3 = [full[k] for k in range(3)]
    row = dict(record=record, ship=score(Zs), zero=zero,
               n_obs_ship=int(len(sc['z'])), arm_ok=bool(arm_ok),
               dt=float(rec.dt), depth=depth, sq=sq,
               box=[float(rec.gx.min()), float(rec.gx.max()),
                    float(rec.gy.min()), float(rec.gy.max())])
    rungs = sorted(set(G.LADDER + (G.PRIMARY_PER_FRAME,)))
    n_prim = 0
    gt_free = True
    for m in rungs:
        sgn, Zd, n_in, n_held = EX.predict(
            model, frames3, rec.gx, rec.gy, m, float(rec.dt), depth,
            G.SAMPLE_SEED, n=G.CNN_N, holdout=G.DIR_HOLDOUT, signs=G.SIGNS)
        Zq = EX.to_query(Zd, rec.gx, rec.gy, gxq, gyq, n=G.CNN_N)
        row[f'cnn_{m}'] = score(Zq)
        row[f'nin_{m}'] = int(n_in)
        if m == G.PRIMARY_PER_FRAME:
            row['dir_sign'] = list(sgn)
            n_prim = int(n_in) + int(n_held)
            # ★G-GT — 공급한 관측 수와 입력 유한 원소 수가 맞는가(참값 유입 0)
            gt_free = bool(n_in <= 3 * m and n_in > 0)
    row['n_obs_cnn_primary'] = n_prim
    row['primary'] = True
    row['gt_free'] = bool(gt_free)
    row['wall_s'] = round(time.time() - t0w, 1)
    return row


def main():
    model = EX.build_model(G.CNN_N)
    pos = positive_control(model)
    kok, krel = kgrid_check()
    eok, emsg = G.ext_clean()
    print(f'  G-POS {pos:.4f} · G-KGRID {krel:.2e} · G-NOMOD {emsg}', flush=True)
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT, encoding='utf-8'):
            if l.strip():
                done.add(json.loads(l).get('record'))
    with open(OUT, 'a', encoding='utf-8') as fh:
        for rec in G.RECORDS:
            if rec in done:
                continue
            try:
                r = run(rec, model)
            except Exception as exc:                                 # noqa: BLE001
                r = dict(record=rec, error=f'{type(exc).__name__}: {exc}')
            fh.write(json.dumps(r, ensure_ascii=False) + '\n'); fh.flush()
            if 'error' in r:
                print(f"  {rec:6s} ⛔{r['error'][:90]}", flush=True)
            else:
                print(f"  {rec:6s} ship={r['ship']:.3f} zero={r['zero']:.3f} "
                      f"cnn200={r.get('cnn_200', float('nan')):.3f} "
                      f"cnn12={r.get('cnn_12', float('nan')):.3f} "
                      f"dir={r.get('dir_sign')} ({r['wall_s']}s)", flush=True)
    json.dump(dict(n_records=sum(1 for l in open(OUT, encoding='utf-8') if l.strip()),
                   pos_nrmse=pos, kgrid_ok=kok, kgrid_relerr=krel,
                   ext_clean=eok, ext_msg=emsg,
                   shipping_expect=G.shipping_expect()),
              open(META, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('✅ 완료')
    return 0


if __name__ == '__main__':
    sys.exit(main())
