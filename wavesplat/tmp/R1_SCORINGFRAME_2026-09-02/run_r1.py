"""`R1-SCORINGFRAME` 러너 — **한 번의 런에서 두 채점 규약을 동시에** 잰다.

★그것이 가능한 이유(PREREG §1): 후보 ① 은 `Zmem` 을 **건드리지 않는다** ⟹ 관측
`(P, z)`·질의 격자·`sg` 가 비트 동일 ⟹ **추정기 출력이 비트 동일** ⟹ 같은 예측장을
두 규약으로 채점하면 된다. 그리고 **옛 규약이 동결 `B-2` 결과를 재현하는지**가 게이트다.

⛔추정 체인은 `tmp/B2_REAL_2026-08-23/run_b2.py:26-61` 을 축자로 옮긴다(팔은 임포트).
"""
import hashlib
import json
import os
import sys
import time
import warnings

warnings.filterwarnings('ignore')
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
          'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[v] = '1'
import numpy as np                                                  # noqa: E402
import torch                                                        # noqa: E402
torch.set_num_threads(1)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
B2DIR = os.path.join(ROOT, 'tmp', 'B2_REAL_2026-08-23')
for p in (HERE, B2DIR, ROOT, os.path.join(ROOT, 'scripts'),
          os.path.join(ROOT, 'tmp', 'bankmmse_port_2026-07-29')):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(ROOT)
import gates_r1 as G                                                # noqa: E402
import scene_station as SS                                          # noqa: E402
from scene_station import scene_station                             # noqa: E402
from d3_sa_multirecord import MRecord, _unpack                      # noqa: E402
import make_gif_bank_deployed as M                                  # noqa: E402

OUT = os.path.join(HERE, 'results_r1.jsonl')
PROBE = os.path.join(HERE, 'probes_r1.jsonl')
FROZEN_B2 = os.path.join(B2DIR, 'results.jsonl')

_MD5_FILES = [
    os.path.join(ROOT, 'wavesplat', 'optimizer', 'sequential_update.py'),
    os.path.join(ROOT, 'wavesplat', 'optimizer', 'learned_init.py'),
    os.path.join(ROOT, 'wavesplat', 'run.py'),
    os.path.join(B2DIR, 'scene_station.py'),
    os.path.join(B2DIR, 'gates_b2.py'),
    os.path.join(HERE, 'gates_r1.py'),
]


def code_md5():
    return {os.path.basename(p): hashlib.md5(open(p, 'rb').read()).hexdigest()[:12]
            for p in _MD5_FILES}


def _raw_window(rec, i0, n):
    """`MRecord.window` 와 같되 **demean 하지 않는다** (`d3_sa_multirecord.py:133-145`)."""
    s = rec.stride
    Z = (_unpack(rec.zd, np.s_[i0:i0 + n, ::s, ::s]) if s > 1
         else _unpack(rec.zd, np.s_[i0:i0 + n]))
    Z[:, ~rec.valid] = np.nan
    return Z


def scene_geometry(record):
    """`scene_station` 이 내부에서 고르는 `i0`·`nfw`·`sq` 를 **같은 규약으로** 재현한다."""
    cfg = SS.RECORDS_12[record]
    rec = MRecord(os.path.join(ROOT, cfg['path']), stride=cfg['stride'])
    nfw = int(round(40.0 / rec.dt))
    n_lead = int(round(G.LEAD_MAX / rec.dt))
    if rec.T_all < nfw + n_lead + 4:
        n_lead = max(2, (rec.T_all - nfw - 4))
    lo, hi = nfw + 1, rec.T_all - n_lead - 2
    i0 = int(np.linspace(lo, hi, 6).astype(int)[G.WIN])
    sq = max(1, int(round(4.0 / rec.dx)))
    return rec, nfw, n_lead, i0, sq


def delta_and_probe(record):
    """후보 ① 의 보정장 `mu_fut − mu_mem` 과 런 0회 진단 두 개를 낸다."""
    rec, nfw, n_lead, i0, sq = scene_geometry(record)
    mu_mem = np.nanmean(_raw_window(rec, i0 - nfw + 1, nfw), axis=0)
    mu_fut = np.nanmean(_raw_window(rec, i0, n_lead + 1), axis=0)
    delta_q = (mu_fut - mu_mem)[::sq, ::sq]

    # 오라클 = 관측이 뽑히는 그 프레임(기억창 규약)을 질의점에 얹은 것
    Zmem = rec.window(i0 - nfw + 1, nfw)
    sg = float(np.nanstd(Zmem))
    oracle = Zmem[-1][::sq, ::sq]
    g_old = rec.window(i0, n_lead + 1)[0][::sq, ::sq]
    g_new = g_old + delta_q
    m = np.isfinite(g_old) & np.isfinite(oracle) & np.isfinite(g_new)
    rms = lambda a: float(np.sqrt(np.mean(a[m] ** 2)) / sg)
    probe = dict(record=record, sg=sg,
                 oracle_new=rms(oracle - g_new), oracle_old=rms(oracle - g_old))

    # ★G-INVAR — `i0` 고정, 채점창 길이만 훑는다
    zs_old, zs_new = [], []
    for nl in G.NLEAD_GRID:
        nl = n_lead if nl is None else nl
        raw = _raw_window(rec, i0, nl + 1)
        mu = np.nanmean(raw, axis=0)
        go = (raw[0] - mu)[::sq, ::sq]
        gn = (raw[0] - mu_mem)[::sq, ::sq]
        mm = np.isfinite(go)
        zs_old.append(float(np.sqrt(np.mean(go[mm] ** 2)) / sg))
        zs_new.append(float(np.sqrt(np.mean(gn[mm] ** 2)) / sg))
    probe['invar_old_spread'] = max(zs_old) / min(zs_old)
    probe['invar_new_spread'] = max(zs_new) / min(zs_new)
    probe['zero_old'] = zs_old[-1]
    probe['zero_new'] = zs_new[-1]
    return delta_q, probe


def frozen_b2():
    out = {}
    for line in open(FROZEN_B2):
        if line.strip():
            r = json.loads(line)
            if 'error' not in r:
                out[(r['cell'], r['record'])] = r
    return out


def run(record, arm, delta_q, frozen):
    from wavesplat.optimizer.sequential_update import make_updater
    from wavesplat.run import DEFAULT_CONFIG, set_seed
    t0w = time.time()
    sc = scene_station(record, win=G.WIN)
    t_shift = float(np.asarray(sc['P'])[:, 2].min())
    t0 = float(sc['Pq'][0, 2]) - t_shift
    frames = M.frames_from_scene(sc, t_shift, dtype=torch.float32)
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(depth=float(sc['depth']), seed=G.SEED, device='cpu')
    cfg.update(G.ARMS[arm])
    set_seed(G.SEED)
    upd = make_updater(cfg)
    res = upd.initialize(frames,
                         x_range=(float(sc['gxq'].min()), float(sc['gxq'].max())),
                         y_range=(float(sc['gyq'].min()), float(sc['gyq'].max())),
                         verbose=False)
    gxq, gyq = np.asarray(sc['gxq']), np.asarray(sc['gyq'])
    curve_old, curve_new = [], []
    for li, l_ in enumerate(sc['leads']):
        g_old = np.asarray(sc['maps_gt'][li], float)
        g_new = g_old + delta_q
        mm = np.isfinite(g_old)
        if not mm.any():
            curve_old.append(float('nan')); curve_new.append(float('nan')); continue
        Z = upd.predict_field(torch.tensor(gxq, dtype=torch.float32),
                              torch.tensor(gyq, dtype=torch.float32),
                              t0 + float(l_)).cpu().numpy().reshape(len(gyq), len(gxq))
        curve_old.append(float(np.sqrt(np.mean((Z[mm] - g_old[mm]) ** 2)) / sc['sg']))
        curve_new.append(float(np.sqrt(np.mean((Z[mm] - g_new[mm]) ** 2)) / sc['sg']))

    h = lambda a: hashlib.md5(np.ascontiguousarray(a, dtype=np.float64)).hexdigest()[:12]
    fz = frozen.get((arm, record))
    scene_match = bool(fz and h(np.column_stack([sc['P'], sc['z']])) == fz['obs_hash']
                       and h(sc['Pq']) == fz['q_hash'])
    rel = 1.0
    if fz:
        a = np.asarray(fz['curve'], float); b = np.asarray(curve_old, float)
        mm = np.isfinite(a) & np.isfinite(b)
        rel = float(np.max(np.abs(b[mm] - a[mm]) / np.maximum(np.abs(a[mm]), 1e-12))) if mm.any() else 1.0
    eff = getattr(upd, 'config', {}) or {}
    arm_ok = all(eff.get(k) == v for k, v in G.ARMS[arm].items()) and bool(G.ARMS[arm]) or arm == 'SHIP'
    return dict(cell=arm, record=record, win=G.WIN,
                nowcast_old=curve_old[0], nowcast_new=curve_new[0],
                curve_old=curve_old, curve_new=curve_new,
                scene_match=scene_match, frozen_relerr=rel, arm_ok=bool(arm_ok),
                n_obs=int(len(sc['z'])), n_atoms=(getattr(getattr(upd, 'model', None),
                                                          'n_gaussians', 0) or 0),
                varpro_fired=bool((res or {}).get('varpro')),
                real_gate=getattr(upd, '_auto_real_gate', None),
                sigma=float(sc['sg']), Hs=float(sc['Hs']),
                pid=os.getpid(), threads=torch.get_num_threads(),
                code_md5=code_md5(), wall_s=round(time.time() - t0w, 3))


def main():
    frozen = frozen_b2()
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            if line.strip():
                r = json.loads(line)
                if 'error' not in r:
                    done.add((r['cell'], r['record']))
    probes = {}
    if os.path.exists(PROBE):
        for line in open(PROBE):
            if line.strip():
                p = json.loads(line)
                probes[p['record']] = p
    with open(PROBE, 'a') as pf, open(OUT, 'a') as fh:
        for rec in G.RECORDS:
            if rec not in probes:
                _, pr = delta_and_probe(rec)
                pf.write(json.dumps(pr, ensure_ascii=False) + '\n'); pf.flush()
                probes[rec] = pr
                print(f'  probe {rec:8s} oracle new={pr["oracle_new"]:.3e} '
                      f'old={pr["oracle_old"]:.3e} · invar old={pr["invar_old_spread"]:.3f}x',
                      flush=True)
            dq, _ = delta_and_probe(rec)
            for arm in G.ARM_ORDER:
                if (arm, rec) in done:
                    continue
                t = time.time()
                row = run(rec, arm, dq, frozen)
                fh.write(json.dumps(row, ensure_ascii=False) + '\n'); fh.flush()
                print(f'  {arm:4s} {rec:8s} {time.time()-t:5.1f}s '
                      f'old={row["nowcast_old"]:.4f} new={row["nowcast_new"]:.4f} '
                      f'frozen_rel={row["frozen_relerr"]:.1e}', flush=True)
    print('✅ 완료')
    return 0


if __name__ == '__main__':
    sys.exit(main())
