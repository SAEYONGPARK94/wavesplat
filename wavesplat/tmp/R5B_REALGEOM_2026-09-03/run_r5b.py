"""`R5b-REALGEOM` 러너 — **센서 기하만** 개입한다. 규칙은 `gates_r5.py`(→`gates_b17` 임포트).

⛔`run_b17.py` 의 `_one` 을 축자로 옮기고 **팔 사양 한 줄**만 `gates_r5.arm_spec` 으로 바꾼다.
⛔`CONST` 는 돌리지 않는다 — 동결 `B-17` 행을 읽는다(`G-CONST`).

    python3 tmp/R5_REALGEOM_2026-09-03/run_r5.py [shard] [nshards]

2 팔 × 7 레코드(BS) = **14 셀**. 샤딩 가능·재개 가능(`done` 집합).

행에 싣는 것 (⛔사후 추론 금지 — 전부 **되읽기·계수기**):
  · `derived`       updater 가 config 와 **따로 스냅샷한 속성** (G-7a)
  · `counters`      **실행이 남긴 수** — 결합적합 스텝 수 · VARPRO 집행 스텝 수 (G-7b)
  · `router`        `_auto_selected` 등 라우터 진단 — 도달 불가 키의 **양성 증명** (G-7c)
  ⛔`upd.config` 되읽기는 **싣지 않는다** — `sequential_update.py:399` 가 입력 dict 을
     그대로 붙들어 `cfg[k]==cfg[k]` 항등식이 되고, 읽히지도 않는 키를 통과시킨다.
  · `p_hash`        관측점 집합 해시 — 팔 간 **비트 동일**해야 한다 (G-3, 짝 개입의 증거)
  · `z_hash`        관측값 해시 — 팔 간 **달라야** 한다 (G-4, 개입 발화)
  · `sigma_stats`   σ_j 의 min/med/max/rms · r · cosθ · **클램프 비율** · mm 척도
  · `window`        ★채점 창(i0·nfw·n_lead·T_all·dt) — 팔 간 같아야 한다 (G-8)
  · `zero_ref`      ★참 제로필드(Z≡0 을 **같은 채점식**에 통과) ⛔상수 1.0 금지
  · `src_md5`       임포트 경로 파일 md5 (git_head 만으로는 더러운 작업트리를 못 잡는다)
"""
import hashlib
import json
import os
import subprocess
import sys
import time
import warnings

warnings.filterwarnings('ignore')
import numpy as np                                       # noqa: E402
import torch                                             # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
B2 = os.path.abspath(os.path.join(HERE, '..', 'B2_REAL_2026-08-23'))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
B17 = os.path.abspath(os.path.join(HERE, '..', 'B17_REALNOISE_2026-08-25'))
for p in (HERE, B17, B2, ROOT, os.path.join(ROOT, 'scripts'),
          os.path.join(ROOT, 'tmp', 'bankmmse_port_2026-07-29')):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(ROOT)

import gates_r5b as G                                     # noqa: E402
from noise_models import scene_noise                     # noqa: E402
import make_gif_bank_deployed as M                       # noqa: E402

OUT = os.path.join(HERE, 'results_r5b')
GEO, GEO_SHA = G.measured_geometry()          # ★R3b 동결 verdict 에서 파싱(손복사 0)
_B17_ROWS = None


def b17_frozen():
    """동결 `B-17` 행 — 앵커 대조와 `CONST` 기준에 쓴다. ⛔재실행하지 않는다."""
    global _B17_ROWS
    if _B17_ROWS is None:
        import glob
        _B17_ROWS = {}
        for p in sorted(glob.glob(G.B17_GLOB)):
            for l in open(p):
                if l.strip():
                    r = json.loads(l)
                    if 'error' not in r:
                        _B17_ROWS[(r['cell'], r['record'])] = r
    return _B17_ROWS


def _src_md5():
    out = {}
    # ★임포트 경로 파일 전부 — `B-17` 것은 그 디렉터리에서, 이 배치 것은 여기서.
    for f in (os.path.join(B17, 'gates_b17.py'), os.path.join(B17, 'noise_models.py'),
              os.path.join(B17, 'run_b17.py'), os.path.join(B2, 'scene_station.py'),
              os.path.join(B2, 'gates_b2.py'),
              os.path.join(HERE, 'gates_r5b.py'), os.path.join(HERE, 'run_r5b.py'),
              os.path.join(ROOT, 'wavesplat', 'optimizer', 'sequential_update.py'),
              os.path.join(ROOT, 'wavesplat', 'run.py')):
        out[os.path.basename(f)] = hashlib.md5(open(f, 'rb').read()).hexdigest()
    return out


def _git_head():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                       cwd=ROOT, text=True).strip()
    except Exception:                                     # noqa: BLE001
        return None


SRC_MD5, GIT_HEAD = _src_md5(), _git_head()


def _one(record, arm):
    from wavesplat.optimizer.sequential_update import make_updater
    from wavesplat.run import DEFAULT_CONFIG, set_seed
    t0w = time.time()
    spec = G.arm_spec(arm, record, GEO)
    sc = scene_noise(record, spec, win=G.WIN, sigma0=G.SIGMA0, sigma_r=G.SIGMA_R,
                     d_ref=G.D_REF, cos_min=G.COS_MIN, mm_target=G.MM_TARGET)
    t_shift = float(np.asarray(sc['P'])[:, 2].min())
    t0 = float(sc['Pq'][0, 2]) - t_shift
    frames = M.frames_from_scene(sc, t_shift, dtype=torch.float32)
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(depth=float(sc['depth']), seed=G.SEED, device='cpu')
    cfg.update(G.ESTIMATOR_CFG)                # ★배포 상수 셋을 **명시** (흡수 금지)
    set_seed(G.SEED)
    upd = make_updater(cfg)
    res = upd.initialize(frames,
                         x_range=(float(sc['gxq'].min()), float(sc['gxq'].max())),
                         y_range=(float(sc['gyq'].min()), float(sc['gyq'].max())),
                         verbose=False)
    gxq, gyq = np.asarray(sc['gxq']), np.asarray(sc['gyq'])
    curve, zcurve = [], []
    for li, l_ in enumerate(sc['leads']):
        g = np.asarray(sc['maps_gt'][li], float)
        mm = np.isfinite(g)
        if not mm.any():
            curve.append(float('nan'))
            zcurve.append(float('nan'))
            continue
        Z = upd.predict_field(torch.tensor(gxq, dtype=torch.float32),
                              torch.tensor(gyq, dtype=torch.float32),
                              t0 + float(l_)).cpu().numpy().reshape(len(gyq), len(gxq))
        curve.append(float(np.sqrt(np.mean((Z[mm] - g[mm]) ** 2)) / sc['sg']))
        # ★참 제로필드 = Z≡0 을 **같은 채점식**에 통과시킨 값
        zcurve.append(float(np.sqrt(np.mean(g[mm] ** 2)) / sc['sg']))
    # ── ★개입 증거 (BLOCKER B-1 수리, 2026-08-26) ────────────────────────────
    # ⛔`upd.config` 되읽기 금지: `sequential_update.py:399` 가 `self.config = config` 로
    #   입력 dict 을 복사 없이 붙든다 ⟹ 대조가 항등식이고 **실패할 수 없다.**
    # (1) 파생 속성 — 생성자가 config 와 **따로** 스냅샷한 것만 증거로 센다.
    derived = {k: getattr(upd, a, G.ATTR_MISSING)
               for k, a in G.DERIVED_READBACK.items()}
    # (2) 계수기 — **실행이 남긴 수**. 설정 echo 와 구별하려고 원 계수만 쓴다.
    vp = (res or {}).get('varpro') or {}
    counters = {
        'n_loss_steps': len((res or {}).get('loss_history') or []),
        'vp_steps_executed': vp.get('vp_steps'),
        'vp_entry': vp.get('vp_entry'),
        'vp_skipped': vp.get('vp_skipped'),
        'final_n': (res or {}).get('final_n'),
        'k_snap': (res or {}).get('k_snap'),
        # ⚠아래 둘은 **설정 echo** 다(같은 config 를 use-site 에서 다시 읽은 값).
        #   비교용으로만 싣고 게이트는 위의 원 계수를 본다.
        'echo_n_iters': (res or {}).get('n_iters'),
        'echo_varpro_steps': (res or {}).get('varpro_steps'),
    }
    # (3) 라우터 진단 — `INERT_KEYS` 가 읽히는 코드가 **실행되지 않았다**는 양성 증거.
    router = {a: getattr(upd, a, G.ATTR_MISSING) for a in G.ROUTER_ATTRS}
    h = lambda a: hashlib.sha256(                                        # noqa: E731
        np.ascontiguousarray(a, dtype=np.float64).tobytes()).hexdigest()[:16]
    return dict(cell=arm, record=record, win=G.WIN, arm_spec=spec,
                geom_sha=GEO_SHA, raw_center=GEO[record]['raw_center'],
                seed=G.SEED, threads=torch.get_num_threads(),
                nowcast=curve[0], leadmean=float(np.nanmean(curve)), curve=curve,
                zero_ref=zcurve[0], zero_curve=zcurve,
                beats_zero=bool(curve[0] < zcurve[0]),
                p_hash=h(sc['P']), z_hash=h(sc['z']), q_hash=h(sc['Pq']),
                sigma_stats=sc['sigma_stats'], window=sc['window'],
                n_obs_used=int(sc['n_obs_used']), npx=int(sc['npx']),
                n_atoms=int(getattr(getattr(upd, 'model', None), 'n_gaussians', 0) or 0),
                varpro_fired=bool((res or {}).get('varpro')),
                varpro_steps=(res or {}).get('varpro_steps'),
                init_iters=(res or {}).get('n_iters'),
                derived=derived, counters=counters, router=router,
                sigma=float(sc['sg']), Hs=float(sc['Hs']),
                lam=float(2 * np.pi / sc['kp']), depth=float(sc['depth']),
                src_md5=SRC_MD5, git_head=GIT_HEAD, wall_s=time.time() - t0w)


def cells():
    # ⛔앵커는 재실행하지 않는다 --- `R5` 가 relerr 0.0 으로 확립했고 `G-INHERIT` 가 대조한다.
    return [(rec, arm) for rec in G.TREAT_RECORDS for arm in G.ARM_ORDER]


def main():
    shard = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    nsh = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    out = f'{OUT}_{shard}.jsonl'
    done = set()
    if os.path.exists(out):
        for l in open(out):
            try:
                r = json.loads(l)
            except json.JSONDecodeError:
                continue
            if 'error' not in r:
                done.add((r['record'], r['cell']))
    todo = [c for i, c in enumerate(cells()) if i % nsh == shard and c not in done]
    print(f'R5b shard {shard}/{nsh}: {len(todo)} 셀 (완료 {len(done)})', flush=True)
    with open(out, 'a') as fh:
        for i, (rec, arm) in enumerate(todo, 1):
            try:
                row = _one(rec, arm)
            except Exception as e:                        # noqa: BLE001
                import traceback
                row = dict(cell=arm, record=rec, error=f'{type(e).__name__}: {e}',
                           tb=traceback.format_exc()[-900:])
            fh.write(json.dumps(row, ensure_ascii=False) + '\n')
            fh.flush()
            if 'error' in row:
                tag = row['error'][:70]
            else:
                s = row['sigma_stats']
                tag = (f"nRMSE {row['nowcast']:.4f} (zero {row['zero_ref']:.4f}) "
                       f"{'WIN' if row['beats_zero'] else '   '} "
                       f"σ[{s['min']:.4f},{s['max']:.4f}] N={row['n_atoms']}")
            print(f'[{i}/{len(todo)}] {rec:6s} {arm:13s} {tag}', flush=True)


if __name__ == '__main__':
    main()
