"""B-18 러너 — **밀도 × 잡음 완전요인** 개입 (실측 12 레코드).

    python3 tmp/B18_DENSITY_NOISE_2026-08-25/run_b18.py [shard] [nshards]
    python3 tmp/B18_DENSITY_NOISE_2026-08-25/run_b18.py --smoke     # 1셀 연기시험

⛔**게이트·팔·문턱·라벨은 `gates_b18.py` 하나에서만 온다.** 여기에는 재기재가 없다.
재개 가능(샤드 파일의 done set) · 샤딩 가능(`results_b18_<shard>.jsonl`, 분석기가 glob).
⚠**인자 없이 돌리면 단일 샤드**(`results_b18_0.jsonl` 한 파일) — 배치 산출물 검사기의
`results~verdict` 열이 그때만 OK 로 읽힌다(샤딩하면 `--`, `STATUS_2026-08-24 §9 ②`).

행에 싣는 것 — ⛔사후 추론 금지, 전부 **되읽기**:
  · `derived`    = updater 객체의 **파생 속성**(`:402 lambda_l1` · `:413 init_method`)
    ⛔**`upd.config` 를 읽지 않는다** — `:399` 가 입력 dict 을 복사 없이 붙들어
    `upd.config[k] == cfg[k]` 는 **항등식**이다(초판 `G-6` 이 그것이었다).
  · `counters`   = `initialize()` 반환 dict = **본문이 실제로 쓴 값**
    (`n_iters` ← `:889` · `varpro_steps` ← `:913` · `vp_steps_exec` ← `varpro.py:138`)
  · `router`     = `_auto_*` 진단 속성 ⟹ 전부 None 이면 `auto` 블록(`:689`) **미진입**
    = 팔의 `real_domain_gate`(`:706`)·`auto_switch_thresh`(`:1008`) 가 **읽히지 않았다**
  · `p_hash`     = 관측 좌표 `P` 의 해시 ⟹ ★**난수 소비 동수**(σ 축 짝 개입)의 증거
  · `z_hash`     = 관측값 `z` 의 해시 ⟹ 잡음 개입 발화의 증거
  · `n_obs_used` / `m_cap`         ⟹ 밀도 되읽기
  · `zero_ref`   = `Z≡0` 을 **같은 채점식**에 통과시킨 값 (⛔상수 1.0 아님)
  · `win`·`i0`·`t0_abs`·`npx`·`n_gt_finite`·`dt` ⟹ ★**채점 창**(두 팔이 다른 창에서
    채점된 사고가 있었다)

★**추정 호출 순서는 `tmp/B14_OBSMODEL_2026-08-24/run_b14.py:_one` 과 글자 그대로 같다.**
그 충실도는 게이트 `G-1`(PC 팔 = `B-14b` 비트 동일, ABORT)이 증명한다 —
⛔해시·메타데이터 계산은 `set_seed` **앞**에서만 하고 난수를 소비하지 않는다.
"""
import hashlib
import json
import os
import sys
import time
import warnings

warnings.filterwarnings('ignore')
import numpy as np                                    # noqa: E402
import torch                                          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
B14 = os.path.join(ROOT, 'tmp', 'B14_OBSMODEL_2026-08-24')
B2 = os.path.join(ROOT, 'tmp', 'B2_REAL_2026-08-23')
for p in (HERE, B14, B2, ROOT, os.path.join(ROOT, 'scripts'),
          os.path.join(ROOT, 'tmp', 'bankmmse_port_2026-07-29')):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(ROOT)

import gates_b18 as G                                 # noqa: E402
from obs_models import scene_obs                      # noqa: E402  (B-14 동결 장면 빌더)
import make_gif_bank_deployed as M                    # noqa: E402

# ★연기시험 셀 — **어느 검정에도 쓰이지 않는 내부 셀**(모서리 아님)로 골랐다.
SMOKE_CELLS = [(G.GRID, 'AA14', 48, 0.005, G.DENSITY_CAP_PER_FRAME * 48)]


def _one(arm, record, m_scan, noise, m_cap):
    from wavesplat.optimizer.sequential_update import make_updater
    from wavesplat.run import DEFAULT_CONFIG, set_seed
    t0w = time.time()
    # 관측 모형은 배포와 같은 `IID`(균등 추출 + 등방 i.i.d.) 하나뿐이다 —
    # 이 배치가 움직이는 것은 **밀도와 잡음 두 축뿐**이다(`B-14b` 가 모형 축을 닫았다).
    sc = scene_obs(record, win=G.WIN, model='IID', noise=noise,
                   m_scan=m_scan, m_cap=m_cap)
    P = np.asarray(sc['P'], float)
    t_shift = float(P[:, 2].min())
    t0 = float(sc['Pq'][0, 2]) - t_shift
    ph = hashlib.sha256(P.tobytes()).hexdigest()[:16]
    zh = hashlib.sha256(np.asarray(sc['z'], float).tobytes()).hexdigest()[:16]

    frames = M.frames_from_scene(sc, t_shift, dtype=torch.float32)
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(depth=float(sc['depth']), seed=G.SEED, device='cpu')
    cfg.update(G.ARM_PROP)                            # ⛔빈 dict 아님 · 배포 상수 명시
    set_seed(G.SEED)
    upd = make_updater(cfg)
    res = upd.initialize(frames,
                         x_range=(float(sc['gxq'].min()), float(sc['gxq'].max())),
                         y_range=(float(sc['gyq'].min()), float(sc['gyq'].max())),
                         verbose=False)
    gxq, gyq = np.asarray(sc['gxq']), np.asarray(sc['gyq'])
    g = np.asarray(sc['maps_gt'][0], float)
    mm = np.isfinite(g)
    Z = upd.predict_field(torch.tensor(gxq, dtype=torch.float32),
                          torch.tensor(gyq, dtype=torch.float32),
                          t0).cpu().numpy().reshape(len(gyq), len(gxq))
    now = float(np.sqrt(np.mean((Z[mm] - g[mm]) ** 2)) / sc['sg'])
    zero = float(np.sqrt(np.mean(g[mm] ** 2)) / sc['sg'])     # ★참 제로필드(상수 아님)

    # ── ★개입 되읽기 — ⛔`upd.config` 는 입력 dict **그 자체**다(:399) ⟹ 항등식 ──
    # 계약은 `gates_b18` 에 있다. 여기서는 세 곳에서 **긁어오기만** 한다.
    #  (1) 파생 속성   — `__init__` 이 config 과 독립으로 스냅샷한 것 (:402/:413)
    derived = {k: getattr(upd, k, '<MISSING>') for k in G.DERIVED_ATTRS}
    #  (2) 개입 계수기 — `initialize()` 가 반환한, **본문이 실제로 쓴** 값
    rr = res or {}
    vp = rr.get('varpro') or {}
    counters = dict(n_iters=rr.get('n_iters'), varpro_steps=rr.get('varpro_steps'),
                    vp_steps_exec=vp.get('vp_steps'), vp_skipped=vp.get('vp_skipped'),
                    vp_obj_before=vp.get('vp_obj_before'), vp_obj_after=vp.get('vp_obj_after'),
                    vp_k_shift=vp.get('vp_k_shift'),
                    final_n=rr.get('final_n'), k_snap=rr.get('k_snap'))
    #  (3) 라우터 진단 — auto 경로에서만 채워진다(:417/:418/:427) ⟹ 전부 None = 미진입
    router = {k: getattr(upd, k, '<MISSING>') for k in G.ROUTER_ATTRS}
    return dict(arm=arm, record=record, m_scan=m_scan, noise=noise, m_cap=m_cap,
                nowcast=now, zero_ref=zero, beats_zero=bool(now < zero),
                n_obs_used=int(sc['n_obs_used']), p_hash=ph, z_hash=zh,
                sigma=float(sc['sg']), nsr=(noise / float(sc['sg'])),
                lam=float(2 * np.pi / sc['kp']), depth=float(sc['depth']),
                win=G.WIN, i0=int(sc['seed']), t0_abs=float(sc['t0']),
                npx=int(sc['npx']), n_gt_finite=int(mm.sum()),
                dt=float(sc['dt']), n_leads=int(len(sc['leads'])), lead_scored=0.0,
                n_atoms=int(getattr(getattr(upd, 'model', None), 'n_gaussians', 0) or 0),
                # ⛔`bool(res['varpro'])` 로 판정하지 않는다 — 건너뛴 경우에도 truthy dict 이다
                # (`varpro.py:201` 이 `{'vp_entry':1,…}` 를 먼저 만들고 `vp_skipped` 를 적는다).
                varpro_fired=bool(vp.get('vp_steps')),
                derived=derived, counters=counters, router=router,
                wall_s=time.time() - t0w)


def _drive(todo, out):
    done = set()
    if os.path.exists(out):
        for line in open(out):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if 'error' not in r:
                done.add((r['arm'], r['record'], r['m_scan'], r['noise'], r['m_cap']))
    todo = [c for c in todo if c not in done]
    print(f'B-18: {len(todo)} 셀 남음 (완료 {len(done)}) -> {os.path.basename(out)}',
          flush=True)
    with open(out, 'a') as fh:
        for i, c in enumerate(todo, 1):
            arm, rec, m_, nz, cap = c
            try:
                row = _one(*c)
            except Exception as e:                    # noqa: BLE001
                import traceback
                row = dict(arm=arm, record=rec, m_scan=m_, noise=nz, m_cap=cap,
                           error=f'{type(e).__name__}: {e}',
                           tb=traceback.format_exc()[-400:])
            fh.write(json.dumps(row, ensure_ascii=False) + '\n')
            fh.flush()
            tag = row.get('error') or (f"nRMSE {row['nowcast']:.4f} "
                                       f"(zero {row['zero_ref']:.4f}) "
                                       f"{'WIN' if row['beats_zero'] else '   '} "
                                       f"obs={row['n_obs_used']} {row['wall_s']:.1f}s")
            print(f'[{i}/{len(todo)}] {arm:10s} {rec:6s} m={m_:<4} σ={nz:<6} {tag}',
                  flush=True)


def main():
    if '--smoke' in sys.argv:
        out = os.path.join(HERE, 'smoke_b18.jsonl')   # ⛔`results*` 아님 — 판정에 안 들어간다
        print('★연기시험 — 내부 셀 1개. 결과 파일은 판정에 쓰지 않는다.', flush=True)
        _drive(SMOKE_CELLS, out)
        return
    shard = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    nsh = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    assert 0 <= shard < nsh, (shard, nsh)
    todo = [c for i, c in enumerate(G.cells()) if i % nsh == shard]
    _drive(todo, os.path.join(HERE, f'results_b18_{shard}.jsonl'))


if __name__ == '__main__':
    main()
