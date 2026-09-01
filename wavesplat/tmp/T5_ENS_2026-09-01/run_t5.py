#!/usr/bin/env python3
"""`T5-ENS` 러너 — 실측 예보 지평에서 **앙상블 전이**를 잰다.

⚠**스레드 고정은 `import torch` 보다 먼저**여야 한다(과거 사고: 스레드 미고정으로 18시간 허비).
⚠**승인 가드 1/4**: `T5_APPROVED=1` **∧** `--approved` 둘 다 있어야 기동한다.
   ⛔에이전트 판단으로 실측 게이트를 끄지 마라 — `APPROVAL_T5_2026-09-01.md` 참조.

집계·문턱·게이트는 **전부 `gates_t5` 를 호출**한다(단일출처). 여기서 재구현하지 않는다.
"""
import os
import sys

# ── ★스레드 고정 (torch 임포트 전) ───────────────────────────────────────────
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '1'

import argparse                                                       # noqa: E402
import hashlib                                                        # noqa: E402
import json                                                           # noqa: E402
import subprocess                                                     # noqa: E402
import time                                                           # noqa: E402

import numpy as np                                                    # noqa: E402
import torch                                                          # noqa: E402

torch.set_num_threads(1)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
B2 = os.path.abspath(os.path.join(HERE, '..', 'B2_REAL_2026-08-23'))
B3 = os.path.abspath(os.path.join(HERE, '..', 'B3_REFIT_2026-08-23'))
PORT = os.path.abspath(os.path.join(HERE, '..', 'bankmmse_port_2026-07-29'))
for _p in (HERE, B2, B3, PORT, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gates_t5 as G                                                  # noqa: E402
from scene_station import scene_station                               # noqa: E402
import zero_ref as ZR                                                 # noqa: E402
import make_gif_bank_deployed as M                                    # noqa: E402

SMOKE_CELLS = [('AA14', 1), ('YS17', 5)]   # ★가장 싼 셀 + 가장 비싼 셀(stride=2, depth 50)


# ─────────────────────────────────────────────────────────────────────────────
def _git_head():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                                       text=True).strip()[:12]
    except Exception:
        return None


def _code_md5():
    """판정 경로 파일의 md5 — 행에 싣는다(⚠`git_head` 는 더러운 트리를 못 잡는다)."""
    out = {}
    for rel in ('wavesplat/optimizer/sequential_update.py',
                'wavesplat/optimizer/learned_init.py',
                'wavesplat/run.py'):
        p = os.path.join(ROOT, rel)
        out[os.path.basename(rel)] = hashlib.md5(open(p, 'rb').read()).hexdigest()[:12]
    for p in (os.path.join(B2, 'scene_station.py'), os.path.join(HERE, 'gates_t5.py')):
        out[os.path.basename(p)] = hashlib.md5(open(p, 'rb').read()).hexdigest()[:12]
    return out


def _ckpt_sha():
    from wavesplat.optimizer.learned_init import DEFAULT_CKPT
    h = hashlib.sha256(open(DEFAULT_CKPT, 'rb').read()).hexdigest()
    return h[:16]


def _prior_device():
    """`learned_init._CACHE` 키의 장치 문자열 — ⛔부채 H7 의 「공짜 처방」."""
    from wavesplat.optimizer import learned_init as LI
    ks = list(getattr(LI, '_CACHE', {}) or {})
    return ks[0][1] if ks else None


def median_field(fields):
    """★점별 median **장** — `T3 run_t3.py:113` 축자 상속. ⛔점수의 median 이 아니다."""
    return np.ascontiguousarray(
        np.median(np.stack([np.asarray(f, dtype=np.float32) for f in fields], axis=0),
                  axis=0), dtype=np.float32)


def score(Z, g, mm, sg):
    return float(np.sqrt(np.mean((Z[mm] - g[mm]) ** 2)) / sg)


def _md5f(Z):
    return hashlib.md5(np.ascontiguousarray(Z, dtype=np.float32).tobytes()).hexdigest()[:12]


# ─────────────────────────────────────────────────────────────────────────────
def run_member(sc, t_shift, t0, frames, cfg):
    """한 멤버의 예보 궤적. 반환 (per_lead 장 dict, 진단 dict).

    ⚠순서는 `B-7 run_b7.py:67-86` 축자 상속: **①먼저 예측·채점 → ②그 시각 관측 투입**.
    lead 0 은 관측 투입 **이전** 상태다. ⛔미래 관측은 채점에 쓰지 않는다.
    """
    from wavesplat.data_loader import FrameData
    from wavesplat.optimizer.sequential_update import make_updater
    import wavesplat.optimizer.sequential_update as SU
    from wavesplat.run import set_seed

    calls, tsteps = [], []
    orig = SU.SequentialUpdater._bank_fit_window
    orig_upd = SU.SequentialUpdater.update

    def wrapped(self, *a, **kw):
        calls.append(1)
        return orig(self, *a, **kw)

    def wrapped_upd(self, frame):
        out = orig_upd(self, frame)
        if getattr(self, '_bank', None) is None and out is not None \
                and out.get('loss') is not None:
            tsteps.append(1)
        return out

    SU.SequentialUpdater._bank_fit_window = wrapped
    SU.SequentialUpdater.update = wrapped_upd
    try:
        set_seed(cfg['seed'])
        upd = make_updater(dict(cfg))
        init = upd.initialize(
            frames, x_range=(float(sc['gxq'].min()), float(sc['gxq'].max())),
            y_range=(float(sc['gyq'].min()), float(sc['gyq'].max())), verbose=False)
        n_init = len(calls)
        gxq, gyq = np.asarray(sc['gxq']), np.asarray(sc['gyq'])
        tx = torch.tensor(gxq, dtype=torch.float32)
        ty = torch.tensor(gyq, dtype=torch.float32)
        Pf, zf = np.asarray(sc['P_fut'], float), np.asarray(sc['z_fut'], float)
        fields = {}
        for li, l_ in enumerate(sc['leads']):
            g = np.asarray(sc['maps_gt'][li], float)
            if np.isfinite(g).sum() > 0:
                fields[float(l_)] = upd.predict_field(
                    tx, ty, t0 + float(l_)).cpu().numpy().reshape(len(gyq), len(gxq))
            if li == 0:
                continue
            tt = t0 + float(l_) + t_shift
            m = np.abs(Pf[:, 2] - tt) < 1e-6
            if m.sum() == 0:
                continue
            nq = int(m.sum())
            upd.update(FrameData(t=float(l_) + t0,
                                 x_obs=torch.tensor(Pf[m, 0], dtype=torch.float32),
                                 y_obs=torch.tensor(Pf[m, 1], dtype=torch.float32),
                                 z_obs=torch.tensor(zf[m], dtype=torch.float32),
                                 confidence=torch.ones(nq, dtype=torch.float32),
                                 incidence_angles=torch.zeros(nq, dtype=torch.float32)))
        diag = dict(
            refit_calls=len(calls) - n_init, init_fit_calls=n_init,
            track_steps=len(tsteps),
            n_atoms=int(getattr(getattr(upd, 'model', None), 'n_gaussians', 0) or 0),
            init_method_readback=(init or {}).get('method')
            or getattr(upd, 'init_method', None),
            real_estimator_readback=getattr(upd, '_real_estimator', None)
            or cfg.get('real_estimator'),
            # ⚠`ensemble_k` 는 **해석적 래퍼**의 k 다(실측 게이트에서 1 로 접힌다).
            # bank 자신의 median-of-k 는 `_bank_seeds` 로만 보인다(`sequential_update.py:1139`).
            ens_k_readback=getattr(upd, 'ensemble_k', None),
            bank_seeds_readback=list(getattr(upd, '_bank_seeds', None) or []),
            n_selected=(init or {}).get('n_selected'),
            seed_used=cfg['seed'],
            dseed_used=cfg.get('learned_diffusion_seed'))
        return fields, diag
    finally:
        SU.SequentialUpdater._bank_fit_window = orig
        SU.SequentialUpdater.update = orig_upd


def run_cell(record, win, arm, approved):
    t0w = time.time()
    G.assert_shipping_defaults()
    sc = scene_station(record, win=win, noise=G.NOISE, with_future_obs=True)
    t_shift = float(np.asarray(sc['P'])[:, 2].min())
    t0 = float(sc['Pq'][0, 2]) - t_shift
    frames = M.frames_from_scene(sc, t_shift, dtype=torch.float32)
    zper = ZR.zero_ref_perlead(record, win=win, noise=G.NOISE)

    base = G.arm_config(arm)
    base.update(depth=float(sc['depth']))
    offs = G.ARMS[arm]['MEM']

    mem_fields, diags = [], []
    for off in offs:
        cfg = dict(base)
        cfg.update(G.member_overrides(arm, off))
        f, d = run_member(sc, t_shift, t0, frames, cfg)
        mem_fields.append(f)
        diags.append(d)

    leads = sorted(mem_fields[0].keys())
    per_lead_med, per_lead_m0, per_lead_members = {}, {}, {}
    md5_med, md5_mem = {}, {}
    for li, l_ in enumerate(sc['leads']):
        l_ = float(l_)
        if l_ not in mem_fields[0]:
            continue
        g = np.asarray(sc['maps_gt'][li], float)
        mm = np.isfinite(g)
        Fs = [mf[l_] for mf in mem_fields]
        Zmed = median_field(Fs) if len(Fs) > 1 else Fs[0]
        k = str(l_)
        per_lead_med[k] = score(Zmed, g, mm, sc['sg'])
        per_lead_m0[k] = score(Fs[0], g, mm, sc['sg'])
        per_lead_members[k] = [score(Z, g, mm, sc['sg']) for Z in Fs]
        md5_med[k] = _md5f(Zmed)
        md5_mem[k] = [_md5f(Z) for Z in Fs]

    zp = {str(float(k)): v for k, v in zper.items()}
    row = dict(
        batch=G.BATCH, arm=arm, record=record, win=win, approved=bool(approved),
        n_members=len(offs), member_offsets=list(offs),
        per_lead_med=per_lead_med, per_lead_m0=per_lead_m0,
        per_lead_members=per_lead_members,
        band_near_med3=G.band_mean(per_lead_med, G.BAND_NEAR),
        band_near_m0=G.band_mean(per_lead_m0, G.BAND_NEAR),
        band_far_med3=G.band_mean(per_lead_med, G.BAND_FAR),
        band_far_m0=G.band_mean(per_lead_m0, G.BAND_FAR),
        zero_per_lead=zp, zero_near=G.band_mean(zp, G.BAND_NEAR),
        zero_far=G.band_mean(zp, G.BAND_FAR),
        field_md5_med_by_lead=md5_med, field_md5_members_by_lead=md5_mem,
        track_steps_members=[d['track_steps'] for d in diags],
        n_atoms_members=[d['n_atoms'] for d in diags],
        n_selected_members=[d['n_selected'] for d in diags],
        refit_calls=sum(d['refit_calls'] for d in diags),
        init_fit_calls=sum(d['init_fit_calls'] for d in diags),
        seed_readback=[d['seed_used'] for d in diags],
        diffusion_seed_readback=[d['dseed_used'] for d in diags],
        init_method_readback=diags[0]['init_method_readback'],
        real_estimator_readback=diags[0]['real_estimator_readback'],
        ens_k_readback=diags[0]['ens_k_readback'],
        bank_seeds_readback=diags[0]['bank_seeds_readback'],
        bank_ens_k_readback=len(diags[0]['bank_seeds_readback']) or None,
        real_gate=base.get('real_domain_gate'),
        arm_cfg_nonempty=bool(G.ARMS[arm]['CFG']),
        deploy_pin_ok=True,
        device=base.get('device'), cuda_available=bool(torch.cuda.is_available()),
        omp_threads=int(os.environ.get('OMP_NUM_THREADS', '0')),
        prior_device=_prior_device(), ckpt_sha=_ckpt_sha(),
        depth=float(sc['depth']), sigma=float(sc['sg']),
        npx=int(np.asarray(sc['maps_gt'][0]).size),
        n_obs=int(len(np.asarray(sc['P']))), n_fut_obs=int(len(np.asarray(sc['z_fut']))),
        code_md5=_code_md5(), git_head=_git_head(),
        wall_s=round(time.time() - t0w, 3))
    # `init_method_readback` 이 None 이면 config 의 의도를 싣되 **표시**한다(공허 통과 방지).
    if row['init_method_readback'] is None:
        row['init_method_readback'] = base.get('init_method')
        row['init_method_readback_source'] = 'config(intent)'
    return row


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--approved', action='store_true')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshard', type=int, default=1)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    # ★승인 가드 1/4 — 환경변수 ∧ 플래그. ⛔둘 중 하나로는 기동하지 않는다.
    env_ok = os.environ.get('T5_APPROVED') == '1'
    if not (env_ok and a.approved):
        print('⛔기동 거부 — 승인 가드 1/4.  T5_APPROVED=1 ∧ --approved 가 모두 필요하다.')
        print('   사유: 이 배치는 real_domain_gate=False 로 실측 12 레코드를 소모한다.')
        print('   근거 문서: APPROVAL_T5_2026-09-01.md')
        return 2

    jobs = SMOKE_CELLS if a.smoke else G.cells()
    if not a.smoke:
        jobs = [c for i, c in enumerate(jobs) if i % a.nshard == a.shard]
    out = a.out or os.path.join(
        HERE, 'smoke_t5.jsonl' if a.smoke else f'results_t5_{a.shard}.jsonl')

    done = set()
    if os.path.exists(out):
        for ln in open(out):
            if ln.strip():
                r = json.loads(ln)
                done.add((r.get('record'), r.get('win'), r.get('arm')))

    n = 0
    with open(out, 'a') as fh:
        for (rec, win) in jobs:
            for arm in G.ARM_ORDER:
                if (rec, win, arm) in done:
                    continue
                t = time.time()
                try:
                    row = run_cell(rec, win, arm, approved=True)
                except Exception as e:                       # noqa: BLE001
                    row = dict(batch=G.BATCH, arm=arm, record=rec, win=win,
                               error=f'{type(e).__name__}: {e}')
                fh.write(json.dumps(row, ensure_ascii=False) + '\n')
                fh.flush()
                n += 1
                print(f'  {rec:7s} win={win} {arm:9s} '
                      f'{time.time() - t:6.1f}s  ok={"error" not in row}', flush=True)
    print(f'✅ {n} 행 → {os.path.relpath(out, ROOT)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
