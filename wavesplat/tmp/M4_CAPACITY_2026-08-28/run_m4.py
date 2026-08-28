"""M4 러너 — **Phase B (실측)** · **Phase C (합성 대조)**. 규칙은 `gates_m4.py` 단일출처.

    CUDA_VISIBLE_DEVICES= python3 tmp/M4_CAPACITY_2026-08-28/run_m4.py real 0 4
    CUDA_VISIBLE_DEVICES= python3 tmp/M4_CAPACITY_2026-08-28/run_m4.py syn  0 8
    CUDA_VISIBLE_DEVICES= python3 tmp/M4_CAPACITY_2026-08-28/run_m4.py pc          # 양성대조

인자 = `<stratum> <shard_idx> <n_shards>`. 출력 = `results_m4_<i>.jsonl`(real 0–3 ·
syn 10–17) / `results_m4_9.jsonl`(PC — 9 는 본배치 샤드 밖의 예약 숫자, `M2 AMENDMENT_1`
가 찾은 검사기 `_sfx()` 함정을 피한다).

⛔GPU 금지: 이 파일도 import 전에 `CUDA_VISIBLE_DEVICES=''` 를 박고 행마다
`cuda_available` 을 되읽어 남긴다(`G-18`).
⛔확산 금지: 실측 원자는 `probe_m4.py` 가, 합성 원자는 `B-22` 가 이미 동결했다.
   `SHIP-K-INL`(캐시 양성대조) 만 예외이고 `G-17` 이 그 예외를 명시적으로 센다.
"""
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('OMP_NUM_THREADS', '1')

import hashlib     # noqa: E402
import json        # noqa: E402
import time        # noqa: E402
import warnings    # noqa: E402

warnings.filterwarnings('ignore')
import numpy as np  # noqa: E402
import torch        # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
for p in (HERE, ROOT, os.path.join(ROOT, 'scripts'),
          os.path.join(ROOT, 'tmp', 'B2_REAL_2026-08-23'),
          os.path.join(ROOT, 'tmp', 'B22_PRIOR_ABLATION_2026-08-27'),
          os.path.join(ROOT, 'tmp', 'bankmmse_port_2026-07-29')):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(ROOT)

import gates_m4 as G                        # noqa: E402
from scene_station import scene_station     # noqa: E402
import make_gif_bank_deployed as M          # noqa: E402

ATOMS_REAL = os.path.join(HERE, 'atoms_real')
ATOMS_SYN = os.path.join(ROOT, 'tmp/B22_PRIOR_ABLATION_2026-08-27/atoms')
_PROPOSE = [0]


def _install_propose_counter():
    """확산 제안 호출 수를 센다 — *"주입했다"* 의 **직접 증거**(B-22 규약 승계)."""
    import wavesplat.optimizer.learned_init as LI
    if getattr(LI.propose_atoms, '_m4_count', False):
        return
    real = LI.propose_atoms

    def counting(*a, **kw):
        _PROPOSE[0] += 1
        return real(*a, **kw)

    counting._m4_count = True
    LI.propose_atoms = counting


def _md5(atoms):
    """원자 풀 해시 — ★`gates_b22.atoms_md5` **그 함수**를 쓴다.

    ⛔자체 규약으로 다시 짜면 수입한 `B-22` 행과 비교가 되지 않는다(`G-19` 가 그것을 잡는다).
    """
    import gates_b22 as B22
    return B22.atoms_md5(atoms)


def _h(a):
    return hashlib.md5(np.ascontiguousarray(a, dtype=np.float64)).hexdigest()[:12]


# ── Phase B: 실측 ────────────────────────────────────────────────────────────
def run_real(record, arm, seed):
    from wavesplat.optimizer.sequential_update import make_updater
    import wavesplat.optimizer.sequential_update as SU
    from wavesplat.run import set_seed
    k = G.arm_knobs(arm)
    t0w = time.time()
    sc = scene_station(record, win=G.WIN)
    t_shift = float(np.asarray(sc['P'])[:, 2].min())
    t0 = float(sc['Pq'][0, 2]) - t_shift
    frames = M.frames_from_scene(sc, t_shift, dtype=torch.float32)
    atoms = None
    if not k['inline']:
        atoms = json.load(open(os.path.join(ATOMS_REAL,
                                            f'{record}__s{seed}.json')))['atoms']
    cfg = G.arm_config(arm, atoms, depth=float(sc['depth']), seed=seed, real=True)
    _install_propose_counter()
    _PROPOSE[0] = 0
    old_si = SU.SCALE_INVARIANT_L1
    SU.SCALE_INVARIANT_L1 = bool(k['si_l1'])
    # 성장 **가지** 계수기 (B-2f 교훈: 함수가 아니라 가지를 센다)
    calls = [0]
    orig = SU.SequentialUpdater._adaptive_prune_grow

    def wrapped(self, *a, **kw):
        n0 = self.model.n_gaussians
        r = orig(self, *a, **kw)
        calls[0] += 1
        wrapped.dn.append(int(self.model.n_gaussians - n0))
        return r

    wrapped.dn = []
    SU.SequentialUpdater._adaptive_prune_grow = wrapped
    si_read = None
    try:
        set_seed(int(seed))
        upd = make_updater(cfg)
        res = upd.initialize(frames,
                             x_range=(float(sc['gxq'].min()), float(sc['gxq'].max())),
                             y_range=(float(sc['gyq'].min()), float(sc['gyq'].max())),
                             verbose=False)
        gxq, gyq = np.asarray(sc['gxq']), np.asarray(sc['gyq'])
        curve = []
        for li, l_ in enumerate(sc['leads']):
            g = np.asarray(sc['maps_gt'][li], float)
            mm = np.isfinite(g)
            if not mm.any():
                curve.append(float('nan'))
                continue
            Z = upd.predict_field(torch.tensor(gxq, dtype=torch.float32),
                                  torch.tensor(gyq, dtype=torch.float32),
                                  t0 + float(l_)).cpu().numpy().reshape(len(gyq),
                                                                       len(gxq))
            curve.append(float(np.sqrt(np.mean((Z[mm] - g[mm]) ** 2)) / sc['sg']))
        # ★모듈 플래그를 **복원 전에** 되읽는다 (⛔팔 정의를 되읽으면 항등식이다)
        si_read = bool(SU.SCALE_INVARIANT_L1)
    finally:
        SU.SCALE_INVARIANT_L1 = old_si
        SU.SequentialUpdater._adaptive_prune_grow = orig
    return dict(
        stratum='real', arm=arm, unit=record, est_seed=int(seed), win=G.WIN,
        depth=float(sc['depth']), n_obs=int(len(sc['z'])), npx=int(sc['npx']),
        obs_hash=_h(np.column_stack([sc['P'], sc['z']])), q_hash=_h(sc['Pq']),
        nowcast=curve[0], leadmean=float(np.nanmean(curve)), curve=curve,
        # ── 개입 되읽기 (⛔항등식 금지: 왼쪽은 추정기 객체, 오른쪽은 러너가 배열에서) ──
        n_gaussians=int(getattr(getattr(upd, 'model', None), 'n_gaussians', 0) or 0),
        K_expected=(None if atoms is None else G.pooled_K(atoms, arm)),
        max_n_readback=int(getattr(upd, 'max_n', -1)),
        atoms_md5=(None if atoms is None else _md5(atoms)),
        n_atoms_injected=(None if atoms is None else len(atoms['kx'])),
        propose_calls=_PROPOSE[0],
        si_l1_readback=si_read,
        apg_calls=calls[0], apg_dn=list(wrapped.dn),
        grow_thresh_readback=float(getattr(upd, 'grow_residual_threshold', -1)),
        real_gate=getattr(upd, '_auto_real_gate', None),
        init_method_readback=str(getattr(upd, 'init_method', '?')),
        varpro_fired=bool((res or {}).get('varpro')),
        varpro_steps=(res or {}).get('varpro_steps'),
        cuda_available=bool(torch.cuda.is_available()),
        Hs=float(sc['Hs']), sigma=float(sc['sg']), wall_s=time.time() - t0w)


# ── Phase C: 합성 대조 ───────────────────────────────────────────────────────
def run_syn(ep, arm, seed):
    from wavesplat.run import run_estimation
    import wavesplat.optimizer.sequential_update as SU
    k = G.arm_knobs(arm)
    p = os.path.join(ATOMS_SYN, f"{ep['episode']}__learned__s{seed}.json")
    atoms = json.load(open(p))['atoms']
    cfg = G.arm_config(arm, atoms, seed=seed, real=False)
    _install_propose_counter()
    _PROPOSE[0] = 0
    old_si = SU.SCALE_INVARIANT_L1
    SU.SCALE_INVARIANT_L1 = bool(k['si_l1'])
    # ★합성에도 같은 **가지 계수기**를 건다 — 합성에서는 성장이 실제로 발화하므로
    #   `G-14`(세 출처 항등)가 두 층 모두에서 성립해야 한다.
    calls = [0]
    orig = SU.SequentialUpdater._adaptive_prune_grow

    def wrapped(self, *a, **kw):
        n0 = self.model.n_gaussians
        rr = orig(self, *a, **kw)
        calls[0] += 1
        wrapped.dn.append(int(self.model.n_gaussians - n0))
        return rr

    wrapped.dn = []
    SU.SequentialUpdater._adaptive_prune_grow = wrapped
    t0 = time.time()
    si_read = None
    try:
        r = run_estimation(ep['path'], config=cfg, verbose=False)
        si_read = bool(SU.SCALE_INVARIANT_L1)     # ★복원 전 되읽기
    finally:
        SU.SCALE_INVARIANT_L1 = old_si
        SU.SequentialUpdater._adaptive_prune_grow = orig
    return dict(
        stratum='syn', arm=arm, unit=ep['episode'], est_seed=int(seed),
        bf=ep['bf'], ncomp=ep['ncomp'], gen_seed=ep['gen_seed'],
        avg_nRMSE=float(r['avg_nRMSE']), zero_nRMSE=r.get('zero_nRMSE'),
        n_gaussians=r.get('n_gaussians'), K_expected=G.pooled_K(atoms, arm),
        atoms_md5=_md5(atoms), n_atoms_injected=len(atoms['kx']),
        propose_calls=_PROPOSE[0], si_l1_readback=si_read,
        apg_calls=calls[0], apg_dn=list(wrapped.dn),
        auto_selected=r.get('auto_selected'), auto_n_selected=r.get('auto_n_selected'),
        auto_real_gate=r.get('auto_real_gate'), ensemble_k=r.get('ensemble_k'),
        varpro_fired=bool(r.get('varpro_fired')), varpro_steps=r.get('varpro_steps'),
        cuda_available=bool(torch.cuda.is_available()), wall_s=time.time() - t0)


# ── 작업 목록 ────────────────────────────────────────────────────────────────
def jobs(stratum):
    out = []
    if stratum == 'real':
        for a in G.LADDER + ['SHIP-K-LSC', 'K-MAX-LSC', 'K-8-NOGROW']:
            for rec in G.RECORDS:
                for s in G.EST_SEEDS:
                    out.append((a, rec, s))
    elif stratum == 'syn':
        eps = {e['episode']: e for e in G.syn_episodes()}
        for a in G.SYN_ARMS:
            if a == G.SYN_IMPORTED_BASE:
                continue                     # ★B-22 `LEARN` 동결 행을 수입한다(재실행 0)
            for e in sorted(eps):
                for s in G.EST_SEEDS:
                    out.append((a, e, s))
    elif stratum == 'pc':
        for rec in G.RECORDS:                # 캐시 양성대조 = 인라인 제안(시드 42)
            for s in G.INLINE_SEEDS:
                out.append(('SHIP-K-INL', rec, s))
    return out


def main():
    stratum = sys.argv[1] if len(sys.argv) > 1 else 'real'
    shard = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    nsh = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    assert not torch.cuda.is_available(), 'GPU 가 보인다 — 중단.'
    base_idx = {'real': 0, 'syn': 10, 'pc': 9}[stratum]
    out_p = os.path.join(HERE, f'results_m4_{base_idx + (shard if stratum != "pc" else 0)}.jsonl')
    done = set()
    if os.path.exists(out_p):
        for line in open(out_p):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if 'error' not in r:
                done.add((r['arm'], r['unit'], r['est_seed']))
    js = [j for i, j in enumerate(jobs(stratum)) if i % nsh == shard]
    eps = {e['episode']: e for e in G.syn_episodes()} if stratum == 'syn' else {}
    with open(out_p, 'a') as fh:
        for (arm, unit, s) in js:
            if (arm, unit, s) in done:
                continue
            try:
                row = (run_syn(eps[unit], arm, s) if stratum == 'syn'
                       else run_real(unit, arm, s))
                if stratum == 'pc':
                    row['pc'] = True
            except Exception as e:            # noqa: BLE001
                import traceback
                row = dict(stratum=stratum, arm=arm, unit=unit, est_seed=int(s),
                           error=repr(e), tb=traceback.format_exc()[-900:])
            fh.write(json.dumps(row) + '\n')
            fh.flush()
            tag = (f"{row.get('nowcast', row.get('avg_nRMSE')):.4f} "
                   f"N={row.get('n_gaussians')} K={row.get('K_expected')}"
                   if 'error' not in row else str(row['error'])[:70])
            print(f'{arm:12s} {unit:34s} s{s} {tag}', flush=True)
    print(f'-> {out_p}')


if __name__ == '__main__':
    main()
