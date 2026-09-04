"""`T15-HOMEREG` 러너 — ⛔`consts_t15.py` → `gates_t15.py`/`analyze_t15.py` **다음에** 쓰였다.

행 스키마는 `analyze_t15.py` 독스트링이 정의한 것을 그대로 따른다(⛔러너가 스키마를 정하지 않는다).

★개입은 컨텍스트 매니저로 걸고 **반드시 되돌린다**(`G-KNOB`/`G-RESTORE`).
⛔`varpro.VP_LR_DEFAULT` **재대입은 안 먹는다** — 기본 인자가 `def` 시점에 바인딩된다
  ⟹ **함수 객체 자체**를 갈아끼운다(`T12` `REACHABILITY.md §5` 항목 5).
⛔`BN.KLIM` 재대입만으로도 안 먹는다 — `beamform` 은 모듈 전역 `KX_G/KY_G/OM_G` 를 읽는다
  ⟹ **`BN.kgrid()` 재호출이 필수**(`PLUMBING.md §4.3`).
⛔`ZERO` 는 추정기를 돌리지 않는다 — h5 에서 산술 계산한다(분석기가 `G-ZERO` 로 독립 재계산).
⛔메모리 캡 밖에서 돌리지 마라:
  `systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 -- python3 -u run_t15.py`
"""
import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')

import hashlib, json, re, sys, time                               # noqa: E402

import numpy as np                                                # noqa: E402
import torch                                                      # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import consts_t15 as C                                            # noqa: E402
sys.path.insert(0, C.ROOT)
sys.path.insert(0, os.path.join(C.ROOT, 'scripts'))
from wavesplat.optimizer import learned_init as LI                # noqa: E402
from wavesplat.optimizer import initializer as INIT               # noqa: E402
from wavesplat.optimizer import varpro as VP                      # noqa: E402
from wavesplat.model import spectral_gaussian as SG               # noqa: E402
from wavesplat.data_loader import EpisodeDataLoader               # noqa: E402
import wavesplat.run as WR                                        # noqa: E402  ⛔bare name 필수
from wavesplat.run import run_estimation, DEFAULT_CONFIG          # noqa: E402
import d3_beamform_net as BN                                      # noqa: E402  ⛔bare name 필수

OUT = os.path.join(HERE, 'results_t15.jsonl')
META = os.path.join(HERE, 'meta_t15.json')
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
NAME_RE = re.compile(r'(BF\d+)_(\d+)comp_seed(\d+)_')


def rng_md5():
    st = np.random.get_state()
    return hashlib.md5(np.asarray(st[1]).tobytes() + str(st[2]).encode()).hexdigest()


def read_knobs(cfg=None):
    """★**살아있는 모듈 상태**를 되읽는다 (`G-KNOB`/`G-RESTORE`).

    ⛔러너가 처치값을 그대로 적는 것이 아니다 — 개입이 조용히 실패하면 여기서 드러난다.
    ⛔`OM_G[0,0]` 이 아니라 **전 텐서 md5** 다(격자 모서리는 심해 포화로 눈이 먼다 — `T12` 사고).
    """
    k = dict(KLIM=float(BN.KLIM), KX_G00=float(BN.KX_G[0, 0].item()),
             OM_Gmd5=hashlib.md5(BN.OM_G.detach().cpu().numpy().tobytes()).hexdigest(),
             MAP_DEPTH=float(BN.DEPTH), MIN_DK=float(INIT.MIN_DK),
             vp_lr=float(getattr(VP.apply_varpro, '_t15_lr', VP.VP_LR_DEFAULT)),
             grow_krange=float(getattr(INIT._beamform_coarse, '_t15_kr', 4.0)))
    try:                        # ★K_LIM 은 (래핑됐을 수도 있는) 로더에서 되읽는다
        k['K_LIM'] = float(LI._load_cond(LI.DEFAULT_CKPT, DEV)[2]['K_LIM'])
    except Exception:                                             # noqa: BLE001
        k['K_LIM'] = None
    src = cfg if cfg is not None else DEFAULT_CONFIG
    k['max_steepness'] = float(src.get('max_steepness', 0.15))
    k['learned_dedup_dk'] = float(src.get('learned_dedup_dk', 0.03))
    return k


class Arm:
    """팔 개입을 걸고 **반드시 되돌린다**. ⛔되돌리기 실패 = 프로세스 전역 오염."""

    def __init__(self, name, dep):
        self.name = name
        self.spec = C.knobs_for(name, dep)      # ⛔리터럴 0 — consts 가 T12 에서 유도한다
        self.grow_calls = 0
        self.grow_atoms = 0
        self.n_raw = None
        self.n_sel = None
        self.n_gathered = None
        self.proposal_md5 = None
        self.device = None
        self.n_init = None
        self.init_t_last = None
        self.seed_applied = None
        self.rng_md5_seeded = None
        self.depth = None
        self.hs = None
        self.tp = None

    def __enter__(self):
        self._undo = []
        if 'KLIM' in self.spec:
            old = BN.KLIM
            BN.KLIM = float(self.spec['KLIM'])
            self._undo.append(lambda: setattr(BN, 'KLIM', old))
            og = (BN.KX_G, BN.KY_G, BN.OM_G)
            BN.KX_G, BN.KY_G, BN.OM_G = BN.kgrid()   # ★재대입만으로는 0회 발화한다

            def _rg(og=og):
                BN.KX_G, BN.KY_G, BN.OM_G = og
            self._undo.append(_rg)
        if 'K_LIM' in self.spec:
            orig, newk = LI._load_cond, float(self.spec['K_LIM'])

            def wrapped(ckpt, device, _o=orig, _k=newk):
                cd, model, stats, abar, betas, den = _o(ckpt, device)
                st = dict(stats); st['K_LIM'] = _k    # ⛔캐시 원본을 변형하지 않는다
                return (cd, model, st, abar, betas, den)
            LI._load_cond = wrapped
            self._undo.append(lambda: setattr(LI, '_load_cond', orig))
        if 'vp_lr' in self.spec:
            orig, lr = VP.apply_varpro, float(self.spec['vp_lr'])

            def wv(updater, frames, _o=orig, _lr=lr, **kw):
                kw['lr'] = _lr            # ⛔모듈 상수 재대입은 안 먹는다(기본 인자 바인딩)
                return _o(updater, frames, **kw)
            wv._t15_lr = lr
            VP.apply_varpro = wv
            self._undo.append(lambda: setattr(VP, 'apply_varpro', orig))

        # ── 계측 (개입 아님 — 되읽기 계수기) ──────────────────────────────
        og = SG.SpectralGaussianModel.add_gaussians

        def wg(slf, n_new, *a, _o=og, _s=self, **kw):
            _s.grow_calls += 1; _s.grow_atoms += int(n_new)
            return _o(slf, n_new, *a, **kw)
        SG.SpectralGaussianModel.add_gaussians = wg
        self._undo.append(lambda: setattr(SG.SpectralGaussianModel, 'add_gaussians', og))

        ops = LI.pool_and_select

        def wp(kx, ky, en, *a, _o=ops, _s=self, **kw):
            _s.n_raw = int(len(kx))
            out = _o(kx, ky, en, *a, **kw)
            skx, sky, sen = (np.asarray(out[0]), np.asarray(out[1]), np.asarray(out[2]))
            _s.n_sel = int(len(skx))
            arr = np.c_[skx.ravel(), sky.ravel(), sen.ravel()] if len(skx) \
                else np.zeros((0, 3))
            _s.proposal_md5 = hashlib.md5(np.round(arr, 9).tobytes()).hexdigest()
            return out
        LI.pool_and_select = wp
        self._undo.append(lambda: setattr(LI, 'pool_and_select', ops))

        olc = LI._load_cond      # ★래퍼 **뒤에** 걸어 실제로 쓰인 device 를 본다

        def wl(ckpt, device, _o=olc, _s=self):
            _s.device = str(device)
            return _o(ckpt, device)
        LI._load_cond = wl
        self._undo.append(lambda: setattr(LI, '_load_cond', olc))

        ogi = LI._gather_init_obs

        def wgi(frames, _o=ogi, _s=self):
            out = _o(frames)
            _s.n_gathered = None if out is None else int(len(out[0]))
            return out
        LI._gather_init_obs = wgi
        self._undo.append(lambda: setattr(LI, '_gather_init_obs', ogi))

        oss = WR.set_seed        # ★시드는 **적은 값**이 아니라 **적용된 값**을 기록한다
        #   ⛔러너 리터럴을 분석기 리터럴과 대조하면 **실패할 수 없는 게이트**가 된다
        #   (`check_prereg C6a`). `G-SEED` 는 여기서 잡은 지문을 분석기가 **독립 재계산**해 댄다.

        def wss(seed, _o=oss, _s=self):
            _o(seed)
            _s.seed_applied = int(seed)
            _s.rng_md5_seeded = rng_md5()
        WR.set_seed = wss
        self._undo.append(lambda: setattr(WR, 'set_seed', oss))

        oiw = EpisodeDataLoader.get_init_window

        def wiw(slf, tp_factor=2.0, _o=oiw, _s=self):
            fr = _o(slf, tp_factor=tp_factor)
            _s.n_init = len(fr)
            _s.init_t_last = float(fr[-1].t) if fr else None
            _s.depth = float(slf.depth); _s.hs = float(slf.Hs_true)
            _s.tp = float(slf.Tp_true)
            return fr
        EpisodeDataLoader.get_init_window = wiw
        self._undo.append(lambda: setattr(EpisodeDataLoader, 'get_init_window', oiw))
        return self

    def __exit__(self, *a):
        for f in reversed(self._undo):
            f()
        return False


def zero_row(path):
    """`ZERO` = `mean_{t>2·Tp} sqrt(mean(gt²))/Hs` — ⛔추정기 런 0회.

    정본 = `tmp/S16_BASELINE_2026-08-14/run_s16.py:240,266`.
    ⚠분석기 `G-ZERO` 가 **같은 값을 독립 경로로 다시 계산**해 대조한다.
    """
    ld = EpisodeDataLoader(path, device='cpu')
    ts = np.asarray(ld.timestamps, dtype=float)
    sel = ts > C.TP_FACTOR * ld.Tp_true
    gt = np.asarray(ld._gt_fields)[sel]
    per = np.sqrt((gt ** 2).reshape(gt.shape[0], -1).mean(axis=1)) / max(ld.Hs_true, 1e-6)
    tt = ts[sel]
    return dict(nrmse=float(per.mean()), n_frames_scored=int(sel.sum()),
                scored_t_first=float(tt[0]), scored_t_last=float(tt[-1]),
                episode_total_frames=int(ld.n_frames),
                n_init_frames=int((ts <= C.TP_FACTOR * ld.Tp_true).sum()),
                init_t_last=float(ts[ts <= C.TP_FACTOR * ld.Tp_true].max()),
                t_split=float(C.TP_FACTOR * ld.Tp_true),
                hs_true=float(ld.Hs_true), tp_true=float(ld.Tp_true),
                depth_readback=float(ld.depth))


def estimate(arm, path, seed):
    """⛔config 를 매번 새로 만든다 (제안 원자 캐시 누출 차단)."""
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(seed=seed, device='cpu')
    assert 'learned_atoms' not in cfg, '제안 캐시가 새 config 에 이미 있다'
    kb = read_knobs(cfg)
    r = run_estimation(path, cfg)
    assert 'learned_atoms' not in cfg, '호출자 config 가 변형됐다 (캐시 역류)'
    return r, kb, cfg


def main(blocks, layer, arms, out, meta_out, limit=None):
    torch.set_num_threads(1)
    md5_start = C.freeze_map()
    dep = C.deployed_knobs()
    ok, mine, theirs = C.treat_pin()
    assert ok, f'G-TREATPIN: f=2 {mine} != T12 A-MAPSCALE {theirs}'
    rows, seen_cache = [], []
    blocks = blocks[:(limit or None)]
    t_all = time.time()
    for i, (name, path, fi, lay) in enumerate(blocks, 1):
        t0 = time.time()
        m = NAME_RE.match(name)
        seed = C.SEED15 + fi
        z = zero_row(path)
        base = dict(layer=lay, block=name, fi=fi, bf=m.group(1), ncomp=int(m.group(2)),
                    ep_seed=int(m.group(3)), path_md5=C.file_md5(path), seed=seed,
                    hs_true=z['hs_true'], tp_true=z['tp_true'], t_split=z['t_split'],
                    episode_total_frames=z['episode_total_frames'],
                    cap_proposal=int(LI.M_OBS_CAP))
        got = {}
        for a in arms:
            r0 = rng_md5(); ta = time.time()
            if a == 'ZERO':
                kb = read_knobs()
                rb = dict(nrmse=z['nrmse'], n_frames_scored=z['n_frames_scored'],
                          scored_t_first=z['scored_t_first'],
                          scored_t_last=z['scored_t_last'],
                          n_init_frames=z['n_init_frames'], init_t_last=z['init_t_last'],
                          depth_readback=z['depth_readback'], has_gt=None,
                          auto_selected=None, auto_n_selected=None, auto_real_gate=None,
                          ensemble_k=None, n_gaussians=None, varpro_fired=None,
                          varpro_k_shift=None, init_iters=None, avg_rmse_near=None,
                          knobs_before=kb, proposal_md5=None, proposal_device=None,
                          n_gathered=None, n_raw=None, n_sel=None, n_grow_calls=None,
                          n_grow_atoms=None, seed_applied=None, rng_md5_seeded=None)
            else:
                with Arm(a, dep) as arm:
                    r, kb, cfg = estimate(arm, path, seed)
                    seen_cache.append('learned_atoms' in cfg)
                rb = dict(nrmse=float(r['avg_nRMSE']),
                          n_frames_scored=int(r['tracking_frames']),
                          scored_t_first=r['scored_t_first'],
                          scored_t_last=r['scored_t_last'],
                          n_init_frames=arm.n_init, init_t_last=arm.init_t_last,
                          depth_readback=arm.depth, has_gt=True,
                          auto_selected=r['auto_selected'],
                          auto_n_selected=r['auto_n_selected'],
                          auto_real_gate=r['auto_real_gate'],
                          ensemble_k=r['ensemble_k'], n_gaussians=r['n_gaussians'],
                          varpro_fired=r['varpro_fired'],
                          varpro_k_shift=r['varpro_k_shift'],
                          init_iters=r['init_iters'],
                          avg_rmse_near=(float(r['avg_RMSE_near'])
                                         if np.isfinite(r['avg_RMSE_near']) else None),
                          knobs_before=kb, proposal_md5=arm.proposal_md5,
                          proposal_device=arm.device, n_gathered=arm.n_gathered,
                          n_raw=arm.n_raw, n_sel=arm.n_sel,
                          n_grow_calls=arm.grow_calls, n_grow_atoms=arm.grow_atoms,
                          seed_applied=arm.seed_applied,
                          rng_md5_seeded=arm.rng_md5_seeded)
            rows.append(dict(base, arm=a, **rb, knobs_after=read_knobs(),
                             zero_nrmse=z['nrmse'], rng_md5_before=r0,
                             rng_md5_after=rng_md5(), wall_s=round(time.time() - ta, 2)))
            got[a] = rb['nrmse']
        eta = (time.time() - t_all) / i * (len(blocks) - i) / 60.0
        print(f"  [{i:3d}/{len(blocks)}] {lay[:3]} {name[:30]:30s} "
              + ' · '.join(f'{a} {got[a]:.4f}' for a in arms)
              + f'  ({time.time() - t0:.0f}s · 남은 {eta:.0f}분)', flush=True)
        with open(out, 'w', encoding='utf-8') as fh:
            for r_ in rows:
                fh.write(json.dumps(r_, ensure_ascii=False) + '\n')
    md5_end = C.freeze_map()
    json.dump(dict(layer=layer, n_runs=sum(1 for r in rows if r['arm'] != 'ZERO'),
                   # ⛔`threads` 와 `threads_readback` 에 **같은 식**을 적으면 하나를 둘로 센 것이다
                   #   (`check_prereg C6c`) ⟹ 앞은 **의도(환경변수)**, 뒤는 **되읽기(라이브)**.
                   threads_env=os.environ.get('OMP_NUM_THREADS'),
                   threads_readback=int(torch.get_num_threads()),
                   blas_env={v: os.environ.get(v) for v in
                             ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                              'MKL_NUM_THREADS')},
                   seed=C.SEED15, code_md5=md5_end, code_md5_start=md5_start,
                   md5_stable=(md5_start == md5_end),
                   cfg_had_learned_atoms=bool(any(seen_cache)),
                   cuda_available=bool(torch.cuda.is_available()),
                   blocks=len(blocks), arms=list(arms)),
              open(meta_out, 'w'), ensure_ascii=False, indent=1)
    print(f'✅{len(rows)} 행 · md5 안정 {md5_start == md5_end} · '
          f'캐시역류 {bool(any(seen_cache))}')
    return rows


def plan():
    """(name, path, fi, layer) — `fi` 는 **정렬된 128 목록의 1-based 위치**(시드 규약)."""
    corp = C.corpus()
    order = [(n, p, L) for L in ('crowded', 'sparse') for n, p in corp[L]]
    sub = {L: {n for n, _ in v} for L, v in C.sub_layer().items()}
    full = [(n, p, i, L) for i, (n, p, L) in enumerate(order, 1)]
    subs = [t for t in full if t[0] in sub[t[3]]]
    return full, subs


if __name__ == '__main__':
    a = sys.argv[1:]
    mode = a[0] if a else 'full'
    lim = int(a[1]) if len(a) > 1 else None
    full, subs = plan()
    if mode == 'smoke':
        # ★스모크는 **가장 비싼 팔과 게이트가 발화할 셀**을 포함해야 한다(상시지시).
        #   ⟹ 혼잡·희소 각 1블록 × **전 9팔**.
        sm = [full[0], [t for t in full if t[3] == 'sparse'][0]]
        print(f'⚠SMOKE — 블록 {[t[0] for t in sm]} × 전 {len(("ZERO",) + C.EST_ARMS)}팔')
        main(sm, 'SMOKE', ('ZERO',) + C.EST_ARMS,
             OUT.replace('.jsonl', '_SMOKE.jsonl'),
             META.replace('.json', '_SMOKE.json'), lim)
    elif mode == 'sub':
        main(subs, 'SUB', C.SUB_ARMS, OUT.replace('.jsonl', '_SUB.jsonl'),
             META.replace('.json', '_SUB.json'), lim)
    else:
        main(full, 'FULL', C.FULL_ARMS, OUT.replace('.jsonl', '_FULL.jsonl'),
             META.replace('.json', '_FULL.json'), lim)
