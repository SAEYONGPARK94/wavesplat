"""`T13-DEPTHFIX` 러너 — ⛔`gates_t13.py`·`analyze_t13.py` **다음에** 쓰였다.

★관측·프레임·오라클·재척도는 `T11 run_t11.py` 를 임포트해 **비트 재사용**한다(앵커 전제).
★처치 = `MapDepth(upd.depth)` — ⛔`DEPTH` 재대입만으로는 **0회 발화**한다
  (`KX_G, KY_G, OM_G = kgrid()` 가 모듈 로드 시 1회 고정, `REACHABILITY §2` 실측).
  반드시 `kgrid()` 재호출. ⛔`scripts.d3_beamform_net` 경로 임포트는 **다른 모듈 객체**라 또 0회.
"""
import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')

import hashlib, json, sys, time, zipfile                          # noqa: E402

import numpy as np                                                # noqa: E402
import torch                                                      # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gates_t13 as G                                             # noqa: E402
sys.path.insert(0, G.T11)
import run_t11 as R11                                             # noqa: E402  ★T11 재사용
import d3_beamform_net as BN                                      # noqa: E402  ⛔bare name
import d3_beamform_cond_diffusion as CD                           # noqa: E402
# ⛔⛔**cwd 를 되돌리지 마라.** `d3_beamform_cond_diffusion.py:25` 의 `os.chdir(repo_root)` 는
#   부작용이 아니라 **의존성**이다 — `:40 PRIOR_CKPT = 'results/direction3/setdiff_prior.pt'`
#   가 **상대 경로**라 cwd 로 해소된다. 되돌리면 `torch.load` 가 FileNotFoundError 를 내고
#   `sequential_update.py:1111` 의 `except (FileNotFoundError, OSError, ImportError)` 가
#   **조용히 `v2_konly` 로 폴백**한다 ⟹ 배치 전체가 **다른 팔**로 돈다.
#   ★스모크가 잡았다(AMENDMENT_1 §1): 경고는 났지만 치명적이지 않았고, 앵커가 T11 과
#   어긋나고서야(SHIP 0.2218 vs 0.2282) 드러났다. T11 러너는 cwd 를 건드리지 않는다.
from wavesplat.optimizer import learned_init as LI                # noqa: E402
from wavesplat.optimizer.sequential_update import make_updater    # noqa: E402
from wavesplat.run import DEFAULT_CONFIG                          # noqa: E402

OUT = os.path.join(HERE, 'results_t13.jsonl'); META = os.path.join(HERE, 'meta_t13.json')
IMPORT_FILES = ('wavesplat/optimizer/sequential_update.py', 'wavesplat/optimizer/learned_init.py',
                'wavesplat/optimizer/varpro.py', 'wavesplat/model/spectral_gaussian.py',
                'wavesplat/model/dispersion.py', 'wavesplat/run.py',
                'scripts/d3_beamform_net.py', 'scripts/d3_beamform_cond_diffusion.py')
CKPT = 'results/direction3/beamform_cond_diffusion_mixed_v1.pt'


def _md5(p):
    return hashlib.md5(open(p, 'rb').read()).hexdigest()


def freeze_map():
    out = {}
    for f in ('gates_t13.py', 'analyze_t13.py', 'run_t13.py', 'PREREG.md', 'REACHABILITY.md'):
        p = os.path.join(HERE, f)
        if os.path.exists(p):
            out[f] = _md5(p)
    for f in IMPORT_FILES:
        p = os.path.join(G.ROOT, f)
        if os.path.exists(p):
            out[f] = _md5(p)
    for f in ('sample_t11.json', 'results_t11.jsonl', 'gates_t11.py', 'run_t11.py'):
        p = os.path.join(G.T11, f)
        if os.path.exists(p):
            out['T11/' + f] = _md5(p)
    h = hashlib.sha256()
    with open(os.path.join(G.ROOT, CKPT), 'rb') as fh:
        for c in iter(lambda: fh.read(1 << 20), b''):
            h.update(c)
    out['sha256:' + CKPT] = h.hexdigest()
    return out


class MapDepth:
    """★처치 — 조건화 맵의 수심만 바꾼다. ⛔`DEPTH` 재대입만으로는 안 먹는다."""

    def __init__(self, depth):
        self.depth = None if depth is None else float(depth)

    def __enter__(self):
        if self.depth is None:
            return self
        self._old = (BN.DEPTH, BN.KX_G, BN.KY_G, BN.OM_G)
        BN.DEPTH = self.depth
        BN.KX_G, BN.KY_G, BN.OM_G = BN.kgrid()      # ★이 재호출이 없으면 0회 발화
        return self

    def __exit__(self, *a):
        if self.depth is not None:
            BN.DEPTH, BN.KX_G, BN.KY_G, BN.OM_G = self._old
        return False


class Probe:
    """★`G-MAP`/`G-BMAPCALLS` 의 되읽기 — **소비자 `_bmap_from_obs` 가 호출 시점에 본 전역**.

    ⛔러너의 변수를 적는 것이 아니다. 모듈 정체가 어긋나 래퍼가 안 걸리면 `None` 이 남아 FAIL 한다.
    """

    def __enter__(self):
        self.calls = 0; self.om_md5 = None; self.om_row = None; self.depth_seen = None
        self.device = None; self.n_gathered = None
        ob = CD._bmap_from_obs

        def wb(P, z, _o=ob, _s=self):
            _s.calls += 1
            om = BN.OM_G.detach().cpu().numpy()
            _s.om_md5 = hashlib.md5(om.tobytes()).hexdigest()
            _s.om_row = [float(x) for x in om[BN.NK // 2, ::4]]
            _s.depth_seen = float(BN.DEPTH)
            return _o(P, z)
        CD._bmap_from_obs = wb
        ol = LI._load_cond

        def wl(ckpt, device, _o=ol, _s=self):
            _s.device = str(device)
            return _o(ckpt, device)
        LI._load_cond = wl
        og = LI._gather_init_obs

        def wg(frames, _o=og, _s=self):
            out = _o(frames)
            _s.n_gathered = 0 if out is None else int(len(out[0]))
            return out
        LI._gather_init_obs = wg
        self._undo = [lambda: setattr(CD, '_bmap_from_obs', ob),
                      lambda: setattr(LI, '_load_cond', ol),
                      lambda: setattr(LI, '_gather_init_obs', og)]
        return self

    def __exit__(self, *a):
        for f in self._undo:
            f()
        return False


def estimate(x, y, tt, z, gx, gy, tq, target, s4, seed, fix):
    """⛔config 를 매번 새로 만든다. `FIX` 와 `SHIP` 의 차이는 **`OM_G` 한 텐서**뿐이다."""
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(depth=G.DEPTH, seed=seed, device='cpu', init_method='auto')
    assert 'learned_atoms' not in cfg, '제안 캐시가 새 config 에 이미 있다'
    upd = make_updater(cfg)
    dcfg = float(getattr(upd, 'depth', cfg['depth']))
    fr = R11.build_frames(x, y, tt, z, G.GT_CONST)
    rb = R11.readback_frames(fr)
    with Probe() as pr:
        with MapDepth(dcfg if fix else None):
            info = upd.initialize(fr, x_range=(float(gx.min()), float(gx.max())),
                                  y_range=(float(gy.min()), float(gy.max())), verbose=False)
    gt_after = float(torch.cat([f.gt_field.reshape(-1) for f in fr]).std(unbiased=False))
    pred = upd.predict_field(torch.tensor(gx, dtype=torch.float32),
                             torch.tensor(gy, dtype=torch.float32), float(tq))
    pred = np.asarray(pred.detach().cpu()).reshape(target.shape)
    la = cfg.get('learned_atoms'); pmd5 = None
    if la:
        arr = np.c_[np.asarray(la['kx'], float).ravel(), np.asarray(la['ky'], float).ravel(),
                    np.asarray(la['energy'], float).ravel()]
        pmd5 = hashlib.md5(np.round(arr, 9).tobytes()).hexdigest()
    vp = (info or {}).get('varpro') or {}
    m = getattr(upd, 'model', None)
    rb.update(map_om_md5=pr.om_md5, map_om_row=pr.om_row, map_depth_seen=pr.depth_seen,
              bmap_calls=pr.calls, depth_config=dcfg, proposal_md5=pmd5,
              proposal_device=pr.device, n_gathered=pr.n_gathered,
              cap_proposal=int(LI.M_OBS_CAP), gt_std_after=gt_after,
              vp_clamp_hits=vp.get('vp_clamp_hits'), prov=upd.arm_provenance(),
              n_atoms_installed=int(getattr(m, 'n_gaussians', 0) or 0))
    return R11.nrmse(pred, target, s4), rb


def main(mode='primary'):
    torch.set_num_threads(1)
    md5_start = freeze_map()
    t11b, fresh = G.layers()
    smoke = mode == 'smoke'
    files = list(G.SMOKE_FILES) if smoke else G.sample()
    out = OUT.replace('.jsonl', '_SMOKE.jsonl') if smoke else OUT
    meta_o = META.replace('.json', '_SMOKE.json') if smoke else META
    if smoke:
        print(f'⚠SMOKE — 노출된 감사 파일 {len(files)}개 · 프리즌 표본 52 미접촉')
    rows, repeat = [], []
    with zipfile.ZipFile(G.ZIP) as zf:
        for fi, name in enumerate(files, 1):
            t0 = time.time()
            E, gx, gy, tarr = R11.read_mat(zf, name)
            nf = int(G.WIN_S)
            Ew, tw = E[G.J0:G.J0 + nf], tarr[G.J0:G.J0 + nf]
            target = Ew[-1]; s4 = 4.0 * float(np.std(Ew))
            ang, hs = R11.parse(name)
            seed = (G.SEED + fi) if smoke else G.seed_of(name)
            x, y, tt, zz = R11.sample_obs(Ew, gx, gy, tw, G.DENSITY,
                                          np.random.default_rng(seed))
            tq = float(tw[-1] - tw[0]); s = G.RESCALE_S
            base = dict(block=name, angle=ang, hs=hs, sigma4=s4, seed=seed, n_frames=nf,
                        layer=('SMOKE' if smoke else
                               ('T11-24' if name in t11b else 'FRESH-28')),
                        dt=float(np.median(np.diff(tarr))), tq=tq,
                        e_shape=[int(v) for v in E.shape])
            none_rb = dict(t_origin=float(tt.min()), gt_std=None, gt_value=None,
                           gt_std_after=None, map_om_md5=None, map_om_row=None,
                           map_depth_seen=None, bmap_calls=None, depth_config=None,
                           proposal_md5=None, proposal_device=None, n_gathered=None,
                           cap_proposal=None, vp_clamp_hits=None, prov={},
                           n_atoms_installed=0, rescale_s=None, x_scale=None,
                           t_scale=None, disp_relerr=None)
            arms = ['ZERO', 'ORACLE', 'SHIP', 'FIX']
            if smoke or name in t11b:
                arms += ['RESCALE', 'RESCALE-FIX']
            got = {}
            for a in arms:
                ta = time.time()
                if a == 'ZERO':
                    val = R11.nrmse(np.zeros_like(target), target, s4)
                    rb = dict(none_rb, n_offered=int(len(zz)))
                elif a == 'ORACLE':
                    val = R11.oracle(x, y, tt, zz, gx, gy, tq, target, s4, G.N_MODE)
                    rb = dict(none_rb, n_offered=int(len(zz)))
                else:
                    resc = a.startswith('RESCALE')
                    fix = a in G.FIX_ARMS
                    ax, ay, at = (x / s, y / s, tt / np.sqrt(s)) if resc else (x, y, tt)
                    agx, agy = (gx / s, gy / s) if resc else (gx, gy)
                    atq = tq / np.sqrt(s) if resc else tq
                    val, rb = estimate(ax, ay, at, zz, agx, agy, atq, target, s4, seed, fix)
                    a_, b_, rel = R11.measured_scales(ax, x, at, tt)
                    rb.update(rescale_s=(s if resc else None), x_scale=a_, t_scale=b_,
                              disp_relerr=rel)
                rows.append(dict(base, arm=a, nrmse=val,
                                 wall_s=round(time.time() - ta, 1), **rb))
                got[a] = val
            if smoke:                       # ★G-REPEAT — 같은 파일 SHIP 2회 비트 동일
                v2, rb2 = estimate(x, y, tt, zz, gx, gy, tq, target, s4, seed, False)
                sr = [r for r in rows if r['block'] == name and r['arm'] == 'SHIP'][0]
                repeat.append(dict(block=name, bit=(v2 == sr['nrmse']
                                                    and rb2['proposal_md5'] == sr['proposal_md5'])))
            print(f"  [{fi:2d}/{len(files)}] {os.path.basename(name):18s} "
                  + ' · '.join(f'{k[:8]} {v:.4f}' for k, v in got.items())
                  + f' ({time.time()-t0:.0f}s)', flush=True)
            with open(out, 'w', encoding='utf-8') as fh:
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + '\n')
    md5_end = freeze_map()
    json.dump(dict(mode=mode, n_runs=sum(1 for r in rows
                                         if r['arm'] not in ('ZERO', 'ORACLE')),
                   threads=int(torch.get_num_threads()),
                   threads_readback=int(torch.get_num_threads()),
                   blas_env={v: os.environ.get(v) for v in
                             ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
                   seed_base=G.SEED, code_md5=md5_end, code_md5_start=md5_start,
                   code_md5_now=md5_end, md5_stable=(md5_start == md5_end),
                   g_repeat=repeat, cuda_available=bool(torch.cuda.is_available()),
                   files=len(files)),
              open(meta_o, 'w'), ensure_ascii=False, indent=1)
    if repeat:
        print(f'★G-REPEAT (GPU 반복 재현성): {repeat}')
    print(f'✅{len(rows)} 행 · md5 안정 {md5_start == md5_end}')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'primary')
