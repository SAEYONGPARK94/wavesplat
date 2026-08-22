"""V1 감사 프로브 — 커밋 d673892 (auto_switch_thresh 8 -> 0).

추정 런 0회 규율: 에피소드를 돌리지 않는다. `learned_atoms` 캐시를 넣어 확산 제안을
건너뛰고, 라우터 판정 / 실측 게이트 / 앙상블 가동 여부 / 되돌리기 비트동일성만 잰다.

사용법:  python3 v1_probe.py <tag>       # tag = NEW | OLD
PYTHONPATH 로 트리를 고른다.
"""
import sys, os, json, math
import numpy as np
import torch

TAG = sys.argv[1]
OUT = os.path.dirname(os.path.abspath(__file__))

from wavesplat.data_loader import FrameData
from wavesplat.optimizer.sequential_update import SequentialUpdater, make_updater
from wavesplat.run import DEFAULT_CONFIG

NY, NX = 8, 8


def frames(with_gt: bool, n=64, nf=3):
    g = torch.Generator().manual_seed(0)
    out = []
    for i in range(nf):
        x = torch.rand(n, generator=g) * 100 + 10
        y = (torch.rand(n, generator=g) - 0.5) * 100
        z = torch.randn(n, generator=g) * 0.1
        gt = torch.ones(NY, NX) * 0.5 if with_gt else None
        out.append(FrameData(t=float(i), x_obs=x, y_obs=y, z_obs=z,
                             confidence=torch.ones(n),
                             incidence_angles=torch.zeros(n), gt_field=gt))
    return out


def atoms(n):
    """결정적 원자 캐시 — 시험 파일과 같은 방식(격자에서 뽑음)."""
    rng = np.random.RandomState(7)
    kx = np.linspace(0.05, 0.05 + 0.2 * (n - 1), n) if n > 0 else np.array([])
    ky = np.zeros(n)
    en = np.linspace(1.0, 0.5, n) if n > 0 else np.array([])
    return dict(kx=kx, ky=ky, energy=en)


def params(u):
    m = getattr(u, 'model', None)
    if m is None:
        return None
    return {k: getattr(m, k).detach().cpu().numpy().astype(np.float64).tolist()
            for k in ('kx', 'ky', 'w', 'phi')}


rec = {'tag': TAG,
       'DEFAULT_CONFIG_has_key': 'auto_switch_thresh' in DEFAULT_CONFIG,
       'DEFAULT_CONFIG_value': DEFAULT_CONFIG.get('auto_switch_thresh'),
       'init_method_default': DEFAULT_CONFIG.get('init_method'),
       'analytic_ensemble_k_default': DEFAULT_CONFIG.get('analytic_ensemble_k')}

# ── P1. 실측 게이트 (GT 없는 프레임) — **배포 생성자 make_updater 로** ────────────
import warnings
p1 = {}
for est in ('bank_mmse', 'v2_konly'):
    cfg = dict(DEFAULT_CONFIG, depth=50.0, seed=42, real_estimator=est)
    u = make_updater(cfg)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = u.initialize(frames(with_gt=False), x_range=(10.0, 110.0),
                         y_range=(-50.0, 50.0))
    p1[est] = {'ctor': type(u).__name__,
               'auto_real_gate': bool(getattr(u, '_auto_real_gate', None)),
               'init_method': getattr(u, 'init_method', None),
               'auto_selected': getattr(u, '_auto_selected', None),
               'auto_n_selected': getattr(u, '_auto_n_selected', None),
               'final_n': r.get('final_n'),
               'ensemble_k': r.get('ensemble_k')}
rec['P1_real_gate_via_make_updater'] = p1

# ── P2. 합성 라우팅 + 앙상블 가동 여부 (배포 기본값) ───────────────────────────
p2 = {}
for n in (0, 1, 2, 7, 8, 20, 30):
    cfg = dict(DEFAULT_CONFIG, depth=50.0, seed=42, learned_atoms=atoms(n))
    u = make_updater(cfg)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = u.initialize(frames(with_gt=True), x_range=(10.0, 110.0),
                         y_range=(-50.0, 50.0))
    p2[str(n)] = {'ctor': type(u).__name__,
                  'auto_real_gate': bool(getattr(u, '_auto_real_gate', None)),
                  'route': getattr(u, 'init_method', None),
                  'n_selected': getattr(u, '_auto_n_selected', None),
                  'final_n': r.get('final_n'),
                  'ensemble_k': r.get('ensemble_k'),
                  'member_seeds': getattr(u, 'member_seeds', None)}
rec['P2_synth_route_and_ensemble'] = p2

# ── P3. 되돌리기 비트동일성 ──────────────────────────────────────────────────
#   NEW 는 auto_switch_thresh=8 을 명시, OLD 는 키 없음(폴백 8).
p3 = {}
for n in (7, 20):
    cfg = dict(DEFAULT_CONFIG, depth=50.0, seed=42, learned_atoms=atoms(n))
    if TAG == 'NEW':
        cfg['auto_switch_thresh'] = 8
    else:
        cfg.pop('auto_switch_thresh', None)
    torch.manual_seed(123); np.random.seed(123)
    u = make_updater(cfg)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = u.initialize(frames(with_gt=True), x_range=(10.0, 110.0),
                         y_range=(-50.0, 50.0))
    fld = u.predict_field(torch.linspace(10., 110., 9), torch.linspace(-50., 50., 9), 2.0)
    fld = np.asarray(fld.detach().cpu().numpy() if hasattr(fld, "detach") else fld, dtype=np.float64)
    p3[str(n)] = {'route': getattr(u, 'init_method', None),
                  'final_n': r.get('final_n'),
                  'ensemble_k': r.get('ensemble_k'),
                  'params': params(u),
                  'field': fld.ravel().tolist()}
rec['P3_revert'] = p3

with open(os.path.join(OUT, f'v1_probe_{TAG}.json'), 'w') as f:
    json.dump(rec, f, indent=1)
print(f'[{TAG}] done')
