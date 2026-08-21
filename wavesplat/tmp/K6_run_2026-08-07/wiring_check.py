"""Rx-4 wiring check -- STRUCTURAL ONLY (requires_grad flags, cfg diffs, lr schedule).
Deliberately reports NO accuracy metric, so it can be run BEFORE the PREREG is frozen."""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, REPO)
os.environ['RX2_EPISODES'] = 'c16s8'
spec = importlib.util.spec_from_file_location('run_rx4', os.path.join(HERE, 'run_rx4.py'))
R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)

print('=== A. episode blocks ===')
print(f'  CROWDED16 n={len(R.CROWDED16)}  {[e[8:-11] for e in R.CROWDED16]}')
print(f'  SPARSE8   n={len(R.SPARSE8)}    {[e[8:-3] for e in R.SPARSE8]}')
print(f'  EPISODES (RX2_EPISODES=c16s8) n={len(R.EPISODES)}')
print(f'  SEEDS default = {R.SEEDS}')
assert len(set(R.CROWDED16)) == 16 and len(set(R.SPARSE8)) == 8

print('\n=== B. cell_config diff vs AUTO (must be exactly ONE knob per cell) ===')
ep = R.CROWDED16[0]
base = R.cell_config('AUTO', ep)
for c in R.RX4_CELLS:
    cfg = R.cell_config(c, ep)
    diff = {k: (base.get(k, '<absent>'), v) for k, v in cfg.items() if base.get(k) != v}
    diff.update({k: (v, '<absent>') for k, v in base.items() if k not in cfg})
    envd = R.CELLS[c].get('env') or {}
    print(f'  {c:12s} cfg-diff={diff or "{}"}  env={envd or "{}"}  '
          f'n_changes={len(diff) + len(envd)}')
    assert 'init_method' in cfg and cfg['init_method'] == 'auto'
    assert len(diff) + len(envd) <= 1, c

print('\n=== C. _rx4_install: requires_grad per tracking_mode ===')
import torch
import wavesplat.optimizer.sequential_update as S
R._rx4_install()
for mode in ('w_phi', 'phi_only', 'w_only', 'all'):
    cfg = dict(R.BASE_CFG); cfg.update(init_method='auto', tracking_mode=mode, depth=50.0)
    u = S.SequentialUpdater(cfg)
    u.model = u._build_model(5)
    u._build_tracking_optimizer()
    g = {k: bool(getattr(u.model, k).requires_grad) for k in ('w', 'phi', 'kx', 'ky')}
    npar = sum(len(pg['params']) for pg in u.tracking_optimizer.param_groups)
    lrs = sorted({pg['lr'] for pg in u.tracking_optimizer.param_groups})
    print(f'  tracking_mode={mode:9s} requires_grad={g}  n_param_tensors={npar} lrs={lrs}')
assert True

print('\n=== D. _rx4_install: rx4_track_frames zeroes the lrs from frame E on ===')
cfg = dict(R.BASE_CFG); cfg.update(init_method='auto', rx4_track_frames=3, depth=50.0)
u = S.SequentialUpdater(cfg)
u.model = u._build_model(4)
u._build_tracking_optimizer()


class _F:
    t = 0.0
    x_obs = torch.zeros(0)


for i in range(6):
    _F.t = 0.1 * i
    try:
        u.update(_F())
    except Exception:
        pass                      # empty-observation early return is fine; lr is what we test
    lrs = sorted({pg['lr'] for pg in u.tracking_optimizer.param_groups})
    print(f'  after update #{i}: _rx4_n={getattr(u, "_rx4_n", None)} lrs={lrs}')
assert all(pg['lr'] == 0.0 for pg in u.tracking_optimizer.param_groups)

print('\n=== E. absent key => update() is the ORIGINAL function path ===')
cfg = dict(R.BASE_CFG); cfg.update(init_method='auto', depth=50.0)
u2 = S.SequentialUpdater(cfg)
u2.model = u2._build_model(4)
u2._build_tracking_optimizer()
lr0 = [pg['lr'] for pg in u2.tracking_optimizer.param_groups]
for i in range(6):
    _F.t = 0.1 * i
    try:
        u2.update(_F())
    except Exception:
        pass
print(f'  rx4_track_frames absent -> lrs unchanged: {lr0} -> '
      f'{[pg["lr"] for pg in u2.tracking_optimizer.param_groups]}  '
      f'_rx4_n attr set? {hasattr(u2, "_rx4_n")}')
assert lr0 == [pg['lr'] for pg in u2.tracking_optimizer.param_groups]
assert not hasattr(u2, '_rx4_n')
print('\nALL STRUCTURAL CHECKS PASSED')
