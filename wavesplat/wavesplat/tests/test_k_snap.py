"""표현 절제 노브 `k_snap_dk` 회귀 시험 (2026-08-24, `B-16`).

★이 노브가 있는 이유: 저장소의 헤드라인은 *"격자 비의존 표현"* 을 기여로 말하는데
**표현만 절제한 팔이 저장소에 0건**이었다(2026-08-24 방법론 심사 A2 지적).

여기서 고정하는 것 둘:
  1. **기본값은 no-op 이다** — `None` 이고, 모델을 건드리지 않는다(배포 거동 불변).
  2. **켜면 실제로 먹는다** — 파수가 격자 위에 놓이고 개입 계수기가 그것을 기록한다
     (*"설정을 넣었다 ≠ 설정이 먹었다"* — W7 에서 대가를 치른 부류).
"""
import numpy as np
import torch

from wavesplat.model.spectral_gaussian import SpectralGaussianModel
from wavesplat.optimizer.sequential_update import (K_SNAP_DK_DEFAULT,
                                                   SequentialUpdater)
from wavesplat.run import DEFAULT_CONFIG

DEPTH = 50.0


def _frames(n_frames=4, n_obs=120, seed=20260824):
    from wavesplat.optimizer.sequential_update import FrameData
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_frames):
        x = rng.uniform(0.0, 150.0, n_obs)
        y = rng.uniform(-60.0, 60.0, n_obs)
        z = rng.normal(0.0, 0.5, n_obs)
        out.append(FrameData(t=float(i) * 0.1,
                             x_obs=torch.as_tensor(x, dtype=torch.float32),
                             y_obs=torch.as_tensor(y, dtype=torch.float32),
                             z_obs=torch.as_tensor(z, dtype=torch.float32),
                             confidence=torch.ones(n_obs, dtype=torch.float32),
                             incidence_angles=torch.zeros(n_obs, dtype=torch.float32)))
    return out


def _updater(**cfg):
    u = SequentialUpdater({'depth': DEPTH, **cfg})
    m = SpectralGaussianModel(n_gaussians=5, depth=DEPTH)
    with torch.no_grad():
        m.kx.copy_(torch.tensor([0.031, 0.0817, -0.0455, 0.12, 0.0], dtype=m.kx.dtype))
        m.ky.copy_(torch.tensor([0.017, -0.0233, 0.0611, 0.0, 0.09], dtype=m.ky.dtype))
        m.w.copy_(torch.tensor([1.0, 0.7, 0.5, 0.3, 0.2], dtype=m.w.dtype))
        m.phi.copy_(torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=m.phi.dtype))
    u.model = m
    u._build_tracking_optimizer()
    return u


def test_default_is_none_and_mirrored():
    """단일 출처 + 미러. ⛔둘이 갈라지면 배포와 기록이 어긋난다."""
    assert K_SNAP_DK_DEFAULT is None
    assert DEFAULT_CONFIG['k_snap_dk'] is None, (
        'run.DEFAULT_CONFIG 의 미러가 단일 출처와 다르다')


def test_default_is_a_true_noop():
    """기본값에서 `_snap_k_to_grid` 는 **모델을 건드리지 않는다**(비트 동일)."""
    u = _updater()
    before = (u.model.kx.detach().clone(), u.model.ky.detach().clone(),
              u.model.w.detach().clone(), u.model.phi.detach().clone())
    rec = u._snap_k_to_grid(_frames(), verbose=False)
    assert rec is None, '기본값인데 개입 기록이 생겼다'
    for a, b in zip(before, (u.model.kx, u.model.ky, u.model.w, u.model.phi)):
        assert torch.equal(a, b.detach()), '기본값에서 파라미터가 바뀌었다'


def test_zero_is_also_a_noop():
    """`0` 도 끄는 값이다 — 0 으로 나누지 않는다."""
    u = _updater(k_snap_dk=0)
    assert u._snap_k_to_grid(_frames(), verbose=False) is None


def test_snap_actually_lands_on_the_grid():
    """켜면 **파수가 격자 위에 놓인다**. 그리고 계수기가 그것을 기록한다."""
    dk = 0.02
    u = _updater(k_snap_dk=dk)
    rec = u._snap_k_to_grid(_frames(), verbose=False)
    assert rec is not None and rec['fired'] is True
    kx = u.model.kx.detach().cpu().numpy()
    ky = u.model.ky.detach().cpu().numpy()
    for arr in (kx, ky):
        resid = np.abs(arr / dk - np.round(arr / dk))
        assert np.all(resid < 1e-6), f'격자 위가 아니다: {arr}'
    assert rec['shift_max'] <= dk * np.sqrt(2) / 2 + 1e-9, (
        '스냅 이동이 격자 반칸을 넘는다 — 반올림이 아니다')
    assert rec['n_atoms'] == 5 and 1 <= rec['n_unique_cells'] <= 5


def test_snap_refits_amplitude_and_phase():
    """스냅 후 (w, φ) 가 **다시 적합된다** — 파수만 옮기고 진폭을 두면 팔이 부당하게 약해진다."""
    u = _updater(k_snap_dk=0.02)
    w0 = u.model.w.detach().clone()
    u._snap_k_to_grid(_frames(), verbose=False)
    assert not torch.equal(w0, u.model.w.detach()), '(w, φ) 재적합이 일어나지 않았다'
