"""Deployment tests for the authorised `auto_v1_m3` configuration (2026-07-27).

The configuration authorised by the second confirmation round on the fresh reserve-seed
40 set (tmp/confirm2_2026-07-27/RESULT_CONFIRM2.md Sec 1 and 4) is:

    beamform_cond_diffusion_mixed_v1.pt  +  auto router (th=8)  +  guard RETIRED
    +  analytic fallback = median-of-3 pointwise-median FIELD ensemble

with one residual condition that is NOT released: on real (measured) input the router
must stay inactive.  These tests pin all four properties, plus the two invariants that
protect published numbers:

  * an explicit ``init_method='v2_konly'`` never reaches the ensemble wrapper;
  * a k=1 ensemble is bit-identical to a single ``v2_konly`` run.

None of these tests needs the diffusion checkpoint on the GPU: the router is driven
through its ``learned_atoms`` cache, and the real-domain gate short-circuits before the
proposal runs.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import glob
import warnings

import numpy as np
import pytest
import torch

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')


def _find_episode():
    episodes = sorted(glob.glob(os.path.join(DATA_DIR, '*.h5')))
    if not episodes:
        pytest.skip("No HDF5 episodes available")
    return episodes[0]


def _init_frames(n=10):
    from wavesplat.data_loader import EpisodeDataLoader
    loader = EpisodeDataLoader(_find_episode())
    frames = loader.get_init_window(tp_factor=2.0)
    if len(frames) < 5:
        pytest.skip("Not enough init frames")
    return loader, frames[:n]


def _strip_gt(frames):
    """Same shape as a real-data front-end: observations, no ground-truth field."""
    from wavesplat.data_loader import FrameData
    return [FrameData(t=f.t, x_obs=f.x_obs, y_obs=f.y_obs, z_obs=f.z_obs,
                      confidence=f.confidence, incidence_angles=f.incidence_angles,
                      gt_field=None) for f in frames]


def _atoms(n, spacing=0.2):
    """n well-separated atoms of equal energy -> pool_and_select keeps exactly n."""
    kx = np.arange(n, dtype=float) * spacing + 0.1
    return dict(kx=kx, ky=np.zeros(n), energy=np.ones(n))


# ── A. checkpoint provenance ────────────────────────────────────────────────

def test_default_checkpoint_is_mixed_v1_with_recorded_sha256():
    """The shipped default is mixed_v1, addressed absolutely, and matches its hash."""
    from wavesplat.optimizer import learned_init as li

    assert os.path.basename(li.DEFAULT_CKPT) == 'beamform_cond_diffusion_mixed_v1.pt'
    assert os.path.isabs(li.DEFAULT_CKPT), "cwd-relative default silently degrades to v2"
    assert os.path.exists(li.DEFAULT_CKPT)
    assert li.ckpt_sha256() == li.DEFAULT_CKPT_SHA256, (
        "deployed checkpoint does not match the recorded provenance hash")


def test_router_uses_the_default_checkpoint():
    """_auto_select_init must pick up DEFAULT_CKPT, not some other path."""
    import inspect
    from wavesplat.optimizer.sequential_update import SequentialUpdater
    src = inspect.getsource(SequentialUpdater._auto_select_init)
    assert "cfg.get('learned_ckpt', DEFAULT_CKPT)" in src


# ── B. router: 경로 통일(2026-08-22), guard retired ──────────────────────────
# ⚠**계약 변경**: 2026-07-26 배포 인가는 임계 8 이었고 이 시험이 그것을 고정했다.
# 2026-08-22 에 사용자 지시로 **합성 경로를 학습으로 통일**하면서 임계를 0 으로 내렸다.
# 근거(같은 B 코퍼스 128ep, 강제 두 팔, median-of-3, 신선 시드):
#   학습 90/128 · p=4.9e-06 · 상대 -38.5% · 제로필드 이김 **128/128 vs 100/128** · 3.4배 빠름
#   (혼잡 63/64 -51.0%; 해석적은 혼잡에서 제로필드에 25/64 패배)
#   라우터는 시드에 불안정했다 — 128 중 34(26.6%)에서 시드만 바꿔도 판정이 뒤집혔다.
#   tmp/{W11_UNIFY,W13_SPARSE_HEAD2HEAD,W14_CROWDED_HEAD2HEAD}_2026-08-22
# ★임계를 상수에서 읽는다 — 이전처럼 리터럴 8 을 박아 두면 배포와 시험이 다시 어긋난다.

def test_auto_router_follows_the_shipped_threshold():
    from wavesplat.optimizer.sequential_update import (SequentialUpdater,
                                                       AUTO_SWITCH_THRESH_DEFAULT)
    for n_atoms in (1, 7, 8, 20):
        cfg = {'init_method': 'auto', 'depth': 50.0, 'seed': 42,
               'learned_atoms': _atoms(n_atoms)}
        upd = SequentialUpdater(cfg)
        want = 'learned' if n_atoms >= AUTO_SWITCH_THRESH_DEFAULT else 'v2_konly'
        assert upd._auto_select_init([], verbose=False) == want
        assert upd._auto_n_selected == n_atoms


def test_unified_router_always_picks_learned_on_synthetic():
    """★통일의 정의: 임계 0 이면 성분 수와 무관하게 항상 학습."""
    from wavesplat.optimizer.sequential_update import (SequentialUpdater,
                                                       AUTO_SWITCH_THRESH_DEFAULT)
    assert AUTO_SWITCH_THRESH_DEFAULT == 0
    for n_atoms in (1, 2, 7, 30):
        upd = SequentialUpdater({'init_method': 'auto', 'depth': 50.0, 'seed': 42,
                                 'learned_atoms': _atoms(n_atoms)})
        assert upd._auto_select_init([], verbose=False) == 'learned'


def test_pre_unify_threshold_still_reproducible():
    """되돌리기·과거 배치 재현: 임계 8 을 명시하면 옛 거동이 그대로 나와야 한다."""
    from wavesplat.optimizer.sequential_update import (SequentialUpdater,
                                                       AUTO_SWITCH_THRESH_PRE_UNIFY)
    for n_atoms, want in ((7, 'v2_konly'), (8, 'learned'), (20, 'learned')):
        upd = SequentialUpdater({'init_method': 'auto', 'depth': 50.0, 'seed': 42,
                                 'auto_switch_thresh': AUTO_SWITCH_THRESH_PRE_UNIFY,
                                 'learned_atoms': _atoms(n_atoms)})
        assert upd._auto_select_init([], verbose=False) == want


def test_auto_guard_is_retired_by_default():
    """Routed 'learned' must NOT be vetoed, and bf_peak_ratio must not even be computed."""
    from wavesplat.optimizer.sequential_update import SequentialUpdater

    cfg = {'init_method': 'auto', 'depth': 50.0, 'seed': 42,
           'learned_atoms': _atoms(12)}
    upd = SequentialUpdater(cfg)
    assert upd._auto_select_init([], verbose=False) == 'learned'
    assert upd._auto_guard_fired is False
    assert upd._auto_bf_peak_ratio is None, (
        "guard block ran: beamform_peak_ratio was evaluated although the guard is retired")


def test_auto_guard_still_reproducible_when_explicitly_requested():
    """The retired code is kept, not deleted: auto_guard=True restores the old behaviour."""
    from wavesplat.optimizer.sequential_update import SequentialUpdater

    loader, frames = _init_frames(6)
    cfg = {'init_method': 'auto', 'depth': loader.depth, 'seed': 42,
           'auto_guard': True, 'learned_atoms': _atoms(12)}
    upd = SequentialUpdater(cfg)
    upd._auto_select_init(frames, verbose=False)
    assert upd._auto_bf_peak_ratio is not None
    assert isinstance(upd._auto_guard_fired, bool)


# ── C. median-of-k analytic ensemble ────────────────────────────────────────

def test_make_updater_wraps_auto_only():
    """Explicit init_method must never be ensembled -- this protects paper numbers."""
    from wavesplat.optimizer.sequential_update import (
        make_updater, MedianEnsembleUpdater, SequentialUpdater)

    assert isinstance(make_updater({'init_method': 'auto'}), MedianEnsembleUpdater)
    for m in ('v2_konly', 'v2', 'v1', 'learned', 'relax'):
        upd = make_updater({'init_method': m})
        assert type(upd) is SequentialUpdater, f"{m} must not be wrapped"
    # k=1 disables the wrapper even on 'auto'
    assert type(make_updater({'init_method': 'auto',
                              'analytic_ensemble_k': 1})) is SequentialUpdater


def test_ensemble_k1_is_bit_identical_to_single_v2_konly():
    from wavesplat.optimizer.sequential_update import (
        MedianEnsembleUpdater, SequentialUpdater)
    from wavesplat.run import set_seed

    loader, frames = _init_frames(8)
    base = {'n_gaussians': [3], 'depth': loader.depth, 'lr': 0.02,
            'lambda_l1': 0.0, 'lambda_smooth': 0.0, 'tracking_steps': 3,
            'seed': 42, 'init_method': 'v2_konly'}
    gx, gy = loader.gt_grid_x, loader.gt_grid_y

    set_seed(42)
    single = SequentialUpdater(dict(base))
    single.initialize(frames)
    ref = single.predict_field(gx, gy, 1.0)

    set_seed(42)
    ens = MedianEnsembleUpdater(dict(base, analytic_ensemble_k=1))
    ens.initialize(frames)
    got = ens.predict_field(gx, gy, 1.0)

    assert ens.ensemble_k == 1
    assert torch.equal(got, ref), "k=1 ensemble diverged from the single run"


def test_ensemble_k3_members_seeds_and_pointwise_median():
    from wavesplat.optimizer.sequential_update import MedianEnsembleUpdater
    from wavesplat.run import set_seed

    loader, frames = _init_frames(8)
    cfg = {'n_gaussians': [3], 'depth': loader.depth, 'lr': 0.02,
           'lambda_l1': 0.0, 'lambda_smooth': 0.0, 'tracking_steps': 3,
           'seed': 42, 'init_method': 'v2_konly', 'analytic_ensemble_k': 3}
    gx, gy = loader.gt_grid_x, loader.gt_grid_y

    set_seed(42)
    ens = MedianEnsembleUpdater(cfg)
    res = ens.initialize(frames)

    assert ens.ensemble_k == 3
    assert ens.member_seeds == [42, 1042, 2042], "frozen offsets are {0, +1000, +2000}"
    assert res['ensemble_k'] == 3 and len(res['member_n']) == 3

    out = ens.predict_field(gx, gy, 1.0)
    members = [m.predict_field(gx, gy, 1.0) for m in ens.members]
    expected = torch.median(torch.stack(members, dim=0), dim=0).values
    assert torch.equal(out, expected)
    # odd k -> the median IS one of the member values at every point (no interpolation)
    assert torch.all(sum((out == m) for m in members) > 0)

    upd = ens.update(loader.get_tracking_frames(tp_factor=2.0)[0])
    assert len(upd['member_loss']) == 3
    assert upd['elapsed_ms'] == pytest.approx(sum(upd['member_elapsed_ms']))


# ── D. real-domain gate ─────────────────────────────────────────────────────

def test_frames_have_gt_discriminates_real_from_synthetic():
    from wavesplat.optimizer.sequential_update import _frames_have_gt

    loader, frames = _init_frames(4)
    assert _frames_have_gt(frames) is True
    assert _frames_have_gt(_strip_gt(frames)) is False


def test_real_domain_gate_routes_to_bank_mmse():
    """Deployed default since 2026-07-29: the gate routes real input to the ported
    bank-MMSE estimator (see test_bank_mmse_port.py for the route's own behaviour)."""
    from wavesplat.optimizer.sequential_update import SequentialUpdater

    loader, frames = _init_frames(8)
    real = _strip_gt(frames)
    cfg = {'n_gaussians': [3], 'depth': loader.depth, 'lr': 0.02,
           'tracking_steps': 3, 'seed': 42, 'init_method': 'auto',
           'real_bank_ensemble_k': 1,
           # deliberately routes 'learned' if the gate fails to fire
           'learned_atoms': _atoms(20)}
    upd = SequentialUpdater(cfg)
    with pytest.warns(RuntimeWarning, match='bank-MMSE'):
        upd.initialize(real)

    assert upd._auto_real_gate is True
    assert upd._auto_selected == 'bank_mmse'
    assert upd.init_method == 'bank_mmse'
    assert upd._auto_n_selected is None, "router must not have run at all"
    assert upd._bank is not None and upd.model is None


def test_real_domain_gate_legacy_v2_konly_escape_hatch():
    """real_estimator='v2_konly' reproduces the pre-port 2026-07-27 behaviour."""
    from wavesplat.optimizer.sequential_update import SequentialUpdater

    loader, frames = _init_frames(8)
    real = _strip_gt(frames)
    cfg = {'n_gaussians': [3], 'depth': loader.depth, 'lr': 0.02,
           'tracking_steps': 3, 'seed': 42, 'init_method': 'auto',
           'real_estimator': 'v2_konly', 'learned_atoms': _atoms(20)}
    upd = SequentialUpdater(cfg)
    with pytest.warns(RuntimeWarning, match='NOT authorised on real'):
        upd.initialize(real)

    assert upd._auto_real_gate is True
    assert upd._auto_selected == 'v2_konly'
    assert upd.init_method == 'v2_konly'
    assert upd._auto_n_selected is None, "router must not have run at all"
    assert upd._bank is None and upd.model is not None


def test_real_domain_gate_suppresses_the_ensemble():
    """Spec: the real path never builds the ANALYTIC m3 wrapper ensemble (the bank
    route runs its own internal field ensemble instead, outside this wrapper)."""
    from wavesplat.optimizer.sequential_update import MedianEnsembleUpdater

    loader, frames = _init_frames(8)
    cfg = {'n_gaussians': [3], 'depth': loader.depth, 'lr': 0.02,
           'tracking_steps': 3, 'seed': 42, 'init_method': 'auto',
           'analytic_ensemble_k': 3, 'real_bank_ensemble_k': 1,
           'learned_atoms': _atoms(20)}
    ens = MedianEnsembleUpdater(cfg)
    with pytest.warns(RuntimeWarning):
        res = ens.initialize(_strip_gt(frames))

    assert ens.ensemble_k == 1
    assert res['ensemble_k'] == 1
    assert len(ens.members) == 1


def test_real_domain_gate_does_not_fire_on_synthetic():
    from wavesplat.optimizer.sequential_update import SequentialUpdater

    loader, frames = _init_frames(8)
    cfg = {'n_gaussians': [3], 'depth': loader.depth, 'lr': 0.02,
           'tracking_steps': 3, 'seed': 42, 'init_method': 'auto',
           'learned_atoms': _atoms(20)}
    upd = SequentialUpdater(cfg)
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        upd.initialize(frames)

    assert upd._auto_real_gate is False
    assert upd._auto_selected == 'learned'
    assert upd._auto_n_selected == 20


def test_real_domain_gate_can_be_overridden_deliberately():
    from wavesplat.optimizer.sequential_update import SequentialUpdater

    loader, frames = _init_frames(6)
    cfg = {'init_method': 'auto', 'depth': loader.depth, 'seed': 42,
           'real_domain_gate': False, 'learned_atoms': _atoms(20)}
    upd = SequentialUpdater(cfg)
    # the router itself needs no GT, so it runs and routes learned
    assert upd._auto_select_init(_strip_gt(frames), verbose=False) == 'learned'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
