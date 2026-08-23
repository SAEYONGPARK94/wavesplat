"""Tests for the bank-MMSE operational port (2026-07-29, user-directed deployment).

Pins four properties:
  A. the ported math is bit-exact against the research implementation
     (scripts/d3_sa_nomp.mmse_bank_estimate) for both kp_hint modes;
  B. the real-domain gate routes to the bank and the route honours the
     SequentialUpdater public surface (initialize/update/predict_field/
     amplitude_refit) including the make_updater wrapper;
  C. the internal median-of-k field ensemble, the deterministic 600-obs cap and
     the rolling-window/refit knobs behave as specced;
  D. explicit init_methods and synthetic (GT-carrying) input never touch the
     bank route -- published synthetic numbers are unaffected.

Real-window accuracy itself is measured, not tested, on the frozen GIF-ledger
windows: tmp/bankmmse_port_2026-07-29/{blind_vs_frozen,m3_field}.json.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import warnings

import numpy as np
import pytest
import torch

from wavesplat.data_loader import FrameData

G = 9.81
DEPTH = 30.0


def _wave_field(P, rng_noise=None):
    """Two dispersion-consistent plane waves, optionally with obs noise."""
    def om(k):
        return np.sqrt(G * k * np.tanh(k * DEPTH))
    k1, th1, k2, th2 = 0.12, 0.3, 0.2, -0.8
    z = (0.8 * np.cos(k1 * np.cos(th1) * P[:, 0] + k1 * np.sin(th1) * P[:, 1]
                      - om(k1) * P[:, 2] + 0.5)
         + 0.4 * np.cos(k2 * np.cos(th2) * P[:, 0] + k2 * np.sin(th2) * P[:, 1]
                        - om(k2) * P[:, 2] + 1.7))
    if rng_noise is not None:
        z = z + rng_noise.normal(0, 0.03, len(z))
    return z


def _real_frames(n_frames=15, n_per=40, t0=0.0, dt=2.0, seed=11):
    """Real-data-shaped frames (gt_field=None) from the synthetic 2-wave sea."""
    rng = np.random.default_rng(seed)
    frames, Ps, zs = [], [], []
    for j in range(n_frames):
        t = t0 + j * dt
        x = rng.uniform(0, 80, n_per)
        y = rng.uniform(0, 80, n_per)
        P = np.column_stack([x, y, np.full(n_per, t)])
        z = _wave_field(P, rng_noise=rng)
        Ps.append(P)
        zs.append(z)
        frames.append(FrameData(
            t=float(t),
            x_obs=torch.tensor(x, dtype=torch.float32),
            y_obs=torch.tensor(y, dtype=torch.float32),
            z_obs=torch.tensor(z, dtype=torch.float32),
            confidence=torch.ones(n_per, dtype=torch.float32),
            incidence_angles=torch.zeros(n_per, dtype=torch.float32),
            gt_field=None))
    return frames, np.vstack(Ps), np.concatenate(zs)


def _bank_cfg(**kw):
    cfg = {'init_method': 'auto', 'depth': DEPTH, 'seed': 7,
           'real_bank_ensemble_k': 1,
           # deliberately routes 'learned' if the gate fails to fire
           'learned_atoms': dict(kx=np.arange(20) * 0.2 + 0.1,
                                 ky=np.zeros(20), energy=np.ones(20))}
    cfg.update(kw)
    return cfg


# ── A. bit-exactness vs the research implementation ─────────────────────────

@pytest.mark.parametrize('kp_hint', [None, 0.13])
def test_port_bit_exact_vs_research(kp_hint):
    from wavesplat.optimizer.bank_mmse import fit_bank_mmse

    cwd = os.getcwd()
    try:
        scripts = os.path.join(os.path.dirname(__file__), '..', '..', 'scripts')
        sys.path.insert(0, scripts)
        from d3_sa_nomp import mmse_bank_estimate   # chdirs to the package root
    finally:
        os.chdir(cwd)

    rng = np.random.default_rng(7)
    n = 150
    P = np.column_stack([rng.uniform(0, 80, n), rng.uniform(0, 80, n),
                         rng.uniform(0, 30, n)])
    z = _wave_field(P, rng_noise=rng)
    Pq = np.column_stack([rng.uniform(0, 80, 60), rng.uniform(0, 80, 60),
                          np.full(60, 31.0)])
    sigma0 = 0.2 * np.std(z)

    ref = mmse_bank_estimate(P, z, DEPTH, Pq, sigma0=sigma0,
                             kp_hint=kp_hint, seed=5)
    fit = fit_bank_mmse(P, z, DEPTH, sigma0, kp_hint=kp_hint, seed=5)
    assert np.array_equal(ref, fit.render(Pq)), (
        "ported fit+render diverged from the research mmse_bank_estimate")


# ── B. gate routing + public surface ────────────────────────────────────────

def test_gate_routes_to_bank_and_public_surface_works():
    from wavesplat.optimizer.sequential_update import (
        make_updater, MedianEnsembleUpdater)

    frames, _, _ = _real_frames()
    upd = make_updater(_bank_cfg())
    assert isinstance(upd, MedianEnsembleUpdater)
    with pytest.warns(RuntimeWarning, match='bank-MMSE'):
        res = upd.initialize(frames, x_range=(0, 80), y_range=(0, 80))

    # wrapper stays k=1 -- the bank has its own internal ensemble
    assert upd.ensemble_k == 1 and res['ensemble_k'] == 1
    assert res['bank_members'] == 1 and res['final_n'] == 0
    assert upd._auto_selected == 'bank_mmse' and upd._auto_real_gate is True
    assert np.isfinite(res['bank_val_misfit']).all()

    gx = torch.linspace(0, 80, 12)
    gy = torch.linspace(0, 80, 9)
    pred = upd.predict_field(gx, gy, 31.0)
    assert pred.shape == (9, 12) and torch.isfinite(pred).all()

    # accuracy sanity on the clean synthetic sea: the bank must reconstruct it
    Xg, Yg = np.meshgrid(np.linspace(0, 80, 12), np.linspace(0, 80, 9))
    Q = np.column_stack([Xg.ravel(), Yg.ravel(), np.full(Xg.size, 31.0)])
    gt = _wave_field(Q).reshape(9, 12)
    nrmse = float(np.sqrt(np.mean((pred.numpy() - gt) ** 2)) / np.std(gt))
    assert nrmse < 0.3, f"bank route lost the synthetic sea (nRMSE {nrmse:.3f})"

    new = _real_frames(n_frames=1, t0=30.0, seed=99)[0][0]
    out = upd.update(new)
    assert set(out) >= {'loss', 'n_pts', 'elapsed_ms'}
    assert np.isfinite(out['loss']) and out['n_pts'] == 40

    assert upd.amplitude_refit([new]) == {}


# ── C. internal ensemble, obs cap, rolling window ───────────────────────────

def test_bank_internal_median_of_k_field():
    from wavesplat.optimizer.sequential_update import SequentialUpdater

    frames, _, _ = _real_frames()
    upd = SequentialUpdater(_bank_cfg(real_bank_ensemble_k=3))
    with pytest.warns(RuntimeWarning):
        upd.initialize(frames)

    assert upd._bank_seeds == [7, 1007, 2007], "offsets are {0, +1000, +2000}"
    assert len(upd._bank) == 3

    gxn, gyn = np.linspace(0, 80, 7), np.linspace(0, 80, 5)
    out = upd.predict_field(torch.tensor(gxn), torch.tensor(gyn), 12.0)
    Xg, Yg = np.meshgrid(gxn, gyn)
    Q = np.column_stack([Xg.ravel(), Yg.ravel(), np.full(Xg.size, 12.0)])
    members = np.stack([f.render(Q) for f in upd._bank], axis=0)
    expected = np.median(members, axis=0).reshape(5, 7)
    assert np.allclose(out.numpy(), expected.astype(np.float32), atol=0)
    # odd k -> the median IS one of the member values at every point
    assert np.all(np.any(out.numpy()[None] == members.reshape(3, 5, 7)
                         .astype(np.float32), axis=0))


def test_bank_obs_cap_is_the_frozen_protocol_subsample():
    from wavesplat.optimizer.sequential_update import SequentialUpdater
    from wavesplat.optimizer.bank_mmse import fit_bank_mmse

    frames, _, _ = _real_frames(n_frames=18, n_per=50)   # 900 obs > 600 cap
    upd = SequentialUpdater(_bank_cfg())
    with pytest.warns(RuntimeWarning):
        upd.initialize(frames)
    assert upd._bank[0].n_obs == 600

    # reference window = the same frame-assembled (float32-rounded) obs the
    # updater saw, capped with the frozen-protocol rng(0) subsample
    P, z = SequentialUpdater._bank_obs_from_frames(frames)
    ii = np.random.default_rng(0).choice(len(z), 600, replace=False)
    Pc, zc = P[ii], z[ii]
    ref = fit_bank_mmse(Pc, zc, DEPTH, 0.05 * 4.0 * float(np.std(zc)),
                        kp_hint=None, seed=7)
    rng = np.random.default_rng(1)
    Q = np.column_stack([rng.uniform(0, 80, 40), rng.uniform(0, 80, 40),
                         np.full(40, 20.0)])
    assert np.array_equal(upd._bank[0].render(Q), ref.render(Q)), (
        "deployed cap diverged from the frozen-protocol rng(0) subsample")


def test_bank_rolling_window_and_refit_trigger():
    from wavesplat.optimizer.sequential_update import SequentialUpdater

    frames, _, _ = _real_frames(n_frames=5, n_per=40, dt=2.0)   # t = 0..8
    upd = SequentialUpdater(_bank_cfg(bank_memory_s=10.0, bank_refit_every_s=5.0))
    with pytest.warns(RuntimeWarning):
        upd.initialize(frames)
    t_fit0 = upd._bank_last_fit_t
    fit0 = upd._bank[0]

    for f in _real_frames(n_frames=4, t0=10.0, dt=2.0, seed=33)[0]:   # t = 10..16
        upd.update(f)

    assert upd._bank_last_fit_t >= t_fit0 + 5.0, "refit never triggered"
    assert upd._bank[0] is not fit0, "refit did not replace the fit"
    P_buf = upd._bank_obs[0]
    assert P_buf[:, 2].min() >= 16.0 - 10.0 - 1e-9, "memory window not evicted"


def test_bank_refit_default_refits_and_revert_freezes():
    """★2026-08-23 배포로 이 시험의 기대가 **뒤집혔다**(동결 -> 5 s 재적합).

    종전 이름은 `test_bank_refit_default_is_frozen` 이었고 *"default must keep the init fit
    frozen (the validated protocol)"* 를 주장했다.  그 protocol 은 **실측 레코드 5개**에
    묶여 있었고, 전방팬 관측모형을 빼면서 **12개**가 되어 `B-3`/`B-3b`/`B-3c`/`B-6c` 가
    재적합을 확증했다(`sequential_update.BANK_REFIT_EVERY_S_DEFAULT` 주석 참조).

    ⟹ 이제 **두 방향을 다 고정**한다: 배포 기본값은 **재적합하고**,
    `pre_2026_08_23_config()` 는 **동결로 되돌린다**.  한쪽만 시험하면 되돌림이 죽은 코드가 된다.
    """
    from wavesplat.optimizer.sequential_update import (SequentialUpdater,
                                                       pre_2026_08_23_config,
                                                       BANK_REFIT_EVERY_S_DEFAULT)

    def _drive(cfg_extra):
        frames, _, _ = _real_frames(n_frames=5, n_per=40, dt=2.0)
        cfg = _bank_cfg(); cfg.update(cfg_extra)
        upd = SequentialUpdater(cfg)
        with pytest.warns(RuntimeWarning):
            upd.initialize(frames)
        fit0 = upd._bank[0]
        for f in _real_frames(n_frames=8, t0=10.0, dt=2.0, seed=33)[0]:
            upd.update(f)
        return fit0, upd._bank[0]

    # (1) 배포 기본값 = 재적합한다.  구동 창이 기본 주기보다 길어야 시험이 공허하지 않다.
    assert 8 * 2.0 > BANK_REFIT_EVERY_S_DEFAULT, '구동 창이 주기보다 짧다 — 시험이 공허하다'
    a, b = _drive({})
    assert b is not a, ('배포 기본값(bank_refit_every_s='
                        f'{BANK_REFIT_EVERY_S_DEFAULT})에서 재적합이 발화하지 않았다')

    # (2) 되돌림 = 동결.  이 절이 없으면 `pre_2026_08_23_config()` 가 죽은 코드가 된다.
    a2, b2 = _drive(pre_2026_08_23_config())
    assert b2 is a2, 'pre_2026_08_23_config() 가 초기 적합을 동결시키지 못한다'



# ── D. nothing else may reach the bank ──────────────────────────────────────

def test_explicit_init_method_never_banks():
    from wavesplat.optimizer.sequential_update import SequentialUpdater

    frames, _, _ = _real_frames(n_frames=8)
    cfg = {'init_method': 'v2_konly', 'depth': DEPTH, 'seed': 7,
           'n_gaussians': [3], 'tracking_steps': 3}
    upd = SequentialUpdater(cfg)
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        upd.initialize(frames)
    assert upd._bank is None and upd.model is not None


def test_synthetic_gt_input_never_banks():
    from wavesplat.optimizer.sequential_update import SequentialUpdater

    frames, _, _ = _real_frames(n_frames=6)
    gt = torch.ones(4, 4)
    synth = [FrameData(t=f.t, x_obs=f.x_obs, y_obs=f.y_obs, z_obs=f.z_obs,
                       confidence=f.confidence,
                       incidence_angles=f.incidence_angles, gt_field=gt)
             for f in frames]
    # 7 atoms -> router picks v2_konly (fast, no checkpoint); small model for speed
    upd = SequentialUpdater(_bank_cfg(
        learned_atoms=dict(kx=np.arange(7) * 0.2 + 0.1, ky=np.zeros(7),
                           energy=np.ones(7)),
        n_gaussians=[3], lr=0.02, tracking_steps=3))
    upd.initialize(synth)
    assert upd._auto_real_gate is False
    assert upd._bank is None, "GT-carrying input must go through the router"


def test_bad_real_estimator_value_raises():
    from wavesplat.optimizer.sequential_update import SequentialUpdater

    frames, _, _ = _real_frames(n_frames=6)
    upd = SequentialUpdater(_bank_cfg(real_estimator='typo'))
    with pytest.raises(ValueError, match='real_estimator'):
        upd.initialize(frames)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
