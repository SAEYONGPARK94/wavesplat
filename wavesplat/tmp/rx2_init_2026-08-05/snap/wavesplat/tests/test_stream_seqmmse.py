"""Unit tests for the streaming sequential MMSE core (2026-07-06).

Covers the load-bearing math of scripts/d3_stream_seqmmse.py:
  - RLS (tau=None) is EXACTLY the batch MMSE posterior mean
  - slow-loop rebuild() reproduces the batch solve on the buffer
  - P stays symmetric over many rank-m downdates (the 07-06 bug)
  - NIS is O(1) for a well-specified model
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..',
                                'scripts'))

from d3_stream_seqmmse import (make_bank, design, StreamMMSE,  # noqa: E402
                               batch_mmse)

G = 9.81
DEPTH = 50.0
KP = (2 * np.pi / 8.0) ** 2 / G


def _sim_obs(rng, kxb, kyb, om, gam, s2, n_frames=25, m=8):
    """Draw a sea from the prior and observe it noisily."""
    B = len(kxb)
    w_true = rng.normal(0, np.sqrt(np.maximum(gam, 1e-12)))
    Ps, zs = [], []
    for j in range(n_frames):
        t = 0.1 * j
        x = rng.uniform(-80, 80, m)
        y = rng.uniform(-80, 80, m)
        P = np.column_stack([x, y, np.full(m, t)])
        z = design(kxb, kyb, om, P) @ w_true \
            + rng.normal(0, np.sqrt(s2), m)
        Ps.append(P)
        zs.append(z)
    return Ps, zs, w_true


@pytest.fixture
def small_bank():
    kxb, kyb, om = make_bank(KP, DEPTH, dk_frac=0.5)   # ~60 cells
    B = len(kxb)
    rng = np.random.default_rng(0)
    g = rng.uniform(0.5, 1.5, B) * 0.01
    gam = np.concatenate([g, g])
    return kxb, kyb, om, gam, 0.02 ** 2


def test_rls_equals_batch(small_bank):
    kxb, kyb, om, gam, s2 = small_bank
    rng = np.random.default_rng(1)
    Ps, zs, _ = _sim_obs(rng, kxb, kyb, om, gam, s2)
    st = StreamMMSE(kxb, kyb, om, gam, s2, tau=None)
    for P, z in zip(Ps, zs):
        st.step(P, z)
    Q = np.column_stack([rng.uniform(-50, 50, 40),
                         rng.uniform(-50, 50, 40),
                         np.full(40, 2.5)])
    f_st, _ = st.predict(Q)
    f_ba = batch_mmse(kxb, kyb, om, gam, s2,
                      np.vstack(Ps), np.concatenate(zs), Q)
    assert np.max(np.abs(f_st - f_ba)) < 1e-8 * (np.std(f_ba) + 1e-12) \
        + 1e-10


def test_rebuild_equals_batch(small_bank):
    kxb, kyb, om, gam, s2 = small_bank
    rng = np.random.default_rng(2)
    Ps, zs, _ = _sim_obs(rng, kxb, kyb, om, gam, s2)
    Pb, zb = np.vstack(Ps), np.concatenate(zs)
    st = StreamMMSE(kxb, kyb, om, gam, s2, tau=None)
    st.rebuild(Pb, zb, gam, kxb, kyb, om)
    Q = np.column_stack([rng.uniform(-50, 50, 30),
                         rng.uniform(-50, 50, 30),
                         np.full(30, 2.0)])
    f_rb, _ = st.predict(Q)
    f_ba = batch_mmse(kxb, kyb, om, gam, s2, Pb, zb, Q)
    assert np.max(np.abs(f_rb - f_ba)) < 1e-8


def test_P_stays_symmetric(small_bank):
    kxb, kyb, om, gam, s2 = small_bank
    rng = np.random.default_rng(3)
    Ps, zs, _ = _sim_obs(rng, kxb, kyb, om, gam, s2, n_frames=200)
    st = StreamMMSE(kxb, kyb, om, gam, s2, tau=10.0, dt=0.1)
    for P, z in zip(Ps, zs):
        st.step(P, z)
    asym = np.max(np.abs(st.P - st.P.T))
    assert asym < 1e-12
    assert np.diag(st.P).min() > -1e-10


def test_nis_calibrated(small_bank):
    kxb, kyb, om, gam, s2 = small_bank
    rng = np.random.default_rng(4)
    Ps, zs, _ = _sim_obs(rng, kxb, kyb, om, gam, s2, n_frames=60)
    st = StreamMMSE(kxb, kyb, om, gam, s2, tau=None)
    nis = [st.step(P, z) for P, z in zip(Ps, zs)]
    m = float(np.mean(nis))
    assert 0.5 < m < 2.0, f"NIS {m} not O(1) for well-specified model"
