"""Regression tests that pin the SHIPPING defaults.

Why this file exists
--------------------
2026-08-13 audit finding (BLOCKER): the three deployment knobs could all be
reverted at once -- ``lambda_l1`` 1e-4 -> 1e-3, ``init_method`` auto ->
v2_konly, ``tracking_mode`` w_phi -> all -- and the whole suite still reported
``111 passed``.  No test referenced ``DEFAULT_CONFIG`` or ``configs/`` at all.

That is the exact failure class that invalidated the S7 and S8 re-runs: a
default moved underneath batch runners whose baseline arms did not pin it, so
the baseline silently received the treatment value.  The defence is cheap --
assert the deployed values, and assert that the places which are supposed to
mirror one definition actually agree.

If a test here fails you are changing a deployed default.  That is allowed, but
it must be a deliberate act: update the value here, in
``configs/estimation.yaml``, and in the auto-memory deployment record together,
and check that every batch runner's BASELINE arm names the old value
explicitly.
"""

from pathlib import Path

import pytest
import yaml

from wavesplat.optimizer.sequential_update import (
    INIT_METHOD_CLASS_FALLBACK,
    INIT_METHOD_DEFAULT,
    LAMBDA_L1_DEFAULT,
    LAMBDA_L1_PRE_S9,
    SequentialUpdater,
)
from wavesplat.run import DEFAULT_CONFIG, load_yaml_config


REPO_ROOT = Path(__file__).resolve().parents[2]
ESTIMATION_YAML = REPO_ROOT / "configs" / "estimation.yaml"

# The deployed operating point.  Each entry carries the date and the batch that
# authorised it so a future reader can find the evidence.
SHIPPING_DEFAULTS = {
    # 2026-08-11, S9 (verdict BOTH); single-sourced in commit 5f0846c.
    "lambda_l1": 1e-4,
    # 2026-07-26 flip (user decision): GT-free regime router.
    "init_method": "auto",
    # kx/ky frozen during tracking; only w and phi are updated.
    "tracking_mode": "w_phi",
    # 2026-07-27 confirm2: median-of-3 on the analytic fallback.
    "analytic_ensemble_k": 3,
}


@pytest.mark.parametrize("key,expected", sorted(SHIPPING_DEFAULTS.items()))
def test_default_config_pins_shipping_value(key, expected):
    assert key in DEFAULT_CONFIG, f"{key} vanished from DEFAULT_CONFIG"
    assert DEFAULT_CONFIG[key] == expected, (
        f"DEFAULT_CONFIG['{key}'] is {DEFAULT_CONFIG[key]!r}, deployed value is "
        f"{expected!r}.  See this module's docstring before changing it."
    )


def test_lambda_l1_has_exactly_one_definition():
    """run.py must import the constant, not re-type the literal."""
    assert LAMBDA_L1_DEFAULT == SHIPPING_DEFAULTS["lambda_l1"]
    assert DEFAULT_CONFIG["lambda_l1"] is LAMBDA_L1_DEFAULT


def test_estimation_yaml_mirrors_default_config():
    """The yaml is a mirror; drift between the two is what bit S7/S8."""
    raw = yaml.safe_load(ESTIMATION_YAML.read_text())
    optimiser = raw["optimizer"]
    assert optimiser["lambda_l1"] == pytest.approx(SHIPPING_DEFAULTS["lambda_l1"])
    assert optimiser["tracking_mode"] == SHIPPING_DEFAULTS["tracking_mode"]
    assert optimiser["init_method"] == SHIPPING_DEFAULTS["init_method"]
    assert optimiser["analytic_ensemble_k"] == SHIPPING_DEFAULTS["analytic_ensemble_k"]


def test_updater_fallback_matches_default_config():
    """A runner that passes an incomplete config must land on the shipped value.

    This is the actual contamination path: a batch runner builds an arm dict
    without ``lambda_l1``, so the arm inherits whatever the fallback happens to
    be that week.
    """
    updater = SequentialUpdater({})
    assert updater.lambda_l1 == SHIPPING_DEFAULTS["lambda_l1"]


def test_no_stale_lambda_literal_in_default_config_comment():
    """Guard the docstring drift found on 2026-08-13.

    ``sequential_update`` documented ``default 0.001`` for 40 lines below the
    line that actually sets 1e-4.  Docstrings are what people read.
    """
    import wavesplat.optimizer.sequential_update as su

    doc = SequentialUpdater.__init__.__doc__ or ""
    assert "0.001" not in doc, (
        "SequentialUpdater.__init__ docstring still advertises lambda_l1=0.001; "
        f"the deployed value is {su.LAMBDA_L1_DEFAULT}."
    )


# ── init_method: one shipping default, one deliberately different class fallback ──
# Added 2026-08-21.  ``init_method`` had the same three-copy layout that bit
# ``lambda_l1``: 'auto' in DEFAULT_CONFIG, 'v2_konly' in the SequentialUpdater
# fallback, and a THIRD value 'v2' in the yaml loader.  Two of those are now
# imported from one definition each; the third was a plain bug and is gone.


def test_init_method_has_exactly_one_definition():
    """run.py must import the constant, not re-type the literal."""
    assert INIT_METHOD_DEFAULT == SHIPPING_DEFAULTS["init_method"]
    assert DEFAULT_CONFIG["init_method"] is INIT_METHOD_DEFAULT


def test_yaml_loader_fallback_tracks_default_config(tmp_path):
    """A yaml with an ``optimizer:`` block but no ``init_method`` must not
    silently downgrade the shipped router.

    Before 2026-08-21 this fallback was a hand-written ``'v2'`` -- neither the
    shipping default nor the class fallback -- and because a yaml-loaded config
    OVERRIDES ``DEFAULT_CONFIG`` in ``run_estimation``, it would have won.
    """
    partial = tmp_path / "partial.yaml"
    partial.write_text("optimizer:\n  lr: 0.01\n")
    cfg = load_yaml_config(str(partial))
    assert cfg["init_method"] == DEFAULT_CONFIG["init_method"]
    # ...and lambda_l1 keeps the property this test's older sibling won in S9
    assert cfg["lambda_l1"] == DEFAULT_CONFIG["lambda_l1"]


def test_class_fallback_is_the_legacy_route_not_the_shipping_default():
    """``SequentialUpdater({})`` must stay 'v2_konly'.

    This fallback is NOT dead (measured 2026-08-21): the shipped
    ``test_phase_convention.py`` and eight ``tmp/`` research runners construct
    the class with a config that omits ``init_method`` and then run it.  It also
    cannot simply be set to 'auto': the router's median-of-k ensemble lives in
    ``make_updater``/``MedianEnsembleUpdater``, so a bare updater carrying 'auto'
    would be a third behaviour that has never shipped and never been measured.
    """
    assert INIT_METHOD_CLASS_FALLBACK != INIT_METHOD_DEFAULT
    assert SequentialUpdater({}).init_method == INIT_METHOD_CLASS_FALLBACK


def test_s8_probe_pin_is_the_pre_s9_value_not_the_current_default():
    """The S8 env-gates pin lambda_l1 to the OLD default, deliberately.

    Two bare ``0.001`` literals sat in those gates under comments calling them
    "the SHIPPED default".  True on 2026-08-10; false from 2026-08-11, when S9
    deployed 1e-4.  The pin must stay at 1e-3 for S8 to reproduce -- this test
    exists so the next reader cannot mistake it for stale drift and "fix" it.
    """
    assert LAMBDA_L1_PRE_S9 == 1e-3
    assert LAMBDA_L1_PRE_S9 != LAMBDA_L1_DEFAULT


# ── VARPRO 배포 회귀 (2026-08-22) ──────────────────────────────────────────────
# 배포 전까지 이 처방은 tmp/ 의 몽키패치로만 존재했다("측정됐는데 배포 안 됨").
# 아래 시험들은 (1) 값이 단일출처이고 (2) yaml·DEFAULT_CONFIG 가 그것을 미러하며
# (3) 처치가 실제로 켜져 있는지를 고정한다. 저장소가 반복해 물린 결함 부류의 방벽이다.

def test_varpro_defaults_are_single_sourced():
    from wavesplat.optimizer.sequential_update import (VARPRO_STEPS_DEFAULT,
                                                       LEARNED_INIT_ITERS_DEFAULT)
    from wavesplat.run import DEFAULT_CONFIG
    assert DEFAULT_CONFIG['varpro_steps'] is VARPRO_STEPS_DEFAULT
    assert DEFAULT_CONFIG['learned_init_iters'] is LEARNED_INIT_ITERS_DEFAULT
    # 배포된 구성 = B 배치 A-VP 팔. 값을 바꾸면 W4/W5/B 재현이 끊긴다.
    assert VARPRO_STEPS_DEFAULT == 400
    assert LEARNED_INIT_ITERS_DEFAULT == 0


def test_estimation_yaml_mirrors_varpro_defaults():
    import os
    import yaml
    from wavesplat.optimizer.sequential_update import (VARPRO_STEPS_DEFAULT,
                                                       LEARNED_INIT_ITERS_DEFAULT)
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, '..', '..', 'configs', 'estimation.yaml')
    opt = yaml.safe_load(open(path))['optimizer']
    assert opt['varpro_steps'] == VARPRO_STEPS_DEFAULT
    assert opt['learned_init_iters'] == LEARNED_INIT_ITERS_DEFAULT


def test_varpro_polar_map_matches_forward_model_sign():
    """a = w cos phi, b = -w sin phi  (전방모델이 w cos(theta+phi) 이므로).

    2026-08-19 에 sequential_update 의 한 사이트가 atan2(b, a) 로 이탈해 있었다.
    varpro 의 역변환이 같은 이탈을 하지 않는지 고정한다.
    """
    import torch
    from wavesplat.optimizer import varpro as V
    w = torch.tensor([0.3, 1.1, 0.7], dtype=torch.float64)
    phi = torch.tensor([0.4, -2.0, 3.0], dtype=torch.float64)
    a, b = V.wphi_to_ab(w, phi)
    assert torch.allclose(a, w * torch.cos(phi))
    assert torch.allclose(b, -w * torch.sin(phi))
    w2, phi2 = V.ab_to_wphi(a, b)
    assert torch.allclose(w, w2)
    d = torch.remainder(phi2 - phi + torch.pi, 2 * torch.pi) - torch.pi
    assert torch.allclose(d, torch.zeros_like(d), atol=1e-12)


def test_varpro_can_be_disabled_to_pre_deployment_behaviour():
    """varpro_steps=0 이면 처치가 완전히 꺼져야 한다 (배포 전 거동 복원 경로)."""
    from wavesplat.optimizer.sequential_update import SequentialUpdater
    u = SequentialUpdater({'varpro_steps': 0, 'learned_init_iters': 500})
    assert int(u.config.get('varpro_steps')) == 0
    assert int(u.config.get('learned_init_iters')) == 500
