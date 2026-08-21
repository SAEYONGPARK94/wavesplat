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
    LAMBDA_L1_DEFAULT,
    SequentialUpdater,
)
from wavesplat.run import DEFAULT_CONFIG


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
