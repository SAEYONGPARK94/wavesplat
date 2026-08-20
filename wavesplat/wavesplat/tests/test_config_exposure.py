"""The config surface is not covered by the rest of the suite.

CLAUDE.md records the measurement: lambda_l1, init_method and tracking_mode could all be
reverted to their pre-deployment values at once and the suite still passed, because no
test referenced DEFAULT_CONFIG or configs/ at all.  Two keys have since been found
implemented-but-unreachable -- learned_merge_frac, and bank_refit_every_s (exposed
2026-08-20) -- where the code honoured a value that no config surface could supply.

These tests pin the chain that makes an exposed key real:
    CLI flag -> merged config -> SequentialUpdater.config -> the reader in the optimizer
and pin that exposing the key did NOT move the shipping default.
"""
import os
import re

import yaml

from wavesplat.optimizer.sequential_update import SequentialUpdater
from wavesplat.run import DEFAULT_CONFIG

_RUN_PY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'run.py')
_YAML = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), 'configs', 'estimation.yaml')


def _yaml_find(node, key):
    if isinstance(node, dict):
        if key in node:
            return True, node[key]
        for v in node.values():
            found, val = _yaml_find(v, key)
            if found:
                return True, val
    return False, None


def test_bank_refit_every_s_default_is_unchanged():
    """Exposure must not deploy the value.  5 independent real records give an exact
    one-sided binomial floor of 0.03125, so refit@2s is DESCRIPTIVE evidence only."""
    assert DEFAULT_CONFIG['bank_refit_every_s'] is None
    found, val = _yaml_find(yaml.safe_load(open(_YAML)), 'bank_refit_every_s')
    assert found, 'configs/estimation.yaml must mirror the key'
    assert val is None


def test_bank_refit_every_s_survives_into_the_updater():
    """The optimizer reads it off self.config (sequential_update.py:824), so the key
    must still be there after construction -- not swallowed by a whitelist."""
    for value in (None, 2.0):
        upd = SequentialUpdater({**DEFAULT_CONFIG, 'bank_refit_every_s': value})
        assert upd.config.get('bank_refit_every_s', '<dropped>') == value


def test_bank_refit_every_s_cli_flag_is_merged_not_just_parsed():
    """run.py merges CLI args with explicit `if args.X is not None` blocks.  Registering
    the flag without adding its block yields a dead flag that parses and is discarded."""
    src = open(_RUN_PY).read()
    assert "'--bank_refit_every_s'" in src, 'flag not registered with argparse'
    assert re.search(r"if args\.bank_refit_every_s is not None:\s*\n\s*"
                     r"config\['bank_refit_every_s'\] = args\.bank_refit_every_s",
                     src), 'flag parsed but never merged into the config'


def test_optimizer_reads_the_key_by_the_exposed_name():
    """A rename on either side would break the chain silently."""
    opt = open(os.path.join(os.path.dirname(_RUN_PY), 'optimizer',
                            'sequential_update.py')).read()
    assert "self.config.get('bank_refit_every_s'" in opt
