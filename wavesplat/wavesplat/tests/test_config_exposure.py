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


def test_bank_refit_cannot_affect_synthetic_runs():
    """`bank_refit_every_s` 를 바꿔도 합성 결과는 **원리적으로** 못 바뀐다.

    2026-08-23 `B-3` 이 실측에서 `refit@2s` 를 36/36 으로 확증했고, 배포 기본값
    (`None`) 변경이 논의된다.  그 변경의 **합성 회귀 위험이 0** 임을 코드 불변량으로 고정한다:

      · `_bank_update` 는 `self._bank is not None` 일 때만 불린다
        (`sequential_update.py:update()` 첫 줄).
      · `self._bank` 는 `_initialize_bank` 에서만 채워지고, 그것은 실측 게이트
        (`not _frames_have_gt(frames)`) 안에서만 도달한다.
      · 합성 프레임은 `EpisodeDataLoader` 가 항상 `gt_field` 를 붙인다.

    ⟹ 합성 프레임으로 초기화하면 `_bank` 가 `None` 으로 남아야 한다.
    이 시험이 깨지면 **`bank_refit_every_s` 기본값 변경이 더 이상 안전하지 않다.**
    """
    import torch
    from wavesplat.data_loader import FrameData
    from wavesplat.optimizer.sequential_update import make_updater
    from wavesplat.run import DEFAULT_CONFIG

    n = 40
    g = torch.linspace(-5.0, 5.0, 8)
    frames = [FrameData(t=float(i) * 0.5,
                        x_obs=torch.linspace(0.0, 20.0, n),
                        y_obs=torch.zeros(n),
                        z_obs=torch.sin(torch.linspace(0.0, 6.0, n) + i * 0.3),
                        confidence=torch.ones(n),
                        incidence_angles=torch.zeros(n),
                        gt_field=torch.outer(g, g))          # ★합성의 표식
              for i in range(6)]
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(device='cpu', seed=0, depth=30.0, bank_refit_every_s=2.0,
               max_iters_per_stage=[1, 1, 1], n_gaussians=2, max_n=2)
    upd = make_updater(cfg)
    upd.initialize(frames, x_range=(0.0, 20.0), y_range=(-5.0, 5.0), verbose=False)
    inner = getattr(upd, '_primary', upd)
    assert getattr(inner, '_bank', None) is None, (
        '합성 입력이 bank 경로에 들어갔다 — bank_refit_every_s 기본값 변경이 '
        '더 이상 합성에 무해하지 않다')
