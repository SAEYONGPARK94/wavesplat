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


def test_bank_refit_every_s_default_is_frozen_after_the_2026_08_23_revert():
    """★2026-08-23 배포로 이 시험의 **기대값이 바뀌었다**(None -> 5.0). 이력을 남긴다.

    종전 문언(축자): *"Exposure must not deploy the value. 5 independent real records give
    an exact one-sided binomial floor of 0.03125, so refit@2s is DESCRIPTIVE evidence only."*
    ⟹ 그 사유는 **레코드 수 5** 였고, 전방팬 관측모형을 빼면서 **12** 가 됐다
    (`tmp/B2_REAL_2026-08-23/scene_station.py`).  근거 사슬은
    `sequential_update.BANK_REFIT_EVERY_S_DEFAULT` 주석에 전부 적혀 있다.
    ⚠종전 문언의 `0.03125` 는 **옳았다** — 같은 날 내 사전등록이 그 값을 0.0625(양측)로
    잘못 적었고, 이 파일이 원래 맞았다.

    ⛔⛔**같은 날 밤 되돌렸다(5.0 -> None).**  사유는 정확도가 아니라 **예산**이다:
    재적합이 `update()` 안에서 **동기로** 돌아 프레임 간격의 **약 20배**(1.35–2.53 s) 블로킹하고,
    내가 근거로 든 `B-6c` 의 `CONTROL-FRAME-OK` 는 **배포 전(재적합 0회) 구성**의 판정이었다.
    ⟹ 출하 기본값은 자기 증거 위에 서야 한다.  권고는
    `sequential_update.recommended_real_refit_config()` 로 남았다.
    """
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


def test_bank_refit_every_s_cli_flag_is_merged_and_can_express_none():
    """run.py merges CLI args with explicit `if args.X is not None` blocks.  Registering
    the flag without adding its block yields a dead flag that parses and is discarded.

    2026-08-23 (RV-6 3): the old form used `type=float`, so there was NO WAY to pass None
    from the command line -- the refit could not be turned off from the CLI.  It now
    accepts 'none'.  This test pins both: the merge actually happens, AND None is
    expressible.
    """
    src = open(_RUN_PY).read()
    assert "'--bank_refit_every_s'" in src, 'flag not registered with argparse'
    i = src.find('if args.bank_refit_every_s is not None:')
    assert i > 0, 'flag parsed but never merged into the config'
    body = src[i:i + 600]
    assert "config['bank_refit_every_s']" in body, 'merge block never writes to config'
    assert "'none'" in body, 'CLI cannot express None -- refit cannot be switched off'


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


def test_bank_refit_default_is_single_sourced():
    """`bank_refit_every_s` 배포값(2026-08-23) 회귀 시험 — 값 + **단일출처**.

    이 프로젝트는 *"같은 상수를 두 곳에 적어 한쪽이 낡는다"* 로 **네 번** 대가를 치렀다
    (`lambda_l1` 3중 기재 · `n` 하드코딩 3회 · 개입 기대값 4중 · 손복사 기준선 3회).
    ⟹ 이 시험은 **값**만이 아니라 **`DEFAULT_CONFIG`·yaml·모듈 상수 셋이 같다**는 것을 고정한다.
    """
    import yaml as _yaml
    from wavesplat.run import DEFAULT_CONFIG
    from wavesplat.optimizer.sequential_update import (BANK_REFIT_EVERY_S_DEFAULT,
                                                       pre_2026_08_23_config)
    assert BANK_REFIT_EVERY_S_DEFAULT is None
    assert DEFAULT_CONFIG['bank_refit_every_s'] == BANK_REFIT_EVERY_S_DEFAULT
    found, val = _yaml_find(_yaml.safe_load(open(_YAML)), 'bank_refit_every_s')
    assert found, 'configs/estimation.yaml 에 bank_refit_every_s 가 없다'
    assert val == BANK_REFIT_EVERY_S_DEFAULT, (
        f'yaml({val}) 이 모듈 상수({BANK_REFIT_EVERY_S_DEFAULT}) 와 어긋난다 — 단일출처 위반')
    # 권고 설정이 **기본값과 다르다**(= 명시적으로 켜야 한다)
    from wavesplat.optimizer.sequential_update import recommended_real_refit_config
    rec = recommended_real_refit_config()
    assert rec == {'bank_refit_every_s': 5.0}
    assert rec['bank_refit_every_s'] != BANK_REFIT_EVERY_S_DEFAULT, (
        '권고가 기본값과 같아지면 recommended_* 가 죽은 코드가 된다')
    assert pre_2026_08_23_config() == {'bank_refit_every_s': None}   # 이제 항등


def test_bank_refit_fallback_matches_shipping_default():
    """설정 키를 **안 준** 추정기도 배포값을 받아야 한다.

    ⚠`lambda_l1` 이 정확히 이 자리에서 깨졌다 — `.get(key, <옛값>)` 폴백이 배포와 갈라져
    S9 배치 하나를 무효화시켰다. 그 부류를 여기서 막는다.
    """
    from wavesplat.optimizer.sequential_update import (SequentialUpdater,
                                                       BANK_REFIT_EVERY_S_DEFAULT)
    upd = SequentialUpdater({'device': 'cpu'})
    assert upd.config.get('bank_refit_every_s', BANK_REFIT_EVERY_S_DEFAULT) \
        == BANK_REFIT_EVERY_S_DEFAULT
    import wavesplat.optimizer.sequential_update as SU
    text = open(SU.__file__).read()
    assert "self.config.get('bank_refit_every_s'," in text, '폴백 위치가 바뀌었다'
    idx = text.index("self.config.get('bank_refit_every_s',")
    tail = text[idx:idx + 200]
    assert 'BANK_REFIT_EVERY_S_DEFAULT' in tail, (
        '폴백이 모듈 상수를 쓰지 않는다 — 옛값 하드코딩이 남아 있다')
