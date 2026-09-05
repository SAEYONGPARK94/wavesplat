"""★`max_n` 분리 (2026-09-05) 회귀 시험 — **다시 합쳐지면 여기서 죽는다.**

⛔이 파일이 있는 이유. `max_n` 이라는 **한 키**가 서로 무관한 두 역할을 했다:

  ① **제안 풀링 상한** — `learned_init.pool_and_select` 가 설치할 원자 수의 상한.
  ② **성장 천장**     — `sequential_update._adaptive_prune_grow` 의 이른 반환
                       (`n_gaussians >= ceiling` 이면 성장 분기가 통째로 죽는다).

⟹ **성분 수 축의 개입이 원리적으로 직교할 수 없었다.** 저장소가 이미 측정했다:
`max_n=10` 은 `SCALE_INVARIANT_GROW=True` 와 묶어도 `max_n=10` 단독과 **비트 동일**하다
(12/12, `tmp/B2_REAL_2026-08-23/RESULT_B2F.md`). 그리고 2026-09-05 제3자 코퍼스 실측에서
배포 팔이 풀링 상한을 **24/24 블록에서 포화**한다 ⟹ 그 코퍼스에서 ②는 **한 번도
발화하지 않는다**.

★이 저장소가 반복해 물린 결함 부류는 *"같은 기능이 두 곳에 구현돼 한쪽만 계측"* 이다.
분리는 그 역(逆)이다 — **다른 두 기능이 한 이름을 공유**했다. 어느 쪽이든 처방은 같다:
**기계가 지키게 하라.** 아래 시험은 값이 아니라 **거동**을 본다(문자열 검사는 항진한다 —
`'real_gate' in getsource` 가 2026-09-01 에 그렇게 통과했다).

⛔**배포 거동 불변이 이 분리의 전제다.** 둘 다 기본값 30 이고 레거시 `max_n` 은 지금도
**둘 다** 설정한다. 아래 `test_legacy_max_n_still_couples_the_two_roles` 가 바로 그
2026-08-23 의 결합(=12/12 를 낳은 성질)을 **보존됐음의 증거로** 다시 잰다.
"""
import os

import pytest
import torch
import yaml

from wavesplat.optimizer.learned_init import (
    POOL_CAP_DEFAULT, learned_init_params, pool_and_select, resolve_pool_cap)
from wavesplat.optimizer.sequential_update import (
    GROWTH_CEILING_DEFAULT, SequentialUpdater, resolve_growth_ceiling)
from wavesplat.run import DEFAULT_CONFIG

_YAML = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), 'configs', 'estimation.yaml')

# 잘 분리된 원자 12개 (dedup_dk=0.03 보다 훨씬 멀다) — 풀링이 상한 말고 다른 이유로
# 잘라내지 않도록 에너지도 전부 문턱(0.03·max) 위에 둔다.
_N_ATOMS = 12
_ATOMS = dict(kx=[0.10 + 0.05 * i for i in range(_N_ATOMS)],
              ky=[0.20 - 0.01 * i for i in range(_N_ATOMS)],
              energy=[1.0 - 0.02 * i for i in range(_N_ATOMS)])


def _pool_cfg(**kw):
    cfg = dict(learned_atoms=_ATOMS, learned_konly=True)
    cfg.update(kw)
    return cfg


def _growth_cfg(**kw):
    """성장 분기까지 도달하는 최소 구성. `prune_threshold=0.0` 이면 가지치기는 no-op
    (target_fraction=1.0 ⟹ n_keep == n_before) 이라 이른 반환 검사가 첫 분기가 된다."""
    cfg = dict(device='cpu', depth=30.0, adaptive_n=True, prune_threshold=0.0,
               grow_residual_threshold=0.05, seed=0)
    cfg.update(kw)
    return cfg


class _GrowthBranchReached(Exception):
    """성장 분기에 들어갔다는 표지 — 이른 반환을 통과했을 때만 던져진다."""


def _updater_at(n_components, **cfg_kw):
    """성분 수 `n` 인 모델을 얹고, 성장 분기 진입을 예외로 관측할 수 있게 만든 updater."""
    upd = SequentialUpdater(_growth_cfg(**cfg_kw))
    upd.model = upd._build_model(n_components)
    with torch.no_grad():
        upd.model.w.fill_(1.0)          # 에너지 균등 ⟹ 가지치기가 아무것도 안 버린다

    def _boom(*_a, **_k):
        raise _GrowthBranchReached()

    upd._compute_residuals = _boom      # 이른 반환 **직후**의 첫 호출
    return upd


def _grow_branch_fires(upd):
    try:
        upd._adaptive_prune_grow([], (0.0, 1.0), (0.0, 1.0), verbose=False)
    except _GrowthBranchReached:
        return True
    return False


# ── ① 배포 거동 불변 ────────────────────────────────────────────────────────
def test_both_caps_ship_at_thirty_and_are_single_sourced():
    """분리는 **이름을 둘로** 했을 뿐 값을 바꾸지 않았다. 손복사도 없어야 한다."""
    assert POOL_CAP_DEFAULT == 30
    assert GROWTH_CEILING_DEFAULT == 30
    assert DEFAULT_CONFIG['max_n'] == 30
    raw = yaml.safe_load(open(_YAML))
    assert raw['adaptive']['max_n'] == 30, 'yaml 미러가 어긋났다'
    # ⛔yaml 에 새 키를 **값으로** 적어 두면 `max_n` 별칭이 죽는다(주석으로만 문서화한다).
    assert 'pool_cap' not in raw['adaptive']
    assert 'growth_ceiling' not in raw['adaptive']


def test_default_config_must_not_carry_the_new_keys():
    """★가장 조용한 회귀 경로. `DEFAULT_CONFIG` 에 `pool_cap`/`growth_ceiling` 을 넣으면
    `{**DEFAULT_CONFIG, 'max_n': 10}` 으로 정의된 **동결 배치 팔**에서 기본 30 이 10 을
    덮어써 기록된 실험의 의미가 바뀐다(`MAXN-10`·`MAXN-4`·`K-8`·`K-15`·`K-60`·`K-128`·
    `S1-A3` … 다섯 배치). 넣지 마라."""
    assert 'pool_cap' not in DEFAULT_CONFIG
    assert 'growth_ceiling' not in DEFAULT_CONFIG
    frozen_arm = {**DEFAULT_CONFIG, 'max_n': 10}
    assert resolve_pool_cap(frozen_arm) == 10
    assert resolve_growth_ceiling(frozen_arm) == 10


def test_pool_and_select_signature_default_is_still_max_n_thirty():
    """⛔동결 게이트가 이 **서명 기본값을 이름으로 되읽는다** —
    `tmp/B22_PRIOR_ABLATION_2026-08-27/gates_b22.py:101` ·
    `tmp/M4_CAPACITY_2026-08-28/gates_m4.py:72` (`SHIP-K` 기준선이 여기서 나온다).
    인자를 개명하거나 기본값을 바꾸면 그 배치들이 죽는다."""
    import inspect
    p = inspect.signature(pool_and_select).parameters
    assert 'max_n' in p, '동결 게이트가 이 인자 이름으로 되읽는다 — 개명 금지'
    assert p['max_n'].default == 30


def test_max_n_attribute_readback_still_works():
    """⛔동결 배치가 **객체를 되읽는다**: `getattr(upd, 'max_n', -1)`
    (`tmp/B2_REAL_2026-08-23/run_b2f.py:94`, `tmp/M4_CAPACITY_2026-08-28/run_m4.py:172`),
    그리고 `tmp/T18_FACTORIAL_2026-09-05/PREREG.md:493` 의 게이트 `G-MAXN` 이 그 값을
    배포값과 대조한다. 별칭은 **성장 천장**을 가리키고, 읽기/쓰기가 갈라지지 않는다."""
    upd = SequentialUpdater(dict(DEFAULT_CONFIG))
    assert getattr(upd, 'max_n', -1) == 30
    assert upd.growth_ceiling == 30
    upd.max_n = 7                       # 레거시 쓰기 경로
    assert upd.growth_ceiling == 7, '별칭 쓰기가 성장 천장에 닿지 않는다'


# ── ② 하위호환 = 2026-08-23 의 결합을 **보존**한다 ──────────────────────────
def test_legacy_max_n_still_couples_the_two_roles():
    """`max_n` 하나만 주면 지금도 **둘 다** 그 값이다.

    ★이것이 `RX-COMBINED`(`max_n=10`) 가 `MAXN-10` 과 비트 동일했던 그 성질이다
    (12/12, `tmp/B2_REAL_2026-08-23/RESULT_B2F.md`). 분리가 그 기록의 의미를 바꾸지
    않았음을 **거동으로** 확인한다: 풀링이 잘리고, 성장 분기는 죽은 채로 남는다."""
    cfg = _pool_cfg(max_n=5)
    assert resolve_pool_cap(cfg) == 5
    assert resolve_growth_ceiling(cfg) == 5
    *_, n = learned_init_params([], 30.0, cfg, seed=0)
    assert n == 5, '레거시 max_n 이 풀링 상한으로 안 먹었다'
    upd = _updater_at(5, max_n=5)
    assert not _grow_branch_fires(upd), (
        '레거시 max_n=5 에서 성장 분기가 살아났다 — 동결 배치 12/12 의 의미가 바뀐다')


# ── ③ ★재병합 방지 — 두 역할이 **거동으로** 갈린다 ─────────────────────────
def test_pooling_reads_pool_cap_and_ignores_growth_ceiling():
    """풀링이 성장 천장을 읽으면 원자 12개가 다 설치돼 5 가 아니게 된다."""
    *_, n = learned_init_params(
        [], 30.0, _pool_cfg(pool_cap=5, growth_ceiling=_N_ATOMS + 20), seed=0)
    assert n == 5, '풀링이 pool_cap 이 아닌 값을 읽었다 — 두 역할이 다시 합쳐졌다'


def test_growth_reads_growth_ceiling_and_ignores_pool_cap():
    """성장 천장이 풀링 상한을 읽으면 40 이 아니라 5 가 된다."""
    upd = SequentialUpdater(_growth_cfg(pool_cap=5, growth_ceiling=40))
    assert upd.growth_ceiling == 40, (
        '성장 천장이 pool_cap 을 읽었다 — 두 역할이 다시 합쳐졌다')
    assert upd.max_n == 40


def test_low_pool_cap_no_longer_kills_the_growth_branch():
    """★★분리의 요점. 2026-08-23 에는 **측정 불가**였던 조합이다.

    풀링 상한을 성분 수까지 내려도(`pool_cap=5`, 모델 성분 5개) 성장 천장이 40 이면
    성장 분기는 **살아 있어야** 한다. 다시 합쳐지면 이른 반환이 발화해 실패한다.
    """
    assert _grow_branch_fires(_updater_at(5, pool_cap=5, growth_ceiling=40))


def test_growth_ceiling_alone_still_stops_growth():
    """음성 대조 — 천장을 성분 수까지 내리면 분기는 죽어야 한다.
    (없으면 위 시험이 *"이른 반환이 아예 사라졌다"* 로도 통과한다 = 항진.)"""
    assert not _grow_branch_fires(_updater_at(5, pool_cap=40, growth_ceiling=5))


@pytest.mark.parametrize('cfg,expect_pool,expect_growth', [
    ({}, 30, 30),
    ({'max_n': 10}, 10, 10),
    ({'pool_cap': 4}, 4, 30),
    ({'growth_ceiling': 4}, 30, 4),
    # 구체적인 키가 레거시 별칭을 이긴다 (run.py 의 CLI 병합 순서와 같은 규약).
    ({'max_n': 10, 'pool_cap': 4}, 4, 10),
    ({'max_n': 10, 'growth_ceiling': 4}, 10, 4),
])
def test_resolution_priority(cfg, expect_pool, expect_growth):
    assert resolve_pool_cap(cfg) == expect_pool
    assert resolve_growth_ceiling(cfg) == expect_growth
