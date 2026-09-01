"""학습 경로 median-of-k 경로 — **기본값 OFF** 를 핀한다 (2026-09-01, 결정 `D3`).

★이 시험이 지키는 것은 *"앙상블이 좋다"* 가 아니라 ***"측정된 팔이 패키지 안에 존재하고,
그리고 기본값은 켜져 있지 않다"*** 이다.

배경: `T3-NOW`(29/36 · −8.65%)와 `T5-ENS`(NEAR 28/36 · FAR 35/36)가 학습 경로 앙상블을
**확증**했지만 **둘 다 패키지가 소유하지 않은 연산자**로 쟀다(행의 `ens_k_readback` 전부 `None`).
그 상태로 기본값을 바꾸는 것은 2026-08-23 사고(`bank_refit_every_s` 당일 철회)의 재연이다.
⟹ 경로는 만들고 기본값은 1 로 둔다. 켜는 길은 `recommended_learned_ensemble_config()`.
"""
import ast
import inspect

from wavesplat.optimizer.sequential_update import (
    DEFAULT_ANALYTIC_ENSEMBLE_K,
    DEFAULT_LEARNED_ENSEMBLE_K,
    MedianEnsembleUpdater,
    SequentialUpdater,
    make_updater,
    recommended_learned_ensemble_config,
)


def test_learned_ensemble_is_off_by_default():
    """★기본값은 **1(OFF)** 이다. 이 한 줄이 D3 의 결정 그 자체다."""
    assert DEFAULT_LEARNED_ENSEMBLE_K == 1


def test_plain_learned_is_not_wrapped():
    """기본 구성의 `learned` 는 감싸이지 않는다 ⟹ 기록된 모든 배치가 불변이다."""
    upd = make_updater({'init_method': 'learned'})
    assert type(upd) is SequentialUpdater


def test_recommended_config_keeps_the_real_domain_gate():
    """★권고 구성은 `auto` 를 유지한다 --- 그래야 실측 게이트가 살아 있다.

    ⛔초판은 `init_method='learned'` 를 돌려줬고, 명시 `learned` 는 실측 게이트 분기를
    **아예 통과하지 않는다** ⟹ 권고를 실측에 쓰면 **게이트 없이** 학습 멤버가 돌았다
    (`W-12` 가 금지한 팔). 독립 감사가 잡았다.
    """
    from wavesplat.run import DEFAULT_CONFIG

    rec = recommended_learned_ensemble_config()
    assert rec['init_method'] == 'auto', '권고가 실측 게이트를 우회한다'

    cfg = dict(DEFAULT_CONFIG)
    cfg.update(rec)
    upd = make_updater(cfg)
    assert isinstance(upd, MedianEnsembleUpdater)
    # 합성(게이트 미발동)에서는 켜지고
    assert upd._effective_k('learned', real_gate=False) == 3
    # ★실측(게이트 발동)에서는 꺼진다
    assert upd._effective_k('learned', real_gate=True) == 1


def test_knob_is_validated_wherever_it_is_read():
    """★결정을 한 곳으로 모았으면 **검증도 거기** 있어야 한다.

    ⛔초판은 `__init__` 에서만 검증해서, `auto` 로 들어온 `learned_ensemble_k=99` 가
    **어디서도 검증되지 않고** 조용히 5 로 잘렸다(요청 99 · 실제 5 · 경고 0).
    """
    import pytest
    from wavesplat.run import DEFAULT_CONFIG

    for bad in (99, 0, -1, None, 'three'):
        cfg = dict(DEFAULT_CONFIG)
        cfg['learned_ensemble_k'] = bad
        with pytest.raises(ValueError):
            MedianEnsembleUpdater(cfg)._effective_k('learned', real_gate=False)


def test_the_two_ensemble_knobs_do_not_leak_into_each_other():
    """★두 상수는 배포 이력이 다르다 — 병합하면 조용히 동작이 바뀐다.

    `analytic_ensemble_k` 는 2026-07-27 배포(k=3), `learned_ensemble_k` 는 2026-09-01
    신설(k=1, OFF). 한쪽을 켜도 다른 쪽 경로가 켜지면 안 된다.
    """
    # 해석적 노브는 learned 경로를 켜지 못한다
    upd = make_updater({'init_method': 'learned', 'analytic_ensemble_k': 5})
    assert type(upd) is SequentialUpdater, 'analytic 노브가 learned 를 켰다'

    # 학습 노브는 v2_konly 를 감싸지 못한다(명시 init_method 는 평 updater 다)
    upd = make_updater({'init_method': 'v2_konly', 'learned_ensemble_k': 3})
    assert type(upd) is SequentialUpdater, 'learned 노브가 v2_konly 를 켰다'

    # auto 경로는 여전히 해석적 노브만 읽는다
    upd = make_updater({'init_method': 'auto'})
    assert isinstance(upd, MedianEnsembleUpdater)
    assert upd.k_key == 'analytic_ensemble_k'
    assert upd.k_requested == DEFAULT_ANALYTIC_ENSEMBLE_K


def test_learned_members_differ_only_in_the_diffusion_draw():
    """★멤버 규약을 소스에서 핀한다 — `seed` 고정 + `learned_diffusion_seed` 만 오프셋.

    ⛔이 규약이 확증 배치(`T3`/`T5`)가 측정한 것이다. `seed` 를 오프셋하면 주변 RNG 스트림이
    함께 움직여 **다른 팔**이 된다(그리고 멤버 0 의 비트 항등이 깨진다).
    ⚠`learned_atoms` 를 물려받으면 멤버 전부가 **같은 추첨 하나**를 재사용한다.
    """
    src = inspect.getsource(MedianEnsembleUpdater.initialize)
    tree = ast.parse(src.strip())
    assigns = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Constant):
                    assigns.append(t.slice.value)
    assert 'learned_diffusion_seed' in assigns, '멤버가 확산 추첨으로 갈리지 않는다'
    assert "cfg.pop('learned_atoms', None)" in src, 'learned_atoms 캐시 공유를 막지 않는다'
    assert "cfg['init_method'] = 'learned'" in src, '학습 멤버가 learned 로 초기화되지 않는다'


def test_initialize_delegates_k_to_the_single_resolver():
    """★멤버 수 결정은 **한 곳**에서만 일어난다.

    ⛔초판은 `initialize` 안에서 직접 계산했고, 그래서 `auto`→`learned` 해소 경로를
    놓쳤다(아래 회귀 시험 참조). 지금은 `_effective_k` 하나가 경계이고 `initialize` 는
    그것을 **호출만** 한다 --- 두 번째 구현이 생기면 이 시험이 깨진다.
    """
    src = inspect.getsource(MedianEnsembleUpdater.initialize)
    assert 'self._effective_k(route, real_gate)' in src, \
        'initialize 가 단일 결정자를 쓰지 않는다'
    assert 'learned_ensemble_k' not in src, \
        'initialize 가 노브를 다시 읽는다 = 같은 기능 두 곳 구현'


# ⛔초판의 `assert 'real_gate' in getsource(_effective_k)` 는 **항진이었다** —
#   그 이름은 시그니처에 있으므로 본문의 게이트 분기를 통째로 지워도 통과한다.
#   독립 감사가 돌연변이로 확인했다. ⟹ 문자열이 아니라 **행동**으로 검사한다.


# ─────────────────────────────────────────────────────────────────────────────
# ★아래 둘은 **초판이 통과시킨 결함**을 잡는다.
#   위의 시험 6개는 전부 통과하는 동안, 배포 기본 구성(`auto` + 임계 0 ⟹ 항상 `learned`)이
#   `analytic_ensemble_k=3` 을 물려받아 **조용히 학습 멤버 3개를 만들고 있었다.**
#   사유: 시험이 전부 **생성 시점**만 봤고 **라우터 해소 이후**를 보지 않았다.
#   ⟹ 교훈: 「기본값이 꺼져 있다」는 **해소된 route 위에서** 확인해야 한다.
# ─────────────────────────────────────────────────────────────────────────────
def test_shipped_default_gets_no_learned_ensemble_after_the_router_resolves():
    """★출하 기본 구성에서 라우터가 `learned` 로 가도 멤버는 1개다."""
    from wavesplat.run import DEFAULT_CONFIG

    upd = MedianEnsembleUpdater(dict(DEFAULT_CONFIG))
    # 라우터가 learned 로 해소한 뒤의 유효 멤버 수
    assert upd._effective_k('learned', real_gate=False) == 1, \
        '배포 기본값이 학습 앙상블을 얻었다 — D3 가 하지 않기로 한 것이다'
    # 해석적으로 해소하면 배포된 대로 k=3 이다(2026-07-27 이후 불변)
    assert upd._effective_k('v2_konly', real_gate=False) == DEFAULT_ANALYTIC_ENSEMBLE_K


def test_real_domain_gate_suppresses_both_ensembles():
    from wavesplat.run import DEFAULT_CONFIG

    cfg = dict(DEFAULT_CONFIG)
    cfg['learned_ensemble_k'] = 3
    upd = MedianEnsembleUpdater(cfg)
    assert upd._effective_k('learned', real_gate=True) == 1
    assert upd._effective_k('v2_konly', real_gate=True) == 1


def test_explicit_knob_survives_the_router():
    """`auto` 로 들어와 `learned` 로 해소돼도, 노브를 명시하면 그 값이 쓰인다."""
    from wavesplat.run import DEFAULT_CONFIG

    cfg = dict(DEFAULT_CONFIG)
    cfg['learned_ensemble_k'] = 3
    assert MedianEnsembleUpdater(cfg)._effective_k('learned', real_gate=False) == 3


def test_unknown_route_never_ensembles():
    from wavesplat.run import DEFAULT_CONFIG

    upd = MedianEnsembleUpdater(dict(DEFAULT_CONFIG))
    for route in ('bank_mmse', 'v2', None, 'learned_konly'):
        assert upd._effective_k(route, real_gate=False) == 1, route
