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


def test_explicit_knob_reaches_the_wrapper():
    cfg = {'init_method': 'learned'}
    cfg.update(recommended_learned_ensemble_config())
    upd = make_updater(cfg)
    assert isinstance(upd, MedianEnsembleUpdater)
    assert upd.k_requested == 3
    assert upd.k_key == 'learned_ensemble_k'


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


def test_real_domain_gate_still_suppresses_the_learned_ensemble():
    """실측 게이트는 여전히 두 앙상블을 모두 누른다 — bank 가 자체 k=3 을 돌린다.

    ⛔`T3`/`T5` 는 `real_domain_gate=False` 로 쟀다. 그 구성은 **연구 팔**이지 배포가 아니다.
    """
    src = inspect.getsource(MedianEnsembleUpdater.initialize)
    assert 'real_gate' in src
    assert 'or real_gate' in src, '실측 게이트가 접기 조건에서 빠졌다'
