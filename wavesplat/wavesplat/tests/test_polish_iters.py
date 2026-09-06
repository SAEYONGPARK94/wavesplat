"""★`learned_polish_iters` (2026-09-06) 회귀 시험 — **배포 기본값이 움직이면 여기서 죽는다.**

⛔이 파일이 있는 이유. 제안 원자는 확산망 출력 뒤에 `polish_atoms`(Adam, `lr=0.02`)로
다듬어지는데, 그 **스텝 수가 `propose_atoms` 의 서명 기본값에만 존재하고 config 로 도달할 수
없었다**. 2026-09-06 프로브가 그 단계를 측정했다
(`tmp/T22_APLADDER_2026-09-06/PROBE_4.md`·`PROBE_5.md`, 추정기 적합 0회):

  · 확산 원출력은 장면 파수를 기울기 **0.79** 로 따라가는데 80스텝 뒤에는 **0.46** 이다.
  · 다듬기는 **80/80 장면**에서 파수를 키우고(단측 p=8.3e-25), **320스텝까지 수렴하지 않으며**
    그 표류가 **긴 파도에 집중**된다.
  · ⚠단 배포 선택기(`pool_and_select`)가 그 대부분을 흡수한다(`PROBE_6.md`, 20/30 p=0.049).

⟹ 그것을 **처치하려면 팔이 패키지 안에 있어야 한다** — 하네스 연산자로 재면 `D3` 토론의
가장 강한 반론(*"측정된 연산자가 패키지에 없다"*)에 그대로 걸린다.

★**이 시험은 값이 아니라 거동을 본다.** 문자열 검사는 항진한다 —
`'real_gate' in getsource` 가 2026-09-01 에 그렇게 통과했다.

⛔**기본값 80 은 배포값이다.** `T3`·`T5-ENS`·`T17` 앵커와 모든 동결 배치가 그 값에서 측정됐다.
이 수를 바꾸면 그 앵커들이 전부 무효다.
"""
import inspect

import pytest

from wavesplat.optimizer import learned_init as LI


def test_default_equals_propose_atoms_signature():
    """★두 곳에 적힌 같은 상수가 어긋나면 죽는다 (이 저장소의 반복 결함 부류)."""
    sig = inspect.signature(LI.propose_atoms).parameters['polish_iters'].default
    assert LI.POLISH_ITERS_DEFAULT == sig == 80, (
        f'POLISH_ITERS_DEFAULT={LI.POLISH_ITERS_DEFAULT} vs '
        f'propose_atoms 서명 기본값={sig} — 단일출처가 깨졌다')


def _spy(monkeypatch):
    """`propose_atoms` 를 가로채 **실제로 전달된** polish_iters 를 되읽는다."""
    seen = {}

    def fake(frames, depth, ckpt=None, nsamp=4, diffusion_seed=0,
             device=None, polish_iters=LI.POLISH_ITERS_DEFAULT, subsample_seed=0):
        seen['polish_iters'] = polish_iters
        seen['nsamp'] = nsamp
        # 잘 분리된 원자 4개 — 풀링이 상한 말고 다른 이유로 자르지 않도록.
        import numpy as np
        return (np.array([0.05, 0.20, 0.35, 0.50]),
                np.array([0.00, 0.10, -0.10, 0.20]),
                np.array([1.0, 0.9, 0.8, 0.7]))

    monkeypatch.setattr(LI, 'propose_atoms', fake)
    return seen


class _F:
    """`ls_fit_wphi` 가 만질 수 있는 최소 프레임 하나."""
    def __init__(self):
        import torch
        self.t = 0.0
        self.x_obs = torch.linspace(-50.0, 50.0, 32)
        self.y_obs = torch.zeros(32)
        self.z_obs = torch.zeros(32)
        self.confidence = torch.ones(32)
        self.incidence_angles = torch.zeros(32)


def test_absent_key_forwards_the_shipped_default(monkeypatch):
    """★키를 안 주면 배포값 80 이 그대로 간다 = 기존 런과 비트 동일."""
    seen = _spy(monkeypatch)
    LI.learned_init_params([_F()], 50.0, {}, seed=0)
    assert seen['polish_iters'] == 80


def test_key_is_actually_forwarded(monkeypatch):
    """★값이 **전달되는지**를 본다 — 조용히 무시되면 팔이 장식이 된다."""
    for want in (0, 20, 320):
        seen = _spy(monkeypatch)
        LI.learned_init_params([_F()], 50.0,
                               {'learned_polish_iters': want}, seed=0)
        assert seen['polish_iters'] == want, (
            f'learned_polish_iters={want} 을 줬는데 {seen["polish_iters"]} 가 전달됐다')


def test_negative_is_rejected(monkeypatch):
    _spy(monkeypatch)
    with pytest.raises(ValueError):
        LI.learned_init_params([_F()], 50.0, {'learned_polish_iters': -1}, seed=0)


def test_cached_atoms_path_does_not_call_the_proposal(monkeypatch):
    """⛔`learned_atoms` 캐시 경로에서는 다듬기가 애초에 일어나지 않는다 —
    그 경로로 이 팔을 재면 **처치가 no-op** 이다. 배치가 그것을 알아야 한다."""
    seen = _spy(monkeypatch)
    LI.learned_init_params([_F()], 50.0, {
        'learned_atoms': {'kx': [0.05, 0.2], 'ky': [0.0, 0.1], 'energy': [1.0, 0.9]},
        'learned_polish_iters': 0}, seed=0)
    assert 'polish_iters' not in seen, '캐시 경로가 제안을 불렀다 — 전제가 깨졌다'


# ── ★출하 `auto` 경로 (2026-09-06 정정) ───────────────────────────────────────
# ⛔초판은 `learned_init_params` 에만 키를 붙였다. 그런데 출하 기본값은 `init_method='auto'`
#   이고, 라우터 `_auto_select_init` 이 **자기가** `propose_atoms` 를 돌린 뒤
#   `cfg['learned_atoms']` 에 캐시한다 ⟹ `learned_init_params` 는 캐시 분기를 타고
#   그 키를 **영영 읽지 않는다** ⟹ 처치가 출하 경로에서 **조용히 no-op**.
#   ★`D3`(2026-09-01)와 같은 부류: *"생성 시점만 보고 라우터 해소 이후를 안 봤다."*
#   적대적 심사(2026-09-06)가 BLOCKER-1 로 잡았다. 아래가 그 회귀 시험이다.

def _router_spy(monkeypatch):
    """라우터가 부르는 `propose_atoms` 를 가로챈다 — `sequential_update` 는
    함수 안에서 `from .learned_init import propose_atoms` 하므로 **원본 모듈**을 패치한다."""
    seen = {}
    import numpy as np

    def fake(frames, depth, ckpt=None, nsamp=4, diffusion_seed=0,
             device=None, polish_iters=LI.POLISH_ITERS_DEFAULT, subsample_seed=0):
        seen.setdefault('calls', []).append(polish_iters)
        return (np.array([0.05, 0.20, 0.35, 0.50]),
                np.array([0.00, 0.10, -0.10, 0.20]),
                np.array([1.0, 0.9, 0.8, 0.7]))

    monkeypatch.setattr(LI, 'propose_atoms', fake)
    return seen


@pytest.mark.parametrize('want', [0, 20, 320])
def test_auto_router_forwards_polish_iters(monkeypatch, want):
    """★출하 `auto` 경로에서 **실제로 전달되는지**. 이것이 없으면 팔이 장식이다."""
    from wavesplat.optimizer.sequential_update import SequentialUpdater
    seen = _router_spy(monkeypatch)
    upd = SequentialUpdater({'init_method': 'auto', 'auto_switch_thresh': 0,
                             'real_domain_gate': False, 'depth': 50.0,
                             'learned_polish_iters': want, 'seed': 0})
    upd._auto_select_init([_F()])
    assert seen.get('calls'), '라우터가 제안을 부르지 않았다 — 전제가 깨졌다'
    assert seen['calls'] == [want], (
        f"auto 라우터가 {want} 대신 {seen['calls']} 를 넘겼다 "
        f"— 처치가 출하 경로에서 no-op 이다")


def test_auto_router_default_is_the_shipped_value(monkeypatch):
    """★키가 없으면 배포값 80 = 종전 런과 비트 동일."""
    from wavesplat.optimizer.sequential_update import SequentialUpdater
    seen = _router_spy(monkeypatch)
    upd = SequentialUpdater({'init_method': 'auto', 'auto_switch_thresh': 0,
                             'real_domain_gate': False, 'depth': 50.0, 'seed': 0})
    upd._auto_select_init([_F()])
    assert seen['calls'] == [80]


def test_auto_router_rejects_negative(monkeypatch):
    from wavesplat.optimizer.sequential_update import SequentialUpdater
    _router_spy(monkeypatch)
    upd = SequentialUpdater({'init_method': 'auto', 'auto_switch_thresh': 0,
                             'real_domain_gate': False, 'depth': 50.0,
                             'learned_polish_iters': -1, 'seed': 0})
    with pytest.raises(ValueError):
        upd._auto_select_init([_F()])
