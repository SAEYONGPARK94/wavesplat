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


def test_pre_varpro_config_flips_both_switches():
    """★배포를 되돌리는 스위치는 하나가 아니라 둘이다.

    2026-08-22 감사(R2-3)가 잡은 지뢰: `varpro_steps=0` 하나만 끄면 배포 전이 아니라
    W4 의 `L-ITER0` 팔이 되고, 그것은 배포 전보다 **나쁘다**(7/64, +23.34%).
    문서 세 곳이 그 거짓을 적고 있었다. 이 시험이 그 재발을 막는다.
    """
    from wavesplat.optimizer.sequential_update import (pre_varpro_config,
                                                       LEARNED_INIT_ITERS_PRE_VARPRO)
    cfg = pre_varpro_config()
    assert cfg == {'varpro_steps': 0,
                   'learned_init_iters': LEARNED_INIT_ITERS_PRE_VARPRO}
    assert LEARNED_INIT_ITERS_PRE_VARPRO == 500
    # 되돌림은 두 키를 **함께** 바꿔야 한다.
    assert len(cfg) == 2


def test_varpro_design_matrix_matches_shipped_forward_model():
    """T1 (원 selftest 에서 이식) — VARPRO 설계행렬이 출하 전방모델과 같은 장을 낸다.

    원본 `tmp/W4_SOLVER_2026-08-19/varpro.py` 의 selftest 8검사는 패키지 이식 때
    통째로 빠졌다(R2-3 지적). 그중 하중이 가장 큰 둘(T1·T3)을 여기에 되살린다.
    이 불변식이 깨지면 VARPRO 는 **출하 모델이 아닌 것을 최적화한다**.
    """
    import torch
    from wavesplat.model.spectral_gaussian import SpectralGaussianModel
    from wavesplat.optimizer import varpro as V

    torch.manual_seed(0)
    n, depth = 5, 50.0
    kx = torch.rand(n, dtype=torch.float64) * 0.2 + 0.02
    ky = (torch.rand(n, dtype=torch.float64) - 0.5) * 0.2
    w = torch.rand(n, dtype=torch.float64) * 0.5 + 0.1
    phi = (torch.rand(n, dtype=torch.float64) - 0.5) * 6.0
    x = torch.rand(40, dtype=torch.float64) * 100.0 + 10.0
    y = (torch.rand(40, dtype=torch.float64) - 0.5) * 100.0
    t = 3.25

    model = SpectralGaussianModel(n, depth=depth)
    with torch.no_grad():
        model.kx.copy_(kx.float()); model.ky.copy_(ky.float())
        model.w.copy_(w.float()); model.phi.copy_(phi.float())
    shipped = model(x.float(), y.float(), t).double()

    Phi, _ = V.design_matrix(kx, ky, x, y, torch.full_like(x, t), depth)
    a, b = V.wphi_to_ab(w, phi)
    vp = Phi @ torch.cat([a, b])
    assert torch.allclose(shipped, vp, atol=2e-5), \
        f'max|diff| = {float(torch.max(torch.abs(shipped - vp))):.3e}'


def test_varpro_negative_control_zero_steps_is_identity():
    """T3 (이식) — steps=0 이면 VARPRO 가 파라미터를 건드리지 않아야 한다."""
    import torch
    from wavesplat.optimizer import varpro as V
    torch.manual_seed(1)
    kx0 = torch.rand(4, dtype=torch.float64) * 0.2 + 0.02
    ky0 = (torch.rand(4, dtype=torch.float64) - 0.5) * 0.2
    x = torch.rand(30, dtype=torch.float64) * 100 + 10
    y = (torch.rand(30, dtype=torch.float64) - 0.5) * 100
    t = torch.zeros(30, dtype=torch.float64)
    z = torch.randn(30, dtype=torch.float64) * 0.1
    conf = torch.ones(30, dtype=torch.float64)
    kx1, ky1, _a, _b, info = V.varpro_refine(kx0, ky0, x, y, t, z, conf, 50.0,
                                            steps=0)
    assert torch.equal(kx0, kx1) and torch.equal(ky0, ky1)
    assert info['k_shift_max'] == 0.0


def test_unifying_via_init_method_default_would_lose_the_real_gate():
    """★2026-08-22 — **통일 방법을 잘못 고르면 실측 게이트가 사라진다**는 것을 고정한다.

    실측 게이트는 `init_method == 'auto'` 안에만 있고, 그것은 의도된 계약이다
    (명시적 선택 = 연구자의 의도적 override; `test_bank_mmse_port.py` 참조).
    ⛔따라서 "한 방법으로 통일" 을 `init_method` 기본값 변경으로 하면 실측이 학습 경로로 간다.
    측정된 결과(W12): 예보 지평에서 제로필드를 **0/5** 로 진다.
    ✅안전한 통일 = 기본값을 'auto' 로 두고 라우터 임계를 0 으로.

    이 시험은 **현재 거동을 바꾸지 않는다** — 함정이 실재함을 기록하고, 누군가
    기본값을 바꿨을 때 여기서 걸리게 한다.
    """
    import torch
    from wavesplat.optimizer.sequential_update import SequentialUpdater
    from wavesplat.data_loader import FrameData
    from wavesplat.run import DEFAULT_CONFIG

    # (1) 배포 기본값은 'auto' 여야 한다 — 이것이 게이트가 작동하는 유일한 조건이다.
    assert DEFAULT_CONFIG['init_method'] == 'auto', (
        "배포 기본값이 'auto' 가 아니면 실측 게이트가 발화하지 않는다. "
        "통일이 목적이라면 auto_switch_thresh 를 쓰라 (W11/W12 참조).")

    def gt_free(n=3, m=64):
        g = torch.Generator().manual_seed(0)
        return [FrameData(t=float(i), x_obs=torch.rand(m, generator=g) * 100 + 10,
                          y_obs=(torch.rand(m, generator=g) - 0.5) * 100,
                          z_obs=torch.randn(m, generator=g) * 0.1,
                          confidence=torch.ones(m),
                          incidence_angles=torch.zeros(m)) for i in range(n)]

    # (2) 'auto' 에서는 게이트가 실측을 잡는다.
    u = SequentialUpdater({'init_method': 'auto', 'depth': 50.0})
    u.initialize(gt_free(), x_range=(10.0, 110.0), y_range=(-50.0, 50.0))
    assert u._auto_real_gate is True
    assert u.init_method != 'learned'


# ── 경로 통일 회귀 (2026-08-22) ────────────────────────────────────────────────

def test_router_threshold_is_single_sourced_and_unified():
    """★합성 경로 통일: 라우터 임계 0 = 항상 학습.

    근거(같은 B 코퍼스 128ep, 강제 두 팔, median-of-3):
    학습 90/128 · p=4.9e-06 · 상대 -38.5% · 제로필드 이김 128/128 vs 100/128.
    """
    import yaml, os
    from wavesplat.optimizer.sequential_update import (AUTO_SWITCH_THRESH_DEFAULT,
                                                       AUTO_SWITCH_THRESH_PRE_UNIFY)
    from wavesplat.run import DEFAULT_CONFIG
    assert DEFAULT_CONFIG['auto_switch_thresh'] is AUTO_SWITCH_THRESH_DEFAULT
    assert AUTO_SWITCH_THRESH_DEFAULT == 0
    assert AUTO_SWITCH_THRESH_PRE_UNIFY == 8          # 되돌리기 값 고정
    here = os.path.dirname(os.path.abspath(__file__))
    opt = yaml.safe_load(open(os.path.join(here, '..', '..', 'configs',
                                           'estimation.yaml')))['optimizer']
    assert opt['auto_switch_thresh'] == AUTO_SWITCH_THRESH_DEFAULT


def test_unification_did_not_disable_the_real_gate():
    """★★통일했어도 실측 게이트는 살아 있어야 한다.

    통일을 `init_method` 기본값 변경으로 했다면 게이트가 사라졌을 것이다.
    임계로 한 이유가 이것이고, 이 시험이 그 성질을 고정한다.
    W12 측정: 실측에 학습을 강제하면 예보 지평에서 제로필드에 0/5 패배.
    """
    import torch
    from wavesplat.optimizer.sequential_update import SequentialUpdater
    from wavesplat.data_loader import FrameData
    from wavesplat.run import DEFAULT_CONFIG

    assert DEFAULT_CONFIG['init_method'] == 'auto'     # 게이트가 사는 유일한 조건
    g = torch.Generator().manual_seed(0)
    frames = [FrameData(t=float(i), x_obs=torch.rand(64, generator=g) * 100 + 10,
                        y_obs=(torch.rand(64, generator=g) - 0.5) * 100,
                        z_obs=torch.randn(64, generator=g) * 0.1,
                        confidence=torch.ones(64),
                        incidence_angles=torch.zeros(64)) for i in range(3)]
    u = SequentialUpdater(dict(DEFAULT_CONFIG, depth=50.0))
    u.initialize(frames, x_range=(10.0, 110.0), y_range=(-50.0, 50.0))
    assert u._auto_real_gate is True, '통일이 실측 게이트를 껐다'
    assert u.init_method != 'learned', '실측이 학습 경로로 갔다'


def test_shell_metrics_are_aggregated_into_the_summary():
    """★2026-08-22 — 거리(셸) 분해가 요약에 남아야 한다.

    `run.py` 가 프레임마다 `shell_rmse`/`shell_frac` 를 채우는데 요약이 집계하지 않아
    저장된 결과 파일이 **거리 분해에 구조적으로 눈이 멀어 있었다** —
    `RMSE_near` 가 46일간 당한 것과 같은 결함이고, 2026-08-19 에는 그것만 고쳤다.
    2026-08-22 검수가 네 배치에서 shell 키 0회를 확인해 재발을 잡았다.
    ⚠이 셸 접두어는 `obs:`(최근접 관측점 거리)이지 배-거리가 아니다.
    """
    import inspect
    from wavesplat import run as R
    src = inspect.getsource(R.run_estimation)
    assert "'avg_shell_rmse'" in src, '요약이 셸 RMSE 를 집계하지 않는다'
    assert "'shell_frac'" in src, '요약이 셸 격자 비중을 기록하지 않는다'
    # 집계가 frame_results 를 실제로 읽어야 한다 (공허한 키가 아니어야 한다)
    assert "r['shell_rmse'][k]" in src
