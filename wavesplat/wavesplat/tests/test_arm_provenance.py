"""★팔 정체는 **의도가 아니라 되읽기**로 적어야 한다 (2026-09-03).

왜 이 파일이 있는가
-------------------
2026-09-03 에 배치 `E1-EXTBASE` 가 저장소 최초의 외부 기준선 행을 냈다. 그 배치의
"우리 팔" 은 `gates_b2.ARMS['SHIP'] = {}` --- **빈 dict** 이었다. 실측(참값 없는)
입력에서 라우터는 **무조건** ported bank-MMSE 로 보내므로, 실제로 비교된 것은
**배포 비교자**이지 논문이 제안하는 학습 사전이 아니었다.

그런데 그 사실이 **결과 행에도, 논문에도, 학위논문 6곳에도 한 번도 적히지 않았다.**
박사 심사 관점의 적대적 감사가 그것을 찾았고, 두 원고를 고쳐야 했다.

⛔**라우팅 동작은 바꾸지 않는다** --- 그것은 측정된 판정이다(`W-12`: 실측에서 학습을
강제하면 예보 지평 0/5; bank 가 v2_konly 를 3/3 창에서 이긴다). 바꾸는 것은
**추적 가능성**이고, 이 시험이 그것을 고정한다.

사용자 지시(2026-09-03, 축자): *"Bank mmse는 코드에서 안타게 분리하자 자꾸 헷갈려하네"*.
⟹ 「안 타게」의 안전한 형태는 **동작 제거가 아니라 정체 노출**이다 --- 제거하면
측정된 판정을 뒤집고 `D3` BLOCKER(인가된 경로가 실측 게이트를 껐다)를 재발시킨다.
"""

import numpy as np
import pytest
import torch

from wavesplat.data_loader import FrameData
from wavesplat.optimizer.sequential_update import SequentialUpdater, make_updater


def _real_frames(n=6, m=60, seed=0):
    """참값 없는(관측만) 프레임 --- 실측 게이트가 보는 형태.

    ⛔`gt_field=None` 이 「참값 없음」의 정의다(그것이 게이트를 발화시킨다).
    """
    rng = np.random.default_rng(seed)
    out = []
    for j in range(n):
        x = rng.uniform(-20, 20, m); y = rng.uniform(-20, 20, m)
        t = float(j) * 0.1
        z = 0.3 * np.cos(0.2 * x + 0.1 * y - 0.8 * t)
        out.append(FrameData(
            t=t,
            x_obs=torch.tensor(x, dtype=torch.float32),
            y_obs=torch.tensor(y, dtype=torch.float32),
            z_obs=torch.tensor(z, dtype=torch.float32),
            confidence=torch.ones(m, dtype=torch.float32),
            incidence_angles=torch.zeros(m, dtype=torch.float32),
            gt_field=None))
    return out


def test_provenance_keys_exist():
    u = SequentialUpdater({'init_method': 'auto', 'depth': 30.0})
    p = u.arm_provenance()
    for k in ('requested_init_method', 'resolved_init_method',
              'real_domain_gate_fired', 'auto_selected', 'real_estimator',
              'is_bank_mmse'):
        assert k in p, f'arm_provenance() 가 {k} 를 빠뜨렸다'


def test_empty_arm_on_real_input_resolves_to_bank_mmse_and_says_so():
    """★핵심: **빈 dict 팔**이 실측 입력에서 무엇이 되는지 되읽기로 드러나야 한다."""
    from wavesplat.run import DEFAULT_CONFIG
    cfg = dict(DEFAULT_CONFIG)
    cfg.update({})                      # ⛔`E1` 이 한 것과 같은 "빈 dict 팔"
    cfg.update(depth=30.0, seed=42, device='cpu')
    u = make_updater(cfg)
    with pytest.warns(RuntimeWarning):
        u.initialize(_real_frames(), x_range=(-20, 20), y_range=(-20, 20),
                     verbose=False)
    p = u.arm_provenance()
    assert p['real_domain_gate_fired'] is True, '실측 게이트가 발화하지 않았다'
    assert p['resolved_init_method'] == 'bank_mmse', (
        f"실측 입력의 빈 dict 팔은 bank_mmse 로 해소돼야 한다 (실제 {p['resolved_init_method']!r})")
    assert p['is_bank_mmse'] is True
    # ★그리고 **의도와 다르다**는 것이 드러나야 한다 --- 이것이 오늘의 사고였다.
    assert p['requested_init_method'] != p['resolved_init_method'], (
        '의도와 실제가 같아 보이면 배치가 팔 정체를 놓친다')


def test_gate_escape_hatch_is_load_bearing_on_real_input():
    """⛔대조: 실측 게이트를 **의도적으로** 끈 팔은 bank 가 아니어야 한다.

    ⛔[2026-09-03 재검이 초판을 REFUTE 했다] 초판 이름은
      `test_deliberate_learned_arm_is_not_bank_mmse` 였는데 **learned 를 한 번도 타지 않고**
      `v2_konly` 를 지정했다. 게다가 계측해 보니 그 런에서 `real_domain_gate` 키를
      **0회** 읽었다 ⟹ 탈출구를 통째로 지워도 통과하는 **항진 시험**이었다.
    ★수리: 이름을 실제로 시험하는 것에 맞추고, **탈출구가 하중을 받는지**를
      두 구성의 **행동 차이**로 검사한다(문자열 검사 아님 --- 오늘 세 번째 교훈).
    """
    from wavesplat.run import DEFAULT_CONFIG
    base = dict(DEFAULT_CONFIG)
    base.update(depth=30.0, seed=42, device='cpu', init_method='auto')

    # (a) 탈출구 없이 --- 실측 입력이면 게이트가 bank 로 보낸다.
    a = make_updater(dict(base))
    a.initialize(_real_frames(), x_range=(-20, 20), y_range=(-20, 20), verbose=False)
    pa = a.arm_provenance()
    assert pa['is_bank_mmse'] is True, f'게이트가 실측에서 bank 로 보내야 한다 ({pa})'
    assert pa['real_domain_gate_fired'] is True

    # (b) 탈출구를 켜면 --- **같은 입력·같은 config** 인데 경로가 달라져야 한다.
    b = make_updater(dict(base, real_domain_gate=False, init_method='v2_konly'))
    b.initialize(_real_frames(), x_range=(-20, 20), y_range=(-20, 20), verbose=False)
    pb = b.arm_provenance()
    assert pb['is_bank_mmse'] is False, f'탈출구가 하중을 받지 않는다 ({pb})'
    assert pb['real_domain_gate_fired'] is False
    assert pa['resolved_init_method'] != pb['resolved_init_method'], (
        '두 구성이 같은 경로로 끝나면 이 시험은 항진이다')
