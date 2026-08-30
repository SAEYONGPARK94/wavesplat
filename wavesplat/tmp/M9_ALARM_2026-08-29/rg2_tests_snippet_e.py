"""RG-2 제품 회귀 시험 — ⛔지금 실행하지 마라 (수리 미착지 상태에서는 (a)(d)가 죽는 게 정상).

★수리 커밋에서 `wavesplat/tests/test_bank_mmse_port.py` **E절 말미로 이동**해 커밋하라
(PREREG §3 RG-2·§5-3). 파일명이 test_* 가 아니므로 이 위치에서는 pytest 가 수집하지
않는다. 이동 시 그 파일의 헬퍼(`_real_frames`, `_bank_cfg`)와 상단 import
(`pytest`, `torch`, `warnings`, `FrameData`)를 그대로 쓴다 — 여기 정의된 `_alarm_msgs`
헬퍼만 함께 옮긴다.

⚠PREREG §5-3 의 취약점 처방: 모든 시험이 **명시 warnings 문맥**을 쓴다 — 훗날 누가
`filterwarnings=error` 를 pytest.ini 에 넣어도 이 시험들은 깨지지 않는다.
경보 문안 매칭 토큰 'bank_refit_async' 는 gates_m9.ALARM_WARN_TOKEN 과 같은 동결
인터페이스다(수리 후 문안도 이 토큰을 포함해야 한다).
"""
import time
import warnings

import numpy as np
import pytest
import torch


def _alarm_msgs(wlog):
    """포집 로그에서 no-op 경보만 센다 (동결 토큰 = 'bank_refit_async')."""
    return [w for w in wlog if issubclass(w.category, RuntimeWarning)
            and 'bank_refit_async' in str(w.message)]


def _drive_async_alarm(fit_behavior, mid_join_before_t=None, extra_frame_t=None):
    """비동기 경보 시나리오 공용 드라이버.

    fit_behavior: '_bank_fit_window' 몽키패치 (계약: self._bank 를 채워야 collect 가
    적용을 센다). 트리거는 t=10+5k (bank_refit_every_s=5, dt=2 프레임).
    mid_join_before_t: 그 t 의 프레임 **직전에** 배경 스레드를 join (결정적 착지/kick).
    extra_frame_t: join 후 트리거 **아님** 프레임을 하나 끼워 collect 만 발화시킨다.
    반환 (updater, 경보 수)."""
    from wavesplat.optimizer.sequential_update import SequentialUpdater

    frames, _, _ = _real_frames(n_frames=5, n_per=40, dt=2.0)      # t = 0..8  # noqa: F821
    cfg = _bank_cfg(bank_refit_every_s=5.0, bank_refit_async=True)  # noqa: F821
    upd = SequentialUpdater(cfg)
    with pytest.warns(RuntimeWarning):
        upd.initialize(frames)
    orig = SequentialUpdater._bank_fit_window
    SequentialUpdater._bank_fit_window = fit_behavior
    try:
        with warnings.catch_warnings(record=True) as wlog:
            warnings.simplefilter('always')
            drive = list(_real_frames(n_frames=6, t0=10.0, dt=2.0, seed=33)[0])  # noqa: F821
            for f in drive:
                if mid_join_before_t is not None and float(f.t) >= mid_join_before_t:
                    fut = getattr(upd, '_bank_async', None)
                    if fut is not None:
                        fut['thread'].join(timeout=60)
                    if extra_frame_t is not None:
                        ex = _real_frames(n_frames=1, t0=extra_frame_t,        # noqa: F821
                                          dt=2.0, seed=44)[0][0]
                        upd.update(ex)                    # 트리거 아님 → collect 만
                    mid_join_before_t = None              # 1회만
                upd.update(f)
    finally:
        SequentialUpdater._bank_fit_window = orig
        fut = getattr(upd, '_bank_async', None)
        if fut is not None:
            fut['thread'].join(timeout=60)
    return upd, len(_alarm_msgs(wlog))


def test_noop_alarm_fires_when_second_trigger_is_skipped():
    """RG-2(a) — skip 소거의 최소 재현: 적합이 두 트리거보다 오래 산다.

    구 코드 = 발화 0 (조기 반환이 검사보다 앞 ⟹ 구조적 도달 불가, M6 88.5% 무경고의
    기제). 수리 후 = skip 사건에서도 검사 ⟹ **정확히 1회** 발화."""
    def slow_fit(self, P, z):
        time.sleep(1.5)                    # 드라이브 루프(ms)보다 훨씬 오래
        self._bank = list(self._bank) if self._bank else self._bank

    upd, n = _drive_async_alarm(slow_fit)
    assert getattr(upd, '_bank_async_skipped', 0) >= 1, '시나리오 무효 — skip 이 없었다'
    assert getattr(upd, '_bank_async_applied', 0) == 0, '시나리오 무효 — 적용이 있었다'
    assert n == 1, f'skip 소거 부류에서 경보 {n}회 (기대 1 — 구 코드는 0)'


def test_noop_alarm_2_0_order_trap_preserved():
    """RG-2(b) — (2,0) 순서 함정: 두 kick 이 모두 실행되고 적용은 0.

    구 경보가 **유일하게 잡던** 부류(M6 무릎 15런). 술어를 증가-전 계수로 옮기면
    ((k+s)>=2 를 그대로) 여기서 침묵한다 — 이 시험이 그 off-by-one 을 영구 고정한다."""
    def instant_fit(self, P, z):
        self._bank = list(self._bank) if self._bank else self._bank

    # 트리거2 직전 join ⟹ kick2 실행. 완료-미수집 적합은 kick2 가 덮어써 applied=0
    # (현행 거동 — M6 오발 3건의 first_apply=14 가 그 증거. §5-2: 스케줄링 불변).
    upd, n = _drive_async_alarm(instant_fit, mid_join_before_t=15.0)
    assert getattr(upd, '_bank_async_kicks', 0) == 2, '시나리오 무효 — kick 2회가 아니다'
    assert getattr(upd, '_bank_async_skipped', 0) == 0
    assert n == 1, f'(2,0) 부류에서 경보 {n}회 (기대 1 — off-by-one 이면 0)'


def test_noop_alarm_silent_when_fit_lands_between_triggers():
    """RG-2(c) — 건강 침묵: 트리거 사이에 착지·적용되면 어느 사건에서도 발화 0."""
    def instant_fit(self, P, z):
        self._bank = list(self._bank) if self._bank else self._bank

    # 트리거2 전에 join + 비-트리거 프레임 ⟹ collect 가 적용 ⟹ kick2 시점 applied=1
    upd, n = _drive_async_alarm(instant_fit, mid_join_before_t=15.0,
                                extra_frame_t=13.0)
    assert getattr(upd, '_bank_async_applied', 0) >= 1, '시나리오 무효 — 적용이 없었다'
    assert n == 0, f'건강 경로에서 경보 {n}회 (기대 0 — 신호 보존)'


def test_noop_alarm_due_is_a_pure_module_function_with_frozen_contract():
    """RG-2(d)+RG-3 — 술어 계약과 health 키 (gates_m9 동결 계약의 시험판 고정)."""
    import inspect
    from wavesplat.optimizer.sequential_update import (SequentialUpdater,
                                                       noop_alarm_due)

    assert inspect.isfunction(noop_alarm_due)
    assert len(inspect.signature(noop_alarm_due).parameters) == 2
    # 이번 사건 **포함** 계수 — (2,0) 부류 보존의 핵 (gates_m9.PREDICATE_PROBES)
    for (n, a), want in [((1, 0), False), ((2, 0), True), ((2, 1), False),
                         ((3, 0), True), ((2, 5), False), ((17, 0), True)]:
        assert noop_alarm_due(n, a) is want, (n, a, want)

    frames, _, _ = _real_frames()                                   # noqa: F821
    upd = SequentialUpdater(_bank_cfg())                            # noqa: F821
    with pytest.warns(RuntimeWarning):
        upd.initialize(frames)
    h = upd.bank_refit_health()
    base = {'enabled', 'kicks', 'applied', 'skipped', 'in_flight', 'effective'}
    assert base <= set(h), '기존 키가 사라졌다 — 하위 호환 위반'
    assert {'first_apply_data_t', 'last_fit_wall_s'} <= set(h), 'RG-3 새 키 결손'
    assert h['effective'] is False and h['first_apply_data_t'] is None
