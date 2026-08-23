"""실측 채점 기준의 **단일 출처** — 상수를 적지 말고 **영(zero) 예측을 채점 함수에 통과시켜라**.

## 왜 이 파일이 있는가 (2026-08-23)
`gates_b2.ZERO_REF = 1.0` 은 주석에 축자로 *"σ 정규화 ⟹ 제로필드 = 1.0 (정의상, 손복사 아님)"*
이라 적혀 있었고 **그 주석이 틀렸다.** σ 는 **과거 기억창 40 s 전해상도**에서 나오고
채점 분자는 **미래 프레임의 4 m 부분추출 격자**다. 두 창은 다른 양이고 `MRecord.window` 가
각각 따로 demean 한다. **실측: 참 제로필드는 레코드별 0.3375–1.1559, lead 까지 벌리면 0.3375–2.0966.**

그 상수 하나가 **여섯 문서의 계수를 동시에 틀리게 했다**(`AMENDMENT_ZEROREF.md`).
⟹ ★**처방**: 기준선은 **자료에서 산출한다.** 이 모듈이 그 유일한 경로다.

## 쓰는 법
    from zero_ref import zero_ref_nowcast, zero_ref_perlead
    z  = zero_ref_nowcast(record, win, noise)     # lead 0 스칼라
    zl = zero_ref_perlead(record, win, noise)     # {lead(float): 값}
⚠캐시가 없으면 **계산한다** — 그래서 상수가 아니라 **함수**다.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_CACHE_NOW = os.path.join(HERE, 'true_zero_field.json')
_CACHE_LEAD = os.path.join(HERE, '..', 'B3_REFIT_2026-08-23', 'true_zero_perlead.json')


def _scene(record, win, noise):
    import sys
    for p in (HERE, os.path.abspath(os.path.join(HERE, '..', '..')),
              os.path.abspath(os.path.join(HERE, '..', '..', 'scripts'))):
        if p not in sys.path:
            sys.path.insert(0, p)
    from scene_station import scene_station
    return scene_station(record, win=win, noise=noise)


def zero_ref_perlead(record, win=3, noise=0.02, use_cache=True):
    """`Z = 0` 을 이 배치의 채점식에 통과시킨 값, lead 별.

    채점식은 러너와 **글자 그대로 같다**: `sqrt(mean((Z - gt)^2)) / sg`, `Z = 0`.
    """
    kk = f'{record}|{win}'
    if use_cache and abs(noise - 0.02) < 1e-12 and os.path.exists(_CACHE_LEAD):
        c = json.load(open(_CACHE_LEAD))
        if kk in c:
            return {float(k): v for k, v in c[kk].items()}
    sc = _scene(record, win, noise)
    out = {}
    for li, l_ in enumerate(sc['leads']):
        g = np.asarray(sc['maps_gt'][li], float)
        mm = np.isfinite(g)
        if mm.sum():
            out[float(l_)] = float(np.sqrt(np.mean(g[mm] ** 2)) / sc['sg'])
    return out


def zero_ref_nowcast(record, win=3, noise=0.02, use_cache=True):
    if use_cache and abs(noise - 0.02) < 1e-12 and win == 3 and os.path.exists(_CACHE_NOW):
        c = json.load(open(_CACHE_NOW))
        if record in c:
            return c[record]
    return zero_ref_perlead(record, win, noise, use_cache)[0.0]


def selftest():
    """상수 `1.0` 과 실제로 다른지 스스로 증명한다 (시험이 공허하지 않음을 보인다)."""
    z = json.load(open(_CACHE_NOW))
    lo, hi = min(z.values()), max(z.values())
    assert not (0.99 < lo and hi < 1.01), '상수 1.0 과 구분되지 않는다 — 시험이 공허하다'
    return {'n': len(z), 'min': round(lo, 4), 'max': round(hi, 4),
            'n_far_from_one': sum(1 for v in z.values() if abs(v - 1.0) > 0.1)}


if __name__ == '__main__':
    print(selftest())
