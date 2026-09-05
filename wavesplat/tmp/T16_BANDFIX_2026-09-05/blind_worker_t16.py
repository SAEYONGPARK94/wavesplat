"""★`G-S-BLIND` 의 **별도 프로세스** — 관측 5배열만 받아 세 규칙을 재실행한다.

⛔표적(`eta`·`Ew`·`sigma4`)·메타(`hs`·`angle`·`block`)는 이 프로세스에 **존재하지 않는다**.
그것이 이 검사의 하중이다 — 규칙이 관측 밖의 무언가를 봤다면 `s` 가 비트 동일하지 않다.

★`_beamform_top` 호출 수도 함께 돌려준다 — §8 의 「파일당 16 호출」을 `beamform_calls_n` 으로
**되읽기** 위해서다(⛔러너가 4를 손으로 더하면 그것은 유도이지 측정이 아니다).
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE,
           os.path.abspath(os.path.join(HERE, '..', '..')),
           os.path.abspath(os.path.join(HERE, '..', 'T14_AUTOSCALE_2026-09-04'))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import scale_rules_t16 as SR                                       # noqa: E402
from wavesplat.optimizer import bank_mmse as BM                    # noqa: E402

_ORIG = BM._beamform_top
_N = {'k': 0}


def _counted(P, r, depth, klim=0.6, n=80):
    _N['k'] += 1
    return _ORIG(P, r, depth, klim=klim, n=n)


BM._beamform_top = _counted

d = np.load(sys.argv[1])
s_out, n_out = {}, {}
for _name, _fn in sorted(SR.RULES.items()):
    _N['k'] = 0
    s_out[_name] = float(_fn(d['x'], d['y'], d['t'], d['z'], float(d['depth'])))
    n_out[_name] = int(_N['k'])
print(json.dumps({'s': s_out, 'n_calls': n_out,
                  'device': str(BM._DEV)}))
