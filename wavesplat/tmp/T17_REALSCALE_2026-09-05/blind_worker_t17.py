"""`T17-REALSCALE` **blind 재실행 일꾼** — `G-S-BLIND` (측정면).

★**별도 프로세스**에서 관측 5배열 `(x, y, t, z, depth)` **만** 받아 두 규칙을 다시 돌린다.
⛔레코드 이름·창 번호·참값·`sc` dict 은 넘기지 않는다 — 넘길 방법이 없다(입력이 `.npz` 5배열이다).
⟹ 규칙이 관측 밖의 무언가를 보면 `s` 가 달라지고 분석기가 그것을 **비트 비교**로 잡는다.

사용: `python3 blind_worker_t17.py <npz 경로>`  → 마지막 줄에 JSON 한 줄.
"""
import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')

import json                                                        # noqa: E402
import sys                                                         # noqa: E402

import numpy as np                                                 # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
T14 = os.path.abspath(os.path.join(HERE, '..', 'T14_AUTOSCALE_2026-09-04'))
T16 = os.path.abspath(os.path.join(HERE, '..', 'T16_BANDFIX_2026-09-05'))
for _p in (HERE, ROOT, os.path.join(ROOT, 'scripts'), T14, T16):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import scale_rules_t14 as SR14                                     # noqa: E402
import scale_rules_t16 as SR16                                     # noqa: E402


def main(path):
    import torch
    torch.set_num_threads(1)
    d = np.load(path)
    x, y, t, z = d['x'], d['y'], d['t'], d['z']
    depth = float(d['depth'])
    n_calls = {'bandr': 0, 'ap': 0, 'band_old': 0}
    from wavesplat.optimizer import bank_mmse as BM
    orig = BM._beamform_top
    which = ['bandr']

    # ⛔인자 이름을 바꾸지 마라 — 호출자가 `n=` 키워드로 넘긴다(스모크가 이것을 잡았다).
    def wrap(P, r, dep, klim=0.6, n=80, _o=orig, _c=n_calls):
        _c[which[0]] += 1
        return _o(P, r, dep, klim=klim, n=n)
    BM._beamform_top = wrap
    try:
        which[0] = 'bandr'
        s_bandr = SR16.clip_s(SR16.s_bandr_raw(x, y, t, z, depth))
        which[0] = 'ap'
        s_ap = SR16.clip_s(SR14.s_ap_raw(x, y, t, z, depth))
        which[0] = 'band_old'
        s_band_old = SR16.clip_s(SR14.s_band_raw(x, y, t, z, depth))
    finally:
        BM._beamform_top = orig
    print(json.dumps({'s': {'bandr': s_bandr, 'ap': s_ap, 'band_old': s_band_old},
                      'n_calls': n_calls, 'device': str(BM._DEV)}))


if __name__ == '__main__':
    main(sys.argv[1])
