"""S2_ALPHA 계측이 결과를 **오염시키지 않음**을 측정한다 (AMENDMENT_1 A21, §5 I20).

주장: `run.py:310` 의 `S2_ALPHA=1` 은 `updater.update()` **이후** `no_grad` 로 프리뷰 4다리만
읽으므로 RNG·파라미터를 건드리지 않는다 ⟹ `nRMSE` 가 **비트 동일**해야 한다.
논증이 아니라 **측정**으로 확인한다(S1 오류 #7 의 교훈: spawn 전파를 논증하다 측정으로 바꿨다).

덤으로 이 스크립트는 **배치 전 스모크 테스트**다 — 추가 계측 키가 실제로 레코드에 들어오는지
1152 런을 시작하기 전에 본다.

사용: python3 alpha_bitcheck.py [episode] [cell] [seed]
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import run_s7 as R7   # noqa: E402


def main():
    ep = sys.argv[1] if len(sys.argv) > 1 else R7.CROWDED32[0]
    cell = sys.argv[2] if len(sys.argv) > 2 else 'AUTO-NT-L1-1e4'
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else R7.SEEDS[0]
    if not os.path.exists(os.path.join(R7.REPO, ep)):
        sys.exit(f'에피소드 없음 (생성 중일 수 있다): {ep}')
    job = {'job_id': f'{cell}|{ep}|{seed}', 'cell': cell, 'episode': ep, 'seed': seed}

    out = []

    def p(s):
        print(s, flush=True)
        out.append(s)

    p(f'S2_ALPHA 비트동일 스팟체크  cell={cell}  seed={seed}')
    p(f'  ep = {ep}')

    os.environ['S2_ALPHA'] = '0'
    rec_off = R7.S5.run_one(job)                 # 계측 OFF (S5 러너 그대로)
    os.environ.pop('S2_ALPHA', None)

    rec_on = R7.run_one(job)                     # 계측 ON (S7 러너)

    for r, lab in ((rec_off, 'OFF'), (rec_on, 'ON ')):
        if 'error' in r:
            p(f'  ⛔ {lab} 에러: {r["error"][:200]}')
            open(os.path.join(HERE, 'alpha_bitcheck.txt'), 'w').write('\n'.join(out) + '\n')
            sys.exit(1)

    a, b = rec_off['nRMSE'], rec_on['nRMSE']
    same = (a == b)
    p(f'  nRMSE  OFF {a!r}')
    p(f'  nRMSE  ON  {b!r}')
    p(f'  ⟹ 비트 동일: {"✅ YES" if same else f"⛔ NO (Δ={b - a:.3e})"}')
    p(f'  window_resid  OFF {rec_off.get("window_resid")!r}  ON {rec_on.get("window_resid")!r}')

    p('\n  [스모크] S7 신규 계측 키:')
    for k in ('regime', 'auto_selected', 'auto_n_selected', 'ensemble_k',
              'RMSE_obs_r30', 'RMSE_unobs_r30', 'n_alpha_frames', 'alpha_stencil_err'):
        p(f'    {k:20s} = {rec_on.get(k)!r}')
    sh = rec_on.get('shell_rmse') or {}
    p(f'    shell_rmse           = ' + '  '.join(
        f'{k}:{v:.4f}' for k, v in sh.items() if np.isfinite(v)))
    al = rec_on.get('alpha_episode') or {}
    p(f'    alpha_episode        = ' + json.dumps(al))
    if rec_on.get('alpha_error'):
        p(f'    ⛔ alpha_error = {rec_on["alpha_error"][:200]}')

    open(os.path.join(HERE, 'alpha_bitcheck.txt'), 'w').write('\n'.join(out) + '\n')
    sys.exit(0 if same else 1)


if __name__ == '__main__':
    main()
