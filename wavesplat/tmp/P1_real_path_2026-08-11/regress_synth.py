"""회귀 증명 — 합성 경로 nRMSE 가 수정 전/후 비트 동일한지 측정한다.
같은 스크립트를 (1) 하네스 원본(HEAD), (2) 수정본 상태에서 각각 실행해 비교한다.
harness 모듈이 애초에 import 되지 않는다는 사실도 함께 기록한다."""
import os, sys, json, warnings
warnings.filterwarnings('ignore')
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

EPISODES = ['data/episode_BF4_5comp_seed0_uniform.h5',
            'data/episode_BF2_1comp_seed0_uniform.h5']
TAG = sys.argv[1] if len(sys.argv) > 1 else 'now'
OUT = f'tmp/P1_real_path_2026-08-11/regress_synth_{TAG}.json'

from wavesplat.run import run_estimation, set_seed

KEYS = ('init_nrmse', 'init_rmse', 'avg_nRMSE', 'avg_RMSE_z', 'avg_Hs_error',
        'avg_obs_loss', 'auto_selected')

res = {}
for ep in EPISODES:
    if not os.path.exists(ep):
        continue
    set_seed(42)
    out = run_estimation(ep, {'seed': 42, 'device': 'cpu'}, verbose=False)
    res[os.path.basename(ep)] = {k: (float(out[k])
                                     if isinstance(out[k], (int, float, np.floating))
                                     else out[k]) for k in KEYS}
    # per-frame nRMSE, full precision -> bit-level comparison
    res[os.path.basename(ep)]['frame_nRMSE_repr'] = [
        repr(float(r['nRMSE'])) for r in out['frame_results'] if 'nRMSE' in r]
    print(ep, {k: res[os.path.basename(ep)][k] for k in KEYS}, flush=True)

res['_harness_imported'] = 'make_gif_bank_deployed' in sys.modules
json.dump(res, open(OUT, 'w'), indent=1, sort_keys=True)
print('harness module imported by synthetic path:', res['_harness_imported'])
print('wrote', OUT)
