import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_rx1
t = time.time()
r = run_rx1.run_one({'job_id': 'smoke', 'cell': 'X-BASE',
                     'episode': 'episode_BF4_15comp_seed0_uniform.h5', 'seed': 1515})
print('wall', time.time() - t, flush=True)
print(json.dumps({k: v for k, v in r.items() if k not in ('cfg', 'spec')}))
