"""CON contamination control: PRO edited sequential_update.py at 21:21 while my probe
pool was still running.  Its gates default OFF, so shipped behaviour should be
bit-identical.  Re-run the FROZEN cell S2-T1f|BF4_15comp|42 under the edited file and
compare against the pre-edit value recorded in tmp/s1s2_2026-08-05/results.jsonl."""
import os, sys, json
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('tmp/s1s2_2026-08-05'))
os.environ.setdefault('OMP_NUM_THREADS','1')
import run_s1s2 as S
job = {'job_id':'S2-T1f|episode_BF4_15comp_seed0_uniform.h5|42','arm':'S2-T1f',
       'episode':'episode_BF4_15comp_seed0_uniform.h5','seed':42}
r = S.run_one(job)
old = [json.loads(l) for l in open('tmp/s1s2_2026-08-05/results.jsonl')
       if json.loads(l)['job_id']==job['job_id']][0]
print('metric            pre-edit (21:0x)     post-edit (now)      identical')
ok=True
for k in ['nRMSE','init_nrmse','Pearson_rho','Variance_Ratio','Hs_error','N']:
    same = abs(float(r[k])-float(old[k])) < 1e-12
    ok &= same
    print(f'{k:18s}{float(old[k]):20.12f}{float(r[k]):20.12f}   {same}')
print('BIT-IDENTICAL:', ok)
