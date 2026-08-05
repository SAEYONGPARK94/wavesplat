import sys, os, time
sys.path.insert(0,'scripts'); sys.path.insert(0,'.')
os.chdir('.')
import numpy as np, torch
from d3_external_figs import Window
from d3_sa_multirecord import MRecord
from d3_prior_net_eval import get_net, DEVICE
from d3_l1_s2head_train import SmultHead
from d3_falcon_real_operator import RECORDS

print('DEVICE', DEVICE)
CK='results/direction3/prior_net_l1s2bw.pt'
net=get_net(CK)
head=SmultHead().to(DEVICE)
head.load_state_dict(torch.load(CK, map_location=DEVICE)['head']); head.eval()
print('net/head loaded')

t=time.time()
rec=MRecord(RECORDS['YS17']['path'], stride=RECORDS['YS17']['stride'])
print('MRecord YS17 loaded in %.1fs; gx %d gy %d dt %.3f T_all %d'%(time.time()-t, len(rec.gx), len(rec.gy), rec.dt, rec.T_all))

t=time.time()
w=Window('YS17', 0, rec=rec, net=net, head=head)
print('Window build %.1fs; i0=%d Hs=%.3f kp=%.4f'%(time.time()-t, w.i0, w.Hs, w.kp))

t=time.time()
gx,gy,truth,fh,fn=w.fields([0.0], stride=2)
print('fields lead0 render %.1fs; truth shape %s fh %s'%(time.time()-t, truth[0].shape, fh[0].shape))
print('truth finite frac %.2f; truth p97 %.3f'%(np.isfinite(truth[0]).mean(), np.nanpercentile(truth[0],97)))
th,tn=w.protocol_corr()
print('protocol_corr hand %.3f net %.3f'%(th,tn))
