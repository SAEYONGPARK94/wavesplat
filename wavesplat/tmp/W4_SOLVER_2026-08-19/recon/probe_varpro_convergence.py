import json,os,sys,numpy as np,torch
sys.path.insert(0,'/home/vexit/wavesplat/wavesplat'); os.chdir('/home/vexit/wavesplat/wavesplat')
sys.path.insert(0,'/home/vexit/wavesplat/wavesplat/tmp/W4_SOLVER_2026-08-19/recon')
import probe_varpro as P
from wavesplat.data_loader import EpisodeDataLoader
rows=[json.loads(l) for l in open(P.S4)]
rows=[r for r in rows if 'error' not in r and r['cell'] in ('AUTO-NT','ORA-K-NT')]
sd=min(r['seed'] for r in rows); rows=[r for r in rows if r['seed']==sd and r['regime']=='crowded']
eps=sorted({r['episode'] for r in rows})[:4]
for cell in ('ORA-K-NT','AUTO-NT'):
  for ep in eps:
    r=[x for x in rows if x['cell']==cell and x['episode']==ep][0]
    ld=EpisodeDataLoader(os.path.join('/home/vexit/wavesplat/wavesplat',ep))
    X,Y,Z,T=P._win(ld); gtc=np.asarray(ld.wave_components,float)
    k0=np.stack([np.asarray(r['kx'],float),np.asarray(r['ky'],float)],1); n=len(k0)
    kx=torch.tensor(k0[:,0],dtype=torch.float64,requires_grad=True)
    ky=torch.tensor(k0[:,1],dtype=torch.float64,requires_grad=True)
    opt=torch.optim.Adam([kx,ky],lr=2e-3); out=[]
    for s in range(1,4001):
        opt.zero_grad(); f,_c,_o=P._fc(kx,ky,X,Y,T,Z,ld.depth); f.backward(); opt.step()
        if s in (400,1000,2000,4000):
            with torch.no_grad(): f1,c1,om1=P._fc(kx,ky,X,Y,T,Z,ld.depth)
            out.append((s,float(np.sqrt(f1.item())),P._score(kx.detach().numpy(),ky.detach().numpy(),
                        c1[:n].numpy(),c1[n:].numpy(),om1.numpy(),ld)))
    print('%-9s %-46s n=%d  '%(cell,ep.split('/')[-1][:44],n)+' | '.join('%d: win %.5f nR %.5f'%o for o in out),flush=True)
