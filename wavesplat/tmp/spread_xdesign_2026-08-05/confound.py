"""사전등록 §6 교란 점검 — 명목 확산만 바꿨을 때 **방향 외의 것이 변했는가**."""
import h5py, numpy as np, os
C='data_finemesh_2026-07-23'; T='data_spread60_2026-08-05'
def comp(p):
    with h5py.File(p,'r') as f:
        c=f['config/wave_components'][:]        # [A, omega, kx, ky, phi, Q]
        a=dict(f['config'].attrs)
    return c, a
print(f"{'cell':16s} {'|A| 동일':>9s} {'omega 동일':>11s} {'|k| 동일':>9s} {'phi 동일':>9s} {'Hs':>6s} {'Tp':>6s} {'mesh':>10s}")
allsame=True
for nc in (5,30):
    for s in range(5):
        fn=f'episode_BF4_{nc}comp_seed{s}_uniform.h5'
        cc,ca=comp(os.path.join(C,fn)); tc,ta=comp(os.path.join(T,fn))
        A=np.allclose(np.sort(cc[:,0]),np.sort(tc[:,0]))
        W=np.allclose(np.sort(cc[:,1]),np.sort(tc[:,1]))
        K=np.allclose(np.sort(np.hypot(cc[:,2],cc[:,3])),np.sort(np.hypot(tc[:,2],tc[:,3])))
        P=np.allclose(np.sort(cc[:,4]),np.sort(tc[:,4]))
        allsame &= (A and W and K and P)
        print(f'{nc:2d}comp/seed{s:<7d} {str(A):>9s} {str(W):>11s} {str(K):>9s} {str(P):>9s} '
              f'{ca["Hs"]:6.2f} {ca["Tp"]:6.2f} {ta["mesh_verdict"]:>10s}')
print(f'\n⟹ 명목 확산 변경이 진폭·주파수·|k|·위상을 **전부 보존**: {allsame}')
print('  (보존이면 단일요인 조작이 성립하고, 바뀐 것은 성분 **방향**뿐이다)')
# 필드 에너지도 확인 — GT 분산이 같아야 nRMSE 정규화가 짝지어진다
for nc in (5,30):
    v=[]
    for s in range(5):
        fn=f'episode_BF4_{nc}comp_seed{s}_uniform.h5'
        with h5py.File(os.path.join(C,fn),'r') as f: a=float(np.std(f['frames/gt_wave_field'][:]))
        with h5py.File(os.path.join(T,fn),'r') as f: b=float(np.std(f['frames/gt_wave_field'][:]))
        v.append((a,b))
    print(f'  {nc:2d}comp GT std  30° {np.median([x[0] for x in v]):.4f}  60° {np.median([x[1] for x in v]):.4f}  '
          f'비 중앙 {np.median([x[1]/x[0] for x in v]):.3f}')
