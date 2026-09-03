"""★제3자 CNN 양성 대조 — **아는 파도장**을 희소화해 넣고 조밀 복원을 되받는다.

⛔저들 코드는 한 줄도 고치지 않는다. 규약은 `wassfast/_wassfast.py:751-870` 축자 복제.
⛔이것은 배치가 아니다(사전등록 전 실행가능성 확인).
"""
import os, sys, numpy as np
os.environ['TF_USE_LEGACY_KERAS']='1'; os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
BASE='/home/vexit/wavesplat/external/wassfast'
sys.path.insert(0, BASE)
import tf_keras; sys.modules['keras']=tf_keras
from wassfast.cnn.wavenet_models import create_model_with_prediction
from wassfast.cnn.sparsecnn_with_prediction_model import compute_phase_diff_matrix

N=256; L=90.0                      # 저들 시험자료와 같은 90 m 박스
dx=L/(N-1)
x=np.linspace(-L/2, L/2, N); X,Y=np.meshgrid(x,x)
# ★KX_ab/KY_ab — 저들 config256.mat 과 같은 규약(fftshift 된 각파수 격자)
kax=np.fft.fftshift(2*np.pi*np.fft.fftfreq(N, d=dx))
KX_ab, KY_ab = np.meshgrid(kax, kax)
print('KX_ab range %.4f..%.4f  (저들 config256.mat: -8.9012..8.8316)'%(KX_ab.min(),KX_ab.max()))

g=9.8; dt=0.2
rng=np.random.default_rng(7)
# 아는 파도장: 심해 분산관계를 지키는 평면파 5개, 대체로 +y 진행
comps=[]
for _ in range(5):
    lam=rng.uniform(18.0, 45.0); th=np.radians(rng.uniform(70,110))
    k=2*np.pi/lam; comps.append((rng.uniform(0.10,0.35), k*np.cos(th), k*np.sin(th),
                                 np.sqrt(g*k), rng.uniform(0,2*np.pi)))
def field(t):
    Z=np.zeros_like(X)
    for a,kx,ky,w,ph in comps: Z += a*np.cos(kx*X+ky*Y-w*t+ph)
    return Z.astype(np.float32)

Zt=[field(-dt), field(0.0), field(dt)]
print('true field std %.4f m'%float(np.std(Zt[1])))

def sparsify(Z, n, seed):
    r=np.random.default_rng(seed)
    S=np.full((N,N), np.nan, np.float32)
    iy=r.integers(0,N,n); ix=r.integers(0,N,n)
    S[iy,ix]=Z[iy,ix]
    return S

# 저들 위상차 행렬 (⛔저들 함수)
ph=np.expand_dims(compute_phase_diff_matrix(KX_ab, KY_ab, 0.0, 1.0, dt), axis=-1)
phc=np.conj(ph)
phcat=np.concatenate((np.real(ph),np.imag(ph),np.real(phc),np.imag(phc)),axis=2).astype(np.float32)
PH=phcat[None,...]

m=create_model_with_prediction(N, sparsecnn_weights=f'{BASE}/wassfast/cnn/modeldata/2021-06-30_16-48-27.h5')
m.load_weights(f'{BASE}/wassfast/cnn/modeldata/2021-07-01_12-00-44_3.h5')

print(f"\n{'n_obs':>7s} {'density':>9s} {'RMSE(m)':>9s} {'nRMSE':>7s} {'corr':>7s}   (nRMSE 1.0 = 무기술)")
for n in (600, 2000, 6000, 20000, 60000):
    B=np.zeros((1,N,N,3), np.float32)
    for c in range(3): B[0,:,:,c]=sparsify(Zt[c], n, 100+c)
    std=np.nanstd(B); bmin,bmax=-3*std, 3*std
    Xn=(B-bmin)/(bmax-bmin)
    Yp=m.predict([Xn,PH], verbose=0)
    Yp=np.squeeze(Yp)*(bmax-bmin)+bmin
    t=Zt[1]; e=Yp-t
    rmse=float(np.sqrt(np.mean(e**2))); sg=float(np.std(t))
    cc=float(np.corrcoef(Yp.ravel(), t.ravel())[0,1])
    print(f'{n:7d} {n/N**2:9.4f} {rmse:9.4f} {rmse/sg:7.3f} {cc:7.3f}')
