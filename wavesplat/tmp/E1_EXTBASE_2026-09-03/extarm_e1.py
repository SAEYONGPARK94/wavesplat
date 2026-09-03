"""`E1-EXTBASE` 제3자 팔 어댑터 — ⛔저들 코드 **무수정**, 규약은 `_wassfast.py:751-870` 축자.

이 파일이 하는 일은 **우리 관측을 저들 입력 규약으로 옮기는 것뿐**이다.
⛔참값(`maps_gt`)은 이 파일에 **들어오지 않는다**(호출자가 채점만 한다).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
EXT = os.path.abspath(os.path.join(ROOT, '..', 'external', 'wassfast'))
if EXT not in sys.path:
    sys.path.insert(0, EXT)


def build_model(n=256):
    os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
    import tf_keras
    sys.modules.setdefault('keras', tf_keras)          # 저들 소스가 `keras` 를 부른다
    from wassfast.cnn.wavenet_models import create_model_with_prediction
    import gates_e1 as G
    m = create_model_with_prediction(n, sparsecnn_weights=os.path.join(EXT, G.CKPT_SPARSE))
    m.load_weights(os.path.join(EXT, G.CKPT_FULL))     # ⛔순서가 규약이다
    return m


def kgrid(n, dx, dy):
    """저들 `config256.mat` 과 같은 규약: fftshift 된 각파수 격자."""
    kx = np.fft.fftshift(2 * np.pi * np.fft.fftfreq(n, d=dx))
    ky = np.fft.fftshift(2 * np.pi * np.fft.fftfreq(n, d=dy))
    return np.meshgrid(kx, ky)


def phase_mat(KX, KY, xsign, ysign, dt, depth):
    """⛔저들 함수를 임포트해서 쓴다(재구현 아님)."""
    from wassfast.cnn.sparsecnn_with_prediction_model import compute_phase_diff_matrix
    ph = np.expand_dims(compute_phase_diff_matrix(KX, KY, xsign, ysign, dt, depth=depth),
                        axis=-1)
    phc = np.conj(ph)
    cat = np.concatenate((np.real(ph), np.imag(ph), np.real(phc), np.imag(phc)),
                         axis=2).astype(np.float32)
    return cat[None, ...]


def rasterise(px, py, pz, n, xmin, xmax, ymin, ymax):
    """저들 `_wassfast.py:759-768` 축자 — floor(( . )*(N−1) + 0.5)."""
    ix = np.floor((px - xmin) / (xmax - xmin) * (n - 1) + 0.5).astype(np.int64)
    iy = np.floor((py - ymin) / (ymax - ymin) * (n - 1) + 0.5).astype(np.int64)
    good = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n)
    Z = np.full((n, n), np.nan, np.float32)
    Z[iy[good], ix[good]] = pz[good]
    return Z, int(good.sum())


def sample_frame(Zf, gx, gy, m, rng, noise):
    """`scene_station` 과 **같은 규약**: 유효 격자에서 균등 추출 + 등방 잡음."""
    ok = np.isfinite(Zf)
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        return np.empty(0), np.empty(0), np.empty(0)
    pick = rng.choice(idx, min(m, idx.size), replace=False)
    iy, ix = np.unravel_index(pick, Zf.shape)
    return gx[ix], gy[iy], Zf[iy, ix] + rng.normal(0, noise, len(pick))


def predict(model, frames, gx, gy, m_per_frame, dt, depth, seed, n=256,
            holdout=0.0, signs=((1, 1),)):
    """3 프레임 희소 관측 → 저들 CNN → (n,n) 조밀 고도.

    `holdout>0` 이면 **가운데 프레임의 그 비율**을 입력에서 빼고 따로 돌려준다
    (§4 GT-free 방향 선택용). ⛔참값은 쓰지 않는다.
    반환: `(best_sign, Zdense, n_in, held)`
    """
    xmin, xmax, ymin, ymax = gx.min(), gx.max(), gy.min(), gy.max()
    dxg, dyg = (xmax - xmin) / (n - 1), (ymax - ymin) / (n - 1)
    KX, KY = kgrid(n, dxg, dyg)
    rng = np.random.default_rng(seed)
    pts = [sample_frame(frames[k], gx, gy, m_per_frame, rng, 0.02) for k in range(3)]

    held = None
    if holdout > 0:
        px, py, pz = pts[1]
        k = max(1, int(round(len(pz) * holdout)))
        sel = np.random.default_rng(seed + 1).permutation(len(pz))
        held = (px[sel[:k]], py[sel[:k]], pz[sel[:k]])
        pts[1] = (px[sel[k:]], py[sel[k:]], pz[sel[k:]])

    B = np.zeros((1, n, n, 3), np.float32)
    n_in = 0
    for c in range(3):
        Z, ng = rasterise(*pts[c], n, xmin, xmax, ymin, ymax)
        B[0, :, :, c] = Z
        n_in += ng
    std = float(np.nanstd(B))
    bmin, bmax = (-3 * std, 3 * std) if std > 0 else (-0.01, 0.01)
    Xn = (B - bmin) / (bmax - bmin)

    best, best_res, best_Z = None, np.inf, None
    for sg in signs:
        PH = phase_mat(KX, KY, sg[0], sg[1], dt, depth)
        Y = np.squeeze(model.predict([Xn, PH], verbose=0)) * (bmax - bmin) + bmin
        if held is None:
            best, best_Z = sg, Y
            break
        hx, hy, hz = held
        ix = np.clip(np.floor((hx - xmin) / (xmax - xmin) * (n - 1) + 0.5).astype(int), 0, n - 1)
        iy = np.clip(np.floor((hy - ymin) / (ymax - ymin) * (n - 1) + 0.5).astype(int), 0, n - 1)
        res = float(np.sqrt(np.mean((Y[iy, ix] - hz) ** 2)))
        if res < best_res:
            best, best_res, best_Z = sg, res, Y
    return best, best_Z, n_in, (0 if held is None else len(held[2]))


def to_query(Zd, gx, gy, gxq, gyq, n=256):
    """저들 256² 출력을 우리 질의 격자로 **최근접 이웃** 되샘플(결과 전 고정)."""
    xmin, xmax, ymin, ymax = gx.min(), gx.max(), gy.min(), gy.max()
    ix = np.clip(np.floor((gxq - xmin) / (xmax - xmin) * (n - 1) + 0.5).astype(int), 0, n - 1)
    iy = np.clip(np.floor((gyq - ymin) / (ymax - ymin) * (n - 1) + 0.5).astype(int), 0, n - 1)
    return Zd[np.ix_(iy, ix)]
