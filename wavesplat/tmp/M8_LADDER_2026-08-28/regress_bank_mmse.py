"""★`bank_mmse.py` **비트동일 회귀 시험** — 2026-08-28 `dk_frac`·`k_set` 가산분.

    python3 tmp/M8_LADDER_2026-08-28/regress_bank_mmse.py           # 전건(12 레코드 x 3 시드)
    python3 tmp/M8_LADDER_2026-08-28/regress_bank_mmse.py --quick   # 3 레코드 x 1 시드

⛔**추정기 런 0회 · GPU 0회 · 자료 추가 0건.**  `fit_bank_mmse` 와 생성자만 부른다.
(`CUDA_VISIBLE_DEVICES=''` 를 torch 임포트 **전에** 세운다 — `PREREG §5.1 BLOCKER-A`,
그리고 오늘 `M1` 이 GPU 를 쓰고 있으므로 건드리지 않는다.)

## 왜 이 시험이 필요한가
`wavesplat/optimizer/bank_mmse.py` 는 **실측 배포 경로**다(REAL-DOMAIN GATE →
`_initialize_bank` → `fit_bank_mmse`).  `M-8` 이 그 파일에 인자 둘을 넣었으므로,
**기본 인자 경로가 한 비트도 안 움직였다**는 것을 기계가 매번 다시 증명해야 한다.
선례 = `k_snap_dk`(B-16) · `bank_refit_every_s`(D) · `cap_mode`(M5b).

## 다섯 겹
  **L1 (전건, 차단)** 수정 전 소스(git blob, md5 대조)와 수정 후 소스가 **기본 인자**에서
        `BankMMSEFit` 의 모든 상태를 비트동일하게 낸다 — 12 레코드 x 3 시드.
  **L2 (차단)** ★새 인자는 **오직 `dk_c` 를 통해서만** 작용한다:
        `fit(dk_c=kp/10.0)` 명시 == 기본(무인자).  ⟹ else 분기가 축자 그대로다.
        ⚠**그리고 여기서 사실 하나가 드러난다**(§L2 참조): `dk_frac=0.1` 은 배포 기본과
        **비트동일이 아니다** — `0.1*kp` 와 `kp/10.0` 이 IEEE754 에서 1 ulp 갈린다.
        ⟹ 이 시험은 *"같다"* 를 주장하지 않고 **세어서 공시**하고, 갈리는 원인이
        `dk_c` **하나뿐**임을 위 항등으로 못박는다.  ⛔러너는 `BANK`/`BANK-K1` 을
        `dk_frac=0.1` 로 정의하면 안 된다 — **인자를 주지 않는 것**으로 정의해야 한다.
  **L3 (차단)** `k_set=(kx,ky)` 주입이 **손으로 만든 같은 격자**의 적합과 비트동일
        (`dk_c` 제외 — 주입 경로에서는 NaN 이 정의다).  ⟹ 주입이 γ·val·EM·solve 를
        건드리지 않는다.
  **L4 (차단, no-op 방지)** 처치가 **실제로 다르다**: `bank_B` 가 `dk_frac` 을 따라
        `premise_m8.json:P2` 의 표대로 움직이고(12/12), 적합 자체도 기본과 다르다.
        ⛔이것이 없으면 "처치를 걸었다고 적고 안 건" 상태다(`M4 G-13` 부류).
  **L5 (차단)** ★**조용한 무시 금지 가드**가 실제로 문다 —
        `real_estimator != 'bank_mmse'` + bank 키 => `ValueError`;
        배타 인자 조합 => `ValueError`; 그리고 **정상 조합은 통과**한다(항진 방지).

⚠**L1·L2·L3 은 수정 전 소스에 새 키워드를 넘기지 않는다** — 넘기면 `TypeError` 다.
"""
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import warnings

warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = ''          # ★torch 임포트 전 (BLOCKER-A)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
for _p in (HERE, ROOT, os.path.join(ROOT, 'scripts'),
           os.path.join(ROOT, 'tmp', 'B2_REAL_2026-08-23')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(ROOT)

import numpy as np                                              # noqa: E402
import torch                                                    # noqa: E402

from gates_b2 import RECORDS                                    # noqa: E402
from scene_station import scene_station                         # noqa: E402
from wavesplat.optimizer.bank_mmse import fit_bank_mmse as post  # noqa: E402
from wavesplat.optimizer.sequential_update import SequentialUpdater  # noqa: E402

# ── 수정 전 판본 핀 ─────────────────────────────────────────────────────────
# ⛔HEAD 를 쓰지 않는다(다른 세션이 커밋하면 흔들린다). blob 을 직접 핀한다 (M5b 선례).
BANK_PRE_BLOB = 'f19871c268d71dd2c12a6fe26f56d79fb4a05ba2'
BANK_PRE_MD5 = 'f6e989c6aabcb71094f0e631b0341809'

WIN = 3                       # PREREG §4: 실측 창 win=3, cap_mode='uniform'(기본)
# 배포 median-of-3 의 실제 val-split 시드 (est_seed=42 + {0,1000,2000});
# `_initialize_bank:1131` 의 오프셋에서 온다. 비트동일 시험이라 시드 선택은 무해하다.
SEEDS = (42, 1042, 2042)
LADDER_FRACS = (0.2, 0.4, 0.8)   # PREREG §2 R2·R3·R4
PREMISE = json.load(open(os.path.join(HERE, 'premise_m8.json')))
# ⛔손복사 상수 금지: 기대 기수는 premise 가 낸 표에서 읽는다.
B_OF_FRAC = PREMISE['P2']['B_of_frac']

# BankMMSEFit 의 상태 전부 (render 는 이것만 본다)
ARRS = ('kxb', 'kyb', 'om', 'gam', 'w_pred')
SCALARS = ('kp', 'seed', 'sigma0', 'smult', 'pexp', 'vr', 'vr_rel', 'n_obs',
           'n_candidates')


def load_pre():
    """수정 전 `bank_mmse.py` 를 git blob 에서 꺼내 임시로 임포트한다.

    ⛔사본을 저장소에 남기지 않는다(같은 것이 N곳 금지)."""
    src = subprocess.run(['git', '-C', ROOT, 'cat-file', 'blob', BANK_PRE_BLOB],
                         capture_output=True, check=True).stdout
    got = hashlib.md5(src).hexdigest()
    if got != BANK_PRE_MD5:
        raise RuntimeError(f'수정 전 bank_mmse md5 불일치: {got} != {BANK_PRE_MD5}')
    text = src.decode()
    assert 'dk_frac' not in text and 'k_set' not in text, 'blob 핀이 틀렸다(이미 패치본)'
    path = os.path.join(HERE, '_bank_mmse_pre_regress.py')
    with open(path, 'w') as fh:
        fh.write(text)
    try:
        spec = importlib.util.spec_from_file_location('bank_mmse_pre_regress', path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        os.remove(path)
    assert mod._DEV.type == 'cpu', 'CUDA_VISIBLE_DEVICES 가 안 먹었다'
    return mod.fit_bank_mmse


def obs(rec):
    """배포 실측 경로가 보는 것과 같은 관측 (`_bank_fit_window` 축자):
    scene_station(win=3) 의 (P, z) — `m_cap=600` 이라 600-cap 분기는 무연산 —
    그리고 `sigma0 = 0.05 * 4*std(z)`."""
    sc = scene_station(rec, win=WIN)
    P = np.asarray(sc['P'], float)
    z = np.asarray(sc['z'], float)
    assert len(z) <= 600, '600-cap 분기가 살아난다 — 하네스 가정이 깨졌다'
    return P, z, float(sc['depth']), 0.05 * 4.0 * float(np.std(z))


def diff_fit(a, b, skip=()):
    """두 `BankMMSEFit` 이 비트동일한지. 다르면 (필드, 사유) 목록."""
    bad = []
    for k in ARRS:
        if k in skip:
            continue
        x, y = np.asarray(getattr(a, k)), np.asarray(getattr(b, k))
        if x.shape != y.shape or x.dtype != y.dtype:
            bad.append((k, f'{x.shape}{x.dtype} vs {y.shape}{y.dtype}'))
        elif x.tobytes() != y.tobytes():
            bad.append((k, f'bits differ max|d|={float(np.max(np.abs(x - y))):.3e}'))
    for k in SCALARS:
        if k in skip:
            continue
        x, y = getattr(a, k), getattr(b, k)
        if np.float64(x).tobytes() != np.float64(y).tobytes():
            bad.append((k, f'{x!r} vs {y!r}'))
    return bad


def main():
    quick = '--quick' in sys.argv
    recs = RECORDS[:3] if quick else RECORDS
    seeds = SEEDS[:1] if quick else SEEDS
    pre = load_pre()
    fails, ncmp = [], 0
    print(f'수정 전 blob={BANK_PRE_BLOB[:12]} md5={BANK_PRE_MD5}  '
          f'| 레코드 {len(recs)} · 시드 {list(seeds)} · win={WIN} · torch={torch.__version__} '
          f'cuda_available={torch.cuda.is_available()}', flush=True)
    if torch.cuda.is_available():
        fails.append('⛔CUDA 가 보인다 — CUDA_VISIBLE_DEVICES 가 안 먹었다')

    SC = {r: obs(r) for r in recs}

    # ── L1 기본 인자 비트동일 ────────────────────────────────────────────────
    print('\nL1 기본 인자 비트동일 (수정 전 blob vs 현재 소스) ...', flush=True)
    base = {}
    for r in recs:
        P, z, d, s0 = SC[r]
        for s in seeds:
            a = pre(P, z, d, s0, kp_hint=None, seed=s)
            b = post(P, z, d, s0, kp_hint=None, seed=s)
            base[(r, s)] = b
            dd = diff_fit(a, b)
            ncmp += 1
            if dd:
                fails.append(f'L1 {r} seed={s}: {dd[:3]}')
            print(f'  {r:7s} seed={s:<5d} B={b.n_candidates:5d} kp={b.kp:.6f} '
                  f'vr={b.vr:.6f}  {"OK" if not dd else "⛔ " + str(dd[:2])}', flush=True)

    # ── L2 새 인자는 오직 dk_c 를 통해서만 작용한다 + 부동소수 공시 ──────────
    print('\nL2 (a) fit(dk_c=kp/10.0) 명시 == 기본  (else 분기 축자성) ...', flush=True)
    for r in recs:
        P, z, d, s0 = SC[r]
        for s in seeds:
            b0 = base[(r, s)]
            b1 = post(P, z, d, s0, kp_hint=None, seed=s, dk_c=b0.kp / 10.0)
            dd = diff_fit(b0, b1)
            ncmp += 1
            if dd:
                fails.append(f'L2a {r} seed={s}: {dd[:3]}')
            print(f'  {r:7s} seed={s:<5d} {"OK" if not dd else "⛔ " + str(dd[:2])}',
                  flush=True)

    print('\nL2 (b) ⚠공시: `dk_frac=0.1` 은 배포 기본의 비트-별칭이 **아니다** '
          '(0.1*kp vs kp/10.0) ...', flush=True)
    n_ulp, n_same = 0, 0
    for r in recs:
        P, z, d, s0 = SC[r]
        s = seeds[0]
        b0 = base[(r, s)]
        b1 = post(P, z, d, s0, kp_hint=None, seed=s, dk_frac=0.1)
        same_expr = (b0.kp / 10.0) == (0.1 * b0.kp)
        dd = diff_fit(b0, b1)
        ncmp += 1
        if same_expr:
            n_same += 1
            if dd:                      # 같은 dk_c 인데 다르면 그건 진짜 결함이다
                fails.append(f'L2b {r}: dk_c 가 같은데 적합이 다르다 {dd[:3]}')
        else:
            n_ulp += 1
        print(f'  {r:7s} dk_c {"동일" if same_expr else "1ulp 상이"} · '
              f'B {b0.n_candidates}->{b1.n_candidates} · '
              f'적합 {"비트동일" if not dd else "상이(" + str(len(dd)) + "필드)"}',
              flush=True)
    print(f'  => 식이 일치한 레코드 {n_same}/{len(recs)} · 1ulp 갈린 레코드 {n_ulp}'
          f'  ⛔따라서 러너는 R0/R1 을 **인자 부재**로 정의해야 한다', flush=True)

    # ── L3 k_set 주입 == 손으로 만든 같은 격자 ───────────────────────────────
    print('\nL3 k_set 주입 == 손으로 만든 동일 격자 (dk_c 제외) ...', flush=True)
    for r in recs:
        P, z, d, s0 = SC[r]
        for s in seeds:
            b0 = base[(r, s)]
            kp, dk = b0.kp, b0.kp / 10.0            # 배포 격자를 손으로 재현
            b_lo, b_hi = (0.35, 2.5)
            ax = np.arange(-b_hi * kp, b_hi * kp + dk / 2, dk)
            KX, KY = np.meshgrid(ax, ax)
            kk = np.hypot(KX, KY)
            sel = (kk > b_lo * kp) & (kk < b_hi * kp)
            b1 = post(P, z, d, s0, kp_hint=None, seed=s,
                      k_set=(KX[sel].astype(float), KY[sel].astype(float)))
            dd = diff_fit(b0, b1, skip=('dk_c',))
            ncmp += 1
            if dd:
                fails.append(f'L3 {r} seed={s}: {dd[:3]}')
            if not np.isnan(b1.dk_c):
                fails.append(f'L3 {r} seed={s}: dk_c 가 NaN 이 아니다 ({b1.dk_c})')
            print(f'  {r:7s} seed={s:<5d} B={b1.n_candidates:5d} dk_c={b1.dk_c} '
                  f'{"OK" if not dd else "⛔ " + str(dd[:2])}', flush=True)

    # ── L4 처치가 실제로 다르다 (no-op 방지) + premise 표와 대조 ─────────────
    print('\nL4 dk_frac 이 기수를 움직인다 (premise_m8.json:P2 대조) + no-op 아님 ...',
          flush=True)
    n_diff = 0
    for r in recs:
        P, z, d, s0 = SC[r]
        s = seeds[0]
        b0 = base[(r, s)]
        got = []
        for f in LADDER_FRACS:
            b1 = post(P, z, d, s0, kp_hint=None, seed=s, dk_frac=f)
            exp = B_OF_FRAC[r][str(f)]
            ncmp += 1
            if b1.n_candidates != exp:
                fails.append(f'L4 {r} frac={f}: B={b1.n_candidates} != premise {exp}')
            if b1.n_candidates >= b0.n_candidates:
                fails.append(f'L4 {r} frac={f}: B 가 안 줄었다 '
                             f'({b0.n_candidates}->{b1.n_candidates})')
            if diff_fit(b0, b1):
                n_diff += 1
            else:
                fails.append(f'L4 {r} frac={f}: ⛔적합이 기본과 비트동일 = 처치 no-op')
            got.append(b1.n_candidates)
        print(f'  {r:7s} B {b0.n_candidates:5d} -> {got}  (기대 '
              f'{[B_OF_FRAC[r][str(f)] for f in LADDER_FRACS]})', flush=True)
    print(f'  => 기본과 다른 셀 {n_diff}/{len(recs) * len(LADDER_FRACS)}', flush=True)

    # ── L5 조용한 무시 금지 가드 (런 0회 — 생성자만) ────────────────────────
    print('\nL5 가드: 조용한 무시 금지 + 항진 방지 ...', flush=True)
    kx = np.array([0.1, 0.2]); ky = np.array([0.0, 0.05])

    def expect_raise(tag, fn):
        try:
            fn()
        except ValueError:
            print(f'  {tag:52s} ValueError OK', flush=True)
            return
        except Exception as e:                                  # noqa: BLE001
            fails.append(f'L5 {tag}: ValueError 가 아니라 {type(e).__name__}')
            return
        fails.append(f'L5 {tag}: ⛔예외가 안 났다 = 조용한 무시')

    cases = [
        ("cfg{bank_dk_frac} + real_estimator='v2_konly'",
         lambda: SequentialUpdater({'bank_dk_frac': 0.4,
                                    'real_estimator': 'v2_konly'})),
        ("cfg{bank_k_set} + real_estimator='v2_konly'",
         lambda: SequentialUpdater({'bank_k_set': (kx, ky),
                                    'real_estimator': 'v2_konly'})),
        ('fit(dk_c=..., dk_frac=...) 동시',
         lambda: post(*SC[recs[0]][:2], SC[recs[0]][2], SC[recs[0]][3],
                      dk_c=0.02, dk_frac=0.4)),
        ('fit(k_set=..., dk_frac=...) 동시',
         lambda: post(*SC[recs[0]][:2], SC[recs[0]][2], SC[recs[0]][3],
                      k_set=(kx, ky), dk_frac=0.4)),
        ('fit(k_set=빈 배열)',
         lambda: post(*SC[recs[0]][:2], SC[recs[0]][2], SC[recs[0]][3],
                      k_set=(np.zeros(0), np.zeros(0)))),
        ('fit(k_set=길이 불일치)',
         lambda: post(*SC[recs[0]][:2], SC[recs[0]][2], SC[recs[0]][3],
                      k_set=(kx, ky[:1]))),
    ]
    for tag, fn in cases:
        ncmp += 1
        expect_raise(tag, fn)
    # ★항진 방지: 정상 조합은 반드시 통과해야 한다 (가드가 전부를 막으면 팔이 빈다)
    for tag, cfg in (("cfg{bank_dk_frac} + 기본 real_estimator", {'bank_dk_frac': 0.4}),
                     ("cfg{bank_k_set} + real_estimator='bank_mmse'",
                      {'bank_k_set': (kx, ky), 'real_estimator': 'bank_mmse'}),
                     ("cfg{} (키 없음) + real_estimator='v2_konly'",
                      {'real_estimator': 'v2_konly'})):
        ncmp += 1
        try:
            u = SequentialUpdater(dict(cfg))
            for k in ('bank_dk_frac', 'bank_k_set'):
                if k in cfg:
                    assert u.config.get(k, '<dropped>') is cfg[k], f'{k} 가 삼켜졌다'
            print(f'  {tag:52s} 통과 OK', flush=True)
        except Exception as e:                                  # noqa: BLE001
            fails.append(f'L5 {tag}: ⛔정상 조합이 막혔다 — {type(e).__name__}: {e}')

    print(f'\n비교 {ncmp} 건')
    print(f'regress_bank_mmse: {"PASS" if not fails else "FAIL"} ({len(fails)} 실패)')
    for f in fails:
        print('  -', f)
    return 0 if not fails else 1


if __name__ == '__main__':
    sys.exit(main())
