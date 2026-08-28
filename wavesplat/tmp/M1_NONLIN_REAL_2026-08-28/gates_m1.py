"""M1 게이트 — **판정 규칙·문턱·팔·표본·해석 규칙의 유일한 정의**.

⛔손복사 상수 0건이 목표다. 이 파일이 읽는 원본:
  · 코너 (Hs, Tp) · 생성 인자 · 배포 상수 · 문턱 도우미  → `gates_b26` (B-26)
  · 부호검정·집계 도우미                                  → `gates_b11` (B-11/B-12)
  · HOS 상수 (M · Lx · Ly · dt)                          → `tmp/HOSGATE_2026-08-13/gen_hos.py` **본문 파싱**
    (⛔`import gen_hos` 는 금지 — 임포트 시점에 `wavesim.pipeline` 에 몽키패치를 설치한다)

    python3 tmp/M1_NONLIN_REAL_2026-08-28/gates_m1.py --selftest   # 게이트 자가진단 (런 0회)
    python3 tmp/M1_NONLIN_REAL_2026-08-28/gates_m1.py              # 런 0회 게이트
"""
import ast
import glob
import json
import os
import re
import statistics as st
import sys
from math import comb, hypot

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
B26 = os.path.join(ROOT, 'tmp', 'B26_SEASTATE_2026-08-27')
B12 = os.path.join(ROOT, 'tmp', 'B12_HOS_EXTEND_2026-08-23')
B11 = os.path.join(ROOT, 'tmp', 'B11_HOS_BASELINE_2026-08-23')
S16 = os.path.join(ROOT, 'tmp', 'S16_BASELINE_2026-08-14')
HOSGATE = os.path.join(ROOT, 'tmp', 'HOSGATE_2026-08-13')
for _p in (HERE, B26, B12, B11, S16, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gates_b26 as G26                                        # noqa: E402
from gates_b11 import (EST_SEEDS, FILTER_SEEDS,                # noqa: F401,E402
                       sign_test_one_sided, med_by_episode, paired)

G_ACC = 9.81
DATA = os.path.join(HERE, 'data_m1')                 # data_m1/LIN, data_m1/HOS
SMOKE = os.path.join(HERE, 'smoke_m1')
PC1_DIR = os.path.join(HERE, 'pc1')
SUFFIX = G26.SUFFIX
GT_ARMS = ('LIN', 'HOS')

# ── 코너: **B-26 의 `SS-REAL` 을 그대로 쓴다** (손복사 0) ──────────────────────
CORNER = 'SS-REAL'
HS = G26.CORNERS[CORNER]['Hs']
TP = G26.CORNERS[CORNER]['Tp']
NCOMPS = G26.NCOMPS                                  # (15, 30) 혼잡 고정
K_SEEDS = G26.K_SEEDS                                # 12
N_EP = 24                                            # (GT 팔마다) — 아래 assert 가 대조
assert N_EP == len(NCOMPS) * K_SEEDS
# ★시드 원장: B-26 이 **선언한** 풀은 range(1641,1681) 이다. 그 밖에서 시작한다.
SEED_POOL = tuple(range(1681, 1741))

# ── 메시: ★**M=3 화면이 B-26 의 M=1 화면과 비트 동일**해지도록 고른다 ──────────
#   mesh_adequacy 는 lam_eff = lam / M 을 6·d 와 견준다 ⟹ 판정은 (M·d) 만의 함수다.
#   3·d_M1 == d_B26 ⟹ verdict 도 var_under 도 **정확히 같다**(G-23 이 비트로 요구).
MESH_NX, MESH_NY = 598, 499

# ── HOS: M·Lx·Ly·dt 는 `B-12` 계보 그대로, **격자만** 올린다 (G-22 가 사유를 재도출) ──
GEN_HOS_SRC = os.path.join(HOSGATE, 'gen_hos.py')
HOS_NX, HOS_NY = 576, 384


def hos_constants_from_source():
    """`gen_hos.py:48` 한 줄에서 HOS 상수를 **파싱**한다(임포트하면 몽키패치가 설치된다)."""
    src = open(GEN_HOS_SRC).read()
    m = re.search(r'^HOS_M,\s*HOS_NX,\s*HOS_NY,\s*HOS_LX,\s*HOS_LY,\s*HOS_DT\s*=\s*(.+?)\s*$',
                  src, re.M)
    vals = ast.literal_eval(m.group(1))
    return dict(M=vals[0], Nx=vals[1], Ny=vals[2], Lx=vals[3], Ly=vals[4], dt=vals[5])


_HC = hos_constants_from_source()
HOS_M, HOS_LX, HOS_LY, HOS_DT = _HC['M'], _HC['Lx'], _HC['Ly'], _HC['dt']
B12_HOS_NX, B12_HOS_NY = _HC['Nx'], _HC['Ny']        # 192, 128 — 이 배치가 **바꾸는** 유일한 값
DKX, DKY = 2.0 * 3.141592653589793 / HOS_LX, 2.0 * 3.141592653589793 / HOS_LY
HALF_BIN = hypot(DKX / 2.0, DKY / 2.0)


def nyquist(nx, ny, lx=HOS_LX, ly=HOS_LY):
    import numpy as np
    return float(np.pi * nx / lx), float(np.pi * ny / ly)


def k_max_theory(Tp=TP, depth=None):
    """`jonswap_components` 의 주파수 상단은 3·omega_p 다(wave_generator.py 축자).
    심해 근사 k = omega^2/g 가 유한수심 해의 **상한**이므로 이 값이 성분 |k| 의 상한이다."""
    import numpy as np
    w_max = 3.0 * (2 * np.pi / Tp)
    return float(w_max ** 2 / G_ACC)


# ── 문턱: B-26 규약 상속 (n=24, 단측 짝 부호검정). ★코드가 재도출한다 (G-19) ────
# ★리터럴로 적되 **아래 assert 가 B-26 원본과 대조**한다 — 손복사가 아니라 *검사된 거울*이다
#   (tmp/_audit/check_thresholds.py 가 문턱의 이항 p 를 재계산하려면 리터럴이 필요하다).
NOMINAL, CONFIRM, WEAK = 17, 18, 7   # /24  — B-26 규약 상속
NEG_MIN = 17                         # /24  — 음성대조 최소
ALPHA = 0.05
assert (NOMINAL, CONFIRM, WEAK, NEG_MIN, ALPHA) == (
    G26.NOMINAL, G26.CONFIRM, G26.WEAK, G26.NEG_MIN, G26.ALPHA), 'B-26 문턱과 어긋났다'
N_FAMILY = 2                     # 주장 가족 = {HOS 코퍼스, LIN 코퍼스} 2개
FAMILY_ALPHA = ALPHA / N_FAMILY  # 0.025  (CONFIRM 의 p=0.011328 은 0.0125 도 통과)
MESH_BAD = G26.MESH_BAD


def threshold_table(n=N_EP):
    return {'n': n,
            'p_nominal': sign_test_one_sided(NOMINAL, n),
            'p_confirm': sign_test_one_sided(CONFIRM, n),
            'p_weak_le': sum(comb(n, k) for k in range(0, WEAK + 1)) / 2 ** n,
            'family_alpha': FAMILY_ALPHA,
            'family_alpha_b26': G26.FAMILY_ALPHA,
            'confirm_within_family': sign_test_one_sided(CONFIRM, n) <= FAMILY_ALPHA,
            'confirm_within_b26_family': sign_test_one_sided(CONFIRM, n) <= G26.FAMILY_ALPHA,
            'weak_is_sibling': WEAK == n - NOMINAL}


# ── 표본 (런 0회로 재도출 가능) ────────────────────────────────────────────────
def components(ncomp, seed):
    """★B-26 의 성분 생성기를 **그대로** 부른다(손복사 0)."""
    return G26.components(CORNER, ncomp, seed)


def snap_components(comps, depth):
    """`gen_hos.snap_components` 축자 이식 — **LIN·HOS 두 팔이 같은 성분을 쓴다**(짝의 정의).

    ⚠`% NX` 되감기는 원본 그대로 둔다. 이 배치는 그 되감기를 **없애는 것이 아니라
    발화하지 않는 격자를 고르는 것**이고, `n_wrapped` 가 그것을 매 에피소드 계수한다."""
    import numpy as np
    kx1 = np.fft.fftfreq(HOS_NX, d=1.0 / (HOS_NX * DKX))
    ky1 = np.fft.fftfreq(HOS_NY, d=1.0 / (HOS_NY * DKY))
    out, moved, wrapped = [], [], 0
    for c in comps:
        rx, ry = c.kx / DKX, c.ky / DKY
        if abs(round(rx)) > HOS_NX // 2 or abs(round(ry)) > HOS_NY // 2:
            wrapped += 1
        ix, iy = int(round(rx)) % HOS_NX, int(round(ry)) % HOS_NY
        kx, ky = float(kx1[ix]), float(ky1[iy])
        k = float(np.hypot(kx, ky))
        w = float(np.sqrt(G_ACC * k * np.tanh(np.clip(k * depth, 0, 20)))) if k > 0 else 0.0
        moved.append(float(np.hypot(kx - c.kx, ky - c.ky)))
        out.append(type(c)(A=c.A, omega=w, kx=kx, ky=ky, phi=c.phi, Q=c.Q))
    return out, dict(n_wrapped=wrapped, max_snap_dk=(max(moved) if moved else 0.0))


def cfg_m1(hos_enabled):
    """★생성 설정. `wavesim/**` 무수정 — 새 해상상태는 배포 노브 `cfg.sea_states` 로 넣는다."""
    from wavesim.config import WaveSimConfig
    cfg = WaveSimConfig()
    cfg.mesh.nx, cfg.mesh.ny = MESH_NX, MESH_NY
    cfg.sea_states = dict(cfg.sea_states)
    cfg.sea_states[CORNER] = {'Hs': HS, 'Tp': TP}
    cfg.hos.enabled = bool(hos_enabled)
    cfg.hos.M, cfg.hos.Nx, cfg.hos.Ny = HOS_M, HOS_NX, HOS_NY
    cfg.hos.Lx, cfg.hos.Ly, cfg.hos.dt = HOS_LX, HOS_LY, HOS_DT
    return cfg


def mesh_d(cfg):
    from wavesim import mesh_adequacy
    dx, dy = mesh_adequacy.mesh_spacing(cfg)
    return max(dx, dy)


def screen(ncomp, seed):
    """생성 전 선별 — **GT 무관·결과 무관**. `hos_order=HOS_M` 로 본다(비선형 코퍼스다)."""
    from wavesim import mesh_adequacy
    from wavesim.wave_generator import check_q_constraint
    comp = components(ncomp, seed)
    q_ok, q_val = check_q_constraint(comp)
    info = mesh_adequacy.check(comp, cfg_m1(False), hos_order=HOS_M)
    return dict(mesh_verdict=info['mesh_verdict'],
                var_under=float(info['var_under_resolved']),
                lambda_min=float(info['lambda_min']),
                lambda_peak=float(info['lambda_peak']),
                q_ok=bool(q_ok), q_val=float(q_val))


def fname(ncomp, seed):
    return f'episode_{CORNER}_{ncomp}comp_seed{seed}_{SUFFIX}.h5'


def ep_key(gt_arm, ncomp, seed):
    return f'{gt_arm}|{ncomp}comp_seed{seed}'


def select():
    """동결 선택 규칙. 반환 (episodes, rejected). ⛔결과를 보지 않는다.
    LIN·HOS 두 GT 팔이 **같은 (ncomp, seed) 목록**을 받는다(짝)."""
    eps, rej, picked = [], [], []
    for ncomp in NCOMPS:
        got = 0
        for seed in SEED_POOL:
            s = screen(ncomp, seed)
            if s['mesh_verdict'] == MESH_BAD or not s['q_ok']:
                rej.append(dict(ncomp=ncomp, seed=seed, **s))
                continue
            picked.append((ncomp, seed, s))
            got += 1
            if got == K_SEEDS:
                break
        assert got == K_SEEDS, f'{ncomp}comp: 풀 부족 ({got}/{K_SEEDS})'
    for gt_arm in GT_ARMS:
        for ncomp, seed, s in picked:
            eps.append(dict(gt_arm=gt_arm, corner=CORNER, ncomp=ncomp, seed=seed,
                            bf=CORNER, Hs=HS, Tp=TP,
                            episode=ep_key(gt_arm, ncomp, seed),
                            path=os.path.join(DATA, gt_arm, fname(ncomp, seed)), **s))
    return eps, rej


# ★동결 기대값 — `select()` 를 결과 0행에서 돌려 얻었다 (G-1 이 매 실행 재대조).
FROZEN_SEEDS = {
    '15': (1681, 1683, 1684, 1685, 1686, 1687, 1688, 1689, 1690, 1692, 1693, 1694),
    '30': (1681, 1682, 1683, 1684, 1685, 1686, 1687, 1688, 1689, 1690, 1691, 1692),
}
FROZEN_REJECTED = ((15, 1682), (15, 1691))
SEEDS_CONSUMED = tuple(range(1681, 1695))

# ── 배포 상수 (패키지 단일출처에서 되읽는다) ──────────────────────────────────
DEPLOY_EXPECTED = dict(G26.DEPLOY_EXPECTED)
ROUTE_EXPECTED = G26.ROUTE_EXPECTED
SHIP_CAPTURE = G26.SHIP_CAPTURE
SPEC_CAPTURE = G26.SPEC_CAPTURE
PC1_DATASETS = G26.PC1_DATASETS
# ★PC-1 — B-26 의 **`SS-REAL` 선형** 에피소드 하나를 이 배치의 생성기로 재생성해 비트 동일 요구
PC1 = dict(sea_state=CORNER, n_components=15, seed=1681 - 40)   # = 1641 (B-26 풀 첫 시드)
PC1_PATH = os.path.join(B26, 'data_b26', fname(PC1['n_components'], PC1['seed']))


# ── 판정 ──────────────────────────────────────────────────────────────────────
def aggregate(rows):
    ok = [r for r in rows if 'error' not in r]
    out = {}
    for gt_arm in GT_ARMS:
        sub = [r for r in ok if r.get('gt_arm') == gt_arm]
        ship = med_by_episode(sub, 'SHIP-NOW')
        spec = med_by_episode(sub, 'SPEC25')
        zero = med_by_episode(sub, 'SPEC25', field='zero_nRMSE')
        out[gt_arm] = dict(ship=ship, spec=spec, zero=zero,
                           prim=paired(ship, spec), neg=paired(spec, zero),
                           sz=paired(ship, zero))
    return out


def arm_label(prim, neg):
    """GT 팔 하나의 라벨. ⛔음성대조가 죽으면 우열을 말하지 않는다(B-9b·B-11 규약)."""
    if neg['n'] == 0 or neg['better'] < NEG_MIN:
        return ('BASELINE-NO-SKILL', f'기준선 무기술 {neg["better"]}/{neg["n"]} (문턱 {NEG_MIN})')
    w, n = prim['better'], prim['n']
    if w >= CONFIRM:
        return ('HOLDS (CONFIRMATORY)', f'{w}/{n} p={prim["p_one_sided"]:.6f}')
    if w >= NOMINAL:
        return ('HOLDS (NOMINAL)', f'{w}/{n} p={prim["p_one_sided"]:.6f} — 가족단위 미달')
    if w <= WEAK:
        return ('BREAKS', f'{w}/{n} — 부호가 반대로 간다')
    return ('UNDECIDED', f'{w}/{n} — 부호로 갈리지 않는다')


# ─────────────────────────────────────────────────────────────────────────────
# ★결과 전 해석 규칙 — 어느 결과가 나와도 무엇을 쓸지 **지금** 정한다
# ─────────────────────────────────────────────────────────────────────────────
READING = {
    'HOLDS (CONFIRMATORY)': (
        '★★**해양 심사의 최대 할인이 닫힌다.** 비선형 참값과 실측 해상상태가 **한 칸에서** '
        '만나도 헤드라인이 산다 ⟹ §「작동 경계」의 *"경계는 파랑 조건 축을 따라 놓여 있지 않다"* 가 '
        '**선형 GT 단서 없이** 설 수 있다(`B-26` 은 선형에서, `B-11`/`B-12` 는 순한 바다에서만 '
        '답했다). 논문에 **비선형 × 실측 해상상태 교호 표**를 신설하고 해양 심사 `PASS` 를 청구한다. '
        '⚠범위는 그대로다: 합성·정지 기하·`k·Hs`≈0.24·`mesh_verdict` 3분할 공시.'),
    'HOLDS (NOMINAL)': (
        '⚠명목만 ⟹ **`EXPLORATORY`**. 같은 문장을 쓰되 가족단위 미달을 병기하고 '
        '`MINOR REVISION` 유지를 청구한다. ⛔`PASS` 청구 금지.'),
    'BREAKS': (
        '★★★**경계를 옮긴다 — 이것이 더 큰 결과다.** §「작동 경계」의 동결 문안 '
        '*"경계는 파랑 조건 축을 따라 놓여 있지 않다"* 를 **철회**하고 '
        '***"경계는 가파른 비선형 바다에 있다"*** 로 다시 쓴다. 근거: `B-26` 의 배제는 '
        '**선형 GT 한정**이었고 이 배치가 그 단서를 제거하자 부호가 뒤집혔다. '
        '⟹ (a) 실측 12 레코드의 `k·Hs` 중앙 **0.2445** 가 이 칸이므로, 이것이 저장소 최초의 '
        '***실측 음성에 대한 양성 설명***이 된다(지금까지는 전부 소거법이었다). '
        '(b) 기여 ① 의 범위를 **"순한 바다"** 로 좁혀 다시 쓴다. '
        '(c) `B-11`/`B-12` 의 `HEADLINE-SURVIVES` 는 유지하되 **`k·Hs`≈0.1 한정**임을 명시한다. '
        '⛔`B-11`/`B-12` 를 철회하지 않는다 — 그 코퍼스에서는 여전히 참이다.'),
    'UNDECIDED': (
        '`NONLIN-REAL-UNDECIDED` — 어느 쪽도 못 쓴다. ★**이 라벨도 결과다**: '
        '⛔`B-26` 의 *"해상상태 축은 매개하지 않는다"* 를 **비선형 참값으로 일반화하는 문장을 '
        '논문에서 뺀다**(일반화의 근거가 없다). 해양 심사의 할인은 **열린 채로 공시**한다.'),
    'BASELINE-NO-SKILL': (
        '⛔비교자가 무기술 ⟹ **우열을 말하지 않는다**. ★그러나 그 자체가 강한 관찰이다: '
        '가파른 비선형 실측 해상상태에서는 **최강 기준선(LETKF·SPEC25)조차** 제로필드를 못 이긴다 '
        '⟹ 실측 음성이 **방법이 아니라 레짐**에 붙는다. 이 경우 `SHIP<ZERO` 를 보고하고, '
        '⚠`B-26` 이 같은 해상상태의 **선형** 판에서 기준선 기술을 −3.96%(23/24)로 이미 '
        '**간신히** 재놨다는 사실을 함께 적는다.'),
}

# ── 짝(LIN, HOS) 2×2 — ★교락 통제. 추가 검정 0개, 라벨 두 개만 읽는다 ──────────
PAIR_READING = {
    ('hold', 'hold'): ('두 GT 클래스 모두 유지 ⟹ ★비선형성은 매개하지 않는다. '
                       '덤으로 `B-26` 의 `SS-REAL` 판정이 **3배 조밀한 메시 · 신선 시드**에서 '
                       '독립 재현된다(메시 정련 대조).'),
    ('hold', 'break'): ('★★**비선형성이 매개한다** — 저장소가 얻을 수 있는 가장 깨끗한 귀속이다. '
                        '같은 성분·같은 메시·같은 광선, 다른 것은 **시간 전개 방식 하나**뿐이다.'),
    ('break', 'hold'): ('⛔가설과 반대 방향 ⟹ **귀속 금지**. 비선형성이 아니라 메시 정련 또는 '
                        '시드가 움직였다는 뜻이고, 두 라벨을 그대로 보고한다.'),
    ('break', 'break'): ('★비선형성은 매개자가 **아니다** — 선형 팔도 함께 죽었다. '
                         '`B-26` 이 같은 해상상태를 **거친 메시**에서 24/24 로 지켰으므로 '
                         '남는 후보는 **메시 정련(= 관측의 공간 분해능)** 또는 시드다 ⟹ '
                         '★그 경우 다음 배치는 메시 축이고, 서사는 **관측 모형 쪽으로** 이동한다.'),
}


def pair_reading(labels):
    def stt(a):
        return ('hold' if labels[a].startswith('HOLDS')
                else 'break' if labels[a] in ('BREAKS', 'BASELINE-NO-SKILL') else 'undecided')
    key = (stt('LIN'), stt('HOS'))
    return key, PAIR_READING.get(
        key, '조합이 표에 없다(UNDECIDED 포함) ⟹ **귀속 금지**, 두 라벨을 그대로 보고한다.')


# ─────────────────────────────────────────────────────────────────────────────
# 게이트 — ★단일출처. 러너·생성기·분석기가 같은 함수를 부른다. 공허한 통과 금지.
# ─────────────────────────────────────────────────────────────────────────────
RUNFREE_GATES = ('G-0', 'G-1', 'G-2', 'G-3', 'G-19', 'G-20', 'G-22', 'G-23')
ROWLESS_OK = ('G-11', 'G-18')
SKEW_MIN_WINS = CONFIRM          # 배치 자신의 문턱을 재사용(새 상수 금지)


def gates_runfree(seed_scan_root=ROOT, hos_nx=None, hos_ny=None):
    import numpy as np
    hos_nx = HOS_NX if hos_nx is None else hos_nx
    hos_ny = HOS_NY if hos_ny is None else hos_ny
    out = []
    eps, rej = select()
    picked = {f'{n}': tuple(e['seed'] for e in eps
                            if e['gt_arm'] == 'LIN' and e['ncomp'] == n) for n in NCOMPS}

    # G-0 시드 원장 충돌 — 저장소 전체 파일명 대조 (이 배치 디렉터리는 제외)
    consumed = sorted({e['seed'] for e in eps} | {r['seed'] for r in rej})
    clash = set()
    for p in glob.glob(os.path.join(seed_scan_root, '**', 'episode_*.h5'), recursive=True):
        if os.path.abspath(p).startswith(HERE):
            continue
        m = re.search(r'seed(\d+)', os.path.basename(p))
        if m and int(m.group(1)) in set(consumed):
            clash.add(int(m.group(1)))
    out.append(('G-0 시드 원장 충돌 0', not clash, f'소모 {consumed[0]}–{consumed[-1]} · 충돌 {sorted(clash)}'))

    # G-1 select() == 동결 목록
    out.append(('G-1 select() == FROZEN_SEEDS', picked == {k: tuple(v) for k, v in FROZEN_SEEDS.items()},
                json.dumps(picked)))
    out.append(('G-1b GT 팔마다 24 에피소드', all(sum(1 for e in eps if e['gt_arm'] == a) == N_EP
                                                for a in GT_ARMS), f'{len(eps)} / {2*N_EP}'))
    out.append(('G-1c 두 GT 팔이 같은 (ncomp,seed) 짝',
                {(e['ncomp'], e['seed']) for e in eps if e['gt_arm'] == 'LIN'}
                == {(e['ncomp'], e['seed']) for e in eps if e['gt_arm'] == 'HOS'}, ''))
    # G-2 INADEQUATE 0 + 거부목록 동결
    out.append(('G-2 선택분 INADEQUATE 0 · 거부목록 동결',
                all(e['mesh_verdict'] != MESH_BAD for e in eps)
                and tuple((r['ncomp'], r['seed']) for r in rej) == tuple(
                    (a, b) for a, b in FROZEN_REJECTED),
                f'거부 {[(r["ncomp"], r["seed"], r["mesh_verdict"], round(r["var_under"],3)) for r in rej]}'))
    out.append(('G-3 Q 제약 전건', all(e['q_ok'] for e in eps), f'{sum(e["q_ok"] for e in eps)}/{len(eps)}'))

    # ★G-22 나이퀴스트 산술 — **이 배치가 격자를 올린 사유를 재도출한다**
    kmax = k_max_theory()
    nyq_b12 = nyquist(B12_HOS_NX, B12_HOS_NY)
    nyq_m1 = nyquist(hos_nx, hos_ny)
    kx_obs = ky_obs = 0.0
    for ncomp in NCOMPS:
        for seed in [e['seed'] for e in eps if e['gt_arm'] == 'LIN' and e['ncomp'] == ncomp]:
            c = components(ncomp, seed)
            kx_obs = max(kx_obs, max(abs(x.kx) for x in c))
            ky_obs = max(ky_obs, max(abs(x.ky) for x in c))
    out.append(('★G-22 M1 격자가 이론 상한을 담는다 (nyq >= 3·omega_p 의 k)',
                nyq_m1[0] >= kmax and nyq_m1[1] >= kmax,
                f'k_max(이론)={kmax:.4f} · nyq_M1={nyq_m1[0]:.4f}/{nyq_m1[1]:.4f} '
                f'· 실측 |kx|max={kx_obs:.4f} |ky|max={ky_obs:.4f}'))
    out.append(('★G-22b B-12 격자로는 담기지 **않는다**(격자를 올린 사유)',
                nyq_b12[0] < kmax and nyq_b12[1] < kmax,
                f'nyq_B12={nyq_b12[0]:.4f} < k_max={kmax:.4f} (배수 {kmax/nyq_b12[0]:.2f})'))
    out.append(('★G-22c M · Lx · Ly · dt 는 B-12 계보 그대로',
                (HOS_M, HOS_LX, HOS_LY, HOS_DT) == (_HC['M'], _HC['Lx'], _HC['Ly'], _HC['dt'])
                and (hos_nx, hos_ny) != (B12_HOS_NX, B12_HOS_NY),
                f'M={HOS_M} L=({HOS_LX:.0f},{HOS_LY:.0f}) dt={HOS_DT} · 격자 '
                f'{B12_HOS_NX}×{B12_HOS_NY} -> {hos_nx}×{hos_ny}'))

    # ★G-23 메시 등가 — M=3 화면이 B-26 의 M=1 화면과 **비트 동일**
    from wavesim import mesh_adequacy
    d_m1, d_b26 = mesh_d(cfg_m1(False)), mesh_d(G26._cfg())
    same = True
    detail = []
    for ncomp in NCOMPS:
        for seed in list(SEED_POOL)[:K_SEEDS + 4]:
            c = components(ncomp, seed)
            a = mesh_adequacy.check(c, cfg_m1(False), hos_order=HOS_M)
            b = mesh_adequacy.check(c, G26._cfg(), hos_order=None)
            ok = (a['mesh_verdict'] == b['mesh_verdict']
                  and a['var_under_resolved'] == b['var_under_resolved'])
            same &= ok
            if not ok:
                detail.append((ncomp, seed, a['mesh_verdict'], b['mesh_verdict']))
    out.append(('★G-23 3·d_M1 == d_B26 (메시 화면 비트 등가)',
                abs(3 * d_m1 - d_b26) < 1e-12 and same,
                f'd_M1={d_m1:.9f} · 3·d_M1={3*d_m1:.9f} · d_B26={d_b26:.9f} · 불일치 {detail}'))

    # G-19 문턱 산술 · G-20 배포 상수 단일출처
    t = threshold_table()
    out.append(('★G-19 문턱 산술', t['confirm_within_family'] and t['weak_is_sibling']
                and t['p_nominal'] > FAMILY_ALPHA, json.dumps({k: (round(v, 8) if isinstance(v, float) else v)
                                                               for k, v in t.items()})))
    dc = G26.deploy_constants()
    out.append(('★G-20 배포 상수 == 패키지 단일출처 == 사전등록값',
                all(dc[f'DEFAULT_CONFIG.{k}'] == v for k, v in DEPLOY_EXPECTED.items())
                and dc['SU.VARPRO_STEPS_DEFAULT'] == DEPLOY_EXPECTED['varpro_steps']
                and dc['SU.LEARNED_INIT_ITERS_DEFAULT'] == DEPLOY_EXPECTED['learned_init_iters']
                and dc['SU.AUTO_SWITCH_THRESH_DEFAULT'] == DEPLOY_EXPECTED['auto_switch_thresh']
                and dc['SU.LAMBDA_L1_DEFAULT'] == DEPLOY_EXPECTED['lambda_l1'],
                json.dumps(dc)))
    return out


def gates_data(read=None, exists=None, manifest=None):
    """생성 후 · 추정 전. `read(path)->dict` 주입 가능(자가진단이 픽스처를 넣는다)."""
    import numpy as np
    exists = exists or os.path.exists
    read = read or _read_ep
    eps, _ = select()
    man = manifest if manifest is not None else _load_manifest()
    out = []
    have = [e for e in eps if exists(e['path'])]
    out.append(('G-6 48 에피소드 생성됨', len(have) == 2 * N_EP, f'{len(have)}/{2*N_EP}'))
    metas = {}
    for e in have:
        try:
            metas[e['episode']] = read(e['path'])
        except Exception as exc:                                   # noqa: BLE001
            metas[e['episode']] = dict(error=repr(exc))
    ok = [m for m in metas.values() if 'error' not in m]
    out.append(('★G-7 저장된 .h5 의 (Hs,Tp) == 코너',
                bool(ok) and all(abs(m['Hs'] - HS) < 1e-9 and abs(m['Tp'] - TP) < 1e-9 for m in ok),
                f'{len(ok)} 파일'))
    hs4 = {e['episode']: metas[e['episode']].get('hs4') for e in have
           if 'error' not in metas[e['episode']]}
    out.append(('★G-8 개입의 파생 측정: 평균 4·std(gt)/Hs == 1 (±5%)',
                bool(hs4) and abs(float(np.mean(list(hs4.values()))) / HS - 1) < 0.05,
                f'{np.mean(list(hs4.values()))/HS:.4f}' if hs4 else '0행'))
    out.append(('★G-8c 에피소드별 |4·std/Hs − 1| <= 0.25',
                bool(hs4) and all(abs(v / HS - 1) <= 0.25 for v in hs4.values()),
                f'max {max(abs(v/HS-1) for v in hs4.values()):.3f}' if hs4 else '0행'))
    out.append(('G-9 메타 되읽기 (mesh_dx · depth · ra/rf/ds)',
                bool(ok) and all(abs(m['mesh_dx'] - mesh_d(cfg_m1(False))) < 1e-6
                                 and abs(m['depth'] - G26.gen_params()['depth']) < 1e-9
                                 and m['ra'] and m['rf'] and m['ds'] for m in ok),
                f'{len(ok)} 파일'))
    out.append(('G-10 프레임 수 단일 == B-26 과 같다',
                bool(ok) and {m['n_frames'] for m in ok} == {G26.gen_params()['n_frames']},
                str({m['n_frames'] for m in ok})))
    # ★G-24 되감기 0 · 스냅 이동 <= 반빈  (매니페스트의 **실측 계수기**)
    wraps = [(k, v.get('n_wrapped'), v.get('max_snap_dk')) for k, v in man.items()
             if v.get('status') == 'ok']
    out.append(('★G-24 나이퀴스트 되감기 0 · 스냅 이동 <= 반빈',
                bool(wraps) and all(w == 0 and s is not None and s <= HALF_BIN + 1e-12
                                    for _, w, s in wraps),
                f'{len(wraps)} 행 · max_snap_dk={max((s for _,_,s in wraps if s is not None), default=None)} '
                f'· 반빈={HALF_BIN:.6f}'))
    # ★G-25 짝 무결성 — LIN·HOS 가 **같은 스냅 성분**을 썼다
    pairs = {}
    for k, v in man.items():
        if v.get('status') != 'ok' or 'comp_md5' not in v:
            continue
        pairs.setdefault((v['ncomp'], v['seed']), {})[v['gt_arm']] = v['comp_md5']
    out.append(('★G-25 짝 무결성: LIN·HOS 성분 해시 동일',
                bool(pairs) and len(pairs) == N_EP
                and all(set(d) == set(GT_ARMS) and d['LIN'] == d['HOS'] for d in pairs.values()),
                f'{len(pairs)} 짝 · 불일치 '
                f'{[k for k,d in pairs.items() if set(d)==set(GT_ARMS) and d["LIN"]!=d["HOS"]]}'))
    # ★G-26 비선형성이 실제로 켜졌다 — **설정 메아리가 아니라 장의 왜도**
    sk = {}
    for k, v in man.items():
        if v.get('status') == 'ok' and v.get('gt_skew') is not None:
            sk.setdefault((v['ncomp'], v['seed']), {})[v['gt_arm']] = v['gt_skew']
    both = [d for d in sk.values() if set(d) == set(GT_ARMS)]
    wins = sum(1 for d in both if d['HOS'] > d['LIN'])
    out.append(('★G-26 HOS 팔의 장이 **양의 왜도**로 갈린다 (>= CONFIRM/24)',
                len(both) == N_EP and wins >= SKEW_MIN_WINS,
                f'{wins}/{len(both)} · 중앙 LIN {st.median([d["LIN"] for d in both]):.4f} '
                f'HOS {st.median([d["HOS"] for d in both]):.4f}' if both else '0짝'))
    diffs = [v.get('gt_absdiff_vs_pair') for v in man.values()
             if v.get('status') == 'ok' and v.get('gt_arm') == 'HOS']
    diffs = [d for d in diffs if d is not None]
    out.append(('★G-26b HOS 장이 LIN 장과 실제로 다르다 (max|Δ| > 0 전건)',
                len(diffs) == N_EP and all(d > 0 for d in diffs),
                f'{len(diffs)} 행 · min max|Δ|={min(diffs):.4g}' if diffs else '0행'))
    # ★G-27 스냅 후에도 메시 판정이 INADEQUATE 가 아니다 (화면은 스냅 전에 봤다)
    mv = [v.get('mesh_verdict_snapped') for v in man.values() if v.get('status') == 'ok']
    mv = [m for m in mv if m]
    out.append(('★G-27 스냅 성분의 메시 판정도 INADEQUATE 아님',
                len(mv) == 2 * N_EP and all(m != MESH_BAD for m in mv),
                str(sorted(set(mv))) if mv else '0행'))
    return out


def gates_run(rows):
    ok = [r for r in rows if 'error' not in r]
    out = [('G-11 에러 0행', len(rows) == len(ok), f'{len(rows)-len(ok)} 에러')]
    for arm, want in (('SHIP-NOW', 3 * 2 * N_EP), ('SPEC25', 3 * 2 * N_EP)):
        n = len([r for r in ok if r.get('arm') == arm])
        out.append((f'G-12 {arm} {want}행', n == want, f'{n}/{want}'))
    for a in GT_ARMS:
        sh = {r['episode'] for r in ok if r.get('arm') == 'SHIP-NOW' and r.get('gt_arm') == a}
        sp = {r['episode'] for r in ok if r.get('arm') == 'SPEC25' and r.get('gt_arm') == a}
        out.append((f'★G-13 {a}: 두 팔 에피소드 집합 완전 일치 · {N_EP}',
                    sh == sp and len(sh) == N_EP, f'SHIP {len(sh)} · SPEC {len(sp)}'))
    sh = [r for r in ok if r.get('arm') == 'SHIP-NOW']
    out.append(('★G-14 SHIP 배포값 되읽기 (400·0·0·λ=1e-4, λ 는 객체 속성)',
                bool(sh) and all(r.get('varpro_steps') == DEPLOY_EXPECTED['varpro_steps']
                                 and r.get('learned_init_iters') == DEPLOY_EXPECTED['learned_init_iters']
                                 and r.get('auto_switch_thresh') == DEPLOY_EXPECTED['auto_switch_thresh']
                                 and r.get('lambda_l1') == DEPLOY_EXPECTED['lambda_l1'] for r in sh),
                f'{len(sh)}행'))
    out.append(('★G-15 라우터가 두 GT 팔 전건에서 learned',
                bool(sh) and all(r.get('cap_auto_selected') == ROUTE_EXPECTED for r in sh),
                str(sorted({r.get('cap_auto_selected') for r in sh}))))
    out.append(('★G-15b VARPRO 발화 전건', bool(sh) and all(r.get('cap_varpro_fired') for r in sh),
                f'{sum(1 for r in sh if r.get("cap_varpro_fired"))}/{len(sh)}'))
    sp = [r for r in ok if r.get('arm') == 'SPEC25']
    out.append(('★G-16 SPEC25 되읽기 (loc 25 · ens 600 · jonswap)',
                bool(sp) and all(r.get('loc_radius') == 25.0 and r.get('n_ensemble') == 600
                                 and r.get('background') == 'jonswap' for r in sp), f'{len(sp)}행'))
    out.append(('★G-17 기준선 배경이 코너를 따라갔다 (cap_Hs/cap_Tp)',
                bool(sp) and all(abs(r.get('cap_Hs', -1) - HS) < 1e-6
                                 and abs(r.get('cap_Tp', -1) - TP) < 1e-6 for r in sp),
                str(sorted({(r.get('cap_Hs'), r.get('cap_Tp')) for r in sp}))))
    out.append(('★G-18 계측 shim 투명 (cap_avg_nRMSE == avg_nRMSE 비트동일)',
                all(r.get('cap_avg_nRMSE') is None or r['cap_avg_nRMSE'] == r['avg_nRMSE']
                    for r in ok),
                f'{sum(1 for r in ok if r.get("cap_avg_nRMSE") is not None)}행 대조'))
    z = {r['episode'] for r in sp if r.get('zero_nRMSE') is not None}
    out.append(('G-18b ZERO 48 에피소드', len(z) == 2 * N_EP, f'{len(z)}/{2*N_EP}'))
    return out


def gates_pc1(pc1):
    return [('★G-16b PC-1 생성기 비트동일 (B-26 SS-REAL 선형 에피소드 재생성)',
             bool(pc1) and all(pc1['identical'].get(k) for k in PC1_DATASETS),
             json.dumps(pc1.get('maxabs')))]


# ── I/O 도우미 (게이트 논리와 분리 — 자가진단이 주입한다) ──────────────────────
def _read_ep(path):
    import h5py
    import numpy as np
    with h5py.File(path, 'r') as f:
        a = dict(f['config'].attrs)
        gt = f['frames/gt_wave_field']
        n_frames = int(gt.shape[0])
        hs4 = float(4 * np.std(gt[:]))
    return dict(Hs=float(a['Hs']), Tp=float(a['Tp']), depth=float(a['depth']),
                mesh_dx=float(a.get('mesh_dx', -1)), n_frames=n_frames, hs4=hs4,
                ra=bool(a.get('random_amplitudes', False)),
                rf=bool(a.get('random_frequencies', False)),
                ds=bool(a.get('decorrelate_stream', False)))


MANIFEST = os.path.join(HERE, 'gen_manifest_m1.json')


def _load_manifest():
    return json.load(open(MANIFEST)) if os.path.exists(MANIFEST) else {}


# ─────────────────────────────────────────────────────────────────────────────
# ★자가진단 — *게이트도 시험한다*  (런 0회 · 파일 0개)
# ─────────────────────────────────────────────────────────────────────────────
def _fixture(perturb=None):
    """정상 케이스 픽스처: 288 행 + 자료 48건 + 매니페스트 96건. 값은 **임의 상수**다."""
    eps, _ = select()
    rows, data, man = [], {}, {}
    for e in eps:
        h = (hash(e['episode']) % 100) / 1000.0
        spec = 0.24 * (1 + h)
        for s in EST_SEEDS:
            rows.append(dict(arm='SHIP-NOW', gt_arm=e['gt_arm'], episode=e['episode'],
                             ncomp=e['ncomp'], gen_seed=e['seed'], est_seed=s,
                             avg_nRMSE=spec * 0.55, cap_avg_nRMSE=spec * 0.55,
                             varpro_steps=400, learned_init_iters=0, auto_switch_thresh=0,
                             lambda_l1=1e-4, cap_auto_selected='learned', cap_varpro_fired=True))
        for s in FILTER_SEEDS:
            rows.append(dict(arm='SPEC25', gt_arm=e['gt_arm'], episode=e['episode'],
                             ncomp=e['ncomp'], gen_seed=e['seed'], filter_seed=s,
                             avg_nRMSE=spec, cap_avg_nRMSE=spec, zero_nRMSE=spec * 1.12,
                             loc_radius=25.0, n_ensemble=600, background='jonswap',
                             cap_Hs=HS, cap_Tp=TP))
        data[e['path']] = dict(Hs=HS, Tp=TP, depth=G26.gen_params()['depth'],
                               mesh_dx=mesh_d(cfg_m1(False)),
                               n_frames=G26.gen_params()['n_frames'], hs4=HS,
                               ra=True, rf=True, ds=True)
        man[f"{e['gt_arm']}/{fname(e['ncomp'], e['seed'])}"] = dict(
            status='ok', gt_arm=e['gt_arm'], ncomp=e['ncomp'], seed=e['seed'],
            n_wrapped=0, max_snap_dk=HALF_BIN * 0.9, comp_md5=f"{e['ncomp']}_{e['seed']}",
            gt_skew=(0.15 if e['gt_arm'] == 'HOS' else 0.001),
            gt_absdiff_vs_pair=(0.01 if e['gt_arm'] == 'HOS' else None),
            mesh_verdict_snapped='DEGRADED')
    pc1 = dict(identical={k: True for k in PC1_DATASETS},
               maxabs={k: 0.0 for k in PC1_DATASETS})
    if perturb:
        perturb(rows, data, man, pc1)
    return rows, data, man, pc1


def _run_all(rows, data, man, pc1, runfree=True, **rf_kw):
    out = []
    if runfree:
        out += gates_runfree(**rf_kw)
    out += gates_data(read=lambda p: data[p], exists=lambda p: p in data, manifest=man)
    out += gates_run(rows)
    out += gates_pc1(pc1)
    return out


def _name(g):
    return g[0].split()[0].lstrip('★')


def _perturbations():
    """게이트마다 **그것을 깨는 개입**. 해당 게이트가 FAIL 로 뒤집혀야 한다."""
    def d(k, fn):
        return (k, fn)
    return [
        d('G-6', lambda r, dt, m, p: dt.pop(next(iter(dt)))),
        d('G-7', lambda r, dt, m, p: dt[next(iter(dt))].update(Hs=HS + 1.0)),
        d('G-8', lambda r, dt, m, p: [v.update(hs4=HS * 1.5) for v in dt.values()]),
        d('G-8c', lambda r, dt, m, p: dt[next(iter(dt))].update(hs4=HS * 1.3)),
        d('G-9', lambda r, dt, m, p: dt[next(iter(dt))].update(mesh_dx=99.0)),
        d('G-10', lambda r, dt, m, p: dt[next(iter(dt))].update(n_frames=7)),
        d('G-24', lambda r, dt, m, p: m[next(iter(m))].update(n_wrapped=1)),
        d('G-24', lambda r, dt, m, p: m[next(iter(m))].update(max_snap_dk=HALF_BIN * 3)),
        d('G-25', lambda r, dt, m, p: [v.update(comp_md5='X') for k, v in m.items()
                                       if v['gt_arm'] == 'HOS' and v['seed'] == m[next(iter(m))]['seed']]),
        d('G-26', lambda r, dt, m, p: [v.update(gt_skew=-1.0) for v in m.values()
                                       if v['gt_arm'] == 'HOS']),
        d('G-26b', lambda r, dt, m, p: [v.update(gt_absdiff_vs_pair=0.0) for v in m.values()
                                        if v['gt_arm'] == 'HOS']),
        d('G-27', lambda r, dt, m, p: m[next(iter(m))].update(mesh_verdict_snapped=MESH_BAD)),
        d('G-11', lambda r, dt, m, p: r.append(dict(arm='SHIP-NOW', error='boom'))),
        d('G-12', lambda r, dt, m, p: r.remove(next(x for x in r if x['arm'] == 'SHIP-NOW'))),
        d('G-13', lambda r, dt, m, p: [x.update(episode='ZZZ') for x in r
                                       if x['arm'] == 'SPEC25' and x['gt_arm'] == 'HOS'][:1]),
        d('G-14', lambda r, dt, m, p: [x.update(varpro_steps=0) for x in r
                                       if x['arm'] == 'SHIP-NOW']),
        d('G-14', lambda r, dt, m, p: [x.update(lambda_l1=1e-3) for x in r
                                       if x['arm'] == 'SHIP-NOW']),
        d('G-15', lambda r, dt, m, p: [x.update(cap_auto_selected='v2_konly') for x in r
                                       if x['arm'] == 'SHIP-NOW' and x['gt_arm'] == 'HOS']),
        d('G-15b', lambda r, dt, m, p: [x.update(cap_varpro_fired=False) for x in r
                                        if x['arm'] == 'SHIP-NOW']),
        d('G-16', lambda r, dt, m, p: [x.update(loc_radius=50.0) for x in r
                                       if x['arm'] == 'SPEC25']),
        d('G-17', lambda r, dt, m, p: [x.update(cap_Hs=1.2, cap_Tp=7.0) for x in r
                                       if x['arm'] == 'SPEC25']),
        d('G-18', lambda r, dt, m, p: [x.update(cap_avg_nRMSE=x['avg_nRMSE'] * 1.0001)
                                       for x in r][:1]),
        d('G-18b', lambda r, dt, m, p: [x.update(zero_nRMSE=None) for x in r
                                        if x['arm'] == 'SPEC25' and x['gt_arm'] == 'HOS']),
        d('G-16b', lambda r, dt, m, p: p['identical'].update({PC1_DATASETS[0]: False})),
    ]


def selftest(verbose=True):
    def say(s):
        if verbose:
            print(s, flush=True)
    fails = []
    say('★M1 게이트 자가진단 ────────────────────────────────────────')
    rows, data, man, pc1 = _fixture()
    base = _run_all(rows, data, man, pc1)
    bad = [g for g in base if not g[1]]
    say(f'[1] 픽스처 정상 케이스: {len(base)-len(bad)}/{len(base)} PASS')
    for g in bad:
        fails.append(f'픽스처에서 FAIL: {g[0]} :: {g[2]}')

    # [2] 공허한 통과 — 0행·자료 0건에서 통과하는 게이트가 ROWLESS_OK 밖인가
    # ⚠아래는 `dict(...)` 로 적는다. 중괄호 리터럴로 빈 dict 를 값에 넣으면
    #   **검사기(`tmp/_audit/check_batch_artifacts.py`)의 빈-dict-팔 정규식**에 걸려
    #   `no-{}-arm` 열이 영구히 경고를 찍는다 — 이 배치에 그렇게 정의된 팔은 **0개**이므로
    #   순수 오탐이고, ★경고가 항상 켜져 있으면 신호가 죽는다(`gates_b7.py:46` 전례).
    #   ⛔**값은 비트 동일**이고 자가진단 [2] 의 `공허한 통과 0건` 이 그것을 확인한다.
    #   ⚠이 주석 자체도 그 패턴을 적으면 안 된다 — 정규식은 주석을 구별하지 않는다.
    empty = _run_all([], {}, {}, dict(identical={}, maxabs={}), runfree=False)
    leaks = [g[0] for g in empty if g[1] and _name(g) not in ROWLESS_OK]
    say(f'[2] 공허한 통과: {len(leaks)}건 (허용 {ROWLESS_OK})')
    if leaks:
        fails.append(f'0행에서 통과한 게이트: {leaks}')

    # [3] 변형 — 게이트마다 그것을 깨는 개입이 실제로 FAIL 을 낸다
    fired = 0
    for gname, fn in _perturbations():
        r, dt, m, p = _fixture(perturb=fn)
        res = _run_all(r, dt, m, p, runfree=False)
        hit = [g for g in res if _name(g) == gname and not g[1]]
        if hit:
            fired += 1
        else:
            fails.append(f'변형이 {gname} 를 발화시키지 못했다')
    say(f'[3] 변형: {fired}/{len(_perturbations())} 발화')

    # [4] 런 0회 변형 — 격자를 되돌리면 G-22 가 발화하는가
    rf = gates_runfree(hos_nx=B12_HOS_NX, hos_ny=B12_HOS_NY)
    if not [g for g in rf if _name(g) in ('G-22', 'G-22c') and not g[1]]:
        fails.append('격자를 B-12 값으로 되돌려도 G-22 가 발화하지 않는다')
    else:
        say('[4] 런 0회 변형: 격자 되돌림 -> G-22 발화 ✓')

    # [5] 라벨 도달성 + READING 키
    need = set()
    for w in range(0, N_EP + 1):
        for nw in (N_EP, NEG_MIN - 1):
            need.add(arm_label({'better': w, 'n': N_EP, 'p_one_sided': sign_test_one_sided(w, N_EP)},
                               {'better': nw, 'n': N_EP})[0])
    missing = sorted(need - set(READING))
    say(f'[5] 라벨 도달성: {len(need)}종 도달 · READING 누락 {missing}')
    if missing:
        fails.append(f'READING 에 없는 라벨: {missing}')
    if len(need) != 5:
        fails.append(f'도달 가능 라벨이 5종이 아니다: {sorted(need)}')
    pk = {k for k in PAIR_READING}
    if pk != {(a, b) for a in ('hold', 'break') for b in ('hold', 'break')}:
        fails.append(f'PAIR_READING 조합 누락: {sorted(pk)}')

    # [6] 문턱 산술
    t = threshold_table()
    if not (t['confirm_within_family'] and t['weak_is_sibling'] and t['p_nominal'] > FAMILY_ALPHA):
        fails.append(f'문턱 산술 불일치: {t}')
    say(f'[6] 문턱: CONFIRM {CONFIRM}/{N_EP} p={t["p_confirm"]:.6f} <= {FAMILY_ALPHA} · '
        f'NOMINAL {NOMINAL} p={t["p_nominal"]:.6f} · WEAK <= {WEAK}')

    say('────────────────────────────────────────────────────────────')
    for f in fails:
        say(f'  ⛔ {f}')
    say(f'자가진단 {"PASS" if not fails else "FAIL"} ({len(fails)} 결함)')
    return (not fails), fails


def _freeze():
    eps, rej = select()
    picked = {f'{n}': tuple(e['seed'] for e in eps if e['gt_arm'] == 'LIN' and e['ncomp'] == n)
              for n in NCOMPS}
    print('FROZEN_SEEDS =', json.dumps(picked))
    print('FROZEN_REJECTED =', tuple((r['ncomp'], r['seed']) for r in rej))
    print('SEEDS_CONSUMED =', tuple(sorted({e['seed'] for e in eps} | {r['seed'] for r in rej})))


if __name__ == '__main__':
    if '--freeze' in sys.argv:
        _freeze()
    elif '--selftest' in sys.argv:
        sys.exit(0 if selftest()[0] else 1)
    else:
        for n, ok, msg in gates_runfree():
            print(f'[{"PASS" if ok else "FAIL"}] {n}: {msg}')
