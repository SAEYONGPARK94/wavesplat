#!/usr/bin/env python3
"""M-10 게이트·문턱·밴드·라벨 — ★**판정의 유일한 출처** (2026-08-30, 결과 0행에서 동결).

    python3 tmp/M10_GEOMETRY_2026-08-30/gates_m10.py       # 자가진단 (자료 0행 · 런 0회)

⛔`PREREG.md` 와 이 파일이 어긋나면 **이 파일이 권위**다.
⛔손복사 상수 0건: 코퍼스·시드·팔·셸 경계는 전부 동결 배치에서 **임포트**하고,
   문턱 다섯 수는 `sign_test_one_sided` 에서 **재도출**한다.

★검사기 규약(`M8b F-2` 처방 (b)): 무인자 로더 `load_rows()` 를 `glob.glob` **무별칭** +
`json.loads` 로 노출해 `check_gate_refs._call_loaders` 가 행 원천을 짚을 수 있게 한다.

═══════════════════════════════════════════════════════════════════════════════
★**AMENDMENT_1 (2026-08-30, ⛔결과 0행 · 판정 표면 불변) — `BLOCKER-D1` 반영.**

이 파일에서 **바뀐 것은 `G-10` 의 절 하나**다: `cuda_available is False` 요구를 뺐다.
사유 = `G-10`(CPU 고정)과 `G-4`(`B-12` 동결 재현)가 **동시에 성립하지 않는다**(양방향 실측,
`AMENDMENT_1.md §2`). `B-12` 동결 자료가 GPU 가 보이는 환경의 산물이고
`learned_init.propose_atoms` 는 `config['device']` 를 **받지 않고** 스스로 cuda 를 고른다.

⛔**빠지지 않은 것**(= 완화가 새지 않는다):
  · `device == 'cpu'` · `omp_threads == '1'` · 단일 `code_md5` — **그대로**.
  · `cuda_available` 은 **스탬프 튜플 안에 남아 있다** ⟹ `len(stamps) == 1` 요구가
    *"배치 중간에 GPU 가시성이 바뀌었다"* 를 **여전히 차단**한다. 뺀 것은 **값의 고정**뿐이다.
★**대신 하나를 더 본다**(`G-15`, 범위축소): `device` 스탬프는 `DEFAULT_CONFIG['device']` 의
  **의도표**이고 확산 사전이 실제로 쓴 장치가 아니다(저장소가 반복해 물린 부류).
  `G-15` 는 `learned_init._CACHE` **되읽기**를 요구한다 — 재도출이 아니라 callee 자신의 상태다.

⛔**문턱·밴드·라벨·해석 규칙·팔·코퍼스·시드는 한 글자도 바뀌지 않았다.**
   기계 증명 = `amend1_surface.py`(판정 표면 비트 대조 + 완화 누수 시험).
═══════════════════════════════════════════════════════════════════════════════
"""
import glob
import json
import os
import statistics as st
import sys
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
B11 = os.path.join(ROOT, 'tmp', 'B11_HOS_BASELINE_2026-08-23')
B12 = os.path.join(ROOT, 'tmp', 'B12_HOS_EXTEND_2026-08-23')
S16 = os.path.join(ROOT, 'tmp', 'S16_BASELINE_2026-08-14')
for _p in (HERE, B11, B12, S16, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── 코퍼스·시드·팔 — ⛔손복사 금지, 동결물에서 임포트 ─────────────────────────
from gates_b11 import HOS, EST_SEEDS, FILTER_SEEDS, SPEC          # noqa: E402,F401
from gates_b12 import ALL_SEEDS, N_EP                             # noqa: E402
import run_s16 as R16                                             # noqa: E402

SHELL_KEYS = list(R16.SHELL_KEYS)          # ['obs:0_5', …, 'obs:150_inf'] — 단일출처
EDGES = list(R16.EDGES)
CROWDED_MIN_NCOMP = 15                     # `run_b12.crowded_episodes` 의 조건절
ARM_ORDER = ('SHIP', 'SPEC25')
B12_FROZEN = os.path.join(B12, 'results_b12.jsonl')

# ── 밴드 (결과 전 확정 · PREREG §4.2) ──────────────────────────────────────────
BAND_REAL = ('obs:0_5',)
BAND_WIDE = ('obs:0_5', 'obs:5_10', 'obs:10_20')
BAND_FAR = ('obs:50_75', 'obs:75_100', 'obs:100_150', 'obs:150_inf')
BANDS = dict(BAND_REAL=BAND_REAL, BAND_WIDE=BAND_WIDE, BAND_FAR=BAND_FAR)
PRIMARY_BAND = 'BAND_REAL'

# ── 문턱 (PREREG §4.3) — ⛔전부 재도출, 리터럴 없음 ─────────────────────────────
ALPHA = 0.05
N_FAMILY = 2                               # PRIMARY-A · SECONDARY-B
FAMILY_ALPHA = ALPHA / N_FAMILY
REPRO_RTOL = 1e-6                          # G-4 (⚠비트 아님 — PREREG §6-5)
FRAC_RTOL = 1e-9                           # G-6
MIN_BAND_CELLS = 50                        # G-13 (범위축소)


def sign_test_one_sided(w, n):
    return sum(comb(n, k) for k in range(w, n + 1)) / 2 ** n


def thresholds(n=N_EP):
    """(NOMINAL, CONFIRM, NOT_EARN, q80, floor_p). 형제 규약 = `B-12`/`B-22`."""
    nominal = next(w for w in range(n + 1) if sign_test_one_sided(w, n) <= ALPHA)
    confirm = next(w for w in range(n + 1) if sign_test_one_sided(w, n) <= FAMILY_ALPHA)
    lo, hi = 0.5, 1.0
    for _ in range(80):                     # 80% 검정력을 주는 참 승률 (이분)
        mid = (lo + hi) / 2
        pw = sum(comb(n, k) * mid ** k * (1 - mid) ** (n - k) for k in range(confirm, n + 1))
        if pw >= 0.80:
            hi = mid
        else:
            lo = mid
    return dict(n=n, NOMINAL=nominal, CONFIRM=confirm, NOT_EARN=n - nominal,
                q80=hi, floor_p=sign_test_one_sided(n, n),
                alpha=ALPHA, family_alpha=FAMILY_ALPHA)


# ── 밴드 합성 (결과 전 고정: 면적 가중 RMS) ────────────────────────────────────
def band_rmse(shell, frac, band):
    """RMSE_band = sqrt( Σ f_k·RMSE_k² / Σ f_k ).  결측 셸은 제외하고 f 도 뺀다."""
    num = den = 0.0
    for k in band:
        r, f = shell.get(k), frac.get(k)
        if r is None or f is None:
            continue
        if r != r or f != f or f <= 0:      # NaN / 빈 셸
            continue
        num += f * r * r
        den += f
    if den <= 0:
        return None
    return (num / den) ** 0.5


def band_frac(frac, band):
    return sum(f for k in band
               for f in [frac.get(k)] if f is not None and f == f)


# ── 판정 ───────────────────────────────────────────────────────────────────────
def med_by_episode(rows, arm, key):
    d = {}
    for r in rows:
        if 'error' in r or r.get('arm') != arm or r.get(key) is None:
            continue
        d.setdefault(r['episode'], []).append(float(r[key]))
    return {e: st.median(v) for e, v in d.items()}


def paired_wins(a, b):
    """a < b 인 에피소드 수 (짝, 공통 키만).  반환 (w, n, rel_median)."""
    ks = sorted(set(a) & set(b))
    w = sum(1 for k in ks if a[k] < b[k])
    rel = [(a[k] - b[k]) / b[k] for k in ks if b[k]]
    return w, len(ks), (st.median(rel) if rel else None)


LABELS = ('NEARFIELD-HOLDS', 'NEARFIELD-HOLDS-NOMINAL', 'NEARFIELD-BREAKS',
          'NEARFIELD-UNDECIDED', 'NEARFIELD-VOID')


def label(w_primary, n, cmp_ok):
    """★결과 전 확정. `cmp_ok` = G-CMP(비교자 유효성)."""
    t = thresholds(n)
    if not cmp_ok:
        return 'NEARFIELD-VOID'
    if w_primary >= t['CONFIRM']:
        return 'NEARFIELD-HOLDS'
    if w_primary >= t['NOMINAL']:
        return 'NEARFIELD-HOLDS-NOMINAL'
    if w_primary <= t['NOT_EARN']:
        return 'NEARFIELD-BREAKS'
    return 'NEARFIELD-UNDECIDED'


def fold(lab):
    """★접기 공역.  ⛔`'break'` 는 `NEARFIELD-BREAKS` **정확히 하나**다.

    본뜬 곳 = `tmp/M3_SUPPORT_2026-08-28/gates_m3.py:750-757 st_()` (정확 일치).
    ⛔`gates_m1.py:314` 상속 금지 — 그것은 `BASELINE-NO-SKILL` 을 `'break'` 로 접어
    *"비교 불능"* 을 *"부호 반전"* 으로 둔갑시킨다
    (`tmp/_audit/DISCIPLINE_verdict_labels_2026-08-30.md` P1)."""
    if lab == 'NEARFIELD-HOLDS':
        return 'hold'
    if lab == 'NEARFIELD-BREAKS':
        return 'break'
    return 'undecided'


READING = {
    'NEARFIELD-HOLDS':
        '합성 이득은 실측 코퍼스가 점유하는 관측 기하(obs:0_5) 안에도 있다 ⟹ 소거표에서 '
        '관측 기하의 **피복** 성분이 배제된다. ★척추 문장 무수정(배제≠지목). '
        '⛔"관측 기하가 무관하다" 금지(등가검정 없음).',
    'NEARFIELD-HOLDS-NOMINAL':
        '명목만(가족단위 α 미달) ⟹ **EXPLORATORY** 병기 의무 + MDE 병기. '
        '⛔확증으로 올리지 마라. ⛔결과를 보고 n 을 늘리지 마라.',
    'NEARFIELD-BREAKS':
        '합성 이득은 실측이 점유하는 기하에 없다 — 헤드라인은 격자 면적의 72%인 외삽 대역에서 '
        '나온다. ⛔인과 지목 금지(재채점은 기제를 재지 않는다); 허용은 "이득의 위치"까지. '
        '★전역 승수를 인용하는 모든 지점에 동반 공시 의무가 생긴다(같은 것이 N곳이면 N곳 전부). '
        '⛔"SUPPORT-TRADEOFF" 라는 이름 금지(그 이름은 내부 처치 대조에 붙어 있다).',
    'NEARFIELD-UNDECIDED':
        '실측이 점유하는 밴드에서 우열이 n=48 로 갈리지 않았다 — **이 라벨도 결과다**'
        '(전역 승수가 그 밴드에서 재현되지 않는다는 사실 자체가 서술). '
        '⛔"차이 없다" 금지(등가검정 없음). MDE 병기 의무.',
    'NEARFIELD-VOID':
        '비교자가 그 밴드에서 무기술 ⟹ ⛔**우열 주장 전면 금지**. '
        '허용 = "그 밴드는 두 방법 모두에게 다른 문제다" · "헤드라인의 비교는 그 밴드 밖에서만 '
        '성립한다". ★우리 재현·우리 기하에 대한 판정이지 문헌 일반에 대한 것이 아니다.',
}


# ── 로더 (★검사기 규약: 무인자 · glob.glob 무별칭 · json.loads) ────────────────
def load_rows():
    out = []
    for path in sorted(glob.glob(os.path.join(HERE, 'results_m10_*.jsonl'))):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
    return out


def load_frozen_b12():
    out = {}
    if not os.path.exists(B12_FROZEN):
        return out
    with open(B12_FROZEN) as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if 'error' in r or r.get('arm') not in ('SHIP-NOW', 'SPEC25'):
                continue
            arm = 'SHIP' if r['arm'] == 'SHIP-NOW' else 'SPEC25'
            s = r.get('est_seed', r.get('filter_seed'))
            out[(arm, r['episode'], s)] = float(r['avg_nRMSE'])
    return out


def episodes():
    """혼잡 48.  조건절은 `run_b12.crowded_episodes` 의 것을 그대로."""
    import re
    out = []
    for p in sorted(glob.glob(os.path.join(HOS, '*.h5'))):
        m = re.match(r'episode_(BF\d)_(\d+)comp_seed(\d+)_(.+)\.h5', os.path.basename(p))
        if m and int(m.group(2)) >= CROWDED_MIN_NCOMP and int(m.group(3)) in ALL_SEEDS:
            out.append(os.path.basename(p)[len('episode_'):-3])
    return out


# ── 게이트 ─────────────────────────────────────────────────────────────────────
def _rel(a, b):
    return abs(a - b) / abs(b) if b else (0.0 if a == b else float('inf'))


def gates(rows, frozen=None):
    """(name, ok, detail, blocking) 목록.  ★러너와 분석기가 **이 함수 하나**를 부른다."""
    frozen = load_frozen_b12() if frozen is None else frozen
    good = [r for r in rows if 'error' not in r]
    err = [r for r in rows if 'error' in r]
    out = []
    cov = {}                                    # G-8: 게이트별 열람 행 수

    def g(name, ok, det, blocking=True, n_seen=1):
        cov[name] = n_seen
        out.append((name, bool(ok), det, blocking))

    per_arm = {a: [r for r in good if r.get('arm') == a] for a in ARM_ORDER}
    eps = sorted({r['episode'] for r in good})
    want = N_EP * len(EST_SEEDS)
    g('G-1 표본 SHIP/SPEC25 각 %d행 · 에피소드 %d' % (want, N_EP),
      all(len(per_arm[a]) == want for a in ARM_ORDER) and len(eps) == N_EP,
      'SHIP %d · SPEC25 %d · ep %d' % (len(per_arm['SHIP']), len(per_arm['SPEC25']), len(eps)),
      n_seen=len(good))
    g('G-2 에러 0', not err, '%d' % len(err), n_seen=len(rows))

    dep = [r for r in good if r.get('arm') == 'SHIP']
    ok3 = all(r.get('obj_init_method') == 'auto'
              and r.get('obj_tracking_mode') == 'w_phi'
              and r.get('cap_varpro_steps') is not None
              and r.get('arm_cfg_nonempty') is True for r in dep)
    g('G-3 팔 되읽기(객체 속성) + ⛔빈 dict 팔 0', ok3 and bool(dep),
      '%d행' % len(dep), n_seen=len(dep))

    pairs = [(r, frozen.get((r['arm'], r['episode'], r.get('seed'))))
             for r in good]
    pairs = [(r, f) for r, f in pairs if f is not None]
    bad4 = [r['episode'] for r, f in pairs if _rel(float(r['avg_nRMSE']), f) > REPRO_RTOL]
    g('G-4 동결 재현 rel Δ ≤ %g (%d짝)' % (REPRO_RTOL, len(pairs)),
      pairs and not bad4, '위반 %d' % len(bad4), n_seen=len(pairs))

    win = {}
    for r in good:
        win.setdefault(r['episode'], set()).add((r.get('n_scored'), r.get('t_first'),
                                                 r.get('t_last')))
    g('G-5 같은 채점 창(에피소드 안 팔·시드 전건 동일)',
      bool(win) and all(len(v) == 1 for v in win.values()),
      '불일치 %d' % sum(1 for v in win.values() if len(v) != 1), n_seen=len(win))

    bad6 = 0
    seen6 = 0
    byep = {}
    for r in good:
        byep.setdefault(r['episode'], {}).setdefault(r['arm'], []).append(r)
    for e, d in byep.items():
        if len(d) < 2:
            continue
        a = d['SHIP'][0].get('shell_frac') or {}
        b = d['SPEC25'][0].get('shell_frac') or {}
        for k in SHELL_KEYS:
            if k in a and k in b:
                seen6 += 1
                if _rel(a[k], b[k]) > FRAC_RTOL:
                    bad6 += 1
    g('G-6 같은 셸 마스크(팔 간 shell_frac 일치, %d 대조)' % seen6,
      seen6 > 0 and bad6 == 0, '위반 %d' % bad6, n_seen=seen6)

    bad7 = seen7 = 0
    for e, d in byep.items():
        for arm, rs in d.items():
            for r in rs:
                zb = r.get('zero_shell_batch') or {}
                zr = r.get('zero_shell_ref')
                if not zb or not zr:
                    continue
                for k in SHELL_KEYS:
                    if k in zb and k in zr:
                        seen7 += 1
                        if zb[k] != zr[k]:          # ★비트 동일 요구
                            bad7 += 1
    g('G-7 교차구현 zero_shell 비트 일치 (%d 대조)' % seen7,
      seen7 > 0 and bad7 == 0, '위반 %d' % bad7, n_seen=seen7)

    zbad = 0
    for e, d in byep.items():
        zs = {round(float(r['zero_nRMSE']), 12) for rs in d.values() for r in rs
              if r.get('zero_nRMSE') is not None}
        if len(zs) > 1:
            zbad += 1
    g('G-9 zero_nRMSE 팔 간 동일', bool(byep) and zbad == 0, '불일치 %d' % zbad,
      n_seen=len(byep))

    # ★AMENDMENT_1(BLOCKER-D1): `s[1] is False`(= `cuda_available` 고정) 절을 **뺐다**.
    #   ⛔`cuda_available` 은 튜플에 **남는다** — `len(stamps) == 1` 이 배치 중간의
    #   가시성 변화를 계속 차단한다. 뺀 것은 *값의 고정*이지 *관측*이 아니다.
    stamps = {(r.get('device'), r.get('cuda_available'), r.get('omp_threads'),
               r.get('code_md5')) for r in good}
    g('G-10 장치·스레드·code_md5 단일 (⚠cuda_available 은 단일성만 · AMENDMENT_1)',
      len(stamps) == 1 and all(s[0] == 'cpu' and str(s[2]) == '1'
                               for s in stamps),
      '%r' % (sorted(map(repr, stamps))[:1],), n_seen=len(good))

    src = open(os.path.abspath(__file__)).read()
    g('G-11 셸 경계 단일출처(리터럴 손복사 0)',
      'R16.SHELL_KEYS' in src and '0_5' in repr(BAND_REAL) and 'EDGES = list(' in src,
      'imported', n_seen=1)

    g('G-8 커버리지 0 게이트 금지', all(v > 0 for v in cov.values()),
      '0 커버리지 %d' % sum(1 for v in cov.values() if v == 0), n_seen=len(cov))

    # ── 범위축소 (blocking=False) ────────────────────────────────────────────
    nbit = sum(1 for r, f in pairs if float(r['avg_nRMSE']) == f)
    g('G-12 동결값 비트 일치 수 (0이어도 유효)', True, '%d/%d' % (nbit, len(pairs)),
      blocking=False, n_seen=len(pairs))
    small = [r['episode'] for r in good
             if (r.get('band_cells') or {}).get(PRIMARY_BAND, 0) < MIN_BAND_CELLS]
    g('G-13 %s 격자점 ≥ %d (미달은 보고만, ⛔밴드 불변)' % (PRIMARY_BAND, MIN_BAND_CELLS),
      True, '미달 %d' % len(set(small)), blocking=False, n_seen=len(good))
    mv = sorted({r.get('mesh_verdict') for r in good})
    g('G-14 mesh_verdict 분할 보고 (⛔"HOS M=3" 단일 서술 금지)', True, '%r' % (mv,),
      blocking=False, n_seen=len(good))
    # ★AMENDMENT_1 신설 (범위축소) — **의도표가 아니라 되읽기**.
    #   `device` 스탬프는 `DEFAULT_CONFIG['device']`(= 'cpu' 리터럴)의 복사이고, 확산 사전이
    #   실제로 올라간 장치가 아니다. `learned_init._CACHE` 의 키 `(ckpt, str(device))` 를
    #   러너가 되읽어 행에 싣는다(`run_m10._effective_prior_device`).
    #   ⛔차단이 아니다 — `G-10` 의 단일 스탬프가 이미 가시성 변화를 막는다. 이것은
    #   *"추론(스탬프가 cuda 를 함의한다)"* 을 *"관측(사전 자신이 그렇게 말한다)"* 으로 바꾼다.
    pdev = sorted({str(r.get('prior_device')) for r in dep})
    g('G-15 학습 사전이 **실제로** 쓴 장치 단일 (되읽기, ⛔선언 device 아님)',
      bool(dep) and len(pdev) == 1 and pdev[0] not in ('None', ''),
      '%r' % (pdev,), blocking=False, n_seen=len(dep))
    return out


def cmp_ok_from(rows):
    """G-CMP — 비교자 유효성: SPEC25 < ZERO on BAND_REAL 이 CONFIRM 이상."""
    spec = med_by_episode(rows, 'SPEC25', 'band_' + PRIMARY_BAND)
    zero = med_by_episode(rows, 'SPEC25', 'zero_' + PRIMARY_BAND)
    w, n, rel = paired_wins(spec, zero)
    return (n > 0 and w >= thresholds(n)['CONFIRM']), dict(w=w, n=n, rel_median=rel)


# ── 자가진단 (자료 0행 · 런 0회) ───────────────────────────────────────────────
def _selftest():
    ok = []

    def chk(name, cond, det=''):
        ok.append((name, bool(cond), det))

    t = thresholds(48)
    chk('① 문턱 재도출 (n=48)',
        t['NOMINAL'] == 31 and t['CONFIRM'] == 32 and t['NOT_EARN'] == 17
        and abs(t['q80'] - 0.7109) < 1e-3,
        'NOM %d CONF %d NOT %d q80 %.4f' % (t['NOMINAL'], t['CONFIRM'],
                                            t['NOT_EARN'], t['q80']))
    chk('①b 문턱이 α 를 실제로 만족',
        sign_test_one_sided(t['CONFIRM'], 48) <= FAMILY_ALPHA
        and sign_test_one_sided(t['CONFIRM'] - 1, 48) > FAMILY_ALPHA)

    # ② 라벨-접기 변형 (DISCIPLINE P1)
    folds = {L: fold(L) for L in LABELS}
    chk('② break 로 접히는 라벨은 정확히 하나',
        sum(1 for v in folds.values() if v == 'break') == 1
        and folds['NEARFIELD-BREAKS'] == 'break', repr(folds))
    chk('②b VOID·UNDECIDED·NOMINAL 은 undecided',
        all(folds[L] == 'undecided' for L in
            ('NEARFIELD-VOID', 'NEARFIELD-UNDECIDED', 'NEARFIELD-HOLDS-NOMINAL')))
    chk('②c 모든 라벨이 READING 에 있다', set(LABELS) == set(READING))

    # ③ 무기술 팔 픽스처 — 비교자가 죽으면 어떤 승수에서도 VOID 이고 귀속 문구가 없다
    voids = {label(w, 48, cmp_ok=False) for w in (0, 17, 31, 32, 48)}
    chk('③ cmp 실패 ⟹ 전 승수에서 VOID', voids == {'NEARFIELD-VOID'}, repr(voids))
    chk('③b VOID 읽기에 우열 문구 없음',
        '우열 주장 전면 금지' in READING['NEARFIELD-VOID']
        and '이긴다' not in READING['NEARFIELD-VOID'])

    # ③c 라벨 경계 (사후 이동 방지)
    chk('③c 라벨 경계', label(32, 48, True) == 'NEARFIELD-HOLDS'
        and label(31, 48, True) == 'NEARFIELD-HOLDS-NOMINAL'
        and label(30, 48, True) == 'NEARFIELD-UNDECIDED'
        and label(18, 48, True) == 'NEARFIELD-UNDECIDED'
        and label(17, 48, True) == 'NEARFIELD-BREAKS')

    # ④ 밴드 합성 — 면적 가중 RMS 의 두 성질
    sh = {'obs:0_5': 2.0, 'obs:5_10': 4.0}
    fr = {'obs:0_5': 0.25, 'obs:5_10': 0.75}
    chk('④ band_rmse 면적가중 RMS',
        abs(band_rmse(sh, fr, ('obs:0_5', 'obs:5_10')) - (0.25 * 4 + 0.75 * 16) ** 0.5) < 1e-12)
    chk('④b 단일 셸이면 그 값 그대로',
        abs(band_rmse(sh, fr, ('obs:0_5',)) - 2.0) < 1e-12)
    chk('④c 결측·0면적 셸은 제외', band_rmse({'obs:0_5': None}, {'obs:0_5': 0.0},
                                       ('obs:0_5',)) is None)

    # ⑤ 삭제 민감도 + 항등 게이트 0 — 깨끗한 픽스처에서
    fx = _fixture()
    gs = gates(fx['rows'], fx['frozen'])
    chk('⑤ 깨끗한 픽스처: 차단 게이트 전건 PASS',
        all(o for n, o, d, b in gs if b), [n for n, o, d, b in gs if b and not o])
    chk('⑤b 커버리지 0 게이트 0개',
        all(o for n, o, d, b in gs if n.startswith('G-8')))
    for drop in ('SHIP', 'SPEC25'):
        g2 = gates([r for r in fx['rows'] if r['arm'] != drop], fx['frozen'])
        chk('⑤c %s 삭제 ⟹ 차단 FAIL 발생' % drop,
            any((not o) for n, o, d, b in g2 if b))
    g3 = gates([dict(r, shell_frac={k: v * 2 for k, v in r['shell_frac'].items()})
                if r['arm'] == 'SPEC25' else r for r in fx['rows']], fx['frozen'])
    chk('⑤d shell_frac 어긋나면 G-6 FAIL',
        any((not o) for n, o, d, b in g3 if n.startswith('G-6')))
    g4 = gates([dict(r, zero_shell_ref={k: v + 1e-9 for k, v in r['zero_shell_ref'].items()})
                for r in fx['rows']], fx['frozen'])
    chk('⑤e zero_shell 어긋나면 G-7 FAIL',
        any((not o) for n, o, d, b in g4 if n.startswith('G-7')))
    g5 = gates([dict(r, avg_nRMSE=r['avg_nRMSE'] * 1.01) for r in fx['rows']], fx['frozen'])
    chk('⑤f 값이 동결에서 벗어나면 G-4 FAIL',
        any((not o) for n, o, d, b in g5 if n.startswith('G-4')))

    # ★⑤g AMENDMENT_1 — `G-10` 완화가 **새지 않는가**(BLOCKER-D1 의 회귀 시험)
    def _g10(**over):
        rs = [dict(r, **over) for r in fx['rows']]
        return [o for n, o, d, b in gates(rs, fx['frozen']) if n.startswith('G-10')][0]
    chk('⑤g G-10: cuda_available=True 가 이제 통과한다 (AMENDMENT_1 의 의도된 유일 변화)',
        _g10(cuda_available=True))
    chk('⑤h G-10: device="cuda" 는 ⛔여전히 FAIL', not _g10(device='cuda'))
    chk('⑤i G-10: omp_threads="8" 은 ⛔여전히 FAIL', not _g10(omp_threads='8'))
    chk('⑤j G-10: code_md5 분열은 ⛔여전히 FAIL',
        not [o for n, o, d, b in gates(
            [dict(r, code_md5=('a' if i % 2 else 'b')) for i, r in enumerate(fx['rows'])],
            fx['frozen']) if n.startswith('G-10')][0])
    chk('⑤k ★G-10: cuda_available 이 **행마다 다르면** ⛔FAIL '
        '(완화는 값의 고정만 뺐고 단일성은 남았다)',
        not [o for n, o, d, b in gates(
            [dict(r, cuda_available=(i % 2 == 0)) for i, r in enumerate(fx['rows'])],
            fx['frozen']) if n.startswith('G-10')][0])
    g15 = lambda rs: [o for n, o, d, b in gates(rs, fx['frozen']) if n.startswith('G-15')][0]
    chk('⑤l G-15: 되읽기 스탬프가 있으면 PASS', g15(fx['rows']))
    chk('⑤m G-15: 되읽기 스탬프가 없으면 FAIL (⛔부재를 통과로 찍지 않는다)',
        not g15([{k: v for k, v in r.items() if k != 'prior_device'} for r in fx['rows']]))
    chk('⑤n G-15: 사전 장치가 행마다 갈리면 FAIL',
        not g15([dict(r, prior_device=('cuda' if i % 2 else 'cpu'))
                 if r['arm'] == 'SHIP' else r for i, r in enumerate(fx['rows'])]))
    chk('⑤o G-15 는 **범위축소**다 (판정을 막지 않는다)',
        all(b is False for n, o, d, b in gates(fx['rows'], fx['frozen'])
            if n.startswith('G-15')))

    # ⑥ 공허 참조 방어 — 외부 원천을 비우면 해당 게이트가 FAIL(항진 금지)
    g6 = gates(fx['rows'], {})
    chk('⑥ 동결 원천이 비면 G-4 FAIL(0짝을 OK 로 찍지 않는다)',
        any((not o) for n, o, d, b in g6 if n.startswith('G-4')))
    g7 = gates([], {})
    chk('⑥b 행 0개면 차단 FAIL', any((not o) for n, o, d, b in g7 if b))

    # ⑦ 검사기 규약 — 무인자 로더 노출
    src = open(os.path.abspath(__file__)).read()
    chk('⑦ load_rows 가 glob.glob 무별칭 + json.loads 를 소스에 노출',
        'glob.glob(' in src and 'json.loads(' in src and 'def load_rows():' in src)

    # ⑧ 외부 임포트가 살아 있다 (손복사 0 의 증거)
    chk('⑧ SHELL_KEYS 9개 · obs: 접두어 · run_s16 유래',
        len(SHELL_KEYS) == 9 and all(k.startswith('obs:') for k in SHELL_KEYS)
        and SHELL_KEYS == list(R16.SHELL_KEYS))
    chk('⑧b 밴드가 SHELL_KEYS 부분집합 · PRIMARY 는 최근접 셸',
        all(set(b) <= set(SHELL_KEYS) for b in BANDS.values())
        and BAND_REAL == (SHELL_KEYS[0],))
    chk('⑧c 시드·표본이 동결물 유래',
        EST_SEEDS == (42, 43, 44) and FILTER_SEEDS == (42, 43, 44) and N_EP == 48)

    print('── gates_m10 자가진단 (자료 0행 · 런 0회) ──')
    nf = 0
    for n, o, d in ok:
        print(('  %s  %s' % ('PASS' if o else '⛔FAIL', n))
              + (('   %s' % (d,)) if (d and not o) else ''))
        nf += (not o)
    t48 = thresholds(48)
    print('  n=%d  NOMINAL %d · CONFIRM %d · NOT_EARN %d · q80 %.4f · 바닥 p %.3e'
          % (t48['n'], t48['NOMINAL'], t48['CONFIRM'], t48['NOT_EARN'],
             t48['q80'], t48['floor_p']))
    print('  %d/%d PASS' % (len(ok) - nf, len(ok)))
    return nf


def _fixture():
    """깨끗한 픽스처 — 자료 0행, 게이트 논리만 시험한다."""
    frac = {k: 1.0 / len(SHELL_KEYS) for k in SHELL_KEYS}
    zero = {k: 0.5 + 0.01 * i for i, k in enumerate(SHELL_KEYS)}
    rows, frozen = [], {}
    for e in range(N_EP):
        ep = 'BF4_15comp_seed%d_uniform_ra_rf_ds' % (1581 + e)
        for arm in ARM_ORDER:
            for s in EST_SEEDS:
                v = 0.10 + 0.001 * e + (0.05 if arm == 'SPEC25' else 0.0)
                frozen[(arm, ep, s)] = v
                rows.append(dict(
                    arm=arm, episode=ep, seed=s, avg_nRMSE=v, zero_nRMSE=0.25,
                    shell_rmse={k: v for k in SHELL_KEYS}, shell_frac=dict(frac),
                    zero_shell_batch=dict(zero), zero_shell_ref=dict(zero),
                    band_BAND_REAL=v, band_BAND_WIDE=v, band_BAND_FAR=v,
                    zero_BAND_REAL=zero[SHELL_KEYS[0]],
                    band_cells={'BAND_REAL': 135, 'BAND_WIDE': 514, 'BAND_FAR': 3462},
                    n_scored=160, t_first=14.1, t_last=30.0,
                    obj_init_method='auto', obj_tracking_mode='w_phi',
                    cap_varpro_steps=400, arm_cfg_nonempty=True,
                    mesh_verdict='DEGRADED', device='cpu', cuda_available=False,
                    omp_threads='1', code_md5='fixture'))
                # ★AMENDMENT_1: `prior_device` 는 **SHIP 행에만** 있다 —
                #   `SPEC25`(LETKF)는 확산 사전을 부르지 않는다. 러너와 같은 모양이다.
                if arm == 'SHIP':
                    rows[-1]['prior_device'] = 'cuda'
    return dict(rows=rows, frozen=frozen)


if __name__ == '__main__':
    sys.exit(1 if _selftest() else 0)
