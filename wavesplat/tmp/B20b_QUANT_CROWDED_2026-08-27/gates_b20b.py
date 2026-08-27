"""B-20b 게이트·문턱·라벨 — ★**단일출처**. 분석기(`analyze_b20b.py`)는 **호출만** 한다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
이 배치는 **신규 런이 0개**다. `B-21`(2026-08-27) 이 남긴 **동결 자료**에서,
그 배치가 **계산하지 않은 짝 하나**를 사전등록해 계산한다.

    PRIMARY (주장 하나) :  혼잡 층에서  `FITSWAP` < `KGRID1`  (n = 48, 짝 부호검정)
                          = `B-20` PRIMARY-B(*"적합 절차를 고정한 채 파수 양자화가
                            해치는가"*)의 **혼잡판**.

⛔**새 통계량을 만들지 않는다.** 집계(`median-of-3`)·표본 선정·부호검정·상대차는 전부
`gates_b21` 의 함수를 **import 해서** 쓴다. 이 저장소가 반복해서 물린 결함 부류
(*"같은 기능이 두 파일에 구현돼 한쪽만 계측"* — λ 3중 기재 · W4 개입 기대값 4중 기재)를
만들지 않기 위해서다. **이 파일이 새로 정의하는 것은 (a) 가족 α 와 문턱, (b) 라벨,
(c) 이 재분석에만 필요한 게이트 H-* 뿐이다.**

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
★**blind 의 정확한 뜻** (증거는 H-2·H-3·H-3b 가 기계로 검사한다)

  · `analyze_b21.py` 는 대조를 **네 개만** 만든다(`:68-71`): `KGRID1`vs`SPEC25` ·
    `KGRID2`vs`SPEC25` · `FITSWAP`vs`SHIP` · `SHIP`vs`KGRID1`.
    **`KGRID1` 과 `FITSWAP` 을 함께 받는 호출은 저장소의 두 판정 파일 어디에도 없다.**
  · `verdict_b21.json` 의 키 집합에도 그 통계량이 없다.
  · `B-20` 은 이 대조의 **희소판**을 쟀으나 그 배치의 행은 **전건 희소**다(`ncomp ∈ {2,5}`).

⛔**그러나 "예측 불가" 라는 뜻은 아니다.** §PREREG 7 에 무엇을 보고 썼는지 전부 적었다.
★**다만 판정은 넘겨받은 수로 결정되지 않는다** — 공개된 두 대조
(`SHIP<KGRID1` 46/48 · `FITSWAP<SHIP` 31/48)로부터 **연역 가능한 하한은
|A ∩ B| ≥ 46 + 31 − 48 = 29** 이고, 이 배치의 문턱 **31 보다 낮다**.
⟹ 가능한 구간 [29, 48] 안에 `UNDECIDED`(29–30) · `NOMINAL`(31) · `CONFIRM`(32–48) 가
   **전부 들어 있다.**

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
★**문턱의 도출** (⛔손복사 금지 — `gates_b21.k_min` 이 `math.comb` 로 낸다)

  주장은 **하나**다. 그러므로 다중비교 보정할 것이 **없다**.
  ⟹ ⛔*"가족단위로 보정했다"* 고 쓰지 않는다. 대신 **형제 배치의 더 엄격한 문턱을
     상속**한다: `B-20` 의 가족은 둘(A·B)이어서 α = 0.05/2 = 0.025 였고, 이 재분석이
     그중 **B 만** 옮겨 오면서 문턱을 **느슨하게 하는 것**은 결과 전이라도 이득 취득이다.
  ⟹ `ALPHA = 0.05`(NOMINAL) · `FAMILY_ALPHA = 0.025`(CONFIRM).
     n=48: **NOMINAL 31**(p=0.029732) · **CONFIRM 32**(p=0.014652) · NOT_EARN 17.
     ★같은 값을 `B-22` 가 같은 n 에서 **독립적으로** 못 박았다(`verdict_b22.json`
     `thresholds` = {31, 32, 17, family_alpha 0.025}) ⟹ **H-11 이 그 동결물과 대조**한다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠**이 재분석이 세울 수 없는 것** (PREREG §10 과 같은 문장, 코드에도 남긴다)

  `B-21` 의 행에는 `B-20` 의 **개입 충실도 계수기가 없다**:
      `k_dpsi_max` · `k_bit_identical_frac` · `w_abs_change_max` ·
      `phi_abs_change_max` · `snap_calls` · `k_dk_max`  ← **전부 부재**
  `B-21` 이 가진 것은 `k_snap_fired` · `k_snap_dk` · `k_snap_shift_median` ·
  `k_snap_cells` 넷이고, 개입 크기 게이트는 `G-4b`(|Δk| 문턱) **하나**다.
  ⟹ **`B-20` 의 `G-14`(재적합 실집행) · `G-15`(Δψ ≤ float32 정밀도)에 해당하는 증거를
     이 재분석은 만들 수 없다.** `H-12` 가 그 부재를 **기계로 확인해 기록**한다
     (부재가 사실이면 PASS — 이것은 주장 게이트가 아니라 **한계 기록 게이트**다).
  ⟹ 계수기가 판정에 필요해지는 조건은 `followup_triggers()` 가 결과 전에 정해 두었다.

⚠**`FITSWAP` 과 `NOOP-REFIT` 은 다른 팔이다** — `k_snap_dk` 가 각각 **1e-12** 와
  **1e-10** 이다. 같은 기계(`_snap_k_to_grid`)를 통과하되 격자가 100배 다르다.
  ⟹ `B-20`(희소, `NOOP-REFIT`)과 이 배치(혼잡, `FITSWAP`)를 **합산하지 않는다**.
"""
import ast
import hashlib
import json
import os
import statistics as _st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))          # …/wavesplat/wavesplat
B21 = os.path.join(ROOT, 'tmp', 'B21_MEDIATION_2026-08-27')
B20 = os.path.join(ROOT, 'tmp', 'B20_REFIT_CONTROL_2026-08-27')
B22 = os.path.join(ROOT, 'tmp', 'B22_PRIOR_ABLATION_2026-08-27')
REPO = os.path.abspath(os.path.join(ROOT, '..'))                # /home/vexit/wavesplat
for _p in (HERE, ROOT, B21, B20):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gates_b21 as G21                                          # noqa: E402
from gates_b20 import pp_band, PP_BAND_PIN                       # noqa: E402

# ── 동결 자료의 위치 (⛔이 배치는 여기에 **쓰지 않는다**) ─────────────────────
B21_RESULTS = os.path.join(B21, 'results_b21_*.jsonl')
B21_VERDICT = os.path.join(B21, 'verdict_b21.json')
B21_ANALYZE = os.path.join(B21, 'analyze_b21.py')
B21_GATES = os.path.join(B21, 'gates_b21.py')
B20_VERDICT = os.path.join(B20, 'verdict_b20.json')
B20_RESULTS = os.path.join(B20, 'results_b20_*.jsonl')
B22_VERDICT = os.path.join(B22, 'verdict_b22.json')
FROZEN = os.path.join(HERE, 'FROZEN_md5.txt')

# ── 이 배치의 대조 ────────────────────────────────────────────────────────────
STRATUM = 'crowded'
ARM_BASE = 'FITSWAP'      # 적합 절차는 그대로, 파수는 옮기지 않는다 (dk = 1e-12)
ARM_TREAT = 'KGRID1'      # 적합 절차 + 파수 양자화                (dk = DK1 = 0.02618)
ARM_SHIP = 'SHIP'         # 배포 (스냅 미발화)
PAIR_ARMS = (ARM_BASE, ARM_TREAT)
N_EP_PIN = 48
N_SEEDS_PIN = 3
N_ROWS_PIN = N_EP_PIN * len(PAIR_ARMS) * N_SEEDS_PIN             # 288

# ── 통계 (⛔손복사 0 — 부호검정 자체는 `gates_b21` 의 것을 쓴다) ──────────────
ALPHA = 0.05
FAMILY_ALPHA = 0.05 / 2          # ★형제(`B-20`)의 가족 α 를 **상속**한다 (독스트링 참조)
N_CLAIMS = 1                     # ⚠주장은 하나. 보정이 아니라 상속이다.
NOMINAL_PIN, CONFIRM_PIN, NOT_EARN_PIN = 31, 32, 17
MIN_N_USED = G21.MIN_N_USED[STRATUM]                             # 42
DEDUCIBLE_LOWER_BOUND = 29       # 46 + 31 − 48 (PREREG §7.2 — 문턱 31 보다 **낮다**)

# ── 크기 결정 (⛔검정 아님 — `B-20` §7.3 규약 상속) ───────────────────────────
MAG_MAIN_PIN = -38.6069561633153     # `verdict_b21` crowded `ablation_ref.rel_median_pct`
#   ★H-6 이 이 값을 **자료에서 재도출해 비트 동일**로 대조한다(핀은 대조용이지 근거가 아니다).
#   PP_BAND 는 `gates_b20.pp_band()` 가 `verdict_b12` 에서 재도출한다 = 2.3596978918107183.
#   ⟹ 결과 전 산술: `MAG_CLEAN` 이 [MAG_MAIN − PP_BAND, MAG_MAIN + PP_BAND]
#      = **[−40.96665, −36.24726]** 밖으로 나가야 인용값을 갈아 끼운다.

# ── 개입 분리 (팔별로 **다른 방향**을 요구한다 — W7 *"설정을 넣었다 ≠ 먹었다"*) ──
KGRID_SHIFT_MIN = G21.KGRID_SHIFT_MIN        # 1e-4  (KGRID1 은 파수를 실제로 옮겨야)
FITSWAP_SHIFT_MAX = G21.FITSWAP_SHIFT_MAX    # 1e-9  (FITSWAP 은 옮기면 안 된다)
SHIFT_RATIO_MIN = 1.0e6                      # ★두 팔의 중앙 이동량 비 (B-21 G-4b 에 없는 검사)

# ── 되읽기 핀 ─────────────────────────────────────────────────────────────────
ROUTE_EXPECTED = G21.ROUTE_EXPECTED           # 'learned'
EST_SEEDS = tuple(G21.EST_SEEDS)              # (42, 43, 44)
VARPRO_STEPS_PIN = 400
ENSEMBLE_K_PIN = 1
SPARSE_MAX_NCOMP = G21.SPARSE_MAX_NCOMP       # 15

# ── ★계수기 부재 핀 (한계의 기계적 기록) ─────────────────────────────────────
ABSENT_COUNTERS = ('k_dpsi_max', 'k_bit_identical_frac', 'w_abs_change_max',
                   'phi_abs_change_max', 'snap_calls', 'k_dk_max')
PRESENT_SNAP_KEYS = ('k_snap_fired', 'k_snap_dk', 'k_snap_shift_median', 'k_snap_cells')

# ── ★blind 증거 핀 ───────────────────────────────────────────────────────────
CMP_FUNCS = ('cmp_pair', 'paired_diffs', 'cmp_zero')
ANALYZE_PAIRS_PIN = [('KGRID1', 'SPEC25'), ('KGRID2', 'SPEC25'),
                     ('FITSWAP', 'SHIP'), ('SHIP', 'KGRID1')]
GATES_SITES_PIN = 4          # gates_b21.py 안의 비교 호출 지점 수 (숨은 호출이 생기면 어긋난다)
BLIND_PAIR = {'KGRID1', 'FITSWAP'}

B21_TOP_KEYS_PIN = {'arms', 'batch', 'family_alpha', 'gates', 'gates_all_pass', 'label',
                    'n_claims', 'n_gate_pass', 'n_gates', 'n_ok', 'n_rows', 'strata',
                    'tost_delta_conf_pct', 'tost_delta_pct'}
B21_STRATUM_KEYS_PIN = {'ablation_ref', 'attribution_share', 'confound', 'confound_label',
                        'confound_note', 'confound_tost', 'dose', 'dose_label', 'dose_note',
                        'n_used', 'negative', 'primary', 'primary_label', 'primary_note',
                        'primary_tost', 'thresholds'}
B21_GATES_PIN = 24           # `B-21` 이 통과시킨 게이트 수 (전건 통과였다)

# 재현해야 하는 `B-21` 대조 (층, a, b, verdict 키)
POSCTRL_CONTRASTS = (
    ('sparse', 'KGRID1', 'SPEC25', 'primary'),
    ('sparse', 'KGRID2', 'SPEC25', 'dose'),
    ('sparse', 'FITSWAP', 'SHIP', 'confound'),
    ('sparse', 'SHIP', 'KGRID1', 'ablation_ref'),
    ('crowded', 'KGRID1', 'SPEC25', 'primary'),
    ('crowded', 'KGRID2', 'SPEC25', 'dose'),
    ('crowded', 'FITSWAP', 'SHIP', 'confound'),
    ('crowded', 'SHIP', 'KGRID1', 'ablation_ref'),
)
POSCTRL_FIELDS = ('n', 'better', 'worse', 'n_tie', 'p_one_sided', 'rel_median_pct',
                  'median_a', 'median_b')

SOFT_GATES = ('H-14', 'H-17')     # ⚠실패해도 무효화하지 않는다 (결과 전에 그렇게 정한다)

# ⛔**공허한 통과 금지** — 결과 0행에서 참이 되는 게이트를 만들지 않는다
#   (W8 `G-4` 부류 · `gates_b18.py:223` · `B-19` §8.1 · `B-21` `ROWLESS_OK`).
#   ★아래 여섯만이 행 없이도 판정할 수 있다 — **동결 코드·동결 산출물·패키지만 읽기
#   때문**이고, 그 여섯도 **자기 입력이 비면 통과하지 않는다**(FROZEN 부재 · verdict 부재 ·
#   `B-20` 행 0개는 전부 FAIL 이다). 자가진단이 이 집합을 **핀으로 검사**한다 —
#   행 의존 게이트가 조용히 이 집합에 들어오면 그 자리에서 FAIL 이다.
ROWLESS_OK = ('H-1', 'H-2', 'H-3', 'H-3b', 'H-11', 'H-15')


# ─────────────────────────────────────────────────────────────────────────────
# 문턱
# ─────────────────────────────────────────────────────────────────────────────
def thresholds(n=N_EP_PIN):
    """(NOMINAL, CONFIRM, NOT_EARN). ⛔손으로 적지 않는다 — `gates_b21.k_min` 이 낸다."""
    if not n:
        return (None, None, None)
    nominal = G21.k_min(n, ALPHA)
    confirm = G21.k_min(n, FAMILY_ALPHA)
    return (nominal, confirm, (n - nominal) if nominal is not None else None)


# ─────────────────────────────────────────────────────────────────────────────
# 자료 적재 (⛔`B-21` 디렉터리는 **읽기 전용**)
# ─────────────────────────────────────────────────────────────────────────────
def load_frozen_rows():
    """`B-21` 동결 결과 전건. `gates_b21._load_jsonl` 은 error 행을 버리므로 쓰지 않는다."""
    import glob
    out = []
    for f in sorted(glob.glob(B21_RESULTS)):
        with open(f) as fh:
            for line in fh:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def _json(path):
    with open(path) as fh:
        return json.load(fh)


def _md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# ★blind 증거 — 동결 판정 코드의 AST 를 읽는다
# ─────────────────────────────────────────────────────────────────────────────
def _resolve(node, mod):
    """AST 노드를 문자열로 푼다 (문자열 리터럴 · 모듈 상수 · `G.CONST`). 아니면 None."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        v = getattr(mod, node.attr, None)
        return v if isinstance(v, str) else None
    if isinstance(node, ast.Name):
        v = getattr(mod, node.id, None)
        return v if isinstance(v, str) else None
    return None


def comparison_sites(path, mod=G21):
    """[(함수명, 줄번호, (a, b))] — 비교 함수 호출의 **마지막 두 위치인자**를 푼 것."""
    with open(path) as fh:
        tree = ast.parse(fh.read())
    out = []
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        fn = (n.func.attr if isinstance(n.func, ast.Attribute)
              else n.func.id if isinstance(n.func, ast.Name) else None)
        if fn not in CMP_FUNCS:
            continue
        args = n.args[-2:] if len(n.args) >= 2 else []
        out.append((fn, n.lineno, tuple(_resolve(a, mod) for a in args)))
    return out


def blind_evidence():
    """★이 짝이 동결 판정 코드에서 **형성된 적이 없다**는 기계적 증거."""
    a_sites = comparison_sites(B21_ANALYZE)
    g_sites = comparison_sites(B21_GATES)
    a_pairs = [s[2] for s in a_sites if s[0] == 'cmp_pair']
    violations = [s for s in a_sites + g_sites
                  if BLIND_PAIR <= {x for x in s[2] if x is not None}]
    return dict(analyze_sites=len(a_sites), gates_sites=len(g_sites),
                analyze_pairs=[list(p) for p in a_pairs],
                analyze_pairs_match=(a_pairs == [tuple(p) for p in ANALYZE_PAIRS_PIN]),
                gates_sites_match=(len(g_sites) == GATES_SITES_PIN),
                violations=[list(v[2]) + [v[1]] for v in violations])


# ─────────────────────────────────────────────────────────────────────────────
# 동결 md5
# ─────────────────────────────────────────────────────────────────────────────
def frozen_check(path=FROZEN):
    """`FROZEN_md5.txt` 의 (md5, 저장소상대경로) 전건 대조.

    ⚠경로는 **`wavesplat/wavesplat` 기준**으로 적는다(= `tmp/…`) — 이 저장소에서
      *"저장소 상대"* 는 그 뜻이다. `tmp/_audit/check_batch_artifacts.py:114 _resolve`
      가 **배치 → 부모(`tmp/`) → `dirname(dirname(배치))`** 순으로만 풀기 때문이다.
      ⛔`wavesplat/tmp/…` 로 적으면 그 검사기에서 **26/26 전부 미해결**이 된다(실측).
      ⟹ 여기서도 **같은 순서**로 푼다(마지막에 `/home/vexit/wavesplat` 만 덧붙인다).
      맨 파일명만 적으면 **조용히 SKIP** 되므로 그렇게 적지 않는다(2026-08-25 실측).
    """
    if not os.path.exists(path):
        return dict(listed=0, checked=0, ok=0, missing=['FROZEN_md5.txt'], bad=[])
    listed, ok, bad, missing = 0, 0, [], []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            want, rel = parts[0], parts[1].strip()
            listed += 1
            cand = [os.path.join(HERE, rel), os.path.join(os.path.dirname(HERE), rel),
                    os.path.join(ROOT, rel), os.path.join(REPO, rel)]
            hit = next((c for c in cand if os.path.exists(c)), None)
            if hit is None:
                missing.append(rel)
                continue
            got = _md5(hit)
            if got == want:
                ok += 1
            else:
                bad.append(f'{rel}: {want[:8]} != {got[:8]}')
    return dict(listed=listed, checked=listed - len(missing), ok=ok,
                missing=missing, bad=bad)


# ─────────────────────────────────────────────────────────────────────────────
# 집계·비교 — ⛔전부 `gates_b21` 의 함수. 이 파일은 **구현하지 않는다.**
# ─────────────────────────────────────────────────────────────────────────────
def analysis(rows):
    """PRIMARY · 크기 · 음성대조 · 양성대조 재현 — 한 번에 낸다 (게이트와 분석기가 공유)."""
    agg = G21.aggregate(rows)
    val, eps = agg['val'], agg['good'][STRATUM]
    zero = agg['zero'][STRATUM]
    prim = G21.cmp_pair(val, eps, STRATUM, ARM_BASE, ARM_TREAT)      # ★blind 통계량
    main = G21.cmp_pair(val, eps, STRATUM, ARM_SHIP, ARM_TREAT)      # = B-21 ablation_ref
    band = pp_band()
    mag_clean = prim['rel_median_pct']
    mag_main = main['rel_median_pct']
    shift = (abs(mag_clean - mag_main)
             if (mag_clean is not None and mag_main is not None) else None)
    neg = dict(
        n1=G21.cmp_zero(val, zero, eps, STRATUM, ARM_SHIP),
        n1b=G21.cmp_zero(val, zero, eps, STRATUM, ARM_BASE),
        n2=G21.cmp_zero(val, zero, eps, STRATUM, ARM_TREAT))
    # 양성대조: `B-21` 의 여덟 대조를 이 파이프라인으로 다시 낸다
    repro = {}
    for st, a, b, key in POSCTRL_CONTRASTS:
        repro[f'{st}:{key}'] = G21.cmp_pair(val, agg['good'][st], st, a, b)
    return dict(agg=agg, primary=prim, main=main, neg=neg,
                magnitudes=dict(mag_clean=mag_clean, mag_main=mag_main,
                                mag_main_pin=MAG_MAIN_PIN, shift_pp=shift,
                                pp_band=band, pp_band_pin=PP_BAND_PIN,
                                band_lo=(MAG_MAIN_PIN - band), band_hi=(MAG_MAIN_PIN + band),
                                magnitude_unmoved=(shift is not None and shift <= band)),
                repro=repro)


def sparse_crosscheck(rows_analysis):
    """⚠**보고 전용** — 희소 층 `FITSWAP < KGRID1` 를 `B-20` PRIMARY-B(14/16)와 견준다.

    ⛔판정에 쓰지 않는다: 교차배치(다른 러너)·교차팔(`dk` 1e-12 대 1e-10)이고,
      `B-20` 의 희소 결과와 이 층은 **같은 16 에피소드**라 독립 확인도 아니다.
    """
    agg = rows_analysis['agg']
    c = G21.cmp_pair(agg['val'], agg['good']['sparse'], 'sparse', ARM_BASE, ARM_TREAT)
    try:
        ref = _json(B20_VERDICT)['primary_b']
    except (OSError, KeyError):
        ref = None
    return dict(here=c, b20_primary_b=ref,
                same_direction=bool(ref and c['n'] and
                                    (c['better'] >= thresholds(c['n'])[0]) ==
                                    (ref['better'] >= 12)))


# ─────────────────────────────────────────────────────────────────────────────
# 게이트 (★단일출처. 분석기는 `gates(...)` 를 **호출만** 한다)
# ⛔**공허한 통과 금지** — 모든 게이트가 자기 입력의 비어있음을 통과로 세지 않는다.
# ─────────────────────────────────────────────────────────────────────────────
def gates(rows, res=None, inherited=None):
    """[(이름, 통과, 상세)]. `inherited` = `gates_b21.gates(rows, agg)` 의 반환.

    `inherited` 가 None 이면 **H-0 은 FAIL** 이다(실행하지 않은 것을 통과로 세지 않는다).
    """
    out = []

    def g(name, cond, detail):
        out.append((name, bool(cond), detail))

    ok = [r for r in rows if 'error' not in r]
    pair = [r for r in ok if r.get('stratum') == STRATUM and r.get('arm') in PAIR_ARMS]
    have = bool(pair)
    byarm = {a: [r for r in pair if r['arm'] == a] for a in PAIR_ARMS}
    byep = {}
    for r in pair:
        byep.setdefault(r['episode'], []).append(r)

    # ── H-0 상속 ─────────────────────────────────────────────────────────────
    if inherited is None:
        g('H-0 상속 게이트 (`gates_b21.gates`) 전건 통과', False,
          '⛔상속 게이트를 실행하지 않았다 — 실행하지 않은 것을 통과로 세지 않는다')
    else:
        n_pass = sum(1 for _n, c, _d in inherited if c)
        fails = [n for n, c, _d in inherited if not c]
        g('H-0 상속 게이트 (`gates_b21.gates`) 전건 통과',
          bool(inherited) and n_pass == len(inherited) == B21_GATES_PIN,
          f'{n_pass}/{len(inherited)} (핀 {B21_GATES_PIN}) 실패={fails}')

    # ── H-1 동결 입력 md5 ────────────────────────────────────────────────────
    fz = frozen_check()
    g('H-1 동결 입력 md5 (판정코드 + 임포트 + 동결 산출물)',
      fz['listed'] > 0 and fz['ok'] == fz['listed'],
      f"목록 {fz['listed']} · 대조 {fz['checked']} · 일치 {fz['ok']} · "
      f"불일치 {fz['bad']} · 부재 {fz['missing']}")

    # ── H-2 blind (코드) ─────────────────────────────────────────────────────
    be = blind_evidence()
    g('H-2 ★blind 코드 증거: 동결 판정 코드가 이 짝을 형성한 적 없다',
      be['analyze_pairs_match'] and be['gates_sites_match'] and not be['violations'],
      f"analyze 짝 {be['analyze_pairs']} (핀 일치 {be['analyze_pairs_match']}) · "
      f"gates 호출지점 {be['gates_sites']}/{GATES_SITES_PIN} · "
      f"KGRID1×FITSWAP 동반 호출 {len(be['violations'])}건 {be['violations']}")

    # ── H-3 blind (산출물) ───────────────────────────────────────────────────
    try:
        v21 = _json(B21_VERDICT)
        top = set(v21.keys())
        sk = {st: set(v21['strata'][st].keys()) for st in v21.get('strata', {})}
        okk = (top == B21_TOP_KEYS_PIN and bool(sk)
               and all(s == B21_STRATUM_KEYS_PIN for s in sk.values()))
        g('H-3 ★blind 산출물 증거: `verdict_b21.json` 키 집합 == 핀 (이 짝의 통계량 없음)',
          okk, f'최상위 Δ={sorted(top ^ B21_TOP_KEYS_PIN)} · '
               f'층 Δ={ {k: sorted(s ^ B21_STRATUM_KEYS_PIN) for k, s in sk.items()} }')
    except OSError as e:
        g('H-3 ★blind 산출물 증거: `verdict_b21.json` 키 집합 == 핀', False, f'{e}')

    # ── H-3b `B-20` 에 혼잡 행 0개 ───────────────────────────────────────────
    import glob
    b20_rows, b20_bad = 0, 0
    for f in sorted(glob.glob(B20_RESULTS)):
        with open(f) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                b20_rows += 1
                if r.get('ncomp') is None or r['ncomp'] >= SPARSE_MAX_NCOMP:
                    b20_bad += 1
    g('H-3b ★`B-20` 행은 전건 희소 (혼잡 직접 대조가 다른 배치에도 없다)',
      b20_rows > 0 and b20_bad == 0,
      f'행 {b20_rows} · ncomp ≥ {SPARSE_MAX_NCOMP} 인 행 {b20_bad}')

    # ── H-4 표본 단일출처 ────────────────────────────────────────────────────
    if res is None or not have:
        g('H-4 표본 = `gates_b21.aggregate().good[crowded]` (재선정 0)', False,
          '⛔분석 결과가 없다')
    else:
        eps = res['agg']['good'][STRATUM]
        try:
            n_ref = _json(B21_VERDICT)['strata'][STRATUM]['n_used']
        except (OSError, KeyError):
            n_ref = None
        g('H-4 표본 = `gates_b21.aggregate().good[crowded]` (재선정 0)',
          len(eps) == N_EP_PIN and n_ref == N_EP_PIN and set(eps) == set(byep),
          f'n_used={len(eps)} (핀 {N_EP_PIN}) · verdict_b21 n_used={n_ref} · '
          f'짝 행이 덮는 에피소드 {len(byep)}')

    # ── H-5 짝 완전성 ────────────────────────────────────────────────────────
    seeds_ok = all(sorted({r['seed'] for r in R if r['arm'] == a}) == sorted(EST_SEEDS)
                   for R in byep.values() for a in PAIR_ARMS) if have else False
    g('H-5 짝 완전성 48ep × 2팔 × 3시드 = 288행 · error 0',
      have and len(pair) == N_ROWS_PIN and len(byep) == N_EP_PIN
      and len(ok) == len(rows) and seeds_ok
      and all(len(R) == len(PAIR_ARMS) * N_SEEDS_PIN for R in byep.values()),
      f'행 {len(pair)}/{N_ROWS_PIN} · 에피소드 {len(byep)}/{N_EP_PIN} · '
      f'error {len(rows) - len(ok)} · 시드집합 {sorted(EST_SEEDS)} 전건 {seeds_ok}')

    # ── H-6 ★양성대조 (통계량 수준, 비트 동일) ───────────────────────────────
    if res is None:
        g('H-6 ★양성대조: `B-21` 여덟 대조를 이 파이프라인이 비트 동일 재현', False,
          '⛔분석 결과가 없다')
    else:
        diffs = []
        try:
            v21 = _json(B21_VERDICT)
            for st, _a, _b, key in POSCTRL_CONTRASTS:
                got, want = res['repro'][f'{st}:{key}'], v21['strata'][st][key]
                for f_ in POSCTRL_FIELDS:
                    if got.get(f_) != want.get(f_):
                        diffs.append(f'{st}.{key}.{f_}: {got.get(f_)} != {want.get(f_)}')
            n_cmp = len(POSCTRL_CONTRASTS) * len(POSCTRL_FIELDS)
        except (OSError, KeyError) as e:
            diffs, n_cmp = [f'{e}'], 0
        g('H-6 ★양성대조: `B-21` 여덟 대조를 이 파이프라인이 비트 동일 재현',
          n_cmp > 0 and not diffs,
          f'대조 {n_cmp}개 · 불일치 {len(diffs)} {diffs[:5]}')

    # ── H-7 개입 분리 (팔별 방향 + ★비율) ────────────────────────────────────
    def _shifts(a):
        return [r.get('k_snap_shift_median') for r in byarm.get(a, [])]
    st_k, st_f = _shifts(ARM_TREAT), _shifts(ARM_BASE)
    fired_ok = have and all(r.get('k_snap_fired') is True for r in pair)
    k_ok = bool(st_k) and all(x is not None and x >= KGRID_SHIFT_MIN for x in st_k)
    f_ok = bool(st_f) and all(x is not None and x <= FITSWAP_SHIFT_MAX for x in st_f)
    ratio = ((_st.median(st_k) / _st.median(st_f))
             if (k_ok and f_ok and _st.median(st_f) > 0) else None)
    g('H-7 개입 분리: KGRID1 ≥ 1e-4 · FITSWAP ≤ 1e-9 · ★중앙 이동량 비 ≥ 1e6',
      fired_ok and k_ok and f_ok and ratio is not None and ratio >= SHIFT_RATIO_MIN,
      f'fired 전건 {fired_ok} · KGRID1 중앙 {_st.median(st_k) if st_k and k_ok else st_k[:1]} · '
      f'FITSWAP 중앙 {_st.median(st_f) if st_f and f_ok else st_f[:1]} · 비 {ratio}')

    # ── H-8 채점 창 짝 (b1_full_comparison 결함 부류) ────────────────────────
    win_keys = ('n_scored', 'scored_t_first', 'scored_t_last', 'tracking_frames',
                'zero_n_frames')
    win_bad = []
    for e, R in byep.items():
        for k in win_keys:
            vs = {r.get(k) for r in R}
            if len(vs) != 1 or None in vs:
                win_bad.append(f'{e}.{k}={sorted(map(str, vs))}')
        if R and R[0].get('zero_n_frames') != R[0].get('tracking_frames'):
            win_bad.append(f'{e}.zero≠tracking')
    g('H-8 채점 창: 에피소드 안 6행이 동일 · ZERO 창 == 추적 창',
      have and not win_bad, f'불일치 {len(win_bad)} {win_bad[:4]}')

    # ── H-9 코드 상태 ────────────────────────────────────────────────────────
    heads = {r.get('git_head') for r in pair}
    imds = {json.dumps(r.get('import_md5'), sort_keys=True) for r in pair}
    g('H-9 코드 상태 동일: `git_head` 1종 · `import_md5` 1종 (더러운 트리 탐지)',
      have and len(heads) == 1 and None not in heads and len(imds) == 1,
      f'git_head {len(heads)}종 {list(heads)[:1]} · import_md5 {len(imds)}종')

    # ── H-10 라우팅·VARPRO 되읽기 ────────────────────────────────────────────
    route_bad = [r['episode'] for r in pair
                 if r.get('auto_selected') != ROUTE_EXPECTED
                 or r.get('auto_real_gate') is not False
                 or r.get('ensemble_k') != ENSEMBLE_K_PIN
                 or r.get('varpro_steps') != VARPRO_STEPS_PIN
                 or r.get('varpro_obj_before') is None or r.get('varpro_obj_after') is None]
    g('H-10 라우팅 `learned` 전건 · 실측게이트 미발화 · VARPRO 실집행(obj 존재)',
      have and not route_bad, f'위반 행 {len(route_bad)} {route_bad[:3]}')

    # ── H-11 문턱 재도출 ─────────────────────────────────────────────────────
    th = thresholds(N_EP_PIN)
    try:
        t22 = _json(B22_VERDICT)['thresholds']
        sib = (t22.get('nominal'), t22.get('confirm'), t22.get('not_earn'))
        sib_a = t22.get('family_alpha')
    except (OSError, KeyError):
        sib, sib_a = (None, None, None), None
    g('H-11 문턱 재도출 (comb) == 핀 == 형제 `B-22` 동결 문턱',
      th == (NOMINAL_PIN, CONFIRM_PIN, NOT_EARN_PIN) and sib == th
      and sib_a == FAMILY_ALPHA,
      f'재도출 {th} · 핀 {(NOMINAL_PIN, CONFIRM_PIN, NOT_EARN_PIN)} · B-22 {sib} '
      f'(family_alpha {sib_a} vs {FAMILY_ALPHA})')

    # ── H-12 ★계수기 부재 기록 (한계가 참인가) ───────────────────────────────
    present = sorted({k for r in pair for k in ABSENT_COUNTERS if k in r})
    snap_missing = [k for k in PRESENT_SNAP_KEYS
                    if have and any(k not in r for r in pair)]
    g('H-12 ★계수기 부재 기록: `B-20` 충실도 계수기 6종이 행에 0회 (한계 서술이 참)',
      have and not present and not snap_missing,
      f'발견된 계수기 {present} (기대: 없음) · 스냅 키 결손 {snap_missing} · '
      f'⟹ `B-20` G-14/G-15 상당의 증거는 이 재분석에 **없다**')

    # ── H-13 / H-13b / H-14 음성대조 (참 제로필드) ───────────────────────────
    nominal = thresholds(len(res['agg']['good'][STRATUM]))[0] if (res and have) else None
    for tag, key, hard in (('H-13 (하드)', 'n1', True), ('H-13b (하드)', 'n1b', True),
                           ('H-14 (소프트)', 'n2', False)):
        arm = {'n1': ARM_SHIP, 'n1b': ARM_BASE, 'n2': ARM_TREAT}[key]
        c = res['neg'][key] if res else None
        cond = bool(c and c['n'] and nominal and c['better'] >= nominal)
        name = (f'{tag.split()[0]} 음성대조 `{arm}` < 참 제로필드 ≥ {nominal} '
                f'{"— 실패 시 무효" if hard else "— ⚠실패해도 무효화하지 않는다(라벨 접미)"}')
        g(name, cond, (f'{c["better"]}/{c["n"]} p={c["p_one_sided"]} '
                       f'rel={c["rel_median_pct"]}' if c else '⛔분석 결과가 없다'))

    # ── H-15 자기동결 ────────────────────────────────────────────────────────
    selfs = ('PREREG.md', 'gates_b20b.py', 'analyze_b20b.py')
    listed = set()
    if os.path.exists(FROZEN):
        with open(FROZEN) as fh:
            listed = {ln.split(None, 1)[1].strip() for ln in fh
                      if ln.strip() and not ln.startswith('#') and len(ln.split(None, 1)) == 2}
    miss = [s for s in selfs
            if not any(os.path.basename(x) == s for x in listed)]
    g('H-15 자기동결: 이 배치의 PREREG·게이트·분석기가 FROZEN 에 있고 일치',
      not miss and fz['listed'] > 0 and fz['ok'] == fz['listed'], f'FROZEN 미등재 {miss}')

    # ── H-17 희소 교차확인 (⚠보고 전용) ──────────────────────────────────────
    if res is None:
        g('H-17 (보고전용) 희소 교차확인 vs `B-20` PRIMARY-B', False, '⛔분석 결과가 없다')
    else:
        sc = sparse_crosscheck(res)
        ref = sc['b20_primary_b']
        g('H-17 (보고전용) 희소 교차확인 vs `B-20` PRIMARY-B — ⛔판정에 쓰지 않는다',
          sc['same_direction'],
          f"여기(B-21 희소, dk=1e-12) {sc['here']['better']}/{sc['here']['n']} "
          f"rel={sc['here']['rel_median_pct']} · "
          f"B-20(dk=1e-10) {ref['better'] if ref else None}/{ref['n'] if ref else None} "
          f"— ⚠교차배치·교차팔·같은 16 에피소드 ⟹ 독립 확인이 아니다")

    return out


def hard_pass(gate_rows):
    """소프트 게이트를 뺀 전건 통과 여부."""
    return all(c for n, c, _d in gate_rows if not n.startswith(SOFT_GATES))


# ─────────────────────────────────────────────────────────────────────────────
# 라벨 (★결과 전 확정 — `B-20` §7.2 의 이름을 물려받되 n=48 문턱으로 옮긴다)
# ─────────────────────────────────────────────────────────────────────────────
def label_b20b(res, gate_rows):
    prim = res['primary']
    n, w, worse, ties = prim['n'], prim['better'], prim['worse'], prim['n_tie']
    nominal, confirm, _ne = thresholds(n)
    neg = res['neg']
    n_used = len(res['agg']['good'][STRATUM])

    if not hard_pass(gate_rows):
        bad = [nm for nm, c, _d in gate_rows if not c and not nm.startswith(SOFT_GATES)]
        return ('REANALYSIS-NOT-VALID',
                f'⛔하드 게이트 실패 {bad} ⟹ 이 재분석으로 **아무것도 쓰지 않는다**. '
                '배선/동결 결함을 보고한다(결과 전 규칙 4).')
    if n_used < MIN_N_USED:
        return ('SAMPLE-LOSS-UNDECIDED',
                f'유효 표본 {n_used} < {MIN_N_USED} ⟹ 문턱 재계산으로 결론을 사지 않는다')
    if not (neg['n1']['n'] and neg['n1']['better'] >= nominal):
        return ('ARM-NO-SKILL-HERE',
                f"⛔배포 팔 `{ARM_SHIP}` 이 이 층에서 참 제로필드를 문턱만큼 이기지 못한다 "
                f"({neg['n1']['better']}/{neg['n1']['n']}) ⟹ 뺄 기술이 없다 (PRIMARY 무효)")
    if not (neg['n1b']['n'] and neg['n1b']['better'] >= nominal):
        return ('ARM-NO-SKILL-HERE',
                f"⛔통제 팔 `{ARM_BASE}`(= 이 대조의 기저) 이 참 제로필드를 문턱만큼 이기지 "
                f"못한다 ({neg['n1b']['better']}/{neg['n1b']['n']}) ⟹ PRIMARY 무효. "
                '⚠`FITSWAP` 은 거의 no-op 이므로 이 실패는 처치가 강했다는 뜻이 아니라 '
                '**배선 경보**다 — H-7 을 먼저 읽어라.')

    suffix = ('' if (neg['n2']['n'] and neg['n2']['better'] >= nominal)
              else '-NO-SKILL-BASE')
    tail = ('' if not suffix else
            f" ⚠음성대조 N-2 실패(`{ARM_TREAT}` < ZERO "
            f"{neg['n2']['better']}/{neg['n2']['n']}) ⟹ ⛔상대 중앙 %를 margin 으로 인용 금지.")

    if ties == n and n:
        return ('QUANT-IS-EXACTLY-A-NOOP' + suffix,
                f'★{n}/{n} 비트 동일 동점 — 혼잡에서 파수 양자화가 장 오차를 전혀 옮기지 '
                '못했다. ⛔단 H-7 이 개입 발화를 통과했는지 **먼저** 보라(배선 의심).' + tail)
    if (n - ties) < nominal:
        return ('UNDECIDABLE-TIES' + suffix,
                f'동점 {ties}/{n} 로 문턱 {nominal} 이 산술적으로 도달 불가 ⟹ 잴 수 없었다. '
                '⛔널을 "효과 없음" 으로 읽지 마라.' + tail)
    if w >= confirm:
        return ('QUANTIZATION-EARNS-INDEPENDENTLY-CROWDED' + suffix,
                f'★{w}/{n}(동점 {ties}), p={prim["p_one_sided"]:.4g}, '
                f'상대 중앙 {prim["rel_median_pct"]:.3f}% — **적합 절차를 고정한 채로도** '
                '파수를 격자에 스냅하면 혼잡 층에서 나빠진다 ⟹ 혼잡의 양자화 귀속이 '
                '**뺄셈에서 직접 측정으로** 바뀐다. ⛔*"독립 확증 하나 추가"* 로 세지 마라 '
                '(결과 전 규칙 1: 인접 통계량이 이 방향을 거의 강제한다). '
                '⛔`B-20` 희소 14/16 과 합산하지 마라(다른 층·다른 dk).' + tail)
    if w == nominal:
        return ('QUANTIZATION-EARNS-NOMINAL-CROWDED' + suffix,
                f'⚠{w}/{n}(동점 {ties}), p={prim["p_one_sided"]:.4g} — 명목 문턱만 넘었다 '
                f'(가족단위 {confirm} 미달) ⟹ **탐색적**. ⛔확증으로 쓰지 마라.' + tail)
    if worse >= nominal:
        return ('QUANTIZATION-DOES-NOT-EARN-CROWDED' + suffix,
                f'⛔{w}/{n}(진 에피소드 {worse}, 동점 {ties}), 반대꼬리 '
                f'p={prim["p_other_tail"]:.4g} — 적합 절차를 고정하면 양자화가 오히려 '
                '**낫다** ⟹ ★`B-16` 이 혼잡에서 잰 것은 표현이 아니라 적합 절차(또는 그 '
                '상호작용)다. **기여 ① 의 혼잡 부분을 다시 쓴다.** ⚠단 H-7·H-12 를 먼저 '
                '읽어라. ⛔동결 판정은 수정하지 않고 새 절을 붙인다.' + tail)
    return ('QUANTIZATION-UNDECIDED-CROWDED' + suffix,
            f'{w}/{n}(진 {worse}, 동점 {ties}), p={prim["p_one_sided"]:.4g} — 어느 꼬리도 '
            f'문턱 {nominal} 에 닿지 않았다. ⛔*"혼잡에서는 양자화 단독이 해치지 않는다"* 로 '
            '쓰지 마라(널은 부재의 증거가 아니다). 쓸 문장 = *"n=48 로도 부호가 서지 '
            '않는다"* + **층간 담지자 분리** 신설(결과 전 규칙 2).' + tail)


# ─────────────────────────────────────────────────────────────────────────────
# ★조건부 후속 `B-20c` (288런) — **결과 전에** 발화 조건을 정한다
# ─────────────────────────────────────────────────────────────────────────────
def followup_triggers(label, res, gate_rows):
    """계수기가 **판정에 필요해지는** 조건. 셋 중 하나라도 참이면 `B-20c` 를 등록한다.

    ⛔*"더 재고 싶어서"* 288런을 돌리지 않는다(`B-8b` 정지규칙의 대칭형).
    """
    m = res['magnitudes']
    t1 = label.startswith(('QUANTIZATION-UNDECIDED', 'QUANTIZATION-DOES-NOT-EARN',
                           'UNDECIDABLE-TIES', 'QUANT-IS-EXACTLY-A-NOOP'))
    t2 = bool(m['shift_pp'] is not None and m['shift_pp'] > m['pp_band'])
    t3 = not hard_pass(gate_rows)
    return dict(
        T1_direct_contrast_not_confirmed=t1,
        T2_clean_magnitude_left_band=t2,
        T3_hard_gate_failed=t3,
        fire=bool(t1 or t2 or t3),
        note=('T1 ⟹ *"혼잡에서 정말 안 해치는가"* 와 *"섭동이 혼잡 원자에서 너무 작았는가"* 를 '
              '가르려면 Δψ·비트동일 비율이 필요하다. '
              'T2 ⟹ 통제 팔이 실제로 얼마나 움직였는지(|Δw|·|Δφ|)를 알아야 크기를 귀속한다. '
              'T3 ⟹ 전제가 무너졌으므로 재측정한다.'))


# ─────────────────────────────────────────────────────────────────────────────
# ★런 전 자가진단 — 게이트를 켜 보지 않고 배치를 판정하지 않는다
#   ⛔**동결 자료를 읽지 않는다** (blind 보존). 전부 합성 픽스처다.
# ─────────────────────────────────────────────────────────────────────────────
def _fx_row(ep, arm, seed, val, zero=0.30):
    return dict(stratum=STRATUM, arm=arm, episode=ep, seed=seed, seed_kind='est',
                avg_nRMSE=val, zero_nRMSE=zero, zero_n_frames=160, tracking_frames=160,
                n_scored=160, scored_t_first=14.1, scored_t_last=30.0, ncomp=15,
                gen_seed=1581, bf='BF4', git_head='deadbeef', import_md5={'a': 'b'},
                auto_selected=ROUTE_EXPECTED, auto_real_gate=False,
                ensemble_k=ENSEMBLE_K_PIN, varpro_steps=VARPRO_STEPS_PIN,
                varpro_obj_before=1.0, varpro_obj_after=0.4, varpro_fired=True,
                n_gaussians=22, k_snap_fired=True,
                k_snap_dk=(G21.DK1 if arm == ARM_TREAT else G21.FITSWAP_DK),
                k_snap_shift_median=(0.0104 if arm == ARM_TREAT else 3.9e-13),
                k_snap_cells=22)


def _fixture(n_ep=N_EP_PIN, w_target=40):
    """48 에피소드 × 2팔 × 3시드. `w_target` 개에서 `FITSWAP < KGRID1`."""
    rows = []
    for i in range(n_ep):
        ep = f'BF4_15comp_seed{1581 + i}_uniform_ra_rf_ds'
        base, treat = (0.13, 0.21) if i < w_target else (0.21, 0.13)
        for s in EST_SEEDS:
            rows.append(_fx_row(ep, ARM_BASE, s, base))
            rows.append(_fx_row(ep, ARM_TREAT, s, treat))
    return rows


def selftest(verbose=True):
    """★게이트가 **깨질 때 깨지는지** 를 확인한다. 동결 자료는 건드리지 않는다.

    ⚠픽스처에는 `SHIP`·`KGRID2`·`SPEC25` 행이 없으므로 `H-0`·`H-4`·`H-6`·`H-17` 처럼
      동결 산출물/전체 팔을 요구하는 게이트는 **FAIL 이 정상**이다. 자가진단은
      **각 게이트가 자기 고장에 반응하는지**만 본다.
    """
    fails = []

    def check(tag, cond, detail=''):
        if not cond:
            fails.append(f'{tag}: {detail}')
        if verbose:
            print(f'{"ok  " if cond else "FAIL"} {tag} {detail}')

    # ① 문턱
    check('문턱(48) == (31, 32, 17)', thresholds(48) == (31, 32, 17), str(thresholds(48)))
    check('문턱(16) == (12, 13, 4)', thresholds(16) == (12, 13, 4), str(thresholds(16)))
    check('연역 하한 29 < NOMINAL 31', DEDUCIBLE_LOWER_BOUND < NOMINAL_PIN)

    # ② blind 증거 (동결 **코드**만 읽는다 — 결과 파일은 열지 않는다)
    be = blind_evidence()
    check('blind: analyze 짝 == 핀', be['analyze_pairs_match'], str(be['analyze_pairs']))
    check('blind: gates 호출지점 == 핀', be['gates_sites_match'], str(be['gates_sites']))
    check('blind: KGRID1×FITSWAP 동반 호출 0건', not be['violations'], str(be['violations']))

    # ③ ⛔공허한 통과 금지 — 행 0개에서 통과하는 게이트는 `ROWLESS_OK` 뿐이어야 한다
    empty = gates([], None, None)
    passed_empty = {n.split()[0] for n, c, _d in empty if c}
    all_names = {n.split()[0] for n, _c, _d in empty}
    check('행 0개에서 통과하는 게이트 ⊆ ROWLESS_OK',
          passed_empty <= set(ROWLESS_OK), str(sorted(passed_empty)))
    check('행 의존 게이트는 행 0개에서 전건 FAIL',
          not (passed_empty & (all_names - set(ROWLESS_OK))),
          str(sorted(passed_empty & (all_names - set(ROWLESS_OK)))))

    # ④ 게이트가 고장에 반응하는가 (픽스처 변형)
    base = _fixture()
    muts = [
        ('H-5', lambda R: R[:-6]),
        ('H-5', lambda R: [dict(r, seed=99) if r is R[0] else r for r in R]),
        ('H-7', lambda R: [dict(r, k_snap_shift_median=1e-9) if r['arm'] == ARM_TREAT
                           else r for r in R]),
        ('H-7', lambda R: [dict(r, k_snap_shift_median=6e-3) if r['arm'] == ARM_BASE
                           else r for r in R]),
        ('H-7', lambda R: [dict(r, k_snap_fired=False) if r['arm'] == ARM_BASE
                           else r for r in R]),
        ('H-8', lambda R: [dict(r, n_scored=301) if r['arm'] == ARM_TREAT else r
                           for r in R]),
        ('H-8', lambda R: [dict(r, zero_n_frames=120) for r in R]),
        ('H-9', lambda R: [dict(r, git_head='x') if r is R[0] else r for r in R]),
        ('H-10', lambda R: [dict(r, auto_selected='v2_konly') for r in R]),
        ('H-10', lambda R: [dict(r, varpro_obj_before=None) for r in R]),
        ('H-10', lambda R: [dict(r, auto_real_gate=None) for r in R]),
        ('H-12', lambda R: [dict(r, k_dpsi_max=4.8) for r in R]),
        ('H-12', lambda R: [{k: v for k, v in r.items() if k != 'k_snap_cells'}
                            for r in R]),
    ]
    clean = {n for n, c, _d in gates(base, None, None) if c}
    for tag, mut in muts:
        rows = mut([dict(r) for r in base])
        got = {n for n, c, _d in gates(rows, None, None) if c}
        broke = any(n.startswith(tag) for n in clean - got)
        check(f'변형 → {tag} FAIL', broke,
              f'(깨끗할 때 통과 {sorted(n.split()[0] for n in clean)})')

    # ⑤ 라벨 분기 전수
    def _res(w, n=48, ties=0, n1=True, n1b=True, n2=True, shift=0.14):
        prim = dict(n=n, better=w, worse=n - w - ties, n_tie=ties,
                    p_one_sided=0.5, p_other_tail=0.5, rel_median_pct=-38.5)
        mk = lambda ok: dict(n=n, better=(n if ok else 0), p_one_sided=0.0,  # noqa: E731
                             rel_median_pct=-50.0)
        return dict(primary=prim, neg=dict(n1=mk(n1), n1b=mk(n1b), n2=mk(n2)),
                    agg=dict(good={STRATUM: list(range(n))}),
                    magnitudes=dict(mag_clean=-38.5, mag_main=MAG_MAIN_PIN,
                                    shift_pp=shift, pp_band=PP_BAND_PIN,
                                    magnitude_unmoved=shift <= PP_BAND_PIN))
    good_gates = [('H-x', True, '')]
    cases = [
        (_res(40), 'QUANTIZATION-EARNS-INDEPENDENTLY-CROWDED'),
        (_res(32), 'QUANTIZATION-EARNS-INDEPENDENTLY-CROWDED'),
        (_res(31), 'QUANTIZATION-EARNS-NOMINAL-CROWDED'),
        (_res(29), 'QUANTIZATION-UNDECIDED-CROWDED'),
        (_res(10), 'QUANTIZATION-DOES-NOT-EARN-CROWDED'),
        (_res(48, ties=48), 'QUANT-IS-EXACTLY-A-NOOP'),
        (_res(20, ties=25), 'UNDECIDABLE-TIES'),
        (_res(40, n1=False), 'ARM-NO-SKILL-HERE'),
        (_res(40, n1b=False), 'ARM-NO-SKILL-HERE'),
        (_res(40, n2=False), 'QUANTIZATION-EARNS-INDEPENDENTLY-CROWDED-NO-SKILL-BASE'),
    ]
    for res, want in cases:
        got = label_b20b(res, good_gates)[0]
        check(f'라벨 w={res["primary"]["better"]} ties={res["primary"]["n_tie"]} → {want}',
              got == want, got)
    check('라벨: 하드 게이트 실패 → REANALYSIS-NOT-VALID',
          label_b20b(_res(40), [('H-9 x', False, '')])[0] == 'REANALYSIS-NOT-VALID')
    check('표본 손실 → SAMPLE-LOSS-UNDECIDED',
          label_b20b(_res(40, n=30), good_gates)[0] == 'SAMPLE-LOSS-UNDECIDED')

    # ⑥ 후속 발화 조건
    check('T1: UNDECIDED 면 후속 발화',
          followup_triggers('QUANTIZATION-UNDECIDED-CROWDED', _res(29), good_gates)['fire'])
    check('T2: 밴드 밖이면 후속 발화',
          followup_triggers('QUANTIZATION-EARNS-INDEPENDENTLY-CROWDED',
                            _res(40, shift=9.9), good_gates)['fire'])
    check('CONFIRM + 밴드 안이면 후속 없음',
          not followup_triggers('QUANTIZATION-EARNS-INDEPENDENTLY-CROWDED',
                                _res(40), good_gates)['fire'])

    print(f'\n자가진단 {"PASS" if not fails else "FAIL"} — 실패 {len(fails)}')
    for f in fails:
        print('  ⛔', f)
    return not fails


if __name__ == '__main__':
    sys.exit(0 if selftest() else 1)
