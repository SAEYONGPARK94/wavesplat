#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""analyze_w4.py -- the W4_SOLVER verdict analyser.

    python3 tmp/W4_SOLVER_2026-08-19/analyze_w4.py smoke_results.jsonl   # dev / dry run
    python3 tmp/W4_SOLVER_2026-08-19/analyze_w4.py                       # results.jsonl

WRITTEN AND FROZEN **BEFORE ANY VERDICT ROW WAS READ.**
S17 wrote its analyser after looking at the rows.  This file inverts that order: it was
authored against `PREREG.md` + `AMENDMENT_1.md` + `AMENDMENT_2.md` and exercised end to end
on `smoke_results.jsonl` (2 episodes/arm -- meaningless as a verdict, sufficient as a code
path).  `results.jsonl` was NOT opened while this file was written.

WHAT THIS FILE OWNS
  - the frozen aggregation (PREREG section 6): episode-level median-of-3 over estimator
    seeds FIRST, pairing on episode SECOND; relative difference (treat - ctrl)/ctrl;
    listwise deletion of missing; exact-float ties dropped from the pair and `n` reduced.
  - the PRIMARY family multiplicity rule.  **AMENDMENT_1 A-1 makes Holm-2 the ONLY valid
    rule** -- the PREREG headline's fixed thresholds were deleted, because an exhaustive
    enumeration found 94 of 4,225 cells where the two rules disagree, including one where
    OPTIMIZER-INERT and OPTIMIZER-EARNS swap.  Exact binomial throughout; no normal
    approximation anywhere in this file.
  - the total order of the verdict (AMENDMENT_1 A-8), including which gates may kill it.
  - the SECONDARY tests, the D descriptives, and the always-printed prohibition list.

WHAT THIS FILE DELIBERATELY DOES NOT OWN
  - the arm table, the shell keys, the DCC contract and the gate thresholds: **imported
    from `run_w4.py`** so there is exactly one definition in the batch.  Importing that
    module is side-effect-free apart from `os.chdir` and two read-only git calls (its heavy
    imports all live inside functions); this file restores the cwd.  If the import fails the
    analyser falls back to a local mirror of the constants AND says so loudly.
  - the run-free gates.  `run_w4.gates()` already decided them and wrote `gates.log`.  G-7
    is re-evaluated live here because it is the one that fails silently (a sign flip).
  - deciding anything.  Every threshold in this file is a transcription with a citation.

DUAL VERIFICATION (standing repo directive)
  The row-dependent gates are evaluated twice: once by this file's own implementation
  (which yields the per-gate booleans the verdict ordering needs) and once by
  `run_w4.posthoc()` (the runner's own).  A disagreement is printed as a BLOCKER.
"""
import json
import math
import os
import sys
from math import comb

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_CWD0 = os.getcwd()

# ── single source of truth: the runner (PREREG section 2/5/6) ────────────────
sys.path.insert(0, HERE)
try:
    import run_w4 as W4
    ARMS = W4.ARMS
    ARM_ORDER = W4.ARM_ORDER
    SHELL_KEYS = list(W4.SHELL_KEYS)
    BARE_KEYS = list(W4.BARE_KEYS)
    SEEDS = tuple(W4.SEEDS)
    VP_STEPS = W4.VP_STEPS
    SHIP_INIT_ITERS = W4.SHIP_INIT_ITERS
    N_EP_EXPECTED = W4.N_EP_EXPECTED
    missing_fields = W4.missing_fields
    IMPORT_OK, IMPORT_ERR = True, None
except Exception as _e:                                    # pragma: no cover
    W4 = None
    IMPORT_OK, IMPORT_ERR = False, repr(_e)[:200]
    BARE_KEYS = ['0_5', '5_10', '10_20', '20_30', '30_50', '50_75', '75_100',
                 '100_150', '150_inf']
    SHELL_KEYS = ['obs:' + k for k in BARE_KEYS]
    SEEDS = (1595, 1596, 1597)
    VP_STEPS, SHIP_INIT_ITERS, N_EP_EXPECTED = 400, 500, 64
    ARM_ORDER = ('ZERO', 'L-SHIP', 'L-SHIP-DUP', 'L-VP', 'L-ITER0', 'L-VP-NW',
                 'L-STEP400', 'L-MERGE', 'ORA-K-CVX')
    ARMS = {a: dict(kind=('zero' if a == 'ZERO' else 'cvx' if a == 'ORA-K-CVX' else 'est'),
                    vp=(VP_STEPS if a in ('L-VP', 'L-VP-NW') else 0)) for a in ARM_ORDER}

    def missing_fields(row):                               # degraded, announced below
        return []
os.chdir(_CWD0)

POLISH_STEPS = 320          # AMENDMENT_2 A2-1: nsamp 4 x polish_iters 80, NOT reachable
                            # by `learned_init_iters`.  L-SHIP 820, L-ITER0 320.
ALPHA = 0.05                # PREREG section 1: family alpha of the PRIMARY family
N_MIN = 56                  # PREREG section 5 D-4: effective pairs below this -> UNDECIDED
IUT_MIN_FRAC = 41.0 / 64.0  # PREREG section 1 IUT: L-SHIP < ZERO on >= 41/64
SEC_ALPHA = 0.025           # PREREG section 3: SECONDARY, one-sided alpha


# ══════════════════════════════════════════════════════════════════════════════
#  EXACT BINOMIAL (sign test).  No scipy, no normal approximation.
#  Cross-checked against AMENDMENT_1 A-2: p_one(42|64)=0.008429, p_two(42|64)=0.016858.
# ══════════════════════════════════════════════════════════════════════════════
def p_ge(k, n):
    return sum(comb(n, i) for i in range(k, n + 1)) / 2.0 ** n


def p_le(k, n):
    return sum(comb(n, i) for i in range(0, k + 1)) / 2.0 ** n


def p_one(k, n):
    """One-sided (greater): P(X >= k | p=0.5)."""
    return p_ge(k, n) if n else float('nan')


def p_two(k, n):
    """Two-sided, the symmetric doubling that the PREREG's 42/22 region encodes."""
    return min(1.0, 2.0 * min(p_ge(k, n), p_le(k, n))) if n else float('nan')


def region_one(n, alpha):
    """{k : one-sided p <= alpha}; PREREG A-1 says >=41 at 0.025 and >=40 at 0.05, n=64."""
    return [k for k in range(n + 1) if p_one(k, n) <= alpha]


def region_two(n, alpha):
    """{k : two-sided p <= alpha}; A-1 says >=42 or <=22 at 0.025, >=41 or <=23 at 0.05."""
    return [k for k in range(n + 1) if p_two(k, n) <= alpha]


def _fmt_region(reg, n):
    if not reg:
        return '(empty)'
    hi = [k for k in reg if k > n / 2.0]
    lo = [k for k in reg if k < n / 2.0]
    parts = []
    if hi:
        parts.append('>=%d' % min(hi))
    if lo:
        parts.append('<=%d' % max(lo))
    return ' or '.join(parts)


def power_of(region, n, pi):
    return sum(comb(n, k) * pi ** k * (1.0 - pi) ** (n - k) for k in region)


def mde_pi(region, n, target=0.80):
    """Smallest win-probability pi>0.5 the region detects with `target` power.

    This is the honest content of a null: `OPTIMIZER-INERT` does not mean "no effect", it
    means "no effect at least this large".  PREREG has no MDE clause -- printing it beside
    every null is a standing repo rule (feedback_detection_rate_needs_null)."""
    if not region or not n:
        return None
    lo, hi = 0.5, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if power_of(region, n, mid) >= target:
            hi = mid
        else:
            lo = mid
    return hi


def sign_ci_median(diffs, alpha=0.05):
    """Distribution-free CI for the median of the paired differences (order statistics,
    exact binomial coverage).  This is the effect size a null fails to exclude."""
    d = sorted(float(x) for x in diffs if x is not None and np.isfinite(x))
    n = len(d)
    if n < 6:
        return None, None, None
    k = 0
    for cand in range(0, n // 2 + 1):
        tail = sum(comb(n, i) for i in range(0, cand)) / 2.0 ** n
        if tail <= alpha / 2.0:
            k = cand
        else:
            break
    if k < 1:
        return d[0], d[-1], 1.0
    cov = 1.0 - 2.0 * sum(comb(n, i) for i in range(0, k)) / 2.0 ** n
    return d[k - 1], d[n - k], cov


def holm2(tests, alpha=ALPHA):
    """Holm step-down on the 2-member PRIMARY family (AMENDMENT_1 A-1, the ONLY rule).

    p ascending: the first is tested at alpha/2, the second at alpha, and -- Holm -- the
    second can only be rejected if the first was."""
    order = sorted(tests, key=lambda t: (t[1], t[0]))
    levels = [alpha / 2.0, alpha]
    out, alive = {}, True
    for i, (name, p) in enumerate(order):
        lvl = levels[i] if i < len(levels) else alpha
        rej = bool(alive and p <= lvl)
        alive = rej
        out[name] = dict(rank=i + 1, alpha=lvl, p=p, reject=rej)
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  AGGREGATION -- PREREG section 6, transcribed
# ══════════════════════════════════════════════════════════════════════════════
def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and np.isfinite(x)


def med3(rows, getter):
    """{(arm, episode): dict(v=median-over-seeds, n_seeds, seeds)} -- STEP ONE of the
    frozen aggregation.  The seed median comes first, the pairing second."""
    acc = {}
    for r in rows:
        v = getter(r)
        if not _num(v):
            continue
        acc.setdefault((r['arm'], r['episode']), []).append((r.get('est_seed'), float(v)))
    out = {}
    for key, sv in acc.items():
        vals = [v for _, v in sv]
        out[key] = dict(v=float(np.median(vals)), n_seeds=len(vals),
                        seeds=sorted(s for s, _ in sv if s is not None))
    return out


def pair(tab, treat, ctrl, episodes):
    """STEP TWO.  Pair unit = episode.  rel = (treat - ctrl)/ctrl.  Missing -> listwise
    deletion.  Exact float equality -> dropped from the pair, `n` reduced (PREREG 6)."""
    rec, drop_missing, drop_tie, ctrl_zero, partial = [], [], [], [], []
    for ep in episodes:
        a, b = tab.get((treat, ep)), tab.get((ctrl, ep))
        if a is None or b is None:
            drop_missing.append(ep)
            continue
        av, bv = a['v'], b['v']
        if av == bv:                          # exact float equality only
            drop_tie.append(ep)
            continue
        rel = ((av - bv) / bv) if bv != 0.0 else float('nan')
        if bv == 0.0:
            ctrl_zero.append(ep)
        if a['n_seeds'] < len(SEEDS) or b['n_seeds'] < len(SEEDS):
            partial.append((ep, a['n_seeds'], b['n_seeds']))
        rec.append(dict(episode=ep, treat=av, ctrl=bv, rel=rel, win=bool(av < bv)))
    rels = [r['rel'] for r in rec if np.isfinite(r['rel'])]
    lo, hi, cov = sign_ci_median(rels)
    return dict(
        treat=treat, ctrl=ctrl, n=len(rec), wins=sum(1 for r in rec if r['win']),
        losses=sum(1 for r in rec if not r['win']),
        n_missing=len(drop_missing), n_tie=len(drop_tie), n_ctrl_zero=len(ctrl_zero),
        partial_seeds=partial,
        median_rel=(float(np.median(rels)) if rels else float('nan')),
        mean_rel=(float(np.mean(rels)) if rels else float('nan')),
        median_rel_ci=(lo, hi, cov),
        episodes_missing=sorted(drop_missing)[:8], episodes_tie=sorted(drop_tie)[:8],
        rows=rec)


def one_sided(pr, alpha):
    pr = dict(pr)
    pr['test'] = 'one-sided'
    pr['p'] = p_one(pr['wins'], pr['n'])
    pr['region'] = region_one(pr['n'], alpha)
    pr['thresh'] = (min(pr['region']) if pr['region'] else None)
    pr['alpha_used'] = alpha
    pr['pass'] = bool(pr['region'] and pr['wins'] >= min(pr['region']))
    return pr


def two_sided(pr, alpha):
    pr = dict(pr)
    pr['test'] = 'two-sided'
    pr['p'] = p_two(pr['wins'], pr['n'])
    pr['region'] = region_two(pr['n'], alpha)
    pr['alpha_used'] = alpha
    pr['pass'] = bool(pr['wins'] in pr['region'])
    return pr


# ══════════════════════════════════════════════════════════════════════════════
#  SHELL KEYS -- the trap the repository keeps stepping into
#  `run_w4.py` writes ONE convention (`obs:`) on every arm.  Some legacy rows are bare.
#  Resolve per row, never strip, and refuse to average across conventions silently.
# ══════════════════════════════════════════════════════════════════════════════
def shell_getter(field, bare):
    def get(r):
        d = r.get(field)
        if not isinstance(d, dict):
            return None
        if ('obs:' + bare) in d:
            return d['obs:' + bare]
        if bare in d:
            return d[bare]
        return None
    return get


def shell_convention_audit(rows):
    """Which convention each arm's rows actually carry.  A mixed arm is a BLOCKER: `obs:`
    and `ship:` differ by 1.8-4.1x on the same sample, so a silent mix is a fabricated
    number, not a rounding issue."""
    seen = {}
    for r in rows:
        d = r.get('shell_rmse')
        if not isinstance(d, dict):
            continue
        conv = set()
        for k in d:
            conv.add('obs:' if k.startswith('obs:')
                     else 'ship:' if k.startswith('ship:')
                     else 'bare')
        seen.setdefault(r['arm'], set()).update(conv)
    return seen


# ══════════════════════════════════════════════════════════════════════════════
#  GATES -- own implementation (AMENDMENT_1 A-8 needs the gates SEPARATED, because only
#  G-1/G-2/G-3/G-5/G-7/G-8 may halt; G-4 and G-9 may not).
# ══════════════════════════════════════════════════════════════════════════════
def eval_gates(rows):
    good = [r for r in rows if 'error' not in r]
    by = {}
    for r in good:
        by.setdefault(r['arm'], []).append(r)
    est = [r for r in good if (ARMS.get(r['arm']) or {}).get('kind') == 'est']
    g = {}

    def add(name, status, detail, halting):
        g[name] = dict(status=status, detail=detail, halting=halting)

    # G-1: every estimator row actually took the learned branch (constructive pin).
    if est:
        bad = [(r['arm'], r['episode']) for r in est
               if r.get('route') != 'learned' or r.get('learned_init_calls') != 1]
        add('G-1', 'PASS' if not bad else 'FAIL',
            '%d/%d est rows route==learned AND learned_init_calls==1%s'
            % (len(est) - len(bad), len(est),
               '' if not bad else '; e.g. %s' % (bad[:3],)), True)
    else:
        add('G-1', 'PENDING', 'no estimator rows', True)

    # G-2: AMENDMENT_2 A2-2 -- exact step counts, not ">0 vs ==0".
    a0 = [r.get('adam_steps_phase1') for r in by.get('L-ITER0', [])]
    a1 = [r.get('adam_steps_phase1') for r in by.get('L-SHIP', [])]
    if a0 or a1:
        ok = ((all(v == POLISH_STEPS for v in a0) if a0 else True)
              and (all(v == POLISH_STEPS + SHIP_INIT_ITERS for v in a1) if a1 else True))
        add('G-2', 'PASS' if ok else 'FAIL',
            'L-ITER0 distinct=%s (must be [%d] = proposal polish only) | L-SHIP distinct=%s '
            '(must be [%d] = polish + %d joint fit)'
            % (sorted(set(a0)), POLISH_STEPS, sorted(set(a1)),
               POLISH_STEPS + SHIP_INIT_ITERS, SHIP_INIT_ITERS), True)
    else:
        add('G-2', 'PENDING', 'neither L-ITER0 nor L-SHIP present', True)

    # G-2c: AMENDMENT_2 A2-4 -- budget parity is what makes S-2b parameterisation-only.
    bv = sorted({r.get('adam_steps_phase1') for r in by.get('L-VP', [])})
    bs = sorted({r.get('adam_steps_phase1') for r in by.get('L-STEP400', [])})
    if bv and bs:
        add('G-2c', 'PASS' if bv == bs else 'FAIL',
            'L-VP %s vs L-STEP400 %s -- S-2b is a parameterisation-only contrast ONLY if '
            'these match' % (bv, bs), False)
    else:
        add('G-2c', 'PENDING', 'L-VP and/or L-STEP400 absent', False)

    # G-2b: unattributed Adam steps (arms_config alarm, not a gate).
    ua = [(r['arm'], r['episode'], r.get('adam_steps_other')) for r in est
          if r.get('adam_steps_other')]
    add('G-2b', 'PASS' if not ua else 'INVESTIGATE',
        '%d est rows with adam_steps_other>0%s'
        % (len(ua), '' if not ua else '; e.g. %s' % (ua[:3],)), False)

    # G-3: AMENDMENT_2 A2-3 -- read the arm table's `vp` declaration, do not name L-VP.
    vps, bad3 = {}, []
    for a, v in by.items():
        s = sorted({r.get('vp_steps') for r in v})
        vps[a] = s
        want = [VP_STEPS] if (ARMS.get(a) or {}).get('vp') else [0]
        if s != want:
            bad3.append((a, s, want))
    add('G-3', ('PASS' if not bad3 else 'FAIL') if vps else 'PENDING',
        'vp_steps %s%s' % (vps, '' if not bad3 else ' MISMATCH %s' % (bad3,)), True)

    # G-4: RECORD ONLY (AMENDMENT_1 A-7 demoted it; it can never kill PRIMARY).
    if 'L-MERGE' in by:
        mr = [r.get('merge_removed') for r in by['L-MERGE'] if _num(r.get('merge_removed'))]
        mc = [r.get('merge_calls') for r in by['L-MERGE'] if _num(r.get('merge_calls'))]
        med = float(np.median(mr)) if mr else 0.0
        add('G-4', 'RECORDED' if med > 0 else 'NO-INTERVENTION',
            'L-MERGE merge_removed median=%s merge_calls median=%s -- record only; D-4 is '
            'a descriptive, so this gate cannot downgrade anything (A-7)'
            % (med, float(np.median(mc)) if mc else None), False)
    else:
        add('G-4', 'PENDING', 'no L-MERGE rows', False)

    # G-5: bit determinism.
    ship = {r['episode']: r['avg_nRMSE'] for r in by.get('L-SHIP', [])
            if r.get('est_seed') == SEEDS[0]}
    dups = [(r['episode'], abs(r['avg_nRMSE'] - ship[r['episode']]))
            for r in by.get('L-SHIP-DUP', []) if r['episode'] in ship]
    if dups:
        nz = [e for e, d in dups if d != 0.0]
        add('G-5', 'PASS' if not nz else 'FAIL',
            '%d pairs, max|delta|=%.3e, nonzero=%d' % (len(dups), max(d for _, d in dups),
                                                       len(nz)), True)
    else:
        add('G-5', 'PENDING', 'no L-SHIP-DUP/L-SHIP seed-%s overlap' % (SEEDS[0],), True)

    # G-6 DCC -- its own step in the verdict ordering, not folded into the halting gates.
    bad6 = [(r['arm'], r['episode'], missing_fields(r)) for r in good if missing_fields(r)]
    add('G-6', 'PASS' if not bad6 else 'FAIL',
        '%d/%d rows satisfy the per-arm-kind contract%s'
        % (len(good) - len(bad6), len(good),
           '' if not bad6 else '; e.g. %s' % (bad6[:3],)), False)

    # G-6b / G-6c: the b1_full_comparison BLOCKER as a gate.
    bad_fr, bad_z = [], []
    for ep in sorted({r['episode'] for r in good}):
        rs = [r for r in good if r['episode'] == ep]
        if len({r.get('n_match') for r in rs}) != 1:
            bad_fr.append((ep, sorted({(r['arm'], r.get('n_match')) for r in rs})))
        zs = [r['zero_nRMSE'] for r in rs if _num(r.get('zero_nRMSE'))]
        if zs and (max(zs) - min(zs)) > 1e-12:
            bad_z.append((ep, max(zs) - min(zs)))
    add('G-6b', 'PASS' if not bad_fr else 'FAIL',
        '%d episodes where arms were scored on different frame counts%s'
        % (len(bad_fr), '' if not bad_fr else '; e.g. %s' % (bad_fr[:2],)), False)
    add('G-6c', 'PASS' if not bad_z else 'FAIL',
        '%d episodes where zero_nRMSE differs across arms' % len(bad_z), False)

    # G-7: phase convention identity.  Re-evaluated live; this is the gate that fails
    # silently (a sign flip reconstructs at RMSE 1.37 instead of 9e-16).
    try:
        ref, badc, got = W4._g7_phase_identity()
        ok7 = (ref <= 1e-12) and (got is None or got <= 1e-12)
        add('G-7', 'PASS' if ok7 else 'FAIL',
            'atan2(-b,a) max|d|=%.3e (tol 1e-12); the sequential_update.py:1335 convention '
            'atan2(+b,a) gives %.3e; varpro.ab_to_wphi max|d|=%s'
            % (ref, badc, ('%.3e' % got) if got is not None else 'n/a (module absent)'),
            True)
    except Exception as e:
        gl = os.path.join(HERE, 'gates.log')
        line = ''
        if os.path.exists(gl):
            for ln in open(gl, encoding='utf-8', errors='replace'):
                if ln.startswith('G-7'):
                    line = ln.strip()
        if line.endswith('PASS'):
            add('G-7', 'PASS(from gates.log)', line, True)
        else:
            add('G-7', 'PENDING', 'not evaluable live (%s) and gates.log gives %r'
                % (repr(e)[:80], line), True)

    # G-8: one code digest over the whole batch.
    digs = sorted({r.get('code_md5') for r in good})
    heads = sorted({r.get('git_head') for r in good})
    add('G-8', ('PASS' if len(digs) == 1 else 'FAIL') if digs else 'PENDING',
        '%d distinct code_md5 %s | %d distinct git_head %s'
        % (len(digs), digs, len(heads), [str(h)[:8] for h in heads]), True)

    # G-9: watchdog.  A-8: supervision NEVER kills PRIMARY.
    add('G-9', 'OUT-OF-SCOPE', 'progress supervision lives in watchdog.py (file state only)'
        ' and by AMENDMENT_1 A-8 cannot kill the PRIMARY family', False)
    return g


HALTING = ('G-1', 'G-2', 'G-3', 'G-5', 'G-7', 'G-8')


# ══════════════════════════════════════════════════════════════════════════════
#  PROHIBITIONS -- PREREG section 7 + AMENDMENT_1 A-4.  ALWAYS printed under the verdict.
# ══════════════════════════════════════════════════════════════════════════════
PROHIBITIONS = [
    "1. VARPRO 를 '발명/구현했다' 를 novelty 로 인용 금지 -- Golub-Pereyra 1973, 그리고 "
    "arXiv:2511.01234(2025-11) 가 '변수 소거가 안장점을 축소 지형의 국소 최대로 바꾼다' 를 "
    "정리로 이미 증명했다. 살 수 있는 것은 영역 주장 + 측정뿐.",
    "2. ORA-K-CVX 대비 비를 순수 솔버 이득으로 인용 금지 -- 6요인 묶음이다.",
    "3. '벽을 닫았다' 금지 -- W2 가 창 잔차를 잡음 바닥까지 내려도 필드가 x37 임을 이미 쟀다. "
    "혼잡은 초해상 영역(성분 분리 1.29 Rayleigh vs 250 m 외삽 요구 0.07).",
    "4. 전역 nRMSE 헤드라인 금지 -- 판정은 혼잡 층 한정.",
    "5. 희소 주장 금지 -- 표현 아티팩트 확정(HOSGATE).",
    "6. 배포 권고 금지 -- 이 배치는 확증이지 배포 결정이 아니다.",
    "7. 'k 표류가 원인' 금지 -- 표류-nRMSE 상관 -0.05, 그리고 k_drift_valid 가 SHIP 0/144 "
    "= 배포 팔의 표류는 미측정이다(A-9).",
    "8. 범위 문구 의무: 합성 코퍼스 - 선형 GT - direction_deg=0 - 단일 체크포인트 - 혼잡 층 - "
    "init_method='learned' 고정(배포 라우터가 아님).",
    "A-4. L-VP 의 이득을 '매개화 덕분' 이라고 단정 금지 -- L-VP 는 >=4요인 묶음이고, S-2 - S-2b "
    "- S-2c 가 함께 통과할 때만 그 귀속이 허용된다.",
    "A-9. '격차는 곱이다' 는 철회됐다. 허용 문장 = '한 표본 안에서 격차는 k 다. (a,b) 는 널이다.'",
    "A-7. L-MERGE(D-4)는 기술통계다. 문턱도 주장도 없다. 2026-07-26 선행 = 혼잡 20셀 0승.",
    "A2-1. P-2 를 '하강을 제거한다' 로 쓰지 마라. 제거되는 것은 500/820 = 61% 의 결합 적합 "
    "단계이고, 제안 자신의 320스텝 폴리싱은 남는다.",
    "PREREG 0.1. 사전노출 신고: W2 가 사전등록 없이 VARPRO 를 이미 측정했다(혼잡 -47.3%, "
    "14/16). 본 배치는 확인이지 발견이 아니며, p 는 무조건부 I종 오류율이 아니다.",
    "PREREG 8. 'preregistered' 대신 'self-attested freeze + git-anchored' 라고 쓸 것.",
]


# ══════════════════════════════════════════════════════════════════════════════
def _need(pr):
    """The count threshold, or an explicit statement that no count can reach alpha at this
    n.  Printing `None` here is how a batch talks itself into a silent 'FAIL'."""
    t = pr.get('thresh')
    return str(t) if t is not None else 'n/a(n=%d 에서 어떤 k 도 alpha 에 못 닿는다)' % pr['n']


def _pct(x):
    return ('%+7.2f%%' % (100 * x)) if (x is not None and np.isfinite(x)) else '    n/a'


def _pr_line(tag, pr, extra='', two=False):
    lo, hi, cov = pr['median_rel_ci']
    ci = ('[%+.1f%%, %+.1f%%]@%.0f%%' % (100 * lo, 100 * hi, 100 * cov)
          if lo is not None else 'n/a')
    return ('%-10s %-12s %s %-12s  %2d/%-2d lower  rel median %s  CI %s  '
            'p=%.6g%s' % (tag, pr['treat'], 'vs' if two else ' <', pr['ctrl'], pr['wins'],
                          pr['n'], _pct(pr['median_rel']), ci, pr['p'], extra))


def self_test():
    """The analyser checks its own arithmetic against the numbers AMENDMENT_1 prints.
    'Verify the prescription too' -- an audit agent has shipped a wrong fix here before."""
    checks = [
        ('A-1 P-1 one-sided @0.025 = >=41', _fmt_region(region_one(64, 0.025), 64), '>=41'),
        ('A-1 P-1 one-sided @0.05  = >=40', _fmt_region(region_one(64, 0.05), 64), '>=40'),
        ('A-1 P-2 two-sided @0.025 = >=42 or <=22',
         _fmt_region(region_two(64, 0.025), 64), '>=42 or <=22'),
        ('A-1 P-2 two-sided @0.05  = >=41 or <=23',
         _fmt_region(region_two(64, 0.05), 64), '>=41 or <=23'),
        ('A-2 p_one(42|64) = 0.008429', '%.6f' % p_one(42, 64), '0.008429'),
        ('A-2 p_two(42|64) = 0.016858', '%.6f' % p_two(42, 64), '0.016858'),
        ('PREREG 1 p_one(41|64) = 0.01638', '%.5f' % p_one(41, 64), '0.01638'),
    ]
    bad = [(n, g, w) for n, g, w in checks if g != w]
    return checks, bad


def analyse(path, out_path=None, cross_check=True):
    rows_all = [json.loads(l) for l in open(path) if l.strip()]
    rows = [r for r in rows_all if 'error' not in r]
    errs = [r for r in rows_all if 'error' in r]
    episodes = sorted({r['episode'] for r in rows})
    by_arm = {}
    for r in rows:
        by_arm.setdefault(r['arm'], []).append(r)

    P = print
    P('=' * 100)
    P('W4_SOLVER_2026-08-19  --  analyze_w4.py  (frozen before any verdict row was read)')
    P('=' * 100)
    P('input            : %s' % path)
    P('rows             : %d usable, %d error rows' % (len(rows), len(errs)))
    P('episodes         : %d (expected %d)' % (len(episodes), N_EP_EXPECTED))
    P('arms             : %s' % {a: len(by_arm.get(a, [])) for a in ARM_ORDER
                                 if a in by_arm})
    P('estimator seeds  : %s (frozen, AMENDMENT_1 A-3)' % (list(SEEDS),))
    P('rule source      : PREREG.md + AMENDMENT_1.md + AMENDMENT_2.md; multiplicity = '
      'Holm-2 ONLY (A-1 deleted the fixed-threshold sentence)')
    if not IMPORT_OK:
        P('!! BLOCKER      : run_w4.py not importable (%s) -- constants are a LOCAL MIRROR '
          'and the DCC check is disabled.  Do not publish this run.' % IMPORT_ERR)
    if errs:
        P('error rows       : %s' % [(e.get('arm'), e.get('episode')) for e in errs[:6]])

    checks, bad_checks = self_test()
    P('self-test        : %d/%d frozen thresholds reproduced from exact binomial%s'
      % (len(checks) - len(bad_checks), len(checks),
         '' if not bad_checks else '   !! BLOCKER %s' % bad_checks))

    # ---- shell convention audit ------------------------------------------------
    conv = shell_convention_audit(rows)
    mixed = {a: sorted(c) for a, c in conv.items() if len(c) > 1}
    P('shell convention : %s%s' % ({a: sorted(c) for a, c in sorted(conv.items())},
                                   '' if not mixed else '   !! MIXED %s' % mixed))
    if mixed:
        P('  !! BLOCKER: an arm carries more than one shell prefix.  obs: and ship: differ '
          'by 1.8-4.1x on the same sample; S-3 is not interpretable until this is one '
          'convention (PREREG section 6).')

    # ---- gates -----------------------------------------------------------------
    P('\n' + '-' * 100)
    P('GATES  (halting = %s; G-4 record-only and G-9 supervision CANNOT kill PRIMARY -- '
      'AMENDMENT_1 A-8)' % ', '.join(HALTING))
    P('-' * 100)
    gates = eval_gates(rows)
    for name in ('G-1', 'G-2', 'G-2b', 'G-2c', 'G-3', 'G-4', 'G-5', 'G-6', 'G-6b',
                 'G-6c', 'G-7', 'G-8', 'G-9'):
        if name not in gates:
            continue
        gd = gates[name]
        P('%-5s %-18s %s%s' % (name, gd['status'], gd['detail'][:150],
                               '  [HALTING]' if gd['halting'] else ''))
    halt_fail = [n for n in HALTING
                 if n in gates and not str(gates[n]['status']).startswith('PASS')]
    dcc_fail = ('G-6' in gates and gates['G-6']['status'] != 'PASS')

    if cross_check and IMPORT_OK:
        P('\n--- DUAL VERIFICATION: run_w4.posthoc() (the runner\'s own implementation) ---')
        try:
            their = W4.posthoc(rows_all, os.path.basename(path))
            mine = not halt_fail and not dcc_fail
            P('cross-check      : runner_ok=%s  analyser_ok=%s  -> %s'
              % (their, mine, 'AGREE' if bool(their) == bool(mine)
                 else '!! DISAGREE -- BLOCKER, do not publish'))
        except Exception as e:
            P('cross-check      : posthoc raised %s (not a verdict input)' % repr(e)[:120])

    # ---- aggregation -----------------------------------------------------------
    tab = med3(rows, lambda r: r.get('avg_nRMSE'))
    P('\n' + '-' * 100)
    P('AGGREGATION (PREREG section 6): seed median FIRST, episode pairing SECOND; '
      'rel=(treat-ctrl)/ctrl; missing -> listwise; exact-float ties dropped, n reduced')
    P('-' * 100)
    for a in ARM_ORDER:
        cells = [(k, v) for k, v in tab.items() if k[0] == a]
        if not cells:
            continue
        ns = sorted({v['n_seeds'] for _, v in cells})
        P('  %-12s %2d episodes  seeds/episode=%s  med3 nRMSE median=%.6f'
          % (a, len(cells), ns, float(np.median([v['v'] for _, v in cells]))))

    # ---- IUT -------------------------------------------------------------------
    P('\n' + '-' * 100)
    P('IUT (PREREG section 1, the direction that is AGAINST us): L-SHIP < ZERO, '
      'crowded, one-sided >= %d/64' % round(IUT_MIN_FRAC * 64))
    P('-' * 100)
    iut = pair(tab, 'L-SHIP', 'ZERO', episodes)
    # The frozen rule is literally ">= 41/64".  At n=64 that is 41; at any other n the
    # proportional threshold is MY interpolation, declared in recon/D1_analyzer.md.
    iut_need = (41 if iut['n'] == 64
                else (max(1, int(math.ceil(IUT_MIN_FRAC * iut['n']))) if iut['n'] else None))
    iut_pass = bool(iut['n'] and iut['wins'] >= iut_need)
    iut['p'] = p_one(iut['wins'], iut['n'])
    iut['need'] = iut_need
    iut['pass'] = iut_pass
    P(_pr_line('IUT', iut, '   need >=%s%s -> %s'
               % (iut_need, '' if iut['n'] == 64 else ' (=ceil(41/64*n); n!=64, 보간은 '
                  '분석기 해석이다)', 'PASS' if iut_pass else 'FAIL')))

    # ---- PRIMARY ---------------------------------------------------------------
    P('\n' + '-' * 100)
    P('PRIMARY family, Holm-2 (family alpha=%.2f).  P-1 one-sided, P-2 two-sided.' % ALPHA)
    P('  P-1 (prescription): L-VP    < L-SHIP')
    P('  P-2 (diagnosis)   : L-ITER0 != L-SHIP  -- what does the 500-step JOINT FIT earn,')
    P('                      on atoms the learned proposal has ALREADY polished (A2-1)?')
    P('-' * 100)
    p1_raw = pair(tab, 'L-VP', 'L-SHIP', episodes)
    p2_raw = pair(tab, 'L-ITER0', 'L-SHIP', episodes)
    p1_p = p_one(p1_raw['wins'], p1_raw['n'])
    p2_p = p_two(p2_raw['wins'], p2_raw['n'])
    holm = holm2([('P-1', p1_p), ('P-2', p2_p)])

    for tag, raw, p, kind in (('P-1', p1_raw, p1_p, 'one'), ('P-2', p2_raw, p2_p, 'two')):
        h = holm[tag]
        reg = (region_one(raw['n'], h['alpha']) if kind == 'one'
               else region_two(raw['n'], h['alpha']))
        raw['p'] = p
        P(_pr_line(tag, raw,
                   '  Holm rank %d @alpha=%.4f  region %s  -> %s'
                   % (h['rank'], h['alpha'], _fmt_region(reg, raw['n']),
                      'REJECT' if h['reject'] else 'no rejection'), two=(kind == 'two')))
        P('%14s n=%d (dropped: missing %d, exact ties %d)%s'
          % ('', raw['n'], raw['n_missing'], raw['n_tie'],
             '' if not raw['partial_seeds']
             else '  !! %d episodes with <%d seeds on one arm'
                  % (len(raw['partial_seeds']), len(SEEDS))))
    P('  (AMENDMENT_1 A-1 reference at n=64: one-sided @0.025 %s / @0.05 %s ; two-sided '
      '@0.025 %s / @0.05 %s.  Regions above are recomputed at the ACTUAL n.)'
      % (_fmt_region(region_one(64, 0.025), 64), _fmt_region(region_one(64, 0.05), 64),
         _fmt_region(region_two(64, 0.025), 64), _fmt_region(region_two(64, 0.05), 64)))

    # ---- P-2 label (three outcomes, all informative; fixed before results) ------
    clamp = [(r.get('counters') or {}).get('clamp_hits_p1') for r in by_arm.get('L-SHIP', [])]
    clamp_hits = [c for c in clamp if _num(c) and c > 0]
    # AMENDMENT_1 A-5 reading rule, fixed before results:
    unit = ('결합 적합(joint fit) 단계' if not clamp_hits
            else 'Phase-1 루프 전체(결합 적합 + clamp_steepness 투영)')
    h2 = holm['P-2']
    reg2 = region_two(p2_raw['n'], h2['alpha'])
    lo_reg = [k for k in reg2 if k < p2_raw['n'] / 2.0]
    hi_reg = [k for k in reg2 if k > p2_raw['n'] / 2.0]
    if h2['reject'] and hi_reg and p2_raw['wins'] >= min(hi_reg):
        p2_label = 'OPTIMIZER-HARMS'
        p2_say = ('현재의 %s 가 해롭다 -- L-ITER0 이 유의하게 낮다. 빼는 것이 처방이고 '
                  '배포 개선이 공짜로 나온다.' % unit)
    elif h2['reject'] and lo_reg and p2_raw['wins'] <= max(lo_reg):
        p2_label = 'OPTIMIZER-EARNS'
        p2_say = ('%s 가 실제로 번다 -- 척추가 배포 경로에서 미지지. 우리 진단이 오라클 '
                  '안에서만 참이었다는 뜻이다.' % unit)
    else:
        p2_label = 'OPTIMIZER-INERT'
        p2_say = ('%s 가 아무것도 못 번다 ==> 벽은 하강이 아니라 제안된 k. '
                  '(A2-1: "모든 하강" 이 아니라 "결합 적합 단계" 로 좁혀진 주장이다.)' % unit)
    P('\nP-2 label        : %s' % p2_label)
    P('  meaning        : %s' % p2_say)
    P('  A-5 read rule  : L-SHIP rows with clamp_hits_p1>0 = %d/%d -> the removed unit is '
      '"%s"' % (len(clamp_hits), len(clamp), unit))
    P('  A2-1 step check: L-SHIP adam_steps_phase1 %s vs L-ITER0 %s (%d polish + %d joint '
      'fit); the arm removes %d/%d = 61%% of Phase-1, NOT all descent'
      % (sorted({r.get('adam_steps_phase1') for r in by_arm.get('L-SHIP', [])}),
         sorted({r.get('adam_steps_phase1') for r in by_arm.get('L-ITER0', [])}),
         POLISH_STEPS, SHIP_INIT_ITERS, SHIP_INIT_ITERS,
         POLISH_STEPS + SHIP_INIT_ITERS))

    mde = None
    if p2_label == 'OPTIMIZER-INERT':
        mde = mde_pi(reg2, p2_raw['n'])
        lo, hi, cov = p2_raw['median_rel_ci']
        P('  !! MDE (널이 배제하지 못하는 크기 -- 널 판정은 반드시 이것과 함께 읽는다):')
        P('     - 이 기각역(%s, n=%d, alpha=%.4f)이 검정력 0.80 으로 잡는 최소 승률 = %s'
          % (_fmt_region(reg2, p2_raw['n']), p2_raw['n'], h2['alpha'],
             ('%.3f (= %.1f/%d 짝)' % (mde, mde * p2_raw['n'], p2_raw['n']))
             if mde else '정의 불가 -- 이 n 에서는 기각역 자체가 비었다 = 검정력 0'))
        P('     - 짝 상대차 중앙값의 분포무관 CI = %s'
          % ('[%+.2f%%, %+.2f%%] (coverage %.0f%%)' % (100 * lo, 100 * hi, 100 * cov)
             if lo is not None else 'n/a'))
        P('     ==> "효과 없음" 이 아니라 "이보다 큰 효과는 없다" 로만 읽어라.')

    # ---- VERDICT ---------------------------------------------------------------
    P('\n' + '=' * 100)
    P('VERDICT  (total order, AMENDMENT_1 A-8): halting gates -> DCC -> IUT -> n<%d -> '
      'Holm-2' % N_MIN)
    P('=' * 100)
    n_eff = min(p1_raw['n'], p2_raw['n'])
    if halt_fail:
        verdict = 'HALTED (%s)' % ','.join(halt_fail)
        why = '중단 게이트 실패 -- 판정을 인쇄하지 않는다.'
    elif dcc_fail:
        verdict = 'NO-VERDICT (DCC)'
        why = 'G-6 필드 계약 위반 -- PREREG section 4 G-6 은 판정 인쇄 없이 종료다.'
    elif not iut_pass:
        verdict = 'FAIL'
        why = ('IUT 실패: L-SHIP < ZERO 가 %d/%d (필요 >=%s). 배포 팔이 자명한 하한을 '
               '이기지 못하면 나머지 비교는 의미가 없다.'
               % (iut['wins'], iut['n'], iut_need))
    elif n_eff < N_MIN:
        verdict = 'UNDECIDED (n)'
        why = ('유효 짝 n=%d < %d (PREREG section 5 D-4). P-1 n=%d, P-2 n=%d.'
               % (n_eff, N_MIN, p1_raw['n'], p2_raw['n']))
    else:
        h1 = holm['P-1']
        if h1['reject'] and p2_label == 'OPTIMIZER-HARMS':
            verdict = 'REMEDY-BY-SUBTRACTION'
            why = ('P-1 통과 AND P-2 = OPTIMIZER-HARMS (A-8 신설 어휘): 처방이 뺄셈 형태로 '
                   '성립한다 -- VARPRO 가 이기고, 그 이득의 일부는 %s 를 그냥 빼는 것으로도 '
                   '나온다.' % unit)
        elif h1['reject']:
            verdict = 'REMEDY-CONFIRMED / %s' % p2_label
            why = 'P-1 통과: L-VP 가 배포 팔보다 낮다. P-2 는 %s.' % p2_label
        else:
            verdict = 'NO-REMEDY / %s' % p2_label
            why = ('P-1 미통과. PREREG section 8 (A-8 개정) 정지 규칙: P-1 실패 AND '
                   'P-2 != OPTIMIZER-HARMS 이면 솔버 축을 여기서 종결한다 -- 현재 P-2 = %s '
                   '==> %s.' % (p2_label,
                                '종결' if p2_label != 'OPTIMIZER-HARMS' else '종결하지 않는다'))
    P('  %s' % verdict)
    P('  %s' % why)
    P('  scope: 합성 코퍼스 - 선형 GT - direction_deg=0 - 단일 체크포인트 - 혼잡 층(ncomp '
      '15/30) - init_method=learned 고정(배포 라우터 아님) - 추정기 시드 median-of-3.')

    # ---- SECONDARY -------------------------------------------------------------
    P('\n' + '-' * 100)
    P('SECONDARY (PREREG section 3; one-sided alpha=%.3f unless marked; NOT part of the '
      'Holm family)' % SEC_ALPHA)
    P('-' * 100)
    sec = {}
    s2 = one_sided(pair(tab, 'L-VP', 'L-ITER0', episodes), SEC_ALPHA)
    s2b = one_sided(pair(tab, 'L-VP', 'L-STEP400', episodes), SEC_ALPHA)
    s2c = two_sided(pair(tab, 'L-VP', 'L-VP-NW', episodes), SEC_ALPHA)
    sec['S-2'], sec['S-2b'], sec['S-2c'] = s2, s2b, s2c
    P(_pr_line('S-2', s2, '   need >=%s -> %s   [VARPRO 이득이 "결합 적합 제거만" 이 아님을 '
               '분리]' % (_need(s2), 'PASS' if s2['pass'] else 'null')))
    P(_pr_line('S-2b', s2b, '   need >=%s -> %s   [A2-4: Phase-1 스텝이 양 팔 동수 ==> '
               '★예산 동수, 매개화 단독 대조]' % (_need(s2b),
                                              'PASS' if s2b['pass'] else 'null')))
    P(_pr_line('S-2c', s2c, '   two-sided region %s -> %s   [가중 요인: L-VP(confidence) '
               'vs L-VP-NW(none). L-VP-NW 는 사전노출 프로브를 정확히 재현하는 설정이다]'
               % (_fmt_region(s2c['region'], s2c['n']),
                  'DIFFER' if s2c['pass'] else 'null'), two=True))
    trio = s2['pass'] and s2b['pass'] and (not s2c['pass'])
    P('  A-4 귀속 조건 : S-2 %s AND S-2b %s AND S-2c %s ==> "매개화 덕분" 이라고 쓸 수 %s'
      % ('PASS' if s2['pass'] else 'FAIL', 'PASS' if s2b['pass'] else 'FAIL',
         'null(=가중 무관)' if not s2c['pass'] else 'DIFFER(=가중이 요인)',
         '있다' if trio else '없다 -- 금지'))

    _r64 = region_one(64, SEC_ALPHA)
    P('\nS-3 shell decomposition of P-1 (L-VP < L-SHIP).  PREREG section 3: `obs:` prefix, '
      'one-sided, >=%d/64 at alpha=%.3f.' % (min(_r64), SEC_ALPHA))
    P('    (셸(obs:)과 배-거리 고리(ship:)는 다른 양이다 -- 같은 표본에서 1.8-4.1배 갈린다.)')
    s3 = {}
    for bare in BARE_KEYS:
        st = med3(rows, shell_getter('shell_rmse', bare))
        pr = one_sided(pair(st, 'L-VP', 'L-SHIP', episodes), SEC_ALPHA)
        s3['obs:' + bare] = pr
        P('   obs:%-9s %2d/%-2d  rel median %s  p=%.6g  need >=%s -> %s'
          % (bare, pr['wins'], pr['n'], _pct(pr['median_rel']), pr['p'], _need(pr),
             'PASS' if pr['pass'] else '-'))
    if s3 and all(v['wins'] == 0 for v in s3.values()) and any(v['n'] for v in s3.values()):
        P('   !! 9셸 전부 0 승 -- PREREG section 6 의 셸 키 함정을 의심하라 (obs: vs bare vs '
          'ship:).  convention audit = %s' % {a: sorted(c) for a, c in conv.items()})
    if s3 and all(v['n'] == 0 for v in s3.values()):
        P('   !! 9셸 전부 n=0 -- shell_rmse 키가 어느 규약과도 맞지 않는다. BLOCKER.')

    # ---- D descriptives ---------------------------------------------------------
    P('\n' + '-' * 100)
    P('D  --  DESCRIPTIVES (PREREG section 3 D-family).  NOT hypotheses: no thresholds, '
      'no p-values, no claims.')
    P('-' * 100)
    desc = {}

    # D-1 window residual vs field
    P('D-1 window residual vs field (does a solver gain transfer to the field?)')
    d1 = {}
    for a in ARM_ORDER:
        rs = by_arm.get(a, [])
        w = [r.get('window_resid_rms') for r in rs if _num(r.get('window_resid_rms'))]
        f = [r.get('avg_nRMSE') for r in rs if _num(r.get('avg_nRMSE'))]
        if not w:
            continue
        d1[a] = dict(window_median=float(np.median(w)), field_median=float(np.median(f)),
                     ratio=float(np.median(f) / np.median(w)) if np.median(w) else None)
        P('    %-12s window %.6f   field %.6f   field/window %s'
          % (a, d1[a]['window_median'], d1[a]['field_median'],
             ('%.1fx' % d1[a]['ratio']) if d1[a]['ratio'] else 'n/a'))
    wt = med3(rows, lambda r: r.get('window_resid_rms'))
    for treat in ('L-VP', 'L-ITER0'):
        dw, df = [], []
        for ep in episodes:
            a_w, b_w = wt.get((treat, ep)), wt.get(('L-SHIP', ep))
            a_f, b_f = tab.get((treat, ep)), tab.get(('L-SHIP', ep))
            if not (a_w and b_w and a_f and b_f) or b_w['v'] == 0 or b_f['v'] == 0:
                continue
            dw.append((a_w['v'] - b_w['v']) / b_w['v'])
            df.append((a_f['v'] - b_f['v']) / b_f['v'])
        if len(dw) >= 4:
            r_p = float(np.corrcoef(dw, df)[0, 1])
            r_s = _spearman(dw, df)
            d1['%s_transfer' % treat] = dict(n=len(dw), pearson=r_p, spearman=r_s,
                                             median_d_window=float(np.median(dw)),
                                             median_d_field=float(np.median(df)))
            P('    %s vs L-SHIP: n=%d  d(window) median %+.2f%%  d(field) median %+.2f%%  '
              'pearson %+.3f  spearman %+.3f'
              % (treat, len(dw), 100 * np.median(dw), 100 * np.median(df), r_p, r_s))
    desc['D-1'] = d1

    # D-2 wall clock
    P('D-2 wall clock per arm (median / IQR seconds)')
    d2 = {}
    for a in ARM_ORDER:
        w = [r.get('wall_s') for r in by_arm.get(a, []) if _num(r.get('wall_s'))]
        if not w:
            continue
        q1, q2, q3 = np.percentile(w, [25, 50, 75])
        d2[a] = dict(n=len(w), median=float(q2), iqr=[float(q1), float(q3)])
        P('    %-12s n=%3d  median %7.2f s  IQR [%.2f, %.2f]' % (a, len(w), q2, q1, q3))
    desc['D-2'] = d2

    # D-3 L-MERGE atom reduction
    P('D-3 L-MERGE atom reduction')
    mrows = by_arm.get('L-MERGE', [])
    if mrows:
        mr = [r.get('merge_removed') for r in mrows if _num(r.get('merge_removed'))]
        na = [r.get('n_atoms_final') for r in mrows if _num(r.get('n_atoms_final'))]
        sa = [r.get('n_atoms_final') for r in by_arm.get('L-SHIP', [])
              if _num(r.get('n_atoms_final'))]
        noop = sum(1 for v in mr if v == 0)
        desc['D-3'] = dict(n=len(mrows), merge_removed_median=float(np.median(mr)) if mr
                           else None, noop_rows=noop,
                           n_atoms_final_median=float(np.median(na)) if na else None,
                           ship_n_atoms_final_median=float(np.median(sa)) if sa else None)
        P('    merge_removed median %s   bit-identical no-op rows %d/%d   '
          'n_atoms_final L-MERGE %s vs L-SHIP %s'
          % (desc['D-3']['merge_removed_median'], noop, len(mrows),
             desc['D-3']['n_atoms_final_median'],
             desc['D-3']['ship_n_atoms_final_median']))
    else:
        desc['D-3'] = None
        P('    no L-MERGE rows')

    # D-4 L-MERGE contrast -- DEMOTED from hypothesis by AMENDMENT_1 A-7
    P('D-4 L-MERGE vs L-SHIP  --  DESCRIPTIVE ONLY (A-7 demoted S-1; no threshold, no p, '
      'no claim)')
    d4 = pair(tab, 'L-MERGE', 'L-SHIP', episodes)
    desc['D-4'] = {k: v for k, v in d4.items() if k != 'rows'}
    P('    %d/%d lower, %d exact ties dropped (n reduced, PREREG section 6), %d missing; '
      'rel median %s' % (d4['wins'], d4['n'], d4['n_tie'], d4['n_missing'],
                         _pct(d4['median_rel'])))
    P('    prior (recorded before results, A-7): 2026-07-26 measured crowded 20 cells, '
      '0 wins; merge_frac=0.5 radius 1.10-1.18 Rayleigh > true median separation; '
      'dedup_dk=0.03 makes a large share of runs mathematically no-op.')
    P('    ==> "공짜 티켓" 은 틀린 서술이었다. 이 줄에서 어떤 판정도 읽지 마라.')

    # ORA-K-CVX upper bound -- descriptive, with the section 7-2 ban attached
    if 'ORA-K-CVX' in by_arm:
        cvx = pair(tab, 'ORA-K-CVX', 'L-SHIP', episodes)
        ratios = [r['ctrl'] / r['treat'] for r in cvx['rows'] if r['treat']]
        desc['D-CVX'] = dict(n=cvx['n'], wins=cvx['wins'],
                             median_ratio=float(np.median(ratios)) if ratios else None,
                             median_rel=cvx['median_rel'])
        P('D-* ORA-K-CVX upper bound: %d/%d lower than L-SHIP, median ratio '
          'L-SHIP/CVX = %s   [PREREG 7-2: 이 비를 순수 솔버 이득으로 인용 금지 -- 6요인 묶음]'
          % (cvx['wins'], cvx['n'],
             ('%.1fx' % desc['D-CVX']['median_ratio'])
             if desc['D-CVX']['median_ratio'] else 'n/a'))

    # free fields (AMENDMENT_1 A-11): summarise whatever the rows actually carry
    FREE = ('RMSE_near', 'rmse_near', 'rho', 'VR', 'decorr_mean', 'shrink_mean',
            'init_nrmse', 'init_time_s', 'obs_resid_rms_mean', 'elapsed_ms_mean',
            'n_atoms_final', 'window_n_obs')
    P('D-free (A-11 무료 필드; 전부 기술통계, 가설로 승격하지 않는다)')
    dfree = {}
    for fld in FREE:
        per = {}
        for a in ARM_ORDER:
            v = [r.get(fld) for r in by_arm.get(a, []) if _num(r.get(fld))]
            if v:
                per[a] = float(np.median(v))
        if per:
            dfree[fld] = per
            P('    %-20s %s' % (fld, {k: round(v, 6) for k, v in per.items()}))
    absent = [f for f in ('RMSE_near', 'rmse_near') if f not in dfree]
    if len(absent) == 2:
        P('    RMSE_near : 행에 없다 (A-11 이 회수를 지시했으나 이 파일들에는 부재). '
          '부재는 음성 증거가 아니다 -- 근거리 주장을 하지 마라.')
    # vp diagnostics: window residual before/after (A-11)
    vpd = [r.get('vp_diag') for r in by_arm.get('L-VP', []) if isinstance(r.get('vp_diag'),
                                                                        dict)]
    if vpd:
        b = [d.get('vp_win_rms_before') for d in vpd if _num(d.get('vp_win_rms_before'))]
        a_ = [d.get('vp_win_rms_after') for d in vpd if _num(d.get('vp_win_rms_after'))]
        ks = [d.get('vp_k_shift') for d in vpd if _num(d.get('vp_k_shift'))]
        dfree['vp_diag'] = dict(win_rms_before=float(np.median(b)) if b else None,
                                win_rms_after=float(np.median(a_)) if a_ else None,
                                k_shift_median=float(np.median(ks)) if ks else None)
        P('    %-20s before %.6f -> after %.6f   median |dk| %.4f'
          % ('vp window resid', np.median(b), np.median(a_), np.median(ks)))
    desc['D-free'] = dfree

    # ---- prohibitions (ALWAYS) --------------------------------------------------
    P('\n' + '=' * 100)
    P('금지 목록 -- PREREG section 7 + AMENDMENT_1 A-4 (판정 아래에 항상 인쇄된다)')
    P('=' * 100)
    for line in PROHIBITIONS:
        P('  ' + line)

    # ---- json -------------------------------------------------------------------
    def strip(pr):
        return {k: v for k, v in pr.items() if k != 'rows'}

    ver = dict(
        batch='W4_SOLVER_2026-08-19', analyzer='analyze_w4.py',
        analyzer_frozen_before_rows=True, input=os.path.abspath(path),
        n_rows=len(rows), n_error_rows=len(errs), n_episodes=len(episodes),
        arms={a: len(v) for a, v in sorted(by_arm.items())},
        run_w4_import_ok=IMPORT_OK, run_w4_import_error=IMPORT_ERR,
        shell_convention={a: sorted(c) for a, c in sorted(conv.items())},
        gates={k: v for k, v in gates.items()},
        halting_gates=list(HALTING), halting_failures=halt_fail, dcc_fail=bool(dcc_fail),
        iut=strip(iut),
        primary=dict(
            rule='Holm-2 (AMENDMENT_1 A-1; the fixed-threshold sentence is deleted)',
            alpha=ALPHA, holm=holm,
            **{'P-1': strip(p1_raw), 'P-2': strip(p2_raw)}),
        p2_label=p2_label, p2_unit=unit,
        p2_clamp_hits_rows=len(clamp_hits), p2_clamp_rows_total=len(clamp),
        p2_mde_pi=mde,
        verdict=verdict, verdict_reason=why,
        n_effective=n_eff, n_min=N_MIN,
        secondary=dict(**{k: strip(v) for k, v in sec.items()},
                       attribution_allowed=bool(trio),
                       **{'S-3': {k: strip(v) for k, v in s3.items()}}),
        descriptives=desc,
        prohibitions=PROHIBITIONS,
        scope=('synthetic corpus; linear GT; direction_deg=0; single checkpoint; crowded '
               'stratum only (ncomp 15/30); init_method=learned pinned (NOT the shipped '
               'router); estimator-seed median-of-3'),
    )
    out_path = out_path or _default_out(path)
    with open(out_path, 'w') as fh:
        json.dump(ver, fh, indent=1, default=str)
    P('\nwrote %s' % out_path)
    return ver


def _spearman(x, y):
    def rank(v):
        v = np.asarray(v, float)
        order = np.argsort(v, kind='mergesort')
        r = np.empty(len(v), float)
        r[order] = np.arange(1, len(v) + 1)
        # average ranks for exact ties
        for val in np.unique(v):
            m = (v == val)
            if m.sum() > 1:
                r[m] = r[m].mean()
        return r
    rx, ry = rank(x), rank(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def _default_out(path):
    base = os.path.basename(path)
    if base == 'results.jsonl':
        return os.path.join(HERE, 'verdict_w4.json')
    stem = base.replace('.jsonl', '')
    return os.path.join(HERE, 'verdict_w4_%s.json' % stem)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('-')]
    path = args[0] if args else os.path.join(HERE, 'results.jsonl')
    if not os.path.isabs(path):
        cand = os.path.join(HERE, path)
        path = cand if os.path.exists(cand) else path
    if not os.path.exists(path):
        print('no such rows file: %s' % path)
        sys.exit(2)
    analyse(path, cross_check=('--no-posthoc' not in sys.argv))


if __name__ == '__main__':
    main()
