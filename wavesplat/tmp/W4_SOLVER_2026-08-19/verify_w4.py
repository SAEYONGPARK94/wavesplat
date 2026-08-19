#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_w4.py — D2 INDEPENDENT VERIFIER for batch W4_SOLVER_2026-08-19.

This is the SECOND, deliberately independent implementation of the frozen
analysis rules.  It was written WITHOUT reading `analyze_w4.py` (D1) or
`recon/D1_analyzer.md`.  Its only inputs are:

    PREREG.md  +  AMENDMENT_1.md  +  AMENDMENT_2.md   (the frozen rules)
    corpus_w4.json                                    (the frozen sample)
    <results>.jsonl                                   (the rows)

Deliberate implementation differences from a "natural" first implementation
(the point is that two bugs must coincide to survive):

  * aggregation container  = dense numpy cubes indexed by a SORTED episode
    list and a SORTED estimator-seed list (no nested dicts, no pandas)
  * median-of-3            = hand-written rank-1-of-3 selector, cross-checked
    against numpy.median on every cell
  * p-values               = exact rational binomial tail via fractions.Fraction
    + math.comb, cross-checked against scipy.stats.binomtest when available
  * critical values        = DERIVED from the exact distribution, then asserted
    equal to the values frozen in PREREG/AMENDMENT_1 (they are not hardcoded
    into the decision path; a mismatch is printed loudly)
  * shell aggregation      = a separate 3-D cube (episode x seed x shell),
    built in one pass, never reusing the scalar path

USAGE
    python3 verify_w4.py                             # -> results.jsonl
    python3 verify_w4.py --results smoke_results.jsonl --out verify_smoke.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from fractions import Fraction

import numpy as np

# --------------------------------------------------------------------------
# frozen constants, transcribed from the rule documents (source noted inline)
# --------------------------------------------------------------------------

BATCH_DIR = os.path.dirname(os.path.abspath(__file__))

EST_SEEDS = (1595, 1596, 1597)          # AMENDMENT_1 A-3
N_REGISTERED = 64                        # AMENDMENT_1 A-3
STRATUM = "crowded"                      # PREREG §1
NCOMP_OK = (15, 30)                      # AMENDMENT_1 A-3
MIN_N = 56                               # PREREG §5 D-4  (n < 56 -> UNDECIDED)

ALPHA_FAMILY = 0.05                      # PREREG §1  (Holm-2 family alpha)
ALPHA_SECONDARY = 0.025                  # PREREG §3   (single, one-sided)

# 9 shells, `obs:` prefix ONLY (PREREG §6; `ship:` is a different quantity)
SHELL_KEYS = ("obs:0_5", "obs:5_10", "obs:10_20", "obs:20_30", "obs:30_50",
              "obs:50_75", "obs:75_100", "obs:100_150", "obs:150_inf")

# PREREG §1 / AMENDMENT_1 A-1 frozen rejection regions at n = 64.
# We DERIVE these below and assert agreement; they are cross-check only.
FROZEN_CRIT = {
    "one_0.025": 41, "one_0.05": 40,
    "two_0.025_hi": 42, "two_0.025_lo": 22,
    "two_0.05_hi": 41, "two_0.05_lo": 23,
}

# AMENDMENT_2 A2-1 / A2-2 / A2-4 : exact phase-1 Adam step budgets
EXPECT_P1_STEPS = {
    "L-SHIP": 820, "L-SHIP-DUP": 820, "L-ITER0": 320,
    "L-VP": 1220, "L-VP-NW": 1220, "L-STEP400": 1220,
}
EXPECT_VP_STEPS = {"L-VP": 400, "L-VP-NW": 400}          # AMENDMENT_2 A2-3

EST_ARMS = ("L-SHIP", "L-ITER0", "L-VP", "L-VP-NW", "L-STEP400", "L-MERGE")
DET_ARMS = ("ZERO", "ORA-K-CVX")
DUP_ARM = "L-SHIP-DUP"


# --------------------------------------------------------------------------
# exact binomial machinery (path A: rational arithmetic)
# --------------------------------------------------------------------------

def _tail_ge(n: int, k: int) -> Fraction:
    """Exact P(X >= k) for X ~ Binom(n, 1/2)."""
    if k <= 0:
        return Fraction(1)
    if k > n:
        return Fraction(0)
    num = sum(math.comb(n, j) for j in range(k, n + 1))
    return Fraction(num, 1 << n)


def _tail_le(n: int, k: int) -> Fraction:
    """Exact P(X <= k) for X ~ Binom(n, 1/2)."""
    if k < 0:
        return Fraction(0)
    if k >= n:
        return Fraction(1)
    num = sum(math.comb(n, j) for j in range(0, k + 1))
    return Fraction(num, 1 << n)


def p_one_sided(n: int, w: int) -> float:
    """One-sided sign-test p for 'treatment wins w of n' (H1: pi > 1/2)."""
    return float(_tail_ge(n, w))


def p_two_sided(n: int, w: int) -> float:
    """Symmetric two-sided sign-test p (pi = 1/2 makes 2*min(tails) exact)."""
    if n <= 0:
        return float("nan")
    return float(min(Fraction(1), 2 * min(_tail_ge(n, w), _tail_le(n, w))))


def crit_one(n: int, alpha: float):
    """Smallest w with P(X >= w) <= alpha, or None."""
    for w in range(0, n + 1):
        if float(_tail_ge(n, w)) <= alpha:
            return w
    return None


def crit_two(n: int, alpha: float):
    """(hi, lo): reject if w >= hi or w <= lo, each tail spending alpha/2."""
    half = alpha / 2.0
    hi = None
    for w in range(0, n + 1):
        if float(_tail_ge(n, w)) <= half:
            hi = w
            break
    lo = None
    for w in range(n, -1, -1):
        if float(_tail_le(n, w)) <= half:
            lo = w
            break
    return hi, lo


# --------------------------------------------------------------------------
# path B: scipy cross-check (optional; absence is reported, never fatal)
# --------------------------------------------------------------------------

try:                                                        # pragma: no cover
    from scipy.stats import binomtest as _scipy_binomtest
    _HAVE_SCIPY = True
except Exception:                                           # pragma: no cover
    _scipy_binomtest = None
    _HAVE_SCIPY = False


def scipy_cross_check(n: int, w: int):
    """Return dict of scipy p-values and their max abs deviation from path A."""
    out = {"available": _HAVE_SCIPY}
    if not _HAVE_SCIPY or n <= 0:
        return out
    try:
        g = _scipy_binomtest(w, n, 0.5, alternative="greater").pvalue
        t = _scipy_binomtest(w, n, 0.5, alternative="two-sided").pvalue
    except Exception as exc:                                # pragma: no cover
        out["error"] = repr(exc)
        return out
    out["p_one_scipy"] = float(g)
    out["p_two_scipy"] = float(t)
    out["max_abs_dev"] = float(max(abs(g - p_one_sided(n, w)),
                                   abs(t - p_two_sided(n, w))))
    out["agree_1e_12"] = bool(out["max_abs_dev"] <= 1e-12)
    return out


# --------------------------------------------------------------------------
# power / MDE
# --------------------------------------------------------------------------

def _power(n: int, region, pi: float) -> float:
    """P(W in region) for W ~ Binom(n, pi); `region` is a predicate on ints."""
    tot = 0.0
    for k in range(n + 1):
        if region(k):
            tot += math.comb(n, k) * (pi ** k) * ((1.0 - pi) ** (n - k))
    return tot


def mde_win_rate(n: int, region, target_power: float = 0.80):
    """Smallest pi > 0.5 whose power over `region` reaches target_power."""
    if n <= 0:
        return None
    lo, hi = 0.5, 1.0
    if _power(n, region, hi) < target_power:
        return None
    for _ in range(80):                       # bisection on a monotone curve
        mid = 0.5 * (lo + hi)
        if _power(n, region, mid) >= target_power:
            hi = mid
        else:
            lo = mid
    return hi


def sign_ci_median(x: np.ndarray, alpha: float = 0.05):
    """Distribution-free CI for the median (order statistics / sign test)."""
    v = np.sort(np.asarray(x, dtype=float))
    n = v.size
    if n == 0:
        return (None, None)
    k = None
    for j in range(0, n // 2 + 1):
        if 2.0 * float(_tail_le(n, j)) <= alpha:
            k = j
        else:
            break
    if k is None:
        return (float(v[0]), float(v[-1]))
    lo_i, hi_i = k, n - 1 - k
    if lo_i > hi_i:
        return (float(v[0]), float(v[-1]))
    return (float(v[lo_i]), float(v[hi_i]))


# --------------------------------------------------------------------------
# median-of-3, hand written + numpy cross-check
# --------------------------------------------------------------------------

def _mid_of_three(a: float, b: float, c: float) -> float:
    """Rank-1-of-3 selector written from comparisons only (no sort call)."""
    if (a <= b <= c) or (c <= b <= a):
        return b
    if (b <= a <= c) or (c <= a <= b):
        return a
    return c


def med3_row(vals: np.ndarray, strict: bool = True):
    """
    Collapse the estimator-seed axis for ONE episode.

    strict=True (PRIMARY, PREREG §6 'listwise'): all three registered seeds
    must be present, otherwise the episode is dropped (NaN).
    strict=False (sensitivity): median over whatever seeds exist.

    Returns (value, n_seeds_present, xcheck_ok).
    """
    v = np.asarray(vals, dtype=float)
    ok = v[np.isfinite(v)]
    m = int(ok.size)
    if m == 0:
        return (float("nan"), 0, True)
    if strict and m != len(EST_SEEDS):
        return (float("nan"), m, True)
    if m == 3:
        hand = _mid_of_three(float(ok[0]), float(ok[1]), float(ok[2]))
    elif m == 1:
        hand = float(ok[0])
    else:
        hand = float(np.median(ok))          # even count: numpy definition
    npv = float(np.median(ok))
    xok = bool(abs(hand - npv) <= 0.0 or np.isclose(hand, npv, rtol=0, atol=0))
    return (hand, m, xok)


# --------------------------------------------------------------------------
# cube construction (dense numpy, sorted axes)
# --------------------------------------------------------------------------

class Cube:
    """Dense [arm][episode, seed] and [arm][episode, seed, shell] containers."""

    def __init__(self, rows):
        self.rows = rows
        self.episodes = sorted({r["episode"] for r in rows})
        self.ep_ix = {e: i for i, e in enumerate(self.episodes)}
        self.seed_ix = {s: i for i, s in enumerate(EST_SEEDS)}
        self.arms = sorted({r["arm"] for r in rows})
        E, S, H = len(self.episodes), len(EST_SEEDS), len(SHELL_KEYS)

        self.scalar = {a: np.full((E, S), np.nan) for a in self.arms}
        self.shell = {a: np.full((E, S, H), np.nan) for a in self.arms}
        self.wall = {a: np.full((E, S), np.nan) for a in self.arms}
        self.wresid = {a: np.full((E, S), np.nan) for a in self.arms}
        self.natoms = {a: np.full((E, S), np.nan) for a in self.arms}
        self.mremoved = {a: np.full((E, S), np.nan) for a in self.arms}
        self.dupes = []          # (arm, episode, seed) collisions

        for r in rows:
            a = r["arm"]
            i = self.ep_ix[r["episode"]]
            es = r.get("est_seed")
            j = self.seed_ix.get(es, 0) if es is not None else 0
            if np.isfinite(self.scalar[a][i, j]):
                self.dupes.append((a, r["episode"], es))
            self.scalar[a][i, j] = _f(r.get("avg_nRMSE"))
            self.wall[a][i, j] = _f(r.get("wall_s"))
            self.wresid[a][i, j] = _f(r.get("window_resid_rms"))
            self.natoms[a][i, j] = _f(r.get("n_atoms_final"))
            self.mremoved[a][i, j] = _f(r.get("merge_removed"))
            sh = r.get("shell_rmse") or {}
            for h, key in enumerate(SHELL_KEYS):
                self.shell[a][i, j, h] = _f(sh.get(key))

    # ---- seed collapse ---------------------------------------------------

    def med3_scalar(self, arm: str, strict: bool = True):
        E = len(self.episodes)
        out = np.full(E, np.nan)
        nseeds = np.zeros(E, dtype=int)
        xbad = 0
        cube = self.scalar.get(arm)
        if cube is None:
            return out, nseeds, xbad
        det = arm in DET_ARMS
        for i in range(E):
            if det:
                v = cube[i, 0]
                out[i] = v
                nseeds[i] = 1 if np.isfinite(v) else 0
            else:
                val, m, ok = med3_row(cube[i], strict=strict)
                out[i], nseeds[i] = val, m
                if not ok:
                    xbad += 1
        return out, nseeds, xbad

    def med3_shell(self, arm: str, h: int, strict: bool = True):
        E = len(self.episodes)
        out = np.full(E, np.nan)
        cube = self.shell.get(arm)
        if cube is None:
            return out
        det = arm in DET_ARMS
        for i in range(E):
            if det:
                out[i] = cube[i, 0, h]
            else:
                out[i], _, _ = med3_row(cube[i, :, h], strict=strict)
        return out


def _f(x):
    try:
        if x is None:
            return float("nan")
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


# --------------------------------------------------------------------------
# paired sign test on episode-level med3 values
# --------------------------------------------------------------------------

def paired(treat: np.ndarray, control: np.ndarray, label: str):
    """
    PREREG §6: pair unit = episode; relative diff = (treat - control)/control;
    listwise deletion of missing; EXACT float ties dropped and n reduced.
    W counts episodes with treat < control.
    """
    t = np.asarray(treat, float)
    c = np.asarray(control, float)
    both = np.isfinite(t) & np.isfinite(c)
    n_missing = int((~both).sum())
    tt, cc = t[both], c[both]
    tie_mask = (tt == cc)                    # exact float equality only
    n_ties = int(tie_mask.sum())
    keep = ~tie_mask
    tt, cc = tt[keep], cc[keep]
    n = int(tt.size)
    W = int((tt < cc).sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(cc != 0, (tt - cc) / cc, np.nan)
    rel_ok = rel[np.isfinite(rel)]
    med_rel = float(np.median(rel_ok)) if rel_ok.size else float("nan")
    ci = sign_ci_median(rel_ok) if rel_ok.size else (None, None)
    out = {
        "label": label,
        "n": n,
        "W": W,
        "L": int(n - W),
        "n_ties_dropped": n_ties,
        "n_missing_pairs": n_missing,
        "median_rel_diff": med_rel,
        "median_rel_diff_ci95": list(ci),
        "mean_rel_diff": float(np.mean(rel_ok)) if rel_ok.size else float("nan"),
        "p_one_sided_treat_lower": p_one_sided(n, W) if n else float("nan"),
        "p_two_sided": p_two_sided(n, W) if n else float("nan"),
        "scipy_xcheck": scipy_cross_check(n, W),
    }
    return out


# --------------------------------------------------------------------------
# gates recomputed from the raw rows (never trusting `gates_pass`)
# --------------------------------------------------------------------------

def recompute_gates(rows, cube: Cube, corpus):
    g = {}
    est_rows = [r for r in rows if r["arm"] in EST_ARMS or r["arm"] == DUP_ARM]

    # -- G-1 routing fixed to `learned`, one learned-init call per run -------
    bad_route = [r["job_id"] for r in est_rows if r.get("route") != "learned"]
    bad_calls = [r["job_id"] for r in est_rows
                 if (r.get("counters") or {}).get("learned_init_calls") != 1]
    g["G-1"] = {"kind": "abort", "pass": not bad_route and not bad_calls,
                "n_checked": len(est_rows),
                "bad_route": bad_route[:5], "bad_calls": bad_calls[:5]}

    # -- G-2 exact phase-1 Adam budgets (AMENDMENT_2 A2-2) ------------------
    p1 = {}
    for r in est_rows:
        p1.setdefault(r["arm"], set()).add(r.get("adam_steps_phase1"))
    g2_bad = {a: sorted(v, key=lambda z: (z is None, z))
              for a, v in p1.items()
              if a in EXPECT_P1_STEPS and v != {EXPECT_P1_STEPS[a]}}
    g["G-2"] = {"kind": "abort", "pass": not g2_bad,
                "observed": {a: sorted(v, key=lambda z: (z is None, z))
                             for a, v in sorted(p1.items())},
                "expected": EXPECT_P1_STEPS, "violations": g2_bad}

    # -- G-2b unattributed Adam steps ---------------------------------------
    other = [r["job_id"] for r in est_rows if (r.get("adam_steps_other") or 0) > 0]
    g["G-2b"] = {"kind": "record", "pass": not other, "n_rows_with_other": len(other)}

    # -- G-2c budget parity L-VP vs L-STEP400 (AMENDMENT_2 A2-4) ------------
    a_set, b_set = p1.get("L-VP", set()), p1.get("L-STEP400", set())
    g["G-2c"] = {"kind": "abort", "pass": bool(a_set) and a_set == b_set,
                 "L-VP": sorted(a_set, key=lambda z: (z is None, z)),
                 "L-STEP400": sorted(b_set, key=lambda z: (z is None, z))}

    # -- G-3 vp_steps 400 on the two VP arms, 0 everywhere else -------------
    vps = {}
    for r in rows:
        vps.setdefault(r["arm"], set()).add(r.get("vp_steps"))
    g3_bad = {}
    for a, v in vps.items():
        want = {EXPECT_VP_STEPS.get(a, 0)}
        if v - {None} and (v - {None}) != want:
            g3_bad[a] = sorted(v - {None})
    g["G-3"] = {"kind": "abort", "pass": not g3_bad,
                "observed": {a: sorted(v - {None}) for a, v in sorted(vps.items())},
                "violations": g3_bad}

    # -- G-4 merge actually happened (RECORD only, AMENDMENT_1 A-7) ---------
    mr = [r.get("merge_removed") for r in rows
          if r["arm"] == "L-MERGE" and r.get("merge_removed") is not None]
    other_mr = [r["job_id"] for r in rows
                if r["arm"] != "L-MERGE" and (r.get("merge_removed") or 0) > 0]
    g["G-4"] = {"kind": "record",
                "median_merge_removed": float(np.median(mr)) if mr else None,
                "n_rows": len(mr), "leak_rows_other_arms": len(other_mr),
                "pass": bool(mr) and float(np.median(mr)) > 0}

    # -- G-5 determinism probe: L-SHIP-DUP must be bit-identical ------------
    ship = {(r["episode"], r.get("est_seed")): r for r in rows if r["arm"] == "L-SHIP"}
    deltas, unmatched = [], 0
    for r in rows:
        if r["arm"] != DUP_ARM:
            continue
        s = ship.get((r["episode"], r.get("est_seed")))
        if s is None:
            unmatched += 1
            continue
        deltas.append(abs(_f(r.get("avg_nRMSE")) - _f(s.get("avg_nRMSE"))))
    mx = float(max(deltas)) if deltas else None
    g["G-5"] = {"kind": "abort",
                "pass": bool(deltas) and mx == 0.0 and unmatched == 0,
                "n_pairs": len(deltas), "max_abs_delta": mx,
                "unmatched_dup_rows": unmatched}

    # -- G-6 data contract --------------------------------------------------
    dcc = [r["job_id"] for r in rows if r.get("dcc_missing")]
    g["G-6"] = {"kind": "terminate", "pass": not dcc, "n_rows_missing": len(dcc),
                "examples": dcc[:5]}

    # -- G-6b frame parity across arms, per episode -------------------------
    fr, zf = {}, {}
    for r in rows:
        fr.setdefault(r["episode"], set()).add(
            (r.get("n_frames_scored"), r.get("n_match"), r.get("n_all")))
        zf.setdefault(r["episode"], set()).add(r.get("zero_nRMSE"))
    bad_fr = {e: len(v) for e, v in fr.items() if len(v) > 1}
    bad_zf = {e: len(v) for e, v in zf.items() if len(v) > 1}
    g["G-6b"] = {"kind": "record", "pass": not bad_fr, "episodes_mismatched": bad_fr}
    g["G-6c"] = {"kind": "record", "pass": not bad_zf, "episodes_mismatched": bad_zf}

    # -- G-7 phase-convention identity: parsed from the pre-run gates.log ----
    g["G-7"] = _parse_gates_log("G-7 phase identity")

    # -- G-8 fingerprint: single code_md5 / single git_head ------------------
    md5s = sorted({r.get("code_md5") for r in rows})
    heads = sorted({r.get("git_head") for r in rows})
    g["G-8"] = {"kind": "abort", "pass": len(md5s) == 1 and len(heads) == 1,
                "code_md5": md5s, "git_head": heads}

    # -- sample integrity vs the frozen corpus ------------------------------
    want = {e["key"] for e in corpus.get("episodes", [])} if corpus else set()
    seen = set(cube.episodes)
    g["SAMPLE"] = {
        "kind": "record",
        "n_episodes_seen": len(seen),
        "n_episodes_frozen": len(want),
        "not_in_frozen_corpus": sorted(seen - want)[:8],
        "frozen_but_absent": len(want - seen),
        "strata_seen": sorted({r.get("stratum") for r in rows}),
        "ncomp_seen": sorted({r.get("ncomp") for r in rows if r.get("ncomp") is not None}),
        "est_seeds_seen": sorted({r.get("est_seed") for r in rows
                                  if r.get("est_seed") is not None}),
        "duplicate_cells": cube.dupes[:8],
        "pass": (not cube.dupes
                 and (not want or not (seen - want))
                 and set(r.get("stratum") for r in rows) == {STRATUM}),
    }

    # -- A-5 clamp readout rule (decides how P-2 is worded) ------------------
    ch = [(r.get("counters") or {}).get("clamp_hits_p1")
          for r in rows if r["arm"] == "L-SHIP"]
    ch = [c for c in ch if c is not None]
    any_hit = any(c > 0 for c in ch)
    g["A5_CLAMP_READOUT"] = {
        "kind": "record", "n_rows": len(ch),
        "max_clamp_hits_p1": max(ch) if ch else None,
        "reading": ("Phase-1 loop as a whole (descent + clamp projection)"
                    if any_hit else "the joint-fit descent"),
        "pass": True,
    }
    return g


def _parse_gates_log(needle: str):
    path = os.path.join(BATCH_DIR, "gates.log")
    if not os.path.exists(path):
        return {"kind": "abort", "pass": None, "note": "gates.log absent"}
    for line in open(path, encoding="utf-8", errors="replace"):
        if needle in line:
            return {"kind": "abort", "pass": line.rstrip().endswith("PASS"),
                    "line": line.strip()[:220], "source": "gates.log (pre-run)"}
    return {"kind": "abort", "pass": None, "note": "line not found in gates.log"}


# --------------------------------------------------------------------------
# Holm-2 over the PRIMARY family
# --------------------------------------------------------------------------

def holm2(p1: float, p2: float, alpha: float = ALPHA_FAMILY):
    """
    PREREG §1 / AMENDMENT_1 A-1: p ascending, first at alpha/2, second at alpha.
    Returns per-test alpha actually spent and the reject flags.
    """
    items = [("P-1", p1), ("P-2", p2)]
    order = sorted(range(2), key=lambda i: (items[i][1], items[i][0]))
    alphas = [alpha / 2.0, alpha]
    rej = {"P-1": False, "P-2": False}
    spent = {"P-1": None, "P-2": None}
    stopped = False
    for rank, idx in enumerate(order):
        name, p = items[idx]
        a = alphas[rank]
        spent[name] = a
        if stopped:
            rej[name] = False
            continue
        if p <= a:
            rej[name] = True
        else:
            rej[name] = False
            stopped = True
    return {"order": [items[i][0] for i in order],
            "alpha_spent": spent, "reject": rej}


# --------------------------------------------------------------------------
# main analysis
# --------------------------------------------------------------------------

def analyse(rows, corpus, strict=True):
    # PREREG §1: the verdict stratum is CROWDED only. Sparse rows, if any,
    # are dropped here and the count is recorded (never silently absorbed).
    n_in = len(rows)
    kept = [r for r in rows if r.get("stratum") == STRATUM]
    n_dropped_stratum = n_in - len(kept)
    if kept:
        rows = kept
    cube = Cube(rows)
    gates = recompute_gates(rows, cube, corpus)

    med = {}
    seed_counts = {}
    xcheck_bad = 0
    for a in cube.arms:
        v, ns, xb = cube.med3_scalar(a, strict=strict)
        med[a], seed_counts[a] = v, ns
        xcheck_bad += xb

    # ---- self-test: derived critical values must equal the frozen ones ----
    hi25, lo25 = crit_two(N_REGISTERED, 0.025)
    hi05, lo05 = crit_two(N_REGISTERED, 0.05)
    derived = {"one_0.025": crit_one(N_REGISTERED, 0.025),
               "one_0.05": crit_one(N_REGISTERED, 0.05),
               "two_0.025_hi": hi25, "two_0.025_lo": lo25,
               "two_0.05_hi": hi05, "two_0.05_lo": lo05}
    selftest = {"derived": derived, "frozen": FROZEN_CRIT,
                "agree": derived == FROZEN_CRIT}

    # ---- IUT (runs against us): L-SHIP < ZERO -----------------------------
    iut = paired(med.get("L-SHIP", np.array([])), med.get("ZERO", np.array([])),
                 "IUT L-SHIP < ZERO")
    iut_thr = crit_one(iut["n"], 0.025) if iut["n"] else None
    iut["threshold_alpha_0.025"] = iut_thr
    # An n so small that NO count can reach alpha is NOT a failure - it is an
    # unevaluable test (the S6 "principally impossible to pass" failure mode).
    # At the registered n=64 the critical value is 41, so this only ever fires
    # on smoke-sized inputs.
    iut["evaluable"] = bool(iut_thr is not None)
    iut["pass"] = bool(iut_thr is not None and iut["W"] >= iut_thr)
    iut["status"] = ("PASS" if iut["pass"] else
                     ("FAIL" if iut["evaluable"] else "NOT-EVALUABLE (n too small)"))

    # ---- PRIMARY ----------------------------------------------------------
    P1 = paired(med.get("L-VP", np.array([])), med.get("L-SHIP", np.array([])),
                "P-1  L-VP < L-SHIP  (one-sided)")
    P2 = paired(med.get("L-ITER0", np.array([])), med.get("L-SHIP", np.array([])),
                "P-2  L-ITER0 != L-SHIP  (two-sided)")

    hp = holm2(P1["p_one_sided_treat_lower"], P2["p_two_sided"])
    a1, a2 = hp["alpha_spent"]["P-1"], hp["alpha_spent"]["P-2"]

    P1["alpha_spent"] = a1
    P1["critical_value_at_alpha"] = crit_one(P1["n"], a1) if P1["n"] else None
    P1["reject_by_p"] = bool(P1["p_one_sided_treat_lower"] <= a1) if P1["n"] else False
    P1["reject_by_threshold"] = bool(
        P1["critical_value_at_alpha"] is not None
        and P1["W"] >= P1["critical_value_at_alpha"])
    P1["holm_reject"] = bool(hp["reject"]["P-1"])
    P1["route_agreement"] = bool(P1["reject_by_p"] == P1["reject_by_threshold"])

    hi, lo = crit_two(P2["n"], a2) if P2["n"] else (None, None)
    P2["alpha_spent"] = a2
    P2["critical_hi"], P2["critical_lo"] = hi, lo
    P2["reject_by_p"] = bool(P2["p_two_sided"] <= a2) if P2["n"] else False
    P2["reject_by_threshold"] = bool(
        (hi is not None and P2["W"] >= hi) or (lo is not None and P2["W"] <= lo))
    P2["holm_reject"] = bool(hp["reject"]["P-2"])
    P2["route_agreement"] = bool(P2["reject_by_p"] == P2["reject_by_threshold"])

    # P-2 label (PREREG §1.1); W counts L-ITER0 < L-SHIP i.e. ITER0 LOWER
    if P2["holm_reject"] and hi is not None and P2["W"] >= hi:
        p2_label = "OPTIMIZER-HARMS"
    elif P2["holm_reject"] and lo is not None and P2["W"] <= lo:
        p2_label = "OPTIMIZER-EARNS"
    else:
        p2_label = "OPTIMIZER-INERT"
    P2["label"] = p2_label
    P2["label_reading"] = gates["A5_CLAMP_READOUT"]["reading"]

    # MDE, mandatory when the label is INERT
    if P2["n"] > 0 and hi is not None:
        region = (lambda k, hi=hi, lo=lo: k >= hi or (lo is not None and k <= lo))
        pi80 = mde_win_rate(P2["n"], region, 0.80)
        P2["MDE"] = {
            "evaluable": True,
            "definition": ("smallest per-episode win fraction pi>0.5 with power "
                           ">= 0.80 over the two-sided rejection region"),
            "n": P2["n"], "alpha": a2, "region": {"hi": hi, "lo": lo},
            "pi_80": pi80,
            "wins_80": (int(math.ceil(pi80 * P2["n"])) if pi80 else None),
            "power_at_observed": _power(P2["n"], region, P2["W"] / P2["n"])
            if P2["n"] else None,
            "observed_median_rel_diff": P2["median_rel_diff"],
            "observed_median_rel_ci95": P2["median_rel_diff_ci95"],
            "note": ("this is a win-rate MDE; a sign test does not bound the "
                     "effect magnitude - the CI on the paired median relative "
                     "difference is the magnitude-side resolution"),
        }
    else:
        P2["MDE"] = {"evaluable": False, "n": P2["n"],
                     "reason": "no two-sided rejection region exists at n=%d, "
                               "alpha=%s" % (P2["n"], a2),
                     "observed_median_rel_diff": P2["median_rel_diff"],
                     "observed_median_rel_ci95": P2["median_rel_diff_ci95"]}
    if P1["n"] > 0 and P1["critical_value_at_alpha"] is not None:
        reg1 = (lambda k, c=P1["critical_value_at_alpha"]: k >= c)
        P1["MDE"] = {"pi_80": mde_win_rate(P1["n"], reg1, 0.80),
                     "power_at_observed": _power(P1["n"], reg1, P1["W"] / P1["n"])}
    else:
        P1["MDE"] = None

    # ---- SECONDARY --------------------------------------------------------
    sec = {}
    sec["S-2"] = paired(med.get("L-VP", np.array([])), med.get("L-ITER0", np.array([])),
                        "S-2  L-VP < L-ITER0 (one-sided)")
    sec["S-2b"] = paired(med.get("L-VP", np.array([])), med.get("L-STEP400", np.array([])),
                         "S-2b L-VP < L-STEP400 (one-sided)")
    sec["S-2c"] = paired(med.get("L-VP", np.array([])), med.get("L-VP-NW", np.array([])),
                         "S-2c L-VP vs L-VP-NW (two-sided)")
    for k in ("S-2", "S-2b"):
        d = sec[k]
        c = crit_one(d["n"], ALPHA_SECONDARY) if d["n"] else None
        d["alpha"] = ALPHA_SECONDARY
        d["critical_value"] = c
        d["reject_by_p"] = bool(d["p_one_sided_treat_lower"] <= ALPHA_SECONDARY) \
            if d["n"] else False
        d["reject_by_threshold"] = bool(c is not None and d["W"] >= c)
        d["route_agreement"] = bool(d["reject_by_p"] == d["reject_by_threshold"])
        d["verdict"] = "PASS" if d["reject_by_p"] else "NULL"
    d = sec["S-2c"]
    hi_c, lo_c = crit_two(d["n"], ALPHA_SECONDARY) if d["n"] else (None, None)
    d["alpha"] = ALPHA_SECONDARY
    d["critical_hi"], d["critical_lo"] = hi_c, lo_c
    d["reject_by_p"] = bool(d["p_two_sided"] <= ALPHA_SECONDARY) if d["n"] else False
    d["reject_by_threshold"] = bool((hi_c is not None and d["W"] >= hi_c)
                                    or (lo_c is not None and d["W"] <= lo_c))
    d["route_agreement"] = bool(d["reject_by_p"] == d["reject_by_threshold"])
    d["verdict"] = ("DIFFERENT (weighting matters)" if d["reject_by_p"]
                    else "NULL (weighting factor not separated)")

    # ---- S-3 shells (9, obs: prefix, independent cube path) ---------------
    shells = {}
    for h, key in enumerate(SHELL_KEYS):
        tv = cube.med3_shell("L-VP", h, strict=strict)
        cv = cube.med3_shell("L-SHIP", h, strict=strict)
        s = paired(tv, cv, "S-3 %s  L-VP < L-SHIP" % key)
        c = crit_one(s["n"], ALPHA_SECONDARY) if s["n"] else None
        s["critical_value"] = c
        s["reject_by_p"] = bool(s["p_one_sided_treat_lower"] <= ALPHA_SECONDARY) \
            if s["n"] else False
        s["verdict"] = "PASS" if s["reject_by_p"] else "NULL"
        shells[key] = s
    n_zero = sum(1 for s in shells.values() if s["W"] == 0 and s["n"] == 0)
    shells_meta = {
        "prefix_used": "obs:",
        "n_shells": len(SHELL_KEYS),
        "all_empty_suspect_key_mismatch": bool(n_zero == len(SHELL_KEYS)),
        "holm9_sensitivity_note": ("PREREG fixes no multiplicity rule inside "
                                   "S-3; raw per-shell p reported, Holm-9 added "
                                   "as sensitivity only"),
    }
    ps = sorted((v["p_one_sided_treat_lower"], k) for k, v in shells.items())
    m = len(ps)
    stop = False
    for i, (p, k) in enumerate(ps):
        a = ALPHA_SECONDARY / (m - i)
        shells[k]["holm9_alpha"] = a
        shells[k]["holm9_reject"] = bool((not stop) and p <= a)
        if p > a:
            stop = True

    # ---- descriptives -----------------------------------------------------
    desc = {}
    desc["D-2_wall_s_median"] = {
        a: (float(np.nanmedian(cube.wall[a])) if np.isfinite(cube.wall[a]).any() else None)
        for a in cube.arms}
    desc["D-3_merge_removed_median"] = {
        a: (float(np.nanmedian(cube.mremoved[a]))
            if np.isfinite(cube.mremoved[a]).any() else None)
        for a in cube.arms}
    desc["n_atoms_final_median"] = {
        a: (float(np.nanmedian(cube.natoms[a]))
            if np.isfinite(cube.natoms[a]).any() else None)
        for a in cube.arms}
    # D-1: does window residual improvement transfer to field nRMSE?
    wt = np.array([np.nanmedian(cube.wresid["L-VP"][i]) if "L-VP" in cube.wresid
                   else np.nan for i in range(len(cube.episodes))])
    wc = np.array([np.nanmedian(cube.wresid["L-SHIP"][i]) if "L-SHIP" in cube.wresid
                   else np.nan for i in range(len(cube.episodes))])
    ok = np.isfinite(wt) & np.isfinite(wc) & np.isfinite(med.get("L-VP", wt)) \
        & np.isfinite(med.get("L-SHIP", wc))
    if ok.sum() >= 3:
        dw = (wt[ok] - wc[ok]) / wc[ok]
        df = (med["L-VP"][ok] - med["L-SHIP"][ok]) / med["L-SHIP"][ok]
        r = float(np.corrcoef(dw, df)[0, 1]) if np.std(dw) > 0 and np.std(df) > 0 else None
    else:
        r = None
    desc["D-1_window_to_field"] = {"n": int(ok.sum()),
                                   "pearson_r_relwindow_vs_relfield": r}
    # D-4 (demoted from S-1 by AMENDMENT_1 A-7): descriptive ONLY
    d4 = paired(med.get("L-MERGE", np.array([])), med.get("L-SHIP", np.array([])),
                "D-4  L-MERGE vs L-SHIP  (DESCRIPTIVE, no hypothesis)")
    d4["status"] = "DESCRIPTIVE (AMENDMENT_1 A-7: S-1 demoted, no threshold)"
    desc["D-4_L-MERGE"] = d4
    if "ORA-K-CVX" in med and np.isfinite(med["ORA-K-CVX"]).any():
        ok2 = np.isfinite(med["ORA-K-CVX"]) & np.isfinite(med.get("L-SHIP", np.array([np.nan])))
        if ok2.any():
            desc["upper_bound_ratio_SHIP_over_CVX_median"] = float(
                np.median(med["L-SHIP"][ok2] / med["ORA-K-CVX"][ok2]))
            desc["upper_bound_note"] = ("PREREG §7-2: 6-factor bundle, MUST NOT be "
                                        "cited as a pure solver gain")

    # ---- death conditions, in the frozen total order (PREREG §5 + A-8) ----
    abort = [k for k, v in gates.items()
             if v.get("kind") == "abort" and v.get("pass") is False]
    order_trace = []
    verdict, reason = None, None
    if abort:
        verdict, reason = "ABORT", "abort gate(s) failed: %s" % ", ".join(sorted(abort))
    order_trace.append(("1 abort gates", abort))
    if verdict is None and gates["G-6"].get("pass") is False:
        verdict, reason = "TERMINATE (G-6 DCC)", "required fields missing"
    order_trace.append(("2 G-6 DCC", gates["G-6"].get("pass")))
    if verdict is None and iut["evaluable"] and not iut["pass"]:
        verdict, reason = "FAIL", ("IUT L-SHIP < ZERO = %d/%d < %s"
                                   % (iut["W"], iut["n"], iut["threshold_alpha_0.025"]))
    order_trace.append(("3 IUT", iut["status"]))
    n_eff = min(P1["n"], P2["n"])
    if verdict is None and n_eff < MIN_N:
        verdict, reason = "UNDECIDED (n)", "effective pairs %d < %d" % (n_eff, MIN_N)
    order_trace.append(("4 n>=56", n_eff))
    order_trace.append(("5 intervention G-2/G-3",
                        gates["G-2"].get("pass") and gates["G-3"].get("pass")))

    if verdict is None:
        if P1["holm_reject"] and p2_label == "OPTIMIZER-HARMS":
            verdict = "REMEDY-BY-SUBTRACTION"       # AMENDMENT_1 A-8
        elif P1["holm_reject"]:
            verdict = "REMEDY (P-1 PASS) / %s" % p2_label
        else:
            verdict = "NO-REMEDY (P-1 NULL) / %s" % p2_label
        reason = "Holm-2 applied: %s" % json.dumps(hp["reject"])
        # PREREG §8 stopping rule, as amended by A-8
        verdict_stop = (not P1["holm_reject"]) and p2_label != "OPTIMIZER-HARMS"
    else:
        verdict_stop = None

    return {
        "meta": {
            "script": "verify_w4.py (D2 independent verifier)",
            "rules": ["PREREG.md", "AMENDMENT_1.md", "AMENDMENT_2.md"],
            "n_rows_in": n_in,
            "n_rows": len(rows),
            "n_rows_dropped_non_crowded": n_dropped_stratum,
            "n_episodes": len(cube.episodes),
            "n_effective_pairs": {"P-1": P1["n"], "P-2": P2["n"],
                                  "family_min": n_eff, "min_required": MIN_N},
            "arms_seen": cube.arms,
            "med3_strict": strict,
            "med3_numpy_crosscheck_mismatches": xcheck_bad,
            "scipy_available": _HAVE_SCIPY,
            "seed_completeness": {a: {"episodes_with_3_seeds":
                                      int((seed_counts[a] == 3).sum()),
                                      "episodes_with_fewer":
                                      int(((seed_counts[a] > 0) & (seed_counts[a] < 3)).sum()),
                                      "episodes_absent":
                                      int((seed_counts[a] == 0).sum())}
                                  for a in cube.arms if a in EST_ARMS or a == DUP_ARM},
        },
        "threshold_selftest": selftest,
        "gates": gates,
        "IUT": iut,
        "PRIMARY": {"P-1": P1, "P-2": P2, "holm2": hp},
        "SECONDARY": {"S-2": sec["S-2"], "S-2b": sec["S-2b"], "S-2c": sec["S-2c"],
                      "S-3": {"meta": shells_meta, "shells": shells}},
        "DESCRIPTIVE": desc,
        "VERDICT": {"verdict": verdict, "reason": reason,
                    "P2_label": p2_label,
                    "stop_solver_axis": verdict_stop,
                    "priority_trace": [{"step": s, "value": _jsonable(v)}
                                       for s, v in order_trace]},
        "SCOPE": ("synthetic corpus; linear GT; direction_deg=0; single "
                  "checkpoint; CROWDED stratum only; init_method='learned' "
                  "FIXED (not the shipping router). PREREG §7 bans apply."),
    }


def _jsonable(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


# --------------------------------------------------------------------------
# printing
# --------------------------------------------------------------------------

def _fmt_p(p):
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "  n/a   "
    return "%.6g" % p


def report(res):
    L = []
    a = L.append
    m = res["meta"]
    a("=" * 78)
    a("W4 · D2 INDEPENDENT VERIFIER  (verify_w4.py)")
    a("=" * 78)
    a("rows=%d  episodes=%d  arms=%s" % (m["n_rows"], m["n_episodes"],
                                         ",".join(m["arms_seen"])))
    a("med3 strict=%s   numpy-median cross-check mismatches=%d   scipy=%s"
      % (m["med3_strict"], m["med3_numpy_crosscheck_mismatches"],
         m["scipy_available"]))
    part = {k: v for k, v in m["seed_completeness"].items()
            if v["episodes_with_fewer"] > 0}
    if part:
        a("!! partial seed coverage (strict med3 drops these episodes): %s"
          % json.dumps(part))
        a("!! rerun with --loose-med3 as the prespecified sensitivity")
    st = res["threshold_selftest"]
    a("threshold self-test (derived vs frozen): %s  %s"
      % ("AGREE" if st["agree"] else "*** MISMATCH ***", json.dumps(st["derived"])))
    a("")
    a("--- GATES (recomputed from rows; `gates_pass` field NOT trusted) ---")
    for k in sorted(res["gates"]):
        g = res["gates"][k]
        a("  %-16s kind=%-9s pass=%s" % (k, g.get("kind"), g.get("pass")))
    a("")
    i = res["IUT"]
    a("--- IUT (against us) ---")
    a("  L-SHIP < ZERO : W=%d/%d  thr>=%s  p=%s  -> %s"
      % (i["W"], i["n"], i["threshold_alpha_0.025"],
         _fmt_p(i["p_one_sided_treat_lower"]), i["status"]))
    a("")
    a("--- PRIMARY (Holm-2, family alpha=0.05) ---")
    p1, p2, h = res["PRIMARY"]["P-1"], res["PRIMARY"]["P-2"], res["PRIMARY"]["holm2"]
    a("  holm order = %s   alpha spent = %s" % (h["order"], h["alpha_spent"]))
    a("  P-1 L-VP < L-SHIP        W=%d/%d (ties %d, missing %d)  p1=%s  "
      "crit>=%s  reject=%s"
      % (p1["W"], p1["n"], p1["n_ties_dropped"], p1["n_missing_pairs"],
         _fmt_p(p1["p_one_sided_treat_lower"]), p1["critical_value_at_alpha"],
         p1["holm_reject"]))
    a("      median rel diff = %+.4f  CI95 %s   two-route agreement=%s"
      % (p1["median_rel_diff"], _rng(p1["median_rel_diff_ci95"]),
         p1["route_agreement"]))
    a("  P-2 L-ITER0 != L-SHIP    W=%d/%d (ties %d, missing %d)  p2=%s  "
      "crit hi>=%s lo<=%s  reject=%s"
      % (p2["W"], p2["n"], p2["n_ties_dropped"], p2["n_missing_pairs"],
         _fmt_p(p2["p_two_sided"]), p2["critical_hi"], p2["critical_lo"],
         p2["holm_reject"]))
    a("      median rel diff = %+.4f  CI95 %s   two-route agreement=%s"
      % (p2["median_rel_diff"], _rng(p2["median_rel_diff_ci95"]),
         p2["route_agreement"]))
    a("      LABEL = %s   (read as: %s)" % (p2["label"], p2["label_reading"]))
    if p2["label"] == "OPTIMIZER-INERT":
        d = p2.get("MDE") or {}
        if d.get("evaluable"):
            a("      MDE  : pi_80=%s (>= %s/%d wins)   power@observed=%.3f"
              % ("%.4f" % d["pi_80"] if d["pi_80"] else None, d["wins_80"],
                 d["n"], d["power_at_observed"] or float("nan")))
        else:
            a("      MDE  : NOT EVALUABLE -- %s" % d.get("reason"))
        a("      MDE  : magnitude side -> median rel diff %+.4f CI95 %s"
          % (d.get("observed_median_rel_diff", float("nan")),
             _rng(d.get("observed_median_rel_ci95"))))
    a("")
    a("--- SECONDARY (single prereg, alpha=0.025) ---")
    CONTRAST = {"S-2": "L-VP < L-ITER0    (one-sided)",
                "S-2b": "L-VP < L-STEP400  (one-sided)",
                "S-2c": "L-VP vs L-VP-NW   (two-sided)"}
    for k in ("S-2", "S-2b", "S-2c"):
        s = res["SECONDARY"][k]
        pk = "p_two_sided" if k == "S-2c" else "p_one_sided_treat_lower"
        a("  %-5s %-30s W=%d/%d  p=%-10s -> %s"
          % (k, CONTRAST[k], s["W"], s["n"], _fmt_p(s[pk]), s["verdict"]))
    a("  S-3 shells (obs: prefix, 9):")
    for key in SHELL_KEYS:
        s = res["SECONDARY"]["S-3"]["shells"][key]
        a("      %-14s W=%2d/%-2d  p=%-10s med rel %+0.4f  %s%s"
          % (key, s["W"], s["n"], _fmt_p(s["p_one_sided_treat_lower"]),
             s["median_rel_diff"], s["verdict"],
             "  [holm9 rej]" if s.get("holm9_reject") else ""))
    if res["SECONDARY"]["S-3"]["meta"]["all_empty_suspect_key_mismatch"]:
        a("      *** all 9 shells empty -> SUSPECT SHELL KEY PREFIX MISMATCH ***")
    a("")
    a("--- DESCRIPTIVE (no hypotheses) ---")
    d4 = res["DESCRIPTIVE"]["D-4_L-MERGE"]
    a("  D-4 L-MERGE vs L-SHIP  W=%d/%d  p=%s  (DESCRIPTIVE ONLY, A-7)"
      % (d4["W"], d4["n"], _fmt_p(d4["p_one_sided_treat_lower"])))
    a("  D-2 wall_s median: %s" % json.dumps(
        {k: (round(v, 2) if v is not None else None)
         for k, v in res["DESCRIPTIVE"]["D-2_wall_s_median"].items()}))
    a("  D-1 window->field pearson r = %s (n=%d)"
      % (res["DESCRIPTIVE"]["D-1_window_to_field"]["pearson_r_relwindow_vs_relfield"],
         res["DESCRIPTIVE"]["D-1_window_to_field"]["n"]))
    a("")
    a("=" * 78)
    a("VERDICT : %s" % res["VERDICT"]["verdict"])
    a("reason  : %s" % res["VERDICT"]["reason"])
    a("P-2 lbl : %s" % res["VERDICT"]["P2_label"])
    a("stop?   : %s" % res["VERDICT"]["stop_solver_axis"])
    a("=" * 78)
    a("SCOPE: %s" % res["SCOPE"])
    return "\n".join(L)


def _rng(ci):
    if not ci or ci[0] is None:
        return "[n/a]"
    return "[%+0.4f, %+0.4f]" % (ci[0], ci[1])


# --------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="W4 D2 independent verifier")
    ap.add_argument("--results", default=os.path.join(BATCH_DIR, "results.jsonl"))
    ap.add_argument("--corpus", default=os.path.join(BATCH_DIR, "corpus_w4.json"))
    ap.add_argument("--out", default=os.path.join(BATCH_DIR, "verify_w4_output.json"))
    ap.add_argument("--loose-med3", action="store_true",
                    help="sensitivity: median over whatever seeds exist")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    if not os.path.exists(args.results):
        print("results file not found: %s" % args.results, file=sys.stderr)
        return 2
    rows = []
    for ln, line in enumerate(open(args.results, encoding="utf-8"), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            print("skipping malformed line %d: %s" % (ln, exc), file=sys.stderr)
    corpus = None
    if os.path.exists(args.corpus):
        corpus = json.load(open(args.corpus, encoding="utf-8"))

    res = analyse(rows, corpus, strict=not args.loose_med3)
    res["meta"]["results_path"] = os.path.abspath(args.results)

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False, default=_jsonable)
    txt = report(res)
    if not args.quiet:
        print(txt)
    print("\n[verify_w4] wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
