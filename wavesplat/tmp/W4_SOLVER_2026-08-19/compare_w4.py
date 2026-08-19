#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_w4.py — automatic D1 vs D2 reconciliation for batch W4_SOLVER_2026-08-19.

The repository standing instruction is "every review gets at least two
independent confirmations".  `analyze_w4.py` (D1) and `verify_w4.py` (D2) were
written independently and therefore have DIFFERENT output shapes.  This script
does not care about shape: it harvests every node that looks like a test result
out of both JSON documents, canonicalises the test names, and compares only the
load-bearing numbers:

    W (win count) · n (effective pairs) · p-value(s) · verdict / label strings
    · each SECONDARY win count · the P-2 label · the MDE (when present)

Exit code 0 = no mismatch, 1 = mismatch found, 2 = input problem.

USAGE
    python3 compare_w4.py analyze_w4_output.json verify_w4_output.json
    python3 compare_w4.py A.json B.json --dump-labels     # reconcile names
    python3 compare_w4.py A.json B.json --rtol 1e-9
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys

BATCH_DIR = os.path.dirname(os.path.abspath(__file__))

# keys that carry a win count / an n / a p-value, in any plausible spelling
W_KEYS = ("w", "wins", "n_wins", "win", "k", "successes", "n_success",
          "wins_treat", "n_better", "count")
N_KEYS = ("n", "n_pairs", "n_eff", "n_effective", "npairs", "n_used", "n_valid")
P_KEY_RE = re.compile(r"^p($|_)|^p_?value|pval|_p$", re.I)

# the official vocabulary, so a label disagreement cannot hide behind wording
P2_LABELS = ("OPTIMIZER-HARMS", "OPTIMIZER-INERT", "OPTIMIZER-EARNS")
VERDICT_TOKENS = ("REMEDY-BY-SUBTRACTION", "NO-REMEDY", "REMEDY",
                  "UNDECIDED (n)", "UNDECIDED", "FAIL", "ABORT", "TERMINATE",
                  "NOT-EVALUABLE", "PASS", "NULL") + P2_LABELS

# shell keys are matched on the underscore form (obs:0_5 must survive intact)
OBS_RE = re.compile(r"obs:(\d+_(?:\d+|inf))")

# everything else is matched on a SPACE-separated form, because `\b` is useless
# against underscores (they are word characters) -- a real bug caught in test.
CANON_RULES = (
    (re.compile(r"\bs[- ]?2c\b"), lambda m: "S-2c"),
    (re.compile(r"\bs[- ]?2b\b"), lambda m: "S-2b"),
    (re.compile(r"\bs[- ]?2\b"), lambda m: "S-2"),
    (re.compile(r"\bs[- ]?1\b"), lambda m: "S-1"),
    (re.compile(r"\bs[- ]?3\b"), lambda m: "S-3"),
    (re.compile(r"\biut\b"), lambda m: "IUT"),
    (re.compile(r"\b(?:p|primary)[- ]?1\b"), lambda m: "P-1"),
    (re.compile(r"\b(?:p|primary)[- ]?2\b"), lambda m: "P-2"),
    (re.compile(r"\bd[- ]?4\b"), lambda m: "D-4"),
    (re.compile(r"\bmerge\b"), lambda m: "D-4"),
)

LABEL_FIELDS = ("label", "name", "test", "id", "contrast", "hypothesis", "key")


def _norm(s: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^0-9a-zA-Z:]+", "_", str(s))).strip("_").lower()


def canon(path, node):
    """Canonical test name from the JSON path plus any self-describing field."""
    parts = [str(p) for p in path]
    for f in LABEL_FIELDS:
        v = node.get(f)
        if isinstance(v, str):
            parts.append(v)
    txt = _norm("_".join(parts))
    # keep the raw `obs:` shell keys recognisable (D1 may spell them obs_0_5)
    und = re.sub(r"obs[_:]", "obs:", txt)
    m = OBS_RE.search(und)
    if m:
        return "S-3:obs:%s" % m.group(1)
    spaced = " " + und.replace("_", " ") + " "
    for rx, fn in CANON_RULES:
        m = rx.search(spaced)
        if m:
            return fn(m)
    return None


def _is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def harvest(doc):
    """
    Walk the document; every dict holding both a win-count-ish key and a
    p-value-ish key is treated as a test node.  Returns {canon_name: record}.
    """
    found = {}

    def visit(node, path):
        if isinstance(node, dict):
            wk = [k for k in node if _norm(k) in W_KEYS and _is_num(node[k])]
            pk = [k for k in node if P_KEY_RE.search(str(k)) and _is_num(node[k])]
            if wk and pk:
                name = canon(path, node)
                if name:
                    rec = {
                        "path": ".".join(str(p) for p in path),
                        "W": int(node[wk[0]]),
                        "W_key": wk[0],
                        "n": next((int(node[k]) for k in node
                                   if _norm(k) in N_KEYS and _is_num(node[k])), None),
                        "p": {str(k): float(node[k]) for k in pk},
                        "strings": sorted({str(node[k]) for k in node
                                           if isinstance(node[k], str)
                                           and any(t in str(node[k])
                                                   for t in VERDICT_TOKENS)}),
                    }
                    # first writer wins, but keep the shallower path
                    prev = found.get(name)
                    if prev is None or len(rec["path"]) < len(prev["path"]):
                        found[name] = rec
            for k, v in node.items():
                visit(v, path + [k])
        elif isinstance(node, list):
            for i, v in enumerate(node):
                visit(v, path + [i])

    visit(doc, [])
    return found


def collect_tokens(doc, tokens):
    """Every official token that appears in any string value of the document."""
    hits = set()

    def visit(node):
        if isinstance(node, dict):
            for v in node.values():
                visit(v)
        elif isinstance(node, list):
            for v in node:
                visit(v)
        elif isinstance(node, str):
            for t in tokens:
                if t in node:
                    hits.add(t)

    visit(doc)
    return hits


def find_p2_label(doc, harvested):
    rec = harvested.get("P-2")
    if rec:
        for s in rec["strings"]:
            for t in P2_LABELS:
                if t in s:
                    return t
    hits = collect_tokens(doc, P2_LABELS)
    if len(hits) == 1:
        return hits.pop()
    if hits:
        return "AMBIGUOUS(%s)" % ",".join(sorted(hits))
    return None


def find_overall_verdict(doc):
    """Prefer an explicit `verdict` key at the shallowest depth."""
    best = (10 ** 6, None)

    def visit(node, depth, key=None):
        nonlocal best
        if isinstance(node, dict):
            for k, v in node.items():
                if "verdict" in _norm(k) and isinstance(v, str) and depth < best[0]:
                    best = (depth, v)
                visit(v, depth + 1, k)
        elif isinstance(node, list):
            for v in node:
                visit(v, depth + 1, key)

    visit(doc, 0)
    return best[1]


def find_mde(doc):
    """
    Numeric leaves under any key containing 'mde'.

    The leaves are SCOPED by the enclosing canonical test name, because P-1 and
    P-2 can both carry an MDE block and comparing one against the other is a
    false alarm (this exact bug was caught by the perturbation test).  PREREG
    requires the MDE for P-2, so P-2-scoped leaves win; an MDE block that sits
    outside any recognised test is treated as the P-2 one.
    """
    scoped, loose = {}, {}

    def visit(node, path, inside):
        if isinstance(node, dict):
            who = _ancestor_test(path)
            for k, v in node.items():
                ins = inside or ("mde" in _norm(k))
                if ins and _is_num(v):
                    if who == "P-2":
                        scoped.setdefault(_norm(k), float(v))
                    elif who is None:
                        loose.setdefault(_norm(k), float(v))
                    # leaves owned by another test (e.g. P-1) are ignored
                visit(v, path + [k], ins)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                visit(v, path + [i], inside)

    visit(doc, [], False)
    return scoped or loose


def _ancestor_test(path):
    """Canonical test name implied by a JSON path, or None."""
    for cut in range(len(path), 0, -1):
        nm = canon(path[:cut], {})
        if nm:
            return nm
    return None


def _close(a, b, rtol):
    """
    RELATIVE tolerance with a tiny absolute floor.  A `max(1.0, ...)` scale here
    would make every p-value below 1e-9 compare equal to every other - i.e. it
    would silently pass the exact regime the PRIMARY tests live in.
    """
    if a is None or b is None:
        return a is None and b is None
    if math.isnan(a) and math.isnan(b):
        return True
    if math.isnan(a) or math.isnan(b):
        return False
    return abs(a - b) <= rtol * max(abs(a), abs(b)) + 1e-300


def compare(a_doc, b_doc, a_name, b_name, rtol=1e-9):
    A, B = harvest(a_doc), harvest(b_doc)
    lines, mism = [], 0
    add = lines.append

    add("=" * 82)
    add("compare_w4.py  —  D1 vs D2 reconciliation")
    add("  A = %s   (%d test nodes harvested)" % (a_name, len(A)))
    add("  B = %s   (%d test nodes harvested)" % (b_name, len(B)))
    add("=" * 82)

    common = sorted(set(A) & set(B), key=_sort_key)
    only_a = sorted(set(A) - set(B), key=_sort_key)
    only_b = sorted(set(B) - set(A), key=_sort_key)

    add("")
    add("%-16s %-13s %-13s %-9s %s" % ("TEST", "W  (A / B)", "n  (A / B)",
                                       "p-match", "STATUS"))
    add("-" * 82)
    for name in common:
        ra, rb = A[name], B[name]
        w_ok = ra["W"] == rb["W"]
        n_ok = (ra["n"] is None or rb["n"] is None) or ra["n"] == rb["n"]
        pa, pb = set(_round(v) for v in ra["p"].values()), \
            set(_round(v) for v in rb["p"].values())
        p_shared = sum(1 for x in ra["p"].values()
                       if any(_close(x, y, rtol) for y in rb["p"].values()))
        # the SMALLER p-set must be fully covered: one side may legitimately
        # store both the one- and two-sided p while the other stores only one.
        p_need = min(len(ra["p"]), len(rb["p"]))
        p_ok = (p_shared >= p_need) if p_need else (not ra["p"] and not rb["p"])
        bad = []
        if not w_ok:
            bad.append("W")
        if not n_ok:
            bad.append("n")
        if not p_ok:
            bad.append("p")
        status = "OK" if not bad else "*** MISMATCH: %s ***" % "+".join(bad)
        if bad:
            mism += 1
        add("%-16s %-13s %-13s %-9s %s"
            % (name, "%d / %d" % (ra["W"], rb["W"]),
               "%s / %s" % (ra["n"], rb["n"]),
               "%d/%d" % (p_shared, p_need), status))
        if bad and "p" in bad:
            add("      A p = %s" % sorted(pa))
            add("      B p = %s" % sorted(pb))

    add("-" * 82)
    if only_a:
        add("only in A (%d): %s" % (len(only_a), ", ".join(only_a)))
    if only_b:
        add("only in B (%d): %s" % (len(only_b), ", ".join(only_b)))
    if only_a or only_b:
        add("  ^ unmatched names are NOT counted as mismatches, but a core test")
        add("    missing on one side must be reconciled by hand (--dump-labels).")
    for core in ("P-1", "P-2", "IUT", "S-2", "S-2b", "S-2c"):
        if core not in A or core not in B:
            add("  !! CORE TEST %s absent from %s" %
                (core, "A" if core not in A else "B"))
            mism += 1

    # ---- label / verdict comparison ---------------------------------------
    add("")
    add("--- labels & verdicts ---")
    la, lb = find_p2_label(a_doc, A), find_p2_label(b_doc, B)
    ok = (la == lb)
    mism += 0 if ok else 1
    add("  P-2 label       A=%-18s B=%-18s %s"
        % (la, lb, "OK" if ok else "*** MISMATCH ***"))

    va, vb = find_overall_verdict(a_doc), find_overall_verdict(b_doc)
    ta = _verdict_tokens(va)
    tb = _verdict_tokens(vb)
    ok = (ta == tb)
    mism += 0 if ok else 1
    add("  overall verdict A=%-28s B=%-28s %s"
        % (str(va)[:28], str(vb)[:28], "OK" if ok else "*** MISMATCH ***"))
    if not ok:
        add("      token sets: A=%s  B=%s" % (sorted(ta), sorted(tb)))

    # ---- MDE --------------------------------------------------------------
    ma, mb = find_mde(a_doc), find_mde(b_doc)
    keys = sorted(set(ma) & set(mb))
    if keys:
        add("")
        add("--- MDE numeric leaves shared by both ---")
        for k in keys:
            ok = _close(ma[k], mb[k], 1e-6)
            if not ok:
                mism += 1
            add("  %-26s A=%-14.6g B=%-14.6g %s"
                % (k, ma[k], mb[k], "OK" if ok else "*** MISMATCH ***"))
    elif ma or mb:
        add("")
        add("--- MDE: no shared numeric leaf names (A=%d, B=%d) -> compare by hand"
            % (len(ma), len(mb)))

    add("")
    add("=" * 82)
    add("RESULT: %d core comparison(s), %d MISMATCH(es)"
        % (len(common) + 2, mism))
    add("=" * 82)
    return "\n".join(lines), mism


def _verdict_tokens(s):
    if not isinstance(s, str):
        return set()
    return {t for t in VERDICT_TOKENS if t in s}


def _round(x):
    try:
        return float("%.12g" % x)
    except (TypeError, ValueError):
        return x


def _sort_key(name):
    order = {"P-1": 0, "P-2": 1, "IUT": 2, "S-1": 3, "S-2": 4, "S-2b": 5,
             "S-2c": 6, "S-3": 7, "D-4": 9}
    return (order.get(name, 8), name)


def _dump(tag, doc):
    print("--- %s harvested test nodes ---" % tag)
    for k, v in sorted(harvest(doc).items(), key=lambda kv: _sort_key(kv[0])):
        print("  %-16s W=%-4d n=%-5s p_keys=%s   <- %s"
              % (k, v["W"], v["n"], ",".join(sorted(v["p"])), v["path"]))
    print()


def _resolve_d1(path):
    """
    Locate D1's output.  D1 names its files `verdict_w4_*.json`; this only ever
    looks at FILE NAMES, never at content, so it does not touch independence
    (and by the time this script runs, both analyzers are already frozen).
    """
    if os.path.exists(path):
        return path
    import glob
    cand = [p for p in sorted(glob.glob(os.path.join(BATCH_DIR, "verdict_w4_*.json")))
            if "fake" not in os.path.basename(p) and "fail" not in os.path.basename(p)]
    real = [p for p in cand if "smoke" not in os.path.basename(p)]
    pick = (real or cand)
    if pick:
        chosen = pick[-1]
        print("[compare_w4] D1 output auto-resolved to %s"
              % os.path.basename(chosen), file=sys.stderr)
        return chosen
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description="Reconcile D1 and D2 W4 outputs")
    ap.add_argument("a", nargs="?",
                    default=os.path.join(BATCH_DIR, "analyze_w4_output.json"),
                    help="D1 output JSON (auto-discovers verdict_w4_*.json)")
    ap.add_argument("b", nargs="?",
                    default=os.path.join(BATCH_DIR, "verify_w4_output.json"),
                    help="D2 output JSON")
    ap.add_argument("--rtol", type=float, default=1e-9)
    ap.add_argument("--dump-labels", action="store_true",
                    help="print the harvested node names for manual mapping")
    ap.add_argument("--out", default=None, help="also write the report here")
    args = ap.parse_args(argv)

    args.a = _resolve_d1(args.a)
    for p in (args.a, args.b):
        if not os.path.exists(p):
            print("input not found: %s" % p, file=sys.stderr)
            print("hint: pass the two JSON paths explicitly", file=sys.stderr)
            return 2
    a_doc = json.load(open(args.a, encoding="utf-8"))
    b_doc = json.load(open(args.b, encoding="utf-8"))

    if args.dump_labels:
        _dump("A: %s" % os.path.basename(args.a), a_doc)
        _dump("B: %s" % os.path.basename(args.b), b_doc)

    txt, mism = compare(a_doc, b_doc, os.path.basename(args.a),
                        os.path.basename(args.b), rtol=args.rtol)
    print(txt)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(txt + "\n")
        print("\n[compare_w4] wrote %s" % args.out)
    return 1 if mism else 0


if __name__ == "__main__":
    raise SystemExit(main())
