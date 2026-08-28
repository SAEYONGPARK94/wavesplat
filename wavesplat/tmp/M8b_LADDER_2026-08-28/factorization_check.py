#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M-8b **단일출처 재유도** — 가법성 두 수 + Holm 도달가능성 증명. (2026-08-29, `AMENDMENT_2.md`)

★**런 0회.** 동결 `verdict_m8b.json` 의 **원 계수만** 읽고(⛔개정으로 *추가*된 필드는
근거로 쓰지 않는다) 두 수를 **다시 유도**한 뒤, 문서에 적힌 값과 어긋나면 `AssertionError` 다.

선례 = `tmp/B3_REFIT_2026-08-23/final6.py` · `tmp/STATUS_2026-08-28/f1_table.py` ·
`tmp/M6_PACING_2026-08-28/alarm_xtab_m6.py`.

무엇을 강제하는가
  A. **가법성 두 수**  ΣΔ_k(6고리 로그비 중앙값의 합) vs 끝점 로그비 중앙값,
     그리고 두 승수 exp(·) 와 그 격차 %.  ⟹ `AMENDMENT_2.md`·`RESULT.md`·논문이 같은 수를 쓴다.
  B. **Holm 도달가능성**  n=12 단측 부호검정에서 α=0.05/6 을 통과하는 **최소 승수 = 11**.
     ⟹ *"10/12 는 이 설계에서 원리적으로 기각 불가"* 가 **결과와 무관하게** 참임을 보인다.
  C. **가족 MDE**  thr=11 의 80% 검정력 요구 승률 q80 = 0.9307(≈0.931),
     단일검정(thr=10) 의 0.8693(≈0.869)와 **다른 수**임을 강제한다.
  D. **개정 필드 대조**  개정이 verdict 에 *추가한* `thresholds.q80_family` 등이
     여기서 재유도한 값과 일치하는지(손복사 금지).
  E. **문서 축자 대조**  세 문서(`AMENDMENT_2.md`·`RESULT.md`·`thesis_main.tex`)에
     필요한 원자가 실제로 들어 있는지.

사용:  cd /home/vexit/wavesplat/wavesplat && python3 tmp/M8b_LADDER_2026-08-28/factorization_check.py
"""
import hashlib
import io
import json
import math
import os
import re
import sys
from math import comb

ROOT = "/home/vexit/wavesplat/wavesplat"
VERDICT = os.path.join(ROOT, "tmp/M8b_LADDER_2026-08-28/verdict_m8b.json")
AMEND = os.path.join(ROOT, "tmp/M8b_LADDER_2026-08-28/AMENDMENT_2.md")
RESULT = os.path.join(ROOT, "tmp/M8b_LADDER_2026-08-28/RESULT.md")
TEX = os.path.join(ROOT, "thesis/thesis_main.tex")

FAIL = []


def chk(cond, msg):
    print(("  ok   " if cond else "  ⛔FAIL ") + msg)
    if not cond:
        FAIL.append(msg)


# ─────────────────────────────────────────────────────────────────────────────
# 0. 동결 자료
# ─────────────────────────────────────────────────────────────────────────────
V = json.load(io.open(VERDICT, encoding="utf-8"))
print("# verdict md5 = %s" % hashlib.md5(io.open(VERDICT, "rb").read()).hexdigest())
print("# ⚠verdict 은 동결 목록(FROZEN_md5.txt)에 없다 — 산출물이다. 판정 **계수** 는 개정이 건드리지 않았고")
print("#   이 스크립트는 그 계수만 근거로 쓴다(개정이 추가한 필드는 D 에서 *대조 대상* 일 뿐이다).")

# ─────────────────────────────────────────────────────────────────────────────
# A. 가법성 — ⛔`magnitude.sum_delta_log` 를 읽지 않고 `chain` 에서 **다시 더한다**
# ─────────────────────────────────────────────────────────────────────────────
print("\n== A. 가법성 재유도 (⛔요약 필드 미사용 — /chain 과 /endpoint 원 계수에서 재계산)")
chain = V["chain"]
assert len(chain) == 6, "고리가 6개가 아니다"
links = [c["log_ratio_median"] for c in chain]
sum_dlog = sum(links)
end_dlog = V["endpoint"]["log_ratio_median"]
mult_sum, mult_end = math.exp(sum_dlog), math.exp(end_dlog)
gap_pct = (mult_end / mult_sum - 1.0) * 100.0

print("   고리     a → b                          Δlog        승수")
for c in chain:
    print("   %-3s %-14s→ %-14s %+10.6f  %8.6f"
          % (c.get("link", "?"), c["a"], c["b"], c["log_ratio_median"],
             math.exp(c["log_ratio_median"])))
print("   %-34s %+10.6f  %8.6f   ← ΣΔ_k (합) / Π (곱)" % ("L1…L6 합/곱", sum_dlog, mult_sum))
print("   %-34s %+10.6f  %8.6f   ← 끝점 BANK→PROP" % ("끝점", end_dlog, mult_end))
print("   승수 격차 = %.4f %%   (Δlog 격차 = %.6f)" % (gap_pct, end_dlog - sum_dlog))

chk(abs(round(sum_dlog, 6) - 0.419195) < 5e-7, "ΣΔ_k = 0.419195 (재유도)")
chk(abs(round(end_dlog, 6) - 0.421363) < 5e-7, "끝점 Δlog = 0.421363 (재유도)")
chk(abs(round(mult_sum, 5) - 1.52074) < 5e-6, "승수 곱 = 1.52074 (재유도)")
chk(abs(round(mult_end, 5) - 1.52404) < 5e-6, "끝점 승수 = 1.52404 (재유도)")
chk(abs(round(gap_pct, 3) - 0.217) < 5e-4, "승수 격차 = 0.217%% (재유도)")

# 곱으로도 같은 수인가 (로그합 == 곱 은 항등이지만, 반올림된 mult_median 열과도 맞는지 본다)
prod = 1.0
for r in V["magnitude"]["per_link"]:
    prod *= r["mult_median"]
chk(abs(prod - mult_sum) < 5e-6, "per_link 의 반올림 승수 곱도 같은 값 (%.6f)" % prod)

# ─────────────────────────────────────────────────────────────────────────────
# B. Holm 도달가능성 증명 — **결과와 무관하게** 참인 설계의 성질
# ─────────────────────────────────────────────────────────────────────────────
print("\n== B. Holm 도달가능성 (⛔자료를 보지 않는다 — n·α 만으로 결정된다)")
N = V["primary"]["n"]
ALPHA = 0.05
FAM = len(chain)


def sign_p(w, n):
    return sum(comb(n, k) for k in range(w, n + 1)) / 2 ** n


def min_wins(n, a):
    for w in range(n + 1):
        if sign_p(w, n) <= a:
            return w
    return n + 1


a0 = ALPHA / FAM
print("   n=%d, 가족 크기=%d, 첫 Holm 유의수준 α₁ = %.9f" % (N, FAM, a0))
print("   도달 가능한 단측 p (승수 w):")
for w in range(N - 6, N + 1):
    p = sign_p(w, N)
    print("     w=%2d  p=%.9f   %s α₁" % (w, p, "≤" if p <= a0 else ">"))
chk(min_wins(N, a0) == 11, "α₁ 을 통과하는 **최소 승수 = 11** (10 은 원리적으로 불가)")
chk(sign_p(10, N) > a0,
    "w=10 의 p=%.6f 가 α₁=%.6f 보다 크다 ⟹ 10/12 는 어떤 자료에서도 첫 단계 기각 불가"
    % (sign_p(10, N), a0))
holm_min = [min_wins(N, ALPHA / (FAM - i)) for i in range(FAM)]
print("   단계별 최소 승수 =", holm_min)
chk(holm_min == V["thresholds"]["holm_min_wins"], "verdict 의 holm_min_wins 와 일치")
chk(holm_min == [11, 11, 11, 11, 10, 10], "= [11,11,11,11,10,10]")
chk(holm_min[0] == 11 and holm_min[1] == 11,
    "★단계적 하강이라 뒤의 10 은 **앞의 넷이 11 로 먼저 기각된 뒤에만** 도달한다")

# ─────────────────────────────────────────────────────────────────────────────
# C. MDE 두 값 — 단일검정 vs 가족
# ─────────────────────────────────────────────────────────────────────────────
print("\n== C. MDE 두 값 (⛔하나로 합치지 마라)")


def power_at(q, n, thr):
    return sum(comb(n, k) * q ** k * (1 - q) ** (n - k) for k in range(thr, n + 1))


def q80(n, thr):
    lo, hi = 0.5, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if power_at(mid, n, thr) < 0.80:
            lo = mid
        else:
            hi = mid
    return round((lo + hi) / 2, 4)


q_single, q_family = q80(N, 10), q80(N, 11)
print("   PRIMARY(단일검정, 문턱 10/12): q80 = %.4f" % q_single)
print("   사슬(가족 Holm 1~4번째, 문턱 11/12): q80 = %.4f" % q_family)
chk(abs(q_single - 0.8693) < 5e-5, "단일검정 q80 = 0.8693 (논문 표기 0.869)")
chk(abs(q_family - 0.9307) < 5e-5, "가족 q80 = 0.9307 (논문 표기 0.931)")
chk(q_family > q_single, "★가족 MDE 가 더 크다 — 단일검정 값을 가족 주장에 붙이면 **과소 보고**다")

# ─────────────────────────────────────────────────────────────────────────────
# D. 개정이 verdict 에 추가한 필드가 재유도값과 일치하는가 (손복사 금지)
# ─────────────────────────────────────────────────────────────────────────────
print("\n== D. 개정 추가 필드 대조 (⛔손복사 금지)")
th = V["thresholds"]
chk("q80_primary" in th and abs(th["q80_primary"] - q_single) < 5e-5,
    "thresholds.q80_primary 보존 + 재유도 일치")
chk("q80_family" in th and abs(th["q80_family"] - q_family) < 5e-5,
    "thresholds.q80_family 추가 + 재유도 일치")
chk(th.get("holm_first_min_wins") == 11, "thresholds.holm_first_min_wins = 11")
NEW = "NO-SINGLE-LINK-DETECTED (n=12, q80=0.931)"
chk(V.get("chain_label") == NEW, "chain_label = 새 라벨")
chk(V.get("label") == NEW, "label = 새 라벨")
chk(V.get("chain_label_as_generated") == "GAP-NOT-FACTORIZABLE",
    "★chain_label_as_generated 로 **생성기 원문을 보존**(계보가 끊기지 않는다)")
# ⚠"상호작용" 이라는 **글자** 를 지우는 것이 목적이 아니다 — 정정 문안이 그것을 금지 대상으로
#   인용한다. 목적은 `⟹ 상호작용` 이라는 **살아 있는 읽기** 를 없애는 것이다.
_why = V.get("chain_why", "")
chk("⟹ **상호작용**" not in _why and "⟹ 상호작용" not in _why,
    "⛔`⟹ 상호작용` 이라는 **읽기** 가 chain_why 에서 사라졌다")
chk(("상호작용" not in _why) or ("⛔" in _why.split("상호작용")[0][-40:]),
    "⛔chain_why 에 남은 `상호작용` 은 금지 표지(⛔)를 단 인용뿐이다")
chk("상호작용" in V.get("amendment_2", {}).get("what_was_removed", ""),
    "★삭제한 문언은 개정 블록에 **인용으로만** 남는다(기록은 지우지 않는다)")
a2 = V.get("amendment_2", {})
fz = a2.get("factorization", {})
chk(abs(fz.get("sum_delta_log", -1) - round(sum_dlog, 6)) < 5e-7, "amendment_2 ΣΔ_k 일치")
chk(abs(fz.get("endpoint_delta_log", -1) - round(end_dlog, 6)) < 5e-7, "amendment_2 끝점 일치")
chk(abs(fz.get("gap_pct", -1) - round(gap_pct, 3)) < 5e-4, "amendment_2 격차 % 일치")

# ─────────────────────────────────────────────────────────────────────────────
# E. 문서 축자 대조
# ─────────────────────────────────────────────────────────────────────────────
print("\n== E. 문서 축자 대조")


def rd(p):
    return io.open(p, encoding="utf-8").read()


def nrm(s):
    return re.sub(r"[\s{}$\\,]|\\textbf|\\emph", "", s)


docs = {"AMENDMENT_2.md": rd(AMEND), "RESULT.md": rd(RESULT), "thesis_main.tex": rd(TEX)}
ATOMS = {
    "새 라벨":            ["NO-SINGLE-LINK-DETECTED"],
    "ΣΔ_k":              ["0.419195", "0.4192"],
    "끝점 Δlog":          ["0.421363", "0.4214"],
    "승수 곱":            ["1.52074", "1.5207"],
    "끝점 승수":          ["1.52404", "1.5240"],
    "격차 %":             ["0.217"],
    "가족 MDE":           ["0.931"],
    "단일 MDE 보존":       ["0.869"],
    "가족 최소 승수":       ["11/12", "11승"],
}
for name, doc in docs.items():
    flat = nrm(doc)
    miss = [k for k, alts in ATOMS.items() if not any(nrm(a) in flat for a in alts)]
    chk(not miss, "%s: 필요 원자 전건 존재%s" % (name, "" if not miss else "  빠짐=%s" % miss))

# 논문에서 구 문안이 **살아 있는 주장으로** 남아 있는가.
# ⚠단순 부재 검사는 쓸 수 없다 — 정정 후 문안이 그 구 문안을 **금지 대상으로 인용**한다.
#   ⟹ 규칙: 본문의 모든 출현이 앞쪽 60자 안에 금지·철회 표지(⛔ 또는 "철회")를 달고 있어야 한다.
tex_body = "\n".join(l for l in docs["thesis_main.tex"].splitlines()
                     if not l.lstrip().startswith("%"))
tex_body = re.sub(r"(?<!\\)%.*$", "", tex_body, flags=re.M)
flat_body = nrm(tex_body)
MARK = re.compile(r"[⛔]|철회|retract", re.I)
for bad_phrase in ("단일 요인으로 인수분해되지 않", "격차가 인수분해되지 않",
                   "does not factorise", "⟹ 상호작용"):
    needle = nrm(bad_phrase)
    hits, k = [], 0
    while True:
        k = flat_body.find(needle, k)
        if k < 0:
            break
        hits.append(k)
        k += 1
    # ⚠표지는 앞에도 뒤에도 올 수 있다("…이었다. 둘 다 **철회한다**") ⟹ 양방향 창.
    unguarded = [h for h in hits
                 if not MARK.search(flat_body[max(0, h - 60):h + len(needle) + 80])]
    chk(not unguarded,
        "논문 본문: %r 이 살아 있는 주장으로 남아 있지 않다 (출현 %d, 무표지 %d)"
        % (bad_phrase, len(hits), len(unguarded)))

print("\n" + "=" * 66)
if FAIL:
    print("⛔ %d 건 불일치" % len(FAIL))
    for m in FAIL:
        print("   -", m)
    raise AssertionError("factorization_check: %d 건 불일치 — 문서와 동결 자료가 어긋난다" % len(FAIL))
print("✅ 전건 일치 — 문서의 두 수와 Holm 증명이 동결 verdict 에서 재유도된다.")
