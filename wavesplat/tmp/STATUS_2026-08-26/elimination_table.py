#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""소거표 단일출처 생성기 — STATUS_2026-08-26 §3.

★왜 이 파일이 있는가 (2026-08-24 교훈, `project_night_audit_2026-08-23.md`):
   *"인용 가능한 수치를 낸 스크립트가 저장소에 없으면 안 된다."*
   `FINAL_REAL_TIMEAXIS.md` 의 최종 세 문장이 대화형 재집계로만 존재해
   재현 불가였던 사고의 재발 방지다.

이 스크립트는 **런을 0회 한다.** 동결된 `verdict_*.json` 만 읽어 표를 만든다.
⛔**수를 손으로 옮기지 않는다** — 표의 모든 정수·비율·p 는 JSON 에서 나온다.
행 이름과 "어느 키를 읽는가" 만 여기 적혀 있고, 그 대응이 곧 이 표의 주장이다.

용법:
    python3 tmp/STATUS_2026-08-26/elimination_table.py            # 표 출력
    python3 tmp/STATUS_2026-08-26/elimination_table.py --check    # 자기검증만 (exit code)

자기검증(`--check`)이 확인하는 것:
  1. 여섯 verdict 파일이 존재하고 파싱된다
  2. 인용하는 배치의 게이트가 전건 통과다 (⛔`B-14` 는 예외 — `G-1` FAIL 이라
     **축 A 를 읽지 않는다**. 이 표는 `B-14` 에서 **축 B(`h2`) 만** 읽는다)
  3. 각 행이 읽는 라벨이 verdict 의 라벨과 글자 그대로 일치한다
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TMP = os.path.dirname(HERE)

V = {
    "B-14":  os.path.join(TMP, "B14_OBSMODEL_2026-08-24",   "verdict_b14.json"),
    "B-14b": os.path.join(TMP, "B14_OBSMODEL_2026-08-24",   "verdict_b14b.json"),
    "B-15":  os.path.join(TMP, "B15_RAYDROP_2026-08-24",    "verdict_b15.json"),
    "B-17":  os.path.join(TMP, "B17_REALNOISE_2026-08-25",  "verdict_b17.json"),
    "B-18":  os.path.join(TMP, "B18_DENSITY_NOISE_2026-08-25", "verdict_b18.json"),
    "B-19":  os.path.join(TMP, "B19_KGRID_SPARSE_2026-08-25",  "verdict_b19.json"),
}


def load():
    d = {}
    for k, p in V.items():
        if not os.path.exists(p):
            raise SystemExit("MISSING: %s" % p)
        with open(p, encoding="utf-8") as f:
            d[k] = json.load(f)
    return d


# ---------------------------------------------------------------- 행 생성기
def row_noise_shape(d):
    """잡음의 **형태** — `B-14b`(관측 모형 4종) + `B-17`(거리·입사각 σ 곡선)."""
    h1 = d["B-14b"]["h1"]
    best = max(h1.items(), key=lambda kv: kv[1]["zero_better"])
    b17 = d["B-17"]
    pv = b17["paired_vs_const"]
    fam = b17["rules"]["FAMILY"]                       # 탐색적 팔은 제외
    fam_rows = {k: v for k, v in pv.items() if k in fam}
    b17_best = max(fam_rows.items(), key=lambda kv: kv[1]["better"])
    n17 = b17_best[1]["n"]
    labels = set(b17["paired_labels"][k] for k in fam_rows)
    ev = (
        "`B-14b` 네 관측 모형 최대 **%d/%d** (%s, p=%.4f) · "
        "★`B-17` 짝 비교 **가족 4팔 전부 널** (최대 %d/%d = %s, p≥%.5f)"
        % (best[1]["zero_better"], best[1]["n"], best[0], best[1]["p_one_sided"],
           b17_best[1]["better"], n17, b17_best[0],
           min(v["p_one_sided"] for v in fam_rows.values()))
    )
    note = ("형태를 **논문 곡선대로 거리·입사각 의존**으로 바꿔도 짝으로 안 움직인다"
            " (`B-17` 짝 라벨 %s)" % "/".join(sorted(labels)))
    return ("잡음의 **형태**", "⛔아니다", ev, note,
            [("B-14b", d["B-14b"]["h1_label"]),
             ("B-17-paired", sorted(labels)[0] if len(labels) == 1 else "MIXED")])


def row_capacity(d):
    """표현의 **용량** — `B-14` 축 B(밀도 사다리, 전부 σ=0)."""
    h2 = d["B-14"]["h2"]
    top = max(h2.items(), key=lambda kv: int(kv[0]))
    ev = ("`B-14` 축 B: 관측 %d 개(m=%s)에서 참 제로필드를 **%d/%d** 로 이기고 "
          "오차가 밀도에 단조 감소 (%s), p=%.5f"
          % (int(top[1]["median_obs"]), top[0], top[1]["beats_zero"], top[1]["n"],
             "→".join("%.3f" % h2[k]["median_nrmse"] for k in sorted(h2, key=int)),
             d["B-14"]["h2_p_one_sided"]))
    note = "⚠**명목만**(가족단위 미달) + 배포의 25배 관측 ⟹ `EXPLORATORY`. ⛔운용 가능성 주장 아님"
    return ("표현의 **용량**", "⛔아니다", ev, note, [("B-14/h2", d["B-14"]["h2_label"])])


def row_sensor(d):
    """센서 모형 — `B-15`(raydrop uniform vs 물리적으로 옳은 slope)."""
    b = d["B-15"]["by_raydrop"]["slope"]
    u = d["B-15"]["by_raydrop"]["uniform"]
    ev = ("`B-15`: 물리적으로 옳은 `slope` 에서 `SHIP<SPEC25` **%d/%d** (p=%.6f, 상대 중앙 %.2f%%), "
          "`uniform` %d/%d — 음성대조 양쪽 %d/%d·%d/%d"
          % (b["ship_beats_spec"], b["n"], b["p_one_sided"], b["rel_median_pct"],
             u["ship_beats_spec"], u["n"],
             b["spec_beats_zero"], b["n"], u["spec_beats_zero"], u["n"]))
    note = "⚠**합성 층**의 교락을 닫은 것이다. ⛔실측 음성의 원인 후보로서만 소거된다 · ⛔크기 비교 금지"
    return ("**센서 모형**", "⛔결론을 바꾸지 않는다", ev, note,
            [("B-15", d["B-15"]["label"]), ("B-15/2nd", d["B-15"]["secondary_label"])])


def row_sparsity(d):
    c = d["B-18"]["primary"]
    th = d["B-18"]["thresholds"]
    B, A = c["B_DENSE"], c["A_DEPLOY"]
    ev = ("★`B-18` `B_DENSE`(m=%d, σ=%s): **%d/%d** — 명목 문턱 **%d** 미달, "
          "배포 근사 `A_DEPLOY`(m=%d) **%d/%d** 과 **동률**"
          % (B["m_scan"], B["noise"], B["beats_zero"], B["n"], th["nominal"],
             A["m_scan"], A["beats_zero"], A["n"]))
    note = ("★**어제(`STATUS_2026-08-24 §7`)의 마지막 생존 후보**였고 오늘 문턱을 못 넘었다. "
            "⚠**신규 증거는 이 한 칸뿐**(σ=0.02 열); σ=0 열은 `B-14` 축 B 의 재추첨이다")
    return ("★**관측의 희소성**", "⛔★아니다 (**오늘 죽었다**)", ev, note,
            [("B-18", d["B-18"]["label"])])


def row_noise_level(d):
    c = d["B-18"]["primary"]
    th = d["B-18"]["thresholds"]
    C = c["C_QUIET"]
    nd = d["B-18"]["secondary"]["noise_dose"]
    ev = ("`B-18` `C_QUIET`(m=%d, σ=%s): **%d/%d** — 문턱 %d 미달이고 "
          "★배포 밀도에서 잡음을 빼면 **나빠진다**(짝 %d/%d 개선, 상대 중앙 **+%.2f%%**, p=%.4f)"
          % (C["m_scan"], C["noise"], C["beats_zero"], C["n"], th["nominal"],
             nd["better"], nd["n"], nd["rel_median_pct"], nd["p_one_sided"]))
    note = ("⚠**탐색적**(사전등록의 가설이 아니다) · ⛔*\"잡음이 이롭다\"* 로 승격 금지 — "
            "라벨은 %s" % nd["label"])
    return ("잡음의 **수준**", "⛔아니다", ev, note, [("B-18/noise_dose", nd["label"])])


def row_both(d):
    c = d["B-18"]["primary"]
    th = d["B-18"]["thresholds"]
    D = c["D_BOTH"]
    it = d["B-18"]["secondary"]["interaction"]
    ev = ("`B-18` `D_BOTH`(m=%d, σ=%s): **%d/%d** — 문턱 %d 미달 · "
          "상호작용 %s (I<0 %d/%d, 양측 p=%.4f)"
          % (D["m_scan"], D["noise"], D["beats_zero"], D["n"], th["nominal"],
             it["label"], it["synergy_count"], it["n"], it["p_two_sided"]))
    note = ("⚠★**라벨을 가른 칸이 여기다** — 이 셀은 `B-14` 축 B 에서 문턱 **정확히 10/12** 였고 "
            "난수 규약 수리 뒤 **%d/%d** 다 ⟹ *\"둘 다 줘도 회복 못 한다\"* 를 **강하게 읽지 마라**"
            % (D["beats_zero"], D["n"]))
    return ("**둘의 조합**", "⛔아니다", ev, note, [("B-18/interaction", it["label"])])


REMAINING = [
    ("**관측 기하** (거리·입사각·폐색 — 잡음의 *크기* 가 아니라 *어디서 오는가*)",
     "⚠**미측정** — `B-18` 은 관측 모형을 `IID` 하나로 고정했다"),
    ("**창 길이 / 시간축** (`win=3` 하나만 돌았다)", "⚠**미측정**"),
    ("**전방모델의 정상성 가정** (창 안에서 스펙트럼 불변 전제)", "⚠**미측정**"),
    ("**스테레오 재구성면 자체의 오차** (참값으로 쓰는 면이 이미 틀렸을 가능성)",
     "⚠**미측정** — 참값을 검증할 독립 자료가 없다"),
    ("**사이트 기하 · 레코드 구성** (12 = 10 일자 · 4 사이트, 58%가 한 사이트)", "⚠**미측정**"),
]

BUILDERS = [row_noise_shape, row_capacity, row_sensor,
            row_sparsity, row_noise_level, row_both]


def check(d):
    """자기검증 — 실패하면 exit 1."""
    errs = []
    # (2) 게이트 전건 통과. ⛔`B-14` 만 예외: `G-1` FAIL 이라 축 A 를 읽지 않는다.
    def gp(v):
        if "gates_pass" in v:
            return v["gates_pass"], v["gates_total"]
        g = v["verdict"]
        return g["gates_pass"], g["gates_total"]
    for k in ("B-14b", "B-15", "B-17", "B-18", "B-19"):
        p, t = gp(d[k])
        if p != t:
            errs.append("%s 게이트 %d/%d — 전건 통과 아님" % (k, p, t))
    p14, t14 = gp(d["B-14"])
    if p14 == t14:
        errs.append("B-14 가 전건 통과다 — 이 스크립트의 예외 주석(G-1 FAIL)이 낡았다")
    # (3) 라벨 일치
    expect = {
        "B-14/h2":  "REPRESENTATION-ADEQUATE-NOMINAL",
        "B-14b":    "OBS-MODEL-DOES-NOT-MEDIATE",
        "B-15":     "HEADLINE-SURVIVES-CORRECT-SENSOR",
        "B-17":     "STRATUM-SURVIVES",
        "B-18":     "NEITHER-SUFFICIENT",
        "B-19":     "CONTINUOUS-K-EARNS-SPARSE",
    }
    got = {
        "B-14/h2":  d["B-14"]["h2_label"],
        "B-14b":    d["B-14b"]["h1_label"],
        "B-15":     d["B-15"]["label"],
        "B-17":     d["B-17"]["label"],
        "B-18":     d["B-18"]["label"],
        "B-19":     d["B-19"]["label"],
    }
    for k, e in expect.items():
        if got[k] != e:
            errs.append("라벨 불일치 %s: 기대 %r · 실제 %r" % (k, e, got[k]))
    # B-17 짝 가족이 전부 널이라는 이 표의 주장
    b17 = d["B-17"]
    fam = b17["rules"]["FAMILY"]
    nom = b17["rules"]["NOMINAL"]
    for a in fam:
        if b17["paired_vs_const"][a]["better"] >= nom:
            errs.append("B-17 짝 %s 가 명목 문턱 %d 이상 — '전 팔 널' 이 거짓" % (a, nom))
    # B-18 네 모서리 전부 문턱 미달
    nom18 = d["B-18"]["thresholds"]["nominal"]
    for a, c in d["B-18"]["primary"].items():
        if c["beats_zero"] >= nom18:
            errs.append("B-18 모서리 %s 가 문턱 %d 이상 — '전 모서리 미달' 이 거짓" % (a, nom18))
    return errs


def main():
    d = load()
    errs = check(d)
    if "--check" in sys.argv:
        for e in errs:
            print("FAIL:", e)
        print("OK — 6 verdict · 게이트 · 라벨 · 두 전칭 주장 전건 확인" if not errs
              else "FAILED (%d)" % len(errs))
        return 1 if errs else 0
    if errs:
        for e in errs:
            print("⛔ SELF-CHECK FAIL:", e)
        print()

    print("# 소거표 — 실측 12 레코드의 음성은 무엇으로 설명되는가 (2026-08-26)")
    print()
    print("⛔**소거법이다.** 각 행은 *\"이 후보가 문턱을 넘지 못했다\"* 이지 "
          "*\"이 후보가 원인이 아님이 증명됐다\"* 가 아니다 — 문턱 미달은 부재의 증명이 아니다.")
    print()
    print("| 후보 | 판정 | 근거 (verdict 에서 읽음) |")
    print("|---|---|---|")
    for b in BUILDERS:
        name, ver, ev, note, _ = b(d)
        print("| %s | %s | %s<br>%s |" % (name, ver, ev, note))
    print("| ★**남은 것** | **측정된 것 밖** | 아래 목록 — ⛔**어느 것도 측정되지 않았다** |")
    print()
    print("### ★남은 후보 (⛔전부 미측정 · 소거의 끝에는 후보 목록이 아니라 **공백**이 남는다)")
    print()
    print("| 남은 후보 | 상태 |")
    print("|---|---|")
    for a, b_ in REMAINING:
        print("| %s | %s |" % (a, b_))
    print()
    # 합성 층 — 소거표 밖이지만 같은 날 자료
    b19, b16p = d["B-19"], None
    pr = b19["primary"]
    ds = b19["dose"]
    print("### ⚠소거표 **밖** — 같은 날의 합성 층 (⛔실측 행과 합산·교차 층화 금지)")
    print()
    print("- `B-19` **%s** + `%s`: 희소 층에서 `SHIP-NOW < KGRID1` **%d/%d**, p=%.5f, 상대 중앙 %.2f%% · "
          "용량-반응 %d/%d(동점 %d)"
          % (b19["label"], b19["dose_label"], pr["better"], pr["n"], pr["p_one_sided"],
             pr["rel_median_pct"], ds["better"], ds["n"], ds["n_tie"]))
    print("  - 음성대조 `SHIP-NOW<ZERO` **%d/%d** (p=%.6f) · `KGRID1<ZERO` **%d/%d** (p=%.4f = 문턱 정확히)"
          % (b19["neg_ship_vs_zero"]["better"], b19["neg_ship_vs_zero"]["n"],
             b19["neg_ship_vs_zero"]["p_one_sided"],
             b19["neg_kgrid1_vs_zero"]["better"], b19["neg_kgrid1_vs_zero"]["n"],
             b19["neg_kgrid1_vs_zero"]["p_one_sided"]))
    print("  - ⚠표본은 `mesh_verdict` 3-분할이다: %s ⟹ ⛔\"HOS M=3\" 로 단일 서술 금지"
          % " · ".join("%s %d" % (k, v) for k, v in
                       sorted(b19["mesh_verdict_counts"].items())))
    print()
    print("---")
    print("생성: `python3 tmp/STATUS_2026-08-26/elimination_table.py` (런 0회 · 동결 verdict 재집계)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
