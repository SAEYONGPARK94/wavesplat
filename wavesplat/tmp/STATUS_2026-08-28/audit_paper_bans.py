#!/usr/bin/env python3
"""paper/ 인용금지 전수 검사 — `audit_thesis_bans.py` 를 `paper/` 에 겨눈 변형.

⛔런 0회(추정기 0회). HDF5 는 **attrs/timestamps 만** 읽는다.

thesis 판과 다른 점 세 가지:
  (1) 대상이 `.tex` 하나가 아니라 **인쇄되는 모든 것**이다 —
      `.tex` 본문(주석 제외) · `.txt`(Elsevier highlights 는 그 자체가 제출물) ·
      `.md`(제출 패키지 문서) · ★**`.pdf`(조판 산출물 — 주석으로 가릴 수단이 없다)**.
  (2) thesis 판은 "단서 동반 의무" 를 검사하지만, `paper/` 는 **동결 원고**이므로
      단서를 붙일 수 없다 ⟹ **존재 자체를 센다**.
  (3) 각 패턴의 근거를 **손으로 옮기지 않고** 기계로 재도출한다(§0).

사용: python3 tmp/STATUS_2026-08-28/audit_paper_bans.py
종료코드: 인쇄되는(=주석 밖) ⚠[2026-09-01 개정] 이제 **양방향 래칫**이다 — 늘어도 실패, 줄어도 실패(기준선을 의도적으로 내리라는 뜻), 대상이 0개여도 실패.
"""
import re, sys, os, glob, hashlib, subprocess

ROOT = "/home/vexit/wavesplat/wavesplat"
PAPER = os.path.join(ROOT, "paper")

# =============================================================================
# §0  근거의 기계 재도출 — "프레임 불일치" 를 이 스크립트가 직접 잰다
#     (금지목록의 사유 (i) 는 results/b1_full_comparison.json 에 대해 쓰였다.
#      paper/ 의 현행 헤드라인 0.115/0.332 는 **다른 코퍼스**
#      = data_finemesh_2026-07-23 + tmp/rerun_2026-07-25 산출이므로,
#      그 코퍼스에서 같은 결함이 성립하는지 여기서 직접 확인한다.)
# =============================================================================
def measure_frame_mismatch():
    import numpy as np, h5py
    D = os.path.join(ROOT, "data_finemesh_2026-07-23")
    if not os.path.isdir(D):
        return None
    tot, trk = [], []
    for e in sorted(f for f in os.listdir(D) if f.endswith(".h5")):
        with h5py.File(os.path.join(D, e), "r") as f:
            ts = np.asarray(f["frames/timestamps"][:])
            tp = float(f["gt_params"].attrs["Tp_true"])
        tot.append(len(ts))                       # 기준선 팔: `for frame in loader:` = 전 프레임
        trk.append(int((ts > 2.0 * tp).sum()))    # WS 팔: run.py:226/311 = t > 2*Tp 만 채점
    return dict(n=len(tot), base_min=min(tot), base_max=max(tot),
                ws_min=min(trk), ws_max=max(trk),
                ws_fewer=sum(1 for a, b in zip(trk, tot) if b > a))

FM = measure_frame_mismatch()
print("# ===== §0 근거 재도출 (런 0회, HDF5 attrs/timestamps 만) =====")
if FM:
    print(f"#   코퍼스 data_finemesh_2026-07-23 = {FM['n']} 에피소드")
    print(f"#   기준선 팔 채점 프레임 = {FM['base_min']}-{FM['base_max']}   "
          f"(batchA_baselines.py:65,87 `for frame in loader:`)")
    print(f"#   WaveSplat 팔 채점 프레임 = {FM['ws_min']}-{FM['ws_max']}  "
          f"(run.py:226 get_init_window(tp_factor=2.0) -> :311 tracking_frames)")
    print(f"#   WS 가 더 적은 에피소드 = {FM['ws_fewer']}/{FM['n']}")
    MISMATCH = (FM["ws_fewer"] == FM["n"])
    print(f"#   ⟹ 프레임 불일치 {'성립' if MISMATCH else '불성립'} "
          f"(= FIX4_ban_reason.md:33 의 사유 (i) 와 같은 구조)")
else:
    MISMATCH = None
    print("#   ⚠코퍼스 부재 — 근거 재도출 불가")
print()

# =============================================================================
# §1  대상 파일 수집 + "인쇄되는가" 판정
# =============================================================================
def tex_body(path):
    """LaTeX 주석을 뺀 줄만 돌려준다 (= 실제로 인쇄되는 것)."""
    out = []
    for i, l in enumerate(open(path, encoding="utf-8", errors="replace").read().splitlines(), 1):
        if l.lstrip().startswith("%"):
            continue
        b = re.sub(r"(?<!\\)%.*$", "", l)
        if b.strip():
            out.append((i, b))
    return out

def plain_body(path):
    return [(i, l) for i, l in
            enumerate(open(path, encoding="utf-8", errors="replace").read().splitlines(), 1)
            if l.strip()]

def pdf_body(path):
    try:
        t = subprocess.run(["pdftotext", "-q", path, "-"], capture_output=True,
                           text=True, timeout=120).stdout
    except Exception:
        return []
    return [(i, l) for i, l in enumerate(t.splitlines(), 1) if l.strip()]

TARGETS = []   # (relpath, kind, [(lineno, text)])
for p in sorted(glob.glob(os.path.join(PAPER, "**", "*"), recursive=True)):
    if not os.path.isfile(p):
        continue
    rel = os.path.relpath(p, ROOT)
    ext = os.path.splitext(p)[1].lower()
    if ext == ".tex":
        TARGETS.append((rel, "tex-body", tex_body(p)))
    elif ext in (".txt", ".md"):
        TARGETS.append((rel, "text", plain_body(p)))
    elif ext == ".pdf":
        TARGETS.append((rel, "pdf", pdf_body(p)))
    elif ext == ".bbl":
        TARGETS.append((rel, "tex-body", tex_body(p)))

# =============================================================================
# §2  금지 패턴 — 각 항목에 **근거 파일:줄**을 동반한다
# =============================================================================
# kind: "num" = 원문 그대로만 대조(정규화 금지, 쉼표 제거가 `{0, 0.02}`->`0.020` 오탐을 만든다)
#       "phr" = LaTeX 정규화 후 대조
BANS = [
 ("WS 헤드라인 nRMSE 0.115/0.1145/0.114",
  r"0\.1145|0\.115|0\.114(?![0-9])",
  "§0 프레임 불일치(WS 120-220 vs 기준선 301, 100/100) = FIX4_ban_reason.md:33 사유(i) 동형"),
 ("ETKF 분모 0.332/0.3321",
  r"0\.332|0\.3321", "동상"),
 ("개선률 65.5%",
  r"65\.5", "동상 (두 불일치 팔의 비)"),
 ("LETKF 0.228 / 49.7%",
  r"0\.228|49\.7", "동상"),
 ("GPR-1 0.263",
  r"0\.263", "동상"),
 ("12-ep 0.182 / 0.211 vs 0.358 / 41.1%",
  r"0\.182|0\.211|0\.358|41\.1", "FIX4_ban_reason.md '유지된 사유' = 구버전 12-ep"),
 ("승패비 84/100 · 85/100 · 66/100",
  r"8[45]/100|66/100", "FIX4_ban_reason.md:33 (프레임 불일치 ⟹ 승패비도 금지)"),
 ("구 헤드라인 0.223/0.319/30.1%",
  r"0\.223|0\.319|30\.1", "reference_citation_bans.md (08-13 3라운드)"),
 ("지연 13.4 / 132.6 / 11.0 / 14.5 ms",
  r"13\.4|132\.6|11\.0~?|14\.5", "reference_citation_bans.md (08-05),(08-13 3라운드 사유 iii)"),
 ("속도배수 ~10x / 8.1x / 8.5x / 9.92x",
  r"10\\?\$?\\times|10\s*×|~?10x|8\.1\\?\$?\\times|8\.1x|8\.5x|9\.92",
  "동상 = 프로세스 간 비교"),
 ("CPU Intel i7-12700H",
  r"i7-?12700H|Intel(?!lig)", "reference_citation_bans.md (08-05) 실제는 AMD Ryzen 7 7800X3D"),
 ("Wilcoxon p=2.5e-16 / 1.6e-7 / Cohen d=1.30/0.65/1.16",
  r"2\.5\\!?\\?times.{0,12}10.{0,4}\{?-?16|1\.6\\!?\\?times.{0,12}10.{0,4}\{?-?7|"
  r"2\.5e-?16|1\.6e-?7|d\s*\{?=\}?\s*1\.30|d\s*\{?=\}?\s*0\.65|d=1\.16",
  "불일치 팔 위에서 계산 + FIX4 '그대로 재현 안 됨'"),
 ("Skill Score +0.538 및 CI [+0.42,+0.65]",
  r"\+?0\.538|\+0\.42|\+0\.65", "동일 코퍼스 산출 ⟹ 결함 상속"),
 ("희소 층 극단값 0.020 / SS +0.988 / 0.016 / 0.029",
  r"0\.020(?![0-9])|0\.988|0\.016(?![0-9])|0\.029(?![0-9])",
  "MEMORY 08-15: 희소 -93% 는 표현 아티팩트(비선형화 시 3.6배 열화)"),
 ("'희소 해상 해결' 류 실용 주장",
  r"희소해상.{0,6}해결|sparseseas?.{0,12}solved|essentiallysolved",
  "reference_citation_bans.md (08-03)"),
 ("'all episodes / every density 에서 우월'",
  r"onall(episodes|densities)|everyepisode|alldensities|outperform.{0,40}all",
  "paper/wavesplat.tex:17-38 배너 2항"),
 ("제목·본문의 'Shipboard LiDAR'",
  r"[Ss]hipboardLiDAR|shipboardlidar", "thesis 제목 철회(실증 0건) = 배너 1항"),
 ("'All phases complete'",
  r"Allphasescomplete|모든단계완료", "CLAUDE.md"),
 ("US 12,054,225 를 파도장 특허로",
  r"12,?054,?225", "MEMORY 08-19 = Candela 수중익선 제어 특허"),
]

NUMERIC = {  # 이 항목들은 원문 대조만 한다
 "WS 헤드라인 nRMSE 0.115/0.1145/0.114","ETKF 분모 0.332/0.3321","개선률 65.5%",
 "LETKF 0.228 / 49.7%","GPR-1 0.263","12-ep 0.182 / 0.211 vs 0.358 / 41.1%",
 "승패비 84/100 · 85/100 · 66/100","구 헤드라인 0.223/0.319/30.1%",
 "지연 13.4 / 132.6 / 11.0 / 14.5 ms","속도배수 ~10x / 8.1x / 8.5x / 9.92x",
 "Wilcoxon p=2.5e-16 / 1.6e-7 / Cohen d=1.30/0.65/1.16",
 "Skill Score +0.538 및 CI [+0.42,+0.65]",
 "희소 층 극단값 0.020 / SS +0.988 / 0.016 / 0.029",
}

# =============================================================================
# §2b  ★오탐 대장 (2026-08-28 손 분류 — 전건 원문 확인)
#   본 검사기는 "인쇄되는 곳에 금지된 수/문구가 있는가" 만 본다.  두 부류가 오탐이다:
#   SELF = 그 줄 자체가 **철회문**이다(금지 수를 "쓰지 마라" 고 적으면서 인쇄한다).
#   COLL = 자릿수만 같고 **다른 양**이다(예: ETKF 가산 인플레이션 σ=0.020,
#          "14.5% 성능 저하", DOI 조각 `11.008`, WS Skill Score `+0.332`).
#   ⚠좌표(파일:줄) 단위 대장이므로, 한 줄이 오탐 패턴과 정탐 패턴에 동시에 걸리면
#     보수적으로 오탐으로 샌다(= 오탐률을 **과대** 보고한다).
FALSE_POSITIVE = {
 ("paper/PAPER_IMPROVEMENTS.md", n) for n in (4, 7, 9, 15, 19)
} | {
 ("paper/submission/graphical_abstract.md", n) for n in (4, 5, 52, 53, 60)
} | {
 ("paper/wavesplat.tex", n) for n in (154, 1151, 1152, 1153, 1592, 1756, 1801,
                                      1887, 1993, 2083, 2407, 2604, 2616, 2843)
} | {
 ("paper/wavesplat.pdf", n) for n in (2041, 2053, 2065, 2776, 3152, 3196,
                                      4178, 5045, 6271, 6456)
}

# ★[2026-09-02] **내용 키 오탐 대장.**
#   ⛔설계 약점: 위 좌표 대장은 `(파일, 줄번호)` 다. 조판 산출 PDF 는 **재조판할 때마다
#   줄번호가 밀리므로** 그 항목들이 조용히 낡고, 래칫이 내용과 무관하게 위아래로 흔들린다
#   (2026-09-02 에 그림 하나를 추가했더니 26 -> 28 로 "악화" 를 찍었는데,
#   실제 원인은 축 눈금 `0.020` 과 밀린 좌표였다).
#   ⟹ 생성 산출물에는 **줄번호가 아니라 그 줄의 내용**으로 키를 잡는다.
#   ⚠엄격하게: 정규화된 **전체 줄**이 일치해야 하고, 사유를 반드시 적는다.
FALSE_POSITIVE_TEXT = {
    # COLL --- 오차-거리 패널의 y축 눈금. 금지된 "희소 층 극단값 0.020" 과
    #          자릿수만 같고 **다른 양**이다(그림에는 팔간 비교가 없다).
    "0.020": "COLL: fig_qualitative 의 축 눈금 (RMS error [m])",
    # COLL --- Delta k_min 사전등록 스윕의 5성분 셀 손해. 희소 극단값이 아니다.
    "on0/5seeds(+0.029nRMSE),becauseasmallerradiusletsthegreedysearch":
        "COLL: Delta k_min 스윕의 셀 손해",
    "caseon0/5seeds($+0.029$~nRMSE),becauseasmallerradiusletsthegreedy":
        "COLL: Delta k_min 스윕의 셀 손해 (tex)",
}


def is_fp(rel, ln, txt):
    if (rel, ln) in FALSE_POSITIVE:
        return True
    return norm(txt).strip() in FALSE_POSITIVE_TEXT


def norm(s):
    s = re.sub(r"\\(?:textbf|emph|texttt|mathbf|textit|text)\s*\{", "{", s)
    s = re.sub(r"[{}$\\,]", "", s)
    return re.sub(r"\s+", "", s)

print("# ===== §2 전수 결과 (인쇄되는 줄만) =====")
total = 0
n_fp = 0
n_raw = 0
per_file = {}
for name, pat, basis in BANS:
    rx = re.compile(pat, re.I)
    hits = []
    for rel, kind, body in TARGETS:
        for ln, txt in body:
            probe = txt if name in NUMERIC else norm(txt)
            if rx.search(probe):
                hits.append((rel, kind, ln, txt))
    tp = [h for h in hits if not is_fp(h[0], h[2], h[3])]
    fp = [h for h in hits if is_fp(h[0], h[2], h[3])]
    total += len(tp); n_fp += len(fp); n_raw += len(hits)
    tag = "⛔HIT" if tp else ("  fp " if fp else "  ok ")
    print(f"{tag} {name}: 정탐 {len(tp)}건 / 오탐 {len(fp)}건   [근거: {basis}]")
    hits = tp
    LIM = 10**9 if os.environ.get("ALL") else 8
    for rel, kind, ln, txt in hits[:LIM]:
        per_file.setdefault(rel, 0)
        per_file[rel] += 1
        print(f"        {rel}:{ln} ({kind})  {txt.strip()[:110]}")
    for rel, kind, ln, txt in hits[LIM:]:
        per_file.setdefault(rel, 0); per_file[rel] += 1
    if len(hits) > LIM:
        print(f"        ... 외 {len(hits)-LIM}건")

# =============================================================================
# §3  ★그림 안의 수 — `audit_thesis_bans.py` 가 **구조적으로 못 보는** 곳 [자문]
#     학위논문은 `paper/figures/` 의 8개 그림을 **바이트 동일 사본**으로 쓴다
#     (`thesis/figures_ws/`). 그 그림들이 금지된 수를 **이미지 안에** 인쇄한다.
#     텍스트 검사기는 `\includegraphics` 줄만 보므로 이 부류를 영원히 놓친다.
# =============================================================================
print("\n# ===== §3 학위논문 그림 안의 금지 수 [자문 — 이 검사기만 본다] =====")
FIGNUM = re.compile(r"0\.115|0\.332|0\.319|0\.223|65\.5|0\.211|0\.358|"
                    r"13\.4|132\.6|0\.228|0\.263|8[45]/100")
n_fig = 0
for _f in sorted(glob.glob(os.path.join(ROOT, "thesis", "figures_*", "*.pdf"))):
    for ln, txt in pdf_body(_f):
        if FIGNUM.search(txt):
            n_fig += 1
            print(f"   ⚠{os.path.relpath(_f, ROOT)}:{ln}  {txt.strip()[:80]}")
print(f"#   그림 내부 히트 {n_fig}건 "
      f"(⚠판정에 넣지 않는다 — 학위논문 처분은 이 검사기의 범위 밖이다)")

# =============================================================================
# §4  ★래칫 — 오늘 수를 동결하고 **늘어나면** 실패한다
#     ⛔`paper/` 는 동결 원고이므로 "히트 0" 을 요구하면 방벽이 영구히 빨간불이 되고
#       사람이 무시하기 시작한다(`run_all.sh` 가 스스로 경고하는 실패 부류).
#     ⟹ 요구는 **"오늘보다 나빠지지 않는다"** 이고, 처분이 집행되면 기준선을 내린다.
# =============================================================================
# ★2026-09-01 (P8): 188 -> 69 로 **의도적으로** 내린다.
#   양방향 래칫이 "개선이 기록되지 않았다" 로 막아섰고, 그 요구에 따라 갱신한다.
#   ⟹ 이 줄의 갱신 자체가 "처분이 집행됐다" 는 기록이다(그러라고 만든 기계다).
#
#   무엇이 줄었나:
#     · paper/wavesplat.tex  84 -> 19  (초록·하이라이트·제목 재작성 +
#       100-ep 레거시 Results 5절·Discussion 5절을 tmp/PAPER_LEGACY_2026-09-01/ 로 **이동**)
#     · paper/wavesplat.pdf  77 -> 0   (조판 갱신 = 56쪽, 미정의 참조 0)
#   ⚠남은 69 는 대부분 **표지·커버레터·그림 PDF·bbl** 이다(본문 tex 는 19).
#   ⛔"0" 이 목표가 아니다 --- paper/ 에는 철회 이력을 보존하는 배너와 주석이 있어야 한다.
#   ★2차 (같은 날): 69 -> 55. 커버레터 두 개의 제목을 갱신하고(‘Shipboard’ 철회 반영)
#   두 PDF 를 재조판했다(각 6 -> 0). ⚠남은 55 는 대부분 **조판 산출 PDF·그림·
#   PAPER_IMPROVEMENTS.md** 이고 본문 tex 는 19 다.
#   3차 (같은 날): 55 -> 53. 적대적 감사(BLOCKER 5)를 수리하며 전칭 자구와
#   범위 문장을 고쳤다. ⚠남은 53 은 대부분 조판 PDF·그림·PAPER_IMPROVEMENTS.md 이고,
#   본문의 'shipboard LiDAR' 잔존 5곳은 **센서 부류·관련연구·한계·향후과제** 문맥이라
#   "선박 탑재 실측을 했다" 는 주장이 아니다(:487 에 범위 문장을 명시했다).
#   4차 (같은 날): 53 -> 51. 2차 적대적 감사(BLOCKER 3·MAJOR 6) 수리 —
#   철회 코퍼스의 **팔간 주장** 2건(Sea state trend · fig:qualitative)을 추가 이동,
#   잔존 전칭 2건 제거, 하이라이트를 의무공시 없는 형태로 재작성.
#   ★5차 (2026-09-02, P9): 51 -> 30. **표.그림 처분**을 집행했다 --- 철회된 코퍼스에서
#   나온 절제.강건성 절 다섯과 표 넷.그림 넷을 tmp/PAPER_LEGACY_2026-09-01/ 로 **옮기고**,
#   대체가 있는 것은 동결 verdict 원자로 만든 Table~\ref{tab:earns}/{tab:boundary} 로
#   교체, 대체가 없는 것(5지표 패널.buoy 층화.12ep 표.강기준선 스윕.스펙트럼 정확도.
#   절대 지연)은 **부재로 남겼다**. 논문 50쪽(58 -> 50).
#   ⚠남은 30 의 대부분은 여전히 **조판 산출 PDF.그림.PAPER_IMPROVEMENTS.md** 이고
#   본문 tex 는 8 인데 그중 4 는 이미 문서화된 문맥이다:
#     · :182/:347/:390 'shipboard LiDAR' = 센서 부류.관련연구.한계 문맥(3차 주석 참조)
#     · :1833 '+0.029 nRMSE' = **Delta k_min 사전등록 스윕의 5성분 셀 손해**이지
#       금지된 희소 극단값이 아니다 --- 우연히 같은 문자열이다(literal 오탐과 같은 부류).
#   ★6차 (2026-09-02, P9 후속): 30 -> 26. 4차 적대적 감사(인쇄면 통독, BLOCKER 9)가
#   **철회 코퍼스의 재인용 두 곳**을 잡았다: §4.4 가 *"수치를 인용하지 않는다"* 라 적은
#   바로 다음 문단에서 ETKF 스윕 0.358/1.187/3.3배 를 인용했고, §6.6 이 §5.7 의
#   *"buoy 층화는 다시 적지 않는다"* 약속을 어기고 16.1/35.3/40/11.5/79.7 을 전부 재진술했다.
#   둘 다 철회. ⛔같은 감사가 §6.6 의 **금지된 팔간 주장**(*"LETKF is more often better"*)과
#   철회 코퍼스를 **긍정 근거**로 쓴 문단도 잡았다(래칫이 세는 대상은 아니었다).
BASELINE = 26

print(f"\n# 원시 히트 {n_raw}건 = 정탐 {total} + 오탐 {n_fp}")
print(f"# ★오탐률 = {n_fp}/{n_raw} = {100.0*n_fp/n_raw:.1f}%" if n_raw else "")
print("# 파일별:")
for rel, n in sorted(per_file.items(), key=lambda kv: -kv[1]):
    print(f"#   {n:>4}  {rel}")
print(f"# 래칫 기준선 {BASELINE} → 오늘 {total} "
      f"({'악화' if total > BASELINE else ('동일' if total == BASELINE else '개선')})")

# ─────────────────────────────────────────────────────────────────────────────
# ★2026-09-01 수리 — **래칫이 「수리」와 「삭제」를 구별하지 못했다.**
#
# 이전 판정은 `sys.exit(1 if total > BASELINE else 0)` 하나였다. 그래서:
#   · `paper/` 가 통째로 사라지면 glob → [] → total=0 → "개선" → **exit 0**
#   · `run_all.sh` 는 그 위에 **"✅검사기 전부 통과"** 를 찍는다
#   ⟹ 방벽이 **처분을 목격하지 못한다.** 미래의 독자는 *"처분이 집행됐다"* 와
#      *"디렉터리가 사고로 사라졌다"* 를 구별할 수 없다.
#   ⟹ 저장소가 반복해서 대가를 치른 **「검사기가 검사하지 않는다」** 부류다
#      (선례: `W-8 G-4` 공허 통과 · `B-26 G-0` 소급 차단 · `T2 G-7b` 구조적 실패).
#
# 수리 = 래칫을 **양방향**으로 만든다. 개선도 «자동 통과»가 아니라 «기준선 갱신 요구»다.
# ⛔`:246` 주석이 이미 *"처분이 집행되면 기준선을 내린다"* 라 적었는데 **강제하는 것이 없었다.**
# ─────────────────────────────────────────────────────────────────────────────
# ★[2026-09-01] PDF 추출이 조용히 죽는 경로를 먼저 막는다(독립 감사 지적).
#   `pdf_body()` 는 `except Exception: return []` 라, `pdftotext` 가 없으면 PDF 가
#   **0줄**이 되고 `TARGETS` 는 여전히 비지 않는다. 오늘 정탐 188 중 PDF 유래가 약 95건이라
#   그 기계에서는 total≈93 < 188 → "기준선을 93 으로 내려라" 라는 **적극적으로 해로운 지시**가
#   나온다(§12 가 존재하는 이유가 "조판 PDF 는 주석으로 가릴 수 없다" 인데 그것을 무력화한다).
_empty_pdf = [rel for rel, kind, body in TARGETS if kind == 'pdf' and not body]
if _empty_pdf:
    print("\n⛔PDF 본문 추출이 0줄이다 — `pdftotext`(poppler) 가 없을 가능성이 높다.")
    print(f"   대상 {len(_empty_pdf)}건: {_empty_pdf[:3]}")
    print("   ⛔이 상태에서 기준선을 내리지 마라 — 히트가 준 것이 아니라 **안 본 것**이다.")
    sys.exit(1)

if not TARGETS:
    print("\n⛔검사 대상이 0개다 — `paper/` 가 비었거나 경로가 틀렸다.")
    print("   처분을 집행했다면 이 스크립트의 BASELINE 을 0 으로 내리고 이 단언을 갱신하라.")
    print(f"   (기대 경로: {PAPER})")
    sys.exit(1)

if total > BASELINE:
    print(f"\n⛔악화 — 기준선 {BASELINE} 을 넘었다.")
    sys.exit(1)

if total < BASELINE:
    print(f"\n⛔개선이 기록되지 않았다 — 기준선 {BASELINE} → 실측 {total}.")
    print(f"   수리했다면 이 파일의 `BASELINE = {total}` 로 **의도적으로** 내려라.")
    print("   ⚠이 실패는 「나빠졌다」가 아니라 「좋아진 것을 아무도 적지 않았다」는 뜻이다.")
    print("   ⛔먼저 확인하라: 추출이 죽지 않았는가(PDF·인코딩), 파일이 사라지지 않았는가.")
    sys.exit(1)

sys.exit(0)
