#!/usr/bin/env python3
"""★**표준 사실 검사기** --- 코드/코퍼스가 정한 값과 다른 값이 문서에 남아 있는가.

⛔왜(2026-09-02):
이 저장소의 **누적 1위 결함**은 *"정정이 한 곳에서 멈춘다"*(N-1)이다. 오늘 하루에만
여섯 번 나왔고, 그중 셋은 **실험 조건 자체**였다:

- 해상상태를 **네 표**가 틀리게 인쇄(같은 장 50쪽 앞은 옳았다)
- 채점 격자 면적이 **2.5배** 과대
- 프레임당 관측 수를 **네 자리**가 틀리게(내가 첫 수리를 한 곳에서 멈췄다)

⟹ 사실을 **코드/코퍼스에서 읽고**, 알려진 **틀린 값**이 두 문서에 남아 있으면 실패한다.
⚠이 검사기는 *"옳게 적혔는가"* 가 아니라 *"틀리게 적힌 것이 남았는가"* 를 본다 ---
후자는 기계로 확실하고, 전자는 아니다.

예외는 **사유와 함께** `ALLOW` 에 적는다(줄 내용 키 --- 재조판에 안 낡는다).
"""
import json
import os
import re
import sys

import h5py

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
from wavesim.pipeline import SEA_STATE                              # noqa: E402

DOCS = ['paper/waveprior.tex', 'thesis/thesis_main.tex']

# ── 코퍼스에서 읽는 사실 ──────────────────────────────────────────────
def corpus_facts():
    p = os.path.join(ROOT, 'data', 'episode_BF4_5comp_seed0_uniform.h5')
    with h5py.File(p, 'r') as f:
        gx, gy = f['frames/gt_grid_x'][:], f['frames/gt_grid_y'][:]
    return dict(x0=gx.min(), x1=gx.max(), y0=gy.min(), y1=gy.max(),
                nx=len(gx), ny=len(gy))


# ── (이름, 틀린 값 정규식, 무엇이 옳은가) ─────────────────────────────
def rules():
    # ⛔[2026-09-02] 첫 판은 *"BF<n> 뒤에 SEA_STATE 와 다른 두 소수가 오면 실패"* 였다.
    #   ⟹ **26건 전부 오탐**이었다(결과 표의 nRMSE 열을 Hs/Tp 로 오인).
    #   ★검사기가 늑대를 외치면 사람이 무시한다 --- 오늘 이미 두 번 배웠다.
    #   ⟹ **알려진 틀린 값의 인접 쌍**만 본다. ⚠한계를 명시한다: 이 규칙은
    #     *"과거의 틀린 값이 되살아나는 것"* 을 막지 *"새로운 틀린 값"* 을 잡지 못한다.
    #     새 값을 막는 것은 `gen_seastate_table.py`(코드에서 생성 + 코퍼스 대조)의 몫이다.
    WRONG_PAIRS = [('BF3', '0.88', '5.0'), ('BF3', '0.88', '5.5'),
                   ('BF4', '1.88', '7.5'), ('BF4', '1.88', '7.0')]
    R = []
    for bf, whs, wtp in WRONG_PAIRS:
        p = SEA_STATE[bf]
        R.append((f'{bf} 옛 Hs/Tp',
                  rf'{re.escape(whs)}\s*(?:&|\\,m)[^\n]{{0,20}}{re.escape(wtp)}',
                  f'{bf} = (Hs {p["Hs"]}, Tp {p["Tp"]}) — wavesim/pipeline.py SEA_STATE'))
    R.append(('옛 BF3 Hs 단독', r'0\.88\\,m', 'BF3 Hs = 0.6 m'))
    c = corpus_facts()
    R += [
        ('채점 격자 x 상한', r'x\s*\\in\s*\[[^\]]*390', f'x 상한 = {c["x1"]:g} (코퍼스)'),
        ('채점 격자 y 범위', r'y\s*\\in\s*\[-150,\s*150\]', f'y = [{c["y0"]:g},{c["y1"]:g}]'),
        ('격자 간격', r'격자 간격 5\\,m', 'dx≈3.04 · dy≈3.39 (코퍼스)'),
        ('BF4 파장', r'87\.7', 'BF4 λ_p = 76.5 m (Tp=7.0, 심해)'),
        ('LiDAR 높이', r'해수면 위 4\\,m|z_L\) = \(3\.0, 0\.0, 4\.0\)',
         'height = 5.0 m, 선수 오프셋·pitch 없음 (configs/default.yaml)'),
        ('LiDAR 수직 범위', r'-12\.5\^\\circ \+ \\Delta\\theta',
         'elev = linspace(-25, 0, 152) (configs/default.yaml)'),
    ]
    # ★★[2026-09-02] **철회된 자구**가 되살아나는 것을 막는다. 이 검사기는 **두 문서를
    #   함께** 보므로(`DOCS`), 「논문은 고쳐지고 학위논문은 안 고쳐진다」 부류를
    #   구조적으로 잡는다 --- 오늘 그 부류가 네 번 나왔다.
    #   ⚠본문만 본다(주석은 이 자구를 **인용해서** 설명하므로 잡히면 안 된다).
    R += [
        # `wavesim/raydrop_threshold.py:10-11` 이 반증: 정반사 게이트가 이 기하에서
        # 발화하지 않고 상수-P 로 퇴화한다 ⟹ `slope` 는 "물리적으로 옳은" 모형이 아니다.
        ('raydrop 물리 전제 (철회)',
         r'physically correct[^.\n]{0,40}(?:raydrop|sensor|slope)'
         r'|물리적으로 옳은[^.\n]{0,30}(?:센서|모형|쪽)',
         'slope 는 기하 가중이 **없는** 모형이다 (raydrop_threshold.py:10-11)'),
        # `verdict_b26.json`: SS-LOWH 도 -17.02% 로 5% 기술 바닥을 통과한다 ⟹ 두 칸이다.
        ('B-26 통과 칸 수 (철회)',
         r'헤드라인 칸만|only the headline cell',
         '긴 주기 **두 칸**이 통과한다 (SS-BF4 -16.17% · SS-LOWH -17.02%)'),
        # ★★[2026-09-05] `T11 AMENDMENT_2 §6` 이 철회한 **초판 읽기**. k_p 기준으로 그
        #   장면들은 학습 밴드 83백분위 **안**이므로 *"우리 대역/학습 사전의 파장 밴드"* 는
        #   과특정이다. 정본 = "배포 추정기의 **절대 스케일** 설계 범위".
        #   ⚠본문의 ⛔*"…로 읽지 않는다"* 형태는 잡히면 안 되므로 **철회된 단정 형태**만 본다.
        ('T11 초판 읽기 (철회)',
         r'실패는 우리 대역|failure is our band',
         '정본 = "배포 추정기의 절대 스케일 설계 범위" (T11 AMENDMENT_2 §6)'),
        # ★★[2026-09-05] `verdict_t13.json` 이 *"수심은 원인이 아니다"* 읽기를 금지한다.
        #   절 제목이 그 읽기가 되는 형태(*"무엇 때문이 아닌가"*)만 본다 --- 본문의
        #   ⛔인용 형태는 잡지 않는다.
        ('T13 「원인이 아니다」 읽기 (금지)',
         r'무엇 \\emph\{때문이 아닌가\}|무엇 \\textbf\{때문이 아닌가\}',
         'FIX-UNDETECTED 는 검정력의 진술이다 (MDE q80=0.6795 병기) — verdict_t13.json'),
        # `T8` 이 2026-09-02 에 측정했고 FAIL 했다(6.00 -> 18.30 ms).
        ('학습 경로 프레임 예산 (철회)',
         r'프레임 예산 미측정|frame budget[^.\n]{0,30}(?:not|never) (?:been )?measured',
         'T8 이 측정했고 3.05배로 FAIL 했다 (verdict_t8.json: frames.pass=false)'),
    ]
    # ★[2026-09-02] 동결 verdict 의 **게이트 수**를 문서가 손으로 적으면 낡는다.
    #   실제로 낡았다: `B-26` 이 `28/29` 인데 학위논문 **여덟 곳**이 `29/29` 라 적었다
    #   (실패 게이트 = `G-0 시드 원장 충돌`, **후속** `M-1` 이 만든 것; 측정 게이트는 전건 통과).
    #   ⟹ 옛 값을 규칙으로 박고, 옳은 값은 **동결 파일에서 읽는다**.
    for tag, rel in (('B-26', 'B26_SEASTATE_2026-08-27/verdict_b26.json'),):
        vp = os.path.join(ROOT, 'tmp', rel)
        if not os.path.exists(vp):
            continue
        v = json.load(open(vp))
        got, tot = int(v['gates_pass']), int(v['gates_total'])
        for wrong in range(got + 1, tot + 1):
            R.append((f'{tag} 옛 게이트 수 {wrong}/{tot}',
                      rf'{wrong}\s*/\s*{tot}\s*(?:gates|게이트)|게이트\s*{wrong}/{tot}',
                      f'{tag} = {got}/{tot} ({vp} 동결값)',
                      True))   # ★주석도 본다 --- 아래 설명
    return R


# ── 문서화된 예외 (줄 내용 키; ⚠사유 필수) ───────────────────────────
ALLOW = {
    # ★[2026-09-02] **정정의 정정.** 수중익선 제어 장(`sec:hydrofoil_exp`)은
    #   `wavesim.SEA_STATE` 를 쓰지 않는다 --- `d3_d17_ctrl_closedloop.py:62-63` 이
    #   *"Hs/Tp from the hydrofoil paper head-sea scenario rows"* 라 적고
    #   `SEAS = {'BF3': (0.88, 5.0), 'BF4': (1.88, 7.5)}` 를 쓴다.
    #   내가 이 표를 SEA_STATE 로 "고쳤다가" 80줄 뒤의 `a_base = 1.631` 과 모순시켰다.
    #   ⟹ **원래가 옳았고**, 이 네 줄은 예외다.
    'BF3 Head    & 0.88 & 5.0': '수중익선 논문 규약 (wavesim 아님)',
    'BF4 Head    & 1.88 & 7.5': '수중익선 논문 규약 (wavesim 아님)',
    'BF3 Beam    & 0.88 & 5.0': '수중익선 논문 규약 (wavesim 아님)',
    'BF3 Quarter & 0.88 & 5.0': '수중익선 논문 규약 (wavesim 아님)',
    # 센서 **부류** 일반 서술이고, 두 초록 모두 "본 논문 코퍼스는 약 50" 을 병기한다.
    '영상 등 --- 는 프레임당 약 50--200개의 희소하고 불균질한 3차원 관측점을 생성한다':
        '센서 부류 일반 범위 (코퍼스 값 병기됨)',
}


def main():
    bad = []
    for doc in DOCS:
        path = os.path.join(ROOT, doc)
        if not os.path.exists(path):
            continue
        for ln, raw in enumerate(open(path, encoding='utf-8'), 1):
            if raw.lstrip().startswith('%'):
                continue                       # 주석은 인쇄되지 않는다
            line = re.sub(r'(?<!\\)%.*$', '', raw).rstrip()
            if any(k in line for k in ALLOW):
                continue
            for rule in rules():
                name, pat, right = rule[0], rule[1], rule[2]
                if re.search(pat, line):
                    bad.append((doc, ln, name, right, line.strip()[:78]))
        # ★★[2026-09-02] **두 번째 통과 --- 주석도 본다**(출처 규칙 한정).
        #   ⛔초판은 `raw.lstrip().startswith('%')` 로 주석을 통째로 건너뛰었고
        #   사유가 *"주석은 인쇄되지 않는다"* 였다. **인쇄면 값 규칙에는 옳다.**
        #   그러나 **출처 규칙**(동결 verdict 의 게이트 수 같은 것)은 주석이 바로
        #   썩는 자리다 --- 실제로 `B-26` 의 `29/29` 가 **여덟 곳 전부 주석**이었고
        #   그래서 이 검사기가 구조적으로 못 봤다.
        #   ⟹ `scan_comments=True` 규칙만 주석에 다시 돌린다.
        #   ★이 저장소의 규율 [[feedback-comments-count-as-N]] 을 기계화한 것이다.
        for ln, raw in enumerate(open(path, encoding='utf-8'), 1):
            if not raw.lstrip().startswith('%'):
                continue
            for rule in rules():
                if len(rule) < 4 or not rule[3]:
                    continue
                if re.search(rule[1], raw):
                    bad.append((doc, ln, rule[0] + ' [주석]', rule[2], raw.strip()[:78]))
    print('── 표준 사실 검사 ─────────────────────────────────────────')
    if not bad:
        print(f'  [OK  ] {len(rules())} 규칙 · {len(DOCS)} 문서 · 틀린 값 0건')
        return 0
    for doc, ln, name, right, txt in bad:
        print(f'  [FAIL] {doc}:{ln}  [{name}]')
        print(f'         옳은 값: {right}')
        print(f'         인쇄됨 : {txt}')
    print(f'\n⛔{len(bad)}건 — 코드/코퍼스와 다른 값이 문서에 남아 있다.')
    print('⚠예외라면 사유와 함께 ALLOW 에 **의도적으로** 적어라.')
    return 1


if __name__ == '__main__':
    sys.exit(main())
