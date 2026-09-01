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
import os
import re
import sys

import h5py

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
from wavesim.pipeline import SEA_STATE                              # noqa: E402

DOCS = ['paper/wavesplat.tex', 'thesis/thesis_main.tex']

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
    return R


# ── 문서화된 예외 (줄 내용 키; ⚠사유 필수) ───────────────────────────
ALLOW = {
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
            for name, pat, right in rules():
                if re.search(pat, line):
                    bad.append((doc, ln, name, right, line.strip()[:78]))
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
