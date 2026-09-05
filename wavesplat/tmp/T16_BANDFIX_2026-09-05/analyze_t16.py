"""`T16-BANDFIX` 분석기 — ⛔러너보다 **먼저** 쓰였다. ★**이 파일이 행 스키마를 정의한다.**

════════════════════════════════════════════════════════════════════════════
행 스키마 (`PREREG §6.5`) — ⛔열이 빠지면 게이트가 **발화 불가**다.
    러너는 아래를 **되읽기로** 채운다. ⛔러너가 스스로 쓴 리터럴을 그대로 다시 비교하는
    열은 게이트가 되지 못한다(항진) — 그런 열에는 「되읽기 지점」을 함께 적었다.

  식별      block(**아카이브 전체 경로**) · tier('TIER-P'|'ANCHOR') · arm · angle · hs
  관측·창   seed · seed_applied(★`rng.bit_generator.seed_seq.entropy` **되읽기**)
            seed_rng_md5(★`json.dumps(rng.bit_generator.state)` 의 md5, **추첨 전**)
            n_frames · density · dt · tq · t_origin · e_shape · frame_index
            n_offered · n_gathered · cap_proposal · mean_convention · axis_convention
  채점      nrmse · nrmse_explicit · sigma4 · invert_maxdiff
  누출      gt_std · gt_value · gt_std_after(★`initialize()` **후**)
  팔 정체   prov(★`upd.arm_provenance()` **전체**) · deploy_consts(9종 `cfg` 되읽기)
            proposal_md5 · atoms_md5 · proposal_device · n_atoms_installed · vp_clamp_hits
            use_stokes2 · use_neural_corrector · ckpt_sha256 · code_md5_row · wall_s
  ★규칙    s_rule(클리핑 후) · s_raw(**클리핑 전**) · s_clipped(bool)
            s_readback(변환된 좌표에서 되읽은 값) · s_blind(**별도 프로세스** 재실행)
            ★s_raw_c05 · s_raw_c20   ⛔옛 이름 `s_c05`/`s_c20` 폐기(R-7) — **비클리핑**이다
            ★k1hat_raw_c05 · k1hat_raw_c20  ⛔옛 이름 `s1_c*` 폐기 — 그것은 `s` 가 아니라 `k̂₁`
            equivar_rule(= `f'{fn.__module__}.{fn.__qualname__}'` of `SR16.RULES_RAW[arm]`)
            x_scale · t_scale · disp_relerr
  ★빔포머  kp_hat(2단 = k̂₂) · kp_hat_stage1(1단 = k̂₁) · klim1 · dk1 · klim2
            klim1_c05 · klim1_c20(★`c` 재라벨 호출에서 **후크가 잡은** 1단 반경)
            beamform_calls(= **본 규칙 호출**에서 소비자가 받은 `(klim, n)` 목록)
            beamform_calls_n(= 이 행의 **전** 평가 호출 수: 본 2 + 등변 4 + blind 2 = 8)
            beamform_device · floor_margin · edge_margin · inner_margin · inner_ok
            kd_min · tp_hat · L_obs_x · L_obs_y · L_TRAIN · K_TRAIN

  ⚠`ZERO` 팔은 추정기를 돌리지 않는다 ⟹ 규칙·빔포머·팔 정체 열이 `None` 이다(게이트가 건너뛴다).
  ⚠`AUTO-AP` 는 빔포머를 **0회** 부른다(§8) ⟹ `beamform_calls == []` · `kp_hat*` 는 `None`.
    ★그것 자체를 `G-EQUIVAR` 가 요구한다 — 조항 면제가 공짜가 아니게.
  ⛔`(x or 기본값)` 을 쓰지 마라 — `0.0` 은 falsy 다. 이 파일은 전부 `is None` 으로 가른다.
════════════════════════════════════════════════════════════════════════════
"""
import ast
import glob
import hashlib
import inspect
import json
import math
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gates_t16 as G                                              # noqa: E402
import scale_rules_t14 as SR14                                     # noqa: E402

ROWS = os.path.join(HERE, 'results_t16.jsonl')
META = os.path.join(HERE, 'meta_t16.json')
PREREG = os.path.join(HERE, 'PREREG.md')
FROZEN = os.path.join(HERE, 'FROZEN_md5.txt')
FROZEN_A = os.path.join(HERE, 'FROZEN_md5_A.txt')
ANALYSIS_SIDE = ('gates_t16.py', 'analyze_t16.py')
OUTDIR = HERE
BANNED = ('target', 'gt', 'eta', 'Ew', 'hs', 'angle', 'sigma4', 'block')
TIER_P = 'TIER-P'


# ── 작은 도우미 ─────────────────────────────────────────────────────────────
def _num(x):
    """숫자면 float, 아니면 None. ⛔`or` 를 쓰지 않는 이유: `0.0` 은 falsy 다."""
    if x is None or isinstance(x, bool):
        return None
    return float(x) if isinstance(x, (int, float)) else None


def _rel(a, b):
    """상대차 `|a/b − 1|`. ★어느 쪽이든 없으면 **`inf`**(= 위반)를 돌려준다.

    ⛔`None` 을 돌려주면 `None > tol` 이 `TypeError` 로 죽고, 그것을 `try` 로 감싸면
    **결측이 조용히 통과**한다. 결측은 통과가 아니라 위반이다.
    """
    if a is None or b is None or b == 0:
        return float('inf')
    return abs(a / b - 1.0)


def _lobs(r):
    """행이 기록한 관측 개구 `max(ptp x, ptp y)`. ⛔`(x or 0.0)` 을 쓰지 않는다."""
    lx, ly = _num(r.get('L_obs_x')), _num(r.get('L_obs_y'))
    if lx is None or ly is None:
        return None
    return max(lx, ly)


def _frozen(p):
    if not os.path.exists(p):
        return None
    out = {}
    for line in open(p, encoding='utf-8'):
        if line.strip() and not line.startswith('#'):
            h, k = line.split(None, 1)
            out[k.strip()] = h.strip()
    return out


def _stat(v):
    v = [x for x in v if x is not None]
    if not v:
        return None
    return {'n': len(v), 'median': float(np.median(v)),
            'min': float(min(v)), 'max': float(max(v))}


def _prereg_gate_names():
    """★사전등록 §5–§6 이 등록한 게이트 이름. ⛔§14 는 스캔하지 않는다 —
    거기엔 개정 R-18 이 **일부러 인쇄한 유령 이름**(`G-HURTS`)이 있다."""
    if not os.path.exists(PREREG):
        return None
    t = open(PREREG, encoding='utf-8').read()
    try:
        sec = t[t.index('## §5 '):t.index('## §7 ')]
    except ValueError:
        return None
    return set(re.findall(r'G-[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*', sec))


def _rule_src_tokens(fn, seen=None):
    """★`getsource(**함수**)` 를 `ast.parse` 해 **식별자·속성명·인자명**만 모은다.

    ⛔`getsource(모듈)`·파일 텍스트 검색 금지 — `scale_rules_t14.py:4` **독스트링**이 금지
    토큰을 전부 담고 있어 모듈 단위로 구현하면 임포트만으로 FAIL 한다(`m-3`).
    ★규칙이 같은 모듈 안에서 부르는 보조 함수까지 **따라간다**(`s_bandr_raw → kp_hat_rel`).
    """
    seen = set() if seen is None else seen
    key = getattr(fn, '__qualname__', repr(fn))
    if key in seen:
        return set()
    seen.add(key)
    try:
        tree = ast.parse(inspect.getsource(fn).lstrip())
    except (OSError, TypeError, SyntaxError, IndentationError):
        return {'<소스를 못 읽었다>'}
    toks, calls = set(), set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name):
            toks.add(n.id)
            calls.add(n.id)
        elif isinstance(n, ast.Attribute):
            toks.add(n.attr)
        elif isinstance(n, ast.arg):
            toks.add(n.arg)
        elif isinstance(n, ast.keyword) and n.arg:
            toks.add(n.arg)
    gl = getattr(fn, '__globals__', {})
    for c in sorted(calls):
        sub = gl.get(c)
        if inspect.isfunction(sub) and getattr(sub, '__module__', '').startswith('scale_rules'):
            toks |= _rule_src_tokens(sub, seen)
    return toks


def _globals_have_target(fn):
    """규칙의 `__globals__` 에 표적 크기 배열이 있는가 (참값 밀반입 탐지)."""
    bad = []
    for k, v in getattr(fn, '__globals__', {}).items():
        if isinstance(v, np.ndarray) and v.size >= 1024:
            bad.append(f'{k}{tuple(v.shape)}')
    return bad


def main():                                                        # noqa: C901
    rows = [json.loads(x) for x in open(ROWS, encoding='utf-8') if x.strip()]
    meta = json.load(open(META, encoding='utf-8')) if os.path.exists(META) else {}
    tier_p, anchor = G.tiers()
    by = {}
    for r in rows:
        by.setdefault((r['tier'], r['arm']), {})[r['block']] = r
    est = [r for r in rows if r['arm'] != 'ZERO']
    auto = [r for r in rows if r['arm'] in G.AUTO_ARMS]
    bf = [r for r in auto if r['arm'] in G.BEAMFORM_ARMS]
    A = []

    def wins(arm, ref, tier=TIER_P):
        d, z = by.get((tier, arm), {}), by.get((tier, ref), {})
        cm = sorted(set(d) & set(z))
        return sum(1 for b in cm if d[b]['nrmse'] < z[b]['nrmse']), len(cm)

    def wins_on(arm, ref, blocks, tier=TIER_P):
        d, z = by.get((tier, arm), {}), by.get((tier, ref), {})
        cm = sorted(set(d) & set(z) & set(blocks))
        return sum(1 for b in cm if d[b]['nrmse'] < z[b]['nrmse']), len(cm)

    # ═══ §6.1 규칙이 well-posed 인가 ══════════════════════════════════════
    # ★G-EQUIVAR — 비클리핑 규칙을 `c ∈ {0.5, 2.0}` 재라벨(수심 고정)에서 재호출한 값으로 잰다.
    #   FAIL 조건 = 재라벨된 관측에서 규칙이 `c` 배로 정확히 따라오지 않는다(> 1e-6).
    eb = []
    for r in [x for x in auto if x['arm'] in G.CANDIDATES]:
        tag = f"{r['arm']}:{r['block'][-14:]}"
        s0, c05, c20 = _num(r.get('s_raw')), _num(r.get('s_raw_c05')), _num(r.get('s_raw_c20'))
        if s0 is None or c05 is None or c20 is None:
            eb.append(f'{tag}:missing'); continue
        if _rel(c05 * 0.5, s0) > G.EQ_TOL:
            eb.append(f'{tag}:s_c05')
        if _rel(c20 * 2.0, s0) > G.EQ_TOL:
            eb.append(f'{tag}:s_c20')
        k1 = _num(r.get('kp_hat_stage1'))
        if r['arm'] in G.BEAMFORM_ARMS:
            k05, k20 = _num(r.get('k1hat_raw_c05')), _num(r.get('k1hat_raw_c20'))
            if k1 is None or k05 is None or k20 is None:
                eb.append(f'{tag}:k1_missing')
            else:
                if _rel(k05, 0.5 * k1) > G.EQ_TOL:
                    eb.append(f'{tag}:k1_c05')
                if _rel(k20, 2.0 * k1) > G.EQ_TOL:
                    eb.append(f'{tag}:k1_c20')
        elif k1 is not None or r.get('beamform_calls'):
            # ★`AUTO-AP` 에 1단 조항을 면제하는 대신 **정말로 빔포머를 안 불렀는지**를 요구한다
            eb.append(f'{tag}:unexpected_stage1')
    A.append((f'★★G-EQUIVAR 규칙 등변성 c∈{{0.5,2.0}} (수심 고정) — s 와 1단 k̂₁ 둘 다 '
              f'≤{G.EQ_TOL} [비클리핑 · 판정 자료 위 · 추정기 런 0회]',
              bool(auto) and not eb, f'위반 {len(eb)} {eb[:6]}'))

    # ★G-EQUIVAR-RAW — 「비클리핑」을 **이름이 아니라 행동**으로 증명한다(R-7).
    #   FAIL 조건 = 러너가 clipped 값을 적었다(그러면 `AUTO-AP` 의 c=0.5 열이 정확히 S_MAX 다).
    rb, ap_c05 = [], []
    want_qn = None
    if G.SR16 is None:
        rb.append('scale_rules_t16 임포트 실패')
    for r in [x for x in auto if x['arm'] in G.CANDIDATES]:
        tag = f"{r['arm']}:{r['block'][-14:]}"
        if G.SR16 is not None:
            fn = G.SR16.RULES_RAW.get(r['arm'])
            want_qn = None if fn is None else f'{fn.__module__}.{fn.__qualname__}'
            if want_qn is None or r.get('equivar_rule') != want_qn:
                rb.append(f"{tag}:rule={r.get('equivar_rule')}≠{want_qn}")
        for k in ('s_raw_c05', 's_raw_c20', 's_raw'):
            v = _num(r.get(k))
            if v is not None and (v == G.S_MIN or v == G.S_MAX):
                rb.append(f'{tag}:{k}=레일값 {v}')          # ⟹ 클리핑된 값을 적었다
        if r['arm'] == 'AUTO-AP':
            v = _num(r.get('s_raw_c05'))
            ap_c05.append(v)
            if v is None or not v > G.S_MAX:
                rb.append(f'{tag}:s_raw_c05={v} ≤ S_MAX={G.S_MAX} (행동 증거 없음)')
    if G.SR16 is not None:
        for a in G.CANDIDATES:            # 원시/클리핑이 **다른 객체**여야 한다
            if G.SR16.RULES_RAW.get(a) is G.SR16.RULES.get(a):
                rb.append(f'{a}:RULES_RAW is RULES (같은 객체)')
    A.append((f'★★G-EQUIVAR-RAW 등변 열이 **비클리핑**임을 행동으로 — AUTO-AP 의 s_raw_c05 > '
              f'S_MAX={G.S_MAX} ∧ 레일값 비트일치 0 ∧ 규칙 객체 되읽기',
              bool(auto) and not rb,
              f'위반 {len(rb)} {rb[:4]} · AUTO-AP s_raw_c05 {_stat(ap_c05)}'))

    # ⚠G-EQUIVAR-OLD [기록 전용] — 옛 규칙의 편차. ⛔막지 않는다(후보가 아니라 비교자다).
    old = []
    for r in [x for x in auto if x['arm'] in G.COMPARATORS]:
        s0, c05, c20 = _num(r.get('s_raw')), _num(r.get('s_raw_c05')), _num(r.get('s_raw_c20'))
        old.append({'block': r['block'][-14:],
                    'dev_c05': None if (s0 is None or c05 is None) else _rel(c05 * 0.5, s0),
                    'dev_c20': None if (s0 is None or c20 is None) else _rel(c20 * 2.0, s0)})
    old_stat = {'c05': _stat([o['dev_c05'] for o in old]),
                'c20': _stat([o['dev_c20'] for o in old]),
                'gt_5.88pct_c05': sum(1 for o in old if o['dev_c05'] is not None
                                      and o['dev_c05'] > 7.0 / 119.0),
                'n_missing': sum(1 for o in old if o['dev_c05'] is None)}
    A.append(('⚠G-EQUIVAR-OLD [기록 전용] 옛 규칙(AUTO-BAND)의 등변 편차 — ⛔허용오차 없음 · '
              '⚠크기는 **규약 종속**이다(§9.3-15)', True, f'{old_stat}'))

    # ★★G-KLIM1 — **소비자 되읽기**. FAIL 조건 = 규칙이 넘겼다고 주장한 klim 과 후크가 잡은
    #   `(klim, n)` 이 다르다 / 새 규칙의 1단이 개구에서 유도되지 않았다(분석기 독립 재계산).
    kb, calls_n = [], {}
    for r in auto:
        tag = f"{r['arm']}:{r['block'][-14:]}"
        lobs = _lobs(r)
        calls = r.get('beamform_calls')
        calls_n.setdefault(r['block'], {})[r['arm']] = r.get('beamform_calls_n')
        if calls is None or lobs is None:
            kb.append(f'{tag}:missing'); continue
        if r['arm'] not in G.BEAMFORM_ARMS:
            if len(calls) != 0:
                kb.append(f'{tag}:AP 가 빔포머를 {len(calls)}회 불렀다')
            continue
        if len(calls) != G.STAGES_PER_EVAL:
            kb.append(f'{tag}:calls={len(calls)}'); continue
        want1 = G.klim1_expected(r['arm'], lobs)          # ★분석기가 행의 L_obs 로 재계산
        k1 = _num(r.get('kp_hat_stage1'))
        got1, n1 = _num(calls[0][0]), calls[0][1]
        got2, n2 = _num(calls[1][0]), calls[1][1]
        if _rel(got1, want1) > 1e-12:
            kb.append(f'{tag}:klim1 {got1}≠{want1}')
        if k1 is None or _rel(got2, G.ZOOM * k1) > 1e-12:
            kb.append(f'{tag}:klim2 {got2}≠3.5·k̂₁')
        if n1 != G.N1 or n2 != G.N1:
            kb.append(f'{tag}:n=({n1},{n2})≠{G.N1}')
        if _rel(_num(r.get('klim1')), got1) > 1e-12 or _rel(_num(r.get('klim2')), got2) > 1e-12:
            kb.append(f'{tag}:기록 klim ≠ 후크 klim')
        if _rel(_num(r.get('dk1')), G.dk1_expected(r['arm'], lobs)) > 1e-12:
            kb.append(f'{tag}:dk1')
    per_file = {b: sum(v for v in d.values() if v is not None) for b, d in calls_n.items()}
    cost_ok = sorted(set(per_file.values()))
    A.append((f'★★G-KLIM1 소비자가 실제로 받은 (klim, n) 되읽기 — BANDR 1단 = 0.5·(N1−1)·2π/'
              f'L_obs(분석기 독립 재계산) · BAND 1단 = {G.KLIM1_OLD} · 2단 = 3.5·k̂₁ · n=={G.N1}',
              bool(auto) and not kb,
              f'위반 {len(kb)} {kb[:4]} · 파일당 호출 {cost_ok} '
              f'(§8 유도 {G.BEAMFORM_CALLS_PER_FILE})'))

    # ★G-KLIM1-LIVE — 개구 의존성 실증. FAIL 조건 = 1단 반경이 재라벨을 **따라가지 않는다**
    #   (상수를 박아 두면 `klim1_c05 == klim1` 이 되어 여기서 죽는다).
    lb, contrast = [], []
    for r in bf:
        tag = f"{r['arm']}:{r['block'][-14:]}"
        k1, a5, a2 = (_num(r.get('klim1')), _num(r.get('klim1_c05')), _num(r.get('klim1_c20')))
        if r['arm'] == 'AUTO-BANDR':
            if k1 is None or a5 is None or a2 is None:
                lb.append(f'{tag}:missing')
            elif a5 != 0.5 * k1 or a2 != 2.0 * k1:
                lb.append(f'{tag}:{a5}/{a2} vs {k1}')
        else:                       # ⚠옛 규칙은 절대 격자다 — **안 움직이는 것이 정상**
            contrast.append((a5, a2))
    fixed_old = all(v is not None and v == G.KLIM1_OLD for p in contrast for v in p)
    A.append(('★G-KLIM1-LIVE 재라벨에서 1단 반경이 c 배로 **비트 정확히** 움직인다 [행동 검사 — '
              '"KLIM=0.6 상수와 구별되지 않는다" 를 닫는다]', bool(bf) and not lb,
              f'위반 {len(lb)} {lb[:4]} · 대조(옛 규칙 klim1 고정 {G.KLIM1_OLD}) {fixed_old}'))

    # ★G-FLOOR — FAIL 조건 = `max(·,1e-3)` 바닥이 결속한다(결속하면 등변성이 구조적으로 깨진다).
    fb, fm = [], []
    for r in bf:
        k1, k2, rec = (_num(r.get('kp_hat_stage1')), _num(r.get('kp_hat')),
                       _num(r.get('floor_margin')))
        if k1 is None or k2 is None:
            fb.append(f"{r['arm']}:missing"); continue
        m = min(k1, k2) / G.FLOOR
        fm.append(m)
        if not (k1 > G.FLOOR and k2 > G.FLOOR):
            fb.append(f"{r['arm']}:{r['block'][-14:]}")
        if rec is None or _rel(rec, m) > 1e-9:
            fb.append(f"{r['arm']}:floor_margin 기록 {rec} ≠ 재계산 {m}")
    A.append((f'★G-FLOOR 바닥 {G.FLOOR} 이 한 번도 결속하지 않는다 (여유 재계산 대조)',
              bool(bf) and not fb, f'위반 {len(fb)} · 여유 {_stat(fm)}'))

    # ★G-EDGE — 1단 첨두가 **바깥** 가장자리 링이 아니다. ⚠BANDR 에서는 여유 17배라
    #   **구조적으로 거의 항상 참**(이름에 표기, `m-2`).
    gb, em = [], []
    for r in [x for x in bf if x['arm'] == 'AUTO-BANDR']:
        k1, kl, rec = _num(r.get('kp_hat_stage1')), _num(r.get('klim1')), _num(r.get('edge_margin'))
        if k1 is None or kl is None:
            gb.append(f"{r['block'][-14:]}:missing"); continue
        lim = kl * (1.0 - 2.0 / (G.N1 - 1))
        em.append(k1 / kl)
        if k1 > lim:
            gb.append(f"{r['block'][-14:]}:k̂₁={k1:.4f}>{lim:.4f}")
        if rec is None or _rel(rec, k1 / kl) > 1e-9:
            gb.append(f"{r['block'][-14:]}:edge_margin 기록 {rec} ≠ 재계산")
    A.append(('★G-EDGE 1단 첨두가 최외곽 링이 아니다 [⚠BANDR 에서는 **구조적으로 거의 항상 참**]',
              not gb, f'위반 {len(gb)} · k̂₁/klim₁ {_stat(em)}'))

    # ★★G-INNER(a) — ★수리가 만든 **반대 방향**의 고장(R-6). FAIL 조건 = 2단 첨두가 창의
    #   반경 밖(= 1단이 안쪽 링에 잘못 앉아 2단 창이 너무 좁았다) = well-posedness 위반.
    ib, im, im_old = [], [], []
    for r in bf:
        k1, k2 = _num(r.get('kp_hat_stage1')), _num(r.get('kp_hat'))
        lo = _lobs(r)
        dk = None if lo is None else G.dk1_expected(r['arm'], lo)
        marg = None if (k1 is None or dk is None or dk == 0) else k1 / dk
        (im if r['arm'] == 'AUTO-BANDR' else im_old).append(marg)
        if r['arm'] != 'AUTO-BANDR':
            continue
        if k1 is None or k2 is None:
            ib.append(f"{r['block'][-14:]}:missing"); continue
        ok = k2 <= G.ZOOM * k1 * (1.0 + 1e-12)
        if not ok:
            ib.append(f"{r['block'][-14:]}:k̂₂={k2:.4f}>3.5·k̂₁={G.ZOOM * k1:.4f}")
        if r.get('inner_ok') is not bool(ok):
            ib.append(f"{r['block'][-14:]}:inner_ok 기록 {r.get('inner_ok')} ≠ 재계산 {ok}")
        rec = _num(r.get('inner_margin'))
        if marg is not None and (rec is None or _rel(rec, marg) > 1e-9):
            ib.append(f"{r['block'][-14:]}:inner_margin 기록 {rec} ≠ 재계산 {marg}")
    A.append(('★★G-INNER(a) 2단 첨두가 줌 창의 **반경 안** (k̂₂ ≤ 3.5·k̂₁) — ★수리가 1단 바닥을 '
              '낮춰 새로 생긴 고장 방향이다', bool(im) and not ib,
              f'위반 {len(ib)} {ib[:4]}'))
    A.append((f'⚠G-INNER(b) [기록 전용] inner_margin = k̂₁/Δk₁ (보고 문턱 {G.INNER_MARGIN_REPORT}) '
              f'— ⛔막지 않는다', True,
              f'BANDR {_stat(im)} · BAND {_stat(im_old)} · 문턱 미만 '
              f'{sum(1 for v in im if v is not None and v < G.INNER_MARGIN_REPORT)}'))

    # ⚠G-BANDRES [기록 전용] — `T14` 의 1.16/0.88 과 **같은 자로** 비교 가능하게.
    bres = {}
    for a in G.BEAMFORM_ARMS:
        vv = []
        for r in [x for x in bf if x['arm'] == a]:
            k2, lo = _num(r.get('kp_hat')), _lobs(r)
            dk = None if lo is None else G.dk1_expected(a, lo)
            vv.append(None if (k2 is None or dk is None) else k2 / dk)
        bres[a] = _stat(vv)
    A.append(('⚠G-BANDRES [기록 전용] 팔별 k̂_p/Δk₁ — ★T14 실측 중앙 1.16 · 최소 0.88 과 같은 자',
              True, f'{bres}'))

    # ★★G-S-BLIND — 관측 5배열만 별도 프로세스에 넘겨 재실행 ⟹ `s` **비트 동일**.
    #   FAIL 조건 = 규칙이 관측 밖의 무언가(파일명·참값·전역 상태)를 봤다.
    sb = [f"{r['arm']}:{r['block'][-14:]}" for r in auto
          if r.get('s_blind') is None or r['s_blind'] != r.get('s_rule')]
    A.append(('★★G-S-BLIND 관측 5배열만 별도 프로세스에 넘겨 규칙 재실행 ⟹ s 비트 동일 '
              '[행동 검사 — ⛔이것이 하중을 진다]', bool(auto) and not sb,
              f'위반 {len(sb)}/{len(auto)}'))

    # ★G-S-SRC — 정적. FAIL 조건 = 규칙(또는 그것이 부르는 같은-모듈 보조 함수)의 **식별자**에
    #   표적·메타 토큰이 있다 / 시그니처가 5인자가 아니다 / 전역에 표적 크기 배열이 있다.
    ss = []
    if G.SR16 is None:
        ss.append('scale_rules_t16 임포트 실패')
    else:
        for nm, fn in sorted(G.SR16.RULES.items()):
            if list(inspect.signature(fn).parameters) != ['x', 'y', 't', 'z', 'depth']:
                ss.append(f'{nm}:sig')
            toks = _rule_src_tokens(fn)
            ss += [f'{nm}:{tk}' for tk in BANNED if tk in toks]
            ss += [f'{nm}:glob:{g}' for g in _globals_have_target(fn)]
    A.append(('★G-S-SRC 규칙 5인자 · **AST 식별자**에 표적/메타 토큰 0회 · 전역에 표적 배열 0개 '
              '[⚠단독 항진 위험 — G-S-BLIND 가 하중을 진다]', not ss, f'위반 {ss[:6] or "없음"}'))

    # ★★G-S-READBACK — 러너 리터럴이 아니라 **프레임이 실제로 가진 좌표**에서 되읽는다.
    rr = []
    for r in est:
        s = _num(r.get('s_rule'))
        xs, ts, dre = _num(r.get('x_scale')), _num(r.get('t_scale')), _num(r.get('disp_relerr'))
        srb = _num(r.get('s_readback'))
        if s is None or xs is None or ts is None or dre is None or srb is None:
            rr.append(f"{r['arm']}:missing"); continue
        if (abs(1.0 / xs - s) > G.READBACK_TOL * max(1.0, s)
                or abs(ts - 1.0 / math.sqrt(s)) > G.READBACK_TOL
                or abs(srb - s) > G.READBACK_TOL * max(1.0, s) or dre > 1e-9):
            rr.append(f"{r['arm']}:{r['block'][-14:]}")
    A.append(("★★G-S-READBACK 변환된 좌표에서 s 되읽기 (1/median(x'/x)==s · median(t'/t)==1/√s · "
              'disp_relerr ≤ 1e-9)', bool(est) and not rr, f'위반 {len(rr)}/{len(est)}'))

    # ★G-RULEID — 단일출처. 함수는 **객체 동일성**, 상수는 `==` 로 차단하고 `id()` 는 기록만.
    #   (⛔float 에 `is` 를 쓰면 CPython 인터닝에 의존하는 구현 종속 비교가 된다, `m-6`.)
    idb, ids = [], {}
    if G.SR16 is None:
        idb.append('scale_rules_t16 임포트 실패')
    else:
        for nm in ('s_ap', 's_band', 's_ap_raw', 's_band_raw'):
            if getattr(G.SR16, nm, None) is not getattr(SR14, nm):
                idb.append(f'{nm}: SR16 이 SR14 와 다른 객체다')
        for nm in ('S_MIN', 'S_MAX', 'L_TRAIN', 'K_TRAIN'):
            a, b = getattr(G.SR16, nm, None), getattr(SR14, nm)
            ids[nm] = {'id_SR16': id(a), 'id_SR14': id(b), 'equal': a == b}
            if a != b:
                idb.append(f'{nm}: {a} != {b}')
    A.append(('★G-RULEID 단일출처 — 함수 4개는 **객체 동일성** · 상수 4개는 == 차단 + id() 기록',
              not idb, f'위반 {idb or "없음"} · id {ids}'))

    # G-LTRAIN · G-KTRAIN — 상수 재생성. FAIL 조건 = 동결 상수가 원 절차로 재생되지 않는다.
    try:
        sys.path.insert(0, os.path.join(G.ROOT, 'scripts'))
        from d40_learned_acquisition import pool_xy
        P = np.asarray(pool_xy())
        lt = max(float(np.ptp(P[:, 0])), float(np.ptp(P[:, 1])))
    except Exception:                                              # noqa: BLE001
        lt = None
    A.append(('★G-LTRAIN pool_xy() 재호출로 L_TRAIN 재도출 == 동결값 (⛔손복사 금지)',
              lt is not None and abs(lt - G.L_TRAIN) < 0.01,
              f'재생성 {None if lt is None else round(lt, 3)} vs 동결 {G.L_TRAIN}'))
    try:
        from scipy.optimize import brentq
        from wavesplat.model.dispersion import G as _g
        Tp = np.linspace(5.0, 11.0, 20001)
        kk = np.array([brentq(lambda k, w=2 * np.pi / tp: _g * k * np.tanh(k * 50.0) - w * w,
                              1e-6, 10.0) for tp in Tp])
        kt = float(np.exp(np.mean(np.log(kk))))
    except Exception:                                              # noqa: BLE001
        kt = None
    A.append(('★G-KTRAIN 사전 밴드에서 K_TRAIN 기하평균 재생성 == 동결값 (허용 2% = MC 오차)',
              kt is not None and abs(kt / G.K_TRAIN - 1) < 0.02,
              f'재생성 {None if kt is None else round(kt, 5)} vs 동결 {G.K_TRAIN}'))

    # G-SRC-PIN — 소스 드리프트. FAIL 조건 = 배포 소스의 리터럴이 움직였다.
    try:
        import d3_beamform_net as bn
        from wavesplat.optimizer.bank_mmse import _beamform_top
        from wavesplat.optimizer.initializer import MIN_DK
        sig = inspect.signature(_beamform_top).parameters
        pin = {'DEPTH': bn.DEPTH, 'KLIM': bn.KLIM, 'NK': bn.NK, 'M_OBS': bn.M_OBS,
               'MIN_DK': float(MIN_DK), 'bank_klim': sig['klim'].default,
               'bank_n': sig['n'].default}
        pin_ok = (pin['DEPTH'] == 50.0 and pin['KLIM'] == 0.6 and pin['NK'] == 64
                  and pin['M_OBS'] == 400 and pin['MIN_DK'] == 0.15
                  and pin['bank_klim'] == 0.6 and pin['bank_n'] == 80)
    except Exception as e:                                         # noqa: BLE001
        pin, pin_ok = {'error': f'{type(e).__name__}'}, False
    A.append(('G-SRC-PIN 소스 리터럴 대조 (d3 DEPTH=50 · KLIM=0.6 · NK=64 · M_OBS=400 · '
              'MIN_DK=0.15 · _beamform_top 기본값 klim=0.6, n=80)', pin_ok, f'{pin}'))

    # ═══ §6.2 표본·교차배치 ══════════════════════════════════════════════
    outcome, prop, miss_ref = G.exposed_blocks()
    xb = sorted(set(tier_p) & outcome)
    A.append(('★★G-XBATCH 형제 배치(T11·T12·T13·T14)의 동결 jsonl 을 **직접 열어** Tier-P 와의 '
              '교집합 == ∅ [⛔참조 파일 부재는 FAIL 이다]',
              not xb and not miss_ref,
              f'교집합 {len(xb)} · 성과노출 {len(outcome)} · 부재 {miss_ref}'))
    samp = G.frozen_sample()
    mats = G.archive_mats()
    pop_ok = (samp is not None and sorted(samp) == tier_p
              and len(tier_p) == G.N_P
              and all(b.startswith(G.POP_PREFIX) for b in tier_p)
              and set(by.get((TIER_P, 'ZERO'), {})) == set(tier_p))
    A.append((f'★G-POP 동결 표본 == archive_mats()[{G.POP_RANGE[0]}:{G.POP_RANGE[1]}] == '
              f'{G.POP_PREFIX}* 전수 {G.N_P} == 결과 행의 블록 집합',
              pop_ok, f'표본 {None if samp is None else len(samp)} · Tier-P {len(tier_p)}'))
    exposed_pre = ({mats[i] for i in G.AUDIT_SMOKE_IDX if i < len(mats)} | set(G.SMOKE_FILES))
    A.append(('★G-SAMPLE 사전노출(감사 스모크 8 ∪ 러너 스모크 2) 과 교집합 0 [zip namelist 직접]',
              not (set(tier_p) & exposed_pre) and len(anchor) == G.N_ANC,
              f'교집합 {len(set(tier_p) & exposed_pre)}'))
    # ★G-GRID — 코퍼스 동형. FAIL 조건 = `mixed_data` 가 `mixed_data2` 와 다른 기하다(§2.2 R3).
    lob, gb2 = [], []
    for r in rows:
        es, dt = r.get('e_shape'), _num(r.get('dt'))
        lx, ly = _num(r.get('L_obs_x')), _num(r.get('L_obs_y'))
        if es != G.GRID_SHAPE or dt is None or abs(dt - G.GRID_DT) > 1e-9:
            gb2.append(f"{r['block'][-14:]}:shape/dt")
        if lx is None or ly is None or abs(lx - ly) > G.LOBS_XY_TOL:
            gb2.append(f"{r['block'][-14:]}:L_obs_xy")
        elif r['arm'] == 'ZERO':
            lob.append(max(lx, ly))
    sd = float(np.std(lob)) if lob else None
    A.append((f'★G-GRID 전 행 e_shape=={G.GRID_SHAPE} ∧ dt==1 ∧ |L_obs_x−L_obs_y|≤{G.LOBS_XY_TOL} '
              f'∧ 파일 간 L_obs sd ≤ {G.LOBS_SD_TOL} [⚠팔 간 sd 는 항진이라 뺐다]',
              not gb2 and sd is not None and sd <= G.LOBS_SD_TOL,
              f'위반 {len(gb2)} · L_obs sd {sd} · 중앙 '
              f'{None if not lob else round(float(np.median(lob)), 3)}'))
    sib = {}
    for d in sorted(glob.glob(os.path.join(os.path.dirname(HERE), 'T1[2-5]_*'))):
        for p in sorted(glob.glob(os.path.join(d, 'results_*.jsonl'))):
            sib[os.path.relpath(p, os.path.dirname(HERE))] = round(os.path.getmtime(p), 1)
    A.append(('⚠G-SIBLING [기록 전용] 형제 결과 파일 존재·mtime (T15-HOMEREG 는 우리 코퍼스를 '
              '쓰므로 교집합 0 이 예상값)', True, f'{sib}'))
    tp = _stat([_num(r.get('tp_hat')) for r in bf if r['arm'] == 'AUTO-BANDR'])
    A.append(('⚠G-TP [기록 전용] Tp̂ = 2π/√(g·k̂_p·tanh(k̂_p·d)) — ⛔T14 의 "전 파일 10 s" 는 '
              'mixed_data2 의 진술이고 이 코퍼스에서는 **미확인**이었다', True, f'{tp}'))

    # ═══ §5 역변환 ═══════════════════════════════════════════════════════
    ivb = []
    for r in est:
        md, s4 = _num(r.get('invert_maxdiff')), _num(r.get('sigma4'))
        if md is None or s4 is None or md > G.INVERT_TOL_REL * s4:
            ivb.append(f"{r['arm']}:{r['block'][-14:]}")
    A.append((f'★G-INVERT 암묵/명시 역변환 두 경로 차 ≤ {G.INVERT_TOL_REL}·σ4 [필드 차이로 잰다]',
              bool(est) and not ivb, f'위반 {len(ivb)}/{len(est)}'))
    pb = [r['arm'] for r in est if r.get('use_stokes2') is not False
          or r.get('use_neural_corrector') is not False]
    A.append(('G-PURE use_stokes2 · use_neural_corrector 둘 다 False [모델 객체 되읽기]',
              bool(est) and not pb, f'위반 {len(pb)}'))
    kd = [_num(r.get('kd_min')) for r in bf]
    kdb = [r['block'] for r in bf if _num(r.get('kd_min')) is None
           or _num(r.get('kd_min')) < G.KD_MIN]
    A.append(('★G-DEEP 모든 행 k̂_p·depth ≥ π [⚠G-FLOOR·코퍼스 대역이 이미 보장 ⟹ **구조적으로 '
              '거의 항상 참**]', bool(bf) and not kdb, f'위반 {len(kdb)} · kd_min {_stat(kd)}'))

    # ═══ §6.3 T11·T14 계승 ═══════════════════════════════════════════════
    cnt = {f'{k[0]}:{k[1]}': len(v) for k, v in sorted(by.items())}
    want_cnt = ({f'{TIER_P}:{a}': G.N_P for a in G.TIERP_ARMS}
                | {f'ANCHOR:{a}': G.N_ANC for a in G.ANCHOR_ARMS})
    A.append((f'★G-N 총 {G.N_ROWS}행 · Tier-P {len(G.TIERP_ARMS)}팔×{G.N_P} · ANCHOR '
              f'{len(G.ANCHOR_ARMS)}팔×{G.N_ANC} [⚠문턱 {G.THRESH_S}/{G.THRESH_D}/{G.THRESH_F} 는 '
              f'n={G.N_P} **전용**이다]', len(rows) == G.N_ROWS and cnt == want_cnt,
              f'rows={len(rows)} · {cnt}'))
    nfin = [r['block'] for r in rows if _num(r.get('nrmse')) is None
            or not np.isfinite(r['nrmse'])]
    A.append(('★G-FINITE 모든 행의 nRMSE 가 유한', not nfin, f'비유한 {len(nfin)}'))
    sz = os.path.getsize(G.ZIP) if os.path.exists(G.ZIP) else -1
    A.append((f'★G-SIZE 분석기가 직접 잰 아카이브 크기 == {G.WANT_BYTES}',
              sz == G.WANT_BYTES, str(sz)))
    leak = [r for r in est if _num(r.get('gt_std')) is None or abs(r['gt_std']) > 1e-12
            or _num(r.get('gt_std_after')) is None or abs(r['gt_std_after']) > 1e-12
            or _num(r.get('gt_value')) is None or abs(r['gt_value'] - G.GT_CONST) > 1e-12]
    A.append(('★★G-NOLEAK 붙인 gt 가 상수(std 0) · 값==GT_CONST · initialize() **후에도** [되읽기]',
              bool(est) and not leak, f'위반 {len(leak)}'))
    dupa = [f'{k[0]}:{k[1]}' for k, d in by.items() if k[1] != 'ZERO'
            and len({r.get('proposal_md5') for r in d.values()}) != len(d)]
    A.append(('★G-ATOMS raw 제안 지문이 **(층, 팔) 안에서 블록 간** 전건 상이 (캐시 누출 탐지) '
              '[⛔팔 사이는 비교하지 않는다 — 동점 파일이 있어도 발화하지 않는다]',
              not dupa, f'중복 {dupa or "없음"}'))
    from wavesplat.optimizer.learned_init import M_OBS_CAP
    cb = [r['arm'] for r in est if r.get('n_gathered') is None
          or r['n_gathered'] != r.get('n_offered') or r.get('cap_proposal') != int(M_OBS_CAP)]
    A.append((f'★G-CAP 모음==제공 · 캡=={int(M_OBS_CAP)} [⛔손복사 0 — learned_init 에서 임포트]',
              bool(est) and not cb, f'위반 {len(cb)}'))
    from wavesplat.optimizer.sequential_update import (
        DEFAULT_LEARNED_ENSEMBLE_K as KDEP, VARPRO_STEPS_DEFAULT, LAMBDA_L1_DEFAULT,
        LEARNED_INIT_ITERS_DEFAULT, AUTO_SWITCH_THRESH_DEFAULT, DEFAULT_ANALYTIC_ENSEMBLE_K,
        INIT_METHOD_DEFAULT)
    arb = [r['arm'] for r in est
           if (r.get('prov') or {}).get('resolved_init_method') != 'learned'
           or (r.get('prov') or {}).get('real_domain_gate_fired') is not False
           or (r.get('prov') or {}).get('ensemble_k') != int(KDEP)]
    A.append((f'★G-ARM resolved_init_method==learned · 실측게이트 **미발화** · '
              f'ensemble_k=={int(KDEP)} [팔 정체는 의도가 아니라 되읽기다]',
              bool(est) and not arb, f'위반 {len(arb)}'))
    A.append(('G-WRAPPER 출하 진입점 make_updater [⚠배포 상수상 **구조적으로 참**]',
              bool(est) and all((r.get('prov') or {}).get('is_ensemble_wrapper') for r in est),
              '—'))
    sh = [r['block'] for r in rows if r['arm'] == 'SHIP'
          and (r.get('x_scale') != 1.0 or r.get('t_scale') != 1.0)]
    A.append(('★G-SHIPID SHIP 은 항등 재라벨 — x_scale==1.0 ∧ t_scale==1.0 **정확히**',
              not sh, f'위반 {len(sh)}'))
    t11s, anc_reg = {}, {}
    if os.path.exists(G.SMOKE_T11):
        for line in open(G.SMOKE_T11, encoding='utf-8'):
            if line.strip():
                d = json.loads(line)
                t11s[(d['arm'], d['block'])] = d['nrmse']
    grade, abad, npair = 'T1', [], 0
    for b in anchor:
        for arm, ref in (('SHIP', 'SHIP'), ('FIX2', 'RESCALE')):
            r = by.get(('ANCHOR', arm), {}).get(b)
            v = t11s.get((ref, b))
            anc_reg[f'{arm}:{b[-14:]}'] = v
            if r is None or v is None:
                continue
            npair += 1
            if r['nrmse'] != v:
                grade = 'T2'
                if _rel(r['nrmse'], v) > 1e-9:
                    abad.append(f'{arm}:{b[-14:]}')
    A.append((f'★★G-ANCHOR-T11 앵커 2파일 SHIP·FIX2 가 T11 스모크와 일치 [등급 {grade}: '
              f'T1=비트 동일 · T2=상대차 ≤1e-9] ⛔짝 4개 미만이면 FAIL',
              npair == 4 and not abad, f'짝 {npair}/4 · 불일치 {len(abad)} · 핀 {anc_reg}'))
    want_dep = {'lambda_l1': float(LAMBDA_L1_DEFAULT), 'varpro_steps': int(VARPRO_STEPS_DEFAULT),
                'learned_init_iters': int(LEARNED_INIT_ITERS_DEFAULT),
                'auto_switch_thresh': int(AUTO_SWITCH_THRESH_DEFAULT),
                'init_method': str(INIT_METHOD_DEFAULT), 'tracking_mode': 'w_phi',
                'analytic_ensemble_k': int(DEFAULT_ANALYTIC_ENSEMBLE_K),
                'learned_ensemble_k': int(KDEP), 'depth': G.DEPTH}
    dbad = [r['arm'] for r in est
            if any((r.get('deploy_consts') or {}).get(k) != v for k, v in want_dep.items())]
    A.append(('★G-DEPLOY 배포 상수 9종을 cfg **되읽기**로 확인 [⛔손복사 0 — 전부 임포트 · '
              '⚠init_method 항은 러너 강제값과 같은 리터럴을 비교하므로 **항진**이다]',
              bool(est) and not dbad, f'위반 {len(dbad)} · 기대 {want_dep}'))
    tb = [r['block'] for r in rows if 't_origin' not in r
          or _num(r.get('t_origin')) is None or abs(r['t_origin']) > 1e-9]
    A.append(('★G-TORIGIN 창 시작 t == 0 [★**키 부재를 위반으로** 센다 — 변화 탐지기]',
              not tb, f'위반 {len(tb)}'))
    db = []
    for r in rows:
        es = r.get('e_shape') or [0]
        dt, tq = _num(r.get('dt')), _num(r.get('tq'))
        if (r.get('n_offered') != G.DENSITY * int(G.WIN_S) or r.get('n_frames') != int(G.WIN_S)
                or es[0] < G.J0 + int(G.WIN_S) or dt is None or tq is None
                or abs(tq - (int(G.WIN_S) - 1) * dt) > 1e-9):
            db.append(r['block'])
    A.append((f'★G-DENSITY 제공=={G.DENSITY}×{int(G.WIN_S)} · 창이 자료 안 · tq==(N−1)·dt',
              not db, f'위반 {len(db)}'))
    # ★G-ENV — 동결을 **지금 다시 계산한다**(⛔`T14 A3 §1` 의 "대입만 되고 안 쓰였다" 재발 금지).
    #   그리고 시드는 **적용된 RNG 상태의 지문**으로 되읽는다(⛔러너 리터럴 왕복은 항진이다).
    env = {}
    try:
        import run_t16 as _R
        live = _R.freeze_map()
    except Exception as e:                                         # noqa: BLE001
        live, env['freeze_error'] = None, f'{type(e).__name__}'
    rec = meta.get('code_md5') or {}
    fz, fz2 = _frozen(FROZEN), _frozen(FROZEN_A)
    d_meta = d_fz = d_fz2 = None
    if live is not None:
        run_keys = sorted(k for k in (set(live) | set(fz or {})) if k not in ANALYSIS_SIDE)
        ana_keys = sorted(k for k in (set(live) | set(fz2 or {})) if k in ANALYSIS_SIDE)
        d_meta = [k for k in run_keys if live.get(k) != rec.get(k)]
        d_fz = None if fz is None else [k for k in run_keys if live.get(k) != fz.get(k)]
        d_fz2 = None if fz2 is None else [k for k in ana_keys if live.get(k) != fz2.get(k)]
        meas = {k: v for k, v in live.items() if k not in ANALYSIS_SIDE}
        digest = hashlib.md5(json.dumps(meas, sort_keys=True).encode()).hexdigest()
        env['code_md5_row_mismatch'] = [r['block'] for r in rows
                                        if r.get('code_md5_row') != digest]
    seed_bad = []
    for r in rows:
        want = G.seed_of(r['block'], r['tier'], tier_p)
        got_a = r.get('seed_applied')
        if r.get('seed') != want or got_a != want:
            seed_bad.append(f"{r['block'][-14:]}:{r.get('seed')}/{got_a}≠{want}")
            continue
        st = np.random.default_rng(want).bit_generator.state       # ★분석기가 독립 재생성
        want_md5 = hashlib.md5(json.dumps(st, sort_keys=True, default=str).encode()).hexdigest()
        if r.get('seed_rng_md5') != want_md5:
            seed_bad.append(f"{r['block'][-14:]}:rng지문")
    ck = sorted({r.get('ckpt_sha256') for r in rows if r.get('ckpt_sha256')})
    A.append(('★G-ENV 측정면 3중 동결(**지금 재계산**==meta==0행동결) · 분석면 동결 · threads 되읽기 '
              '· ★시드는 **적용된 RNG 상태 지문**으로 되읽는다 · ckpt sha256',
              meta.get('threads_readback') == 1 and bool(rec) and d_meta == []
              and d_fz == [] and d_fz2 == [] and not seed_bad
              and not env.get('code_md5_row_mismatch') and len(ck) == 1,
              f'측정면 런대비 {d_meta} · 0행동결대비 {d_fz} · 분석면 {d_fz2} · '
              f'시드위반 {len(seed_bad)} {seed_bad[:3]} · ckpt {len(ck)} · {env}'))
    # ═══ §6.4 기록 전용 (⛔막지 않는다 — 조건이 리터럴 True 다) ═════════════
    sclip = {a: sum(1 for r in rows if r['arm'] == a and r.get('s_clipped'))
             for a in G.AUTO_ARMS}
    A.append((f'⚠G-SCLIP [기록 전용] 안전 레일 [{G.S_MIN}, {G.S_MAX}] 접촉 팔별 횟수',
              True, f'{sclip}'))
    svar = {}
    for a in G.AUTO_ARMS + ('SHIP', 'FIX2'):
        v = [_num(r.get('s_rule')) for r in rows if r['arm'] == a]
        v = [x for x in v if x is not None]
        svar[a] = (None if not v else
                   {'n': len(v), 'sd': float(np.std(v)), 'min': min(v), 'max': max(v),
                    'single_value': float(np.std(v)) < 1e-12})
    A.append(('⚠G-SVAR [기록 전용] 팔별 s 의 (sd, min, max) — ★sd==0 이면 이 배치는 '
              '**단일값 검정**이다(적응성이 아니라 착지점을 쟀다는 뜻)', True, f'{svar}'))
    clamp = {a: sum(1 for r in rows if r['arm'] == a
                    and (_num(r.get('vp_clamp_hits')) is not None
                         and r['vp_clamp_hits'] > 0)) for a in G.TIERP_ARMS[1:]}
    A.append(('⚠G-CLAMP [기록 전용] VARPRO 경사 클램프 발화 팔별', True, f'{clamp}'))

    # ── D 축의 처치 크기 (`SEC-DRIVE` 의 원자료 · `T1`·`T2` 의 입력) ────────
    braw = {b: _num(r.get('s_raw')) for b, r in by.get((TIER_P, 'AUTO-BAND'), {}).items()}
    rraw = {b: _num(r.get('s_raw')) for b, r in by.get((TIER_P, 'AUTO-BANDR'), {}).items()}
    dln = {}
    for b in sorted(set(braw) & set(rraw)):
        a, c = braw[b], rraw[b]
        if a is not None and c is not None and a > 0 and c > 0:
            dln[b] = math.log(c) - math.log(a)          # ⛔**클리핑 전** 값으로 잰다
    vals = list(dln.values())
    n_pos = sum(1 for v in vals if v > 0)
    n_neg = sum(1 for v in vals if v < 0)
    med_abs = float(np.median([abs(v) for v in vals])) if vals else None
    iqr = (float(np.percentile([abs(v) for v in vals], 75)
                 - np.percentile([abs(v) for v in vals], 25)) if vals else None)
    t1 = med_abs is not None and med_abs < G.DRIVE_T1_LN
    t2 = max(n_pos, n_neg) < G.DRIVE_T2_SIGN
    D_drive = {'median_abs_dln_s': med_abs, 'iqr': iqr, 'n': len(vals),
               'n_pos': n_pos, 'n_neg': n_neg, 'n_zero': len(vals) - n_pos - n_neg,
               'n_gt_ln105': sum(1 for v in vals if abs(v) > G.DRIVE_T1_LN),
               'T1_treatment_below_scale': bool(t1), 'T2_direction_mixed': bool(t2),
               'ln105': G.DRIVE_T1_LN, 'T2_sign_thresh': G.DRIVE_T2_SIGN}
    A.append(('⚠G-DRIVE [기록 전용] Δ ln s = ln s_bandr_raw − ln s_band_raw 의 중앙·IQR·부호분할 '
              '— ★D 축의 **라벨을 고르는 입력**이다(§3.3 T1·T2). ⛔막지 않는다', True,
              f'{D_drive}'))

    # ═══ 통계량 ══════════════════════════════════════════════════════════
    w_f, n_f = wins('FIX2', 'ZERO')
    w_s, n_s = wins('AUTO-BANDR', 'ZERO')
    w_e, n_e = wins('AUTO-BAND', 'ZERO')
    w_ap, n_ap = wins('AUTO-AP', 'ZERO')
    w_sh, n_sh = wins('SHIP', 'ZERO')
    bd, rd = by.get((TIER_P, 'AUTO-BAND'), {}), by.get((TIER_P, 'AUTO-BANDR'), {})
    common = sorted(set(bd) & set(rd))
    tied = [b for b in common
            if abs(rd[b]['nrmse'] - bd[b]['nrmse']) / max(abs(bd[b]['nrmse']), 1e-12) < G.TIE_REL]
    untied = [b for b in common if b not in set(tied)]
    w_d, n_eff = sum(1 for b in untied if rd[b]['nrmse'] < bd[b]['nrmse']), len(untied)
    kplus = G.k_two_sided(n_eff) if n_eff else None

    blind = G.blind_files(tier_p)
    ang30 = G.angle30_files(tier_p)

    def _pair(arm, ref, blocks=None):
        w, n = (wins(arm, ref) if blocks is None else wins_on(arm, ref, blocks))
        return {'w': w, 'n': n, 'p_one_sided': G.sign_p(w, n)}

    SEC = {
        'SEC-A AUTO-BANDR<SHIP': _pair('AUTO-BANDR', 'SHIP'),
        'SEC-B AUTO-AP<ZERO (⚠노출된 예측)': {'w': w_ap, 'n': n_ap, 'p_one_sided':
                                              G.sign_p(w_ap, n_ap)},
        'SEC-C AUTO-BANDR vs FIX2 (양측 · ⛔동치 주장 금지)': _pair('AUTO-BANDR', 'FIX2'),
        'SEC-D AUTO-BANDR vs AUTO-AP (양측 · EXPLORATORY · q80 병기 의무)':
            _pair('AUTO-BANDR', 'AUTO-AP'),
        'SEC-E AUTO-BAND<ZERO (★사다리 안 · 무보정)': {
            'w': w_e, 'n': n_e, 'p_one_sided': G.sign_p(w_e, n_e),
            'thresh': G.THRESH_SECE, 'pass': w_e >= G.THRESH_SECE, 'q80': G.MDE['SECE_q80']},
        'SEC-BLIND (미접촉 24 · ⛔angle 과 완전 교락)': {
            'n_files': len(blind),
            'S': _pair('AUTO-BANDR', 'ZERO', blind), 'F': _pair('FIX2', 'ZERO', blind),
            'D': {'w': sum(1 for b in blind if b in rd and b in bd
                           and rd[b]['nrmse'] < bd[b]['nrmse']),
                  'n': sum(1 for b in blind if b in rd and b in bd)},
            'thresh': G.THRESH_BLIND, 'q80': G.MDE['BLIND_q80']},
        'SEC-BLIND-ANGLE30 (교락 대조군 · ⚠n 이 달라 검정 비교 불가)': {
            'n_files': len(ang30), 'S': _pair('AUTO-BANDR', 'ZERO', ang30),
            'F': _pair('FIX2', 'ZERO', ang30)},
        'SEC-DRIVE': D_drive,
    }
    layers = {}
    for key, getter in (('angle', lambda b: G.angle_hs(b)[0]), ('hs', lambda b: G.angle_hs(b)[1])):
        for lv in sorted({getter(b) for b in tier_p}):
            sel = [b for b in tier_p if getter(b) == lv]
            layers[f'{key}={lv}'] = _pair('AUTO-BANDR', 'ZERO', sel)
    SEC['SEC-F 층 (기술통계 · 만장일치일 때만 클러스터 주장 가능)'] = layers

    def R(arm):
        d, sh_, fx = (by.get((TIER_P, arm), {}), by.get((TIER_P, 'SHIP'), {}),
                      by.get((TIER_P, 'FIX2'), {}))
        v, ex = [], 0
        for b in sorted(set(d) & set(sh_) & set(fx)):
            lo = math.log(fx[b]['nrmse']) - math.log(sh_[b]['nrmse'])
            if lo <= -G.DRIVE_T1_LN:                    # 포함 조건 = `ln r_FIX2 − ln r_SHIP ≤ −ln1.05`
                v.append((math.log(d[b]['nrmse']) - math.log(sh_[b]['nrmse'])) / lo)
            else:
                ex += 1
        return {'median': float(np.median(v)) if v else None,
                'iqr': (float(np.percentile(v, 75) - np.percentile(v, 25)) if v else None),
                'n_used': len(v), 'n_excluded': ex}

    lab_S = G.label_S(w_s, w_e)
    lab_D = G.label_D(w_d, n_eff, t1, t2)

    # ── §6 게이트 ↔ 사전등록 1:1 + 병기 의무 자가검사 (`G-COV`) ─────────────
    # ★`G-COV`·`G-BLOCK` 은 **마지막에** 붙는다 ⟹ 자기 이름을 미리 넣어 두 게이트 자신도
    #   덮는다(⛔늦게 붙은 게이트가 조용히 장식이 되는 것을 막는다 — `T5-ENS` 부류).
    emitted = sorted({re.search(r'G-[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*', n).group(0)
                      for n, _, _ in A if re.search(r'G-[A-Za-z0-9]+', n)}
                     | {'G-COV', 'G-BLOCK'})
    reg = _prereg_gate_names()
    drift = G.registration_drift()
    cov_bad = []
    if reg is not None:
        cov_bad += [f'사전등록에만: {sorted(reg - set(G.GATE_NAMES))}'] if reg - set(G.GATE_NAMES) else []
        cov_bad += [f'구현에만: {sorted(set(G.GATE_NAMES) - reg)}'] if set(G.GATE_NAMES) - reg else []
    else:
        cov_bad.append('PREREG.md 를 못 읽었다')
    if set(emitted) != set(G.GATE_NAMES):
        cov_bad.append(f'인쇄되지 않은 게이트 {sorted(set(G.GATE_NAMES) - set(emitted))}')
    cov_bad += [f'등록 드리프트: {d}' for d in drift]

    # ★★병기 의무(§3.4) — ⛔분석기가 **기계로** 넣고, 넣었는지 **기계로** 확인한다.
    sec_e_txt = (f"SEC-E {w_e}/{n_e} "
                 f"({'통과' if w_e >= G.THRESH_SECE else '미달'}, 문턱 {G.THRESH_SECE}, 무보정)")
    drive_txt = (f"median|Δ ln s| = {'n/a' if med_abs is None else round(med_abs, 6)} "
                 f"(부호 +{n_pos}/−{n_neg}, ln1.05 = {round(G.DRIVE_T1_LN, 6)})")
    disclosures = [
        # ── §9.2 금지 ──────────────────────────────────────────────────
        f'⛔[§9.2-1] 어떤 라벨도 "우리 방법이 저들 방법보다 낫다" 로 읽지 마라 — 외부 기준선 행이 '
        f'**0개**다. `AUTOSCALE-BEATS-ZERO-*` 의 비교자는 **제로필드**이지 제3자 방법이 아니다. '
        f'라벨은 **귀속**이다.',
        f'⛔[§9.2-2] 음성 라벨(`NO-AUTOSCALE-SKILL-DETECTED`·`D-NO-DIFFERENCE-DETECTED`·'
        f'`D-UNPOWERED`)을 "자동 스케일은 불가능"·"도달 불가" 로 읽지 마라 — 이름이 「미검출」이고 '
        f'MDE 병기가 의무다: S 널 q80={G.MDE["S_q80"]} · D 널 q80={G.MDE["D_q80"]} · '
        f'SEC-BLIND q80={G.MDE["BLIND_q80"]}. ⛔세 값을 **섞지 마라**(`M8b` 선례).',
        f'⛔[§9.2-3] `T11`(n=24, mixed_data2)·`T14`(n=28, Hs 불균형)의 수와 이 배치의 수를 '
        f'**직접 비교하지 마라** — 모집단이 다르다.',
        f'⛔[§9.2-4] `E2`·`T11`·`T14` 와 대각선 비교 금지 · 저들 논문 수치와 비교 금지 · '
        f'solver 미명시("HOS"·"완전 비선형" 이라 쓰지 않는다).',
        f'⛔[§9.2-5] `AUTO-BAND`(옛 규칙)의 착지점을 **규칙의 성능**으로 읽지 마라 — ill-posed 다'
        f'(`T14` 가 그것으로 `BLOCKED`). 차단 게이트에서 뺀 것은 완화가 아니라 **지위 변경**'
        f'(후보 → 비교자)이고, 후보 둘의 허용오차는 `T14`(5.88%)보다 **1e-6 로 더 엄하다**.',
        f'⛔[§9.2-6] `D` 축을 `SEC-E` **와** `D_drive` 없이 인용 금지 — {sec_e_txt} · {drive_txt}. '
        f'(i) 두 팔이 같은 방향으로 실패하면 `D` 는 "덜 나쁜 쪽" 을 고른다(비교자 붕괴, `F1` 부류) '
        f'(ii) 처치가 자보다 작거나 방향이 섞이면 `D` 의 통계량은 **혼합을 센다**.',
        f'⛔[§9.2-6b] `S` 축을 `SEC-E` 없이 인용 금지 — {sec_e_txt}. `S` 는 **처치를 판별하지 '
        f'못한다**; `SEC-E` 가 함께 발화했다면 그 문장은 "자동 스케일이 이긴다" 이지 '
        f'"밴드 수리가 이긴다" 가 아니다. ⛔라벨 접두어 `BANDFIX-` 는 폐기됐다.',
        f'⛔[§9.2-6c] `D-TREATMENT-BELOW-SCALE` 를 "수리가 무의미하다" 로 읽지 마라 — 그것은 '
        f'**이 코퍼스에서** 착지점 차이가 배제 문턱보다 작다는 진술이고 `G-EQUIVAR` 의 판정과 '
        f'**독립**이다. 수리의 근거는 well-posedness 이지 이 코퍼스의 정확도가 아니다. '
        f'⛔`D-NO-DIFFERENCE-DETECTED` 로 접지 마라(`M8b` 선례) · 병기: {drive_txt}.',
        f'⛔[§9.2-7] `SEC-D`(개구 vs 밴드)로 규칙을 **다시 고르지 마라**("질 때까지 조율" 회피). '
        f'새 규칙은 새 사전등록으로만. ⚠`SEC-D` 는 EXPLORATORY 이고 q80={G.MDE["D_q80"]} 병기 의무.',
        f'⛔[§9.2-8] 회수율 `R` 을 평균하지 마라 · CI 를 붙이지 마라 · 라벨로 승격하지 마라 '
        f'(비의 비 · 기술통계).',
        f'⛔[§9.2-9] **배포 권고 아님.** 양성 라벨도 필요조건 하나다(선례 `S-ENS §5`) — 최소한 '
        f'(i) 실측 경로 영향 (ii) 단가 (iii) 우리 코퍼스 무해성이 남는다.',
        f'⛔[§9.2-10] 층·부분집합 진술(`SEC-F`·`SEC-BLIND`·`SEC-BLIND-ANGLE30`)은 전부 '
        f'**기술통계**이고 라벨을 만들지 않는다.',
        # ── §9.3 공시 ──────────────────────────────────────────────────
        f'⚠[§9.3-1] **완전 blind 아님** — 설계자가 `T14 AMENDMENT_2 §2`(AUTO-AP 28/28 · 회수율 '
        f'1.634, mixed_data2) 와 앵커 2파일 nRMSE, `T14` band_fragile 전건을 읽었다. 그것이 '
        f'`q_F`·`q_S`·`q_SEC-E` 를 높게 잡은 근거다. ✅Tier-P 60 은 **성과 노출 0**(`G-XBATCH`).',
        f'⚠[§9.3-2] `s` 와 `klim₁` 이 이 코퍼스에서 거의/완전히 상수다 ⟹ **착지점 진술이지 적응 '
        f'진술이 아니다**. ⛔축퇴는 넓다(`M-1`): `klim₁` 을 0.6033→2.0 으로 3.3배 움직여도 최종 '
        f'`k̂_p` 는 약 1% 만 변한다 ⟹ 자료는 이 규칙을 `[k_p, 2.0]` 의 **어떤 상수와도** 구별하지 '
        f'못한다. 구별은 `G-KLIM1-LIVE`(행동)가 하고 ⛔**자료는 하지 않는다** — provenance 로는 '
        f'충분, 해석으로는 불충분. ⛔"자유 파라미터 0개" 를 **측정 결과로** 서술하지 마라.',
        f'⚠[§9.3-3] **재척도는 단일요인이 아니다**(`M-2`): 두 팔의 차이는 `klim` 이 아니라 `s` 이고, '
        f'그 `s` 가 사전 밴드·`learned_dedup_dk`·`MIN_DK`·성장 DC블록·VARPRO lr·조건화 지도의 '
        f'개구/픽셀 비를 **동시에** 움직인다 ⟹ ⛔단일 기제 귀속 금지. ⛔*"D 다리는 예외적으로 '
        f'단일요인"* 은 **철회**됐다(R-10) — 개입이 하나이지 기제가 하나가 아니다.',
        f'⚠[§9.3-4] `d3_beamform_net.py:30` 의 `DEPTH=50.0` 하드코딩이 config depth 를 조건화 '
        f'지도에서 무효로 만든다 ⟹ `s` 를 키우면 그 불일치도 줄어드는 **비대칭 교락**이다. '
        f'⛔이 배치에서 고치지 않는다(고치면 출하 팔이 아니게 된다).',
        f'⚠[§9.3-5] **예산 비대칭** — 제안 단계는 프레임당 제공 8,000 중 캡 {int(M_OBS_CAP)}만 '
        f'쓴다. "고른 400" 과 "있는 것 전부" 는 다르다.',
        f'⚠[§9.3-6] `Tp` 는 이 코퍼스에서 **미확인**이다 — `T14` 의 "전 파일 10 s" 는 '
        f'`mixed_data2` 의 진술이다. `G-TP` 가 `Tp̂` 를 행마다 적었다.',
        f'⚠[§9.3-7] `SEC-*` 의 p 는 전부 **무보정**이다. Holm/Bonferroni 는 PRIMARY 가족(**2축** = '
        f'`S`·`D`)에만 적용한다. ⚠`SEC-E` 는 무보정인데 사다리 안에 있다 ⟹ '
        f'`AUTOSCALE-BEATS-ZERO-BANDR-ONLY` 는 **EXPLORATORY** 이고 확증 등급이 아니다.',
        f'⚠[§9.3-8] 규칙의 목표 상수(`L_TRAIN`={G.L_TRAIN} · `K_TRAIN`={G.K_TRAIN})는 **우리 학습 '
        f'분포**에서 나온다 — 규칙은 "장면을 우리 사전이 학습된 곳으로 옮긴다" 이고 '
        f'⛔"제3자 자료가 규칙을 정했다" 는 거짓이다.',
        f'⚠[§9.3-9] `k̂_p` 는 FFT 첨두 대비 **편향**이 있다(`T14 §9.3-9`: +7–13%). ⛔보정하지 않았다.',
        f'⚠[§9.3-10] `proposal_device` 는 config 가 정하지 않는다(`propose_atoms` 가 cuda if '
        f'available) ⟹ 되읽어 적었다.',
        f'⚠[§9.3-11] `torch.set_num_threads(1)` 은 numpy/BLAS 를 제약하지 않는다 ⟹ '
        f'⛔`wall_s` 를 팔 간 비교에 쓰지 마라.',
        f'⚠[§9.3-12] `G-WRAPPER` 와 `G-DEPLOY` 의 `init_method` 항은 배포 상수상 **구조적으로 '
        f'참/항진**이다(`T14 A3 §3`) ⟹ ⛔`gates_pass` 를 오독하지 마라. 같은 부류로 `G-DEEP`·'
        f'`G-EDGE`(BANDR)도 구조적으로 거의 항상 참이다.',
        f'⚠[§9.3-13] **추정기 시드 1개**(median-of-k 아님) ⟹ ⛔시드 간 변동 주장 금지'
        f'(`S13b` 선례: 시드가 부호를 가른 적이 있다).',
        f'⚠[§9.3-14] **검정력이 아니라 blind 가 표본 크기를 정했다** — `mixed_data2`(60)와 OOD(6)는 '
        f'소진됐다 ⟹ 이 아카이브에서 이 규약으로 남은 blind 층은 Tier-P 가 **마지막**이다. '
        f'재실행은 blind 가 아니다.',
        f'⚠[§9.3-15] **등변 편차의 크기는 규약 종속**이다 — 설계 규약 `|k̂_c/(c·k̂₁)−1|` 과 분석기 '
        f'규약 `|s_c·c/s_1−1|` 은 `s ∝ 1/k̂` 라 1차 근사에서만 같다(같은 셀이 1.138e+03 과 0.9991). '
        f'⛔두 수를 나란히 놓아 서로 다른 크기인 것처럼 쓰지 마라. 이 게이트는 **분석기 규약**이다.',
        f'⚠[§9.3-16] `T14` 의 허용오차 5.88% 는 **`c=2.0` 방향에만** 적용됐다 ⟹ `AUTO-BAND` 의 '
        f'`c=0.5` 방향은 `T14` 에서 **한 번도 게이트된 적이 없다**. 이 배치의 `G-EQUIVAR` 는 '
        f'후보 둘에 **1e-6 양방향**이고 `G-EQUIVAR-OLD` 는 허용오차 없는 기록 전용이다.',
        f'⚠[§9.3-17] `SEC-BLIND` 의 미접촉 {len(blind)}파일은 `angle ∈ {{40,50,60}}` 과 **완전 '
        f'교락**이다 ⟹ ⛔PRIMARY 와의 차이를 「노출 효과」로 읽는 것 금지. 대조 = '
        f'`SEC-BLIND-ANGLE30`(n={len(ang30)}, 기술통계, ⚠n 이 달라 검정 비교 불가).',
        f'⚠[§9.3-18] `G-INNER` 는 이 배치가 신설한 게이트다 — 수리는 고장 방향을 **바깥에서 안쪽**'
        f'으로 옮긴다(옛 1단 바닥 klim₂ ≥ 0.0832 는 코퍼스를 항상 덮었고 새 바닥 ≥ 0.0251 은 '
        f'k_p=0.034 를 구조적으로 못 덮을 수 있다) ⟹ ⛔"새 규칙이 더 튼튼하다" 서술 금지. '
        f'잰 것은 **등변성과 격자 여유**이지 **강건성**이 아니다.',
        f'⚠[§0-4 · §3.6] 이 배치는 **적응성을 재지 못한다**(격자가 전 파일 동일 ⟹ `s` 도 `klim₁` 도 '
        f'코퍼스 상수) — 잰 것은 (i) 착지점과 (ii) 등변성이다. 그리고 설계는 결과 전에 '
        f'P(S 축 양성)≈1, P(`D-TREATMENT-BELOW-SCALE`)=0.937 이라고 **스스로 적었다** ⟹ '
        f'⛔이 사실을 결과 뒤에 재서술하지 마라. 병기: {sec_e_txt} · {drive_txt}.',
        f'⚠[§10] 이 배치가 재는 것은 배포 **합성 경로**다 — 상수 `gt_field` 때문에 라우터가 '
        f'`learned` 로 가고(`G-ARM` 되읽기), 규칙이 빌려 쓰는 `bank_mmse._beamform_top` 은 '
        f'배포 학습 경로에서 **0회** 실행된다 ⟹ ⛔실측 경로에 대해 아무 말도 하지 않는다.',
    ]
    cocite = []
    for s in disclosures:
        if ('`D` 축' in s or 'D-TREATMENT' in s) and 'median|Δ ln s|' not in s:
            cocite.append(f'D 진술에 D_drive 없음: {s[:40]}')
        if '`S` 축' in s and 'SEC-E' not in s:
            cocite.append(f'S 진술에 SEC-E 없음: {s[:40]}')
    for lb_ in (lab_S, lab_D):
        neg = ('DETECTED' in lb_ and 'NO-' in lb_) or lb_ == 'D-UNPOWERED'
        if neg and not any('q80' in s for s in disclosures):
            cocite.append(f'음성 라벨 {lb_} 에 MDE 병기 없음')
    cov_bad += cocite
    A.append(('G-COV 사전등록 §5–§6 게이트 표 ↔ 구현 **1:1** · 등록 상수 드리프트 0 · '
              '★§3.4 **병기 의무** 자가검사(모든 D 진술에 D_drive · 모든 S 진술에 SEC-E · '
              '음성 라벨에 MDE) · 게이트가 자료를 봤다',
              not cov_bad and len(rows) > 0,
              f'{cov_bad if cov_bad else "이상 없음"} · rows={len(rows)}'))
    missed, folded = G.selfcheck_blocked_by(emitted + list(G.GATE_NAMES))
    A.append(('G-BLOCK blocked_by 자가진단 — **분석기가 실제로 낸 이름 전건**(자기 자신 포함) + '
              '등록된 전 이름을 합성 위반 튜플로 넣어 빈 리스트가 아님을 확인 [긴 이름 우선]',
              not missed and not folded, f'미포착 {missed} · 접힘 {folded}'))

    # ★라벨 인쇄 벨트 — ⛔차단이 하나라도 있거나 게이트가 하나라도 FAIL 이면 **인쇄하지 않는다**.
    blocked = G.blocked_by(A)
    all_ok = all(bool(o) for _, o, _ in A)
    lab = G.label(w_f, w_s, w_e, w_d, n_eff, t1, t2) if (not blocked and all_ok) else None

    v = {
        'batch': 'T16-BANDFIX', 'kind': 'CONFIRMATORY',
        'grade': ('CONFIRMATORY (Tier-P 60 은 성과 미노출 blind · ⚠속성 노출 36/60 · '
                  '⛔SEC-BLIND 24 는 angle 과 완전 교락)'),
        'n_rows': len(rows), 'n_runs_expected': G.N_RUNS,
        'gates': [{'name': n, 'pass_': bool(o), 'detail': d} for n, o, d in A],
        'gates_pass': f'{sum(1 for _, o, _ in A if o)}/{len(A)}', 'blocked_by': blocked,
        'primary_family': {
            'S (기술 축, 단측, Bonferroni α/2)': {
                'stat': 'W[AUTO-BANDR < ZERO]', 'w': w_s, 'n': n_s,
                'p': G.sign_p(w_s, n_s), 'thresh': G.THRESH_S, 'pass': w_s >= G.THRESH_S},
            'D (착지 축, 양측, Bonferroni α/2)': {
                'stat': 'W[AUTO-BANDR < AUTO-BAND]', 'w': w_d, 'n_eff': n_eff,
                'n_tied': len(tied), 'tie_rel': G.TIE_REL, 'k_plus': kplus,
                'p_two_sided': (None if n_eff == 0 else
                                2 * min(G.sign_p(w_d, n_eff), G.sign_p_hurt(w_d, n_eff))),
                'q80': G.MDE['D_q80']},
            'bonferroni': f'α/{G.FAMILY_SIZE} = {G.ALPHA_FAM}'},
        'positive_control': {'FIX2<ZERO': {'w': w_f, 'n': n_f, 'p': G.sign_p(w_f, n_f)},
                             'thresh': G.THRESH_F, 'pass': w_f >= G.THRESH_F},
        'ship_vs_zero': {'w': w_sh, 'n': n_sh, 'p': G.sign_p(w_sh, n_sh)},
        'sig_worse': {a: (w <= G.SIG_WORSE) for a, (w, _) in
                      (('AUTO-BANDR', (w_s, n_s)), ('AUTO-BAND', (w_e, n_e)),
                       ('AUTO-AP', (w_ap, n_ap)), ('SHIP', (w_sh, n_sh)),
                       ('FIX2', (w_f, n_f)))},
        'D_drive': D_drive,
        'SEC': SEC,
        'recovery_R': {a: R(a) for a in G.AUTO_ARMS},
        'medians': {f'{k[0]}:{k[1]}': float(np.median([r['nrmse'] for r in d.values()]))
                    for k, d in sorted(by.items()) if d},
        'anchor_grade': grade, 's_clip': sclip, 's_var': svar, 'clamp': clamp,
        'equivar_old': old_stat, 'band_res': bres,
        'cost_readback': {'beamform_calls_per_file': cost_ok,
                          'registered': G.BEAMFORM_CALLS_PER_FILE,
                          'device': sorted({r.get('beamform_device') for r in bf
                                            if r.get('beamform_device')})},
        'mde': G.MDE, 'thresholds': {'F': G.THRESH_F, 'S': G.THRESH_S,
                                     'SEC-E': G.THRESH_SECE, 'D': G.THRESH_D,
                                     'SIG_WORSE': G.SIG_WORSE,
                                     'N_EFF_UNPOWERED': G.N_EFF_UNPOWERED},
        'n_eff_table': {k: val for k, val in G.n_eff_thresholds().items() if k % 5 == 0 or k <= 12},
        'prediction': {'q': G.PRED_Q, 't': G.PRED_T, 'label_probs': G.label_probs(),
                       'null_label_probs': G.null_label_probs()},
        'registration_drift': drift,
        'proposal_device': sorted({r.get('proposal_device') for r in rows
                                   if r.get('proposal_device')}),
        'label_S': lab_S, 'label_D': lab_D,
        'label': lab, 'label_msg': lab or 'BLOCKED',
        'disclosures': disclosures,
    }
    json.dump(v, open(os.path.join(OUTDIR, 'verdict_t16.json'), 'w'),
              ensure_ascii=False, indent=1)

    def _p(k, n):
        q = G.sign_p(k, n)
        return 'n/a' if q is None else f'{q:.4g}'

    print(f"T16-BANDFIX :: {v['label_msg']}")
    print(f'  축 S  AUTO-BANDR < ZERO   {w_s}/{n_s}  p={_p(w_s, n_s)}  '
          f'(문턱 {G.THRESH_S}, Bonferroni α/2)  ⟹ {lab_S}')
    print(f'        ★병기 의무 — {sec_e_txt}')
    print(f'  축 D  AUTO-BANDR < AUTO-BAND  {w_d}/{n_eff}  (동점 {len(tied)}, k⁺={kplus})  '
          f'⟹ {lab_D}')
    print(f'        ★병기 의무 — {drive_txt}')
    print(f'  양성대조 FIX2<ZERO {w_f}/{n_f} p={_p(w_f, n_f)} (문턱 {G.THRESH_F})')
    print(f'  하한 SHIP<ZERO {w_sh}/{n_sh} · SEC-B AUTO-AP<ZERO {w_ap}/{n_ap}')
    print(f'  MDE q80  S={G.MDE["S_q80"]} · D={G.MDE["D_q80"]} · BLIND={G.MDE["BLIND_q80"]} '
          f'(⛔섞지 마라)')
    print(f'  s 분포 {svar} · 레일접촉 {sclip}')
    print(f"  회수율 R {v['recovery_R']}")
    print(f"  호출 되읽기 {v['cost_readback']} · 앵커 등급 {grade}")
    print(f"  gates {v['gates_pass']} · blocked_by {blocked}")
    for n, o, d in A:
        print(f"   [{'OK ' if o else 'FAIL'}] {n[:96]:96s} {d}")


if __name__ == '__main__':
    main()
