"""`T17-REALSCALE` 분석기 — ⛔러너보다 **먼저** 쓰였다. ★**이 파일이 행 스키마를 정의한다.**

════════════════════════════════════════════════════════════════════════════════════════
행 스키마 (`PREREG §6.4`) — ⛔열이 빠지면 게이트가 **발화 불가**다.
    러너는 아래를 **되읽기로** 채운다. ⛔러너가 스스로 쓴 리터럴을 그대로 다시 비교하는
    열은 게이트가 되지 못한다(항진) — 그런 열에는 「되읽기 지점」을 함께 적었다.

  식별      cell_id(`record|wN`) · record · win · site · date · arm · phase
  관측·창   n_obs · sigma · Hs · m_scan · m_cap · noise · W_obs · cap_mode · leads
            t_start · t_end (★**관측창의 record 시각** — `G-WIN` 이 인접 창 겹침을 잰다.
                             ⚠`meta` 의 동명 키는 **벽시계**다: `PREREG §8` 이 그렇게 등록했다)
            obs_hash(★**재라벨 전** P,z — 셀 안에서 팔 간 동일) · q_hash
            obs_hash_used(★재라벨 **후** 실제로 프레임에 들어간 배열 — `s` 가 다르면 달라야 한다)
            n_offered · n_gathered · cap_proposal · gt_present(★프레임 되읽기: 참값 부재)
  채점      nowcast · leadmean · curve · nrmse_explicit · zero_true(★`zero_ref` **함수** 호출)
  ★규칙    s(팔이 실제로 쓴 값) · s_raw(**클리핑 전**) · s_clipped
            s_rule_bandr · s_raw_bandr · s_rule_ap · s_raw_ap · s_raw_band_old(`G-DRIVE`)
            s_blind_bandr · s_blind_ap(★**별도 프로세스** 재실행)
            s_readback(★프레임이 실제로 가진 좌표의 ptp 비) · x_scale · t_scale · disp_relerr
            rule_ids(`id()` — `G-RULEID`)
  ★빔포머  klim1 · dk1 · kp1_hat · kp2_hat · floor_margin
            edge_hit · edge_margin(★**1단** — §6.3 "1단 argmax 가 격자 가장자리")
            margin(★**2단** `k̂₂/Δk₁` ≡ `L_obs/λ̂_p` — `PREREG §0.2` 프로브 B 의 정의.
                   ⛔초판이 1단을 썼고 **스모크가 잡았다**: BS11a 67.974 vs 등록 226.136)
            margin_stage1(1단 자 — ⚠다른 양이다. ⛔둘을 섞지 마라)
            klim1_prime · dk1_prime · kp1_hat_prime · kp2_hat_prime
              (★팔의 **재라벨된 관측**에서 규칙을 다시 돌린 값 — `G-DEPTH-EXACT` 의 kd 항)
            beamform_calls · beamform_calls_rule · beamform_calls_bank(★`PREREG §10` 분리 계수)
            beamform_device · proposal_device
  ★수심    depth(원) · depth_used(★**소비자 되읽기**: 모델 객체 / 빔포머가 받은 인자)
            depth_cfg_readback(`upd.config`) · kd(= kp2_hat·depth) · kd_prime(= kp2_hat_prime·depth_used)
  팔 정체   arm_provenance(★`upd.arm_provenance()` **전체**) · effective(배포 상수 6종 `cfg` 되읽기)
            arm_cfg(그 팔이 `DEFAULT_CONFIG` 위에 명시한 키) · n_atoms · varpro_fired
            varpro_steps · init_iters · vp_clamp_hits · proposal_md5 · atoms_md5
            use_stokes2 · use_neural_corrector
  시드      seed · seed_applied(★`run.set_seed` 후크가 잡은 **적용된** 값)
            rng_md5_seeded(★적용 직후 전역 RNG 지문 — 분석기가 **독립 재계산**해 댄다)
  동결      git_head · import_md5 · ckpt_sha256 · wall_s
  ⚠오류행   error · tb   (⛔`G-N` 이 0 을 요구한다)

  ⚠`BANK` 팔은 학습 경로가 아니다 ⟹ `proposal_md5`·`atoms_md5`·`nrmse_explicit`·
    `use_stokes2`·`use_neural_corrector` 가 `None`, `n_atoms == 0` 이다(게이트가 건너뛴다).
  ⛔`(x or 기본값)` 을 쓰지 마라 — `0.0` 은 falsy 다. 이 파일은 전부 `is None` 으로 가른다.
════════════════════════════════════════════════════════════════════════════════════════

⛔**스모크 판정을 배치 디렉터리에 쓰지 마라** — `--out <다른 경로>` 로 보낸다
  (`T15` 가 스모크 verdict 으로 배치를 오염시켰다).
"""
import argparse
import ast
import hashlib
import inspect
import json
import math
import os
import statistics as st
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gates_t17 as G                                              # noqa: E402
import scale_rules_t14 as SR14                                     # noqa: E402
import zero_ref as ZR                                              # noqa: E402

ROWS = os.path.join(HERE, 'results_t17.jsonl')
META = os.path.join(HERE, 'meta_t17.json')
PREREG = os.path.join(HERE, 'PREREG.md')
FROZEN = os.path.join(HERE, 'FROZEN_md5.txt')
FROZEN_A = os.path.join(HERE, 'FROZEN_md5_A.txt')
ANALYSIS_SIDE = ('gates_t17.py', 'analyze_t17.py')
# ★규칙이 보면 안 되는 토큰 — **장면 dict 과 참값 경로**의 이름들.
#   ⛔초판에 있던 `'kp'` 를 뺐다: 그것은 `scale_rules_t16.kp_hat_rel` **자신의 지역 변수 이름**
#   (`kp = max(hypot(bx,by), FLOOR)`)이라 임포트만으로 FAIL 했다 — **스모크가 잡았다**.
#   ⚠이것은 완화가 아니라 **오탐 제거**다: 그 이름은 규칙의 산출이지 참값이 아니고,
#   실제 누출 경로(`sc['maps_gt']`·`sc['sg']`·`sc['kp']` = FFT 첨두)는 아래 이름으로 잡힌다.
#   그리고 하중은 `G-S-BLIND`(별도 프로세스, 행동 검사)가 진다.
BANNED = ('target', 'gt', 'eta', 'maps_gt', 'Zfut', 'Zmem', 'sg', 'Hs', 'sigma4',
          'record', 'win', 'scene_station', 'directional_spread', 'zero_ref', 'MRecord')


# ── 작은 도우미 ─────────────────────────────────────────────────────────────────────
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


def _stat(v):
    v = [x for x in v if x is not None]
    if not v:
        return None
    return {'n': len(v), 'median': float(np.median(v)), 'min': float(min(v)),
            'max': float(max(v))}


def _frozen(p):
    if not os.path.exists(p):
        return None
    out = {}
    for line in open(p, encoding='utf-8'):
        if line.strip() and not line.startswith('#'):
            h, k = line.split(None, 1)
            out[k.strip()] = h.strip()
    return out


def _prereg_gate_names():
    """★사전등록 **§6** 이 등록한 게이트 이름(표의 첫 칸).

    ⛔§16(개정 이력)은 스캔하지 않는다 — 거기엔 R-1·R-2 가 **일부러 인쇄한 유령 이름**
    (`…G-CHANGE-DETECTED`·`SI G-WORSE`)이 있다.
    """
    if not os.path.exists(PREREG):
        return None
    t = open(PREREG, encoding='utf-8').read()
    try:
        sec = t[t.index('## §6 '):t.index('## §7 ')]
    except ValueError:
        return None
    out = set()
    for ln in sec.splitlines():
        s = ln.strip()
        if not s.startswith('|'):
            continue
        first = s.strip('|').split('|')[0].strip()
        m = G.PHANTOM_RE.pattern and __import__('re').search(
            r'G-[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*',
            first.replace('★', '').replace('⚠', '').replace('⛔', ''))
        if m:
            out.add(m.group(0))
    return out


def _rule_src_tokens(fn, seen=None):
    """★`getsource(**함수**)` 를 `ast.parse` 해 **식별자·속성명·인자명**만 모은다.

    ⛔`getsource(모듈)`·파일 텍스트 검색 금지 — `scale_rules_t14.py` **독스트링**이 금지
    토큰을 담고 있어 모듈 단위로 구현하면 임포트만으로 FAIL 한다.
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


def _seed_fp(seed):
    """★분석기가 **독립 재계산**하는 전역 RNG 지문 (`G-SEED`).

    ⛔러너 리터럴을 분석기 리터럴과 대조하면 실패할 수 없는 게이트가 된다.
    """
    st0 = np.random.get_state()
    try:
        np.random.seed(int(seed))
        s = np.random.get_state()
        return hashlib.md5(np.asarray(s[1]).tobytes() + str(s[2]).encode()).hexdigest()
    finally:
        np.random.set_state(st0)


def main(rows_path=ROWS, meta_path=META, outdir=HERE, tag='t17'):    # noqa: C901
    rows = [json.loads(x) for x in open(rows_path, encoding='utf-8') if x.strip()]
    meta = json.load(open(meta_path, encoding='utf-8')) if os.path.exists(meta_path) else {}
    ok_rows = [r for r in rows if 'error' not in r]
    err_rows = [r for r in rows if 'error' in r]
    by = {}
    for r in ok_rows:
        by.setdefault(r['arm'], {})[r['cell_id']] = r
    cellset = sorted({r['cell_id'] for r in ok_rows})
    prop = [r for r in ok_rows if r['arm'] in G.PROP_ARMS]
    bank = [r for r in ok_rows if r['arm'] == 'BANK']
    A = []

    # ── 통계 도우미 ──────────────────────────────────────────────────────────────
    def wins_zero(arm, cells=None):
        d = by.get(arm, {})
        ks = sorted(d) if cells is None else sorted(set(d) & set(cells))
        ks = [k for k in ks if _num(d[k].get('nowcast')) is not None
              and _num(d[k].get('zero_true')) is not None]
        return sum(1 for k in ks if d[k]['nowcast'] < d[k]['zero_true']), len(ks)

    def paired(arm, ref, cells=None, field='nowcast'):
        """짝 부호검정 — ⛔동점은 `TIE_REL` 로 제외하고 `n_eff` 로 검정한다(§3.1)."""
        a, b = by.get(arm, {}), by.get(ref, {})
        ks = sorted(set(a) & set(b))
        if cells is not None:
            ks = [k for k in ks if k in set(cells)]
        ks = [k for k in ks if _num(a[k].get(field)) is not None
              and _num(b[k].get(field)) is not None]
        tied = [k for k in ks
                if abs(a[k][field] - b[k][field]) / max(abs(b[k][field]), 1e-12) < G.TIE_REL]
        un = [k for k in ks if k not in set(tied)]
        w = sum(1 for k in un if a[k][field] < b[k][field])
        rel = [(a[k][field] - b[k][field]) / b[k][field] for k in ks if b[k][field] != 0]
        return {'w': w, 'n_eff': len(un), 'n': len(ks), 'n_tied': len(tied),
                'tie_rel': G.TIE_REL,
                'rel_median_pct': (float(np.median(rel)) * 100.0 if rel else None),
                'p_one_sided': G.sign_p(w, len(un)),
                'p_two_sided': (None if not un else
                                2 * min(G.sign_p(w, len(un)), G.sign_p_hurt(w, len(un))))}

    def fold(arm, ref, field='nowcast'):
        """`C-FOLD` 입력 — 레코드 안 6창의 짝별 상대차 `(a−b)/b` (§3.4)."""
        a, b = by.get(arm, {}), by.get(ref, {})
        rel = {}
        for k in sorted(set(a) & set(b)):
            va, vb = _num(a[k].get(field)), _num(b[k].get(field))
            if va is None or vb is None or vb == 0:
                continue
            rel.setdefault(a[k]['record'], []).append((va - vb) / vb)
        return G.fold_sign(rel)

    def fold_zero(arm):
        """`C-FOLD` 입력 (S 축) — 상대차는 `zero_true` 대비."""
        d = by.get(arm, {})
        rel = {}
        for k in sorted(d):
            va, vz = _num(d[k].get('nowcast')), _num(d[k].get('zero_true'))
            if va is None or vz is None or vz == 0:
                continue
            rel.setdefault(d[k]['record'], []).append((va - vz) / vz)
        return G.fold_sign(rel)

    # ═══ §6.1 차단 게이트 ══════════════════════════════════════════════════════════
    # ★★G-ARM — 팔 정체는 **의도(config)** 가 아니라 `arm_provenance()` **되읽기**다.
    #   FAIL 조건 = 라우터가 실제로 간 곳이 팔 표와 다르다(`E1-EXTBASE` 사고의 형태).
    ab, fin = [], {}
    for r in ok_rows:
        pv = r.get('arm_provenance')
        tag_ = f"{r['arm']}:{r['cell_id']}"
        if not isinstance(pv, dict):
            ab.append(f'{tag_}:prov 없음')
            continue
        fin.setdefault(r['arm'], set()).add(pv.get('resolved_is_final'))
        want_req = G.ARM_CFG[r['arm']]['init_method']
        if pv.get('requested_init_method') != want_req:
            ab.append(f"{tag_}:requested={pv.get('requested_init_method')}≠{want_req}")
        na = r.get('n_atoms')
        if r['arm'] == 'BANK':
            if pv.get('resolved_init_method') != 'bank_mmse':
                ab.append(f"{tag_}:resolved={pv.get('resolved_init_method')}")
            if pv.get('real_domain_gate_fired') is not True:
                ab.append(f'{tag_}:실측게이트 미발화')
            if pv.get('is_bank_mmse') is not True:
                ab.append(f'{tag_}:is_bank_mmse')
            if na != 0:
                ab.append(f'{tag_}:n_atoms={na}')
        else:
            if pv.get('resolved_init_method') != 'learned':
                ab.append(f"{tag_}:resolved={pv.get('resolved_init_method')}")
            if pv.get('real_domain_gate_fired') is not False:
                ab.append(f'{tag_}:실측게이트 발화')
            if na is None or na <= 0:
                ab.append(f'{tag_}:n_atoms={na}')
            if r.get('varpro_fired') is not True:
                ab.append(f'{tag_}:varpro 미발화')
    A.append(('★★G-ARM 팔 정체 = `arm_provenance()` **전체 되읽기** — requested 는 팔 표대로 · '
              'BANK: resolved==bank_mmse ∧ 실측게이트 발화 ∧ is_bank_mmse ∧ n_atoms==0 · '
              'PROP*: resolved==learned ∧ 게이트 미발화 ∧ n_atoms>0 ∧ varpro 발화 '
              '[⛔`requested != resolved` 일 수 있다는 것이 요점이다 — BANK 이 바로 그 경우다]',
              bool(ok_rows) and not ab,
              f'위반 {len(ab)} {ab[:6]} · ⚠[기록] resolved_is_final {fin} — '
              f'⛔`_initialized` 는 패키지 어디에도 대입되지 않아(전수 grep 0건) 명시 '
              f'`init_method` 팔에서는 **구조적으로 False** 다 ⟹ 이 항목을 요구하지 않는다'))

    # ★G-NOEMPTY — ⛔`{}` 팔 부재 + 배포 상수 6종 **cfg 되읽기**.
    #   FAIL 조건 = 팔 정의가 비었다 / 러너가 명시한 키가 객체에 없다 / 배포 상수가 다르다.
    nb = [a for a, c in G.ARM_CFG.items() if not c]
    want_dep = G.deploy_consts()
    for r in ok_rows:
        eff = r.get('effective')
        if not isinstance(eff, dict):
            nb.append(f"{r['arm']}:{r['cell_id']}:effective 없음")
            continue
        for k in G.DEPLOY_KEYS:
            if k not in eff:
                nb.append(f"{r['arm']}:{r['cell_id']}:{k} 누락")
            elif isinstance(want_dep[k], float):
                if _rel(_num(eff[k]), want_dep[k]) > 0:
                    nb.append(f"{r['arm']}:{r['cell_id']}:{k}={eff[k]}")
            elif eff[k] != want_dep[k]:
                nb.append(f"{r['arm']}:{r['cell_id']}:{k}={eff[k]}")
        acfg = r.get('arm_cfg')
        if not isinstance(acfg, dict) or not acfg:
            nb.append(f"{r['arm']}:{r['cell_id']}:arm_cfg 비었다")
        elif acfg != G.ARM_CFG[r['arm']]:
            nb.append(f"{r['arm']}:{r['cell_id']}:arm_cfg 불일치")
    A.append(('★G-NOEMPTY ⛔빈 dict 팔 0개 · 배포 상수 6종을 `upd.config` **되읽기**로 대조 '
              '(⛔손복사 0 — `sequential_update` 에서 임포트) · 팔이 명시한 키 되읽기',
              bool(ok_rows) and not nb, f'위반 {len(nb)} {nb[:4]} · 기대 {want_dep}'))

    # ★★G-ZEROREF — 채점 기준이 **함수**다. FAIL 조건 = 상수 리터럴이 있다 / 분석기의 독립
    #   재계산(캐시 미사용)이 행과 다르다 / win=3 값이 동결 캐시와 비트 불일치 / selftest 실패.
    zb, zvals = [], []
    try:
        zself = ZR.selftest()
    except Exception as e:                                          # noqa: BLE001
        zself, _ = None, zb.append(f'selftest 실패: {type(e).__name__}: {e}')
    zb += [f'리터럴 {h}' for h in G.zero_ref_literal_hits()]
    cache = G.zero_true_cached()
    seen_cell = {}
    for r in ok_rows:
        seen_cell.setdefault(r['cell_id'], r)
    for cid, r in sorted(seen_cell.items()):
        got = _num(r.get('zero_true'))
        zvals.append(got)
        try:
            want = G.zero_true(r['record'], r['win'], use_cache=False)
        except Exception as e:                                      # noqa: BLE001
            zb.append(f'{cid}:재계산 실패 {type(e).__name__}')
            continue
        if got is None or got != want:
            zb.append(f'{cid}:행 {got} ≠ 재계산 {want}')
        if int(r['win']) == G.ANCHOR_WIN and cache is not None:
            if r['record'] not in cache:
                zb.append(f'{cid}:캐시에 없다')
            elif float(cache[r['record']]) != want:
                zb.append(f'{cid}:캐시 {cache[r["record"]]} ≠ 재계산 {want}')
    # ⛔상수 1.0 과 구분되지 않으면 시험이 공허하다
    vv = [v for v in zvals if v is not None]
    if vv and 0.99 < min(vv) and max(vv) < 1.01:
        zb.append('전 셀이 1.0 과 구분되지 않는다 — 시험이 공허하다')
    A.append(('★★G-ZEROREF 채점 기준은 상수가 아니라 **함수**다 — `zero_ref.zero_ref_nowcast` 를 '
              '분석기가 **캐시 없이** 다시 돌려 행과 대조 · win=3 은 동결 캐시와 비트 일치 · '
              'AST 에 `ZERO_REF = 1.0` 형태 0건 · selftest 통과',
              bool(ok_rows) and not zb, f'위반 {len(zb)} {zb[:4]} · selftest {zself} · '
              f'값 {_stat(zvals)}'))

    # ★G-S-READBACK — 러너 리터럴이 아니라 **프레임이 실제로 가진 좌표**에서 되읽는다.
    #   FAIL 조건 = `1/median(x'/x) != s` · `median(t'/t) != 1/√s` (지수 오류 탐지).
    rb = []
    for r in ok_rows:
        s, sr = _num(r.get('s')), _num(r.get('s_readback'))
        ts, dr = _num(r.get('t_scale')), _num(r.get('disp_relerr'))
        tag_ = f"{r['arm']}:{r['cell_id']}"
        if s is None or sr is None or ts is None or dr is None:
            rb.append(f'{tag_}:missing')
            continue
        if _rel(sr, s) > G.EQ_TOL:
            rb.append(f'{tag_}:s_readback {sr}≠{s}')
        if _rel(ts, 1.0 / math.sqrt(s)) > G.EQ_TOL:
            rb.append(f'{tag_}:t_scale {ts}≠1/√s')
        if dr > G.EQ_TOL:
            rb.append(f'{tag_}:disp_relerr {dr}')
    A.append((f'★G-S-READBACK 변환된 좌표에서 s 되읽기 — 1/median(x\'/x)==s ∧ median(t\'/t)==1/√s '
              f'∧ 분산관계 잔차 ≤ {G.EQ_TOL} [⛔해석적 항등식이 아니다 — 지수를 틀리면 죽는다]',
              bool(ok_rows) and not rb, f'위반 {len(rb)} {rb[:4]}'))

    # ★★G-S-BLIND — 관측 5배열만 별도 프로세스에 넘겨 재실행 ⟹ `s` **비트 동일**.
    #   FAIL 조건 = 규칙이 관측 밖의 무언가(파일명·참값·전역 상태)를 봤다.
    sb = []
    for cid, r in sorted(seen_cell.items()):
        for k in ('bandr', 'ap'):
            got, want = r.get(f's_blind_{k}'), r.get(f's_rule_{k}')
            if got is None or want is None or got != want:
                sb.append(f'{cid}:{k} {got}≠{want}')
    A.append(('★★G-S-BLIND 관측 5배열만 별도 프로세스에 넘겨 규칙 재실행 ⟹ s 비트 동일 '
              '[행동 검사 — ⛔이것이 하중을 진다]',
              bool(seen_cell) and not sb, f'위반 {len(sb)}/{2 * len(seen_cell)}'))

    # ★G-S-SRC — 정적. FAIL 조건 = 규칙(또는 같은-모듈 보조 함수)의 **식별자**에 표적/메타
    #   토큰이 있다 / 시그니처가 5인자가 아니다 / 전역에 표적 크기 배열이 있다.
    ss = []
    if G.SR16 is None:
        ss.append('scale_rules_t16 임포트 실패')
    else:
        for nm in ('AUTO-BANDR', 'AUTO-AP'):
            fn = G.SR16.RULES.get(nm)
            if fn is None:
                ss.append(f'{nm}:규칙 없음')
                continue
            if list(inspect.signature(fn).parameters) != ['x', 'y', 't', 'z', 'depth']:
                ss.append(f'{nm}:sig')
            toks = _rule_src_tokens(fn)
            ss += [f'{nm}:{tk}' for tk in BANNED if tk in toks]
            ss += [f'{nm}:glob:{g}' for g in _globals_have_target(fn)]
    A.append(('★G-S-SRC 규칙 5인자 · **AST 식별자**에 표적/메타 토큰 0회 · 전역에 표적 배열 0개 '
              '[⚠단독 항진 위험 — `G-S-BLIND` 가 하중을 진다]', not ss, f'위반 {ss[:6] or "없음"}'))

    # ★G-RULEID — 규칙 단일출처. FAIL 조건 = 손복사본이 끼어들었다.
    ri, ids = [], {}
    if G.SR16 is None:
        ri.append('scale_rules_t16 임포트 실패')
    else:
        ids = {'s_bandr': id(G.SR16.s_bandr), 's_ap': id(SR14.s_ap),
               'RULES[AUTO-BANDR]': id(G.SR16.RULES.get('AUTO-BANDR')),
               'RULES[AUTO-AP]': id(G.SR16.RULES.get('AUTO-AP'))}
        if G.SR16.RULES.get('AUTO-BANDR') is not G.SR16.s_bandr:
            ri.append('RULES[AUTO-BANDR] is not SR16.s_bandr')
        if G.SR16.RULES.get('AUTO-AP') is not SR14.s_ap:
            ri.append('RULES[AUTO-AP] is not SR14.s_ap')
        if G.SR16.s_ap is not SR14.s_ap:
            ri.append('SR16.s_ap is not SR14.s_ap (손복사)')
        for k, v in (('S_MIN', SR14.S_MIN), ('S_MAX', SR14.S_MAX),
                     ('L_TRAIN', SR14.L_TRAIN), ('K_TRAIN', SR14.K_TRAIN)):
            if getattr(G, k) != v:
                ri.append(f'{k} 손복사')
        for k, v in (('N1', G.SR16.N1), ('ZOOM', G.SR16.ZOOM), ('FLOOR', G.SR16.FLOOR)):
            if getattr(G, k) != v:
                ri.append(f'{k} 손복사')
        # ★러너가 **실제로 부른 함수**의 `module.qualname` 이 등록된 것과 같아야 한다.
        #   ⛔`id()` 로 대조하지 않는다 — 별도 프로세스라 **원리적으로 다르다**(스모크가 잡았다).
        live_qn = {k: G.qualname_of(f) for k, f in
                   (('s_bandr', G.SR16.s_bandr), ('s_bandr_raw', G.SR16.s_bandr_raw),
                    ('s_ap', SR14.s_ap), ('s_ap_raw', SR14.s_ap_raw),
                    ('s_band_raw', SR14.s_band_raw),
                    ('kp_hat_rel', G.SR16.kp_hat_rel), ('clip_s', G.SR16.clip_s))}
        if live_qn != G.RULE_QUALNAMES:
            ri.append(f'라이브 qualname ≠ 등록 {live_qn}')
        for cid, r in sorted(seen_cell.items()):
            got = r.get('rule_qualnames')
            if got != G.RULE_QUALNAMES:
                ri.append(f'{cid}:러너 qualname 불일치 {got}')
                break
    A.append(('★G-RULEID 규칙 단일출처 — 함수는 분석기 안에서 **객체 동일성**(`is`) · 상수 7종은 '
              '임포트 대조 · 러너가 부른 함수의 `module.qualname` == 등록값 [⛔손복사 0 · '
              '⛔`id()` 로 프로세스 간 대조를 하지 않는다]',
              not ri, f'위반 {ri[:5] or "없음"} · id(분석기 안) {ids}'))

    # ★★G-DEPTH-EXACT — §5 규약 `d' = d/s`. FAIL 조건 = 러너가 수심을 안 나눴다(그러면
    #   `k'd' ≠ kd` 로 분산관계가 깨진다) / 소비자가 받은 수심이 다르다.
    db, kdrel = [], []
    for r in ok_rows:
        s, d0, du = _num(r.get('s')), _num(r.get('depth')), _num(r.get('depth_used'))
        kd, kdp = _num(r.get('kd')), _num(r.get('kd_prime'))
        tag_ = f"{r['arm']}:{r['cell_id']}"
        if s is None or d0 is None or du is None:
            db.append(f'{tag_}:missing depth')
            continue
        if abs(du - d0 / s) > G.DEPTH_TOL * d0:
            db.append(f'{tag_}:depth_used {du} ≠ {d0}/{s}')
        if kd is None or kdp is None:
            db.append(f'{tag_}:missing kd')
            continue
        kdrel.append(abs(kdp - kd) / abs(kd) if kd else float('inf'))
        if abs(kdp - kd) > G.KD_TOL * abs(kd):
            db.append(f'{tag_}:k̂\'d\' {kdp} ≠ k̂d {kd}')
    A.append((f'★★G-DEPTH-EXACT §5 규약 — |depth_used − depth/s| ≤ {G.DEPTH_TOL}·depth [소비자 '
              f'되읽기] ∧ |k̂\'·d\' − k̂·d| ≤ **{G.KD_TOL}**·k̂d [★재라벨된 관측에서 규칙 재실행]. '
              f'⚠kd 항의 자는 사전등록 §6.1 의 `{G.EQ_TOL}` 에서 **완화됐다** — 빔포머 격자가 '
              f'float32(eps {G.FLOAT32_EPS:.3e})라 `1e-9` 은 원리적으로 도달 불가다'
              f'(`AMENDMENT_1_PREDATA.md §2`). 수심 항의 자 `{G.DEPTH_TOL}` 는 **그대로**다',
              bool(ok_rows) and not db, f'위반 {len(db)} {db[:4]} · kd 상대차 {_stat(kdrel)}'))

    # ★★G-RAILID — 레일 축퇴를 **팔이 가른다**. FAIL 조건 = 클립 셀에서 두 팔이 다르다
    #   (= 팔 정의가 거짓) / 클립 셀이 하나도 없다(= 게이트가 공허하다).
    clip_cells = [cid for cid, r in sorted(seen_cell.items())
                  if _num(r.get('s_rule_bandr')) is not None
                  and float(r['s_rule_bandr']) == G.S_MIN]
    rid = []
    a1, a2 = by.get('PROP-BANDR', {}), by.get('PROP-S025', {})
    for cid in clip_cells:
        x, y = a1.get(cid), a2.get(cid)
        if x is None or y is None:
            rid.append(f'{cid}:행 없음')
            continue
        if x.get('s') != y.get('s'):
            rid.append(f"{cid}:s {x.get('s')}≠{y.get('s')}")
        if x.get('nowcast') != y.get('nowcast'):
            rid.append(f"{cid}:nRMSE {x.get('nowcast')}≠{y.get('nowcast')}")
        if x.get('proposal_md5') != y.get('proposal_md5'):
            rid.append(f'{cid}:제안 지문')
    A.append((f'★★G-RAILID 레일(`S_MIN={G.S_MIN}`) 클립 셀에서 PROP-BANDR 과 PROP-S025 가 '
              f'**비트 동일**(s · nRMSE · 제안 지문) [⛔다르면 팔 정의가 거짓 · 클립 0셀이면 공허]',
              bool(clip_cells) and not rid, f'클립 {len(clip_cells)}셀 · 위반 {len(rid)} {rid[:4]}'))

    # ★G-OBS — 같은 자료를 받았나 + `PREREG §10` 의 호출 분리.
    #   FAIL 조건 = 셀 안에서 팔이 다른 관측을 받았다 / 재라벨이 좌표를 안 바꿨다 /
    #              학습 경로가 bank 빔포머를 불렀다(§10 이 0회라 주장한다).
    ob = []
    for cid in cellset:
        hs = {r['obs_hash'] for r in ok_rows if r['cell_id'] == cid}
        qs = {r['q_hash'] for r in ok_rows if r['cell_id'] == cid}
        if len(hs) != 1 or len(qs) != 1:
            ob.append(f'{cid}:obs/q 해시 {len(hs)}/{len(qs)}종')
        used = {}
        for r in ok_rows:
            if r['cell_id'] != cid:
                continue
            used.setdefault(round(float(r['s']), 12), set()).add(r.get('obs_hash_used'))
        for sval, hh in used.items():
            if len(hh) != 1:
                ob.append(f'{cid}:s={sval} 에서 재라벨 해시 {len(hh)}종')
        if len(used) > 1 and len({next(iter(h)) for h in used.values()}) != len(used):
            ob.append(f'{cid}:서로 다른 s 가 같은 재라벨 배열을 냈다')
    for r in ok_rows:
        nb_ = r.get('beamform_calls_bank')
        if nb_ is None:
            ob.append(f"{r['arm']}:{r['cell_id']}:호출 계수 없음")
        elif r['arm'] == 'BANK' and nb_ <= 0:
            ob.append(f"BANK:{r['cell_id']}:bank 빔포머 0회")
        elif r['arm'] in G.PROP_ARMS and nb_ != 0:
            ob.append(f"{r['arm']}:{r['cell_id']}:학습 경로가 bank 빔포머를 {nb_}회 불렀다")
    A.append(('★G-OBS 셀 안에서 **재라벨 전** 관측·질의 해시 동일 ∧ 재라벨 후 배열은 s 별로 갈린다 '
              '∧ `PREREG §10` 호출 분리(bank 빔포머: BANK>0 · PROP* ==0)',
              bool(ok_rows) and not ob, f'위반 {len(ob)} {ob[:4]}'))

    # ★G-ATOMS — 캐시 누출. FAIL 조건 = 같은 팔의 서로 다른 셀이 같은 제안 지문을 냈다.
    at = []
    for arm in G.PROP_ARMS:
        d = by.get(arm, {})
        fps = [(cid, r.get('proposal_md5')) for cid, r in sorted(d.items())]
        miss = [c for c, f in fps if f is None]
        if miss:
            at.append(f'{arm}:지문 없음 {len(miss)}')
        vals = [f for _, f in fps if f is not None]
        if len(set(vals)) != len(vals):
            at.append(f'{arm}:중복 {len(vals) - len(set(vals))}')
    for r in bank:
        if r.get('proposal_md5') is not None:
            at.append(f"BANK:{r['cell_id']}:제안 지문이 있다(bank 경로인데)")
    A.append(('★G-ATOMS 제안 지문이 **같은 팔의 셀 사이**에서 전건 상이 (캐시 누출 탐지) · '
              'BANK 은 제안 0 [러너는 팔마다 `dict(DEFAULT_CONFIG)` 를 새로 만들고 assert 한다]',
              bool(prop) and not at, f'위반 {at[:4] or "없음"}'))

    # G-PURE — 역변환 전제. FAIL 조건 = 2차 Stokes/신경 보정이 켜졌다(그러면 §5 역변환이 깨진다).
    pb = [f"{r['arm']}:{r['cell_id']}" for r in prop
          if r.get('use_stokes2') is not False or r.get('use_neural_corrector') is not False]
    A.append(('G-PURE use_stokes2 · use_neural_corrector 둘 다 False [모델 객체 되읽기 · '
              'PROP* 전건 — §5 역변환의 전제다]', bool(prop) and not pb, f'위반 {len(pb)} {pb[:4]}'))

    # ★G-INVERT — §5 두 경로 일치. FAIL 조건 = 암묵/명시 역변환이 갈린다(= 재라벨 사상이 틀렸다).
    ib, iv = [], []
    for r in prop:
        md, sg = _num(r.get('invert_maxdiff')), _num(r.get('sigma'))
        if md is None or sg is None:
            ib.append(f"{r['arm']}:{r['cell_id']}:missing")
            continue
        iv.append(md / sg)
        if md > G.INVERT_TOL_REL * sg:
            ib.append(f"{r['arm']}:{r['cell_id']}:{md:.3e} > {G.INVERT_TOL_REL}·σ")
    A.append((f'★G-INVERT 암묵(재라벨 격자 질의) / 명시(원자를 원 단위로 되돌려 원 격자 렌더) '
              f'두 경로 차 ≤ {G.INVERT_TOL_REL}·σ [필드 차이로 잰다 · ⛔비트 동일 요구 아님]',
              bool(prop) and not ib, f'위반 {len(ib)} {ib[:3]} · maxdiff/σ {_stat(iv)}'))

    # ★G-N — 표본 전수. FAIL 조건 = 부분 실행 / 에러 / 동결 표본과 불일치.
    samp = G.frozen_sample()
    want_cells = [G.cell_id(rec, w) for rec, w in G.cells()]
    nbad = []
    if samp is None:
        nbad.append('sample_t17.json 없다')
    elif sorted(samp) != sorted(want_cells):
        nbad.append(f'동결 표본 {len(samp)} ≠ 계산 {len(want_cells)}')
    if err_rows:
        nbad.append(f'에러 {len(err_rows)}')
    if len(rows) != G.N_ROWS:
        nbad.append(f'행 {len(rows)} ≠ {G.N_ROWS}')
    if sorted(cellset) != sorted(want_cells):
        nbad.append(f'셀 {len(cellset)} ≠ {len(want_cells)}')
    for arm in G.ARM_ORDER:
        if len(by.get(arm, {})) != G.N_CELL:
            nbad.append(f'{arm}:{len(by.get(arm, {}))}')
    A.append((f'★G-N 총 {G.N_ROWS}행 = {G.N_ARM}팔 × {G.N_CELL}셀 · 에러 0 · 동결 표본과 일치 '
              f'[⛔부분 실행으로 라벨을 만들지 않는다]', not nbad,
              f'{nbad if nbad else "이상 없음"} · rows={len(rows)}'))

    # ★★G-SEED — `run.set_seed` 후크가 잡은 **적용된** 시드 + 분석기의 **독립 재계산**.
    #   FAIL 조건 = 러너가 적은 시드가 적용되지 않았다 / RNG 지문이 다르다.
    #   ⛔러너 리터럴을 분석기 리터럴과 대조하면 실패할 수 없는 게이트다(`check_prereg C6a`).
    sd = [f"{r['arm']}:{r['cell_id']}" for r in ok_rows
          if r.get('seed_applied') != r.get('seed')
          or r.get('rng_md5_seeded') != _seed_fp(r['seed'])]
    A.append(('★★G-SEED 추정기에 **적용된** 시드를 `run.set_seed` 후크에서 되읽고, 그 직후 전역 '
              'RNG 지문을 분석기가 **독립 재계산**해 대조 [⛔사전등록 §6 표에 없는 추가 게이트 — '
              'GATES_ADDED 가 사유를 적는다]', bool(ok_rows) and not sd, f'위반 {len(sd)}'))

    # ★G-ENV — 동결 3중(지금 재계산 == meta == 0행 동결) + 분석면 + 스레드 되읽기.
    import run_t17 as R17                                            # noqa: E402
    now = R17.freeze_map()
    rec_md5 = meta.get('code_md5')
    fz0, fza = _frozen(FROZEN), _frozen(FROZEN_A)
    run_keys = sorted(k for k in now if k not in ANALYSIS_SIDE)
    ana_keys = sorted(k for k in now if k in ANALYSIS_SIDE)
    d_meta = [] if not isinstance(rec_md5, dict) else [k for k in run_keys
                                                       if now[k] != rec_md5.get(k)]
    d_fz0 = None if fz0 is None else sorted(set(run_keys) | set(fz0) - set(ANALYSIS_SIDE))
    d_fz0 = None if fz0 is None else [k for k in run_keys if now[k] != fz0.get(k)]
    d_fza = None if fza is None else sorted(k for k in set(ana_keys) | set(fza)
                                            if now.get(k) != fza.get(k))
    eb = []
    if not isinstance(rec_md5, dict):
        eb.append('meta.code_md5 없다')
    eb += [f'meta 대비 {k}' for k in d_meta]
    if d_fz0 is None:
        eb.append('FROZEN_md5.txt 없다')
    else:
        eb += [f'0행동결 대비 {k}' for k in d_fz0]
    if d_fza is None:
        eb.append('FROZEN_md5_A.txt 없다')
    else:
        eb += [f'분석면 대비 {k}' for k in d_fza]
    if meta.get('md5_stable') is not True:
        eb.append('러너 시작/끝 md5 불안정')
    if meta.get('threads_readback') != 1:
        eb.append(f"스레드 되읽기 {meta.get('threads_readback')}")
    ck = {r.get('ckpt_sha256') for r in ok_rows}
    if len(ck) != 1 or None in ck:
        eb.append(f'체크포인트 지문 {ck}')
    A.append((f'★G-ENV 측정면 3중 동결(**지금 재계산**==meta==0행동결, {len(run_keys)}키) · '
              f'분석면 동결 · 러너 시작/끝 md5 안정 · 체크포인트 sha256 · threads[되읽기]==1',
              not eb, f'{eb[:5] if eb else "이상 없음"}'))

    # ═══ §6.3 기록 전용 게이트 (⛔막지 않는다) ═════════════════════════════════════
    # ⚠아래는 전부 조건이 리터럴 `True` 다 — **구조적으로 FAIL 할 수 없다**. 이름을
    #   `GATE_NAMES` 에 두는 이유는 나중에 누가 조건을 붙일 때 조용히 장식이 되지 않게 하는 것.
    w3 = {cid: r for cid, r in seen_cell.items() if int(r['win']) == G.ANCHOR_WIN}
    rail_b = sum(1 for r in seen_cell.values()
                 if _num(r.get('s_rule_bandr')) is not None
                 and float(r['s_rule_bandr']) != float(r['s_raw_bandr']))
    rail_a = sum(1 for r in seen_cell.values()
                 if _num(r.get('s_rule_ap')) is not None
                 and float(r['s_rule_ap']) != float(r['s_raw_ap']))
    rail_b3 = sum(1 for r in w3.values() if float(r['s_rule_bandr']) != float(r['s_raw_bandr']))
    rail_a3 = sum(1 for r in w3.values() if float(r['s_rule_ap']) != float(r['s_raw_ap']))
    A.append((f'⚠G-RAIL [기록 전용] 안전 레일 [{G.S_MIN}, {G.S_MAX}] 접촉 — 밴드 {rail_b}/'
              f'{len(seen_cell)} · 개구 {rail_a}/{len(seen_cell)} '
              f'(win=3: {rail_b3}·{rail_a3} / 등록 {G.RAIL_BANDR_W3}·{G.RAIL_AP_W3}) '
              f'⛔막지 않는다 — 레일은 규칙의 **일부**이고 축퇴는 `PROP-S025` 가 가른다',
              True, f'클립 셀 {sorted(c for c in clip_cells)[:6]}'))

    edge_all = sum(1 for r in seen_cell.values() if r.get('edge_hit') is True)
    edge3 = sum(1 for r in w3.values() if r.get('edge_hit') is True)
    A.append((f'⚠G-EDGE [기록 전용] 1단 첨두가 격자 최외곽 링 — 전체 {edge_all}/{len(seen_cell)} · '
              f'win=3 **{edge3}** (등록 {G.EDGE_W3}, `BS11a`) ★사전등록이 «실측에서 발화한다» 고 '
              f'결과 전에 적었다 ⟹ 발화는 **결과**이지 결함이 아니다. ⛔막으면 배치가 죽는다',
              True, f'발화 셀 {[c for c, r in sorted(seen_cell.items()) if r.get("edge_hit")][:8]}'))

    marg = [_num(r.get('margin')) for r in seen_cell.values()]
    marg3 = [_num(r.get('margin')) for r in w3.values()]
    lt1 = sum(1 for v in marg if v is not None and v < 1.0)
    lt1_3 = sum(1 for v in marg3 if v is not None and v < 1.0)
    A.append((f'⚠G-MARGIN [기록 전용] k̂_p/Δk₁ ≡ L_obs/λ̂_p — 전체 {_stat(marg)} · <1 {lt1}/'
              f'{len(seen_cell)} · win=3 <1 {lt1_3} (등록 {G.MARGIN_LT1_W3}, 범위 {G.MARGIN_W3}) '
              f'⚠개구 한계이지 코드 결함이 아니다 · ⛔«실측 개구/λ 가 1 미만» 을 일반 진술로 '
              f'확장하지 마라(규약 종속, §9.3-13)', True, f'win=3 {_stat(marg3)}'))

    drv = []
    for r in seen_cell.values():
        a, b = _num(r.get('s_raw_bandr')), _num(r.get('s_raw_band_old'))
        if a is not None and b is not None and a > 0 and b > 0:
            drv.append(abs(math.log(a) - math.log(b)))
    drv3 = []
    for r in w3.values():
        a, b = _num(r.get('s_raw_bandr')), _num(r.get('s_raw_band_old'))
        if a is not None and b is not None and a > 0 and b > 0:
            drv3.append(abs(math.log(a) - math.log(b)))
    A.append((f'⚠G-DRIVE [기록 전용] |Δ ln s_raw|(새−옛) — 전체 {_stat(drv)} · win=3 중앙 '
              f'{None if not drv3 else round(float(np.median(drv3)), 6)} (등록 {G.DRIVE_MED_W3}) · '
              f'> ln1.05 win=3 {sum(1 for v in drv3 if v > G.DRIVE_T1_LN)} (등록 {G.DRIVE_GT_T1_W3}) '
              f'⟹ `T16` 의 `D-TREATMENT-BELOW-SCALE` 이 실측에서도 성립함을 기록한다. '
              f'⛔옛/새 규칙 대조는 이 배치의 축이 **아니다**', True, f'n={len(drv)}'))

    adapt = {}
    for rec in G.RECORDS:
        vs = [math.log(float(r['s'])) for cid, r in sorted(by.get('PROP-BANDR', {}).items())
              if r['record'] == rec and _num(r.get('s')) is not None and r['s'] > 0]
        if len(vs) >= 2:
            adapt[rec] = {'n': len(vs), 'sd_ln_s': float(np.std(vs, ddof=1)),
                          'min_s': float(np.exp(min(vs))), 'max_s': float(np.exp(max(vs)))}
    A.append(('⚠G-ADAPT [기록 전용] ★**`T16` 이 원리적으로 못 잰 것** — 레코드 안 6창의 sd(ln s). '
              '`T16` 의 격자는 전 파일 동일이라 `s` 가 코퍼스 상수였다. 여기서 처음 측정된다. '
              '⚠sd==0 이면 이 배치도 **착지점 진술이지 적응 진술이 아니다**',
              True, f'{ {k: round(v["sd_ln_s"], 4) for k, v in sorted(adapt.items())} }'))

    winrec = {}
    for rec in G.RECORDS:
        ws = sorted((int(r['win']), _num(r.get('t_start')), _num(r.get('t_end')),
                     r.get('obs_hash'))
                    for cid, r in seen_cell.items() if r['record'] == rec)
        if len(ws) < 2:
            continue
        gaps = [ws[i + 1][2] - ws[i][2] for i in range(len(ws) - 1)
                if ws[i + 1][2] is not None and ws[i][2] is not None]
        ov = []
        for i in range(len(ws) - 1):
            a0, a1_ = ws[i][1], ws[i][2]
            b0, b1 = ws[i + 1][1], ws[i + 1][2]
            if None in (a0, a1_, b0, b1):
                continue
            inter = max(0.0, min(a1_, b1) - max(a0, b0))
            ov.append(100.0 * inter / max(a1_ - a0, 1e-12))
        winrec[rec] = {'n': len(ws), 'gap_min': (min(gaps) if gaps else None),
                       'gap_max': (max(gaps) if gaps else None),
                       'overlap_pct_max': (max(ov) if ov else None),
                       'distinct_hash': len({h for _, _, _, h in ws})}
    A.append(('⚠G-WIN [기록 전용] 인접 창 관측창 겹침 · 간격 · 해시 상이 — §0.1-(b) 의 근거를 '
              '런 시점에 재확인한다 (프로브 C: 겹침 0.0% 60/60 · 간격 79.1–349.1 s · 해시 6/6 상이). '
              '⚠셀 72 는 **독립 72 가 아니다** — 레코드 수준 잡음을 공유한다(`C-FOLD` 가 집행)',
              True, f'{ {k: (v["overlap_pct_max"], v["distinct_hash"]) for k, v in sorted(winrec.items())} }'))

    kdv = [_num(r.get('kd')) for r in seen_cell.values()]
    kd3 = [_num(r.get('kd')) for r in w3.values()]
    A.append((f'⚠G-DEEP [기록 전용] k̂_p·d — 전체 {_stat(kdv)} · < π 전체 '
              f'{sum(1 for v in kdv if v is not None and v < math.pi)}/{len(kdv)} · win=3 '
              f'{sum(1 for v in kd3 if v is not None and v < math.pi)} (등록 {G.DEEP_LT_PI_W3}) '
              f'⛔차단하면 배치가 죽는다 — 이 코퍼스는 **심해가 아니다**. 그래서 §5 가 `d\'=d/s` 를 '
              f'골랐다', True, f'win=3 {_stat(kd3)}'))

    bases = G.sibling_seed_bases()
    clash = sorted(b for b, vv2 in bases.items()
                   if any(v == G.SEED17 for _, v in vv2)
                   or any(G.SEED17 < v <= G.SEED17 + G.N_CELL for _, v in vv2))
    A.append((f'⚠G-XBATCH [기록 전용] 시드 대역 — Phase B 밑변 {G.SEED17} → '
              f'{G.SEED17 + 1}..{G.SEED17 + G.N_CELL} · 형제 선언 밑변과 충돌 {clash or "없음"}. '
              f'⚠Phase A 시드 **{G.SEED}** 는 `B-2`·`B-3`·`B-14b` 와 **일부러 같다**(앵커). '
              f'원장 대조 = `tmp/_audit/check_seed_bands.py`', True,
              f'형제 {len(bases)}배치'))

    dep_rb = {k: sorted({json.dumps(r.get('effective', {}).get(k)) for r in ok_rows})
              for k in G.DEPLOY_KEYS}
    A.append(('⚠G-DEPLOY [기록·항진] 배포 상수 6종 되읽기 — ⛔**구조적으로 참이다**(§1 이 '
              '`DEFAULT_CONFIG` 위에 같은 값을 명시하므로). 이름에 표기하고 `gates_pass` 를 '
              '오독하지 마라. 명시하는 이유는 `B-7` BLOCKER: 빈 dict 팔이 그날의 런타임 기본값을 '
              '스냅샷했다', True, f'{dep_rb}'))

    # ★G-ANCHOR [기록·등급] — §6.2 2등급. ⛔차단이 아니라 **등급 강등**이다.
    ab_want, ap_want = G.anchor_bank(), G.anchor_prop()
    ab_got = {r['record']: r['nowcast'] for cid, r in by.get('BANK', {}).items()
              if int(r['win']) == G.ANCHOR_WIN}
    ap_got = {r['record']: r['nowcast'] for cid, r in by.get('PROP', {}).items()
              if int(r['win']) == G.ANCHOR_WIN}
    gr_b, off_b = G.anchor_grade(ab_got, ab_want)
    gr_p, off_p = G.anchor_grade(ap_got, ap_want)
    A.append((f'⚠G-ANCHOR [기록·등급] A-BANK(B3 FROZEN win=3 lead0) **{gr_b}** · '
              f'A-PROP(B14b axis=A IID σ=0.02 nowcast) **{gr_p}**. ⛔T3 은 차단이 아니라 '
              f'`verdict.grade` 에 `ANCHOR-T3 (패키지 드리프트)` 를 인쇄하는 것이고 '
              f'⛔**조용한 재기준선은 금지**다. ⛔`B-14b` 는 런 전 동결이 없는 배치라 앵커로만 쓴다',
              True, f'BANK 이탈 {off_b} · PROP 이탈 {off_p}'))

    A.append(('⚠G-SIBLING [기록 전용] 형제 배치 결과 파일 존재·mtime (CPU 경합 · 패키지 드리프트 '
              '단서 — §11 R8·R11)', True, f'{G.sibling_outputs()}'))

    # ═══ 통계량 (PREREG §3) ═══════════════════════════════════════════════════════
    w_f, n_f = wins_zero('BANK')
    w_s, n_s = wins_zero('PROP-BANDR')
    w_p0, n_p0 = wins_zero('PROP')            # ★S 축의 병기 의무 (§3.4)
    w_ap, n_ap = wins_zero('PROP-AP')
    w_s025, n_s025 = wins_zero('PROP-S025')
    w_f2, n_f2 = wins_zero('PROP-FIX2')
    D = paired('PROP-BANDR', 'PROP')
    w_d, n_eff = D['w'], D['n_eff']
    kplus = G.k_two_sided(n_eff) if n_eff else None

    wS_rec, nS_rec, tS_rec, perS = fold_zero('PROP-BANDR')
    wD_rec, nD_rec, tD_rec, perD = fold('PROP-BANDR', 'PROP')
    S_pos = w_s >= G.THRESH_S
    D_pos = (kplus is not None) and (w_d >= kplus or (n_eff - w_d) >= kplus)
    dem_S, why_S = G.fold_demote(S_pos, wS_rec, nS_rec)
    dem_D, why_D = G.fold_demote(D_pos, wD_rec, nD_rec)

    lab_S = G.label_S(w_s)
    lab_D = G.label_D(w_d, n_eff)

    # ── 부차 (⛔전부 무보정 · 라벨 없음 · 기술통계) ────────────────────────────────
    dates_ = G.dates()
    by_date = {dt: [G.cell_id(r, w) for r in G.RECORDS if G.date_of(r) == dt for w in G.WINS]
               for dt in dates_}
    by_site = {s: [G.cell_id(r, w) for r in G.RECORDS if G.site_of(r) == s for w in G.WINS]
               for s in sorted({G.site_of(r) for r in G.RECORDS})}
    SEC = {
        '★SEC-A PROP-BANDR < BANK [⛔무보정 · 라벨 없음]': paired('PROP-BANDR', 'BANK'),
        'SEC-B PROP-AP < PROP [⚠8/12 레일 클립 ⟹ 사실상 상수 팔]': paired('PROP-AP', 'PROP'),
        'SEC-C PROP-FIX2 < PROP [⛔`T11` 재현이 아니다 — 실측에서 **반대 방향** 대조]':
            paired('PROP-FIX2', 'PROP'),
        '★SEC-D PROP-S025 < PROP [레일 대조 — `SEC-D ≈ D` 면 «규칙이 아니라 상수가 한 일»]':
            paired('PROP-S025', 'PROP'),
        '★SEC-E PROP-BANDR vs PROP-AP (양측) [규칙 계열 판별자 · q80 병기 의무]':
            paired('PROP-BANDR', 'PROP-AP'),
        'SEC-LEAD 종점을 leadmean 으로 [기술통계 · ⛔주지표를 바꾸지 마라]':
            {'S': {'w': sum(1 for cid, r in by.get('PROP-BANDR', {}).items()
                            if _num(r.get('leadmean')) is not None), 'note': 'lead 곡선'},
             'D': paired('PROP-BANDR', 'PROP', field='leadmean')},
        'SEC-DATE 일자 10 접기 [문턱 w>=%d · q80=%s · 기술통계]'
        % (G.THRESH_DATE, G.MDE['DATE_q80']):
            {'S': {dt: wins_zero('PROP-BANDR', by_date[dt]) for dt in dates_},
             'D': {dt: paired('PROP-BANDR', 'PROP', by_date[dt])['w'] for dt in dates_}},
        '⛔SEC-SITE 사이트 4 접기 [**기각 불가** — 바닥 p 0.0625 · 수만 인쇄]':
            {'S': {s: wins_zero('PROP-BANDR', by_site[s]) for s in by_site},
             'D': {s: paired('PROP-BANDR', 'PROP', by_site[s])['w'] for s in by_site},
             'thresh': G.THRESH_SITE},
        'SEC-BOX 대형5/소형7 층 [⛔`B-2` 층 분해 수치 인용 금지 · 라벨을 만들지 않는다]':
            {'large': paired('PROP-BANDR', 'PROP',
                             [G.cell_id(r, w) for r in G.LARGE_BOX for w in G.WINS]),
             'small': paired('PROP-BANDR', 'PROP',
                             [G.cell_id(r, w) for r in G.SMALL_BOX for w in G.WINS])},
        '★SEC-ADAPT 레코드 안 6창의 sd(ln s) [★`T16` 이 원리적으로 못 잰 것]': adapt,
        'SEC-DRIVE Δ ln s(새−옛) 분포': _stat(drv),
        'SEC-SEED Phase B [⛔이 실행에 없다 — Phase A 게이트 통과 후 별도]':
            {'present': False, 'base': G.SEED17},
    }

    # ── §6 게이트 ↔ 사전등록 1:1 + 병기 의무 자가검사 (`G-COV`) ─────────────────────
    import re as _re
    emitted = sorted({_re.search(r'G-[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*', n).group(0)
                      for n, _, _ in A if _re.search(r'G-[A-Za-z0-9]+', n)}
                     | {'G-COV', 'G-BLOCK'})
    reg = _prereg_gate_names()
    drift = G.registration_drift()
    cov_bad = []
    if reg is None:
        cov_bad.append('PREREG.md §6 을 못 읽었다')
    else:
        only_p = sorted(reg - set(G.GATE_NAMES))
        only_g = sorted(set(G.GATE_NAMES) - reg - set(G.GATES_ADDED))
        if only_p:
            cov_bad.append(f'사전등록에만: {only_p}')
        if only_g:
            cov_bad.append(f'구현에만(⛔GATES_ADDED 에 선언되지 않았다): {only_g}')
    if set(emitted) != set(G.GATE_NAMES):
        cov_bad.append(f'인쇄되지 않은 게이트 {sorted(set(G.GATE_NAMES) - set(emitted))}'
                       f' · 등록되지 않은 인쇄 {sorted(set(emitted) - set(G.GATE_NAMES))}')
    cov_bad += [f'등록 드리프트: {d}' for d in drift]

    # ★★병기 의무(§3.4) — ⛔분석기가 **기계로** 넣고, 넣었는지 **기계로** 확인한다.
    prop_txt = (f'항등 스케일 `PROP < ZERO_true` = {w_p0}/{n_p0} '
                f'(p={G.sign_p(w_p0, n_p0)})')
    rail_txt = (f'레일 클립 밴드 {rail_b}/{len(seen_cell)} · 개구 {rail_a}/{len(seen_cell)} · '
                f'`G-EDGE` 발화 {edge_all}/{len(seen_cell)}')
    fold_txt = (f'레코드 접기 S {wS_rec}/{nS_rec}(동점 {tS_rec}) · D {wD_rec}/{nD_rec}'
                f'(동점 {tD_rec}) · `C-FOLD` S:{why_S} · D:{why_D}')
    sec_a = SEC['★SEC-A PROP-BANDR < BANK [⛔무보정 · 라벨 없음]']
    seca_txt = (f"`SEC-A` PROP-BANDR<BANK {sec_a['w']}/{sec_a['n_eff']} "
                f"(**무보정 · 라벨 없음**)")
    disclosures = [
        # ── §9.2 금지 ──────────────────────────────────────────────────────────
        f'⛔[§9.2-1] "우리 방법이 실측에서 경쟁자보다 낫다" 금지 — 외부 기준선 행 **0개**다. '
        f'실측 외부 기준선은 두 번 연속 자기검정에서 죽었다(`B2_WANGFIX` · `B-5` 0/12). '
        f'⟹ 그 부재를 **측정된 한계**로 보고한다. 라벨은 **귀속**이다.',
        f'⛔[§9.2-2] `ZERO_REF = 1.0` 인용·사용 금지 — 참 제로필드는 레코드별 **0.3375–1.1559** '
        f'다. 이 배치의 채점 기준은 `zero_ref.zero_ref_nowcast` **함수**이고 `G-ZEROREF` 가 '
        f'분석기의 독립 재계산으로 그것을 증명한다. 실측 분포 = {_stat(zvals)}.',
        f'⛔[§9.2-3] `T11`(n=24)·`T14`(n=28)·`T16`(n=60, mixed_data)의 수와 이 배치의 수를 '
        f'**직접 비교 금지** — 모집단이 다르고 §5 의 **수심 규약이 다르다**(`d\'=d/s`). '
        f'`PROP`(s=1) 만 `B-14b` 앵커와 같은 규약이다.',
        f'⛔[§9.2-4] 사이트 수준 일반화 금지 — 사이트 4개, 층 바닥 p 0.0625 ⟹ `SEC-SITE` 는 '
        f'**어떤 효과 크기에서도** α=0.05 를 달성할 수 없다(문턱 {G.THRESH_SITE}). '
        f'일자·층 진술도 전부 **기술통계**다.',
        f'⛔[§9.2-5] `B-2` 의 「대형/소형 층 분해」 수치 인용 금지(수심 오염 + 채점 기준, '
        f'Fisher 0.2475 널). 이 배치가 층(`SEC-BOX`)을 다시 선언해도 **라벨을 만들지 않는다**.',
        f'⛔[§9.2-6] **셀 수만으로 진술 금지** — `C-FOLD`(§3.4)의 레코드 접기 병기가 의무다: '
        f'{fold_txt}. ⚠MDE 를 섞지 마라: 셀 S q80={G.MDE["S_q80"]} · 셀 D q80={G.MDE["D_q80"]} · '
        f'레코드 S q80={G.MDE["REC_S_q80"]} · 레코드 D q80={G.MDE["REC_D_q80"]} · '
        f'일자 q80={G.MDE["DATE_q80"]} (`M8b` 선례).',
        f'⛔[§9.2-7] `SIG_WORSE`(w<={G.SIG_WORSE}) 와 `SEC-*` 의 p 는 전부 **무보정**이다. '
        f'Holm/Bonferroni 는 PRIMARY 가족(**2축** = `S`·`D`, α/2={G.ALPHA_FAM})에만 적용한다.',
        f'⛔[§9.2-8] `PROP-FIX2` 를 "`T11` 결과의 실측 재현" 으로 읽지 마라 — 실측에서 그 상수는 '
        f'**규칙이 요구하는 방향의 반대**다(s<1 이 11/12). 그 팔은 **반대 방향 대조**다.',
        f'⛔[§9.2-9] **배포 권고 아님.** 양성 라벨도 **필요조건 하나**다(`S-ENS §5` 선례). '
        f'남는 것 최소 셋: (i) 합성 경로 무해성 (ii) 단가 (iii) 실측 라우팅과의 상호작용.',
        f'⛔[§9.2-10] `G-EDGE`·`G-RAIL` 발화를 "규칙의 견고성" 으로도 "규칙의 실패" 로도 단정하지 '
        f'마라 — 등록된 것은 **수**이고 해석은 `PROP-S025` 팔이 가른다. 병기: {rail_txt}.',
        f'⛔[§9.2-11] "학습 경로가 실측에서 작동한다" 로 확장 금지 — `B-7`(순차 추적 무기술) · '
        f'`B-8b`(합성 학습 사전분포는 실측에서 학습 없는 해석적 초기화보다 열등)가 그대로 유효하다. '
        f'`S` 축 진술의 병기 의무: {prop_txt}.',
        # ── §9.3 공시 ──────────────────────────────────────────────────────────
        f'⚠[§9.3-1] **완전 blind 아님** — 설계자가 결과 전에 본 것: 프로브 A/B/C 전건'
        f'(`L_obs`·`s_ap`·`s_band`·`s_bandr`·`k̂_p`·λ̂·여유·레일·창 분리) + 동결 성과 재집계'
        f'(`BANK<ZERO_true` {G.ANCHOR_BANK_W3}/12 · `PROP<ZERO_true` {G.ANCHOR_PROP_W3}/12 · '
        f'`PROP<BANK` {G.ANCHOR_PAIR_W3}/12, 전부 win=3). ⟹ 등록 예측 q_S={G.PRED_Q["S"]}·'
        f'q_D={G.PRED_Q["D"]} 와 `PROP-S025` 팔의 존재가 그 관찰의 산물이다. '
        f'✅어떤 처치 팔의 nRMSE 도 결과 전에 존재하지 않았고 win≠3 의 s 도 보지 않았다(60/72 미노출).',
        f'⚠[§9.3-2] 규칙의 목표 상수(`L_TRAIN`={G.L_TRAIN} · `K_TRAIN`={G.K_TRAIN})는 **우리 '
        f'학습 분포**에서 나온다 — 규칙은 "장면을 우리 사전이 학습된 곳으로 옮긴다" 이고 '
        f'⛔"실측 자료가 규칙을 정했다" 는 거짓이다.',
        f'⚠[§9.3-3] **재척도는 단일요인이 아니다** — `s` 가 사전 밴드·`learned_dedup_dk=0.03`·'
        f'`MIN_DK=0.15`·성장 DC블록·VARPRO lr·조건화 지도의 개구/픽셀 비를 **동시에** 움직인다 '
        f'⟹ ⛔단일 기제 귀속 금지. `D` 는 **개입이 하나(`s`)이지 기제가 하나가 아니다**.',
        f'⚠[§9.3-4] `d3_beamform_net.py:30 DEPTH = 50.0` 하드코딩이 조건화 지도에서 config depth '
        f'를 무효로 만든다 ⟹ §5 의 `d\'=d/s` 는 **분산관계에는 닿고 조건화 지도에는 닿지 않는다**'
        f'(비대칭 교락). ⛔이 배치에서 고치지 않는다(고치면 출하 팔이 아니게 된다).',
        f'⚠[§9.3-5] **예산 비대칭** — 제안 단계는 프레임당 제공 관측 중 캡 '
        f'{sorted({r.get("cap_proposal") for r in prop})} 만 쓴다(`learned_init.py:65 M_OBS_CAP`).',
        f'⚠[§9.3-6] **셀 72 는 독립 72 가 아니다** — 6창은 바다 상태로는 분리되나(겹침 0%) '
        f'레코드 수준 잡음(개구·첨두 파장·상자 크기·관측 잡음·수심)을 **공유한다**. '
        f'`C-FOLD` 가 그 대가를 집행한다: {fold_txt}.',
        f'⚠[§9.3-7] **추정기 시드 1개**(Phase A, `SEED`={G.SEED}) ⟹ ⛔시드 간 변동 주장 금지'
        f'(`S13b` 선례: 시드가 부호를 가른 적이 있다). Phase B(`SEC-SEED`)가 기술통계로만 본다.',
        f'⚠[§9.3-8] 관측은 실측 WASS 표면의 **합성 희소화**다(noise=0.02 · m_cap=600 · 균등추출). '
        f'실제 LiDAR 반환이 아니고, 이동 플랫폼·조우 기하에 대해 **아무 말도 하지 않는다**'
        f'(전부 고정 관측소).',
        f'⚠[§9.3-9] `k̂_p` 는 FFT 첨두 대비 **편향**이 있다(`T14 §9.3-9`: +7–13%). ⛔보정하지 않았다.',
        f'⚠[§9.3-10] `proposal_device`·`beamform_device` 는 `config` 가 정하지 않는다 ⟹ 되읽어 '
        f'적었다: 제안 {sorted({r.get("proposal_device") for r in prop if r.get("proposal_device")})} '
        f'· 빔포머 {sorted({r.get("beamform_device") for r in ok_rows if r.get("beamform_device")})}. '
        f'⛔`wall_s` 를 팔 간 비교에 쓰지 마라(`set_num_threads(1)` 은 BLAS 를 제약하지 않는다).',
        f'⚠[§9.3-11] `BS` 7 레코드의 수심은 **30.0** 이다(2026-08-23 정정). ⛔`B-2`…`B-2f` 는 '
        f'17.0 으로 돌았다 ⟹ 그 배치들의 수를 나란히 놓을 때 수심을 확인하라(`B3`·`B14b` 는 30).',
        f'⚠[§9.3-12] `B-14b` 는 **런 전 동결이 없는 배치**다(자기 공시) ⟹ **앵커로만** 쓰고 '
        f'그 배치의 라벨을 인용하지 않는다. 앵커 등급: A-BANK {gr_b} · A-PROP {gr_p}.',
        f'⚠[§9.3-13] 「개구/파장」은 **규약 종속**이다 — 이 배치가 인쇄하는 여유는 규칙의 2단 '
        f'빔포머 `k̂_p^REL` 로 잰 자이고, 형제 문서 `RETRACT_T11T14_2026-09-05/TARGETS.md` 는 '
        f'같은 12 레코드를 1.0–4.2 로 적는다. ⛔두 수를 나란히 놓아 서로 다른 대상인 것처럼 쓰지 '
        f'마라 — **같은 양의 두 자**다. ⛔"실측 개구/λ 가 1 미만" 을 일반 진술로 확장하지 마라.',
        f'⚠[§10] `PROP*` 팔은 **실측 게이트를 끈다**(`real_domain_gate=False` + 명시 '
        f'`init_method=\'learned\'`) — 그것은 `W-12` 가 금지한 팔이다(실측에서 학습 강제 시 예보 '
        f'지평 0/5 패배). ⟹ ⛔**이 배치는 배포 라우팅을 바꾸자는 제안이 아니다.** 라우팅은 '
        f'측정된 판정이고 없애면 `D3` BLOCKER 재발이다. 묻는 것은 «그 팔이 스케일을 고치면 '
        f'달라지는가» 뿐이다. `BANK` 팔은 라우터를 그대로 두어 게이트가 **실제로 발화**하게 했고 '
        f'`G-ARM` 이 그것을 되읽어 증명한다.',
        f'⚠[§0.1-(d)] 이 배치가 **원리적으로 답하지 못하는 것**: (1) 경쟁자 대비 우위 '
        f'(2) 실해역 운용 검증 (3) 사이트 수준 일반화 (4) «규칙이 옳은 스케일을 찾았다»'
        f'(참 s 가 없다 — 재는 것은 **착지점과 정확도**이지 정확성이 아니다) '
        f'(5) «자유 파라미터 0개의 규칙이 승리했다»(레일 클립 셀에서 처치는 **규칙이 아니라 상수**다) '
        f'(6) 옛/새 규칙 대조(`AUTO-BAND` vs `AUTO-BANDR`)는 이 코퍼스의 축이 아니다. '
        f'⛔결과 뒤에 재서술하지 마라.',
        f'⚠[§3.6] 설계는 결과 전에 스스로 적었다: 최빈 결과가 '
        f'`NO-REAL-SKILL-DETECTED ∧ SCALE-CHANGES-REAL-LANDING`(0.390540) 이고 두 축 모두 널일 '
        f'확률이 0.229312 다. ⟹ ⛔이 사실을 결과 뒤에 재서술하지 마라. 병기: {prop_txt} · '
        f'{seca_txt} · {rail_txt}.',
    ]
    cocite = []
    for s_ in disclosures:
        if '`S` 축' in s_ and 'PROP < ZERO_true' not in s_:
            cocite.append(f'S 진술에 항등 스케일 병기 없음: {s_[:40]}')
        if '`D` 축' in s_ and ('레일' not in s_ and 'G-EDGE' not in s_):
            cocite.append(f'D 진술에 레일·G-EDGE 병기 없음: {s_[:40]}')
    if not any('레코드 접기' in s_ for s_ in disclosures):
        cocite.append('어떤 공시에도 레코드 접기(`C-FOLD`)가 없다')
    for lb_ in (lab_S, lab_D):
        neg = 'UNDETECTED' in lb_ or lb_ == 'D-UNPOWERED'
        if neg and not any('q80' in s_ for s_ in disclosures):
            cocite.append(f'음성 라벨 {lb_} 에 MDE 병기 없음')
    if not any('SEC-A' in s_ for s_ in disclosures):
        cocite.append('`SEC-A` 무보정 표기가 공시에 없다')
    cov_bad += cocite
    A.append(('G-COV 사전등록 §6 게이트 표 ↔ 구현 **1:1**(추가는 `GATES_ADDED` 선언만 허용) · '
              '등록 상수 드리프트 0 · ★§3.4 **병기 의무** 자가검사 · 게이트가 자료를 봤다 '
              '[⛔사전등록 §6 표에 없는 추가 게이트 — GATES_ADDED 가 사유를 적는다]',
              not cov_bad and len(ok_rows) > 0,
              f'{cov_bad if cov_bad else "이상 없음"} · rows={len(rows)}'))
    missed, folded = G.selfcheck_blocked_by(emitted + list(G.GATE_NAMES))
    A.append(('G-BLOCK blocked_by 자가진단 — **분석기가 실제로 낸 이름 전건**(자기 자신 포함) + '
              '등록된 전 이름을 합성 위반 튜플로 넣어 빈 리스트가 아님을 확인 [긴 이름 우선]',
              not missed and not folded, f'미포착 {missed} · 접힘 {folded}'))

    # ★라벨 인쇄 벨트 — ⛔차단이 하나라도 있거나 게이트가 하나라도 FAIL 이면 **인쇄하지 않는다**.
    blocked = G.blocked_by(A)
    all_ok = all(bool(o) for _, o, _ in A)
    lab = (G.label(w_f, w_s, w_d, n_eff, dem_S, dem_D)
           if (not blocked and all_ok) else None)

    # ★프로브 재현 — ⛔막지 않는다(§6.3). 그러나 **묻어 두지도 않는다**.
    probe_now = {'rail_bandr': rail_b3, 'rail_ap': rail_a3, 'edge': edge3,
                 'margin_lt1': lt1_3,
                 'drive_gt_t1': sum(1 for v in drv3 if v > G.DRIVE_T1_LN),
                 'deep_lt_pi': sum(1 for v in kd3 if v is not None and v < math.pi)}
    probe_full = (len(w3) == G.N_REC)
    probe_drift = ([f'{k}: 등록 {v} vs 측정 {probe_now[k]}'
                    for k, v in G.PROBE_W3.items() if probe_now[k] != v]
                   if probe_full else ['⚠미평가 (win=3 셀 %d/%d — 전수가 아니면 수를 비교할 수 '
                                       '없다)' % (len(w3), G.N_REC)])

    grade = 'CONFIRMATORY (2축 · Bonferroni α/2)'
    if gr_b.startswith('T3') or gr_p.startswith('T3'):
        grade += ' · ANCHOR-T3 (패키지 드리프트)'
    if dem_S or dem_D:
        grade += f' · {G.FOLD_SUFFIX}'

    v = {
        'batch': 'T17-REALSCALE', 'kind': 'CONFIRMATORY', 'grade': grade,
        'n_rows': len(rows), 'n_rows_expected': G.N_ROWS, 'n_runs_expected': G.N_RUNS,
        'n_errors': len(err_rows),
        'gates': [{'name': n, 'pass_': bool(o), 'detail': d} for n, o, d in A],
        'gates_pass': f'{sum(1 for _, o, _ in A if o)}/{len(A)}', 'blocked_by': blocked,
        'primary_family': {
            'S (기술 축, 단측, Bonferroni α/2)': {
                'stat': 'W[PROP-BANDR < ZERO_true]', 'w': w_s, 'n': n_s,
                'p': G.sign_p(w_s, n_s), 'thresh': G.THRESH_S,
                'pass': w_s >= G.THRESH_S, 'q80': G.MDE['S_q80'],
                '병기_항등스케일': {'w': w_p0, 'n': n_p0, 'p': G.sign_p(w_p0, n_p0)},
                '병기_레코드접기': {'w': wS_rec, 'n': nS_rec, 'tied': tS_rec,
                                    'thresh': G.FOLD_MAJORITY, 'q80': G.MDE['REC_S_q80'],
                                    'per_record': perS}},
            'D (착지 축, 양측, Bonferroni α/2)': dict(
                D, stat='W[PROP-BANDR < PROP]', k_plus=kplus, thresh=G.THRESH_D,
                q80=G.MDE['D_q80'],
                병기_레코드접기={'w': wD_rec, 'n': nD_rec, 'tied': tD_rec,
                                 'thresh': G.FOLD_MAJORITY, 'q80': G.MDE['REC_D_q80'],
                                 'per_record': perD}),
            'bonferroni': f'α/{G.FAMILY_SIZE} = {G.ALPHA_FAM}'},
        'positive_control': {'stat': 'W[BANK < ZERO_true]', 'w': w_f, 'n': n_f,
                             'p': G.sign_p(w_f, n_f), 'thresh': G.THRESH_F,
                             'pass': w_f >= G.THRESH_F,
                             'note': '★예측이 아니라 **복제 주장**이다(§3.2). 거짓이면 '
                                     '`ANCHOR-DID-NOT-REPLICATE` 하나만 인쇄한다'},
        'C_FOLD': {'S': {'demote': dem_S, 'why': why_S},
                   'D': {'demote': dem_D, 'why': why_D},
                   'rule': '셀 양성 ∧ 레코드 접기 < %d ⟹ 강등(⛔승격은 없다)' % G.FOLD_MAJORITY},
        'other_vs_zero': {'PROP': {'w': w_p0, 'n': n_p0},
                          'PROP-AP': {'w': w_ap, 'n': n_ap},
                          'PROP-S025': {'w': w_s025, 'n': n_s025},
                          'PROP-FIX2': {'w': w_f2, 'n': n_f2}},
        'sig_worse': {a: (w <= G.SIG_WORSE) for a, (w, _) in
                      (('PROP-BANDR', (w_s, n_s)), ('PROP', (w_p0, n_p0)),
                       ('BANK', (w_f, n_f)))},
        'SEC': SEC,
        'medians': {a: float(np.median([r['nowcast'] for r in d.values()]))
                    for a, d in sorted(by.items()) if d},
        'zero_true_stat': _stat(zvals),
        'anchor': {'A-BANK': {'grade': gr_b, 'outliers': off_b, 'n': len(ab_got)},
                   'A-PROP': {'grade': gr_p, 'outliers': off_p, 'n': len(ap_got)}},
        'probe_w3': {'registered': G.PROBE_W3, 'measured': probe_now,
                     'evaluable': probe_full, 'drift': probe_drift,
                     'note': '⛔막지 않는다(§6.3) — 등록은 **사후 재도출을 봉하기 위한 것**이다'},
        'rail': {'bandr': rail_b, 'ap': rail_a, 'clip_cells': sorted(clip_cells)},
        'edge': {'n': edge_all,
                 'cells': [c for c, r in sorted(seen_cell.items()) if r.get('edge_hit')]},
        'margin': _stat(marg), 'kd': _stat(kdv), 'drive': _stat(drv), 'adapt': adapt,
        'window_separation': winrec,
        'mde': G.MDE,
        'thresholds': {'F': G.THRESH_F, 'S': G.THRESH_S, 'D': G.THRESH_D,
                       'SIG_WORSE': G.SIG_WORSE, 'N_EFF_UNPOWERED': G.N_EFF_UNPOWERED,
                       'REC_S': G.THRESH_S_REC, 'REC_D': G.THRESH_D_REC,
                       'DATE': G.THRESH_DATE, 'SITE': G.THRESH_SITE},
        'prediction': {'q': G.PRED_Q, 'label_probs': G.label_probs(),
                       'null_label_probs': G.null_label_probs(),
                       'sensitivity': G.sensitivity_table(),
                       'rejected_design_n12': G.rejected_design_probs()},
        'registration_drift': drift,
        'gates_added': G.GATES_ADDED,
        'label_S': lab_S, 'label_D': lab_D,
        'demoted': {'S': dem_S, 'D': dem_D},
        'label': lab, 'label_msg': lab or 'BLOCKED',
        'disclosures': disclosures,
    }
    out_p = os.path.join(outdir, f'verdict_{tag}.json')
    json.dump(v, open(out_p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

    def _p(k, n):
        q = G.sign_p(k, n)
        return 'n/a' if q is None else f'{q:.4g}'

    print(f"T17-REALSCALE :: {v['label_msg']}")
    print(f'  양성대조 F  BANK < ZERO_true   {w_f}/{n_f}  p={_p(w_f, n_f)}  '
          f'(문턱 {G.THRESH_F}) ⟹ {"복제" if w_f >= G.THRESH_F else "⛔미복제"}')
    print(f'  축 S  PROP-BANDR < ZERO_true  {w_s}/{n_s}  p={_p(w_s, n_s)}  '
          f'(문턱 {G.THRESH_S}, Bonferroni α/2) ⟹ {lab_S}')
    print(f'        ★병기 의무 — {prop_txt}')
    print(f"  축 D  PROP-BANDR < PROP  {w_d}/{n_eff}  (동점 {D['n_tied']}, k⁺={kplus}) ⟹ {lab_D}")
    print(f'        ★병기 의무 — {rail_txt}')
    print(f'  ★C-FOLD — {fold_txt}')
    print(f'  ★{seca_txt}  rel median {sec_a["rel_median_pct"]}')
    print(f'  MDE q80  셀 S={G.MDE["S_q80"]} · 셀 D={G.MDE["D_q80"]} · 레코드 S='
          f'{G.MDE["REC_S_q80"]} · 레코드 D={G.MDE["REC_D_q80"]} (⛔섞지 마라)')
    print(f'  앵커  A-BANK {gr_b} · A-PROP {gr_p}')
    print(f'  프로브(win=3) 등록 {G.PROBE_W3} vs 측정 {probe_now} ⟹ '
          f'{probe_drift if probe_drift else "일치"}'
          + ('' if probe_full else '  ⛔전수가 아니다'))
    print(f"  중앙값 {v['medians']}")
    print(f"  gates {v['gates_pass']} · blocked_by {blocked}")
    for n, o, d in A:
        print(f"   [{'OK ' if o else 'FAIL'}] {n[:100]:100s} {str(d)[:170]}")
    print(f'  → {out_p}')
    return v


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--rows', default=ROWS)
    ap.add_argument('--meta', default=META)
    ap.add_argument('--out', default=HERE,
                    help='⛔스모크 판정을 배치 디렉터리에 쓰지 마라 (T15 오염 선례)')
    ap.add_argument('--tag', default='t17')
    a = ap.parse_args()
    main(a.rows, a.meta, a.out, a.tag)
