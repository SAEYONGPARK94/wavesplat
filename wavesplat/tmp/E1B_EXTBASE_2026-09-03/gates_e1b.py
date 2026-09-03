"""`E1b-EXTBASE` 규칙 단일출처. ⛔`PREREG.md` 가 정본이고 이 파일이 코드로 옮긴다."""
import os
import subprocess
import sys
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
B2DIR = os.path.join(ROOT, 'tmp', 'B2_REAL_2026-08-23')
EXT = os.path.abspath(os.path.join(ROOT, '..', 'external', 'wassfast'))
for _p in (B2DIR, ROOT, os.path.join(ROOT, 'scripts')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import gates_b2 as B2                                                # noqa: E402
import sys as _sys, os as _os
_E1 = _os.path.join(ROOT, 'tmp', 'E1_EXTBASE_2026-09-03')
if _E1 not in _sys.path:
    _sys.path.insert(0, _E1)
import gates_e1 as E1                                                 # noqa: E402

RECORDS = list(B2.RECORDS)          # 12 (LJ18b 결과 전 배제 상속)
N = len(RECORDS)
WIN = B2.WIN                        # 3
SEED = B2.SEED                      # 42
SHIP_ARM = dict(B2.ARMS['SHIP'])    # ⛔재선언하지 않는다 — 되읽기로 확인

CNN_N = 256                         # 저들 격자
LADDER = (12, 40, 120, 400, 1200)   # 프레임당 점수 (결과 전 고정)
PRIMARY_BUDGET = 600                # ★총 관측 예산 고정 ⟹ 200/프레임 × 3
PRIMARY_PER_FRAME = PRIMARY_BUDGET // 3
SCENE_NOISE = 0.02                  # `scene_station` 규약
SAMPLE_SEED = 20260903              # ⛔결과 전 고정
DIR_HOLDOUT = 0.20                  # §4 GT-free 방향 선택의 유보 비율
SIGNS = ((1, 1), (1, -1), (-1, 1), (-1, -1))

EXT_COMMIT = '90faf98face61a18bb8ba79d1bd5d3e55385fd21'
CKPT_SPARSE = 'wassfast/cnn/modeldata/2021-06-30_16-48-27.h5'
CKPT_FULL = 'wassfast/cnn/modeldata/2021-07-01_12-00-44_3.h5'

SELF_MIN = 10                       # ★G-SELF: ZERO 를 이기는 레코드 수
POS_NRMSE_MAX = 0.30                # ★G-POS: 양성 대조 재현 문턱
KGRID_TOL = 1e-9
THRESH_SHIP = 10                    # PRIMARY 판정표
THRESH_EXT = 2
ALPHA = 0.05

SHIPPING_KEYS = ('varpro_steps', 'learned_init_iters', 'auto_switch_thresh')


def shipping_expect():
    from wavesplat.run import DEFAULT_CONFIG
    return {k: DEFAULT_CONFIG[k] for k in SHIPPING_KEYS}


def sign_test_one_sided(w, n):
    return sum(comb(n, k) for k in range(w, n + 1)) / 2.0 ** n


def ext_clean():
    """★`G-NOMOD` — 제3자 저장소가 무수정이고 HEAD 가 핀과 같은가.

    ⚠결과 0행에서 명세를 고쳤다: 초판은 `--porcelain`(=미추적 파일 포함)이라
    **내가 흘린 `texput.log` 하나에 발화**했다. 게이트가 재려던 것은
    *"저들 코드를 고쳤는가"* 이므로 **추적 파일**만 본다(`-uno`) — 그리고
    실체를 더 직접 재도록 **두 체크포인트의 sha256** 을 함께 확인한다.
    """
    import hashlib
    try:
        head = subprocess.run(['git', '-C', EXT, 'rev-parse', 'HEAD'],
                              capture_output=True, text=True, check=True).stdout.strip()
        st = subprocess.run(['git', '-C', EXT, 'status', '--porcelain', '-uno'],
                            capture_output=True, text=True, check=True).stdout.strip()
    except Exception as e:                                           # noqa: BLE001
        return False, f'git 실패: {e}'
    sh = {}
    for c in (CKPT_SPARSE, CKPT_FULL):
        p = os.path.join(EXT, c)
        if not os.path.exists(p):
            return False, f'체크포인트 없음 {c}'
        sh[os.path.basename(c)] = hashlib.sha256(open(p, 'rb').read()).hexdigest()[:12]
    ok = (head == EXT_COMMIT) and (st == '')
    return ok, f'HEAD={head[:12]} tracked-dirty={"yes" if st else "no"} ckpt={sh}'


E1_RUNG = 400            # `E1` 의 G-SELF 통과 최초 칸 (결과 전 고정, 동결 verdict 에서)


def label_bias(new_rung):
    """⛔결과 전 동결 판정표(PREREG §2)."""
    if new_rung is None:
        return ('BIAS-NO-PASS', '어느 칸에서도 자기검정 미통과 — ⛔E1 과 비교 불가')
    if new_rung < E1_RUNG:
        return ('BIAS-INFLATED-THRESHOLD',
                f'{new_rung} < {E1_RUNG} — ★`E1` 의 문턱은 위로 편향돼 있었다')
    if new_rung == E1_RUNG:
        return ('BIAS-IMMATERIAL', f'{new_rung} = {E1_RUNG} — 비대칭은 문턱을 움직이지 않는다')
    return ('BIAS-OPPOSITE', f'{new_rung} > {E1_RUNG} — ⛔내 편향 분석이 틀렸다')


def label(self_ok, rung, w_ship):
    """⛔결과 전 동결 판정표(PREREG §5)."""
    if not self_ok:
        return ('NO-VALID-COMPETITOR-AT-ANY-DENSITY',
                '제3자 팔이 어느 밀도에서도 레코드별 참 제로필드를 넘지 못한다 — '
                '★"외부 기준선 행 0개" 가 이번엔 **밀도 축을 따라 측정**됐다')
    p = sign_test_one_sided(max(w_ship, N - w_ship), N)
    if w_ship >= THRESH_SHIP:
        return ('SHIP-BEATS-EXTERNAL',
                f'{w_ship}/{N} (p={p:.4f}) @ 사다리 {rung} 점/프레임 — '
                '⛔예산 규약 병기 없이 인용 금지')
    if w_ship <= THRESH_EXT:
        return ('EXTERNAL-BEATS-SHIP',
                f'{w_ship}/{N} (p={p:.4f}) @ 사다리 {rung} 점/프레임 — ★그대로 보고한다')
    return ('EXTBASE-UNDECIDED', f'{w_ship}/{N} @ 사다리 {rung} 점/프레임')


def gates(rows, meta):
    ok = [r for r in rows if 'error' not in r]
    A = []
    A.append((f'G-0 레코드 {N} 전건', meta.get('n_records') == N,
              f"{meta.get('n_records')}/{N}"))
    A.append(('G-0b 에러 0', len(rows) == len(ok), str(len(rows) - len(ok))))
    A.append(('★G-NOMOD 제3자 저장소 무수정 · HEAD 핀 일치',
              bool(meta.get('ext_clean')), str(meta.get('ext_msg'))))
    A.append((f'★G-POS 양성 대조 nRMSE ≤ {POS_NRMSE_MAX}',
              meta.get('pos_nrmse') is not None and meta['pos_nrmse'] <= POS_NRMSE_MAX,
              f"{meta.get('pos_nrmse')}"))
    A.append(('★G-KGRID KX_ab 규약이 저들 config256.mat 과 일치',
              bool(meta.get('kgrid_ok')), f"relerr={meta.get('kgrid_relerr')}"))
    bad = [r['record'] for r in ok if not r.get('gt_free')]
    A.append(('★G-GT 제3자 입력에 참값 0 (유한 원소 == 공급 관측 수)', not bad,
              ';'.join(sorted(set(bad))[:3]) or '전건'))
    bad = [r['record'] for r in ok
           if r.get('primary') and r['n_obs_ship'] != r['n_obs_cnn_primary']]
    A.append((f'★G-BUDGET PRIMARY 에서 두 팔 관측 수 동일 ({PRIMARY_BUDGET})', not bad,
              ';'.join(sorted(set(bad))[:3]) or '전건'))
    bad = [r['record'] for r in ok if r.get('dir_sign') is None]
    A.append(('G-DIR GT-free 방향 선택 결과가 행에 기록됨', not bad,
              ';'.join(sorted(set(bad))[:3]) or '전건'))
    bad = [r['record'] for r in ok if not r.get('arm_ok')]
    A.append((f"★G-ARM SHIP 구성 되읽기 {meta.get('shipping_expect')}", not bad,
              ';'.join(sorted(set(bad))[:3]) or '전건'))
    # ★★G-SELF — PREREG §3 의 **가장 중요한 차단 게이트**인데 초판은 이것을 판정 로직
    #   안에만 두고 **게이트 행으로 내지 않았다**(방벽 「게이트 구현 대조」가 잡았다).
    #   ⚠결과 후 추가지만 **가법적 계측**이고 라벨 계산은 같은 값을 이미 쓰고 있었다
    #   ⟹ 수치 불변. ⛔`blocked_by` 에는 **넣지 않는다** — PREREG §5 가 그 실패에
    #   **자기 라벨**(`NO-VALID-COMPETITOR-AT-ANY-DENSITY`)을 주기 때문이다.
    A.append((f'★G-SELF 제3자 팔이 어느 칸에서든 참 제로필드를 ≥ {SELF_MIN}/{N}',
              bool(meta.get('self_ok')), str(meta.get('self_detail', ''))))
    # ★★G-ONELINE / G-MU — PREREG §3 이 약속했는데 초판이 **행으로 내지 않았다**
    #   (방벽 `check_gate_coverage` 가 잡았다 — 오늘 두 번째로 같은 부류다).
    bad = [r['record'] for r in ok if not r.get('ship_bit_identical')]
    A.append(('★G-ONELINE SHIP 열이 E1 과 비트 동일 (개입이 제3자에만 걸렸다)', not bad,
              ';'.join(sorted(set(bad))[:3]) or f'{len(ok)}행 전건'))
    bad = [r['record'] for r in ok if r.get('mu_rms_over_sigma') is None]
    A.append(('★G-MU rms(mu_mem)/σ 를 행에 기록', not bad,
              (f"중앙 {sorted(r['mu_rms_over_sigma'] for r in ok)[len(ok)//2]:.4f}"
               if ok and not bad else ';'.join(sorted(set(bad))[:3]))))
    A.append(('G-COV 게이트가 자료를 봤다', len(ok) > 0, f'rows={len(ok)}'))
    probe = (blocked_by([('G-0 합성', False, ''), ('★G-POS 합성', True, '')]) == ['G-0'])
    A.append(('G-BLOCK blocked_by 자가진단', bool(probe),
              'synthetic-fail → [G-0]' if probe else '⛔자가진단 실패'))
    return A


def blocked_by(gate_rows):
    block = ('G-0', 'G-0b', 'G-NOMOD', 'G-POS', 'G-KGRID', 'G-GT', 'G-BUDGET',
             'G-DIR', 'G-ARM', 'G-ONELINE', 'G-MU', 'G-COV')
    out = []
    for name, ok, _ in gate_rows:
        head = name.replace('★', '').split()[0]
        if head in block and not ok:
            out.append(head)
    return out
