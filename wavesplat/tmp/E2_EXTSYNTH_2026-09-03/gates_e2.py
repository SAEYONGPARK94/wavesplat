"""`E2-EXTSYNTH` 규칙 단일출처. ⛔`PREREG.md` 가 정본이고 이 파일이 코드로 옮긴다."""
import os
import sys
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
E1 = os.path.join(ROOT, 'tmp', 'E1_EXTBASE_2026-09-03')
C1 = os.path.join(ROOT, 'tmp', 'C1_MODELGAP_2026-09-03')
for _p in (E1, C1):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import gates_e1 as E                                                 # noqa: E402
import gates_c1 as C                                                 # noqa: E402

HOS = C.HOS                       # 절제 사슬이 실제로 도는 64 에피소드
is_crowded = C.is_crowded
CNN_N = E.CNN_N                   # 256
EXT_COMMIT, CKPT_SPARSE, CKPT_FULL = E.EXT_COMMIT, E.CKPT_SPARSE, E.CKPT_FULL
ext_clean = E.ext_clean
POS_NRMSE_MAX = E.POS_NRMSE_MAX
KGRID_TOL = E.KGRID_TOL
SAMPLE_SEED = E.SAMPLE_SEED
SIGNS = E.SIGNS
DIR_HOLDOUT = E.DIR_HOLDOUT

N_CROWDED, N_SPARSE = 48, 16
LADDER = (200, 800, 3200)         # ⛔상위 칸은 **참값 표본**(PREREG §1)
SELF_MIN = 34                     # 단측 부호검정 p = 0.0303 (결과 전 고정)


def sign_test_one_sided(w, n):
    return sum(comb(n, k) for k in range(w, n + 1)) / 2.0 ** n


def label(deploy_w, ladder_pass, rung):
    """⛔결과 전 동결 판정표(PREREG §1)."""
    if deploy_w >= SELF_MIN:
        return ('SYNTH-EXTERNAL-VALID',
                f'배포 밀도에서 {deploy_w}/{N_CROWDED} — 합성에도 유효 외부 상대가 있다')
    if ladder_pass:
        return ('SYNTH-EXTERNAL-NEEDS-DENSITY',
                f'배포 밀도 {deploy_w}/{N_CROWDED} 미달, {rung} 점/프레임에서 통과 — '
                '★밀도 문턱을 수로 낸다. ⛔상위 칸은 **참값 표본**이라 실현 가능한 관측이 아니다')
    return ('SYNTH-NO-VALID-COMPETITOR',
            f'어느 칸에서도 미달(배포 {deploy_w}/{N_CROWDED})')


def gates(rows, meta):
    ok = [r for r in rows if 'error' not in r]
    crow = [r for r in ok if r['crowded']]
    spar = [r for r in ok if not r['crowded']]
    A = []
    A.append(('★G-NOMOD 제3자 저장소 무수정 · HEAD 핀', bool(meta.get('ext_clean')),
              str(meta.get('ext_msg'))))
    A.append((f'★G-POS 양성 대조 nRMSE ≤ {POS_NRMSE_MAX}',
              meta.get('pos_nrmse') is not None and meta['pos_nrmse'] <= POS_NRMSE_MAX,
              str(meta.get('pos_nrmse'))))
    A.append(('★G-KGRID 파수 격자가 저들 config256.mat 과 원소별 일치',
              bool(meta.get('kgrid_ok')), f"relerr={meta.get('kgrid_relerr')}"))
    bad = [r['episode'] for r in ok if not r.get('deploy_is_stored')]
    A.append(('★G-DEPLOY 배포 칸 관측 == 저장된 LiDAR 반환 그대로', not bad,
              ';'.join(sorted(set(bad))[:3]) or '전건'))
    bad = [r['episode'] for r in ok if not r.get('gt_free_deploy')]
    A.append(('★G-GT 배포 칸 입력에 참값 0', not bad,
              ';'.join(sorted(set(bad))[:3]) or '전건'))
    A.append((f'G-N 혼잡 {N_CROWDED} · 희소 {N_SPARSE}',
              len(crow) == N_CROWDED and len(spar) == N_SPARSE,
              f'{len(crow)}/{len(spar)}'))
    A.append(('G-0b 에러 0', len(rows) == len(ok), str(len(rows) - len(ok))))
    A.append(('G-COV 게이트가 자료를 봤다', len(ok) > 0, f'rows={len(ok)}'))
    probe = (blocked_by([('G-NOMOD 합성', False, ''), ('★G-POS 합성', True, '')]) == ['G-NOMOD'])
    A.append(('G-BLOCK blocked_by 자가진단', bool(probe),
              'synthetic-fail → [G-NOMOD]' if probe else '⛔자가진단 실패'))
    return A


def blocked_by(gate_rows):
    block = ('G-NOMOD', 'G-POS', 'G-KGRID', 'G-DEPLOY', 'G-GT', 'G-N', 'G-0b', 'G-COV')
    out = []
    for name, ok, _ in gate_rows:
        head = name.replace('★', '').split()[0]
        if head in block and not ok:
            out.append(head)
    return out
