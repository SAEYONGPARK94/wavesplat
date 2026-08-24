"""B-14 게이트·팔·문턱 — **단일 출처**. 러너와 분석기는 이 파일만 읽는다.

⛔이 파일은 **결과가 하나라도 나오기 전에** 커밋된다(git 앵커). 문턱은 아래에 있고
사후에 바꾸지 않는다. 라벨도 여기서 낸다.

## 왜 이 배치인가 (2026-08-24)
실측 12 레코드에서 제안 표현이 음성이다(`B-7`/`B-8`). 그런데 그 음성의 원인이
**표현**인지 **관측 모형**인지 분리된 적이 없다. 그리고 `B-2b` 가 이상한 것을 쟀다:
관측 잡음을 0.02 → 0.005 로 줄이면 좋아지는데 **0 으로 만들면 오히려 나빠진다**(비단조).
⟹ 저장소는 그것을 *"잡음 모형이 자료와 안 맞는다"* 로 적었으나 **개입으로 확인하지 않았다.**

★그리고 2026-08-24 외부 심사가 더 무거운 사실을 지목했다: 실측의 "관측"은
**실제 센서 반환이 아니다** — `scene_station.py:110,114` 가 유효 격자에서 12점을
**균등 추출**하고 **등방·거리무관 i.i.d. 가우스**를 더한다.

## 두 가설 (결과 전 기재, 서로 배타적이지 않다)
- **H1 관측 모형 오설정**: 백색·균등·등분산 가정이 스테레오 자료와 맞지 않는다.
  ⟹ 물리적으로 더 그럴듯한 모형에서는 **비단조가 사라진다**.
- **H2 표현 오차 바닥**: 실제 해면에 우리 스펙트럼 표현이 담을 수 없는 구조가 있고,
  백색잡음이 그 부적합을 **가려 주는 정규화**로 작동해 왔다.
  ⟹ 관측을 **조밀·무잡음**으로 줘도 무기술 기준을 못 넘는다.
★**둘 중 하나는 반드시 발화한다** — 이 배치는 "아무 일도 없었다" 로 끝날 수 없다.
"""
from math import comb

# ── 표본 ────────────────────────────────────────────────────────────────────
RECORDS = ['AA14', 'AA15a', 'AA15b', 'BS11a', 'BS11b', 'BS11c', 'BS11d',
           'BS13a', 'BS13b', 'BS13c', 'LJ18a', 'YS17']
WIN = 3          # B-2/B-2b 와 같은 창 (재현 대조를 위해 필수)
SEED = 42        # 추정기 시드
N = len(RECORDS)

# ── 추정기 팔: 제안 표현 강제 (B-2 의 `PROP` 과 **글자 그대로 같다**) ──────────
# ⛔빈 dict 로 팔을 정의하지 않는다(2026-08-23 BLOCKER 부류).
PROP_CFG = {'real_domain_gate': False, 'init_method': 'learned'}

# ── 축 A: 관측 모형 × 잡음 ──────────────────────────────────────────────────
# 전부 **스테레오** 계측에서 물리적으로 동기가 있는 것만 넣는다(LiDAR raydrop 아님).
OBS_MODELS = {
    'IID':   '유효 격자 균등 추출 + 등방 i.i.d. 가우스  (= 배포 = 재현 대조)',
    'CORR':  '균등 추출 + **공간 상관** 잡음(상관거리 8 m, 주변분포 σ 동일)',
    'CLUST': '**군집 추출**(중심 3개 반경 5 m) + i.i.d. 가우스',
    'SLOPE': '균등 추출 + **국소 경사에 비례**하는 이분산 잡음(평균 분산 동일)',
}
NOISE_LEVELS = [0.02, 0.005, 0.0]      # 배포값 · 1/4 · 무잡음
M_SCAN_A = 12                          # 배포값

# ── 축 B: 표현 용량 바닥 (무잡음, 밀도만 올린다) ─────────────────────────────
# ⚠**프레임당 점수**다. 총 관측 수 = 프레임당 × 40프레임 = 480 / 1,920 / 6,000 / 12,000.
# ⛔`m_cap` 을 함께 올리지 않으면 세 수준이 전부 600 으로 잘려 **축이 죽는다**
#   (분석기 사전 스모크가 결과 전에 이 결함을 잡았다).
DENSITIES = [12, 48, 150, 300]
DENSITY_CAP_PER_FRAME = 40      # m_cap = m_scan * 이 값
NOISE_B = 0.0

FIELD = 'nowcast'

# ── 문턱 (결과 전 계산) ─────────────────────────────────────────────────────
# 단측 부호검정, n=12:  10/12 -> 0.01929 · 11/12 -> 0.00317 · 12/12 -> 0.00024
NOMINAL = 10          # 명목
CONFIRM = 11          # 가족단위(4 모형 검정 ⟹ α=0.05/4=0.0125; 11/12 가 이를 넘는다)
FLOOR_FAIL = 6        # 이 이하이면 "표현이 벽" 쪽 증거로 읽는다(우연 근처 = 무기술)


def sign_test_one_sided(w, n=N):
    return sum(comb(n, k) for k in range(w, n + 1)) / 2 ** n


def label_h1(best_model, wins):
    """H1: 어떤 관측 모형에서 σ→0 비단조가 사라지는가."""
    if wins >= CONFIRM:
        return ('OBS-MODEL-MEDIATES',
                f'{best_model} 에서 σ→0 비단조가 사라진다 ({wins}/{N}, 가족단위 통과) '
                f'⟹ 백색·균등·등분산 가정이 실측 음성의 원인 쪽이다')
    if wins >= NOMINAL:
        return ('OBS-MODEL-MEDIATES-NOMINAL',
                f'{best_model} {wins}/{N} — 명목만. 탐색적으로 읽는다')
    return ('OBS-MODEL-DOES-NOT-MEDIATE',
            f'어떤 관측 모형에서도 비단조가 사라지지 않는다 (최대 {wins}/{N}) '
            f'⟹ 원인은 잡음의 *형태* 가 아니다')


def label_h2(wins):
    """H2: 조밀·무잡음 관측에서 제안 표현이 무기술 기준을 넘는가."""
    if wins >= CONFIRM:
        return ('REPRESENTATION-ADEQUATE',
                f'조밀·무잡음에서 {wins}/{N} 로 참 제로필드를 이긴다 '
                f'⟹ 표현은 충분하고 벽은 **희소성/잡음** 쪽이다')
    if wins >= NOMINAL:
        return ('REPRESENTATION-ADEQUATE-NOMINAL', f'{wins}/{N} — 명목만')
    if wins <= FLOOR_FAIL:
        return ('REPRESENTATION-IS-THE-WALL',
                f'조밀·무잡음을 줘도 {wins}/{N} 에 그친다 '
                f'⟹ ★실제 해면에 이 표현이 담을 수 없는 구조가 있다')
    return ('REPRESENTATION-UNDECIDED', f'{wins}/{N} — 문턱 사이')


def gates(rows):
    """러너 산출을 받아 게이트를 판정한다. (분석기가 이 함수만 부른다)"""
    ok = [r for r in rows if 'error' not in r]
    out = []
    out.append(('G-5 오류 0행', len(ok) == len(rows), f'{len(rows)-len(ok)} 행 오류'))
    need_a = len(OBS_MODELS) * len(NOISE_LEVELS) * N
    need_b = len(DENSITIES) * N
    got_a = sum(1 for r in ok if r['axis'] == 'A')
    got_b = sum(1 for r in ok if r['axis'] == 'B')
    out.append(('G-0 표본 수', got_a == need_a and got_b == need_b,
                f'A {got_a}/{need_a} · B {got_b}/{need_b}'))
    # G-2 개입이 실제로 발화했는가 — 팔마다 표본값이 서로 달라야 한다
    sig = {}
    for r in ok:
        if r['axis'] == 'A' and r['noise'] == 0.02:
            sig.setdefault(r['obs_model'], {})[r['record']] = r['z_hash']
    diff = all(sig.get(m, {}) != sig.get('IID', {}) for m in ('CORR', 'CLUST', 'SLOPE'))
    out.append(('G-2 개입 발화(관측 표본이 팔마다 다르다)', bool(sig) and diff,
                '팔별 z 해시 비교'))
    # G-3 밀도 되읽기
    dens_ok = all(r['n_obs_used'] >= min(r['m_scan'], 12) for r in ok if r['axis'] == 'B')
    out.append(('G-3 밀도 되읽기', dens_ok, 'n_obs_used ≥ m_scan(또는 유효점 상한)'))
    # G-4 참 제로필드 기준이 상수가 아니다
    zs = {round(r['zero_ref'], 6) for r in ok if 'zero_ref' in r}
    out.append(('G-4 참 제로필드가 상수 1.0 이 아니다',
                len(zs) > 1 and 1.0 not in zs, f'{len(zs)} 종'))
    # G-7 팔 정의가 비어 있지 않다
    out.append(('G-7 팔 정의 비어있지 않음', bool(PROP_CFG), str(PROP_CFG)))
    # G-8 배포 상수가 행에 실려 있다 (재실행 시 조용한 흡수 방지)
    keys = ('varpro_steps', 'learned_init_iters', 'auto_switch_thresh', 'lambda_l1')
    have = all(all(k in (r.get('effective') or {}) for k in keys) for r in ok)
    out.append(('G-8 배포 상수 되읽기 기록', have, ' · '.join(keys)))
    return out
