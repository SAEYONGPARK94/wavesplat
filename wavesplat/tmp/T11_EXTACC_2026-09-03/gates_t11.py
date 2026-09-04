"""`T11-EXTACC` **v2 + 개정문 2** 규칙 단일출처.
⛔`PREREG.md` + `AMENDMENT_2_PREDATA.md` 가 정본이고 이 파일이 코드로 옮긴다.

★게이트·분석기는 **러너보다 먼저** 쓰인다(2026-09-04 규율).
⛔v1 은 폐기됐다 — `AMENDMENT_1_PREDATA.md`.
⛔사다리·문턱·ORACLE 정의는 `AMENDMENT_2_PREDATA.md` 에서 **결과 0행 시점에** 교체됐다.
⚠`AMENDMENT_3_POSTRESULT.md`(**결과 이후**)가 게이트를 더했다 — 전부 **막을 수만 있는** 것이고
  `run_t11.py`·임포트 경로·체크포인트는 **한 비트도 안 바뀌었다**(`G-ENV` 측정면 3중 대조).
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
ZIP = os.path.abspath(os.path.join(ROOT, '..', 'data',
                                   'thirdparty_synth_zenodo17359600', 'Data.zip'))
WANT_BYTES = 12828148812

# ── 결과 전 고정 (PREREG v2 §1·§2·§4) ──────────────────────────────────────
ARMS = ('SHIP', 'RESCALE', 'ORACLE', 'ZERO')
DENSITY = 200                 # 점/프레임 (⛔사다리 폐기 — v1 BLOCKER-1)
WIN_S = 40.0                  # 창 (저들 dt=1.0 s ⟹ 40 프레임)
RESCALE_S = 2.0               # ★분산관계 보존 재라벨 x'=x/s, t'=t/√s
DEPTH = 400.0                 # T10 의 NOT-SHALLOW 와 정합
GT_CONST = 0.5                # 붙이는 상수 장 (참 표면 정보 0비트)
SEED = 20260904
HS_SPLIT = 2.0
N_FILES = 24                  # ★AMENDMENT_2 §7 `G-N` — 문턱 17 은 n=24 **전용**이다
J0 = 100                      # 창 시작 프레임 (초판은 러너 지역 리터럴이었다)
N_MODE = 100                  # ORACLE 모드 수 (초판은 러너 지역 리터럴이었다)

# ── 판정 (AMENDMENT_2 §1) ──────────────────────────────────────────────────
THRESH_WIN = 17               # P(W>=17 | n=24, q=.5) = 0.031957  ← 유일한 검정
SIG_WORSE = 7                 # P(W<=7 | n=24, q=.5) = 0.031957   ← 라벨 아님, 기록만
# ⛔`THRESH_LOSE` 는 삭제됐다. 옛 13(= w<=12)은 귀무 아래 **0.5806** 으로 발화하는데
#   `PREREG.md:55` 가 그것을 **양성 주장**으로 정의했다(AMENDMENT_2 §1).
#   ⛔대칭화(w<=7)도 기각했다 — `BAND-MISMATCH` 가 *"SHIP 이 유의하게 더 나쁘다"* 를
#   요구하게 되어 **질문이 바뀐다**. 채택안은 「이김 다리만 검정 + 이름으로 정직화」다.

PRED_Q = (0.75, 0.55)         # ★사전등록 예측 (q_rescale, q_ship) — AMENDMENT_2 §2
MDE_Q80 = 0.760               # α=.05 단측 · 검정력 .80 · n=24  ⟹ ★인용 시 병기 의무
MDE_Q50, MDE_Q90 = 0.685, 0.7955   # ⚠A3 §8: 0.795 의 검정력은 0.8998 < 0.90
UNREJECTABLE = (13, 16)       # w 가 이 구간이면 **어떤 자료로도** 기각 불가(p .4194~.0758)

# ── 사전노출 (PREREG §4 · AMENDMENT_2 §7 `G-SAMPLE`) ───────────────────────
# `NOTE.md §8`: 인덱스는 아카이브의 `.mat` 을 `sorted()` 한 목록 기준. 60–119 = mixed_data2.
AUDIT_SMOKE_IDX = (60, 63, 65, 68, 71, 74, 77, 80)
# ★러너의 스모크도 여기서 단일출처화한다(⛔러너가 따로 들고 있으면 게이트가 못 본다).
SMOKE_FILES = ('Data/mixed_data2/angle10h0.5.mat',
               'Data/mixed_data2/angle10h2.0.mat')

# ── 동결 대상 (AMENDMENT_2 §8) ─────────────────────────────────────────────
BATCH_FILES = ('gates_t11.py', 'analyze_t11.py', 'run_t11.py', 'sample_t11.json',
               'PREREG.md', 'AMENDMENT_1_PREDATA.md', 'AMENDMENT_2_PREDATA.md',
               'AMENDMENT_3_POSTRESULT.md', 'NOTE.md')
# ★A3 §2: 동결을 **측정면 / 분석면**으로 가른다. 개정문 3 이 아래 셋만 고치므로
#   `meta['code_md5']`(런 시점 기록)와 어긋나는 것이 **정상**이고, 나머지 전부는
#   원래 `FROZEN_md5.txt` 와 **비트 동일**해야 한다 ⟹ "측정을 안 건드렸다" 의 기계 증명.
ANALYSIS_SIDE = ('gates_t11.py', 'analyze_t11.py', 'AMENDMENT_3_POSTRESULT.md')
POP_RANGE = (60, 120)      # ★A3 §4: `mixed_data2` 의 sorted() 인덱스 창 (PREREG §4)
# ⛔초판은 셋만 덮었다. 판정 경로에서 **값을 실제로 움직이는** 파일을 전부 넣는다.
IMPORT_FILES = ('wavesplat/optimizer/sequential_update.py',
                'wavesplat/optimizer/learned_init.py',
                'wavesplat/optimizer/varpro.py',          # kx/ky 를 재적합해 덮어쓴다
                'wavesplat/model/spectral_gaussian.py',   # 예측장
                'wavesplat/model/dispersion.py',
                'wavesplat/run.py',
                'scripts/d3_beamform_net.py',             # ★DEPTH=50 하드코딩 지점
                'scripts/d3_beamform_cond_diffusion.py')


def deployed_cap():
    """★제안 단계의 관측 캡을 **배포 모듈에서 임포트**한다.

    ⛔초판은 `CAP_PROPOSAL = 400` 을 **손으로 베껴** 적었다. 저장소가 그 부류로 여러 번
    대가를 치렀다(「손복사 기준 상수 사냥꾼」이 생긴 이유).
    ⛔캡을 올리지 않는다 — 올리면 출하 팔이 아니게 된다.
    """
    import sys as _sys
    if ROOT not in _sys.path:
        _sys.path.insert(0, ROOT)
    from wavesplat.optimizer.learned_init import M_OBS_CAP
    return int(M_OBS_CAP)


def deployed_ensemble_k():
    """★`G-ARM` 이 핀하는 배포 앙상블 폭 (⛔손복사 금지 — D3 결정으로 1 이다)."""
    import sys as _sys
    if ROOT not in _sys.path:
        _sys.path.insert(0, ROOT)
    from wavesplat.optimizer.sequential_update import DEFAULT_LEARNED_ENSEMBLE_K
    return int(DEFAULT_LEARNED_ENSEMBLE_K)


def deployed_ckpt_sha():
    """★확산 체크포인트 지문 — **학습 팔 정체의 절반**인데 초판은 기록조차 안 했다."""
    import sys as _sys
    if ROOT not in _sys.path:
        _sys.path.insert(0, ROOT)
    from wavesplat.optimizer.learned_init import ckpt_sha256
    return ckpt_sha256()


def deployed_g():
    """★중력상수 단일출처 (⛔러너의 손복사 9.80665 를 대체)."""
    import sys as _sys
    if ROOT not in _sys.path:
        _sys.path.insert(0, ROOT)
    from wavesplat.model.dispersion import G as _G
    return float(_G)


def file_md5(path):
    import hashlib
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def freeze_map():
    """배치 + 임포트 경로 + 체크포인트의 md5. ⛔런 **시작 전**에도 잰다."""
    out = {}
    for f in BATCH_FILES:
        p = os.path.join(HERE, f)
        if os.path.exists(p):
            out[f] = file_md5(p)
    for f in IMPORT_FILES:
        p = os.path.join(ROOT, f)
        if os.path.exists(p):
            out[f] = file_md5(p)
    try:
        out['ckpt_sha256'] = deployed_ckpt_sha()
    except Exception as e:                                    # noqa: BLE001
        out['ckpt_sha256'] = f'ERROR:{type(e).__name__}'
    return out


def sample():
    """★표본은 **파일에 동결**돼 있다 — 러너도 분석기도 여기서 읽는다."""
    return json.load(open(os.path.join(HERE, 'sample_t11.json'), encoding='utf-8'))


def archive_mats():
    """아카이브의 `.mat` 을 `sorted()` 한 목록 (`NOTE.md §8` 의 인덱스 규약)."""
    import zipfile
    with zipfile.ZipFile(ZIP) as zf:
        return sorted(n for n in zf.namelist() if n.endswith('.mat'))


def exposed_blocks():
    """사전노출 = 감사 스모크 8(인덱스) + 러너 스모크 2. ⛔둘 다 표본에서 배제돼야 한다."""
    mats = archive_mats()
    idx = [mats[i] for i in AUDIT_SMOKE_IDX if i < len(mats)]
    return sorted(set(idx) | set(SMOKE_FILES)), mats


def label(w_rescale, w_ship, w_oracle):
    """귀속 사다리 — **AMENDMENT_2 §1 이 교체했다**.

    이김 다리(`>= THRESH_WIN`)만 검정이고, 나머지는 **이름 자체가 「미검출」**이다.
    ⛔`NEITHER-DETECTED` 를 *"도달 불가"*·*"대역 말고 다른 것"* 으로 읽지 마라 —
    그것은 검정력의 진술이다(`MDE_Q80` 병기 의무). 선례 = `T5-ENS` 개정문.
    ★**5라벨**이 9조합을 **전부** 덮는다 ⟹ `UNDECIDED` 가 없다 (⛔A3 §8: 개정문 2 의
    *"4라벨"* 은 오기였다 — 음성 칸이 `o_win` 으로 둘로 갈린다).
    """
    r_win = w_rescale >= THRESH_WIN
    s_win = w_ship >= THRESH_WIN
    o_win = w_oracle >= THRESH_WIN
    if r_win and not s_win:
        return 'BAND-MISMATCH'
    if r_win and s_win:
        return 'BOTH-WORK'
    if s_win and not r_win:
        return 'RESCALE-HURTS'        # ★신설 — 대역가설의 가장 강한 반증
    return 'NEITHER-DETECTED' if o_win else 'NO-INFORMATION-IN-OBS'


def blocked_by(rows):
    """⛔**긴 이름부터** 접두 비교 (`G-NOLEAK` 이 `G-N` 보다 먼저 잡혀야 한다)."""
    block = ('G-NOLEAK', 'G-ATOMS', 'G-CAP', 'G-CLAMP', 'G-WRAPPER', 'G-ARM',
             'G-TORIGIN', 'G-SIZE', 'G-SAMPLE', 'G-RESCALE', 'G-ENV', 'G-BLOCK',
             'G-N', 'G-FINITE', 'G-COV', 'G-DENSITY', 'G-POP')
    order = sorted(block, key=len, reverse=True)
    out = []
    for name, ok, _ in rows:
        if ok:
            continue
        nm = name.replace('★', '').strip()
        for b in order:
            if nm.startswith(b):
                out.append(b)
                break
    return out
