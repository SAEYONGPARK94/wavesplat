"""`T12-ISOLATE` **상수 단일출처** — 측정면.

⛔`PREREG.md` + `REACHABILITY.md` 가 정본이고 이 파일이 코드로 옮긴다.
★`T11` 은 러너가 `gates_t11.py` 를 상수 단일출처로 임포트했는데 개정문 3 이 그 파일을 고쳐
  **측정면/분석면 경계가 한 파일 안에서 갈렸다**(`T11 AMENDMENT_3 §2`).
  `T12` 는 **상수(이 파일 = 측정면)와 판정(`gates_t12.py` = 분석면)을 파일로 분리**한다.
⛔이 파일을 고치면 `G-FREEZE` 의 측정면 3중 대조가 깨진다.
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))          # …/wavesplat/wavesplat
REPO = os.path.abspath(os.path.join(ROOT, '..'))                # …/wavesplat
T11 = os.path.join(HERE, '..', 'T11_EXTACC_2026-09-03')
ZIP = os.path.join(REPO, 'data', 'thirdparty_synth_zenodo17359600', 'Data.zip')
WANT_BYTES = 12828148812

# ── T11 규약 축자 계승 (PREREG §2.1) ───────────────────────────────────────
SEED = 20260904
J0 = 100
WIN_S = 40.0
DENSITY = 200
DEPTH = 400.0
GT_CONST = 0.5
RESCALE_S = 2.0
N_MODE = 100                 # ORACLE 모드 수 (T11 개정 정의: 켤레쌍 정준키 + ±ω 4열)
HS_SPLIT = 2.0

# ★표본은 T11 파일을 **읽는다** (⛔복사 금지 — PREREG §2.1)
SAMPLE_JSON = os.path.join(T11, 'sample_t11.json')
SAMPLE_MD5_PIN = '56a0adc053b046081d538aa15b381766'
T11_RESULTS = os.path.join(T11, 'results_t11.jsonl')
AUDIT_SMOKE_IDX = (60, 63, 65, 68, 71, 74, 77, 80)
SMOKE_FILES = ('Data/mixed_data2/angle10h0.5.mat',
               'Data/mixed_data2/angle10h2.0.mat')
POP_RANGE = (60, 120)        # mixed_data2 의 sorted() 인덱스 창
N_PRIMARY = 24
N_SECR = 28

# ── 팔 (PREREG §1) — ⛔빈 dict `{}` 로 정의하지 않는다 ──────────────────────
ANCHOR_ARMS = ('ZERO', 'ORACLE', 'SHIP', 'RESCALE')
TREAT_ARMS = ('A-MAPSCALE', 'A-MAPCELL', 'A-BAND', 'B-DEDUP', 'C-MINDK',
              'C-GROWGRID', 'D-VPLR', 'E-MAPDEPTH', 'F-STEEP')
ARMS = ANCHOR_ARMS + TREAT_ARMS                   # 블록 내 순서를 못박는다 (§2.4)
SECR_ARMS = ('ZERO', 'SHIP', 'RESCALE', 'A-MAPSCALE')
PRIMARY_ARM = 'A-MAPSCALE'
SEC8 = ('A-MAPCELL', 'A-BAND', 'B-DEDUP', 'C-MINDK', 'C-GROWGRID',
        'D-VPLR', 'E-MAPDEPTH', 'F-STEEP')
RECOVERY_ARMS = ('A-MAPSCALE', 'A-MAPCELL', 'A-BAND', 'B-DEDUP', 'C-MINDK',
                 'C-GROWGRID', 'D-VPLR')          # ★ρ 를 인쇄해도 되는 축 (§3)
PRE_PROPOSAL = ('A-MAPSCALE', 'A-MAPCELL', 'A-BAND', 'E-MAPDEPTH')   # G-PROPID
POST_PROPOSAL = ('B-DEDUP', 'C-MINDK', 'C-GROWGRID', 'D-VPLR', 'F-STEEP')
GROWTH_ARMS = ('C-MINDK', 'C-GROWGRID')           # G-GROWFIRE 가 NOT-EXERCISED 판정

# 처치값 — 파수 차원[rad/m] 은 전부 ÷ RESCALE_S (§1)
TREAT = {
    'A-MAPSCALE': dict(KLIM=0.3, K_LIM=0.55),
    'A-MAPCELL':  dict(KLIM=0.3),
    'A-BAND':     dict(K_LIM=0.55),
    'B-DEDUP':    dict(learned_dedup_dk=0.015),
    'C-MINDK':    dict(MIN_DK=0.075),
    'C-GROWGRID': dict(grow_krange=2.0),
    'D-VPLR':     dict(vp_lr=1e-3),
    'E-MAPDEPTH': dict(MAP_DEPTH=400.0),
    'F-STEEP':    dict(max_steepness=0.30),
}

# ── 살아있는 노브 되읽기 키 (G-KNOB/G-RESTORE, PREREG §6) ─────────────────
#   ⛔`OM_G00`(모서리 한 칸)은 **구조적으로 눈이 멀었다** — k=0.8485 에서 tanh(k·50)
#   과 tanh(k·400) 이 **둘 다 정확히 1.0** 으로 포화한다(AMENDMENT_1 §2) ⟹ 전체 텐서 지문.
KNOB_KEYS = ('KLIM', 'KX_G00', 'OM_Gmd5', 'MAP_DEPTH', 'K_LIM', 'MIN_DK',
             'max_steepness', 'learned_dedup_dk', 'vp_lr', 'grow_krange')
# ★사전등록이 **기대하는** 차이 집합. 러너는 이것을 보고 처치하지 않는다(TREAT 를 본다)
#   ⟹ 개입이 조용히 no-op 이면 측정된 차이가 비어 `G-KNOB` 이 **FAIL 한다**.
#   (⛔`D-VPLR` 이 정확히 그 위험이다 — `REACHABILITY.md §5` 항목 5)
EXPECT_DIFF = {
    'SHIP': frozenset(), 'ZERO': frozenset(), 'ORACLE': frozenset(),
    'RESCALE': frozenset(),
    'A-MAPSCALE': frozenset({'KLIM', 'KX_G00', 'OM_Gmd5', 'K_LIM'}),
    'A-MAPCELL':  frozenset({'KLIM', 'KX_G00', 'OM_Gmd5'}),
    'A-BAND':     frozenset({'K_LIM'}),
    'B-DEDUP':    frozenset({'learned_dedup_dk'}),
    'C-MINDK':    frozenset({'MIN_DK'}),
    'C-GROWGRID': frozenset({'grow_krange'}),
    'D-VPLR':     frozenset({'vp_lr'}),
    'E-MAPDEPTH': frozenset({'MAP_DEPTH', 'OM_Gmd5'}),
    'F-STEEP':    frozenset({'max_steepness'}),
}

# ── 배포값 핀 (G-SHIPPIN 이 **임포트해** 대조한다 · ⛔손복사 0) ─────────────
SHIP_PIN = {
    'lambda_l1': 1e-4, 'init_method': 'auto', 'varpro_steps': 400,
    'learned_init_iters': 0, 'auto_switch_thresh': 0,
    'learned_ensemble_k': 1, 'M_OBS_CAP': 400, 'MIN_DK': 0.15,
    'VP_LR_DEFAULT': 2e-3, 'KLIM': 0.6, 'NK': 64, 'MAP_DEPTH': 50.0,
    'K_LIM': 1.1, 'learned_dedup_dk': 0.03, 'max_steepness': 0.15,
    'grow_krange': 4.0,
}

# ── 판정 임계 (PREREG §4.1 · §6-A) ─────────────────────────────────────────
THRESH_PRIMARY = 17     # n=24, α=.05 단측, p(≥17|.5)=0.031957
THRESH_SEC = 19         # n=24, Holm 첫 단계 α/8=0.00625, p(≥19|.5)=0.003310
THRESH_SECR = 19        # n=28, α=.05 단측, p(≥19|.5)=0.04358
GROWFIRE_MIN = 6        # SHIP 성장 발화 행이 이 미만이면 C-* 는 NOT-EXERCISED

PRED_Q = {'A-MAPSCALE': 0.75, 'A-MAPCELL': 0.50, 'A-BAND': 0.55,
          'B-DEDUP': 0.55, 'C-MINDK': 0.50, 'C-GROWGRID': 0.55,
          'D-VPLR': 0.50, 'E-MAPDEPTH': 0.60, 'F-STEEP': 0.50}
MDE = {'primary_q50': 0.685, 'primary_q80': 0.760, 'primary_q90': 0.7955,
       'sec_q80': 0.8332, 'secr_q80': 0.7304, 'stratum_q80': 0.8693,
       'unrejectable_w': (13, 16)}

# ── 동결 (PREREG §7) ───────────────────────────────────────────────────────
ANALYSIS_SIDE = ('gates_t12.py', 'analyze_t12.py')
BATCH_FILES = ('consts_t12.py', 'run_t12.py', 'gates_t12.py', 'analyze_t12.py',
               'PREREG.md', 'REACHABILITY.md')
IMPORT_FILES = (
    'wavesplat/optimizer/sequential_update.py', 'wavesplat/optimizer/learned_init.py',
    'wavesplat/optimizer/initializer.py',       # ★NEW — 이 배치가 MIN_DK 를 움직인다
    'wavesplat/optimizer/varpro.py',
    'wavesplat/model/spectral_gaussian.py',     'wavesplat/model/dispersion.py',
    'wavesplat/model/tnse.py',                  # ★NEW
    'wavesplat/run.py',
    'scripts/d3_beamform_net.py',               'scripts/d3_beamform_cond_diffusion.py',
    'scripts/d3_setdiff_prior.py',              # ★NEW
    'scripts/d3_setdiff_posterior_b2.py',       # ★NEW
)
CKPTS = ('results/direction3/beamform_cond_diffusion_mixed_v1.pt',
         'results/direction3/setdiff_prior.pt')
CKPT_SHA_PIN = {'results/direction3/beamform_cond_diffusion_mixed_v1.pt':
                '636c72a797ee6d0b725449293d5cd230dbd64c8eb03e345da4561b0c265239d9'}
# ⛔T11 참조 파일 — 이 배치가 **읽기만** 하지만 바뀌면 앵커·표본이 흔들린다
#   ★러너는 T11 의 관측 표집·오라클을 **임포트해 재사용**한다(시드 규약 비트 보존)
T11_REFS = ('sample_t11.json', 'results_t11.jsonl', 'run_t11.py', 'gates_t11.py')


def file_md5(path):
    import hashlib
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def file_sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def freeze_map():
    """측정면 + 분석면 전부. ⛔런 **시작 전과 종료 후** 둘 다 잰다 (PREREG §7)."""
    out = {}
    for f in BATCH_FILES:
        p = os.path.join(HERE, f)
        if os.path.exists(p):
            out[f] = file_md5(p)
    for f in IMPORT_FILES:
        p = os.path.join(ROOT, f)
        if os.path.exists(p):
            out[f] = file_md5(p)
    for f in CKPTS:
        p = os.path.join(ROOT, f)
        if os.path.exists(p):
            out['sha256:' + f] = file_sha256(p)
    for f in T11_REFS:
        p = os.path.join(T11, f)
        if os.path.exists(p):
            out['T11/' + f] = file_md5(p)
    return out


def sample():
    """★T11 의 동결 표본을 **읽는다** (⛔복사 금지)."""
    return json.load(open(SAMPLE_JSON, encoding='utf-8'))


def archive_mats():
    import zipfile
    with zipfile.ZipFile(ZIP) as zf:
        return sorted(n for n in zf.namelist() if n.endswith('.mat'))


def exposed_blocks():
    """사전노출 = 감사 스모크 8(인덱스) ∪ 러너 스모크 2."""
    mats = archive_mats()
    return sorted(set(mats[i] for i in AUDIT_SMOKE_IDX if i < len(mats))
                  | set(SMOKE_FILES)), mats


def heldout_blocks():
    """★SEC-R 층 — 분석기·러너가 namelist 에서 **계산**한다 (⛔손 목록 금지, §2.2)."""
    exposed, mats = exposed_blocks()
    pop = mats[POP_RANGE[0]:POP_RANGE[1]]
    return sorted(set(pop) - set(sample()) - set(exposed))


def deployed():
    """★배포 상수를 **임포트해** 되읽는다 (G-SHIPPIN · ⛔손복사 0)."""
    import sys as _sys
    for p in (ROOT, os.path.join(ROOT, 'scripts')):   # ★scripts·results 는 ROOT 아래다
        p = os.path.abspath(p)
        if p not in _sys.path:
            _sys.path.insert(0, p)
    from wavesplat.optimizer.learned_init import M_OBS_CAP
    from wavesplat.optimizer.initializer import MIN_DK
    from wavesplat.optimizer.varpro import VP_LR_DEFAULT
    from wavesplat.optimizer.sequential_update import (
        DEFAULT_LEARNED_ENSEMBLE_K, VARPRO_STEPS_DEFAULT,
        LEARNED_INIT_ITERS_DEFAULT, AUTO_SWITCH_THRESH_DEFAULT, LAMBDA_L1_DEFAULT)
    from wavesplat.run import DEFAULT_CONFIG
    import d3_beamform_net as bn
    out = {'lambda_l1': float(LAMBDA_L1_DEFAULT),
           'init_method': DEFAULT_CONFIG.get('init_method'),
           'varpro_steps': int(VARPRO_STEPS_DEFAULT),
           'learned_init_iters': int(LEARNED_INIT_ITERS_DEFAULT),
           'auto_switch_thresh': int(AUTO_SWITCH_THRESH_DEFAULT),
           'learned_ensemble_k': int(DEFAULT_LEARNED_ENSEMBLE_K),
           'M_OBS_CAP': int(M_OBS_CAP), 'MIN_DK': float(MIN_DK),
           'VP_LR_DEFAULT': float(VP_LR_DEFAULT),
           'KLIM': float(bn.KLIM), 'NK': int(bn.NK),
           'MAP_DEPTH': float(bn.DEPTH),
           # ★KNOB_KEYS 와 **키 이름이 정확히 맞아야** G-KNOB 이 비교한다
           #   (스모크가 잡았다: 이름이 어긋나면 그 키가 조용히 비교에서 빠진다)
           'KX_G00': float(bn.KX_G[0, 0].item()),
           'OM_Gmd5': __import__('hashlib').md5(
               bn.OM_G.detach().cpu().numpy().tobytes()).hexdigest(),
           'vp_lr': float(VP_LR_DEFAULT),
           'learned_dedup_dk': float(DEFAULT_CONFIG.get('learned_dedup_dk', 0.03)),
           'max_steepness': float(DEFAULT_CONFIG.get('max_steepness', 0.15)),
           'grow_krange': 4.0}          # ⚠하드코딩 리터럴(sequential_update.py:1693) — 핀만 한다
    import torch
    out['K_LIM'] = float(torch.load(
        os.path.join(ROOT, 'results/direction3/setdiff_prior.pt'),
        map_location='cpu')['stats']['K_LIM'])
    return out
