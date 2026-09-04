"""`T15-HOMEREG` **상수 단일출처** — 측정면. ⛔`PREREG.md` 가 정본.

★처치는 `T12` 에서 **임포트**한다(⛔손복사 0). `G-TREATPIN` 이 `assert` 로 핀한다.
⛔`f` 는 **노브 스케일 인자**다 — `T11`·`T14` 의 `s`(자료 재라벨)와 **다른 양**이다.
"""
import glob, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))          # …/wavesplat/wavesplat
T12 = os.path.abspath(os.path.join(HERE, '..', 'T12_ISOLATE_2026-09-04'))
W15 = os.path.abspath(os.path.join(HERE, '..', 'W15_DEPLOY_VERIFY_2026-08-22'))
CORPUS_JSON = os.path.join(W15, 'corpus.json')
DATA_DIRS = ('data_s8_crowded_2026-08-10', 'data_s8_sparse_2026-08-10',
             'data_s9_crowded_2026-08-11', 'data_s9_sparse_2026-08-11',
             'data_indep2_2026-08-07', 'data')

# ⚠형제 대역: T11/T12 20260905–928 · T14 20260915–948 · T13-FRESH 20261005–032
#   ⟹ 20261101–20261228 로 옮긴다(128 연속·전부 서로소). ⛔동결 전에 고쳤다.
SEED15 = 20261100
N_FULL, N_SUB, N_LAYER = 128, 64, 64
DEPTH_HOME = 50.0            # ⚠자국 코퍼스 기본 — ⛔T11~T14 의 400 이 아니다. 되읽어 기록한다
TP_FACTOR = 2.0              # 채점창 t > 2·Tp (run.py:272)

# ── 판정 (PREREG §5·§6) ────────────────────────────────────────────────────
THRESH_HURT = 23      # n=64, Holm 첫 단계 α/2=0.025 ⟹ P(W<=23)=0.016380
THRESH_HELP = 41      # 같은 문턱의 반대편 (기술통계)
PRED_Q = 0.30         # ★등록 예측 — 근거는 PREREG §6 (⛔믿음 선언 아님)
# ★양성대조의 등록 예측. 근거 = `W16_REBASELINE` 동결 중앙 nRMSE(혼잡 0.1275 · 희소 0.0202)
#   대 `ZERO` ≈ 0.25 ⟹ 셀별 승률은 1 에 가깝다. ⛔0.95 로 **보수적으로** 건다.
PRED_Q_CTRL = 0.95
MDE = {'q50': 0.3617, 'q80': 0.3185, 'q90': 0.2952,
       'unrejectable_w': (24, 40), 'cell_q80': 0.0975}

# ── 팔 (PREREG §2) — ⛔`{}` 로 정의하지 않는다 ─────────────────────────────
ARM_SPEC = {
    'SHIP':       dict(f=1.0, which='both'),
    'F2':         dict(f=2.0, which='both'),      # ★= T12 의 A-MAPSCALE
    'F1.414':     dict(f=2 ** 0.5, which='both'),
    'F2.828':     dict(f=2 ** 1.5, which='both'),
    'F4':         dict(f=4.0, which='both'),
    'A-MAPCELL':  dict(f=2.0, which='cell'),      # KLIM 만
    'A-BAND':     dict(f=2.0, which='band'),      # K_LIM 만
    'FLOOR-VPLR': dict(vp_lr=1e-3),               # ★처치정합 잡음 바닥
}
FULL_ARMS = ('ZERO', 'SHIP', 'F2', 'FLOOR-VPLR')          # 128 블록 전부
SUB_ARMS = ('F1.414', 'F2.828', 'F4', 'A-MAPCELL', 'A-BAND')   # 64 부분층
EST_ARMS = tuple(a for a in FULL_ARMS + SUB_ARMS if a != 'ZERO')
KNOB_KEYS = ('KLIM', 'KX_G00', 'OM_Gmd5', 'K_LIM', 'vp_lr', 'MAP_DEPTH', 'MIN_DK',
             'max_steepness', 'learned_dedup_dk', 'grow_krange')
# ★기대 노브 차이 — 러너는 이것을 보고 처치하지 않는다(ARM_SPEC 을 본다)
#   ⟹ 개입이 조용히 no-op 이면 측정된 차이가 비어 `G-KNOB` 이 **FAIL 한다**
EXPECT_DIFF = {
    'ZERO': frozenset(), 'SHIP': frozenset(),
    'F2':         frozenset({'KLIM', 'KX_G00', 'OM_Gmd5', 'K_LIM'}),
    'F1.414':     frozenset({'KLIM', 'KX_G00', 'OM_Gmd5', 'K_LIM'}),
    'F2.828':     frozenset({'KLIM', 'KX_G00', 'OM_Gmd5', 'K_LIM'}),
    'F4':         frozenset({'KLIM', 'KX_G00', 'OM_Gmd5', 'K_LIM'}),
    'A-MAPCELL':  frozenset({'KLIM', 'KX_G00', 'OM_Gmd5'}),
    'A-BAND':     frozenset({'K_LIM'}),
    'FLOOR-VPLR': frozenset({'vp_lr'}),
}
ANALYSIS_SIDE = ('gates_t15.py', 'analyze_t15.py')
BATCH_FILES = ('consts_t15.py', 'run_t15.py', 'gates_t15.py', 'analyze_t15.py',
               'PREREG.md', 'PLUMBING.md')
IMPORT_FILES = (
    'wavesplat/optimizer/sequential_update.py', 'wavesplat/optimizer/learned_init.py',
    'wavesplat/optimizer/initializer.py', 'wavesplat/optimizer/varpro.py',
    'wavesplat/model/spectral_gaussian.py', 'wavesplat/model/dispersion.py',
    'wavesplat/model/tnse.py', 'wavesplat/run.py', 'wavesplat/data_loader.py',
    'wavesplat/evaluation/metrics.py', 'wavesim/data_writer.py',
    'scripts/d3_beamform_net.py', 'scripts/d3_beamform_cond_diffusion.py',
    'scripts/d3_setdiff_prior.py', 'scripts/d3_setdiff_posterior_b2.py',
)
CKPTS = ('results/direction3/beamform_cond_diffusion_mixed_v1.pt',
         'results/direction3/setdiff_prior.pt')
T12_REFS = ('consts_t12.py', 'run_t12.py', 'FROZEN_md5.txt')   # 처치 단일출처

# ★형제 배치의 **동결 결과행** — `G-SIBLING` 이 시드·(block,seed,proposal_md5)·값 비트일치를
#   전부 교집합 0 으로 확인한다. ⛔`T13` 이 `T12` 와 24행 비트 동일이면서 *"독립"* 이라 적었다가
#   철회한 사고의 재발 방지 — **시드가 다르다는 것만으로는 증명이 아니다**.
SIBLINGS = ('T11_EXTACC_2026-09-03/results_t11.jsonl',
            'T12_ISOLATE_2026-09-04/results_t12.jsonl',
            'T12_ISOLATE_2026-09-04/results_t12_SECR.jsonl',
            'T13_DEPTHFIX_2026-09-04/results_t13.jsonl',
            'T14_AUTOSCALE_2026-09-04/results_t14.jsonl')

# ── 행 수 (⛔analyze 가 이 값으로 G-N 을 건다) ─────────────────────────────
N_ROWS_FULL = N_LAYER * 2 * len(FULL_ARMS)          # 128 x 4 = 512
N_ROWS_SUB = N_SUB * len(SUB_ARMS)                  #  64 x 5 = 320
N_ROWS = N_ROWS_FULL + N_ROWS_SUB                   # = 832  (PREREG §9)


def file_md5(p):
    import hashlib
    return hashlib.md5(open(p, 'rb').read()).hexdigest()


def file_sha256(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def freeze_map():
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
    for f in T12_REFS:
        p = os.path.join(T12, f)
        if os.path.exists(p):
            out['T12/' + f] = file_md5(p)
    if os.path.exists(CORPUS_JSON):
        out['W15/corpus.json'] = file_md5(CORPUS_JSON)
    return out


def _index():
    idx = {}
    for d in DATA_DIRS:
        for p in glob.glob(os.path.join(ROOT, d, '*.h5')):
            b = os.path.basename(p)
            if b.startswith('episode_') and b.endswith('.h5'):
                idx.setdefault(b[len('episode_'):-3], []).append(p)
    return idx


def corpus():
    """★정본 128 — `W15/corpus.json` 에서 **읽고** 실물을 해소한다 (⛔손목록 금지)."""
    c = json.load(open(CORPUS_JSON, encoding='utf-8'))
    idx = _index()
    out = {}
    for layer in ('crowded', 'sparse'):
        out[layer] = [(n, sorted(idx[n])[0]) for n in c[layer] if n in idx]
    return out


def sub_layer():
    """스윕 층(64) — 각 (BF, comp) 셀에서 **시드 오름차순 앞 8개** (⛔손목록 금지)."""
    import re
    out = {}
    for layer, items in corpus().items():
        cells = {}
        for n, p in items:
            m = re.match(r'(BF\d+)_(\d+comp)_seed(\d+)_', n)
            cells.setdefault((m.group(1), m.group(2)), []).append((int(m.group(3)), n, p))
        out[layer] = [(n, p) for k in sorted(cells)
                      for _, n, p in sorted(cells[k])[:8]]
    return out


def seed_of(name):
    order = [n for layer in ('crowded', 'sparse') for n, _ in corpus()[layer]]
    return SEED15 + order.index(name) + 1


def deployed_knobs():
    """배포 노브를 **임포트해** 되읽는다 (⛔손복사 0)."""
    import sys as _sys
    for p in (ROOT, os.path.join(ROOT, 'scripts')):
        if p not in _sys.path:
            _sys.path.insert(0, p)
    import hashlib, torch
    import d3_beamform_net as bn
    from wavesplat.optimizer.initializer import MIN_DK
    from wavesplat.optimizer.varpro import VP_LR_DEFAULT
    from wavesplat.run import DEFAULT_CONFIG
    return {'KLIM': float(bn.KLIM), 'KX_G00': float(bn.KX_G[0, 0].item()),
            'OM_Gmd5': hashlib.md5(bn.OM_G.detach().cpu().numpy().tobytes()).hexdigest(),
            'MAP_DEPTH': float(bn.DEPTH), 'MIN_DK': float(MIN_DK),
            'vp_lr': float(VP_LR_DEFAULT), 'grow_krange': 4.0,
            'K_LIM': float(torch.load(os.path.join(ROOT, CKPTS[1]),
                                      map_location='cpu')['stats']['K_LIM']),
            'max_steepness': float(DEFAULT_CONFIG.get('max_steepness', 0.15)),
            'learned_dedup_dk': float(DEFAULT_CONFIG.get('learned_dedup_dk', 0.03))}


def knobs_for(arm, dep=None):
    """팔의 **처치값**. ⛔`f=2, which='both'` 는 `T12 TREAT['A-MAPSCALE']` 와 같아야 한다."""
    dep = deployed_knobs() if dep is None else dep
    sp = ARM_SPEC[arm]
    if 'vp_lr' in sp:
        return {'vp_lr': float(sp['vp_lr'])}
    f = float(sp['f'])
    if f == 1.0:
        return {}
    out = {}
    if sp['which'] in ('both', 'cell'):
        out['KLIM'] = dep['KLIM'] / f
    if sp['which'] in ('both', 'band'):
        out['K_LIM'] = dep['K_LIM'] / f
    return out


def treat_pin():
    """★`G-TREATPIN` — `f=2` 가 `T12` 의 `A-MAPSCALE` 과 **같은 값**임을 임포트로 확인."""
    import sys as _sys
    if T12 not in _sys.path:
        _sys.path.insert(0, T12)
    import consts_t12 as C12
    mine = knobs_for('F2')
    theirs = dict(C12.TREAT['A-MAPSCALE'])
    ok = (set(mine) == set(theirs)
          and all(abs(mine[k] - theirs[k]) < 1e-12 for k in mine))
    return ok, mine, theirs


def deployed_cap():
    """★제안 단계의 관측 캡을 **배포 모듈에서 임포트** (⛔손복사 0 — T11 선례)."""
    import sys as _sys
    if ROOT not in _sys.path:
        _sys.path.insert(0, ROOT)
    from wavesplat.optimizer.learned_init import M_OBS_CAP
    return int(M_OBS_CAP)


def deployed_ensemble_k():
    """★`G-ARM` 이 핀하는 배포 앙상블 폭 (⛔손복사 금지 — `D3` 결정으로 1 이다).

    ⚠`arm_provenance()` 의 `ensemble_collapsed=True`(요청 3 vs 실효 1)는 자국 학습 경로의
      **정상값**이다(`sequential_update.py:2090-2126 _effective_k`). 위반이 아니다.
    """
    import sys as _sys
    if ROOT not in _sys.path:
        _sys.path.insert(0, ROOT)
    from wavesplat.optimizer.sequential_update import DEFAULT_LEARNED_ENSEMBLE_K
    return int(DEFAULT_LEARNED_ENSEMBLE_K)
