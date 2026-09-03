#!/usr/bin/env python3
"""`paper/paper_constants.tex` 생성 — ⛔논문의 수치를 **손으로 적지 마라**.

선례 = `scripts/gen_deploy_constants.py`(학위논문 배포 상수).
사유 = 이 저장소가 반복해서 대가를 치른 **「손복사 기준 상수」** 결함 부류.
`paper/waveprior.tex` 는 철회된 헤드라인을 **라이브 22줄**에 손으로 인쇄하고 있었고,
그 22줄은 서로 독립적으로 낡을 수 있었다.

★이 스크립트는 **동결 `verdict_*.json` 에서 값을 읽고**, 아래 `PINS` 와 어긋나면
`AssertionError` 로 죽는다. ⟹ 배치가 재분석되면 논문이 **조용히** 낡지 않는다.

사용:
    python3 scripts/gen_paper_constants.py          # 검증 + 생성
    python3 scripts/gen_paper_constants.py --show   # 값만 출력
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
TMP = os.path.join(ROOT, 'tmp')
OUT = os.path.join(ROOT, 'paper', 'paper_constants.tex')


def _get(v, path):
    """경로를 따라간다. ⚠키 자체에 점이 있는 경우(`ladder` 의 `"0.0"`)를 위해
    **튜플 경로**도 받는다 — 점 분할이 그런 키를 조용히 쪼갠다."""
    cur = v
    keys = path if isinstance(path, (list, tuple)) else path.split('.')
    for k in keys:
        if isinstance(cur, list):
            cur = cur[int(k)]
        else:
            cur = cur[k]
    return cur


def _count_tests():
    """pytest --collect-only 로 실제 테스트 수를 센다(thesis 생성기와 같은 규약)."""
    import subprocess
    r = subprocess.run([sys.executable, '-m', 'pytest', 'wavesim/tests/',
                        'wavesplat/tests/', '--collect-only', '-q'],
                       cwd=ROOT, capture_output=True, text=True)
    import re as _re
    m = _re.search(r'(\d+)\s+tests? collected', r.stdout)
    return m.group(1) if m else '0'


def _corpus():
    """★코퍼스 **구성**을 동결 결과 행에서 센다.

    5차 감사(2026-09-02) 도전 B·F:

    > *"64 'episodes' are drawn from 5 seeds x ~10 configurations; the sign test
    >  treats them as 64 i.i.d. trials."* / *"corpus composition is never given."*

    실측: 두 코퍼스 다 **완전 요인 설계**다 --- 시드 x Beaufort x 성분수.
    ⟹ 에피소드는 독립 추첨이 아니고, **한 시드가 네 에피소드를 낳는다.**
    ⛔손으로 적지 않는다: 동결 `results.jsonl` 에서 센다.
    """
    import re as _re
    out = {}
    for macro, rel, crowded_only in (
            ('HeadlineSeeds', 'W16_REBASELINE_2026-08-22/results.jsonl', True),
            ('AblationSeeds', 'B16_KGRID_2026-08-24', False)):
        path = os.path.join(TMP, rel)
        rows = []
        if os.path.isdir(path):
            for f in sorted(os.listdir(path)):
                if f.endswith('.jsonl'):
                    rows += [json.loads(x) for x in open(os.path.join(path, f))]
        else:
            rows = [json.loads(x) for x in open(path)]
        eps = {os.path.basename(r['episode']) for r in rows if r.get('episode')}
        if crowded_only:
            eps = {e for e in eps if _re.search(r'_(15|30)comp', e)}
        seeds = {int(m.group(1)) for e in eps
                 for m in [_re.search(r'seed(\d+)', e)] if m}
        out[macro] = str(len(seeds))
        out[macro.replace('Seeds', 'Eps')] = str(len(eps))
    return out


def _prior_arch():
    """★학습 사전의 **아키텍처를 체크포인트 가중치에서** 읽는다.

    5차 감사(2026-09-02) REJECT 경계 지적:

    > *"제목이 파는 학습 사전을 논문이 두 문장으로 정의한다. 아키텍처.학습 자료.학습
    >  절차가 없다. 심사자는 재현도 과적합 판단도 할 수 없다."*

    ⛔손으로 적으면 체크포인트를 갈 때 조용히 낡는다 --- 그래서 **가중치에서 센다**.
    ⚠학습 절차(스텝 수.혼합비)는 가중치에 없다: 그것은 배치 문서에서 오고,
      그 파일이 동결 후 움직였다는 사실을 논문이 **공시**한다.
    """
    import re as _re
    import torch
    from wavesplat.optimizer.learned_init import DEFAULT_CKPT
    sd = torch.load(DEFAULT_CKPT, map_location='cpu', weights_only=False)
    blocks = len({int(m.group(1)) for k in sd
                  for m in [_re.match(r'blocks\.(\d+)\.', k)] if m})
    return {
        'PriorParams': f"{sum(v.numel() for v in sd.values()):,}",
        'PriorDim': str(sd['enc.pos'].shape[-1]),
        'PriorTokens': str(sd['enc.pos'].shape[1]),
        'PriorBlocks': str(blocks),
    }


def _sci_seed(v):
    """시드 대역 끝점을 인쇄 가능한 수식으로. ⛔값이 다르면 죽는다."""
    if v == 2_000_000_000:
        return r'2\times10^{9}'
    if v == 2_147_483_647:
        return r'2^{31}-1'
    raise SystemExit(f'⛔시드 대역이 바뀌었다: {v} — 인쇄 형태를 다시 정하라')


def _training():
    """★학습 절차의 수치를 **동결 학습 산출물에서** 읽는다.

    6차 감사(2026-09-02) 지적:

    > *"train/val split is not addressed --- the word 'validation' does not appear
    >  in the paper. One exists and is committed: 256 held-out scenes, seed
    >  987654321, evaluated every 1000 steps, curve saved."*

    ⛔전부 커밋된 파일에 있다. 손으로 적지 않는다 --- `train_curve.json` 과
    `gen_mixed.py` 소스에서 읽는다.
    ⚠이 값들은 **아키텍처와 달리** 가중치에 없다. 그래서 동결 기록이 낡았다는
      §3.4.1 의 공시가 함께 가야 한다(그 공시는 코드가 아니라 *freeze record* 를 겨눈다).
    """
    import re as _re
    B = os.path.join(TMP, 'mixed_retrain_2026-07-26')
    cur = json.load(open(os.path.join(B, 'train_curve.json')))
    args, curve = cur['args'], cur['curve']
    src = open(os.path.join(B, 'gen_mixed.py')).read()
    lo, hi = _re.search(r'SEED_LO,\s*SEED_HI\s*=\s*([\d_]+),\s*([\d_]+)', src).groups()
    log = open(os.path.join(B, 'train.log'), errors='replace').read()
    donors = _re.search(r'geometry bank: (\d+) donor episodes', log).group(1)
    val_n, val_seed = _re.search(r'held-out val: (\d+) scenes \(seed (\d+)\)', log).groups()
    return {
        'TrainSteps': f"{int(args['steps']):,}",
        'TrainBatch': str(args['bsz']),
        'TrainScenes': f"{sum(cur['strata_seen'].values()):,}",
        'TrainValScenes': val_n,
        'TrainValSeed': val_seed,
        'TrainValEvery': f"{int(args['val_every']):,}",
        'TrainLossFinal': f"{curve[-1]['train']:.4f}",
        'TrainValLossFinal': f"{curve[-1]['val']:.4f}",
        # ★수학 모드에서 천단위 쉼표는 `[2, 000, 000, 000, 2, 147, ...)` 로 **깨져서**
        #   인쇄된다(구간 쉼표와 구별이 안 된다). ⟹ 지수 형태로 낸다. 값은 검증한다.
        'TrainSeedLo': _sci_seed(int(lo.replace('_', ''))),
        'TrainSeedHi': _sci_seed(int(hi.replace('_', ''))),
        'TrainDonors': donors,
    }


def _eval_seed_max():
    """★논문이 보고하는 **모든** 코퍼스의 최대 시드.

    ⛔*"학습 시드 대역이 평가와 겹치지 않는다"* 는 주장을 손으로 적지 않는다 ---
    동결 결과 행의 에피소드 이름에서 **센다**. 겹치면 이 값이 학습 대역에 닿는다.
    """
    import glob as _g
    import re as _re
    seeds = set()
    for f in _g.glob(os.path.join(TMP, '*', 'results*.jsonl')):
        for line in open(f, errors='replace'):
            m = _re.search(r'seed(\d+)', line)
            if m:
                seeds.add(int(m.group(1)))
    hi = max(seeds)
    assert hi < 2_000_000_000, f'⛔평가 시드가 학습 대역에 닿는다: {hi}'
    return f"{hi:,}"


def _ckpt_sha(n=12):
    """배포 체크포인트의 **코드에 핀된** SHA256.

    ⛔[2026-09-02] 6차 감사 BLOCKER: 논문이 *"the estimator **refuses to load** a
      checkpoint whose SHA-256 does not begin ..."* 라 적었는데 **거짓**이다.
      `learned_init._load_cond` 는 검사 없이 `torch.load` 한다 --- `ckpt_sha256()` 은
      패키지 어디서도 호출되지 않고, 대조하는 것은 **단위 시험 하나**뿐이다
      (`tests/test_deploy_auto_v1_m3.py`). 문장을 참으로 고쳤다.
    """
    from wavesplat.optimizer.learned_init import DEFAULT_CKPT_SHA256
    return DEFAULT_CKPT_SHA256[:n] if n else DEFAULT_CKPT_SHA256


def _mesh_grades():
    """★비선형(HOS) 코퍼스의 **실현된 메시 적정성**을 에피소드 HDF5 에서 센다.

    6차 감사(2026-09-02) 지적:

    > *"the batch's own gate file instructs that the sample must not be described
    >  uniformly as HOS M=3 ... the reader is never given a fraction."*

    ⛔*"HOS M=3 참값"* 은 **생성기에 요청한 것**이고, 실현된 것은 셀마다 다르다
      (`config.mesh_verdict`). 그 분포를 손으로 적지 않고 **자료에서 센다**.
    """
    import collections
    import glob as _g
    import h5py
    per = {}
    for f in sorted(_g.glob(os.path.join(TMP, 'HOSGATE_2026-08-13', 'data_hos', '*.h5'))):
        with h5py.File(f, 'r') as h:
            v = h['config'].attrs.get('mesh_verdict')
        per[os.path.basename(f)[len('episode_'):-3]] = (
            v.decode() if isinstance(v, bytes) else str(v))

    def grade(pattern):
        eps = set()
        for f in _g.glob(os.path.join(TMP, pattern)):
            for line in open(f):
                e = json.loads(line).get('episode')
                if e:
                    eps.add(os.path.basename(e).replace('episode_', '').replace('.h5', ''))
        c = collections.Counter(per.get(e, '?') for e in eps)
        assert c.get('?', 0) == 0, f'⛔HOS 코퍼스 밖 에피소드가 있다: {pattern}'
        return c

    cr = grade('B12_HOS_EXTEND_2026-08-23/results_b12.jsonl')
    sp = grade('B19_KGRID_SPARSE_2026-08-25/results_b19_*.jsonl')
    return {
        'HosCrowdedAdequate': str(cr['ADEQUATE']),
        'HosCrowdedDegraded': str(cr['DEGRADED']),
        'HosCrowdedN': str(sum(cr.values())),
        'HosSparseAdequate': str(sp['ADEQUATE']),
        'HosSparseDegraded': str(sp['DEGRADED']),
        'HosSparseInadequate': str(sp['INADEQUATE']),
        'HosSparseN': str(sum(sp.values())),
    }


def _spec_controls():
    """`T7b` 의 **음성 대조 두 개**를 게이트 문자열에서 읽는다.

    ⛔[2026-09-02] 6차 감사: 논문이 이 둘(`21`, `two`)을 **손으로 적었다** ---
      *"a paper whose method is that numbers come from macros."* 그 지적이 옳다.
    ⚠이 배치는 두 값을 `contrasts` 가 아니라 **게이트 사유 문자열**에 넣었다.
      동결 파일은 고치지 않는다 ⟹ 여기서 파싱하고, 형태가 바뀌면 죽는다.
    """
    import re as _re
    v = load('T7b_SPECTRAL_2026-09-02/verdict_t7b.json')
    g = {row[0].split()[0]: row[2] for row in v['gates']}
    ana = _re.match(r'(\d+)/(\d+)', g['G-5c'])
    learn = _re.match(r'(\d+)/(\d+)', g['G-5b'])
    assert ana and learn, f'⛔게이트 문자열 형태가 바뀌었다: {g}'
    n = int(ana.group(2))
    assert int(learn.group(2)) == n
    return {'SpecCmpBetter': ana.group(1),
            'SpecOwnBelowZero': str(n - int(learn.group(1)))}


def _est_seeds():
    """★**추정기 시드** 수를 동결 행에서 센다 --- 장면 시드와 다른 축이다.

    6차 감사(2026-09-02):

    > *"the number of estimator seeds is never stated anywhere, yet Table 6's
    >  caption depends on it ('median over estimator seeds')."*

    ⚠균일하지 않다: 대부분 3개(42/43/44 또는 1629--1631)인데 `S-ENS` 는 **하나**다
      (그 배치가 스스로 공시한다). ⟹ 두 값을 따로 낸다. ⛔뭉치지 마라.
    """
    import glob as _g
    counts = {}
    # ⚠[2026-09-02] 검증 감사 22: 표 행 셋(B-15·B-26·B-1 의 기준선 팔 S16)이 목록에서
    #   빠져 있었다 --- 매크로가 참이긴 했으나 **셋은 검사되지 않은 채** 인쇄됐다.
    for tag, pat in (('main', ('W16_REBASELINE_2026-08-22', 'S16_BASELINE_2026-08-14',
                               'B12_HOS_EXTEND_2026-08-23', 'B15_RAYDROP_2026-08-24',
                               'B16_KGRID_2026-08-24', 'B19_KGRID_SPARSE_2026-08-25',
                               'B22_PRIOR_ABLATION_2026-08-27', 'B26_SEASTATE_2026-08-27',
                               'M10_GEOMETRY_2026-08-30', 'T4_GEOMETRY_2026-08-30')),
                     ('sens', ('S_ENS_2026-09-01',))):
        per = {}
        for d in pat:
            seeds = set()
            for f in _g.glob(os.path.join(TMP, d, '*.jsonl')):
                if 'result' not in os.path.basename(f):
                    continue
                for line in open(f):
                    r = json.loads(line)
                    # ⚠열 이름이 배치마다 다르다: `est_seed` 또는 `seed`.
                    #   ⛔`gen_seed`(장면 시드)는 **절대** 세지 마라.
                    v = r.get('est_seed', r.get('filter_seed', r.get('seed')))
                    if v is not None:
                        seeds.add(v)
            assert seeds, f'⛔추정기 시드 열이 없다: {d}'
            per[d] = len(seeds)
        assert len(set(per.values())) == 1, f'⛔{tag} 배치들의 추정기 시드 수가 갈린다: {per}'
        counts[tag] = next(iter(per.values()))
    return {'EstSeeds': str(counts['main']), 'EstSeedsSEns': str(counts['sens'])}


def _frame_span():
    """`T6` 8칸의 **프레임 중앙값 최소/최대**. ⛔손으로 적으면 낡는다.

    ⛔[2026-09-02] 6차 감사: 논문이 *"varied only between 5.9 and 6.3 ms"* 와
      *"a factor of 1.06"* 를 손으로 적었다. 매크로 규약의 예외를 둘 이유가 없다.
    """
    v = load('T6_BUDGET_2026-09-02/verdict_t6.json')
    ms = [c['frame_ms_median'] for c in v['cells'].values()]
    return {'FrameMsMin': f'{min(ms):.2f}', 'FrameMsMax': f'{max(ms):.2f}',
            'FrameMsSpread': f'{max(ms) / min(ms):.2f}'}


def _breakable(h, every=8):
    """긴 16진수를 조판에서 끊을 수 있게 만든다(문자는 하나도 바뀌지 않는다)."""
    return r'\allowbreak '.join(h[i:i + every] for i in range(0, len(h), every))


def _qualitative():
    """정성 그림의 캡션 수치를 **그 그림의 출처 json 에서** 읽는다.

    ⛔[2026-09-02] 6차 감사 N6: 캡션이 *"a ten-times finer colour scale"* 이라 적었는데
      그림은 생성기가 계산한 실제 비(`x9.2`)를 인쇄했다 --- **같은 float 안에서
      손으로 적은 수가 생성된 수와 모순**됐다.
    """
    p = os.path.join(ROOT, 'paper', 'figures', 'fig_qualitative.provenance.json')
    d = json.load(open(p))
    u, sp = d['unmatched_true'], d['spurious_est']
    return {'QualScaleRatio': f"{d['colour_scale_ratio']:g}",
            'QualFrameRmse': f"{d['rmse_this_frame']:.3f}",
            'QualNObs': str(d['n_obs_this_frame']),
            'QualMissKx': f"{u['kx']:.2f}", 'QualMissKy': f"{u['ky']:.2f}",
            'QualSpurKx': f"{sp['kx']:.2f}", 'QualSpurKy': f"{sp['ky']:.2f}"}


def _b12_seeds():
    """`B-12` 의 **장면 시드**와 그중 이 프로젝트의 앞선 배치에 없던 것의 수.

    ⛔[2026-09-02] 6차 감사 N19: 논문이 *"scene seeds drawn from outside the training
      corpus"* 를 **B-12 를 구별하는 성질**로 팔았다. 그것은 **모든 배치에 참**이다
      (학습 대역은 2e9 이상, 평가 최대 시드는 `EvalSeedMax`) ⟹ 구별하지 못한다.
    ★실제로 B-12 가 더하는 것은 **이 저장소의 앞선 배치가 쓰지 않은 시드**다.
      그 수를 여기서 센다(`HOSGATE` 와의 차집합).
    """
    import glob as _g
    import re as _re

    def seeds_of(pat, key='gen_seed'):
        out = set()
        for f in _g.glob(os.path.join(TMP, pat)):
            for line in open(f):
                r = json.loads(line)
                v = r.get(key)
                if v is None:
                    m = _re.search(r'seed(\d+)', str(r.get('episode', '')))
                    v = int(m.group(1)) if m else None
                if v is not None:
                    out.add(int(v))
        return out

    b12 = seeds_of('B12_HOS_EXTEND_2026-08-23/results_b12.jsonl')
    prior = set()
    for f in _g.glob(os.path.join(TMP, 'HOSGATE_2026-08-13', '*.jsonl')):
        for line in open(f):
            m = _re.search(r'seed(\d+)', line)
            if m:
                prior.add(int(m.group(1)))
    fresh = b12 - prior
    assert b12 and fresh, '⛔B-12 시드를 세지 못했다'
    return {'BTwelveSeeds': str(len(b12)), 'BTwelveFreshSeeds': str(len(fresh))}


def _donor_disjoint():
    """★논문의 *"어느 평가 코퍼스에도 도너 에피소드가 없다"* 를 **기계로** 확인한다.

    ⛔[2026-09-02] 이 문장을 쓰기 전에 `tmp/*/results*.jsonl` 만 훑고 참이라 적을
      뻔했다. 넓혀서 훑으니 도너 14개가 **`results/100ep_ws_letkf.json` 등 옛 파일에
      전부 나온다**(그 코퍼스는 철회됐고 논문이 보고하지 않는다).
    ⟹ 주장의 범위는 **논문이 인용하는 배치**여야 하고, 그 범위를 여기서 강제한다.
      범위를 넓힌 주장을 쓰면 이 함수가 죽는다.
    """
    import glob as _g
    donors = ([f'BF{bf}_{nc}comp_seed0_uniform' for bf in (6, 7) for nc in (1, 5, 15, 30)]
              + [f'BF{bf}_5comp_seed{sd}_slope' for bf in (3, 4) for sd in (2, 3, 4)])
    dirs = {a[1].split('/')[0] for a in ATOMS}
    for d in sorted(dirs):                       # B-1 처럼 행이 다른 배치에 있는 경우
        vp = _g.glob(os.path.join(TMP, d, 'verdict*.json'))
        for f in vp:
            for src in (json.load(open(f)).get('source_results') or []):
                dirs.add(src.split('/')[1])
    files = [f for d in dirs
             for f in _g.glob(os.path.join(TMP, d, '*.jsonl'))]
    assert files, '⛔인용 배치의 결과 행을 찾지 못했다'
    bad = {}
    for f in files:
        t = open(f, errors='replace').read()
        for dn in donors:
            if dn in t:
                bad.setdefault(dn, set()).add(os.path.basename(os.path.dirname(f)))
    assert not bad, f'⛔학습 도너가 인용 배치의 평가 코퍼스에 있다: {bad}'
    return {'CitedBatchFiles': str(len(files))}


def _corpus_identity():
    """★표 5 각 행이 **어느 코퍼스를 쓰는지**를 동결 행에서 판정한다.

    ⛔[2026-09-02] 검증 감사 BLOCKER: 표 주 (i) 가 *"세 행(B-12·B-16·B-19)이 HOS 코퍼스를
      공유하고 나머지는 선형 참값"* 이라 적었다. **틀렸다** --- `B-22`·`M-10`·`T-4` 도
      같은 `HOSGATE_2026-08-13/data_hos` 를 읽고, 게다가 `B-12`·`B-16`·`B-22`·`M-10`·`T-4`
      다섯 행은 **완전히 같은 48 에피소드**다(∩ = 48/48). `B-19` 만 같은 디렉터리의
      **서로소인 희소 16개**다.
    ★그래서 이 함수는 자구를 고치는 것이 아니라 **구조를 센다**: 손으로 적으면 다음 배치에서
      또 틀린다. 논문은 이 매크로들을 쓴다.
    """
    import glob as _g
    import h5py
    hos = {os.path.basename(f)[len('episode_'):-3]
           for f in _g.glob(os.path.join(TMP, 'HOSGATE_2026-08-13', 'data_hos', '*.h5'))}
    assert hos, '⛔HOS 코퍼스를 찾지 못했다'

    def eps(pat):
        out = set()
        for f in _g.glob(os.path.join(TMP, pat)):
            for line in open(f):
                e = json.loads(line).get('episode')
                if e:
                    out.add(os.path.basename(str(e)).replace('episode_', '').replace('.h5', ''))
        return out

    ROWS = [('B-1', 'W16_REBASELINE_2026-08-22/results*.jsonl'),
            ('B-12', 'B12_HOS_EXTEND_2026-08-23/results*.jsonl'),
            ('B-15', 'B15_RAYDROP_2026-08-24/results*.jsonl'),
            ('B-16', 'B16_KGRID_2026-08-24/results*.jsonl'),
            ('B-19', 'B19_KGRID_SPARSE_2026-08-25/results*.jsonl'),
            ('B-22', 'B22_PRIOR_ABLATION_2026-08-27/results*.jsonl'),
            ('B-26', 'B26_SEASTATE_2026-08-27/results*.jsonl'),
            ('M-10', 'M10_GEOMETRY_2026-08-30/results*.jsonl'),
            ('T-4', 'T4_GEOMETRY_2026-08-30/results*.jsonl'),
            ('S-ENS', 'S_ENS_2026-09-01/results*.jsonl')]
    sets = {n: eps(p) for n, p in ROWS}
    missing = [n for n, e in sets.items() if not e]
    assert not missing, f'⛔결과 행을 찾지 못한 배치: {missing}'
    in_hos = [n for n, e in sets.items() if e <= hos]
    lin = [n for n, e in sets.items() if not (e & hos)]
    assert set(in_hos) | set(lin) == set(sets), '⛔HOS 를 부분적으로만 쓰는 배치가 있다'
    ref = sets['B-12']
    same = sorted(n for n in in_hos if sets[n] == ref)
    disj = sorted(n for n in in_hos if sets[n] != ref and not (sets[n] & ref))
    assert len(same) + len(disj) == len(in_hos), '⛔부분 겹침 배치가 있다'
    return {
        'HosRows': ', '.join(sorted(in_hos, key=lambda x: in_hos.index(x))),
        'HosSameRows': ', '.join(same),
        'HosSameN': str(len(same)),
        'HosDisjointRows': ', '.join(disj),
        'LinearRows': ', '.join(lin),
        'LinearN': str(len(lin)),
    }


def _w15_reagg():
    """★§5.6 의 *"희소는 배포 해석적 패키지 상대로 분리되지 않는다"* 를 **재집계로 낸다**.

    ⛔[2026-09-02] 검증 감사 MAJOR: 그 주장이 논문에 **배치 이름도 수치도 없이** 있었고
      *"we cite it as such"* 라 적었는데 **아무것도 인용하지 않았다**. 게다가 Data
      availability 는 *"Every result in Section 5 carries a batch identifier"* 라 적는다.
    ★출처는 `W15_DEPLOY_VERIFY_2026-08-22` 의 `AMENDMENT_1` 팔(진짜 배포 해석적,
      앙상블 k=3) 대 학습 팔이다. ⚠**그 수는 동결 `verdict_w15.json` 에 없다**
      (그 파일의 `sparse` 는 PRE/POST 대조 14/64 로 **다른 양**이다) ⟹ 여기서 동결 행을
      다시 집계하고, 논문은 그것을 **재집계로 명시**해 인용한다.
    """
    import statistics as _st
    rows = [json.loads(x) for x in
            open(os.path.join(TMP, 'W15_DEPLOY_VERIFY_2026-08-22', 'results.jsonl'))]
    by = {}
    for r in rows:
        if r.get('avg_nRMSE') is None:
            continue
        by.setdefault((r['cell'], r['stratum'], r['episode']), []).append(r['avg_nRMSE'])
    med = {k: _st.median(v) for k, v in by.items()}
    # ★음성 대조 — 비교자(해석적 패키지)가 **제로필드를 이기는가**.
    #   ⛔이것 없이 "분리되지 않는다" 를 인용하면 `F1` 부류가 된다(비교자가 무너져서
    #   생긴 널인지 구별할 수 없다). 제로 기준은 `W-15` 가 쓴 것과 **같은 출처**다.
    zero = {}
    for line in open(os.path.join(TMP, 'B_DEPLOY_2026-08-19', 'results.jsonl')):
        r = json.loads(line)
        if 'error' not in r and r.get('zero_nRMSE') is not None:
            zero[r['episode']] = r['zero_nRMSE']
    assert zero, '⛔제로필드 기준을 찾지 못했다'
    out = {}
    for stratum, tag in (('crowded', 'Crowded'), ('sparse', 'Sparse')):
        eps = sorted({k[2] for k in med if k[1] == stratum and k[0] == 'POST'})
        pairs = [(med[('POST', stratum, e)], med[('ALL-ANALYTIC', stratum, e)])
                 for e in eps if ('ALL-ANALYTIC', stratum, e) in med]
        assert pairs, f'⛔W-15 재집계 실패: {stratum}'
        out[f'WFifteen{tag}Better'] = str(sum(1 for a, b in pairs if a < b))
        out[f'WFifteen{tag}N'] = str(len(pairs))
        out[f'WFifteen{tag}Rel'] = f"{_st.median([(a - b) / b * 100 for a, b in pairs]):+.1f}"
        ks = [e for e in eps if e in zero]
        out[f'WFifteen{tag}CmpZero'] = str(
            sum(1 for e in ks if med[('ALL-ANALYTIC', stratum, e)] < zero[e]))
    return out


def _b1_control():
    """★`B-1` 비교자의 **제로필드 기술** — 논문이 *"측정되지 않았다"* 고 적던 것.

    ⛔[2026-09-02] 7차 심사 BLOCKER: §5.1 이 *"The comparator's own skill against a zero
      field **was not measured on this corpus**, so the dash ... is a **real absence**"*
      라 적었다. **거짓이다.** `verdict_b1.json.source_results` 가 가리키는
      `S16_BASELINE_2026-08-14/results.jsonl` 의 `SPEC25` 행 **전건**에 에피소드별
      `zero_nRMSE` 가 들어 있다(1,920/1,920).
    ★그리고 재집계하면 **64/64** --- 논문이 자기에게 **불리한 거짓**을 적고 있었다.
    ⚠재집계임을 공시해야 한다(선례 `_w15_reagg`). 사전등록된 대조가 아니다.
    """
    import statistics as _st
    v = load('B1_BASELINE_2026-08-22/verdict_b1.json')
    src = [x for x in v['source_results'] if 'S16_BASELINE' in x]
    assert len(src) == 1, f'⛔B-1 의 기준선 출처가 하나가 아니다: {v["source_results"]}'
    arm = v['baseline_arm']
    by = {}
    for line in open(os.path.join(ROOT, src[0])):
        r = json.loads(line)
        if r.get('arm') != arm or r.get('stratum') != 'crowded':
            continue
        if r.get('avg_nRMSE') is None or r.get('zero_nRMSE') is None:
            continue
        by.setdefault(r['episode'], []).append((r['avg_nRMSE'], r['zero_nRMSE']))
    assert by, '⛔B-1 기준선 행을 찾지 못했다'
    beats = sum(1 for e in by
                if _st.median(a for a, _ in by[e]) < _st.median(z for _, z in by[e]))
    assert len(by) == v['crowded']['n'], (
        f'⛔에피소드 수가 판정과 다르다: {len(by)} vs {v["crowded"]["n"]}')
    # ★[2026-09-02] 7차 심사 A2: `B-1` 의 두 팔은 **여드레 떨어진 두 배치**에서 왔고
    #   `SHIP` 행에는 프레임 수 열이 **없다** ⟹ 프레임 정합을 행에서 확인할 수 없다.
    #   ★그런데 확인할 방법이 있다: 채점 프레임은 `run.py:263` 의
    #   `get_tracking_frames(tp_factor=2.0)` 이 **결정론적으로** 고른다. 기준선 행이
    #   기록한 `n_match` 를 그 함수로 다시 계산해 대조하면 정합이 **검증된다**
    #   (2026-09-02 실측 64/64). ⛔에피소드 HDF5 를 여는 계산이라 여기서는 핀만 둔다.
    return {'BOneCmp': str(beats), 'BOneCmpN': str(len(by)),
            'BOneFrameMatch': str(len(by))}


def _seed_clustered():
    """★**시드 수준** 재집계 — 에피소드가 아니라 **장면 시드**를 단위로 부호검정.

    ⛔[2026-09-02] 7차 심사 B14: 논문이 스스로 *"the effective sample size lies somewhere
      between 12 and 48, and **every p we print is correspondingly optimistic**"* 라
      적으면서 표는 $p = 1.13\\times10^{-16}$ 을 인쇄한다. 심사자 축자:
      *"A referee cannot let a table assert what the surrounding text says is wrong."*
    ★그래서 **가장 보수적인 규약**으로 다시 집계한다: 한 시드의 네 에피소드를 중앙값으로
      접고, 시드를 시행 단위로 부호검정. 이 규약의 p 바닥은 $2^{-n}$ 이다.
    ⚠**사전등록된 대조가 아니다** --- 강건성 재집계다. 논문이 그렇게 밝힌다.
    """
    import glob as _g
    import re as _re
    import statistics as _st
    from math import comb as _comb

    def _p(w, n):
        return sum(_comb(n, k) for k in range(w, n + 1)) / 2 ** n

    def med_by(pat, arm, stratum='crowded'):
        by = {}
        for f in _g.glob(os.path.join(TMP, pat)):
            for line in open(f):
                r = json.loads(line)
                if (r.get('arm') or r.get('cell')) != arm:
                    continue
                if stratum and r.get('stratum') not in (None, stratum):
                    continue
                v = r.get('avg_nRMSE')
                if v is not None:
                    by.setdefault(r['episode'], []).append(v)
        return {k: _st.median(v) for k, v in by.items()}

    def fold(A, B):
        ks = sorted(set(A) & set(B))
        assert ks, '⛔짝이 없다'
        seeds = {}
        for k in ks:
            sd = int(_re.search(r'seed(\d+)', k).group(1))
            seeds.setdefault(sd, []).append((A[k], B[k]))
        w = sum(1 for v in seeds.values()
                if _st.median(a for a, _ in v) < _st.median(b for _, b in v))
        return w, len(seeds)

    out = {}
    for tag, (pa, aa, pb, ab) in {
        'BOne': ('W16_REBASELINE_2026-08-22/results*.jsonl', 'SHIP',
                 'S16_BASELINE_2026-08-14/results*.jsonl', 'SPEC25'),
        'BTwelve': ('B12_HOS_EXTEND_2026-08-23/results*.jsonl', 'SHIP-NOW',
                    'B12_HOS_EXTEND_2026-08-23/results*.jsonl', 'SPEC25'),
        'BSixteen': ('B16_KGRID_2026-08-24/results*.jsonl', 'SHIP-NOW',
                     'B16_KGRID_2026-08-24/results*.jsonl', 'KGRID1'),
        'BTwentyTwo': ('B22_PRIOR_ABLATION_2026-08-27/results*.jsonl', 'SHIP',
                       'B22_PRIOR_ABLATION_2026-08-27/results*.jsonl', 'ANA-SHIP'),
    }.items():
        w, n = fold(med_by(pa, aa), med_by(pb, ab))
        out[f'{tag}SeedW'] = str(w)
        out[f'{tag}SeedN'] = str(n)
        out[f'{tag}SeedP'] = _sci(_p(w, n))
    return out


def _prior_ckpt_sha(n=12):
    """★두 번째 체크포인트(`setdiff_prior.pt`)의 SHA256 앞 12자리.

    ⛔[2026-09-02] 6차 감사 MAJOR: 논문이 정규화 상수를 *"a constant stored in **the**
      checkpoint"* 라 적었다 --- 배포 학습 경로는 체크포인트를 **둘** 읽는다
      (`learned_init._load_cond` 가 `cd.PRIOR_CKPT` 에서 `stats` 를 가져온다).
      논문은 하나만 이름 짓고 핀했다.
    """
    import hashlib
    ck = os.path.join(ROOT, 'results', 'direction3', 'setdiff_prior.pt')
    h = hashlib.sha256()
    with open(ck, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()[:n] if n else h.hexdigest()


def _k_lim():
    """학습 사전의 파수 지지 상한 --- `setdiff_prior.pt` 의 `stats['K_LIM']`.

    샘플러가 정규화 좌표를 `clamp(-1,1)` 하므로(`d3_beamform_cond_diffusion.py`)
    `|kx|,|ky| <= K_LIM` 이 **구조적** 상한이다.  ⛔손으로 적지 마라 --- 체크포인트를
    갈면 조용히 낡는다.
    """
    import torch
    ck = os.path.join(ROOT, 'results', 'direction3', 'setdiff_prior.pt')
    if not os.path.exists(ck):
        raise SystemExit(f'⛔체크포인트가 없다: {ck} — 지지 상한을 셀 수 없다')
    d = torch.load(ck, map_location='cpu', weights_only=False)
    st = d.get('stats', d)
    return f"{float(st['K_LIM']):g}"


def _ana_kmax():
    """해석적 빔포밍의 탐색 상한 --- 함수 서명 기본값에서 읽는다."""
    import inspect
    from wavesplat.optimizer import initializer as _init
    hi = inspect.signature(_init.initialize_from_frames_v2).parameters['kx_range'].default[1]
    return f"{float(hi):g}"


def load(rel):
    p = os.path.join(TMP, rel)
    if not os.path.exists(p):
        raise SystemExit(f'⛔동결 판정이 없다: {p}')
    return json.load(open(p))

def _e1_equal_budget():
    """★`E1` 의 **동일 예산** 정면대결 — verdict 이 저장하지 않는 값이라 여기서 센다.

    ⛔라벨 칸(400/프레임)은 저들이 관측을 **두 배** 받는다. 동일 예산 칸은 200/프레임이고,
    그 칸에서 저들은 **자기검정 미달**(7/12)이다. 두 사실을 같이 인쇄해야 한다.
    ⚠런 0회 — 동결 `verdict_e1.json` 의 `per_record` 재집계.
    """
    v = load('E1_EXTBASE_2026-09-03/verdict_e1.json')
    pr = v['per_record']
    w = sum(1 for r in pr.values() if r['ship'] < r['cnn_200'])
    return {'EOneEqualShipWin': str(w), 'EOneEqualN': str(len(pr))}



# ─────────────────────────────────────────────────────────────────────────────
# ★인용 원자 — (매크로, 배치 verdict, json 경로, 형식, 핀)
#   ⛔핀은 "오늘 값" 이다. 값이 바뀌면 **의도적으로** 갱신하라(래칫과 같은 규약).
#   ⚠`kind`: 'int' | 'pct'(부호 포함 소수1) | 'p'(과학표기) | 'ratio'(a/b)
# ─────────────────────────────────────────────────────────────────────────────
ATOMS = [
    # ── 헤드라인 (혼잡 층, 최강 적격 격자 기준선 대비)
    ('BOneBetter',  'B1_BASELINE_2026-08-22/verdict_b1.json', 'crowded.better', 'int', 62),
    ('BOneN',       'B1_BASELINE_2026-08-22/verdict_b1.json', 'crowded.n', 'int', 64),
    ('BOneRel',     'B1_BASELINE_2026-08-22/verdict_b1.json', 'crowded.rel_median_pct', 'pct', -38.6),
    ('BOneP',       'B1_BASELINE_2026-08-22/verdict_b1.json', 'crowded.p_one_sided', 'p', 1.13e-16),
    # ── 비선형 참값 · 신선 시드 확장 (확증)
    ('BTwelveBetter', 'B12_HOS_EXTEND_2026-08-23/verdict_b12.json', 'ship_vs_spec.better', 'int', 43),
    ('BTwelveN',      'B12_HOS_EXTEND_2026-08-23/verdict_b12.json', 'ship_vs_spec.n', 'int', 48),
    ('BTwelveRel',    'B12_HOS_EXTEND_2026-08-23/verdict_b12.json', 'ship_vs_spec.rel_median_pct', 'pct', -40.2),
    ('BTwelveP',      'B12_HOS_EXTEND_2026-08-23/verdict_b12.json', 'ship_vs_spec.p_one_sided', 'p', 6.84e-09),
    # ★음성대조 — 비교자에게도 기술이 있는가 (F1 교훈)
    ('BTwelveCmpBetter', 'B12_HOS_EXTEND_2026-08-23/verdict_b12.json', 'spec_vs_zero.better', 'int', 47),
    ('BTwelveCmpN',      'B12_HOS_EXTEND_2026-08-23/verdict_b12.json', 'spec_vs_zero.n', 'int', 48),
    # ── 표현 절제: 파수 연속성 (혼잡)
    ('BSixteenBetter', 'B16_KGRID_2026-08-24/verdict_b16.json', 'primary.better', 'int', 46),
    ('BSixteenN',      'B16_KGRID_2026-08-24/verdict_b16.json', 'primary.n', 'int', 48),
    ('BSixteenRel',    'B16_KGRID_2026-08-24/verdict_b16.json', 'primary.rel_median_pct', 'pct', -38.6),
    ('BSixteenP',      'B16_KGRID_2026-08-24/verdict_b16.json', 'primary.p_one_sided', 'p', 4.18e-12),
    ('BSixteenDose',   'B16_KGRID_2026-08-24/verdict_b16.json', 'dose.better', 'int', 47),
    # ── 표현 절제: 파수 연속성 (희소)
    ('BNineteenBetter', 'B19_KGRID_SPARSE_2026-08-25/verdict_b19.json', 'primary.better', 'int', 14),
    ('BNineteenN',      'B19_KGRID_SPARSE_2026-08-25/verdict_b19.json', 'primary.n', 'int', 16),
    ('BNineteenRel',    'B19_KGRID_SPARSE_2026-08-25/verdict_b19.json', 'primary.rel_median_pct', 'pct', -67.4),
    ('BNineteenP',      'B19_KGRID_SPARSE_2026-08-25/verdict_b19.json', 'primary.p_one_sided', 'p', 2.09e-03),
    # ── 사전 절제: 학습 패키지 < 해석적 패키지 (L2) ★%를 margin 으로 쓸 수 있는 성분
    ('BTwentyTwoBetter', 'B22_PRIOR_ABLATION_2026-08-27/verdict_b22.json', 'comparisons.L2.better', 'int', 45),
    ('BTwentyTwoN',      'B22_PRIOR_ABLATION_2026-08-27/verdict_b22.json', 'comparisons.L2.n', 'int', 48),
    ('BTwentyTwoRel',    'B22_PRIOR_ABLATION_2026-08-27/verdict_b22.json', 'comparisons.L2.rel_median_pct', 'pct', -36.3),
    ('BTwentyTwoP',      'B22_PRIOR_ABLATION_2026-08-27/verdict_b22.json', 'comparisons.L2.p_one_sided', 'p', 6.56e-11),
    # ⛔⛔[2026-09-02] **정정(BLOCKER)**: 종전 주석은 *"L2 의 우변은 참 제로필드를 못 넘는다"*
    #   라 적었고 논문 7곳이 그 전제로 `BTwentyTwoAnaZero`(=**ANA** 의 15/48)를 병기했다.
    #   ⛔틀렸다. `L2` 의 우변은 **`ANA-SHIP`** 이고(`RESULT.md:16,92,116`) 그 팔의
    #   제로필드 대비는 **32/48 · p=0.0147** 이다. 15/48 은 **`L1a`** 의 우변(`ANA`)이다.
    #   ⟹ 두 원자를 **각각 명명**한다. L2 를 인용하는 자리에는 `…AnaShipZero` 를 쓴다.
    #   ★결과적으로 논문은 **자기에게 불리한 거짓**을 인쇄하고 있었다 --- 그 병기가
    #     참이었다면 §5.1 의 자기 원칙에 의해 헤드라인 절제가 인용 불가가 된다.
    ('BTwentyTwoAnaShipZero', 'B22_PRIOR_ABLATION_2026-08-27/verdict_b22.json',
     'neg_vs_zero.ANA-SHIP.better', 'int', 32),
    ('BTwentyTwoAnaShipZeroP', 'B22_PRIOR_ABLATION_2026-08-27/verdict_b22.json',
     'neg_vs_zero.ANA-SHIP.p_one_sided', 'p', 0.0147),
    # ⚠아래는 **`L1a`** 의 우변(`ANA`)이다. ⛔`L2` 옆에 쓰지 마라.
    ('BTwentyTwoAnaZero', 'B22_PRIOR_ABLATION_2026-08-27/verdict_b22.json', 'neg_vs_zero.ANA.better', 'int', 15),
    ('BTwentyTwoLearnZero', 'B22_PRIOR_ABLATION_2026-08-27/verdict_b22.json', 'neg_vs_zero.LEARN.better', 'int', 48),
    # ── 센서 모형(raydrop) 강건성
    # ── ★[2026-09-02] 유일하게 살아있는 시간 계수(4차 감사 B9).
    #   ⛔프로세스간 지연은 전부 철회됐지만, **질의 렌더 단가**는 다른 양이고 측정돼 있다.
    #   ⚠병기 의무 2건: **격자 기반 실측 경로**의 계측이고(학습 렌더러 미측정),
    #   **초기화 이후 정상 상태**만 포괄한다(제안 단계 불포함).
    ('BudgetUsPerPoint', 'B6_BUDGET_2026-08-23/verdict_b6b.json',
     'pooled.per_point_s', 'us', 137.0),
    # ── ★[2026-09-02] `T6-BUDGET` --- **배포 학습 경로**의 프레임 예산.
    #   ⛔이 배치 전까지 학습 경로의 지연은 **0건 측정**이었고 논문이 그렇게 적고 있었다.
    #   결정표가 결과 전에 지정한 **가장 보수적인 칸** = 혼잡 · CPU 전용 · 1 스레드.
    ('FrameMsLearn', 'T6_BUDGET_2026-09-02/verdict_t6.json',
     ('cells', 'CPU|LEARN|crowded', 'frame_ms_median'), 'ms2', 5.91),
    ('FramePctLearn', 'T6_BUDGET_2026-09-02/verdict_t6.json',
     ('cells', 'CPU|LEARN|crowded', 'budget_frac_at_10hz'), 'pct100', 5.9),
    ('InitSLearn', 'T6_BUDGET_2026-09-02/verdict_t6.json',
     ('cells', 'CPU|LEARN|crowded', 'init_s_median'), 'ms2', 5.18),
    # ★[2026-09-02] `T8` 의 비용 문턱은 **희소** 칸이다(같은 층에서 비교해야 한다).
    ('InitSAnaSparse', 'T6_BUDGET_2026-09-02/verdict_t6.json',
     ('cells', 'CPU|ANA|sparse', 'init_s_median'), 'ms2', 17.08),
    ('InitSAna', 'T6_BUDGET_2026-09-02/verdict_t6.json',
     ('cells', 'CPU|ANA|crowded', 'init_s_median'), 'ms2', 26.61),
    ('InitSLearnGpu', 'T6_BUDGET_2026-09-02/verdict_t6.json',
     ('cells', 'GPU|LEARN|crowded', 'init_s_median'), 'ms2', 4.86),
    ('BudgetFrameS', 'B6_BUDGET_2026-08-23/verdict_b6b.json', 'frame_budget_s', 'sec', 0.1),
    ('BFifteenN',   'B15_RAYDROP_2026-08-24/verdict_b15.json', 'by_raydrop.slope.n', 'int', 20),
    ('BFifteenRel', 'B15_RAYDROP_2026-08-24/verdict_b15.json', 'by_raydrop.slope.rel_median_pct', 'pct', -42.6),
    ('BFifteenBetter', 'B15_RAYDROP_2026-08-24/verdict_b15.json',
     'by_raydrop.slope.ship_beats_spec', 'int', 17),
    ('BFifteenP', 'B15_RAYDROP_2026-08-24/verdict_b15.json',
     'by_raydrop.slope.p_one_sided', 'p', 1.29e-03),
    # ★음성대조: 이 코너에서 **비교자에게도 기술이 있다**(F1 부류 방지)
    ('BFifteenCmp', 'B15_RAYDROP_2026-08-24/verdict_b15.json',
     'by_raydrop.slope.spec_beats_zero', 'int', 20),
    # ── 이득의 위치: 공간(M-10) ⚠rel_median 이 **분수**다(다른 배치와 단위가 다르다)
    ('MTenBetter', 'M10_GEOMETRY_2026-08-30/verdict_m10.json', 'primary.primary.better', 'int', 45),
    ('MTenN',      'M10_GEOMETRY_2026-08-30/verdict_m10.json', 'primary.primary.n', 'int', 48),
    ('MTenRel',    'M10_GEOMETRY_2026-08-30/verdict_m10.json', 'primary.primary.rel_median', 'frac', -31.7),
    ('MTenCmp',    'M10_GEOMETRY_2026-08-30/verdict_m10.json', 'primary.spec_vs_zero.better', 'int', 48),
    ('MTenP',      'M10_GEOMETRY_2026-08-30/verdict_m10.json', 'primary.primary.p_one_sided', 'p', 6.56e-11),
    # ★[2026-09-02] **절대** nRMSE. 4차 감사: 논문 52쪽에 우리 팔의 절대 정확도가 0회였다
    #   --- 인쇄된 유일한 절대값이 **철회된 기준선의 것**이었다.
    ('MTenAbsShip', 'M10_GEOMETRY_2026-08-30/verdict_m10.json', 'primary.abs_median.SHIP', 'abs3', 0.043),
    ('MTenAbsSpec', 'M10_GEOMETRY_2026-08-30/verdict_m10.json', 'primary.abs_median.SPEC25', 'abs3', 0.063),
    ('MTenAbsZero', 'M10_GEOMETRY_2026-08-30/verdict_m10.json', 'primary.abs_median.ZERO', 'abs3', 0.339),
    # ── 이득의 위치: 시간(T-4) ⚠비교자 붕괴를 **함께** 싣는다(F1 부류 방지)
    ('TFourBetter', 'T4_GEOMETRY_2026-08-30/verdict_t4.json', 'primary.better', 'int', 47),
    ('TFourN',      'T4_GEOMETRY_2026-08-30/verdict_t4.json', 'primary.n', 'int', 48),
    ('TFourRel',    'T4_GEOMETRY_2026-08-30/verdict_t4.json', 'primary.rel_median_pct', 'pct', -46.3),
    ('TFourCmp',    'T4_GEOMETRY_2026-08-30/verdict_t4.json', 'comparator.better', 'int', 34),
    ('TFourCmpAtZero', 'T4_GEOMETRY_2026-08-30/verdict_t4.json',
     ('ladder', '0.0', 'spec_vs_zero', 'better'), 'int', 47),

    # ═════════════════════════════════════════════════════════════════════════
    # ★[2026-09-02 P9] 레거시 표·그림 처분에 필요한 원자.
    #   사유: 논문의 절제·강건성 표 다섯 개와 그림 넷이 **철회된 코퍼스** 산물이라
    #   옮긴다. 그 자리에 들어가는 대체 표는 **전부 동결 verdict 에서만** 나와야 한다.
    #   ⛔손으로 옮겨 적으면 이 저장소가 반복해 대가를 치른 결함으로 되돌아간다.
    # ═════════════════════════════════════════════════════════════════════════

    # ── 해상상태 2x2 (B-26). ⚠F1: 승수 한 척도에서만 "최강" 이다 --- **절대 정확도는 반대**.
    #    ⟹ 절대 중앙값 두 개를 **함께** 낸다(코너 순위 주장을 문면에서 불가능하게 만든다).
    ('BTwentySixBetter', 'B26_SEASTATE_2026-08-27/verdict_b26.json',
     'corners.SS-REAL.ship_vs_spec.better', 'int', 24),
    ('BTwentySixN', 'B26_SEASTATE_2026-08-27/verdict_b26.json',
     'corners.SS-REAL.ship_vs_spec.n', 'int', 24),
    ('BTwentySixRel', 'B26_SEASTATE_2026-08-27/verdict_b26.json',
     'corners.SS-REAL.ship_vs_spec.rel_median_pct', 'pct', -42.4),
    # ★F1 병기 원자 --- 비교자의 기술이 이 코너에서 붕괴한다(-16.2% -> -4.0%)
    ('BTwentySixCmpRel', 'B26_SEASTATE_2026-08-27/verdict_b26.json',
     'corners.SS-REAL.spec_vs_zero.rel_median_pct', 'pct', -4.0),
    ('BTwentySixAnchorCmpRel', 'B26_SEASTATE_2026-08-27/verdict_b26.json',
     'corners.SS-BF4.spec_vs_zero.rel_median_pct', 'pct', -16.2),
    # ★F1 병기 원자 --- 절대 정확도는 **반대 방향**(0.1366 vs 0.1176)
    ('BTwentySixAbs', 'B26_SEASTATE_2026-08-27/verdict_b26.json',
     'corners.SS-REAL.medians.ship', 'abs3', 0.137),
    ('BTwentySixAnchorAbs', 'B26_SEASTATE_2026-08-27/verdict_b26.json',
     'corners.SS-BF4.medians.ship', 'abs3', 0.118),
    ('BTwentySixCmpBetter', 'B26_SEASTATE_2026-08-27/verdict_b26.json',
     'corners.SS-REAL.spec_vs_zero.better', 'int', 23),

    # ── 실측 음성 셋 (경계 표). ⛔전부 **참 제로필드 대비 승수**이지 팔간 비교가 아니다.
    ('BSeventeenBeats', 'B17_REALNOISE_2026-08-25/verdict_b17.json',
     'beats_zero.RANGEANG.wins', 'int', 4),
    ('BSeventeenN', 'B17_REALNOISE_2026-08-25/verdict_b17.json',
     'beats_zero.RANGEANG.n', 'int', 12),
    ('BEighteenBeats', 'B18_DENSITY_NOISE_2026-08-25/verdict_b18.json',
     'primary.D_BOTH.beats_zero', 'int', 8),
    ('BEighteenN', 'B18_DENSITY_NOISE_2026-08-25/verdict_b18.json',
     'primary.D_BOTH.n', 'int', 12),
    ('BEighteenConfirm', 'B18_DENSITY_NOISE_2026-08-25/verdict_b18.json',
     'thresholds.confirm', 'int', 11),
    ('BFourteenBBeats', 'B14_OBSMODEL_2026-08-24/verdict_b14b.json',
     'h1.IID.zero_better', 'int', 8),
    ('BFourteenBN', 'B14_OBSMODEL_2026-08-24/verdict_b14b.json',
     'h1.IID.n', 'int', 12),

    # ── 앙상블 대칭 (2026-09-01 두 배치)
    ('SEnsBetter', 'S_ENS_2026-09-01/verdict_sens.json', 'contrasts.PRIMARY.better', 'int', 55),
    ('SEnsN', 'S_ENS_2026-09-01/verdict_sens.json', 'contrasts.PRIMARY.n', 'int', 64),
    ('SEnsRel', 'S_ENS_2026-09-01/verdict_sens.json',
     'contrasts.PRIMARY.rel_median_pct', 'pct', -17.9),
    ('SEnsP', 'S_ENS_2026-09-01/verdict_sens.json',
     'contrasts.PRIMARY.p_one_sided', 'p', 1.77e-09),
    # ⚠희소는 `UNDECIDED` --- **혼잡만 인용하면 층 한정이 사라진다** ⟹ 함께 낸다.
    ('SEnsSparseBetter', 'S_ENS_2026-09-01/verdict_sens.json',
     'contrasts.SEC-A.better', 'int', 35),
    ('SEnsSparseN', 'S_ENS_2026-09-01/verdict_sens.json', 'contrasts.SEC-A.n', 'int', 64),
    ('SEnsNominal', 'S_ENS_2026-09-01/verdict_sens.json', 'thresholds.NOMINAL', 'int', 40),

    # ── ★[2026-09-02] `T7b-SPECTRAL` --- 스펙트럼 영역 지표(등급 EXPLORATORY 고정).
    #   ⚠`G-5c`(비교자>제로장 21/32)는 **동반 공시 의무**다: 우변이 얇다(F1 부류).
    ('SpecBetter', 'T7b_SPECTRAL_2026-09-02/verdict_t7b.json',
     'contrasts.A_learn_better', 'int', 25),
    ('SpecN', 'T7b_SPECTRAL_2026-09-02/verdict_t7b.json', 'contrasts.n', 'int', 32),
    ('SpecP', 'T7b_SPECTRAL_2026-09-02/verdict_t7b.json',
     'contrasts.A_p_one_sided', 'p', 1.05e-03),
    # ── ★[2026-09-02] `T8-ENSDEFAULT` --- 앙상블을 배포 기본값으로 켤 것인가.
    #   ⛔라벨은 `ENS-DEFAULT-UNDECIDED`(희소 비열등성 문턱 미달)이나 **결정을 만든 것은
    #   프레임 비용**이다: 앙상블은 매 프레임 k 멤버를 **순차로** 돌린다
    #   (`sequential_update.py:2128` · 축자 주석 *"additive by construction"*).
    ('TEightFrameOne', 'T8_ENSDEFAULT_2026-09-02/verdict_t8.json',
     ('frames', 'M1', 'frame_ms_median'), 'ms2', 6.00),
    ('TEightFrameThree', 'T8_ENSDEFAULT_2026-09-02/verdict_t8.json',
     ('frames', 'M3', 'frame_ms_median'), 'ms2', 18.30),
    ('TEightInitOne', 'T8_ENSDEFAULT_2026-09-02/verdict_t8.json',
     ('cost', 'M1', 'init_s_median'), 'ms2', 2.15),
    ('TEightInitThree', 'T8_ENSDEFAULT_2026-09-02/verdict_t8.json',
     ('cost', 'M3', 'init_s_median'), 'ms2', 10.49),
    ('TEightK', 'T8_ENSDEFAULT_2026-09-02/verdict_t8.json', 'primary.K', 'int', 53),
    ('TEightN', 'T8_ENSDEFAULT_2026-09-02/verdict_t8.json', 'primary.n', 'int', 64),
    ('TEightThresh', 'T8_ENSDEFAULT_2026-09-02/verdict_t8.json',
     'thresholds.K_MIN', 'int', 56),
    ('TEightRel', 'T8_ENSDEFAULT_2026-09-02/verdict_t8.json',
     'primary.rel_median_pct', 'pct', -8.1),
    ('SpecHs', 'T7b_SPECTRAL_2026-09-02/verdict_t7b.json',
     'contrasts.B_learn_better', 'int', 20),

    ('TFiveBetter', 'T5_ENS_2026-09-01/verdict_t5.json', 'contrasts.PRIMARY.better', 'int', 28),
    ('TFiveN', 'T5_ENS_2026-09-01/verdict_t5.json', 'contrasts.PRIMARY.n', 'int', 36),
    ('TFiveRel', 'T5_ENS_2026-09-01/verdict_t5.json',
     'contrasts.PRIMARY.rel_median_pct', 'pct', -8.5),
    ('TFiveP', 'T5_ENS_2026-09-01/verdict_t5.json',
     'contrasts.PRIMARY.p_one_sided', 'p', 5.97e-04),
    ('TFiveFarBetter', 'T5_ENS_2026-09-01/verdict_t5.json', 'contrasts.SEC-A.better', 'int', 35),
    ('TFiveFarRel', 'T5_ENS_2026-09-01/verdict_t5.json',
     'contrasts.SEC-A.rel_median_pct', 'pct', -11.1),
    # ── 채점 규약 수리 (R1b, 2026-09-02) — ⛔배포 기본값 변경 아님. 규약을 고쳐도
    #    판정이 사는가만 잰다. ⚠`Old` 계수를 `B-2` 공표값과 직접 비교 금지(수심 상이).
    ('ROneBShipFixed', 'R1B_SCORINGFRAME_2026-09-02/verdict_r1b.json', 'primary.w', 'int', 11),
    ('ROneBN', 'R1B_SCORINGFRAME_2026-09-02/verdict_r1b.json', 'primary.n', 'int', 12),
    ('ROneBP', 'R1B_SCORINGFRAME_2026-09-02/verdict_r1b.json',
     'primary.p_one_sided', 'p', 3.174e-03),
    ('ROneBShipOld', 'R1B_SCORINGFRAME_2026-09-02/verdict_r1b.json',
     'secondary.SHIP_old.w', 'int', 10),
    ('ROneBPropOld', 'R1B_SCORINGFRAME_2026-09-02/verdict_r1b.json',
     'secondary.PROP_old.w', 'int', 7),
    ('ROneBPropFixed', 'R1B_SCORINGFRAME_2026-09-02/verdict_r1b.json',
     'secondary.PROP_new.w', 'int', 8),
    # ★수리가 움직인 것은 **비교자**다 --- 중앙은 거의 0 이고 최대는 한 레코드가 만든다.
    ('ROneBZeroMed', 'R1B_SCORINGFRAME_2026-09-02/verdict_r1b.json',
     'zero_shift_summary.median_pct', 'pct', 0.3),
    ('ROneBZeroMax', 'R1B_SCORINGFRAME_2026-09-02/verdict_r1b.json',
     'zero_shift_summary.max_pct', 'pct', 49.0),
    # ── 외부 자료의 체제 판정 (R4b, 2026-09-02) — ⛔정확도 주장 아님(추정기 런 0회).
    ('RFourBCtrlPass', 'R4B_DISPERSION_2026-09-02/verdict_r4b.json',
     'groups.control.n_pass', 'int', 12),
    ('RFourBCtrlN', 'R4B_DISPERSION_2026-09-02/verdict_r4b.json', 'groups.control.n', 'int', 12),
    ('RFourBTreatPass', 'R4B_DISPERSION_2026-09-02/verdict_r4b.json',
     'groups.agulhas.n_pass', 'int', 0),
    ('RFourBTreatN', 'R4B_DISPERSION_2026-09-02/verdict_r4b.json', 'groups.agulhas.n', 'int', 6),
    ('RFourBCtrlMin', 'R4B_DISPERSION_2026-09-02/verdict_r4b.json',
     'groups.control.obs_min', 'abs3', 0.071),
    ('RFourBTreatMax', 'R4B_DISPERSION_2026-09-02/verdict_r4b.json',
     'groups.agulhas.obs_max', 'abs3', 0.011),
    # ── 재구성면 잡음 바닥 (R2b, 2026-09-02) — ⛔**상한**이다(게이지 자신의 잡음과
    #    공간 비균질이 섞인다) · ⛔격자점별 오차장 주장 금지(6점, 반경 0.25 m).
    ('RTwoBBetter', 'R2B_STEREONOISE_2026-09-02/verdict_r2b.json', 'primary.better', 'int', 7),
    ('RTwoBN', 'R2B_STEREONOISE_2026-09-02/verdict_r2b.json', 'primary.n', 'int', 7),
    ('RTwoBP', 'R2B_STEREONOISE_2026-09-02/verdict_r2b.json',
     'primary.p_one_sided', 'p', 7.8125e-03),
    ('RTwoBFloorMin', 'R2B_STEREONOISE_2026-09-02/verdict_r2b.json',
     'noise_floor.rms_mm.min', 'mm1', 12.8),
    ('RTwoBFloorMed', 'R2B_STEREONOISE_2026-09-02/verdict_r2b.json',
     'noise_floor.rms_mm.median', 'mm1', 56.9),
    ('RTwoBFloorMax', 'R2B_STEREONOISE_2026-09-02/verdict_r2b.json',
     'noise_floor.rms_mm.max', 'mm1', 129.0),
    # ── 계수 인공물 재검 (R4c, 2026-09-03, 런 0회) — 감사 BLOCKER B2 를 닫는다.
    #    ⛔`0/6` 은 여전히 **비초과**이지 부재가 아니다(R4b/AMENDMENT_1 §1).
    ('RFourCCtrlPass', 'R4C_DISPERSION_2026-09-03/verdict_r4c.json',
     'groups.control.n_pass', 'int', 12),
    ('RFourCTreatPass', 'R4C_DISPERSION_2026-09-03/verdict_r4c.json',
     'groups.agulhas.n_pass', 'int', 0),
    ('RFourCDropMed', 'R4C_DISPERSION_2026-09-03/verdict_r4c.json',
     'groups.agulhas.p95_drop_pct_median', 'pct', -0.6),
    # ── 측정된 센서 기하로 잡음 모형 재측정 (R5, 2026-09-03)
    #    ⛔`B-17` 판정 갱신 아님 · ⛔측정 기하는 **BS 7 뿐**(대형 박스 5 는 선언값 그대로).
    ('RFiveGap', 'R5_REALGEOM_2026-09-03/verdict_r5.json', 'primary.gap', 'abs3', 0.514),
    ('RFiveVsConst', 'R5_REALGEOM_2026-09-03/verdict_r5.json',
     'secondary.meas_vs_const.better', 'int', 8),
    ('RFiveN', 'R5_REALGEOM_2026-09-03/verdict_r5.json', 'secondary.meas_vs_const.n', 'int', 12),
    ('RFiveRelSmall', 'R5_REALGEOM_2026-09-03/verdict_r5.json',
     'secondary.meas_vs_anchor_small.rel_median_pct', 'pct', -45.4),
    # ── ★`R5b` — `R5` 의 **축 반사 결함**을 고친 후속(그 배치는 ⛔인용 불가).
    #    ⛔`RFive*` 원자는 **인쇄에 쓰지 않는다**(생성기에는 남겨 이력을 보존한다).
    ('RFiveBGap', 'R5B_REALGEOM_2026-09-03/verdict_r5b.json', 'primary.gap', 'abs3', 0.657),
    ('RFiveBVsConst', 'R5B_REALGEOM_2026-09-03/verdict_r5b.json',
     'primary.vs_const.better', 'int', 3),
    ('RFiveBN', 'R5B_REALGEOM_2026-09-03/verdict_r5b.json', 'primary.vs_const.n', 'int', 7),
    ('RFiveBMMGap', 'R5B_REALGEOM_2026-09-03/verdict_r5b.json',
     'arms.RANGEANG-MEASFIX-MM.gap', 'abs3', 0.514),
    # ── ★[2026-09-03 절제 사슬 감사 BLOCKER] `tab:earns` 의 `cmp.` 열이 B-16·B-19 행에서
    #    비어 있었고, 표 주가 그 공백을 *"설계에 없었다"* 로 **거짓 설명**했다.
    #    · B-19 는 **자기 배치의 명명된 게이트**(`N-2 음성대조 KGRID1 < ZERO`)다.
    #    · B-16 은 2026-08-27 재집계로 측정됐다(⚠사전등록 대조가 **아니다** --- 병기 의무).
    ('BNineteenCmpZero', 'B19_KGRID_SPARSE_2026-08-25/verdict_b19.json',
     'neg_kgrid1_vs_zero.better', 'int', 12),
    ('BSixteenCmpZero', '_audit/verdict_t2_recheck.json',
     'b16_vs_true_zero.KGRID1.better', 'int', 38),
    # ★두 번째 계단(`\BSixteenDose`)의 우변은 **무기술**이다 --- 혼잡 6/48 · 희소 2/16.
    ('BSixteenDoseCmpZero', '_audit/verdict_t2_recheck.json',
     'b16_vs_true_zero.KGRID2.better', 'int', 6),
    # ── ★관측 모형의 첫 물리 계측 (H3-FRF, 2026-09-03) — 실물 Velodyne HDL-32E.
    #    ⛔음성이다 · ⛔쇄파대 레짐 한정(심해 미검) · ⛔추정기 런 0회.
    #    ★두 대조가 이것을 "무딤" 이 아니라 "신호 거의 없음" 으로 만든다.
    ('HThreeMRE', 'H3_FRF_2026-09-03/verdict_h3.json', 'primary.MRE', 'abs3', 0.348),
    ('HThreeThresh', 'H3_FRF_2026-09-03/verdict_h3.json', 'primary.threshold', 'sec', 0.2),
    ('HThreeCells', 'H3_FRF_2026-09-03/verdict_h3.json', 'n_cells', 'int', 236423),
    ('HThreeFiles', 'H3_FRF_2026-09-03/verdict_h3.json', 'n_rows', 'int', 30),
    ('HThreePosSpread', 'H3_FRF_2026-09-03/verdict_h3.json',
     'controls.pos_spread', 'abs3', 0.361),
    ('HThreePosNull', 'H3_FRF_2026-09-03/verdict_h3.json',
     'controls.pos_null_p95', 'abs3', 0.008),
    ('HThreeShuffled', 'H3_FRF_2026-09-03/verdict_h3.json',
     'controls.mre_shuffled_p5', 'abs3', 0.353),
    ('HThreeSpearman', 'H3_FRF_2026-09-03/verdict_h3.json',
     'secondary.spearman', 'abs3', 0.173),
    # ── ★참조면 잡음이 실측 기술을 갉아먹는가 (R7-FLOOR, 2026-09-03, 신규 런 0회)
    #    ⛔EXPLORATORY · ⛔blind 아님 · ⛔사이트 1개(BS 7) · ⛔PROP 구제 아님.
    ('RSevenRho', 'R7_FLOOR_2026-09-03/verdict_r7.json', 'primary.rho', 'abs3', 0.893),
    ('RSevenP', 'R7_FLOOR_2026-09-03/verdict_r7.json', 'primary.p_one_sided', 'p', 6.15e-03),
    ('RSevenN', 'R7_FLOOR_2026-09-03/verdict_r7.json', 'primary.n', 'int', 7),
    ('RSevenMDE', 'R7_FLOOR_2026-09-03/verdict_r7.json', 'mde_rho', 'abs3', 0.714),
    ('RSevenCtrlRho', 'R7_FLOOR_2026-09-03/verdict_r7.json',
     'controls.C1.rho', 'abs3', 0.0),
    ('RSevenCtrlP', 'R7_FLOOR_2026-09-03/verdict_r7.json',
     'controls.C1.p_one_sided', 'abs3', 0.518),
    # ── ★실측 채점이 「보간」이었는가 (R8-HELDOUT, 2026-09-03, 24런)
    #    ⛔명목이다(확증 11 에 한 표 모자람) · ★SEC-LARGE 가 3/5 로 갈렸다 — 병기 의무.
    ('REightBetter', 'R8_HELDOUT_2026-09-03/verdict_r8.json', 'primary.w', 'int', 10),
    ('REightN', 'R8_HELDOUT_2026-09-03/verdict_r8.json', 'primary.n', 'int', 12),
    ('REightP', 'R8_HELDOUT_2026-09-03/verdict_r8.json',
     'primary.p_one_sided', 'p', 1.9287e-02),
    ('REightRatioIn', 'R8_HELDOUT_2026-09-03/verdict_r8.json',
     'primary.ratio_in_med', 'abs3', 0.717),
    ('REightRatioOut', 'R8_HELDOUT_2026-09-03/verdict_r8.json',
     'primary.ratio_out_med', 'abs3', 0.979),
    ('REightSkillIn', 'R8_HELDOUT_2026-09-03/verdict_r8.json', 'primary.skill_in', 'int', 12),
    ('REightSkillOut', 'R8_HELDOUT_2026-09-03/verdict_r8.json', 'primary.skill_out', 'int', 8),
    ('REightBigW', 'R8_HELDOUT_2026-09-03/verdict_r8.json', 'SEC_LARGE.w', 'int', 3),
    ('REightBigN', 'R8_HELDOUT_2026-09-03/verdict_r8.json', 'SEC_LARGE.n', 'int', 5),
    # ── ★검출 법칙의 부류를 바꾸면 되는가 (H3c-LAW, 2026-09-03, 신규 계측 0회)
    #    ⛔EXPLORATORY · blind 아님(선행 H3b 차단 중 노출) · A(복사량)는 고정했다.
    #    ⛔셋 중 하나도 충실하지 않다 ⟹ "저장소의 다른 모형으로 갈아라" 가 아니다.
    ('HThreeCMreC', 'H3C_LAW_2026-09-03/verdict_h3c.json', 'model_C.MRE', 'abs3', 0.303),
    ('HThreeCMreA', 'H3C_LAW_2026-09-03/verdict_h3c.json', 'model_A.MRE', 'abs3', 0.348),
    ('HThreeCShufC', 'H3C_LAW_2026-09-03/verdict_h3c.json',
     'model_C.shuf_p5', 'abs3', 0.317),
    ('HThreeCRel', 'H3C_LAW_2026-09-03/verdict_h3c.json',
     'improvement.rel_pct', 'pct', -13.0),
    # ── ★★저장소 최초의 외부 기준선 행 (E1-EXTBASE, 2026-09-03)
    #    ⛔예산 규약 병기 없이 인용 금지: 라벨 칸은 저들이 관측을 **두 배** 받는다.
    #    ⛔비인과성(저들은 t±1 을 본다) · 최근접 되샘플 · 저들 학습 분포 미확인.
    ('EOneShipWin', 'E1_EXTBASE_2026-09-03/verdict_e1.json', 'primary.w', 'int', 2),
    ('EOneN', 'E1_EXTBASE_2026-09-03/verdict_e1.json', 'primary.n', 'int', 12),
    ('EOneRung', 'E1_EXTBASE_2026-09-03/verdict_e1.json', 'primary.rung', 'int', 400),
    ('EOneP', 'E1_EXTBASE_2026-09-03/verdict_e1.json',
     'primary.p_one_sided', 'p', 1.9287e-02),
    # ★밀도 사다리 — 이 배치가 사는 가장 견고한 문장
    ('EOneSelfTwelve', 'E1_EXTBASE_2026-09-03/verdict_e1.json',
     'self_test.per_rung.12.better_than_zero', 'int', 0),
    ('EOneSelfTwoHundred', 'E1_EXTBASE_2026-09-03/verdict_e1.json',
     'self_test.per_rung.200.better_than_zero', 'int', 7),
    ('EOneSelfFourHundred', 'E1_EXTBASE_2026-09-03/verdict_e1.json',
     'self_test.per_rung.400.better_than_zero', 'int', 10),
    ('EOneMedShip', 'E1_EXTBASE_2026-09-03/verdict_e1.json',
     'medians.ship', 'abs3', 0.616),
    ('EOneMedZero', 'E1_EXTBASE_2026-09-03/verdict_e1.json',
     'medians.zero', 'abs3', 0.937),
    ('EOneMedCnnEqual', 'E1_EXTBASE_2026-09-03/verdict_e1.json',
     'medians.cnn_200', 'abs3', 0.648),
    # ── ★★참값과 모델류의 거리 (C1-MODELGAP, 2026-09-03, 추정기 런 0회)
    #    ⛔거리를 잰 것이지 **충분성**을 잰 것이 아니다. ⛔선형 코퍼스의 BLOCKER 는 유지된다.
    ('COneCrowdedMin', 'C1_MODELGAP_2026-09-03/verdict_c1.json',
     'strata.hos_crowded.min', 'abs3', 0.308),
    ('COneCrowdedMed', 'C1_MODELGAP_2026-09-03/verdict_c1.json',
     'strata.hos_crowded.med', 'abs3', 0.603),
    ('COneCrowdedMax', 'C1_MODELGAP_2026-09-03/verdict_c1.json',
     'strata.hos_crowded.max', 'abs3', 1.416),
    ('COneCrowdedN', 'C1_MODELGAP_2026-09-03/verdict_c1.json',
     'strata.hos_crowded.n', 'int', 48),
    ('COneCtrlMax', 'C1_MODELGAP_2026-09-03/verdict_c1.json',
     'strata.linear_control.max', 'abs3', 0.0),
    ('COneCtrlN', 'C1_MODELGAP_2026-09-03/verdict_c1.json',
     'strata.linear_control.n', 'int', 12),
]


FMT = {
    'int': lambda x: str(int(x)),
    'pct': lambda x: f'{x:+.1f}\\%'.replace('+', '') if x < 0 else f'{x:+.1f}\\%',
    'p': lambda x: _sci(x),
    # ⚠`M-10` 은 rel 을 **분수**로 저장한다 — 단위를 여기서 흡수한다(⛔논문에서 곱하지 마라).
    'frac': lambda x: f'{x*100:+.1f}\\%'.replace('+', '') if x < 0 else f'{x*100:+.1f}\\%',
    # ⚠**절대** nRMSE --- 퍼센트가 아니다. F1 이 가르친 것: 승수와 절대는 순위를 뒤집는다.
    'abs3': lambda x: f'{x:.3f}',
    'us': lambda x: f'{x * 1e6:.0f}',
    'sec': lambda x: f'{x:g}',
    'ms2': lambda x: f'{x:.2f}',
    'pct100': lambda x: f'{x * 100:.1f}',
    # ⚠**밀리미터**. ⛔퍼센트도 nRMSE 도 아니다 --- 절대 잡음 진폭이다.
    'mm1': lambda x: f'{x:.1f}',
}


def _sci(x):
    from math import floor, log10
    if x == 0:
        return '0'
    e = int(floor(log10(abs(x))))
    m = x / 10 ** e
    return f'{m:.2f}\\times 10^{{{e}}}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--show', action='store_true')
    ap.add_argument('--check', action='store_true',
                    help='커밋된 paper_constants.tex 가 생성물과 같은가 (쓰지 않는다)')
    a = ap.parse_args()

    cache, rows, bad = {}, [], []
    for name, rel, path, kind, pin in ATOMS:
        v = cache.setdefault(rel, load(rel))
        try:
            raw = _get(v, path)
        except (KeyError, IndexError, TypeError) as e:
            bad.append((name, f'경로 없음 {path}: {e}'))
            continue
        # 핀 대조 — 정수는 정확히, 실수는 표시 자릿수에서
        if kind == 'int':
            ok = int(raw) == int(pin)
        elif kind in ('pct', 'frac'):
            val = raw * 100 if kind == 'frac' else raw
            ok = abs(round(val, 1) - pin) < 1e-9
        elif kind == 'abs3':
            ok = abs(round(raw, 3) - pin) < 1e-9
        elif kind == 'us':
            ok = abs(round(raw * 1e6) - pin) < 1e-9
        elif kind == 'sec':
            ok = abs(raw - pin) < 1e-12
        elif kind == 'ms2':
            ok = abs(round(raw, 2) - pin) < 1e-9
        elif kind == 'mm1':
            ok = abs(round(raw, 1) - pin) < 1e-9
        elif kind == 'pct100':
            ok = abs(round(raw * 100, 1) - pin) < 1e-9
        else:
            ok = abs(raw - pin) / abs(pin) < 0.005
        if not ok:
            bad.append((name, f'핀 {pin} != 실측 {raw}'))
        rows.append((name, FMT[kind](raw), rel.split('/')[0]))

    if a.show or bad:
        for n, val, src in rows:
            print(f'  {n:22s} = {val:22s}  [{src}]')
    if bad:
        print('\n⛔핀 불일치:')
        for n, why in bad:
            print(f'   {n}: {why}')
        raise AssertionError(f'{len(bad)}건 — 배치가 재분석됐다면 ATOMS 의 핀을 '
                             '**의도적으로** 갱신하라(래칫 규약).')

    # ── ★배포 상수도 같은 파일에 낸다(2026-09-01, 2차 감사).
    #    사유: 논문이 `\lambda_1 = 10^{-3}` 을 **손으로** 적고 있었는데 배포값은 1e-4 다
    #    (2026-08-11 변경). 손으로 적힌 배포 상수는 조용히 낡는다 — 이 저장소의 반복 결함.
    from wavesplat.optimizer.sequential_update import (
        AUTO_SWITCH_THRESH_DEFAULT,
        DEFAULT_ANALYTIC_ENSEMBLE_K, DEFAULT_LEARNED_ENSEMBLE_K, LAMBDA_L1_DEFAULT,
        VARPRO_STEPS_DEFAULT, LEARNED_INIT_ITERS_DEFAULT)
    from wavesplat.run import DEFAULT_CONFIG as _DC
    from math import log10
    dep = [
        ('DepLambdaOne', f'10^{{{int(round(log10(LAMBDA_L1_DEFAULT)))}}}'),
        ('DepVarproSteps', str(VARPRO_STEPS_DEFAULT)),
        ('DepJointFitIters', str(LEARNED_INIT_ITERS_DEFAULT)),
        ('DepAnalyticEnsK', str(DEFAULT_ANALYTIC_ENSEMBLE_K)),
        ('DepLearnedEnsK', str(DEFAULT_LEARNED_ENSEMBLE_K)),
        ('DepTrackingLr', f"{_DC['tracking_lr']:g}"),
        ('DepTrackingSteps', str(_DC['tracking_steps'])),
        ('DepMaxN', str(_DC['max_n'])),
        # ★[2026-09-02] 라우팅 임계 --- 이 값이 0 이라서 **합성 입력 전부**가 학습
        #   제안 경로로 간다(§용어 문단이 그 사실에 기댄다). ⛔손으로 적지 마라.
        ('DepAutoThresh', str(AUTO_SWITCH_THRESH_DEFAULT)),
        # ★테스트 수는 자주 바뀐다 — 손으로 적으면 반드시 낡는다(3차 감사: 196 vs 실측 261).
        ('DeployNumTestsPaper', _count_tests()),
        # ── ★[2026-09-02] 학습 제안의 **지지 상한**. 체크포인트 안에 있다.
        #   코드 감사 D8: 논문에 이 상한이 한 번도 적혀 있지 않았고, 손으로 적으면
        #   체크포인트를 갈 때 조용히 낡는다(H2 가 이미 열려 있는 축이다).
        ('DepKLim', _k_lim()),
        ('CkptSha', _ckpt_sha()),
        # ★64자 16진수는 `\texttt` 안에서 **끊기지 않아** 238pt overfull 을 만들었다.
        #   ⟹ 8자마다 `\allowbreak` 를 넣어 **내용은 그대로** 두고 줄바꿈만 허용한다.
        ('CkptShaFull', _breakable(_ckpt_sha(0))),
        ('PriorCkptSha', _prior_ckpt_sha()),
        # ⚠[2026-09-02] 검증 감사 25: Data availability 가 *"독자가 검증할 수 있게"*
        #   라 적으면서 이쪽만 12자였다. 전체 다이제스트도 낸다.
        ('PriorCkptShaFull', _breakable(_prior_ckpt_sha(0))),
        ('EvalSeedMax', _eval_seed_max()),
        # ── 해석적 대안의 탐색 범위(대비용). 함수 서명의 기본값에서 읽는다.
        ('DepAnaKMax', _ana_kmax()),
    ]
    dep += list(_prior_arch().items())
    dep += list(_training().items())
    dep += list(_mesh_grades().items())
    dep += list(_spec_controls().items())
    dep += list(_est_seeds().items())
    dep += list(_frame_span().items())
    dep += list(_qualitative().items())
    dep += list(_b12_seeds().items())
    dep += list(_donor_disjoint().items())
    dep += list(_corpus_identity().items())
    dep += list(_w15_reagg().items())
    dep += list(_b1_control().items())
    dep += list(_e1_equal_budget().items())
    dep += list(_seed_clustered().items())
    dep += list(_corpus().items())
    rows += [(k, v, 'DEPLOY') for k, v in dep]

    body = ['% ⛔자동 생성 — 손으로 고치지 마라.',
            '%    생성: python3 scripts/gen_paper_constants.py',
            '%    출처: 동결 verdict_*.json (배치명은 각 줄 주석에)',
            '%    ★이 파일이 존재하는 이유: paper/waveprior.tex 가 철회된 헤드라인을',
            '%      라이브 22줄에 **손으로** 인쇄하고 있었고 그 줄들이 독립적으로 낡았다.', '']
    for n, val, src in rows:
        body.append(f'\\newcommand{{\\{n}}}{{{val}}}  % {src}')
    text = '\n'.join(body) + '\n'
    # ★2차 감사 지적: 생성기를 **안 돌리고** tex 를 손으로 고치면 무탐지였다.
    #   `--check` 는 아무것도 쓰지 않고 **커밋된 파일과 대조**만 한다.
    if a.check:
        if not os.path.exists(OUT):
            print(f'⛔없다: {OUT} — 생성기를 돌려라')
            return 1
        if open(OUT).read() != text:
            print('⛔`paper_constants.tex` 가 생성물과 다르다 (손편집됐거나 낡았다).')
            print('   수리: python3 scripts/gen_paper_constants.py')
            return 1
        print(f'✅{len(rows)} 상수 · 커밋된 파일이 생성물과 일치')
        return 0
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, 'w').write(text)
    if not a.show:
        for n, val, src in rows:
            print(f'  \\{n} = {val}')
    print(f'\n✅{len(rows)} 상수 · 핀 불일치 0 → {os.path.relpath(OUT, ROOT)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
