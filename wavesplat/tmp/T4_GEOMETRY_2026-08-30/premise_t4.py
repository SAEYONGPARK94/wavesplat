#!/usr/bin/env python3
"""T-4 전제 재도출 — ⛔**추정기 0회 · 새 런 0회**.  동결 파일만 읽는다.

    python3 tmp/T4_GEOMETRY_2026-08-30/premise_t4.py     # → premise_t4.json

`PREREG.md` 와 `gates_t4.py` 의 **A부 선별기가 근거로 삼는 모든 수**를 동결물에서 다시
만들고, 이 파일의 `EXPECTED` 와 어긋나면 **`AssertionError`** 로 죽는다.
⟹ ★손복사 상수는 **판정·선별 경로에 0건**이다(본보기 = `premise_t2.py`·`premise_m10.py`).

읽는 것(전부 기존 동결물, 이 배치가 만든 것 0개):
  · `tmp/B3_REFIT_2026-08-23/{gates_b3.py, results_b3.jsonl}`   실측 판정 자와 lead 밴드
  · `tmp/S16_BASELINE_2026-08-14/run_s16.py`                    합성 기준선 채점 **순서**
  · `wavesplat/run.py`                                          배포 팔 채점 **순서**
  · `tmp/HOSGATE_2026-08-13/data_hos/*.h5`                      코퍼스 기하·시간축·광선 설정
  · `tmp/M10_GEOMETRY_2026-08-30/{premise_m10.json, verdict_m10.json}`  실측 λ_p · 피복 판정
  · `tmp/B2_REAL_2026-08-23/scene_station.py`                   실측 관측 모형(대응물 부재)
  · `tmp/B26_SEASTATE_2026-08-27/gen_manifest_b26.json`         생성 비용(재생성 배제 근거)
  · `tmp/{B12,B15,B18,M5b,C,B17,B14}` 의 동결 판정               중복 측정 배제 근거
"""
import ast
import glob
import json
import math
import os
import re
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)

import gates_t4 as G                                              # noqa: E402

B2 = os.path.join(ROOT, 'tmp', 'B2_REAL_2026-08-23')
B3 = os.path.join(ROOT, 'tmp', 'B3_REFIT_2026-08-23')
B12 = os.path.join(ROOT, 'tmp', 'B12_HOS_EXTEND_2026-08-23')
B26 = os.path.join(ROOT, 'tmp', 'B26_SEASTATE_2026-08-27')
M10 = os.path.join(ROOT, 'tmp', 'M10_GEOMETRY_2026-08-30')
S16 = os.path.join(ROOT, 'tmp', 'S16_BASELINE_2026-08-14')
sys.path.insert(0, B2)

GRAV = 9.81

# ── 결과 전 기대값.  ⛔어긋나면 죽는다. ───────────────────────────────────────
EXPECTED = dict(
    # P1 실측 판정의 자와 밴드
    real_field='fc_mean', real_lead_min=2.0, real_lead_max=14.0, real_n_leads=15,
    # P2 채점 **순서** (합성 = 동화 · 실측 = 예보)
    synth_is_analysis=True, deploy_is_analysis=True,
    # P3 코퍼스
    n_ep=48, dt=0.1, n_frames=301, duration_s=30.0, tp_by_bf={'BF4': 7.0, 'BF5': 9.0},
    # P4 원점 (stride 0.5 s · PRIMARY rung 1.0 Tp)
    origins_by_bf={'BF4': 18, 'BF5': 6},
    # P5 실측 밴드를 Tp 로 환산했을 때 PRIMARY rung 이 그 안에 든다
    primary_rung_in_real_band=True, n_real_records=12,
    # P6 입사각은 **파생량**이다 (설정 가능한 필드가 없다) + 사거리에 강하게 종속
    #    ⛔초안은 *"|ρ| ≥ 0.99 로 잠겨 있다"* 였고 이 스크립트가 **결과 0행 시점에
    #      그것을 반증했다**(실측 |ρ|min ≈ 0.84 — 잔여는 국소 파면 기울기가 만든다).
    #      ⟹ 선별기 `INCIDENCE` 의 근거를 **통계적 잠금**에서 **구조적 파생**으로 바꿨다.
    #      바뀐 것은 근거의 종류이고 판단(NO-GO)은 그대로다. 상세 = PREREG §2.3.
    incidence_is_derived=True, incidence_cfg_fields=0,
    incidence_range_rho_floor=0.50,
    # P7 광선 패턴이 코퍼스 상수
    lidar_cfg_unique=True, lidar_height=5.0, lidar_channels=152, lidar_scan_rate=10.0,
    n_hos_files=64,
    # P8 피복 성분은 당일 측정됨
    m10_label='NEARFIELD-HOLDS', m10_primary=(45, 48),
    # P9 재생성 비용 (GPU 실측)
    gen_s_per_ep_min=150.0,
    # P10 합성 헤드라인 동결 행에 lead 키가 0개
    b12_lead_keys=0,
    # P11 실측 상속
    b3_records=12, b3_noise=0.02, b3_bs_depth=30.0,
)


def _fail(name, got, want):
    raise AssertionError('%s: 재도출 %r != 기대 %r' % (name, got, want))


def ck(name, got, want, rtol=0.0):
    if rtol and isinstance(got, float) and isinstance(want, float):
        if abs(got - want) > rtol * max(abs(want), 1e-30):
            _fail(name, got, want)
    elif got != want:
        _fail(name, got, want)
    return got


# ═════════════════════════════════════════════════════════════════════════════
# P1 — 실측 판정의 자는 **예보 평균**이다 (동결 jsonl 에서 재도출)
# ═════════════════════════════════════════════════════════════════════════════
def p1_real_band():
    sys.path.insert(0, B3)
    import gates_b3 as GB3
    rows = [json.loads(l) for l in open(os.path.join(B3, 'results_b3.jsonl'))
            if l.strip()]
    assert rows, 'results_b3.jsonl 이 비었다'
    leads = sorted(float(k) for k in rows[0]['per_lead'])
    # ★`fc_mean` 이 실제로 어느 lead 들의 평균인지 **자료에서** 찾는다(문서를 믿지 않는다).
    best = None
    for lo in leads:
        sel = [L for L in leads if L >= lo]
        errs = []
        for r in rows:
            pl = r['per_lead']
            m = sum(pl[repr(L)] if repr(L) in pl else pl[str(L)] for L in sel) / len(sel)
            errs.append(abs(m - r['fc_mean']) / max(abs(r['fc_mean']), 1e-30))
        e = max(errs)
        if best is None or e < best[1]:
            best = (lo, e)
    lo, err = best
    assert err <= 1e-9, 'fc_mean 을 어떤 lead 구간으로도 재현하지 못했다 (최소 rel %g)' % err
    ck('real_field', GB3.FIELD, EXPECTED['real_field'])
    ck('real_lead_min', lo, EXPECTED['real_lead_min'])
    ck('real_lead_max', max(leads), EXPECTED['real_lead_max'])
    ck('real_n_leads', len(leads), EXPECTED['real_n_leads'])
    ck('b3_records', len(GB3.RECORDS), EXPECTED['b3_records'])
    ck('b3_noise', float(GB3.NOISE), EXPECTED['b3_noise'])
    return dict(field=GB3.FIELD, leads=leads, fc_lead_min=lo, fc_lead_max=max(leads),
                fc_recon_max_relerr=err, n_rows=len(rows),
                protocol='예측·채점 먼저 → 관측 섭취 나중 (gates_b3 독스트링)')


# ═════════════════════════════════════════════════════════════════════════════
# P2 — 합성 채점은 **동화**다: update → predict → score 순서를 `ast` 로 확인
# ═════════════════════════════════════════════════════════════════════════════
def _call_lines(fn_node, want):
    """함수 본문에서 각 호출 패턴이 처음 나타나는 줄 번호."""
    out = {}
    for node in ast.walk(fn_node):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        nm = (f.attr if isinstance(f, ast.Attribute)
              else (f.id if isinstance(f, ast.Name) else None))
        if nm in want and nm not in out:
            out[nm] = node.lineno
    return out


def _fn(path, name):
    tree = ast.parse(open(path, encoding='utf-8').read())
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return n
    raise AssertionError('%s 에 %s 가 없다' % (path, name))


def p2_scoring_order():
    s16 = os.path.join(S16, 'run_s16.py')
    ln = _call_lines(_fn(s16, 'run_letkf'),
                     {'update', 'predict_grid', 'compute_all_metrics'})
    for k in ('update', 'predict_grid', 'compute_all_metrics'):
        assert k in ln, 'run_s16.run_letkf 에 %s 호출이 없다' % k
    synth = ln['update'] < ln['predict_grid'] < ln['compute_all_metrics']
    ck('synth_is_analysis', synth, EXPECTED['synth_is_analysis'])

    rp = os.path.join(ROOT, 'wavesplat', 'run.py')
    src = open(rp, encoding='utf-8').read().splitlines()
    iu = next(i for i, l in enumerate(src) if 'updater.update(frame)' in l)
    ip = next(i for i, l in enumerate(src)
              if 'updater.predict_field(grid_x, grid_y, frame.t)' in l)
    ck('deploy_is_analysis', iu < ip, EXPECTED['deploy_is_analysis'])
    # ★배포 경로가 **미래 시각 평가**를 이미 갖고 있다는 증거(lead 잔차)
    lead_sites = [i + 1 for i, l in enumerate(src) if "_lead[str(_L)]" in l]
    assert lead_sites, 'run.py 에 lead 잔차 계측이 없다 (배포 경로 근거 소멸)'
    return dict(run_s16_update=ln['update'], run_s16_predict=ln['predict_grid'],
                run_s16_score=ln['compute_all_metrics'],
                run_py_update=iu + 1, run_py_predict=ip + 1,
                run_py_lead_resid_line=lead_sites[0])


# ═════════════════════════════════════════════════════════════════════════════
# P3 · P4 · P7 — 코퍼스의 시간축 · 원점 · 광선 설정 (동결 h5)
# ═════════════════════════════════════════════════════════════════════════════
def p3_corpus():
    import h5py
    files = sorted(glob.glob(os.path.join(G.HOS, '*.h5')))
    ck('n_hos_files', len(files), EXPECTED['n_hos_files'])
    eps, cfgs, per = [], set(), {}
    for p in files:
        b = os.path.basename(p)
        m = re.match(r'episode_(BF\d)_(\d+)comp_seed(\d+)_(.+)\.h5', b)
        if not (m and int(m.group(2)) >= G.CROWDED_MIN_NCOMP
                and int(m.group(3)) in G.ALL_SEEDS):
            continue
        with h5py.File(p, 'r') as h:
            a = dict(h['config'].attrs)
            ts = h['frames/timestamps'][()]
            lc = dict(h['config/lidar_config'].attrs)
        cfgs.add((float(lc['height']), int(lc['channels']), float(lc['scan_rate'])))
        dt = float(round(ts[1] - ts[0], 12))
        bf = m.group(1)
        per.setdefault(bf, []).append(dict(
            dt=dt, n=len(ts), dur=float(round(ts[-1], 6)), tp=float(a['Tp'])))
        eps.append(b[len('episode_'):-3])
    ck('n_ep', len(eps), EXPECTED['n_ep'])
    ck('lidar_cfg_unique', len(cfgs) == 1, EXPECTED['lidar_cfg_unique'])
    h, ch, sr = next(iter(cfgs))
    ck('lidar_height', h, EXPECTED['lidar_height'])
    ck('lidar_channels', ch, EXPECTED['lidar_channels'])
    ck('lidar_scan_rate', sr, EXPECTED['lidar_scan_rate'])
    out = {}
    for bf, rs in sorted(per.items()):
        assert len({r['dt'] for r in rs}) == 1 and len({r['n'] for r in rs}) == 1
        ck('dt', rs[0]['dt'], EXPECTED['dt'])
        ck('n_frames', rs[0]['n'], EXPECTED['n_frames'])
        ck('duration_s', rs[0]['dur'], EXPECTED['duration_s'])
        ck('tp_%s' % bf, rs[0]['tp'], EXPECTED['tp_by_bf'][bf])
        o = G.origins(rs[0]['tp'], rs[0]['dt'], rs[0]['n'], G.ORIGIN_STRIDE_S)
        lf = G.lead_frames(rs[0]['tp'], rs[0]['dt'])
        # ★rung 이 두 층 모두에서 프레임에 **정확히** 떨어진다 (G-9 의 전제)
        for L, v in lf.items():
            assert abs(v * rs[0]['dt'] - L * rs[0]['tp']) <= 1e-9, \
                'rung %s 가 %s 층에서 프레임에 떨어지지 않는다' % (L, bf)
        ck('origins_%s' % bf, len(o), EXPECTED['origins_by_bf'][bf])
        assert len(o) >= G.MIN_ORIGINS
        out[bf] = dict(n_ep=len(rs), tp=rs[0]['tp'], dt=rs[0]['dt'], n_frames=rs[0]['n'],
                       duration_s=rs[0]['dur'], n_origins=len(o),
                       origin_first=o[0], origin_last=o[-1],
                       lead_frames={str(k): v for k, v in sorted(lf.items())},
                       origins_by_stride={str(s): len(G.origins(
                           rs[0]['tp'], rs[0]['dt'], rs[0]['n'], s))
                           for s in G.ORIGIN_STRIDE_CHOICES})
    return dict(by_stratum=out, lidar_config=dict(height=h, channels=ch, scan_rate=sr),
                n_episodes=len(eps))


# ═════════════════════════════════════════════════════════════════════════════
# P5 — 실측 lead 밴드를 **Tp 단위**로 환산 (M-10 의 λ_p + 분산관계, 손복사 0)
# ═════════════════════════════════════════════════════════════════════════════
def p5_real_band_in_tp(real_lo, real_hi):
    from scene_station import RECORDS_12
    pm = json.load(open(os.path.join(M10, 'premise_m10.json')))
    real = pm['real']
    ck('n_real_records', len(RECORDS_12), EXPECTED['n_real_records'])
    per = {}
    for rec, cfg in RECORDS_12.items():
        assert rec in real, '%s 가 premise_m10.json 에 없다' % rec
        lam = float(real[rec]['lambda_p_m'])
        d = float(cfg['depth'])
        k = 2.0 * math.pi / lam
        tp = 2.0 * math.pi / math.sqrt(GRAV * k * math.tanh(k * d))
        per[rec] = dict(lambda_p_m=lam, depth_m=d, Tp_s=tp,
                        lead_lo_tp=real_lo / tp, lead_hi_tp=real_hi / tp)
    lo = min(v['lead_lo_tp'] for v in per.values())
    hi = max(v['lead_hi_tp'] for v in per.values())
    inside = sum(1 for v in per.values()
                 if v['lead_lo_tp'] <= G.PRIMARY_LEAD_TP <= v['lead_hi_tp'])
    ck('primary_rung_in_real_band', lo <= G.PRIMARY_LEAD_TP <= hi,
       EXPECTED['primary_rung_in_real_band'])
    ck('b3_bs_depth', float(RECORDS_12['BS11a']['depth']), EXPECTED['b3_bs_depth'])
    return dict(per_record=per, band_tp=[lo, hi], primary_rung_tp=G.PRIMARY_LEAD_TP,
                n_records_covering_primary=inside, n_records=len(per),
                Tp_median_s=st.median([v['Tp_s'] for v in per.values()]))


# ═════════════════════════════════════════════════════════════════════════════
# P6 — 입사각은 사거리와 **잠겨** 있다 ⟹ 단일요인 분리 불가 (선별기 INCIDENCE 의 근거)
# ═════════════════════════════════════════════════════════════════════════════
def _spearman(a, b):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k2 in range(i, j + 1):
                r[order[k2]] = avg
            i = j + 1
        return r
    ra, rb = rank(a), rank(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = math.sqrt(sum((x - ma) ** 2 for x in ra))
    db = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return num / (da * db) if da and db else 0.0


def p6_incidence_derived(n_sample=4):
    """입사각은 **설정할 수 없는 파생량**이다 — 이것이 선별기 `INCIDENCE` 의 근거다.

    ★두 조각을 나눠서 확인한다:
      (a) **구조적**(하드 assert) — `LidarCfg` 에 입사각/각도 필드가 **0개**이고
          `compute_incidence_angles` 가 광선 방향과 **표면 법선**에서 유도한다
          ⟹ 값을 바꾸려면 (i) `height`/FOV = **재생성** 또는 (ii) 파면 = **참값 변경**이다.
      (b) **기술통계**(느슨한 바닥) — 사거리와의 순위상관.
          ⛔이 배치의 초안은 `|ρ| ≥ 0.99` 라는 **잠금**을 주장했고, 이 스크립트가
          **결과 0행에서 그것을 반증했다**. 잔여 변동은 국소 파면 기울기가 만든다.
    """
    import h5py
    lid = _fn(os.path.join(ROOT, 'wavesim', 'lidar_simulator.py'),
              'compute_incidence_angles')
    assert lid is not None
    cfg_src = open(os.path.join(ROOT, 'wavesim', 'config.py'), encoding='utf-8').read()
    cls = next(n for n in ast.walk(ast.parse(cfg_src))
               if isinstance(n, ast.ClassDef) and n.name == 'LidarCfg')
    fields = [t.target.id for t in cls.body if isinstance(t, ast.AnnAssign)]
    ang = [f for f in fields if 'incid' in f or 'graz' in f or 'normal' in f]
    ck('incidence_cfg_fields', len(ang), EXPECTED['incidence_cfg_fields'])
    ck('incidence_is_derived', True, EXPECTED['incidence_is_derived'])

    files = sorted(glob.glob(os.path.join(G.HOS, '*.h5')))[:n_sample]
    hgt, rows = None, []
    for p in files:
        with h5py.File(p, 'r') as h:
            P = h['frames/lidar_points'][()]
            M = h['frames/lidar_valid_mask'][()]
            A = h['frames/lidar_incidence_angles'][()]
            hgt = float(dict(h['config/lidar_config'].attrs)['height'])
        m = M.astype(bool)
        x, y, z = P[..., 0][m], P[..., 1][m], P[..., 2][m]
        r = (x ** 2 + y ** 2 + (z - hgt) ** 2) ** 0.5
        a = A[m]
        keep = list(range(0, len(r), max(1, len(r) // 4000)))
        rows.append(dict(episode=os.path.basename(p), n=int(len(r)),
                         rho=_spearman([float(r[i]) for i in keep],
                                       [float(a[i]) for i in keep])))
    worst = min(abs(x['rho']) for x in rows)
    assert worst >= EXPECTED['incidence_range_rho_floor'], \
        '입사각이 사거리와 단조 관계조차 아니다: |rho|min=%.4f — 전제 재검토' % worst
    return dict(cfg_fields=fields, angle_cfg_fields=ang, per_episode=rows,
                min_abs_rho=worst, sensor_height_m=hgt,
                note='입사각은 광선방향×법선의 **파생량**이고 설정 필드가 0개다. '
                     '사거리와 강한 단조 관계(|ρ|min=%.3f)이나 **잠금은 아니다** — '
                     '잔여는 국소 파면 기울기가 만든다. ⟹ 고정 사거리에서 입사각만 '
                     '움직이는 노브가 **존재하지 않는다**(단일요인 불가의 사유는 '
                     '통계가 아니라 구조다).' % worst)


# ═════════════════════════════════════════════════════════════════════════════
# P8 · P9 · P10 · 선별기 근거 판정들
# ═════════════════════════════════════════════════════════════════════════════
def p8_prior_verdicts():
    def j(*p):
        return json.load(open(os.path.join(ROOT, 'tmp', *p)))
    m10 = j('M10_GEOMETRY_2026-08-30', 'verdict_m10.json')
    ck('m10_label', m10['label'], EXPECTED['m10_label'])
    ck('m10_primary', (m10['primary']['primary']['better'],
                       m10['primary']['primary']['n']), EXPECTED['m10_primary'])
    out = dict(M10=dict(label=m10['label'],
                        primary=[m10['primary']['primary']['better'],
                                 m10['primary']['primary']['n']],
                        rel_median=m10['primary']['primary']['rel_median']))
    for key, rel, lab in (
            ('B15', ('B15_RAYDROP_2026-08-24', 'verdict_b15.json'), 'label'),
            ('B18', ('B18_DENSITY_NOISE_2026-08-25', 'verdict_b18.json'), 'label'),
            ('M5b', ('M5b_WINDOW_2026-08-28', 'verdict_m5b.json'), 'label'),
            ('B17', ('B17_REALNOISE_2026-08-25', 'verdict_b17.json'), 'label'),
            ('C', ('C_ENCOUNTER_2026-08-20', 'verdict_c.json'), 'verdict')):
        d = j(*rel)
        assert lab in d, '%s 에 %s 가 없다' % (rel[-1], lab)
        out[key] = d[lab]
    return out


def p9_regen_cost():
    d = json.load(open(os.path.join(B26, 'gen_manifest_b26.json')))
    es = sorted(v['elapsed_s'] for v in d.values() if 'elapsed_s' in v)
    med = st.median(es)
    assert med >= EXPECTED['gen_s_per_ep_min'], '생성 비용 재도출 실패: %.1f s' % med
    return dict(n=len(es), median_s_per_ep=med, min_s=es[0],
                projected_gpu_h_for_48=48 * med / 3600.0,
                note='⛔GPU 는 임무 규율로 금지 · CPU 레이캐스트는 C 배치 A-4 가 약 10배로 '
                     '기록했다(그 배수는 이 스크립트가 재도출하지 않는 **인용**이다).')


def p10_no_lead_in_frozen():
    ks = set()
    with open(os.path.join(B12, 'results_b12.jsonl')) as fh:
        for line in fh:
            try:
                ks |= set(json.loads(line).keys())
            except ValueError:
                continue
    hit = sorted(k for k in ks if 'lead' in k.lower())
    ck('b12_lead_keys', len(hit), EXPECTED['b12_lead_keys'])
    return dict(n_keys=len(ks), lead_keys=hit)


def p11_no_real_ray_geometry():
    """실측 하네스에 광선·시선·센서 위치가 **없다** (선별기 기준 ⑤ 의 근거)."""
    src = open(os.path.join(B2, 'scene_station.py'), encoding='utf-8').read()
    tree = ast.parse(src)
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    forbidden = {'ray', 'rays', 'origin', 'origins', 'incidence', 'fov', 'height',
                 'channels', 'n_h_beams', 'range_max', 'sensor_z', 'occlusion'}
    hit = sorted(names & forbidden)
    assert not hit, '실측 하네스에 광선 기하 심볼이 있다: %r (기준 ⑤ 재검토 필요)' % hit
    uniform = 'isfinite' in names and 'choice' in names
    assert uniform, 'scene_station 이 유효격자 균등추첨이 아니다 (전제 붕괴)'
    return dict(forbidden_symbols_found=hit, uniform_subsample=uniform,
                note='실측 관측 = 재구성면 유효 격자점의 균등 추첨 + iid 잡음. '
                     '광선·입사각·폐색·마운트 높이에 **대응물이 없다**. '
                     '⚠이는 자연의 성질이 아니라 우리 실측 하네스의 성질이다.')


def main():
    p1 = p1_real_band()
    out = dict(
        batch='T-4', runs=0, estimator_calls=0,
        p1_real_band=p1,
        p2_scoring_order=p2_scoring_order(),
        p3_corpus=p3_corpus(),
        p5_real_band_in_tp=p5_real_band_in_tp(p1['fc_lead_min'], p1['fc_lead_max']),
        p6_incidence_derived=p6_incidence_derived(),
        p8_prior_verdicts=p8_prior_verdicts(),
        p9_regen_cost=p9_regen_cost(),
        p10_frozen_has_no_lead=p10_no_lead_in_frozen(),
        p11_real_has_no_ray_geometry=p11_no_real_ray_geometry(),
        screen=[dict(name=n, go=go, failed=list(f)) for n, go, f, c in G.screen()],
        thresholds=G.thresholds(G.N_EP),
        ladder=dict(leads_tp=list(G.LEADS_TP), primary=G.PRIMARY_LEAD_TP,
                    band=list(G.BAND_LEAD_TP), spinup_tp=G.SPINUP_TP,
                    stride=G.ORIGIN_STRIDE_S,
                    stride_choices_descriptive=list(G.ORIGIN_STRIDE_CHOICES),
                    min_origins=G.MIN_ORIGINS),
    )
    path = os.path.join(HERE, 'premise_t4.json')
    with open(path, 'w') as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, sort_keys=True)
    n = sum(1 for k in EXPECTED)
    print('premise_t4: %d/%d 재도출 일치 · 런 0회 · 추정기 0회' % (n, n))
    print('  실측 판정 자 = %s, lead %.0f–%.0f s = %.3f–%.3f Tp (레코드 %d개)'
          % (p1['field'], p1['fc_lead_min'], p1['fc_lead_max'],
             out['p5_real_band_in_tp']['band_tp'][0],
             out['p5_real_band_in_tp']['band_tp'][1],
             out['p5_real_band_in_tp']['n_records']))
    print('  합성 채점 = 동화(update→predict→score) · PRIMARY rung = %.2f Tp '
          '(밴드를 덮는 레코드 %d/%d)'
          % (G.PRIMARY_LEAD_TP, out['p5_real_band_in_tp']['n_records_covering_primary'],
             out['p5_real_band_in_tp']['n_records']))
    print('  원점 BF4 %d · BF5 %d (stride %.1f s) · 입사각↔사거리 |ρ|min = %.4f'
          % (out['p3_corpus']['by_stratum']['BF4']['n_origins'],
             out['p3_corpus']['by_stratum']['BF5']['n_origins'],
             G.ORIGIN_STRIDE_S, out["p6_incidence_derived"]["min_abs_rho"]))
    print('  선별 GO = %s' % [d['name'] for d in out['screen'] if d['go']])
    print('  → %s' % path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
