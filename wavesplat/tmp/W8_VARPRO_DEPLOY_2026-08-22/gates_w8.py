"""W8 단일출처 — 팔 · 게이트 · 판정. 러너와 분석기가 둘 다 이 파일만 읽는다."""

# 배포 검증. 3팔 × 혼잡 16ep × 시드 1 = 48런.
ARMS = {
    # 배포된 기본값 그대로 (아무것도 지정하지 않는다 = 진짜 출하 경로)
    'PKG-VP': {},
    # 몽키패치 복제 — 패키지 처치를 끄고 tmp/W4 의 varpro.install(400) 을 얹는다.
    'MP-VP':  {'varpro_steps': 0, 'learned_init_iters': 0},
    # 배포 전 거동
    'PRE':    {'varpro_steps': 0, 'learned_init_iters': 500},
}
ARM_ORDER = ['PRE', 'MP-VP', 'PKG-VP']
MONKEYPATCH_ARM = 'MP-VP'
VP_STEPS = 400
REGIME, N_EPISODES, SEEDS = 'crowded', 16, [1618]
FIELD = 'avg_nRMSE'
FIDELITY_TOL = 1e-9      # 상대차. 결과 보기 전 확정.
FIDELITY_THRESHOLD = 15  # /16
EFFECT_THRESHOLD = 13    # /16 (저장소 표준)


def sign_test_one_sided(w, n):
    from math import comb
    return sum(comb(n, k) for k in range(w, n + 1)) / 2 ** n


def paired(rows, arm, ref, field=FIELD):
    import statistics as st
    a = {r['episode']: r[field] for r in rows if r['cell'] == arm}
    b = {r['episode']: r[field] for r in rows if r['cell'] == ref}
    ks = sorted(set(a) & set(b)); n = len(ks)
    lower = sum(1 for k in ks if a[k] < b[k])
    rel = st.median([(a[k] - b[k]) / b[k] * 100 for k in ks]) if ks else float('nan')
    ident = sum(1 for k in ks if b[k] and abs(a[k] - b[k]) / abs(b[k]) <= FIDELITY_TOL)
    return {'arm': arm, 'ref': ref, 'n': n, 'lower': lower, 'rel_median_pct': rel,
            'identical': ident, 'p_one_sided': sign_test_one_sided(lower, n) if n else None}


def gates(rows):
    ok = [r for r in rows if 'error' not in r]
    out = [('G-0 런 수', len(ok) == len(ARMS) * N_EPISODES * len(SEEDS),
            f'{len(ok)}/{len(ARMS) * N_EPISODES * len(SEEDS)}'),
           ('G-0b 에러 0', len(rows) == len(ok), f'{len(rows) - len(ok)} 에러')]
    # ★G-1 처치가 실제로 발화했나 — 추론이 아니라 카운터로 (누적 26축이 이 결함으로 죽었다)
    pkg = [r for r in ok if r['cell'] == 'PKG-VP']
    pre = [r for r in ok if r['cell'] == 'PRE']
    out.append(('G-1 PKG-VP 전건 발화', all(r.get('varpro_fired') for r in pkg),
                f'{sum(1 for r in pkg if r.get("varpro_fired"))}/{len(pkg)}'))
    out.append(('G-1b PRE 발화 0 (음성대조)', not any(r.get('varpro_fired') for r in pre),
                f'{sum(1 for r in pre if r.get("varpro_fired"))}/{len(pre)}'))
    out.append(('G-2 PKG-VP steps=400', all(r.get('varpro_steps') == VP_STEPS for r in pkg), ''))
    out.append(('G-3 MP-VP 패키지 처치 off',
                all(r.get('varpro_steps') == 0 for r in ok if r['cell'] == 'MP-VP'), ''))
    # G-4 라우팅이 팔 간 동일해야 한다 (처치가 라우팅을 바꾸면 ITT 가 아니다)
    import collections
    per = collections.defaultdict(set)
    for r in ok:
        per[r['episode']].add(r.get('route'))
    out.append(('G-4 팔 간 라우팅 일치',
                all(len(v) == 1 for v in per.values()),
                f'{sum(1 for v in per.values() if len(v) != 1)} 불일치'))
    return out


def label(fid, eff):
    """PRIMARY = 이식 충실도. SECONDARY = 배포 전후 효과."""
    if fid['identical'] < FIDELITY_THRESHOLD:
        return ('PORT-DIVERGES',
                f'패키지와 몽키패치가 갈린다({fid["identical"]}/{fid["n"]} 동일) '
                '⟹ W4/W5/B 의 측정이 이 배포로 전이되지 않는다.')
    if eff['lower'] >= EFFECT_THRESHOLD:
        return ('DEPLOY-VERIFIED',
                '이식이 충실하고 배포 전 대비 개선이 재현된다.')
    return ('PORT-OK-EFFECT-NULL',
            '이식은 충실하나 이 표본에서 배포 전 대비 개선이 문턱에 못 미친다.')
