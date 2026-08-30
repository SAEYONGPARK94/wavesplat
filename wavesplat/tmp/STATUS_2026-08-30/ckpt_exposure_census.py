"""A3 노출 census — ⛔런 0회, 소스만 읽는다.

질문 하나: **배치의 러너가 팔마다 다른 `.pt` 를 싣는가.**
학습 시드는 체크포인트 파일 바이트로만 배포 경로에 닿으므로(`learned_init.py:351`),
이 한 질문이 「학습 시드 편차에 대한 대비의 노출」을 완전히 결정한다.

출력이 `SEED_AXIS_AUDIT.md §2` 의 표와 어긋나면 실패한다.
"""
import os, re, sys

ROOT = '/home/vexit/wavesplat/wavesplat'
SKIP = ('M3b_DEPTH', 'M9_ALARM')          # 병행 실행 중 — 건드리지 않는다
JUDGE = re.compile(r'^(B\d|B_|B2_|B3_|B5_|B6_|B7_|B8_|B9_|B2[0-9]|S1[3467]|W\d|M[1-6]_'
                   r'|HOSGATE|C_ENCOUNTER)')
PT = re.compile(r'[A-Za-z0-9_./-]+\.pt\b')
CK = re.compile(r'learned_ckpt')
ASSIGN = re.compile(r"""learned_ckpt['"]?\s*\]?\s*=|['"]learned_ckpt['"]\s*:""")


def scan(d):
    pts, ck, assign = set(), 0, 0
    for dp, _dn, fn in os.walk(os.path.join(ROOT, 'tmp', d)):
        if '__pycache__' in dp or f'{os.sep}snap' in dp:
            continue
        for f in fn:
            if not f.endswith(('.py', '.sh')):
                continue
            s = open(os.path.join(dp, f), errors='ignore').read()
            pts |= {m for m in PT.findall(s) if len(m) > 3}
            ck += len(CK.findall(s))
            assign += len(ASSIGN.findall(s))
    return pts, ck, assign


dirs = [d for d in sorted(os.listdir(os.path.join(ROOT, 'tmp')))
        if os.path.isdir(os.path.join(ROOT, 'tmp', d))
        and not any(s in d for s in SKIP)]
judge = [d for d in dirs if JUDGE.match(d)]

exposed, clean = {}, 0
for d in judge:
    pts, ck, assign = scan(d)
    if pts or assign:
        exposed[d] = dict(pt=sorted(pts), ckpt_assign=assign, ckpt_mentions=ck)
    else:
        clean += 1

print(f'판정 배치 {len(judge)}개 검사')
print(f'  체크포인트 주입 0건 = {clean}개')
print(f'  이탈 = {exposed if exposed else "없음"}')
assert clean == len(judge), f'노출 배치 발견: {exposed}'

# 전체 tmp 에서 팔이 체크포인트로 갈리는 배치 (판정 배치 밖 포함)
multi = {}
for d in dirs:
    pts, ck, assign = scan(d)
    real = {p for p in pts if '/direction' in p or 'diffusion' in p or 'prior_net' in p}
    if len(real) > 1 or assign:
        multi[d] = dict(pt=sorted(real), ckpt_assign=assign)
print('\n★팔이 체크포인트로 갈릴 수 있는 배치(판정 밖 포함):')
for k, v in multi.items():
    print(f'  {k}: {v}')
EXPECT = {'M7_SUPERVISION_2026-08-28', 'mixed_retrain_2026-07-26',
          'wp_e2e_2026-07-25', 'G1_waveprior_power'}
assert set(multi) == EXPECT, f'문서 §2 와 불일치: {set(multi) ^ EXPECT}'
print('\n✅ SEED_AXIS_AUDIT.md §2 와 일치')

# ── ⚠`.pt` 스캔의 사각지대: **원자 캐시 경로** ───────────────────────────────
# `learned_atoms` 를 주면 확산 제안이 통째로 건너뛰므로(`learned_init.py:347-350`),
# 서로 다른 체크포인트로 만든 **캐시 두 개**를 쓰면 팔이 체크포인트로 갈리는데도
# 러너에는 `.pt` 문자열이 하나도 없다.  `confirm2` 가 정확히 그 모양이다.
CACHE = re.compile(r'atoms[A-Za-z0-9_]*_v(\d)[A-Za-z0-9_]*\.jsonl')
cache_multi = {}
for d in dirs:
    vs = set()
    p = os.path.join(ROOT, 'tmp', d)
    for f in os.listdir(p) if os.path.isdir(p) else []:
        vs |= set(CACHE.findall(f))
    if len(vs) > 1:
        cache_multi[d] = sorted(vs)
print('\n★원자 캐시가 체크포인트별로 둘 이상인 배치:')
for k, v in cache_multi.items():
    print(f'  {k}: 캐시 판본 {v}')
assert set(cache_multi) == {'mixed_retrain_2026-07-26', 'confirm40_2026-07-27',
                           'confirm2_2026-07-27'}, cache_multi

# ── 은퇴 체크포인트 고정 (부류 C) ────────────────────────────────────────────
RETIRED = 'beamform_cond_diffusion.pt'
# ⚠**언급**과 **적재**를 구분한다.  M7 `train_m7.py:68,513` 은 은퇴본을 *덮어쓰지 않기 위한
#   가드*로만 이름을 쓴다(적재 0회) — 그것까지 세면 오탐이다.
LOADS = re.compile(r'(torch\.load|load_state_dict|--ckpt|default\s*=|learned_ckpt)')
after_switch = []          # 배포 전환 2026-07-27 16:50 (`e4209eec`) **이후** 만들어진 배치
for d in dirs:
    if not re.search(r'2026-0[89]-', d):
        continue
    for dp, _dn, fn in os.walk(os.path.join(ROOT, 'tmp', d)):
        if '__pycache__' in dp:
            continue
        for f in fn:
            if not f.endswith('.py'):
                continue
            for line in open(os.path.join(dp, f), errors='ignore'):
                if RETIRED in line and 'mixed' not in line and LOADS.search(line):
                    after_switch.append(d)
                    break
            else:
                continue
            break
        else:
            continue
        break
print('\n★배포 전환 이후 배치인데 **은퇴 체크포인트를 적재**하는 것:', after_switch)
assert after_switch == ['A2_realproto_2026-08-06'], after_switch
print('\n✅ §2 표 전건(부류 A·B·C) 재현')
