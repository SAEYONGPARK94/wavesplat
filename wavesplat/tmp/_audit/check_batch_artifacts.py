#!/usr/bin/env python3
"""배치 산출물 검사기 — **산문 처방이 두 번 실패해서 만든 기계적 방벽** (2026-08-23).

오늘 같은 결함이 세 번 났다:
  · W15  — 동결 분석기를 안 돌리고 라벨을 손으로 붙였다(`verdict_w15.json` 부재)
  · W17  — 분석기 자체가 없었고 `verdict` 에 `gates`·`label` 이 없었다
  · W12  — verdict 는 있으나 게이트 결과가 안 들어갔다

검사 항목 (배치 디렉터리 하나당):
  A. `PREREG.md` 존재
  B. `gates_*.py` 존재
  C. `analyze_*.py` 존재                     ← 판정이 코드에서 나왔다는 증거
  D. `verdict_*.json` 존재 + `label` 키 보유  ← 라벨이 손으로 붙지 않았다는 증거
  E. `verdict_*.json` 이 `gates` 키 보유      ← 게이트가 실제로 돌았다는 증거
  F. `FROZEN_md5*.txt` 존재
  G. `analyze_*.py` 가 `results*.jsonl` 보다 **먼저** 쓰였는가 (mtime)
  H. **결과 파일마다 짝이 되는 verdict 이 있는가** (디렉터리 단위가 아니라 **분석 단위**)
  I. **`FROZEN_md5*.txt` 에 적힌 해시가 현재 파일과 일치하는가** (동결 위반 탐지)

⚠**RV-5(2026-08-23)가 지적한 이 검사기의 사각지대**를 H·I 로 메웠다:
  · 종전에는 **디렉터리 단위**라 `results_b2b/c/d/e.jsonl` 처럼 **한 디렉터리에 여러 분석**이
    있으면 하나만 있어도 OK 로 셌다 ⟹ 라벨만 붙고 분석기가 없는 배치를 못 잡았다.
  · 종전에는 **동결물 md5 를 전혀 안 봤다** ⟹ `gates_b3b.py` 가 결과 뒤에 바뀐 것을 못 잡았다.

사용:  python3 tmp/_audit/check_batch_artifacts.py [tmp/BATCH_DIR ...]
인자가 없으면 `tmp/` 아래 `results*.jsonl` 을 가진 디렉터리를 전부 검사한다.
종료코드 1 = D 또는 E 위반이 하나라도 있음(= 인용 불가 배치가 있음).
"""
import glob, json, os, sys

COLS = ['PREREG', 'gates', 'analyze', 'verdict.label', 'verdict.gates', 'FROZEN',
        'analyzer<results', 'results~verdict', 'md5 OK']


def check(d):
    has = lambda pat: bool(glob.glob(os.path.join(d, pat)))
    verdicts = glob.glob(os.path.join(d, 'verdict*.json'))
    lab = gts = False
    for f in verdicts:
        try:
            v = json.load(open(f))
        except Exception:
            continue
        if isinstance(v, dict):
            lab |= any(k in v for k in ('label', 'primary_label', 'verdict'))
            gts |= 'gates' in v
    an = sorted(glob.glob(os.path.join(d, 'analyze*.py')))
    rs = sorted(f for f in glob.glob(os.path.join(d, 'results*.jsonl'))
                if 'ABANDONED' not in os.path.basename(f))   # 폐기 선언된 결과는 짝을 요구하지 않는다
    order = bool(an and rs) and min(os.path.getmtime(a) for a in an) < max(os.path.getmtime(r) for r in rs)
    # H. 결과 파일마다 짝 verdict — `results_b2b.jsonl` ↔ `verdict_b2b.json` 식의 접미사 매칭
    def _sfx(f, pre):
        b = os.path.basename(f)
        b = b[len(pre):] if b.startswith(pre) else b
        return b.split('.')[0].lstrip('_')
    rs_sfx = {_sfx(f, 'results') for f in rs}
    vd_sfx = {_sfx(f, 'verdict') for f in verdicts}
    # 결과가 하나뿐이면 접미사가 달라도 짝으로 인정한다(`results.jsonl` ↔ `verdict_w17.json`).
    # 결과가 여럿이면 **각각** 짝이 있어야 한다 — 그게 RV-5 가 지적한 사각지대다.
    # 접미사 없는 `results.jsonl` 은 아무 verdict 과도 짝으로 인정한다(관례상 배치 본 결과).
    need = {x for x in rs_sfx if x}
    paired = bool(verdicts) if len(rs) <= 1 else (bool(verdicts) and need <= vd_sfx)

    # I. 동결 해시 대조
    md5_ok = True
    for ff in glob.glob(os.path.join(d, 'FROZEN*.txt')):
        for ln in open(ff):
            parts = ln.split()
            if len(parts) != 2 or len(parts[0]) != 32 or ln.startswith('#'):
                continue
            h, tgt = parts
            if not os.path.isabs(tgt):
                tgt = os.path.join(os.path.dirname(os.path.dirname(d)), tgt)
            if os.path.exists(tgt):
                import hashlib
                if hashlib.md5(open(tgt, 'rb').read()).hexdigest() != h:
                    md5_ok = False
    return [has('PREREG*.md'), has('gates*.py'), bool(an), lab, gts,
            has('FROZEN*.txt'), order, paired, md5_ok]


def main(argv):
    dirs = argv[1:] or sorted({os.path.dirname(f) for f in glob.glob('tmp/**/results*.jsonl', recursive=True)})
    bad = 0
    print(f"{'배치':44s} " + ' '.join(f'{c[:9]:>9s}' for c in COLS))
    for d in dirs:
        r = check(d)
        if not (r[3] and r[4] and r[7]):
            bad += 1
        print(f"{os.path.basename(d)[:44]:44s} " + ' '.join(f'{"OK" if x else "--":>9s}' for x in r))
    print(f"\n인용 불가(라벨 또는 게이트 증거 없음): {bad}/{len(dirs)}")
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
