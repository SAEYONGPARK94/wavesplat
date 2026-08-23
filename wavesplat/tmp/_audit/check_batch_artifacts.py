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

사용:  python3 tmp/_audit/check_batch_artifacts.py [tmp/BATCH_DIR ...]
인자가 없으면 `tmp/` 아래 `results*.jsonl` 을 가진 디렉터리를 전부 검사한다.
종료코드 1 = D 또는 E 위반이 하나라도 있음(= 인용 불가 배치가 있음).
"""
import glob, json, os, sys

COLS = ['PREREG', 'gates', 'analyze', 'verdict.label', 'verdict.gates', 'FROZEN', 'analyzer<results']


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
    rs = sorted(glob.glob(os.path.join(d, 'results*.jsonl')))
    order = bool(an and rs) and min(os.path.getmtime(a) for a in an) < max(os.path.getmtime(r) for r in rs)
    return [has('PREREG*.md'), has('gates*.py'), bool(an), lab, gts,
            has('FROZEN*.txt'), order]


def main(argv):
    dirs = argv[1:] or sorted({os.path.dirname(f) for f in glob.glob('tmp/**/results*.jsonl', recursive=True)})
    bad = 0
    print(f"{'배치':44s} " + ' '.join(f'{c[:9]:>9s}' for c in COLS))
    for d in dirs:
        r = check(d)
        if not (r[3] and r[4]):
            bad += 1
        print(f"{os.path.basename(d)[:44]:44s} " + ' '.join(f'{"OK" if x else "--":>9s}' for x in r))
    print(f"\n인용 불가(라벨 또는 게이트 증거 없음): {bad}/{len(dirs)}")
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
