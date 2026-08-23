# RV-1 재현 스크립트 (추정 런 0회)

`RV1_adversary.md` 의 모든 수치를 내는 스크립트. 전부 numpy/scipy + 자료 로더만 쓴다.

| 파일 | 무엇을 내는가 | 보고서 절 |
|---|---|---|
| `zf.py` | ★**레코드별 참 제로필드**(`zf.json`) · σ · npx · 상자 · λ_p | §2 |
| `bias.py` | demean 창 불일치 편향 `|B|/σ` + **잡음 불변성**(P·sg·kp·GT 비트동일) | §2 끝 · §4 |
| `lam.py` | λ_p 의 **상자 기본파 포화** + APERTURE_AXIS 상관 재현/재정규화 | §3 |
| `support.py` | 질의점→최근접 관측 거리 (팬 vs 고정 관측소) | §1c |
| `defect2.py` | 12 레코드 원자료 결함 스캔 (클램프·결측·포화·중복) | §5 |

실행: `python3 <파일>` (cwd 무관, 스크립트가 `os.chdir(ROOT)` 한다).
`zf.py` 를 먼저 돌려 `zf.json` 을 만든 뒤 `lam.py` 를 돌린다.
집계 재계산은 `RV1_adversary.md §10` 규약대로 `gates_b2.sign_test_one_sided` 를 그대로 쓴다.
