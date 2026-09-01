# 이동한 그림 파일 (2026-09-02)

⛔**삭제가 아니라 이동이다.** 이 파일들은 `paper/figures/` 에 있었는데
**원고가 한 장도 참조하지 않는다**(`\includegraphics` 0회).

사유: 이들을 싣던 절·표·그림 환경이 2026-09-01~02 처분에서
`../retracted_tables_2026-09-02.tex` 와 `../retracted_results_100ep.tex` 로 옮겨졌는데,
**파일만 원고 디렉터리에 남았다.** 원고 디렉터리를 통째로 묶는 사람이 그것을 함께 가져가고,
그중 넷은 **인용금지 수치를 그림 안에 담고 있다**.

⚠**공시**: 이 이동으로 인용금지 래칫이 **4 내려간다**.
⛔**내용 개선이 아니라 정리다** --- 금지 수치는 하나도 지워지지 않았고 이 그림들 안에 그대로 있다.

현재 원고가 싣는 그림은 셋뿐이고 **전부 생성기가 있다**:
`fig_pipeline_deployed`(`scripts/gen_fig_pipeline.py`) ·
`fig_results`(`scripts/gen_fig_results.py`) ·
`fig_qualitative`(`scripts/gen_fig_qualitative.py`, 출처 json 동반).
