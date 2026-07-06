# 데이터 요청 이메일 초안 (C-13, 사용자 발송용)
(2026-07-06. 대상 후보: ①Scripps AirSea Lab (L. Grare — shipborne/
platform LiDAR wave data) ②OE 2023 "experimental study on monitoring
wave profiles with LiDAR" 저자 ③OE 2025 "Wave characteristics analysis
by LiDAR point cloud" 저자. 목적: 실기 LiDAR 해수면 원시 점군 —
리턴 통계(인접성·raydrop)와 센서-실물 검증용.)

---
Subject: Request for LiDAR sea-surface point-cloud data for a wave
encounter-forecasting study

Dear Dr. [NAME],

I am a Ph.D. student at [INSTITUTION] working on real-time
phase-resolved wave encounter forecasting from forward-looking marine
LiDAR, for feedforward control of foiling vessels. We validate our
estimator on the open stereo dataset of Guimarães et al. (2020) and on
spec-faithful simulation of a commercial 1550-nm LiDAR, and we have
been careful to separate what the simulation can and cannot establish.

The one element we cannot obtain from stereo data is the statistics of
actual LiDAR returns from the sea surface at grazing incidence —
return density, dropout behaviour, and the spatial adjacency of
surviving returns — which set the information available for
slope-resolved estimation. Your [PAPER, YEAR] presents exactly such
measurements.

Would you be willing to share a sample of raw (or lightly processed)
point-cloud data from that campaign — even a few minutes of returns
with timestamps and geometry would be highly valuable? We would of
course cite the data appropriately, follow any licensing terms you
specify, and share our derived return-statistics analysis back with
you before publication.

Thank you for considering this request.

Best regards,
[USER NAME / affiliation / link to preprint when available]

---
비고: FloWave(에든버러) 수조 게이지 어레이는 datashare 컬렉션에서
개별 데이터셋 확인 필요(landing page엔 목록 없음; flowave@ed.ac.uk)
— 관측 기하가 달라(고정점 어레이) 보조 일반화용으로만.
