"""전역 상수 및 기본 설정.

설계서 §4(포아송 베이스라인)의 리그 평균 득점·홈 어드밴티지 등을 모은다.
실제 KBO 시즌 평균에 가깝게 잡은 출발값이며, 파라미터 튜닝(설계서 §4 말미)의
초기값으로 사용한다.
"""

from __future__ import annotations

# KBO 한 팀의 경기당 평균 득점(최근 시즌대 ~4.5~5.5점 범위). 베이스라인 λ 기준.
LEAGUE_AVG_RUNS_PER_TEAM = 4.8

# 홈 어드밴티지: 홈팀 기대득점에 곱하는 승수(약 +3~5%).
HOME_ADVANTAGE = 1.04

# 음이항 분포로 교체할 때의 분산 과대(over-dispersion) 계수.
# variance = mean * NB_DISPERSION (>1 이면 포아송보다 꼬리가 두껍다).
NB_DISPERSION = 1.25

# 득점 분포를 자를 최대 점수(이 이상은 누적 꼬리로 처리).
MAX_RUNS = 25

# EV 후보 채택 임계값(설계서 §5: 예시 0.05).
DEFAULT_EV_THRESHOLD = 0.05

# 시뮬용 북메이커 마진(overround). 설계서 §0: 북메이커 마진 4~7%.
DEFAULT_MARGIN = 0.05
