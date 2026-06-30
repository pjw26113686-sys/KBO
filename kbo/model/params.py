"""모델 파라미터 (설계서 §4 말미: 보정계수 가중치는 과거 시즌으로 튜닝).

각 보정항은 '관측 지표 → 기대득점 승수'로 변환할 때의 민감도(탄력성)를 정한다.
값이 클수록 해당 변수가 λ(기대득점)를 더 강하게 움직인다.

설계서의 변수 영향력 순위(§2)를 반영해 기본값을 잡았다:
선발(★★★★★) > 불펜(★★★★) > 타선/구장(★★★) > 일정 피로(★★).
"""

from __future__ import annotations

from dataclasses import dataclass

from .. import config


@dataclass(frozen=True)
class ModelParams:
    """포아송 λ 추정용 보정계수.

    각 `*_elasticity` 는 표준화된 지표 1단위 변화가 λ 로그스케일에 주는 영향.
    `home_advantage` 는 홈팀 λ 에 곱하는 승수.
    """

    league_avg_runs: float = config.LEAGUE_AVG_RUNS_PER_TEAM
    home_advantage: float = config.HOME_ADVANTAGE

    # 상대 선발의 억제력(FIP 표준화). 양수: 좋은 선발일수록 상대 득점↓.
    starter_elasticity: float = 0.55
    # 자기 타선 강도(wOBA 표준화). 양수: 강한 타선일수록 득점↑.
    offense_elasticity: float = 0.35
    # 구장 파크팩터(이미 승수 형태, 1.0 중심)는 그대로 곱한다.
    # 불펜 피로(연투/누적 투구수 표준화). 양수: 피로할수록 상대 득점↑.
    bullpen_fatigue_elasticity: float = 0.22
    # 일정 피로(원정연속/이동/연장 등 표준화). 양수: 피로할수록 자기 득점↓.
    schedule_fatigue_elasticity: float = 0.12

    # 음이항 사용 여부 및 분산계수(분산 과대 대비, 설계서 §4 Step2).
    use_nbinom: bool = False
    nb_dispersion: float = config.NB_DISPERSION
