"""포아송 결합 베이스라인 모델 (설계서 4장).

Step 1: 팀별 기대득점 λ 를 설명 가능한 곱셈 구조로 추정.
Step 2: 각 팀 득점 ~ Poisson(λ)  (과대분산 시 음이항 nbinom).
Step 3: 두 분포의 결합으로 승패 / 오버언더 / 핸디캡 확률 산출.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import nbinom, poisson

# 리그 평균 팀 득점(경기당). 합성/실데이터 모두 여기를 기준으로 보정계수를 곱한다.
LEAGUE_AVG_RUNS = 4.6
# 홈 어드밴티지(득점 곱 보정). 1.0보다 크면 홈 득점 상향.
HOME_ADVANTAGE = 1.03


@dataclass
class TeamMatchupFeatures:
    """한 팀의 한 경기 공격 기대치를 만드는 입력(설계서 2장 변수 순위 반영).

    각 값은 '리그 평균 대비 배수'로 정규화된 보정계수다(1.0 = 평균).
    pitcher_suppression 은 *상대 선발*의 억제력 → 작을수록 우리 팀 득점이 줄어든다.
    """

    offense_strength: float = 1.0       # wOBA/최근 OPS 기반 자팀 타선 강도
    opp_pitcher_suppression: float = 1.0  # 상대 선발 FIP 기반 억제력 (1=평균)
    park_factor: float = 1.0            # 구장 득점 환경
    bullpen_fatigue: float = 1.0        # 상대 불펜 피로 → >1 이면 우리 득점 상향
    schedule_fatigue: float = 1.0       # 자팀 일정 피로 → <1 이면 득점 하향
    is_home: bool = False


def expected_runs(f: TeamMatchupFeatures, league_avg: float = LEAGUE_AVG_RUNS) -> float:
    """단일 팀의 기대득점 λ (설계서 4장 Step1)."""
    lam = (
        league_avg
        * f.offense_strength
        * f.opp_pitcher_suppression
        * f.park_factor
        * f.bullpen_fatigue
        * f.schedule_fatigue
    )
    if f.is_home:
        lam *= HOME_ADVANTAGE
    return max(lam, 1e-6)


def estimate_lambda(
    home: TeamMatchupFeatures,
    away: TeamMatchupFeatures,
    league_avg: float = LEAGUE_AVG_RUNS,
) -> tuple[float, float]:
    """홈/원정 기대득점 (λ_home, λ_away)."""
    home = TeamMatchupFeatures(**{**home.__dict__, "is_home": True})
    away = TeamMatchupFeatures(**{**away.__dict__, "is_home": False})
    return expected_runs(home, league_avg), expected_runs(away, league_avg)


def score_distribution(
    lam: float, max_runs: int = 20, dist: str = "poisson", overdispersion: float = 0.25
) -> np.ndarray:
    """0..max_runs 득점 확률벡터. 합=1로 정규화.

    dist="nbinom" 이면 과대분산 모델. overdispersion 은 분산/평균 초과분을 키운다
    (Var = lam * (1 + overdispersion * lam)).
    """
    k = np.arange(0, max_runs + 1)
    if dist == "poisson":
        pmf = poisson.pmf(k, lam)
    elif dist == "nbinom":
        # Var = lam + overdispersion*lam^2 → r, p 파라미터로 변환
        var = lam + overdispersion * lam * lam
        p = lam / var
        r = lam * lam / (var - lam)
        pmf = nbinom.pmf(k, r, p)
    else:
        raise ValueError(f"unknown dist: {dist}")
    s = pmf.sum()
    return pmf / s if s > 0 else pmf


def _joint(
    lam_home: float, lam_away: float, max_runs: int = 20, dist: str = "poisson"
) -> np.ndarray:
    """결합 확률표 J[h, a] = P(home=h, away=a). 두 팀 득점 독립 가정."""
    ph = score_distribution(lam_home, max_runs, dist)
    pa = score_distribution(lam_away, max_runs, dist)
    return np.outer(ph, pa)


@dataclass
class MarketProbabilities:
    p_home_win: float
    p_away_win: float
    p_draw: float
    p_over: float           # P(total > ou_line)
    p_under: float
    p_home_cover: float     # P(home - away > handicap)
    ou_line: float
    handicap: float
    lam_home: float = field(default=0.0)
    lam_away: float = field(default=0.0)


def market_probabilities(
    lam_home: float,
    lam_away: float,
    ou_line: float = 8.5,
    handicap: float = -1.5,
    max_runs: int = 20,
    dist: str = "poisson",
) -> MarketProbabilities:
    """세 마켓(승패/오버언더/핸디캡) 확률을 결합표에서 파생 (설계서 4장 Step3).

    handicap 은 홈 기준. 예: -1.5 → 홈이 2점차 이상 이겨야 커버.
    """
    j = _joint(lam_home, lam_away, max_runs, dist)
    h_idx = np.arange(j.shape[0])[:, None]
    a_idx = np.arange(j.shape[1])[None, :]

    diff = h_idx - a_idx           # home - away
    total = h_idx + a_idx          # 총득점

    p_home_win = float(j[diff > 0].sum())
    p_away_win = float(j[diff < 0].sum())
    p_draw = float(j[diff == 0].sum())

    p_over = float(j[total > ou_line].sum())
    p_under = float(j[total < ou_line].sum())  # total==ou_line(정수선) 은 push로 양쪽 제외

    p_home_cover = float(j[diff > handicap].sum())

    return MarketProbabilities(
        p_home_win=p_home_win,
        p_away_win=p_away_win,
        p_draw=p_draw,
        p_over=p_over,
        p_under=p_under,
        p_home_cover=p_home_cover,
        ou_line=ou_line,
        handicap=handicap,
        lam_home=lam_home,
        lam_away=lam_away,
    )
