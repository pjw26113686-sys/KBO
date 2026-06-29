"""합성 시장(배당) 생성.

시장도 진실 λ 의 노이즈 관측치(σ_market)로 확률을 추정한 뒤, 마진(overround)을
얹어 decimal odds 를 만든다. σ_model vs σ_market 가 (a)/(b) 시나리오를 통제한다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..model.poisson import market_probabilities


@dataclass
class MarketLines:
    """한 경기에 대해 시장이 내건 선과 양면 배당."""
    ou_line: float
    handicap: float
    odds: dict[str, float]   # selection -> decimal odds


def _nearest_half(x: float) -> float:
    """오버언더 선을 .5 단위로(푸시 방지)."""
    return round(x - 0.5) + 0.5


def two_way_odds(p: float, margin: float) -> tuple[float, float]:
    """양면 마켓(확률 p / 1-p)에 overround 를 대칭 적용한 decimal odds.

    implied_i = p_i * (1 + margin) → odds_i = 1 / implied_i.
    합산 implied = (1 + margin) > 1 → 북메이커 마진.
    """
    p = min(max(p, 1e-6), 1 - 1e-6)
    o1 = 1.0 / (p * (1.0 + margin))
    o2 = 1.0 / ((1.0 - p) * (1.0 + margin))
    return o1, o2


def make_market(
    lam_home_obs: float,
    lam_away_obs: float,
    margin: float = 0.05,
    handicap: float = -1.5,
) -> MarketLines:
    """시장 관측 λ 로부터 WIN/OU/HDC 양면 배당을 만든다.

    OU 선은 시장 추정 총득점 근처(.5 단위)로 설정 → 시장 기준 거의 공정.
    엣지는 모델이 시장과 다른 확률을 줄 때만 발생한다.
    """
    ou_line = _nearest_half(lam_home_obs + lam_away_obs)
    mp = market_probabilities(lam_home_obs, lam_away_obs, ou_line=ou_line, handicap=handicap)

    odds: dict[str, float] = {}

    # 승패(무승부 제외, 홈/원정으로 정규화한 양면)
    p_home = mp.p_home_win / (mp.p_home_win + mp.p_away_win)
    o_home, o_away = two_way_odds(p_home, margin)
    odds["HOME"] = o_home
    odds["AWAY"] = o_away

    # 오버언더(푸시 제외 정규화)
    p_over = mp.p_over / (mp.p_over + mp.p_under)
    o_over, o_under = two_way_odds(p_over, margin)
    odds["OVER"] = o_over
    odds["UNDER"] = o_under

    # 핸디캡(홈 커버 vs 미커버)
    o_cover, o_nocover = two_way_odds(mp.p_home_cover, margin)
    odds["HDC_HOME"] = o_cover
    odds["HDC_AWAY"] = o_nocover

    return MarketLines(ou_line=ou_line, handicap=handicap, odds=odds)
