"""EV 엔진 + 시뮬 시장 단위 테스트 (설계서 §5)."""

import math

from kbo.ev import market
from kbo.ev.engine import ev_of_bet, fair_odds, filter_candidates


def test_ev_example_positive():
    # 설계서 §5 예시: 모델 57%, 시장배당 1.90 → EV +0.083 (후보)
    assert math.isclose(ev_of_bet(0.57, 1.90), 0.083, abs_tol=1e-3)


def test_ev_example_negative():
    # 설계서 §5 예시: 모델 57%, 시장배당 1.65 → EV −0.06 (제외)
    assert math.isclose(ev_of_bet(0.57, 1.65), -0.0595, abs_tol=1e-3)


def test_fair_odds_inverse():
    assert math.isclose(fair_odds(0.57), 1.0 / 0.57, rel_tol=1e-9)


def test_filter_threshold():
    rows = [
        {"game_id": 1, "market": "WIN", "line": None, "model_prob": 0.57, "decimal_odds": 1.90},
        {"game_id": 2, "market": "OU", "line": 8.5, "model_prob": 0.57, "decimal_odds": 1.65},
    ]
    cands = filter_candidates(rows, threshold=0.05)
    assert len(cands) == 1
    assert cands[0].game_id == 1
    assert cands[0].ev > 0.05


def test_market_overround_sums_above_one():
    # 양 사이드 implied 확률 합 = 1 + margin
    p = 0.6
    margin = 0.05
    odds_yes = market.decimal_odds(p, margin)
    odds_no = market.decimal_odds(1 - p, margin)
    implied = 1 / odds_yes + 1 / odds_no
    assert math.isclose(implied, 1 + margin, rel_tol=1e-9)


def test_fair_market_bet_has_negative_ev():
    # 시장이 진짜 확률을 정확히 알면(노이즈 0), 그 배당에 베팅하면 EV = −margin/(1+margin) < 0
    p = 0.55
    margin = 0.05
    odds = market.decimal_odds(p, margin)
    assert ev_of_bet(p, odds) < 0
