"""EV 엔진 수식 검증 (설계서 5장 예시 숫자 재현)."""
import math

import pytest

from kbo.ev import engine


def test_fair_odds():
    assert engine.fair_odds(0.5) == pytest.approx(2.0)
    assert engine.fair_odds(0.57) == pytest.approx(1.754, abs=1e-3)


def test_ev_positive_example():
    # 설계서 5장: 모델 57%, 시장배당 1.90 → EV = 0.57*1.90-1 = +0.083
    assert engine.ev(0.57, 1.90) == pytest.approx(0.083, abs=1e-3)


def test_ev_negative_example():
    # 시장배당 1.65 → EV = 0.57*1.65-1 = -0.0595 → 제외
    assert engine.ev(0.57, 1.65) == pytest.approx(-0.0595, abs=1e-3)


def test_filter_threshold():
    cands = [
        engine.evaluate(1, "WIN", "HOME", 0.57, 1.90),   # ev +0.083
        engine.evaluate(2, "WIN", "HOME", 0.57, 1.65),   # ev -0.06
        engine.evaluate(3, "OU", "OVER", 0.55, 1.95),    # ev +0.0725
    ]
    kept = engine.filter_candidates(cands, threshold=0.05)
    assert [c.game_id for c in kept] == [1, 3]


def test_invalid_prob():
    with pytest.raises(ValueError):
        engine.fair_odds(0.0)
    with pytest.raises(ValueError):
        engine.fair_odds(1.5)
