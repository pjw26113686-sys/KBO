"""EV 엔진 단위 테스트."""

from kbo.ev.engine import ev, fair_odds, find_candidates


def test_ev_formula():
    # 모델 57%, 시장배당 1.90 → EV = 0.57*1.90 - 1 = +0.083
    assert abs(ev(0.57, 1.90) - 0.083) < 1e-9


def test_ev_negative_when_odds_short():
    # 모델 57%, 시장배당 1.65 → EV 음수 → 제외 대상
    assert ev(0.57, 1.65) < 0


def test_fair_odds_inverse():
    assert abs(fair_odds(0.5) - 2.0) < 1e-9
    assert abs(fair_odds(0.25) - 4.0) < 1e-9


def test_find_candidates_filters_by_threshold():
    probs = {"WIN": 0.57, "OU": 0.50}
    odds = {"WIN": 1.90, "OU": 1.90}   # WIN EV=+0.083, OU EV=-0.05
    cands = find_candidates(1, probs, odds, threshold=0.05, enabled=True)
    assert len(cands) == 1
    assert cands[0].market == "WIN"
    assert cands[0].ev > 0.05


def test_reliability_gate_blocks_when_disabled():
    """신뢰도 게이트: enabled=False 면 후보 없음(실전 신호 차단)."""
    probs = {"WIN": 0.9}
    odds = {"WIN": 3.0}   # 명백한 +EV 라도
    assert find_candidates(1, probs, odds, threshold=0.05, enabled=False) == []
