"""수동 배당 EV 리포트 검증."""
import pytest

from kbo.ev import report


def _feature_game():
    return {
        "game_id": 1, "home": "A", "away": "B",
        "ou_line": 8.5, "handicap": -1.5,
        "features": {
            "home": {"offense_strength": 1.1, "opp_pitcher_suppression": 0.95},
            "away": {"offense_strength": 0.95, "opp_pitcher_suppression": 1.05},
        },
        "odds": {"HOME": 1.95, "AWAY": 2.0, "OVER": 1.95, "UNDER": 1.9},
    }


def test_selection_probs_complementary():
    p = report.selection_probs(_feature_game())
    assert p["HOME"] + p["AWAY"] == pytest.approx(1.0, abs=1e-9)
    assert p["OVER"] + p["UNDER"] == pytest.approx(1.0, abs=1e-9)
    assert p["HDC_HOME"] + p["HDC_AWAY"] == pytest.approx(1.0, abs=1e-9)


def test_probs_passthrough():
    g = {"game_id": 2, "probs": {"HOME": 0.6, "AWAY": 0.4}, "odds": {"HOME": 1.8}}
    p = report.selection_probs(g)
    assert p["HOME"] == 0.6


def test_evaluate_filters_by_ev():
    # 모델 60% × 배당 1.8 = EV +0.08 (후보), 모델 60% × 1.5 = -0.10 (탈락)
    g = {"game_id": 3, "probs": {"HOME": 0.6, "AWAY": 0.4},
         "odds": {"HOME": 1.8, "AWAY": 1.5}}
    cands = report.evaluate_game(g, threshold=0.05)
    sels = {c.selection for c in cands}
    assert "HOME" in sels and "AWAY" not in sels


def test_message_banner_toggle():
    games = [_feature_game()]
    unverified = report.build_message(games, gate_passed=False)
    verified = report.build_message(games, gate_passed=True)
    assert "참고용" in unverified
    assert "참고용" not in verified
    assert "⚾" in unverified
