"""수동 배당 입력 → EV 분석 테스트."""

from pathlib import Path

from kbo.ev.manual import analyze_game, load_games

SAMPLE = Path(__file__).resolve().parent.parent / "examples" / "games_sample.json"


def _game(odds: dict) -> dict:
    """중립 피처(홈 어드밴티지로 홈 약우세)의 테스트 게임."""
    return {
        "game_id": 99,
        "home": "HOME", "away": "AWAY",
        "features": {
            "home_woba": 0.330, "away_woba": 0.330,
            "home_sp_fip": 4.20, "away_sp_fip": 4.20,
            "park_factor_run": 1.0,
        },
        "odds": odds,
    }


def test_positive_ev_becomes_candidate():
    # 홈 승률 ~0.52, 배당 2.2 → EV = 0.52*2.2-1 ≈ +0.14 → 후보
    res = analyze_game(_game({"WIN_HOME": 2.2}), threshold=0.05)
    markets = {c.market for c in res["candidates"]}
    assert "WIN_HOME" in markets


def test_negative_ev_filtered_out():
    # 홈 승률 ~0.52, 배당 1.5 → EV = 0.52*1.5-1 ≈ -0.22 → 제외
    res = analyze_game(_game({"WIN_HOME": 1.5}), threshold=0.05)
    assert res["candidates"] == []


def test_message_contains_teams():
    res = analyze_game(_game({"WIN_HOME": 2.2}))
    assert "HOME" in res["message"] and "AWAY" in res["message"]


def test_sample_file_loads_and_runs():
    games = load_games(SAMPLE)
    assert len(games) == 2
    # 1경기(불펜 피로 blind spot): OU_OVER 후보 발생, 2경기(효율적): 후보 없음.
    r1 = analyze_game(games[0])
    r2 = analyze_game(games[1])
    assert any(c.market == "OU_OVER" for c in r1["candidates"])
    assert r2["candidates"] == []
