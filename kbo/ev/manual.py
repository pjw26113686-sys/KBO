"""수동 배당 입력 → 분석 → 디스코드 메시지 (설계서 §5 step6 / §7).

자동 스크래핑은 봇 차단·법적 회색지대로 보류한다(설계서 §5).
대신 발송 직전 배당 숫자 몇 개를 JSON 으로 붙여넣어 분석한다.

입력 JSON 스키마(games_sample.json 참고):
{
  "games": [
    {
      "game_id": 1, "home": "...", "away": "...",
      "start_time": "18:30", "stadium": "...",
      "home_sp": "투수X (FIP 3.2)", "away_sp": "투수Y (FIP 4.8)",
      "features": { home_woba, away_woba, home_sp_fip, away_sp_fip,
                    home_bullpen_fatigue, ... , park_factor_run },
      "ou_line": 9.5, "handicap": -1.5,
      "odds": { "WIN_HOME": 1.95, "WIN_AWAY": 1.95,
                "OU_OVER": 1.90, "OU_UNDER": 1.90,
                "HDC_HOME": 2.00, "HDC_AWAY": 1.80 },
      "note": "..."
    }
  ]
}
"""

from __future__ import annotations

import json
from pathlib import Path

from .. import config
from ..model.poisson import GameFeatures, predict_markets
from ..scheduler.discord import format_message
from .engine import find_candidates, two_sided_probs


def load_games(path: str | Path) -> list[dict]:
    """입력 JSON 을 읽어 게임 dict 리스트를 반환한다."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return raw["games"]


def _features(g: dict) -> GameFeatures:
    """게임 dict 의 features 를 GameFeatures 로 변환(누락 필드는 기본값)."""
    f = g["features"]
    return GameFeatures(
        home_woba=f["home_woba"],
        away_woba=f["away_woba"],
        home_sp_fip=f["home_sp_fip"],
        away_sp_fip=f["away_sp_fip"],
        home_bullpen_fatigue=f.get("home_bullpen_fatigue", 0.0),
        away_bullpen_fatigue=f.get("away_bullpen_fatigue", 0.0),
        home_schedule_fatigue=f.get("home_schedule_fatigue", 0.0),
        away_schedule_fatigue=f.get("away_schedule_fatigue", 0.0),
        park_factor_run=f.get("park_factor_run", 1.0),
    )


def analyze_game(g: dict, threshold: float | None = None) -> dict:
    """한 경기를 분석해 예측·EV 후보·디스코드 메시지를 만든다."""
    threshold = config.EV_THRESHOLD if threshold is None else threshold
    ou_line = g.get("ou_line", config.DEFAULT_OU_LINE)
    handicap = g.get("handicap", config.DEFAULT_HANDICAP)

    pred = predict_markets(_features(g), ou_line, handicap)
    probs = two_sided_probs(pred)
    odds = g.get("odds", {})

    lines = {
        "WIN_HOME": None, "WIN_AWAY": None,
        "OU_OVER": ou_line, "OU_UNDER": ou_line,
        "HDC_HOME": handicap, "HDC_AWAY": handicap,
    }
    candidates = find_candidates(
        g.get("game_id", 0), probs, odds, lines=lines, threshold=threshold,
    )

    message = format_message(
        home=g.get("home", "HOME"), away=g.get("away", "AWAY"),
        start_time=g.get("start_time", "--:--"),
        stadium=g.get("stadium", ""),
        home_sp=g.get("home_sp", "?"), away_sp=g.get("away_sp", "?"),
        p_win=pred["WIN"], p_over=pred["OU"], ou_line=ou_line,
        p_handicap=pred["HDC"], handicap=handicap,
        candidates=candidates, note=g.get("note", ""),
    )
    return {"pred": pred, "candidates": candidates, "message": message}
