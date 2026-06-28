"""N경기 시뮬레이션 오케스트레이터.

각 경기마다:
  generator → 진짜 λ·실제 스코어·시장 관측 → model 예측확률·예측구간
            → 양면(home/away, over/under, cover/not) 시장배당 산출
            → EV 계산 → 후보 표시 → (옵션) SQLite 적재 → 레코드 수집
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .. import config
from ..ev.engine import ev as compute_ev
from ..ev.engine import two_sided_probs as _two_sided
from ..model.poisson import predict_markets
from ..store import db as store_db
from . import generator as gen


@dataclass
class OutcomeRecord:
    market: str           # WIN_HOME / WIN_AWAY / OU_OVER / OU_UNDER / HDC_HOME / HDC_AWAY
    model_prob: float
    market_prob: float
    odds: float           # decimal odds (vig 포함)
    ev: float
    occurred: bool        # 실제로 적중했는가
    is_candidate: bool    # EV > threshold (베팅 후보)


@dataclass
class GameRecord:
    game_id: int
    home_score: int
    away_score: int
    total: int
    diff: int
    interval_total: tuple[int, int]
    interval_diff: tuple[int, int]
    interval_level: float
    home_win_prob: float
    home_won: bool
    outcomes: list[OutcomeRecord] = field(default_factory=list)


@dataclass
class SimResult:
    records: list[GameRecord]
    n_games: int
    seed: int
    ou_line: float
    handicap: float


def _odds_from_prob(prob: float, vig: float) -> float:
    """시장 확률 → 마진 포함 decimal odds. 양면 합 implied = (1+vig)."""
    prob = min(max(prob, 1e-6), 1.0 - 1e-6)
    return 1.0 / (prob * (1.0 + vig))


def _evaluate(outcome: str, home: int, away: int,
              ou_line: float, handicap: float) -> bool:
    """실제 스코어로 각 outcome 적중 여부 판정. 라인은 .5 단위라 push 없음."""
    total = home + away
    diff = home - away
    return {
        "WIN_HOME": home > away,
        "WIN_AWAY": away > home,
        "OU_OVER": total > ou_line,
        "OU_UNDER": total < ou_line,
        "HDC_HOME": diff > handicap,
        "HDC_AWAY": diff < handicap,
    }[outcome]


def run_simulation(
    n_games: int = 1000,
    seed: int = 42,
    ou_line: float | None = None,
    handicap: float | None = None,
    db_path: str | None = None,
    threshold: float | None = None,
) -> SimResult:
    """N경기를 시뮬레이션하고 검증용 레코드를 반환한다.

    db_path 지정 시 teams/games/odds/predictions/results 를 SQLite 에 적재한다.
    """
    ou_line = config.DEFAULT_OU_LINE if ou_line is None else ou_line
    handicap = config.DEFAULT_HANDICAP if handicap is None else handicap
    threshold = config.EV_THRESHOLD if threshold is None else threshold
    vig = config.DEFAULT_VIG

    rng = np.random.default_rng(seed)
    league = gen.make_league(rng)

    conn = None
    if db_path is not None:
        conn = store_db.connect(db_path)
        store_db.init_db(conn)
        store_db.insert_teams(
            conn,
            [(t.team_id, t.name, t.park_factor_run, 1.0) for t in league],
        )

    records: list[GameRecord] = []
    for gid in range(1, n_games + 1):
        game = gen.make_game(gid, league, rng)

        # 진짜 λ → 실제 스코어
        true_lh, true_la = gen.true_lambdas(game)
        home, away = gen.simulate_result(true_lh, true_la, rng)

        # 모델 예측 (추정오차 포함 입력)
        model_pred = predict_markets(gen.model_view(game, rng), ou_line, handicap)
        model_probs = _two_sided(model_pred)

        # 시장 예측 (피로 blind spot) → 배당
        market_pred = predict_markets(gen.market_view(game, rng), ou_line, handicap)
        market_probs = _two_sided(market_pred)
        odds = {k: _odds_from_prob(p, vig) for k, p in market_probs.items()}

        outcomes: list[OutcomeRecord] = []
        for mk in model_probs:
            e = compute_ev(model_probs[mk], odds[mk])
            occurred = _evaluate(mk, home, away, ou_line, handicap)
            is_cand = config.EV_ENGINE_ENABLED and e > threshold
            outcomes.append(
                OutcomeRecord(
                    market=mk,
                    model_prob=model_probs[mk],
                    market_prob=market_probs[mk],
                    odds=odds[mk],
                    ev=e,
                    occurred=occurred,
                    is_candidate=is_cand,
                )
            )

        rec = GameRecord(
            game_id=gid,
            home_score=home,
            away_score=away,
            total=home + away,
            diff=home - away,
            interval_total=model_pred["interval"]["total"],
            interval_diff=model_pred["interval"]["diff"],
            interval_level=model_pred["interval"]["level"],
            home_win_prob=model_probs["WIN_HOME"],
            home_won=home > away,
            outcomes=outcomes,
        )
        records.append(rec)

        if conn is not None:
            _persist(conn, game, rec, ou_line, handicap)

    if conn is not None:
        conn.commit()
        conn.close()

    return SimResult(records, n_games, seed, ou_line, handicap)


def _persist(conn, game: gen.Game, rec: GameRecord,
             ou_line: float, handicap: float) -> None:
    store_db.insert_game(conn, {
        "game_id": game.game_id,
        "date": "2026-06-28",
        "start_time": "18:30",
        "home_id": game.home.team_id,
        "away_id": game.away.team_id,
        "home_sp_id": None,
        "away_sp_id": None,
        "stadium": game.home.name,
        "status": "final",
    })
    store_db.insert_result(conn, game.game_id, rec.home_score, rec.away_score)
    line_for = {"WIN_HOME": None, "WIN_AWAY": None,
                "OU_OVER": ou_line, "OU_UNDER": ou_line,
                "HDC_HOME": handicap, "HDC_AWAY": handicap}
    for o in rec.outcomes:
        store_db.insert_odds(conn, game.game_id, o.market,
                             line_for[o.market], o.odds)
        store_db.insert_prediction(conn, game.game_id, o.market,
                                   o.model_prob, o.ev)
