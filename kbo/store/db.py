"""SQLite 연결 및 적재 헬퍼.

시뮬레이션 산출물(predictions/results/odds)을 실제 스키마에 적재해
스키마가 동작함을 검증하는 용도로도 쓰인다.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

_SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def connect(db_path: str = ":memory:") -> sqlite3.Connection:
    """SQLite 연결을 반환한다. 외래키 제약을 켠다."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """schema.sql을 실행해 테이블을 (없으면) 생성한다. idempotent."""
    conn.executescript(_SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.commit()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# 적재 헬퍼
# ---------------------------------------------------------------------------
def insert_teams(conn: sqlite3.Connection, teams: Iterable[Sequence]) -> None:
    """teams: (team_id, name, park_factor_run, park_factor_hr) 시퀀스."""
    conn.executemany(
        "INSERT OR REPLACE INTO teams "
        "(team_id, name, park_factor_run, park_factor_hr) VALUES (?, ?, ?, ?)",
        teams,
    )
    conn.commit()


def insert_game(conn: sqlite3.Connection, game: dict) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO games "
        "(game_id, date, start_time, home_id, away_id, home_sp_id, away_sp_id, "
        " stadium, status) VALUES "
        "(:game_id, :date, :start_time, :home_id, :away_id, :home_sp_id, "
        " :away_sp_id, :stadium, :status)",
        game,
    )


def insert_odds(conn: sqlite3.Connection, game_id: int, market: str,
                line: float | None, odds_value: float) -> None:
    conn.execute(
        "INSERT INTO odds (game_id, market, line, odds_value, captured_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (game_id, market, line, odds_value, _now()),
    )


def insert_prediction(conn: sqlite3.Connection, game_id: int, market: str,
                      model_prob: float, ev: float) -> None:
    conn.execute(
        "INSERT INTO predictions (game_id, market, model_prob, ev, captured_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (game_id, market, model_prob, ev, _now()),
    )


def insert_result(conn: sqlite3.Connection, game_id: int,
                  home_score: int, away_score: int) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO results (game_id, home_score, away_score) "
        "VALUES (?, ?, ?)",
        (game_id, home_score, away_score),
    )
