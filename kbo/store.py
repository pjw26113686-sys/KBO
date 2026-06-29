"""DB 연결 및 적재/조회 헬퍼.

시뮬레이션과 (2차) 실데이터 파이프라인이 동일한 적재 경로를 쓰도록 한다.
백테스트(metrics)는 predictions/odds/results 를 여기서 읽는다.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Iterable, Iterator, Sequence

from . import schema


@contextmanager
def connect(path: str = ":memory:") -> Iterator[sqlite3.Connection]:
    """초기화된 SQLite 커넥션 컨텍스트매니저. 기본은 인메모리(시뮬용)."""
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        schema.init_db(conn)
        yield conn
    finally:
        conn.close()


def insert_result(conn: sqlite3.Connection, game_id: int, home_score: int, away_score: int) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO results (game_id, home_score, away_score) VALUES (?, ?, ?)",
        (game_id, home_score, away_score),
    )


def insert_odds(conn: sqlite3.Connection, rows: Iterable[Sequence]) -> None:
    """odds 행 일괄 삽입. row = (game_id, market, line, selection, odds_value, captured_at)."""
    conn.executemany(
        "INSERT INTO odds (game_id, market, line, selection, odds_value, captured_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )


def insert_predictions(conn: sqlite3.Connection, rows: Iterable[Sequence]) -> None:
    """predictions 행 일괄 삽입. row = (game_id, market, selection, line, model_prob, ev, captured_at)."""
    conn.executemany(
        "INSERT INTO predictions (game_id, market, selection, line, model_prob, ev, captured_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows,
    )


def fetch_predictions(conn: sqlite3.Connection, market: str | None = None) -> list[sqlite3.Row]:
    if market is None:
        return list(conn.execute("SELECT * FROM predictions"))
    return list(conn.execute("SELECT * FROM predictions WHERE market = ?", (market,)))


def fetch_results(conn: sqlite3.Connection) -> dict[int, tuple[int, int]]:
    return {
        r["game_id"]: (r["home_score"], r["away_score"])
        for r in conn.execute("SELECT game_id, home_score, away_score FROM results")
    }
