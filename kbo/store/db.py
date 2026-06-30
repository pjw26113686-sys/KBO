"""SQLite 초기화 및 헬퍼 (설계서 모듈 2, 구현 단계 1).

스키마(`schema.sql`)를 적용하고, 시뮬레이션/백테스트가 사용하는 가벼운
insert/query 헬퍼를 제공한다. 파일 DB와 인메모리(`:memory:`) 모두 지원한다.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable, Sequence

_SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def connect(path: str = ":memory:") -> sqlite3.Connection:
    """SQLite 연결을 연다. row_factory 를 sqlite3.Row 로 설정."""
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(path: str = ":memory:") -> sqlite3.Connection:
    """스키마를 적용한 새 연결을 반환한다."""
    conn = connect(path)
    conn.executescript(_SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.commit()
    return conn


def insert_many(
    conn: sqlite3.Connection,
    table: str,
    columns: Sequence[str],
    rows: Iterable[Sequence],
) -> None:
    """다중 행 삽입. columns 순서대로 값 바인딩."""
    placeholders = ", ".join("?" for _ in columns)
    col_sql = ", ".join(columns)
    sql = f"INSERT INTO {table} ({col_sql}) VALUES ({placeholders})"
    conn.executemany(sql, rows)
    conn.commit()


def query(conn: sqlite3.Connection, sql: str, params: Sequence = ()) -> list[sqlite3.Row]:
    """SELECT 결과를 리스트로 반환."""
    return list(conn.execute(sql, params).fetchall())
