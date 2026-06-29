"""SQLite 스키마 정의 (설계서 3장).

시뮬레이션 결과와 (2차) 실데이터가 동일한 테이블 구조를 공유하도록 한다.
백테스트는 항상 이 스키마(특히 predictions / results / odds)에서 읽는다.
"""
from __future__ import annotations

import sqlite3

# 설계서 3장 스키마 초안을 그대로 옮긴 DDL. 멱등(IF NOT EXISTS) 생성.
DDL = """
CREATE TABLE IF NOT EXISTS teams (
    team_id          INTEGER PRIMARY KEY,
    name             TEXT NOT NULL,
    park_factor_run  REAL DEFAULT 1.0,
    park_factor_hr   REAL DEFAULT 1.0
);

CREATE TABLE IF NOT EXISTS players (
    player_id  INTEGER PRIMARY KEY,
    name       TEXT NOT NULL,
    team_id    INTEGER,
    hand       TEXT,                      -- L / R / S
    FOREIGN KEY (team_id) REFERENCES teams(team_id)
);

CREATE TABLE IF NOT EXISTS pitcher_game (
    game_id     INTEGER,
    player_id   INTEGER,
    ip          REAL,
    pitches     INTEGER,
    fip_recent  REAL,
    rest_days   INTEGER,
    vs_team_id  INTEGER,
    PRIMARY KEY (game_id, player_id)
);

CREATE TABLE IF NOT EXISTS bullpen_log (
    date         TEXT,
    player_id    INTEGER,
    pitches      INTEGER,
    is_back2back INTEGER,                 -- 0/1
    PRIMARY KEY (date, player_id)
);

CREATE TABLE IF NOT EXISTS batter_form (
    player_id  INTEGER,
    date       TEXT,
    woba       REAL,
    ops_14d    REAL,
    vs_LHP     REAL,
    vs_RHP     REAL,
    PRIMARY KEY (player_id, date)
);

CREATE TABLE IF NOT EXISTS games (
    game_id     INTEGER PRIMARY KEY,
    date        TEXT,
    start_time  TEXT,
    home_id     INTEGER,
    away_id     INTEGER,
    home_sp_id  INTEGER,
    away_sp_id  INTEGER,
    stadium     TEXT,
    status      TEXT DEFAULT 'scheduled'  -- scheduled / final / postponed
);

CREATE TABLE IF NOT EXISTS schedule_load (
    team_id     INTEGER,
    date        TEXT,
    road_streak INTEGER,
    travel_km   REAL,
    prev_extra  INTEGER,                  -- 직전 연장 여부 0/1
    dbl_next    INTEGER,                  -- 더블헤더 다음날 0/1
    PRIMARY KEY (team_id, date)
);

CREATE TABLE IF NOT EXISTS odds (
    game_id     INTEGER,
    market      TEXT,                     -- WIN / OU / HDC
    line        REAL,                     -- OU/HDC 기준선 (WIN은 NULL)
    selection   TEXT,                     -- HOME/AWAY/OVER/UNDER 등
    odds_value  REAL,                     -- decimal odds
    captured_at TEXT
);

CREATE TABLE IF NOT EXISTS results (
    game_id     INTEGER PRIMARY KEY,
    home_score  INTEGER,
    away_score  INTEGER
);

CREATE TABLE IF NOT EXISTS predictions (
    game_id     INTEGER,
    market      TEXT,
    selection   TEXT,
    line        REAL,
    model_prob  REAL,
    ev          REAL,
    captured_at TEXT
);
"""


def init_db(conn: sqlite3.Connection) -> None:
    """주어진 커넥션에 모든 테이블을 멱등 생성한다."""
    conn.executescript(DDL)
    conn.commit()
