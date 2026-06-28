-- KBO EV 베팅 시스템 스키마 (설계서 §3)
-- 모든 테이블은 idempotent 하게 생성된다.

CREATE TABLE IF NOT EXISTS teams (
    team_id          INTEGER PRIMARY KEY,
    name             TEXT NOT NULL,
    park_factor_run  REAL DEFAULT 1.0,
    park_factor_hr   REAL DEFAULT 1.0
);

CREATE TABLE IF NOT EXISTS players (
    player_id  INTEGER PRIMARY KEY,
    name       TEXT NOT NULL,
    team_id    INTEGER REFERENCES teams(team_id),
    hand       TEXT CHECK (hand IN ('L', 'R', 'S'))   -- 좌/우/스위치
);

CREATE TABLE IF NOT EXISTS pitcher_game (
    game_id     INTEGER,
    player_id   INTEGER,
    ip          REAL,        -- innings pitched
    pitches     INTEGER,
    fip_recent  REAL,        -- 최근 FIP
    rest_days   INTEGER,
    vs_team_id  INTEGER,
    PRIMARY KEY (game_id, player_id)
);

CREATE TABLE IF NOT EXISTS bullpen_log (
    date         TEXT,
    player_id    INTEGER,
    pitches      INTEGER,
    is_back2back INTEGER DEFAULT 0,   -- 연투 여부 (0/1)
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
    home_id     INTEGER REFERENCES teams(team_id),
    away_id     INTEGER REFERENCES teams(team_id),
    home_sp_id  INTEGER,
    away_sp_id  INTEGER,
    stadium     TEXT,
    status      TEXT DEFAULT 'scheduled'   -- scheduled/final/postponed
);

CREATE TABLE IF NOT EXISTS schedule_load (
    team_id      INTEGER,
    date         TEXT,
    road_streak  INTEGER DEFAULT 0,   -- 원정 연속 경기수
    travel_km    REAL DEFAULT 0,
    prev_extra   INTEGER DEFAULT 0,   -- 직전 경기 연장 여부
    dbl_next     INTEGER DEFAULT 0,   -- 더블헤더 다음날 여부
    PRIMARY KEY (team_id, date)
);

CREATE TABLE IF NOT EXISTS odds (
    game_id     INTEGER,
    market      TEXT,        -- WIN / OU / HDC
    line        REAL,        -- OU/HDC 기준선 (WIN은 NULL)
    odds_value  REAL,        -- decimal odds
    captured_at TEXT
);

CREATE TABLE IF NOT EXISTS results (
    game_id     INTEGER PRIMARY KEY,
    home_score  INTEGER,
    away_score  INTEGER
);

CREATE TABLE IF NOT EXISTS predictions (
    game_id     INTEGER,
    market      TEXT,        -- WIN / OU / HDC
    model_prob  REAL,
    ev          REAL,
    captured_at TEXT
);

-- 조회 성능을 위한 인덱스
CREATE INDEX IF NOT EXISTS idx_odds_game   ON odds(game_id);
CREATE INDEX IF NOT EXISTS idx_pred_game   ON predictions(game_id);
