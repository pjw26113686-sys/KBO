-- KBO EV 베팅 시스템 스키마 (설계서 §3)
-- 정규화된 핵심 테이블. raw 크롤링 결과를 정제해 적재하고,
-- predictions/results/odds 가 백테스트·EV 엔진의 입력이 된다.

PRAGMA foreign_keys = ON;

-- 팀 메타 + 구장 파크팩터(설계서: 득점/홈런 환경)
CREATE TABLE IF NOT EXISTS teams (
    team_id          INTEGER PRIMARY KEY,
    name             TEXT NOT NULL,
    park_factor_run  REAL NOT NULL DEFAULT 1.0,
    park_factor_hr   REAL NOT NULL DEFAULT 1.0
);

-- 선수. hand: L/R/S(좌/우/스위치)
CREATE TABLE IF NOT EXISTS players (
    player_id  INTEGER PRIMARY KEY,
    name       TEXT NOT NULL,
    team_id    INTEGER REFERENCES teams(team_id),
    hand       TEXT CHECK (hand IN ('L', 'R', 'S'))
);

-- 선발 등판 기록(★★★★★ 최우선 변수)
CREATE TABLE IF NOT EXISTS pitcher_game (
    game_id     INTEGER,
    player_id   INTEGER REFERENCES players(player_id),
    ip          REAL,           -- 이닝
    pitches     INTEGER,        -- 투구수
    fip_recent  REAL,           -- 최근 FIP
    rest_days   INTEGER,        -- 휴식일
    vs_team_id  INTEGER REFERENCES teams(team_id),
    PRIMARY KEY (game_id, player_id)
);

-- 불펜 등판 로그(★★★★ 연투 추적)
CREATE TABLE IF NOT EXISTS bullpen_log (
    date         TEXT,
    player_id    INTEGER REFERENCES players(player_id),
    pitches      INTEGER,
    is_back2back INTEGER NOT NULL DEFAULT 0,   -- 0/1 연투 여부
    PRIMARY KEY (date, player_id)
);

-- 타자 폼(★★★ 최근 14일 OPS, 좌우 스플릿)
CREATE TABLE IF NOT EXISTS batter_form (
    player_id  INTEGER REFERENCES players(player_id),
    date       TEXT,
    woba       REAL,
    ops_14d    REAL,
    vs_lhp     REAL,           -- 좌투 상대 성적
    vs_rhp     REAL,           -- 우투 상대 성적
    PRIMARY KEY (player_id, date)
);

-- 경기 일정/상태
CREATE TABLE IF NOT EXISTS games (
    game_id     INTEGER PRIMARY KEY,
    date        TEXT NOT NULL,
    start_time  TEXT,
    home_id     INTEGER REFERENCES teams(team_id),
    away_id     INTEGER REFERENCES teams(team_id),
    home_sp_id  INTEGER REFERENCES players(player_id),
    away_sp_id  INTEGER REFERENCES players(player_id),
    stadium     TEXT,
    status      TEXT NOT NULL DEFAULT 'scheduled'  -- scheduled/final/canceled
);

-- 일정 피로(★★ 원정 연속, 이동거리, 직전 연장, 더블헤더 다음날)
CREATE TABLE IF NOT EXISTS schedule_load (
    team_id     INTEGER REFERENCES teams(team_id),
    date        TEXT,
    road_streak INTEGER NOT NULL DEFAULT 0,
    travel_km   REAL NOT NULL DEFAULT 0,
    prev_extra  INTEGER NOT NULL DEFAULT 0,   -- 직전 경기 연장 여부 0/1
    dbl_next    INTEGER NOT NULL DEFAULT 0,   -- 더블헤더 다음날 0/1
    PRIMARY KEY (team_id, date)
);

-- 시장 배당. market: WIN/OU/HDC
CREATE TABLE IF NOT EXISTS odds (
    game_id     INTEGER REFERENCES games(game_id),
    market      TEXT NOT NULL CHECK (market IN ('WIN', 'OU', 'HDC')),
    line        REAL,           -- OU 기준점 / HDC 핸디캡 (WIN 은 NULL)
    odds_value  REAL NOT NULL,  -- decimal odds
    captured_at TEXT NOT NULL,
    PRIMARY KEY (game_id, market, line, captured_at)
);

-- 실제 결과(백테스트용)
CREATE TABLE IF NOT EXISTS results (
    game_id     INTEGER PRIMARY KEY REFERENCES games(game_id),
    home_score  INTEGER NOT NULL,
    away_score  INTEGER NOT NULL
);

-- 모델 예측(model_prob + EV)
CREATE TABLE IF NOT EXISTS predictions (
    game_id     INTEGER REFERENCES games(game_id),
    market      TEXT NOT NULL CHECK (market IN ('WIN', 'OU', 'HDC')),
    line        REAL,
    model_prob  REAL NOT NULL,
    ev          REAL,
    captured_at TEXT NOT NULL,
    PRIMARY KEY (game_id, market, line, captured_at)
);
