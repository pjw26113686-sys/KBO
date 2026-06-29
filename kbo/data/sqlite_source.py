"""SQLite 데이터 소스 (실데이터 파이프라인의 토대).

GameRecord 를 game_features / results / games 테이블에 저장하고 되읽는다.
(2차) 크롤러는 raw → 정제 후 game_features 에 적재하면 동일 인터페이스로 동작한다.
"""
from __future__ import annotations

import sqlite3

from ..model.poisson import TeamMatchupFeatures
from .base import DataSource, GameRecord


def _feature_row(game_id: int, side: str, f: TeamMatchupFeatures):
    return (
        game_id,
        side,
        f.offense_strength,
        f.opp_pitcher_suppression,
        f.park_factor,
        f.bullpen_fatigue,
        f.schedule_fatigue,
    )


def save_records(conn: sqlite3.Connection, records: list[GameRecord]) -> None:
    """GameRecord 들을 games/game_features/results 테이블에 적재."""
    for r in records:
        conn.execute(
            "INSERT OR REPLACE INTO games "
            "(game_id, date, start_time, home_id, away_id, stadium, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                r.game_id, r.date, r.start_time, None, None, r.stadium,
                "final" if r.is_final else "scheduled",
            ),
        )
        conn.executemany(
            "INSERT OR REPLACE INTO game_features "
            "(game_id, side, offense_strength, opp_suppression, park_factor, "
            " bullpen_fatigue, schedule_fatigue) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                _feature_row(r.game_id, "HOME", r.home_features),
                _feature_row(r.game_id, "AWAY", r.away_features),
            ],
        )
        if r.is_final:
            conn.execute(
                "INSERT OR REPLACE INTO results (game_id, home_score, away_score) "
                "VALUES (?, ?, ?)",
                (r.game_id, r.home_score, r.away_score),
            )
    conn.commit()


def _features_from_row(row: sqlite3.Row, is_home: bool) -> TeamMatchupFeatures:
    return TeamMatchupFeatures(
        offense_strength=row["offense_strength"],
        opp_pitcher_suppression=row["opp_suppression"],
        park_factor=row["park_factor"],
        bullpen_fatigue=row["bullpen_fatigue"],
        schedule_fatigue=row["schedule_fatigue"],
        is_home=is_home,
    )


class SqliteDataSource(DataSource):
    """game_features(+results) 에서 GameRecord 를 복원한다."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def games(self) -> list[GameRecord]:
        feats: dict[int, dict[str, sqlite3.Row]] = {}
        for row in self.conn.execute("SELECT * FROM game_features"):
            feats.setdefault(row["game_id"], {})[row["side"]] = row
        results = {
            r["game_id"]: (r["home_score"], r["away_score"])
            for r in self.conn.execute("SELECT * FROM results")
        }
        out: list[GameRecord] = []
        for gid, sides in sorted(feats.items()):
            if "HOME" not in sides or "AWAY" not in sides:
                continue
            hs, as_ = results.get(gid, (None, None))
            out.append(
                GameRecord(
                    game_id=gid,
                    home_features=_features_from_row(sides["HOME"], True),
                    away_features=_features_from_row(sides["AWAY"], False),
                    home_score=hs,
                    away_score=as_,
                )
            )
        return out
