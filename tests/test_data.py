"""데이터 어댑터(DataSource) 검증."""
from kbo.data import GameRecord, SyntheticDataSource
from kbo.data.sqlite_source import SqliteDataSource, save_records
from kbo.store import connect


def test_synthetic_source_shape():
    src = SyntheticDataSource(n_games=200, seed=1)
    games = src.games()
    assert len(games) == 200
    g = games[0]
    assert isinstance(g, GameRecord)
    assert g.is_final
    assert g.home_features.is_home is True
    assert g.away_features.is_home is False
    assert g.home_score >= 0 and g.away_score >= 0


def test_feature_noise_changes_features():
    clean = SyntheticDataSource(n_games=50, seed=2, feature_noise=0.0).games()
    noisy = SyntheticDataSource(n_games=50, seed=2, feature_noise=0.2).games()
    # 점수(진실)는 동일 시드라 같지만, 관측 피처는 달라야 한다.
    assert clean[0].home_score == noisy[0].home_score
    assert clean[0].home_features.offense_strength != noisy[0].home_features.offense_strength


def test_sqlite_roundtrip():
    records = SyntheticDataSource(n_games=30, seed=3).games()
    with connect(":memory:") as conn:
        save_records(conn, records)
        back = SqliteDataSource(conn).games()
    assert len(back) == len(records)
    a, b = records[0], next(r for r in back if r.game_id == records[0].game_id)
    assert a.home_score == b.home_score
    assert abs(a.home_features.offense_strength - b.home_features.offense_strength) < 1e-9
    assert b.home_features.is_home is True
