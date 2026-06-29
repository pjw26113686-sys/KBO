"""합성 데이터 소스 (generator 를 DataSource 인터페이스로 감싼다)."""
from __future__ import annotations

import numpy as np

from ..model.params import ModelParams
from ..model.poisson import TeamMatchupFeatures
from ..sim import generator
from .base import DataSource, GameRecord


def _noisy(f: TeamMatchupFeatures, sigma: float, rng: np.random.Generator) -> TeamMatchupFeatures:
    """피처에 곱셈 관측 노이즈 주입(σ=0 이면 그대로). is_home 은 보존."""
    if sigma <= 0:
        return f
    m = np.exp(rng.normal(-0.5 * sigma * sigma, sigma, size=5))
    return TeamMatchupFeatures(
        offense_strength=f.offense_strength * m[0],
        opp_pitcher_suppression=f.opp_pitcher_suppression * m[1],
        park_factor=f.park_factor * m[2],
        bullpen_fatigue=f.bullpen_fatigue * m[3],
        schedule_fatigue=f.schedule_fatigue * m[4],
        is_home=f.is_home,
    )


class SyntheticDataSource(DataSource):
    """진짜 규칙(true_params)으로 결과를 만들고, 관측 피처에는 feature_noise 를 입힌다.

    feature_noise=0 이면 모델이 진짜 피처를 그대로 본다(튜닝이 파라미터를 정확히 회복 가능).
    """

    def __init__(
        self,
        n_games: int = 1000,
        n_teams: int = 10,
        seed: int = 42,
        true_params: ModelParams | None = None,
        feature_noise: float = 0.0,
    ):
        self.n_games = n_games
        self.n_teams = n_teams
        self.seed = seed
        self.true_params = true_params or ModelParams()
        self.feature_noise = feature_noise

    def games(self) -> list[GameRecord]:
        teams = generator.make_league(self.n_teams, seed=self.seed)
        sim = generator.generate_games(
            self.n_games, teams, seed=self.seed + 1, params=self.true_params
        )
        rng = np.random.default_rng(self.seed + 500)
        records: list[GameRecord] = []
        for g in sim:
            records.append(
                GameRecord(
                    game_id=g.game_id,
                    home_features=_noisy(g.home_features, self.feature_noise, rng),
                    away_features=_noisy(g.away_features, self.feature_noise, rng),
                    home_score=g.home_score,
                    away_score=g.away_score,
                    home_name=g.home.name,
                    away_name=g.away.name,
                )
            )
        return records
