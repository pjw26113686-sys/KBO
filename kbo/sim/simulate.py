"""몬테카를로 경기 시뮬레이션 (설계서 §4 Step2 를 정답지로 사용).

진짜 팀/구장 + 경기별 진짜 선발·불펜피로·일정피로 → 진짜 λ → Poisson 으로 스코어
draw. 모델은 이 진짜 λ 를 *모르고*, 노이즈 낀 관측 z 만 보고 추정해야 한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..model.params import ModelParams
from ..model.poisson import Matchup, TeamInputs, estimate_lambda
from .world import Team


@dataclass(frozen=True)
class GameTruth:
    """한 경기의 객관적 사실(정답지). 배당/예측은 runner 가 따로 계산."""

    game_id: int
    home_id: int
    away_id: int
    # 경기별 진짜 입력(z). 모델은 이 값에 노이즈를 더해 관측한다.
    home_true: TeamInputs
    away_true: TeamInputs
    park_factor_run: float
    lam_home: float
    lam_away: float
    home_score: int
    away_score: int


def simulate_season(
    teams: list[Team],
    n_games: int,
    rng: np.random.Generator,
    true_params: ModelParams,
) -> list[GameTruth]:
    """무작위 매치업으로 n_games 경기를 시뮬레이션해 진짜 결과를 만든다."""
    games: list[GameTruth] = []
    n_teams = len(teams)
    for gid in range(n_games):
        h, a = rng.choice(n_teams, size=2, replace=False)
        home, away = teams[int(h)], teams[int(a)]

        home_in = TeamInputs(
            starter_fip_z=float(rng.choice(home.rotation_fip_z)),
            offense_woba_z=home.offense_z,
            bullpen_fatigue_z=float(rng.normal(0.0, 1.0)),
            schedule_fatigue_z=float(rng.normal(0.0, 1.0)),
        )
        away_in = TeamInputs(
            starter_fip_z=float(rng.choice(away.rotation_fip_z)),
            offense_woba_z=away.offense_z,
            bullpen_fatigue_z=float(rng.normal(0.0, 1.0)),
            schedule_fatigue_z=float(rng.normal(0.0, 1.0)),
        )
        matchup = Matchup(home=home_in, away=away_in, park_factor_run=home.park_factor_run)
        lam_home, lam_away = estimate_lambda(matchup, true_params)

        home_score = int(rng.poisson(lam_home))
        away_score = int(rng.poisson(lam_away))

        games.append(
            GameTruth(
                game_id=gid,
                home_id=home.team_id,
                away_id=away.team_id,
                home_true=home_in,
                away_true=away_in,
                park_factor_run=home.park_factor_run,
                lam_home=lam_home,
                lam_away=lam_away,
                home_score=home_score,
                away_score=away_score,
            )
        )
    return games
