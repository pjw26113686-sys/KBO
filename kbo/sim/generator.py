"""합성 데이터 생성기 = '진실(λ_true)'.

진실을 우리가 알고 있으므로, 모델/시장이 보는 값에 통제된 노이즈를 주입해
'모델이 시장보다 정확/부정확한 상황'을 직접 만들어 검증할 수 있다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..model.poisson import LEAGUE_AVG_RUNS, TeamMatchupFeatures, expected_runs


@dataclass
class Team:
    team_id: int
    name: str
    offense: float       # 리그평균 대비 타선 강도 (배수, 중앙값 1.0)
    suppression: float   # 선발/투수진 억제력 (배수, <1 이면 상대 득점 억제)
    park_factor: float


@dataclass
class SimGame:
    game_id: int
    home: Team
    away: Team
    lam_home_true: float
    lam_away_true: float
    home_score: int
    away_score: int


def _lognormal(rng: np.random.Generator, sigma: float, size=None) -> np.ndarray | float:
    """중앙값 1.0 의 곱셈 노이즈. sigma=0 이면 정확히 1.0(노이즈 없음)."""
    if sigma <= 0:
        return 1.0 if size is None else np.ones(size)
    return np.exp(rng.normal(-0.5 * sigma * sigma, sigma, size=size))


def make_league(n_teams: int = 10, seed: int = 0) -> list[Team]:
    """팀별 진짜 전력을 난수로 생성(시드 고정)."""
    rng = np.random.default_rng(seed)
    teams: list[Team] = []
    for i in range(n_teams):
        teams.append(
            Team(
                team_id=i,
                name=f"T{i:02d}",
                offense=float(_lognormal(rng, 0.12)),
                suppression=float(_lognormal(rng, 0.12)),
                park_factor=float(np.clip(rng.normal(1.0, 0.06), 0.85, 1.20)),
            )
        )
    return teams


def _true_lambdas(home: Team, away: Team, rng: np.random.Generator) -> tuple[float, float]:
    """경기별 진짜 λ. 불펜/일정 피로는 경기마다 난수로 부여(설계서 2장 ★ 변수)."""
    # 불펜 피로(>1 이면 상대 득점 상향), 일정 피로(<1 이면 자팀 득점 하향)
    home_bullpen_fatigue = float(np.clip(rng.normal(1.0, 0.05), 0.85, 1.25))
    away_bullpen_fatigue = float(np.clip(rng.normal(1.0, 0.05), 0.85, 1.25))
    home_sched = float(np.clip(rng.normal(1.0, 0.04), 0.85, 1.10))
    away_sched = float(np.clip(rng.normal(1.0, 0.04), 0.85, 1.10))

    fh = TeamMatchupFeatures(
        offense_strength=home.offense,
        opp_pitcher_suppression=away.suppression,
        park_factor=home.park_factor,           # 홈 구장
        bullpen_fatigue=away_bullpen_fatigue,    # 상대(원정) 불펜 피로 → 홈 득점에 영향
        schedule_fatigue=home_sched,
        is_home=True,
    )
    fa = TeamMatchupFeatures(
        offense_strength=away.offense,
        opp_pitcher_suppression=home.suppression,
        park_factor=home.park_factor,           # 같은 구장에서 경기
        bullpen_fatigue=home_bullpen_fatigue,
        schedule_fatigue=away_sched,
        is_home=False,
    )
    return expected_runs(fh), expected_runs(fa)


def generate_games(n_games: int, teams: list[Team], seed: int = 1) -> list[SimGame]:
    """N경기를 생성: 진짜 λ → 실제 스코어 샘플(Poisson(λ_true))."""
    rng = np.random.default_rng(seed)
    games: list[SimGame] = []
    n = len(teams)
    for gid in range(n_games):
        i, j = rng.choice(n, size=2, replace=False)
        home, away = teams[i], teams[j]
        lh, la = _true_lambdas(home, away, rng)
        hs = int(rng.poisson(lh))
        as_ = int(rng.poisson(la))
        games.append(SimGame(gid, home, away, lh, la, hs, as_))
    return games


def observe_lambda(
    lam_true: float, sigma: float, rng: np.random.Generator
) -> float:
    """진실 λ 의 노이즈 관측치. sigma 가 클수록 추정이 부정확.

    모델과 시장이 각각 독립적으로 호출 → σ_model vs σ_market 로 상대 정확도를 통제.
    """
    return float(lam_true * _lognormal(rng, sigma))
