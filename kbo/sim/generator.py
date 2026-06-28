"""합성 그라운드트루스 생성기.

핵심 설계:
  - '진짜' 팀/투수/피로 파라미터를 분포에서 샘플 → true λ → 실제 스코어 생성.
  - 모델이 보는 입력에는 추정오차(노이즈)를 주입 → 모델이 진실을 못 본다(현실적 갭).
  - 시장(배당)은 진짜에 가깝되 불펜·일정 피로를 부분적으로만 반영(blind spot)
    → 모델이 진짜 엣지를 가질 구간을 만들어 EV 베팅 ROI 검증이 의미있게 한다.

true_lambdas 와 모델은 동일한 함수형(model.estimate_lambda)을 공유한다.
차이는 '입력 파라미터의 품질'뿐 — 그래서 모델 오차가 곧 입력 추정오차로 환원된다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import config
from ..model.poisson import GameFeatures, estimate_lambda

# ---------------------------------------------------------------------------
# 진짜 파라미터 분포
# ---------------------------------------------------------------------------
TRUE_WOBA_STD = 0.018
TRUE_FIP_STD = 0.60
TRUE_PARK_STD = 0.06

# 입력 추정오차 — 모델과 시장 모두 woba/fip 에는 같은 크기의 (독립) 노이즈.
#   → 평균적으로 둘 다 편향 없음. 차이는 아래 '피로 인지'에서만 체계적으로 발생.
WOBA_NOISE = 0.010
FIP_NOISE = 0.35
# 모델은 피로를 노이즈만 얹어 거의 정확히 본다. 시장은 config 의 awareness 만큼만 반영.
MODEL_FATIGUE_NOISE = 0.08

# 고피로(연투 경고 등) 경기 비율 — 시장 blind spot 이 드러나는 구간.
HIGH_FATIGUE_PROB = 0.20

KBO_TEAMS = [
    "두산", "LG", "키움", "SSG", "KT",
    "NC", "롯데", "삼성", "한화", "KIA",
]


# ---------------------------------------------------------------------------
@dataclass
class Team:
    team_id: int
    name: str
    true_woba: float
    park_factor_run: float


@dataclass
class Game:
    game_id: int
    home: Team
    away: Team
    home_sp_fip_true: float
    away_sp_fip_true: float
    home_bullpen_fatigue_true: float
    away_bullpen_fatigue_true: float
    home_schedule_fatigue_true: float
    away_schedule_fatigue_true: float

    def true_features(self) -> GameFeatures:
        return GameFeatures(
            home_woba=self.home.true_woba,
            away_woba=self.away.true_woba,
            home_sp_fip=self.home_sp_fip_true,
            away_sp_fip=self.away_sp_fip_true,
            home_bullpen_fatigue=self.home_bullpen_fatigue_true,
            away_bullpen_fatigue=self.away_bullpen_fatigue_true,
            home_schedule_fatigue=self.home_schedule_fatigue_true,
            away_schedule_fatigue=self.away_schedule_fatigue_true,
            park_factor_run=self.home.park_factor_run,
        )


# ---------------------------------------------------------------------------
def make_league(rng: np.random.Generator) -> list[Team]:
    """KBO 10개 팀의 진짜 실력/파크팩터를 샘플한다."""
    teams = []
    for i, name in enumerate(KBO_TEAMS):
        teams.append(
            Team(
                team_id=i + 1,
                name=name,
                true_woba=float(rng.normal(config.LEAGUE_AVG_WOBA, TRUE_WOBA_STD)),
                park_factor_run=float(
                    np.clip(rng.normal(1.0, TRUE_PARK_STD), 0.85, 1.18)
                ),
            )
        )
    return teams


def _sample_fatigue(rng: np.random.Generator) -> float:
    """[0,1] 피로도. 대부분 낮으나 일부 경기는 고피로(연투/연장 다음날)."""
    if rng.random() < HIGH_FATIGUE_PROB:
        return float(rng.uniform(0.6, 1.0))
    return float(rng.beta(1.5, 4.0))


def make_game(game_id: int, league: list[Team],
              rng: np.random.Generator) -> Game:
    """무작위 홈/원정 매치업과 선발·피로 상태를 생성한다."""
    home, away = rng.choice(len(league), size=2, replace=False)
    return Game(
        game_id=game_id,
        home=league[home],
        away=league[away],
        home_sp_fip_true=float(rng.normal(config.LEAGUE_AVG_FIP, TRUE_FIP_STD)),
        away_sp_fip_true=float(rng.normal(config.LEAGUE_AVG_FIP, TRUE_FIP_STD)),
        home_bullpen_fatigue_true=_sample_fatigue(rng),
        away_bullpen_fatigue_true=_sample_fatigue(rng),
        home_schedule_fatigue_true=_sample_fatigue(rng),
        away_schedule_fatigue_true=_sample_fatigue(rng),
    )


def simulate_result(lam_home: float, lam_away: float,
                    rng: np.random.Generator) -> tuple[int, int]:
    """진짜 λ로부터 실제 스코어를 추출한다. 동점이면 연장 1점으로 승부 결정."""
    h = int(rng.poisson(lam_home))
    a = int(rng.poisson(lam_away))
    if h == a:  # 야구는 무승부가 드물다 → 연장 가정, 50:50 승부.
        if rng.random() < 0.5:
            h += 1
        else:
            a += 1
    return h, a


# ---------------------------------------------------------------------------
# 모델 / 시장의 '관측' 피처
# ---------------------------------------------------------------------------
def model_view(game: Game, rng: np.random.Generator) -> GameFeatures:
    """모델이 보는 입력. 진짜값 + 추정오차. 피로는 거의 정확히 관측."""
    t = game.true_features()
    clip01 = lambda x: float(np.clip(x, 0.0, 1.0))
    return GameFeatures(
        home_woba=t.home_woba + rng.normal(0, WOBA_NOISE),
        away_woba=t.away_woba + rng.normal(0, WOBA_NOISE),
        home_sp_fip=t.home_sp_fip + rng.normal(0, FIP_NOISE),
        away_sp_fip=t.away_sp_fip + rng.normal(0, FIP_NOISE),
        home_bullpen_fatigue=clip01(t.home_bullpen_fatigue + rng.normal(0, MODEL_FATIGUE_NOISE)),
        away_bullpen_fatigue=clip01(t.away_bullpen_fatigue + rng.normal(0, MODEL_FATIGUE_NOISE)),
        home_schedule_fatigue=clip01(t.home_schedule_fatigue + rng.normal(0, MODEL_FATIGUE_NOISE)),
        away_schedule_fatigue=clip01(t.away_schedule_fatigue + rng.normal(0, MODEL_FATIGUE_NOISE)),
        park_factor_run=t.park_factor_run,
    )


def market_view(game: Game, rng: np.random.Generator) -> GameFeatures:
    """시장이 보는 입력. woba/fip 는 모델과 같은 수준으로 본다.

    핵심 차이: 불펜·일정 피로를 awareness 비율만큼만 반영(blind spot).
    """
    t = game.true_features()
    return GameFeatures(
        home_woba=t.home_woba + rng.normal(0, WOBA_NOISE),
        away_woba=t.away_woba + rng.normal(0, WOBA_NOISE),
        home_sp_fip=t.home_sp_fip + rng.normal(0, FIP_NOISE),
        away_sp_fip=t.away_sp_fip + rng.normal(0, FIP_NOISE),
        home_bullpen_fatigue=t.home_bullpen_fatigue * config.MARKET_BULLPEN_AWARENESS,
        away_bullpen_fatigue=t.away_bullpen_fatigue * config.MARKET_BULLPEN_AWARENESS,
        home_schedule_fatigue=t.home_schedule_fatigue * config.MARKET_SCHEDULE_AWARENESS,
        away_schedule_fatigue=t.away_schedule_fatigue * config.MARKET_SCHEDULE_AWARENESS,
        park_factor_run=t.park_factor_run,
    )


def true_lambdas(game: Game) -> tuple[float, float]:
    """진짜 파라미터로부터의 진짜 λ (실제 스코어 생성의 기준)."""
    return estimate_lambda(game.true_features())
