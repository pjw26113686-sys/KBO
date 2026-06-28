"""보정계수 그리드서치 튜닝 (설계서 §4 / §8-4).

흐름:
  1) build_dataset: 시뮬레이터(generator)를 재사용해 라벨드 데이터셋 생성
     = [(모델이 관측하는 features, 실제 home/away 스코어), ...]
  2) grid_search: 핵심 보정계수의 작은 격자를 훑어 승패 log-loss 최소 조합 선택

주의: 합성 데이터 기준이라 '튜닝 기계'의 동작 검증이 목적이다.
실데이터에서는 같은 인터페이스로 build_dataset 만 실데이터 로더로 교체하면 된다.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, replace

import numpy as np

from ..backtest.metrics import brier_score, log_loss
from ..model.poisson import GameFeatures, ModelParams, estimate_lambda, p_home_win
from ..sim import generator as gen

# 라벨드 한 경기: 관측 피처 + 실제 스코어.
LabeledGame = tuple[GameFeatures, int, int]


def build_dataset(n_games: int = 4000, seed: int = 42) -> list[LabeledGame]:
    """시뮬레이터로 (관측 features, home_score, away_score) 데이터셋을 만든다.

    모델이 실제로 보는 입력(gen.model_view, 추정오차 포함)과 진짜 결과를 짝짓는다.
    """
    rng = np.random.default_rng(seed)
    league = gen.make_league(rng)
    data: list[LabeledGame] = []
    for gid in range(1, n_games + 1):
        game = gen.make_game(gid, league, rng)
        true_lh, true_la = gen.true_lambdas(game)
        home, away = gen.simulate_result(true_lh, true_la, rng)
        observed = gen.model_view(game, rng)
        data.append((observed, home, away))
    return data


def _win_probs(data: list[LabeledGame], params: ModelParams) -> np.ndarray:
    """주어진 params 로 각 경기의 홈 승리확률을 계산한다."""
    probs = np.empty(len(data))
    for i, (f, _h, _a) in enumerate(data):
        lh, la = estimate_lambda(f, params)
        probs[i] = p_home_win(lh, la)
    return probs


def _win_outcomes(data: list[LabeledGame]) -> np.ndarray:
    return np.array([1.0 if h > a else 0.0 for _f, h, a in data])


def evaluate(data: list[LabeledGame], params: ModelParams) -> dict:
    """승패 기준 log-loss / Brier 를 반환한다."""
    probs = _win_probs(data, params)
    y = _win_outcomes(data)
    return {"log_loss": log_loss(probs, y), "brier": brier_score(probs, y)}


# 튜닝 대상 계수와 기본 탐색 격자 (각 3값). config 기본값 부근을 훑는다.
DEFAULT_GRID: dict[str, list[float]] = {
    "sp_fip_sensitivity": [0.08, 0.11, 0.14],
    "lineup_woba_sensitivity": [0.06, 0.09, 0.12],
    "home_advantage": [0.02, 0.04, 0.06],
}


@dataclass
class TuneResult:
    best_params: ModelParams
    best_score: float          # train log-loss
    default_score: float       # 기본 계수의 train log-loss
    n_combos: int


def grid_search(train: list[LabeledGame],
                grid: dict[str, list[float]] | None = None) -> TuneResult:
    """train 데이터에서 승패 log-loss 를 최소화하는 ModelParams 를 찾는다."""
    grid = DEFAULT_GRID if grid is None else grid
    base = ModelParams.from_config()
    default_score = evaluate(train, base)["log_loss"]

    keys = list(grid.keys())
    best_params, best_score = base, float("inf")
    n = 0
    for combo in itertools.product(*(grid[k] for k in keys)):
        n += 1
        cand = replace(base, **dict(zip(keys, combo)))
        score = evaluate(train, cand)["log_loss"]
        if score < best_score:
            best_score, best_params = score, cand
    return TuneResult(best_params, best_score, default_score, n)
