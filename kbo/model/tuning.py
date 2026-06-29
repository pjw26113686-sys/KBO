"""파라미터 튜닝 (설계서 4장: '과거 시즌으로 회귀/그리드서치 튜닝').

관측 피처 + 실제 스코어로 ModelParams 를 최대우도(MLE)로 적합한다.
목적함수 = 관측 득점에 대한 포아송 음의 로그우도(벡터화). L-BFGS-B 로 최적화.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

from .params import DEFAULT_PARAMS, ModelParams
from .poisson import expected_runs, win_prob


@dataclass
class _Design:
    """records 에서 한 번만 추출하는 설계행렬(최적화 루프에서 재사용)."""
    # 각 (game, side) 1행: home 먼저, away 뒤 → 길이 2N
    log_offense: np.ndarray
    log_suppression: np.ndarray
    log_park: np.ndarray
    log_bullpen: np.ndarray
    log_schedule: np.ndarray
    is_home: np.ndarray
    scores: np.ndarray


def _design(records) -> _Design:
    finals = [r for r in records if r.is_final]
    if not finals:
        raise ValueError("튜닝하려면 결과가 있는 경기가 필요합니다.")
    log_off, log_sup, log_park, log_bp, log_sch, home, scores = ([] for _ in range(7))
    for r in finals:
        for f, score in ((r.home_features, r.home_score), (r.away_features, r.away_score)):
            log_off.append(np.log(f.offense_strength))
            log_sup.append(np.log(f.opp_pitcher_suppression))
            log_park.append(np.log(f.park_factor))
            log_bp.append(np.log(f.bullpen_fatigue))
            log_sch.append(np.log(f.schedule_fatigue))
            home.append(1.0 if f.is_home else 0.0)
            scores.append(score)
    a = np.asarray
    return _Design(a(log_off), a(log_sup), a(log_park), a(log_bp), a(log_sch),
                   a(home), a(scores, dtype=float))


def _lambdas(d: _Design, p: ModelParams) -> np.ndarray:
    """벡터화 λ 계산. log 공간 합산 후 exp."""
    log_lam = (
        np.log(p.league_avg)
        + p.w_offense * d.log_offense
        + p.w_suppression * d.log_suppression
        + p.w_park * d.log_park
        + p.w_bullpen * d.log_bullpen
        + p.w_schedule * d.log_schedule
        + d.is_home * np.log(p.home_advantage)
    )
    return np.exp(log_lam)


def neg_log_likelihood(records, params: ModelParams) -> float:
    """관측 득점에 대한 평균 포아송 음의 로그우도(낮을수록 적합 좋음)."""
    d = _design(records)
    lam = np.clip(_lambdas(d, params), 1e-6, None)
    return float(-np.mean(poisson.logpmf(d.scores, lam)))


def win_eval(records, params: ModelParams) -> tuple[float, float, int]:
    """홀드아웃 평가: 승패 마켓(무승부 제외) log-loss / Brier 와 표본 수.

    파라미터 적합의 '결과'를 검증하는 지표 — 적합은 득점 우도로 하고 평가는 승률로 한다.
    """
    probs, outs = [], []
    for r in records:
        if not r.is_final or r.home_score == r.away_score:
            continue
        # 데이터 소스가 home_features.is_home=True / away_features.is_home=False 로 둔다.
        lh = expected_runs(r.home_features, params)
        la = expected_runs(r.away_features, params)
        ph, pa, _ = win_prob(lh, la)
        probs.append(ph / (ph + pa))
        outs.append(1.0 if r.home_score > r.away_score else 0.0)
    p = np.clip(np.asarray(probs), 1e-12, 1 - 1e-12)
    y = np.asarray(outs)
    log_loss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    brier = float(np.mean((p - y) ** 2))
    return log_loss, brier, len(y)


@dataclass
class TuneResult:
    params: ModelParams
    nll_before: float
    nll_after: float
    success: bool
    n_iter: int


def tune(records, init: ModelParams | None = None) -> TuneResult:
    """records(피처+결과)로 ModelParams 를 MLE 적합."""
    init = init or DEFAULT_PARAMS
    d = _design(records)

    def objective(vec) -> float:
        p = ModelParams.from_vector(vec)
        lam = np.clip(_lambdas(d, p), 1e-6, None)
        return float(-np.mean(poisson.logpmf(d.scores, lam)))

    nll_before = objective(init.to_vector())
    res = minimize(
        objective,
        x0=np.asarray(init.to_vector(), dtype=float),
        method="L-BFGS-B",
        bounds=ModelParams.bounds(),
    )
    tuned = ModelParams.from_vector(res.x)
    return TuneResult(
        params=tuned,
        nll_before=nll_before,
        nll_after=float(res.fun),
        success=bool(res.success),
        n_iter=int(res.nit),
    )
