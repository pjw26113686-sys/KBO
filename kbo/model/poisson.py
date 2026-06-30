"""포아송 결합 예측 모델 (설계서 §4).

복잡한 ML 이전의 베이스라인. 과적합을 피하기 위해 반드시 이 구조부터 검증한다.

흐름(설계서 §4):
  Step 1  팀별 기대득점 λ 추정  (log-linear 보정)
  Step 2  득점 ~ Poisson(λ)  (분산 과대 시 음이항으로 교체)
  Step 3  두 팀 분포 결합 → 승패 / 오버언더 / 핸디캡 확률

모든 함수는 순수(pure)하며 DB 에 의존하지 않는다 → 시뮬레이션과 (향후)실전이
동일 코드를 재사용한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from .. import config
from .params import ModelParams


@dataclass(frozen=True)
class TeamInputs:
    """한 팀의 표준화(z-score) 관측 지표.

    모두 리그 평균 0, 표준편차 1 로 표준화된 값으로 들어온다고 가정한다.
    부호 규약:
      starter_fip_z      : 높을수록 '나쁜' 선발(FIP↑ → 실점↑)
      offense_woba_z     : 높을수록 강한 타선
      bullpen_fatigue_z  : 높을수록 불펜 피로(연투/누적 투구수↑)
      schedule_fatigue_z : 높을수록 일정 피로(원정연속/이동/연장↑)
    """

    starter_fip_z: float = 0.0
    offense_woba_z: float = 0.0
    bullpen_fatigue_z: float = 0.0
    schedule_fatigue_z: float = 0.0


@dataclass(frozen=True)
class Matchup:
    """한 경기의 양 팀 입력 + 구장 파크팩터."""

    home: TeamInputs
    away: TeamInputs
    park_factor_run: float = 1.0


# ---------------------------------------------------------------------------
# Step 1 — 기대득점 λ
# ---------------------------------------------------------------------------
def estimate_lambda(m: Matchup, params: ModelParams | None = None) -> tuple[float, float]:
    """홈/원정 기대득점 (λ_home, λ_away) 를 추정한다 (설계서 §4 Step1).

    λ = 리그평균 × [홈어드밴티지] × 파크팩터
        × exp( 상대선발_억제력 + 자기타선_강도
               + 상대불펜_피로 − 자기일정_피로 )

    상대 선발/불펜이 자기 득점에 영향을 주고, 자기 타선/일정이 자기 득점에
    영향을 준다는 점에 주의(혼동하기 쉬움).
    """
    p = params or ModelParams()
    base = p.league_avg_runs * m.park_factor_run

    def log_adj(scorer: TeamInputs, opponent: TeamInputs) -> float:
        return (
            p.starter_elasticity * opponent.starter_fip_z
            + p.offense_elasticity * scorer.offense_woba_z
            + p.bullpen_fatigue_elasticity * opponent.bullpen_fatigue_z
            - p.schedule_fatigue_elasticity * scorer.schedule_fatigue_z
        )

    lam_home = base * p.home_advantage * np.exp(log_adj(m.home, m.away))
    lam_away = base * np.exp(log_adj(m.away, m.home))
    return float(lam_home), float(lam_away)


# ---------------------------------------------------------------------------
# Step 2 — 득점 분포
# ---------------------------------------------------------------------------
def score_distribution(
    lam: float, params: ModelParams | None = None, max_runs: int = config.MAX_RUNS
) -> np.ndarray:
    """0..max_runs 점의 확률질량 벡터. 마지막 칸이 꼬리를 흡수해 합이 1이 된다.

    기본은 Poisson(λ). params.use_nbinom 이면 음이항(분산 과대)으로 교체한다
    (설계서 §4 Step2: "분산 과대 시 음이항으로 교체").
    """
    p = params or ModelParams()
    k = np.arange(max_runs + 1)
    if p.use_nbinom:
        # variance = mean * dispersion 가 되도록 (n, prob) 환산.
        # nbinom: mean = n(1-prob)/prob, var = mean/prob → dispersion = 1/prob.
        prob = 1.0 / p.nb_dispersion
        n = lam * prob / (1.0 - prob)
        pmf = stats.nbinom.pmf(k, n, prob)
    else:
        pmf = stats.poisson.pmf(k, lam)
    # 잘려나간 꼬리를 마지막 칸에 더해 정규화.
    pmf[-1] += max(0.0, 1.0 - pmf.sum())
    total = pmf.sum()
    return pmf / total if total > 0 else pmf


# ---------------------------------------------------------------------------
# Step 3 — 3종 마켓 확률
# ---------------------------------------------------------------------------
def _total_pmf(home_pmf: np.ndarray, away_pmf: np.ndarray) -> np.ndarray:
    """총득점(home+away) 분포 = 합성곱."""
    return np.convolve(home_pmf, away_pmf)


def prob_over(home_pmf: np.ndarray, away_pmf: np.ndarray, line: float) -> float:
    """P(home+away > line) — 오버언더의 '오버' 확률 (설계서 §4 Step3)."""
    total = _total_pmf(home_pmf, away_pmf)
    totals = np.arange(total.size)
    return float(total[totals > line].sum())


def prob_home_win(home_pmf: np.ndarray, away_pmf: np.ndarray) -> float:
    """승패 마켓의 홈 승리 확률.

    야구 머니라인은 무승부가 (연장으로) 해소되므로, '결판난 경기' 기준으로
    조건부 확률 P(home>away | home≠away) 을 반환한다.
    """
    joint = np.outer(home_pmf, away_pmf)
    h = np.arange(home_pmf.size)[:, None]
    a = np.arange(away_pmf.size)[None, :]
    p_home = float(joint[h > a].sum())
    p_away = float(joint[h < a].sum())
    decided = p_home + p_away
    return p_home / decided if decided > 0 else 0.5


def prob_handicap(home_pmf: np.ndarray, away_pmf: np.ndarray, handicap: float) -> float:
    """P(home_score − away_score > handicap) — 핸디캡 커버 확률 (설계서 §4 Step3).

    예: handicap=-1.5 → 홈이 2점차 이상 승리할 확률.
    """
    joint = np.outer(home_pmf, away_pmf)
    h = np.arange(home_pmf.size)[:, None]
    a = np.arange(away_pmf.size)[None, :]
    diff = h - a
    return float(joint[diff > handicap].sum())


def markets_from_lambda(
    lam_home: float,
    lam_away: float,
    ou_line: float,
    handicap: float,
    params: ModelParams | None = None,
) -> dict[str, float]:
    """주어진 (λ_home, λ_away) 에서 3종 마켓 확률을 산출한다.

    모델과 (시뮬)북메이커가 동일한 분포 코어를 공유하도록 분리한 함수.
    반환: {'WIN', 'OU', 'HDC'}.
    """
    p = params or ModelParams()
    home_pmf = score_distribution(lam_home, p)
    away_pmf = score_distribution(lam_away, p)
    return {
        "WIN": prob_home_win(home_pmf, away_pmf),
        "OU": prob_over(home_pmf, away_pmf, ou_line),
        "HDC": prob_handicap(home_pmf, away_pmf, handicap),
    }


def market_probabilities(
    m: Matchup,
    ou_line: float,
    handicap: float,
    params: ModelParams | None = None,
) -> dict[str, float]:
    """한 경기(Matchup)의 3종 마켓 모델 확률을 한 번에 산출한다.

    반환: {'WIN': 홈승, 'OU': 오버, 'HDC': 홈 핸디캡 커버, 'lam_home', 'lam_away'}.
    """
    p = params or ModelParams()
    lam_home, lam_away = estimate_lambda(m, p)
    out = markets_from_lambda(lam_home, lam_away, ou_line, handicap, p)
    out["lam_home"] = lam_home
    out["lam_away"] = lam_away
    return out
