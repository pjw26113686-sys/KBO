"""포아송 결합 예측 모델 (설계서 §4).

복잡한 ML 이전의 베이스라인. 과적합 방지가 목적.

흐름:
  1) 팀별 기대득점 λ 추정 (보정계수 곱)
  2) 각 팀 득점 ~ Poisson(λ) 분포 생성
  3) 두 분포를 결합해 승패 / 오버언더 / 핸디캡 확률 + 예측구간 파생
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from scipy.stats import poisson as _sp_poisson

    def _poisson_pmf(k: np.ndarray, lam: float) -> np.ndarray:
        return _sp_poisson.pmf(k, lam)

except ImportError:  # pragma: no cover - scipy 미설치 fallback
    from math import exp, lgamma

    def _poisson_pmf(k: np.ndarray, lam: float) -> np.ndarray:
        k = np.asarray(k, dtype=float)
        log_pmf = k * np.log(lam) - lam - np.array([lgamma(int(x) + 1) for x in k])
        return np.exp(log_pmf)


from .. import config


# ---------------------------------------------------------------------------
# 입력 피처
# ---------------------------------------------------------------------------
@dataclass
class GameFeatures:
    """한 경기에 대한 모델 입력. 시뮬레이터/크롤러가 동일 형태로 채운다.

    fatigue 값들은 [0, 1] 범위(0=완전 휴식, 1=극도 피로)로 정규화되어 있다.
    """

    home_woba: float
    away_woba: float
    home_sp_fip: float          # 홈팀 선발 FIP
    away_sp_fip: float          # 원정팀 선발 FIP
    home_bullpen_fatigue: float = 0.0
    away_bullpen_fatigue: float = 0.0
    home_schedule_fatigue: float = 0.0
    away_schedule_fatigue: float = 0.0
    park_factor_run: float = 1.0


# ---------------------------------------------------------------------------
# 모델 파라미터 (튜닝 대상)
# ---------------------------------------------------------------------------
@dataclass
class ModelParams:
    """λ 추정에 쓰이는 보정계수 묶음. 그리드서치 튜닝의 대상이다.

    기본값은 config 상수에서 가져오며(from_config), 호출부가 params 를 주지 않으면
    이 기본값이 쓰여 기존 동작과 동일하다(하위호환).
    """

    league_avg_runs: float
    home_advantage: float
    sp_fip_sensitivity: float
    lineup_woba_sensitivity: float
    bullpen_fatigue_sensitivity: float
    schedule_fatigue_sensitivity: float
    league_avg_fip: float
    league_avg_woba: float
    woba_scale: float
    lambda_clip: tuple[float, float]

    @classmethod
    def from_config(cls) -> "ModelParams":
        return cls(
            league_avg_runs=config.LEAGUE_AVG_RUNS,
            home_advantage=config.HOME_ADVANTAGE,
            sp_fip_sensitivity=config.SP_FIP_SENSITIVITY,
            lineup_woba_sensitivity=config.LINEUP_WOBA_SENSITIVITY,
            bullpen_fatigue_sensitivity=config.BULLPEN_FATIGUE_SENSITIVITY,
            schedule_fatigue_sensitivity=config.SCHEDULE_FATIGUE_SENSITIVITY,
            league_avg_fip=config.LEAGUE_AVG_FIP,
            league_avg_woba=config.LEAGUE_AVG_WOBA,
            woba_scale=config.WOBA_SCALE,
            lambda_clip=config.LAMBDA_MULTIPLIER_CLIP,
        )


def _default_params() -> ModelParams:
    return ModelParams.from_config()


# ---------------------------------------------------------------------------
# 보정계수
# ---------------------------------------------------------------------------
def _sp_suppression(opp_sp_fip: float, p: ModelParams) -> float:
    """상대 선발의 억제력 배수. FIP가 리그평균보다 높을수록 실점 증가(>1)."""
    rel = opp_sp_fip - p.league_avg_fip
    return 1.0 + p.sp_fip_sensitivity * rel


def _lineup_strength(woba: float, p: ModelParams) -> float:
    """타선 강도 배수. wOBA가 리그평균보다 높을수록 득점 증가(>1)."""
    rel = (woba - p.league_avg_woba) / p.woba_scale
    return 1.0 + p.lineup_woba_sensitivity * rel


def _bullpen_fatigue_adj(opp_bullpen_fatigue: float, p: ModelParams) -> float:
    """상대 불펜 피로 배수. 상대 불펜이 지칠수록 우리 득점 증가(>1)."""
    return 1.0 + p.bullpen_fatigue_sensitivity * opp_bullpen_fatigue


def _schedule_fatigue_adj(own_schedule_fatigue: float, p: ModelParams) -> float:
    """자팀 일정 피로 배수. 우리가 지칠수록 우리 득점 감소(<1)."""
    return 1.0 - p.schedule_fatigue_sensitivity * own_schedule_fatigue


def estimate_lambda(f: GameFeatures,
                    params: ModelParams | None = None) -> tuple[float, float]:
    """홈/원정 기대득점 (λ_home, λ_away)를 추정한다 (설계서 §4 Step 1).

    λ_home = 리그평균 × 상대선발억제 × 홈타선 × 파크팩터
             × 상대불펜피로 × 홈일정피로 × (1 + 홈어드밴티지)
    λ_away = 동일 구조, 홈어드밴티지 미적용.

    params=None 이면 config 기본 계수를 쓴다(기존 동작과 동일).
    """
    p = params or _default_params()
    lo, hi = p.lambda_clip

    home_mult = (
        _sp_suppression(f.away_sp_fip, p)
        * _lineup_strength(f.home_woba, p)
        * f.park_factor_run
        * _bullpen_fatigue_adj(f.away_bullpen_fatigue, p)
        * _schedule_fatigue_adj(f.home_schedule_fatigue, p)
        * (1.0 + p.home_advantage)
    )
    away_mult = (
        _sp_suppression(f.home_sp_fip, p)
        * _lineup_strength(f.away_woba, p)
        * f.park_factor_run
        * _bullpen_fatigue_adj(f.home_bullpen_fatigue, p)
        * _schedule_fatigue_adj(f.away_schedule_fatigue, p)
    )

    lam_home = p.league_avg_runs * float(np.clip(home_mult, lo, hi))
    lam_away = p.league_avg_runs * float(np.clip(away_mult, lo, hi))
    return lam_home, lam_away


# ---------------------------------------------------------------------------
# 분포
# ---------------------------------------------------------------------------
def score_distribution(lam: float, max_runs: int | None = None) -> np.ndarray:
    """0..max_runs 득점에 대한 Poisson PMF 벡터 (절단 후 재정규화)."""
    max_runs = config.MAX_RUNS if max_runs is None else max_runs
    k = np.arange(0, max_runs + 1)
    pmf = _poisson_pmf(k, lam)
    return pmf / pmf.sum()


def _joint(lam_home: float, lam_away: float,
           max_runs: int | None = None) -> np.ndarray:
    """joint[i, j] = P(home=i, away=j). 두 팀 득점 독립 가정."""
    ph = score_distribution(lam_home, max_runs)
    pa = score_distribution(lam_away, max_runs)
    return np.outer(ph, pa)


def total_distribution(lam_home: float, lam_away: float,
                       max_runs: int | None = None) -> np.ndarray:
    """총득점 t = home+away 의 분포. 인덱스 = 총득점."""
    ph = score_distribution(lam_home, max_runs)
    pa = score_distribution(lam_away, max_runs)
    return np.convolve(ph, pa)


def diff_distribution(lam_home: float, lam_away: float,
                      max_runs: int | None = None):
    """점수차 d = home-away 분포. (support, pmf) 반환. support는 -max..max."""
    joint = _joint(lam_home, lam_away, max_runs)
    n = joint.shape[0] - 1
    support = np.arange(-n, n + 1)
    # joint[i, j] 에서 d = i - j. np.trace(joint, offset=-d) 가 해당 대각 합.
    pmf = np.array([np.trace(joint, offset=-d) for d in support])
    return support, pmf


# ---------------------------------------------------------------------------
# 3종 마켓 확률
# ---------------------------------------------------------------------------
def p_home_win(lam_home: float, lam_away: float) -> float:
    """P(home 승). 무승부(연장)는 50:50으로 분배."""
    joint = _joint(lam_home, lam_away)
    p_win = np.tril(joint, k=-1).sum()   # home > away
    p_tie = np.trace(joint)              # home == away
    return float(p_win + 0.5 * p_tie)


def p_over(lam_home: float, lam_away: float, line: float) -> float:
    """P(총득점 > line). line은 보통 .5 단위라 동점 라인 없음."""
    total = total_distribution(lam_home, lam_away)
    idx = np.arange(len(total))
    return float(total[idx > line].sum())


def p_handicap(lam_home: float, lam_away: float, handicap: float) -> float:
    """P(home_score - away_score > handicap). 예: handicap=-1.5 → 홈 -1.5 커버."""
    support, pmf = diff_distribution(lam_home, lam_away)
    return float(pmf[support > handicap].sum())


# ---------------------------------------------------------------------------
# 예측구간 (커버리지 검증용)
# ---------------------------------------------------------------------------
def _central_interval(support: np.ndarray, pmf: np.ndarray,
                      level: float) -> tuple[int, int]:
    """중심 level% 확률을 담는 [lo, hi] 정수 구간을 반환한다."""
    cdf = np.cumsum(pmf)
    alpha = (1.0 - level) / 2.0
    lo_idx = int(np.searchsorted(cdf, alpha))
    hi_idx = int(np.searchsorted(cdf, 1.0 - alpha))
    lo_idx = min(lo_idx, len(support) - 1)
    hi_idx = min(hi_idx, len(support) - 1)
    return int(support[lo_idx]), int(support[hi_idx])


def prediction_interval(lam_home: float, lam_away: float,
                        level: float | None = None) -> dict:
    """총득점과 점수차의 중심 level% 예측구간을 반환한다.

    반환: {"total": (lo, hi), "diff": (lo, hi), "level": level}
    실제 결과가 이 구간에 들어오는 비율을 backtest에서 검증한다.
    """
    level = config.PREDICTION_INTERVAL_LEVEL if level is None else level
    total = total_distribution(lam_home, lam_away)
    total_support = np.arange(len(total))
    diff_support, diff_pmf = diff_distribution(lam_home, lam_away)
    return {
        "total": _central_interval(total_support, total, level),
        "diff": _central_interval(diff_support, diff_pmf, level),
        "level": level,
    }


# ---------------------------------------------------------------------------
# 통합 산출
# ---------------------------------------------------------------------------
def predict_markets(f: GameFeatures, ou_line: float | None = None,
                    handicap: float | None = None,
                    params: ModelParams | None = None) -> dict:
    """한 경기에 대한 3종 마켓 확률 + 예측구간 + λ를 한 번에 산출한다."""
    ou_line = config.DEFAULT_OU_LINE if ou_line is None else ou_line
    handicap = config.DEFAULT_HANDICAP if handicap is None else handicap

    lam_home, lam_away = estimate_lambda(f, params)
    return {
        "lam_home": lam_home,
        "lam_away": lam_away,
        "WIN": p_home_win(lam_home, lam_away),
        "OU": p_over(lam_home, lam_away, ou_line),
        "HDC": p_handicap(lam_home, lam_away, handicap),
        "ou_line": ou_line,
        "handicap": handicap,
        "interval": prediction_interval(lam_home, lam_away),
    }
