"""확률 예측 평가 지표 (설계서 §6).

log-loss, Brier, 캘리브레이션. 모두 이진(binary) 결과 기준.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_EPS = 1e-12


def brier_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Brier = mean((p − y)^2). 완벽 예측이면 0, 무작위(0.5)면 0.25."""
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    return float(np.mean((probs - outcomes) ** 2))


def log_loss(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """log-loss = −mean(y·ln p + (1−y)·ln(1−p)). 무작위(0.5) ≈ ln2 ≈ 0.693."""
    probs = np.clip(np.asarray(probs, dtype=float), _EPS, 1 - _EPS)
    outcomes = np.asarray(outcomes, dtype=float)
    return float(-np.mean(outcomes * np.log(probs) + (1 - outcomes) * np.log(1 - probs)))


@dataclass(frozen=True)
class CalibrationBin:
    lo: float
    hi: float
    count: int
    mean_pred: float    # bin 내 평균 예측 확률
    mean_obs: float     # bin 내 실제 발생 비율


def calibration_bins(
    probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10
) -> list[CalibrationBin]:
    """예측 확률을 동일 폭 bin 으로 나눠 (평균 예측 vs 실제 비율) 을 모은다.

    캘리브레이션이 좋으면 mean_pred ≈ mean_obs (설계서 §5: 60% 예측 경기가
    실제 ~60% 승리하는가).
    """
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[CalibrationBin] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (probs >= lo) & (probs < hi) if i < n_bins - 1 else (probs >= lo) & (probs <= hi)
        count = int(mask.sum())
        if count == 0:
            bins.append(CalibrationBin(lo, hi, 0, float("nan"), float("nan")))
        else:
            bins.append(
                CalibrationBin(lo, hi, count, float(probs[mask].mean()), float(outcomes[mask].mean()))
            )
    return bins


def calibration_error(probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error: bin별 |예측−관측| 의 표본수 가중 평균."""
    bins = calibration_bins(probs, outcomes, n_bins)
    total = sum(b.count for b in bins)
    if total == 0:
        return float("nan")
    return sum(b.count * abs(b.mean_pred - b.mean_obs) for b in bins if b.count > 0) / total
