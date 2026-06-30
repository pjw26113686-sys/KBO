"""백테스트 지표 단위 테스트 (설계서 §6)."""

import math

import numpy as np

from kbo.backtest import backtest as bt
from kbo.backtest.metrics import brier_score, calibration_error, log_loss


def test_perfect_prediction_brier_zero():
    probs = np.array([1.0, 0.0, 1.0, 0.0])
    outcomes = np.array([1, 0, 1, 0])
    assert brier_score(probs, outcomes) == 0.0


def test_random_prediction_logloss_near_ln2():
    rng = np.random.default_rng(0)
    outcomes = rng.integers(0, 2, size=20000)
    probs = np.full(outcomes.shape, 0.5)
    assert math.isclose(log_loss(probs, outcomes), math.log(2), abs_tol=1e-3)


def test_calibration_error_zero_for_calibrated():
    # 예측이 0.3 인 사건이 정확히 30% 발생하면 캘리브레이션 오차 ~0
    rng = np.random.default_rng(1)
    p = 0.3
    n = 50000
    probs = np.full(n, p)
    outcomes = (rng.random(n) < p).astype(float)
    assert calibration_error(probs, outcomes, n_bins=10) < 0.01


def test_flat_betting_roi_known_case():
    # 배당 2.0 에 2건 베팅, 1승 1패 → 손익 (1.0) + (−1.0) = 0, ROI 0
    res = bt.flat_betting_roi(np.array([2.0, 2.0]), np.array([True, False]))
    assert res.n_bets == 2
    assert res.n_wins == 1
    assert math.isclose(res.total_profit, 0.0, abs_tol=1e-9)
    assert math.isclose(res.roi, 0.0, abs_tol=1e-9)


def test_time_split_no_overlap():
    train, test = bt.time_split(100, test_frac=0.3)
    assert len(train) == 70 and len(test) == 30
    assert set(train).isdisjoint(set(test))
    assert train.max() < test.min()  # test 가 미래
