"""포아송 모델 단위 테스트."""

import numpy as np

from kbo.model import poisson as P
from kbo.model.poisson import GameFeatures


def _neutral_features(**kw) -> GameFeatures:
    base = dict(
        home_woba=0.330, away_woba=0.330,
        home_sp_fip=4.20, away_sp_fip=4.20,
        park_factor_run=1.0,
    )
    base.update(kw)
    return GameFeatures(**base)


def test_score_distribution_sums_to_one():
    pmf = P.score_distribution(4.5)
    assert abs(pmf.sum() - 1.0) < 1e-9
    assert (pmf >= 0).all()


def test_total_distribution_sums_to_one():
    total = P.total_distribution(4.5, 4.0)
    assert abs(total.sum() - 1.0) < 1e-9


def test_diff_distribution_sums_to_one():
    support, pmf = P.diff_distribution(4.5, 4.0)
    assert abs(pmf.sum() - 1.0) < 1e-9
    assert support[0] == -support[-1]


def test_probabilities_in_unit_range():
    f = _neutral_features()
    pred = P.predict_markets(f)
    for mk in ("WIN", "OU", "HDC"):
        assert 0.0 <= pred[mk] <= 1.0


def test_home_advantage_makes_home_favored():
    """동일 실력이면 홈 어드밴티지로 홈 승률 > 50%."""
    f = _neutral_features()
    assert P.predict_markets(f)["WIN"] > 0.5


def test_better_pitcher_lowers_opponent_runs():
    """상대 선발 FIP가 낮을수록(좋을수록) 우리 기대득점 감소."""
    good_opp = _neutral_features(away_sp_fip=3.0)   # 원정 선발이 좋음 → 홈 득점↓
    bad_opp = _neutral_features(away_sp_fip=5.5)
    lam_good, _ = P.estimate_lambda(good_opp)
    lam_bad, _ = P.estimate_lambda(bad_opp)
    assert lam_good < lam_bad


def test_win_prob_monotonic_in_lineup():
    """홈 타선이 강할수록 홈 승률 단조 증가."""
    weak = P.predict_markets(_neutral_features(home_woba=0.300))["WIN"]
    strong = P.predict_markets(_neutral_features(home_woba=0.360))["WIN"]
    assert strong > weak


def test_bullpen_fatigue_raises_opponent_scoring():
    """상대 불펜이 지칠수록 우리 기대득점 증가."""
    fresh, _ = P.estimate_lambda(_neutral_features(away_bullpen_fatigue=0.0))
    tired, _ = P.estimate_lambda(_neutral_features(away_bullpen_fatigue=1.0))
    assert tired > fresh


def test_prediction_interval_brackets_mean():
    pred = P.predict_markets(_neutral_features())
    lo, hi = pred["interval"]["total"]
    mean_total = pred["lam_home"] + pred["lam_away"]
    assert lo <= mean_total <= hi
    assert lo < hi
