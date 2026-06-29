"""파라미터 튜닝 검증 (튜닝 머신러리가 올바르게 동작하는가)."""
from kbo.data import SyntheticDataSource
from kbo.model import tuning
from kbo.model.params import DEFAULT_PARAMS, ModelParams


def _records(n=3000, feature_noise=0.0, seed=42):
    true = ModelParams(league_avg=4.9, home_advantage=1.08)
    src = SyntheticDataSource(
        n_games=n, seed=seed, true_params=true, feature_noise=feature_noise
    )
    return src.games(), true


def test_tune_reduces_nll_and_recovers_params():
    records, true = _records(n=4000, feature_noise=0.0)
    res = tuning.tune(records, init=DEFAULT_PARAMS)
    assert res.nll_after <= res.nll_before
    # 노이즈 없으면 진짜 파라미터를 근접 회복해야 한다.
    assert abs(res.params.home_advantage - true.home_advantage) < 0.03
    assert abs(res.params.league_avg - true.league_avg) < 0.30


def test_tune_improves_holdout_winmarket():
    records, _ = _records(n=4000, feature_noise=0.05, seed=7)
    cut = int(len(records) * 0.7)
    train, test = records[:cut], records[cut:]
    res = tuning.tune(train, init=DEFAULT_PARAMS)
    ll0, bs0, _ = tuning.win_eval(test, DEFAULT_PARAMS)
    ll1, bs1, _ = tuning.win_eval(test, res.params)
    assert ll1 <= ll0 + 1e-6
    assert bs1 <= bs0 + 1e-6


def test_perfect_params_minimize_nll():
    # 진짜 파라미터로 평가한 NLL 이 기본값보다 낮아야 한다(정합성).
    records, true = _records(n=2000, feature_noise=0.0)
    assert tuning.neg_log_likelihood(records, true) < tuning.neg_log_likelihood(
        records, DEFAULT_PARAMS
    )
