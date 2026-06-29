"""포아송 모델 검증."""
import numpy as np
import pytest

from kbo.model import poisson


def test_distribution_sums_to_one():
    for lam in (2.0, 4.6, 8.0):
        d = poisson.score_distribution(lam, max_runs=30)
        assert d.sum() == pytest.approx(1.0, abs=1e-6)


def test_nbinom_more_dispersed_than_poisson():
    lam = 4.6
    p = poisson.score_distribution(lam, max_runs=40, dist="poisson")
    nb = poisson.score_distribution(lam, max_runs=40, dist="nbinom", overdispersion=0.4)
    k = np.arange(len(p))
    var_p = (p * k**2).sum() - (p * k).sum() ** 2
    var_nb = (nb * k**2).sum() - (nb * k).sum() ** 2
    assert var_nb > var_p


def test_equal_lambdas_symmetric_winprob():
    # 동일 λ(홈 어드밴티지 끄고)면 승/패 확률이 같아야 한다.
    mp = poisson.market_probabilities(4.0, 4.0, ou_line=8.5, handicap=-1.5)
    assert mp.p_home_win == pytest.approx(mp.p_away_win, abs=1e-6)


def test_higher_lambda_higher_winprob():
    mp = poisson.market_probabilities(5.5, 3.5, ou_line=8.5, handicap=-1.5)
    assert mp.p_home_win > mp.p_away_win
    assert mp.p_home_win > 0.5


def test_probabilities_consistent():
    mp = poisson.market_probabilities(4.8, 4.2, ou_line=8.5, handicap=-1.5)
    assert mp.p_home_win + mp.p_away_win + mp.p_draw == pytest.approx(1.0, abs=1e-4)
    # OU 선이 .5 면 push 없음 → over+under≈1
    assert mp.p_over + mp.p_under == pytest.approx(1.0, abs=1e-4)
    for p in (mp.p_home_win, mp.p_over, mp.p_home_cover):
        assert 0.0 <= p <= 1.0


def test_estimate_lambda_home_advantage():
    f = poisson.TeamMatchupFeatures()
    lh, la = poisson.estimate_lambda(f, f)
    assert lh > la  # 동일 전력이면 홈이 약간 높아야
