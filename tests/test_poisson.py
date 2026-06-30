"""포아송 모델 단위 테스트 (설계서 §4)."""

import numpy as np

from kbo.model.params import ModelParams
from kbo.model.poisson import (
    Matchup,
    TeamInputs,
    estimate_lambda,
    market_probabilities,
    prob_handicap,
    prob_home_win,
    prob_over,
    score_distribution,
)


def test_score_distribution_sums_to_one():
    pmf = score_distribution(4.8)
    assert abs(pmf.sum() - 1.0) < 1e-9
    assert (pmf >= 0).all()


def test_nbinom_has_higher_variance_than_poisson():
    lam = 4.8
    k = np.arange(score_distribution(lam).size)
    pois = score_distribution(lam, ModelParams(use_nbinom=False))
    nb = score_distribution(lam, ModelParams(use_nbinom=True, nb_dispersion=1.5))
    var_p = float((pois * k**2).sum() - (pois * k).sum() ** 2)
    var_nb = float((nb * k**2).sum() - (nb * k).sum() ** 2)
    assert var_nb > var_p


def test_market_probs_in_unit_interval():
    m = Matchup(TeamInputs(), TeamInputs(), 1.0)
    probs = market_probabilities(m, ou_line=8.5, handicap=-1.5)
    for key in ("WIN", "OU", "HDC"):
        assert 0.0 <= probs[key] <= 1.0


def test_over_prob_monotonic_decreasing_in_line():
    home = score_distribution(5.0)
    away = score_distribution(4.5)
    lines = [5.5, 7.5, 9.5, 11.5, 13.5]
    overs = [prob_over(home, away, ln) for ln in lines]
    assert all(overs[i] >= overs[i + 1] for i in range(len(overs) - 1))


def test_handicap_prob_monotonic_decreasing():
    home = score_distribution(5.5)
    away = score_distribution(4.0)
    hs = [-2.5, -1.5, -0.5, 0.5, 1.5]
    covers = [prob_handicap(home, away, h) for h in hs]
    assert all(covers[i] >= covers[i + 1] for i in range(len(covers) - 1))


def test_stronger_offense_raises_home_lambda():
    base = Matchup(TeamInputs(), TeamInputs(), 1.0)
    strong = Matchup(TeamInputs(offense_woba_z=2.0), TeamInputs(), 1.0)
    lh0, _ = estimate_lambda(base)
    lh1, _ = estimate_lambda(strong)
    assert lh1 > lh0


def test_better_opposing_starter_lowers_home_lambda():
    # away 선발 FIP-z 가 낮을수록(좋은 투수) 홈 득점 기대치↓.
    good_opp = Matchup(TeamInputs(), TeamInputs(starter_fip_z=-2.0), 1.0)
    bad_opp = Matchup(TeamInputs(), TeamInputs(starter_fip_z=2.0), 1.0)
    lh_good, _ = estimate_lambda(good_opp)
    lh_bad, _ = estimate_lambda(bad_opp)
    assert lh_bad > lh_good


def test_home_advantage_applied():
    m = Matchup(TeamInputs(), TeamInputs(), 1.0)
    lh, la = estimate_lambda(m)
    # 동일 입력이면 홈 어드밴티지 때문에 홈 λ 가 더 크다.
    assert lh > la


def test_symmetric_matchup_win_prob_above_half():
    m = Matchup(TeamInputs(), TeamInputs(), 1.0)
    home = score_distribution(estimate_lambda(m)[0])
    away = score_distribution(estimate_lambda(m)[1])
    assert prob_home_win(home, away) > 0.5
