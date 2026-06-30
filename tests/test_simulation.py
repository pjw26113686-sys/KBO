"""시뮬레이션 end-to-end 테스트 (사용자 핵심 요청).

핵심 가설 검증:
  - 시드 고정 시 재현 가능.
  - 대수의 법칙: 시뮬 평균 득점 ≈ 진짜 λ 평균.
  - edge 시나리오는 모델이 시장보다 정확(GO), fair/worse 는 그렇지 않음(NO-GO).
"""

import numpy as np

from kbo.sim import runner, simulate, world


def test_reproducible_with_seed():
    t1, g1, _ = runner.simulate_games(300, seed=7)
    t2, g2, _ = runner.simulate_games(300, seed=7)
    assert [x.home_score for x in g1] == [x.home_score for x in g2]
    assert [x.away_score for x in g1] == [x.away_score for x in g2]


def test_law_of_large_numbers_scores_match_lambda():
    _, games, _ = runner.simulate_games(8000, seed=3)
    mean_home_score = np.mean([g.home_score for g in games])
    mean_home_lam = np.mean([g.lam_home for g in games])
    # 표본 평균 득점이 진짜 λ 평균과 가깝다(±0.15).
    assert abs(mean_home_score - mean_home_lam) < 0.15


def test_edge_scenario_beats_market():
    teams, games, tp = runner.simulate_games(6000, seed=11)
    rep = runner.evaluate_scenario(games, tp, "edge", obs_seed=12)
    # 시장이 피로를 놓치므로 모델 log-loss 가 더 낮아야(엣지).
    assert rep.backtest.model_logloss < rep.backtest.market_logloss
    assert rep.backtest.passed  # GO


def test_fair_scenario_does_not_beat_market():
    teams, games, tp = runner.simulate_games(6000, seed=11)
    rep = runner.evaluate_scenario(games, tp, "fair", obs_seed=12)
    # 동일 정보면 모델이 시장을 이기지 못한다 → NO-GO.
    assert not rep.backtest.passed


def test_edge_roi_better_than_fair():
    teams, games, tp = runner.simulate_games(6000, seed=11)
    edge = runner.evaluate_scenario(games, tp, "edge", obs_seed=12)
    fair = runner.evaluate_scenario(games, tp, "fair", obs_seed=12)
    assert edge.betting.roi > fair.betting.roi
