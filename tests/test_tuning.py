"""튜닝 + 워크포워드 백테스트 테스트."""

from kbo.model.poisson import GameFeatures, ModelParams
from kbo.model.tuning import build_dataset, evaluate, grid_search
from kbo.backtest.walk_forward import time_split, walk_forward


def test_build_dataset_shape():
    data = build_dataset(n_games=100, seed=1)
    assert len(data) == 100
    f, h, a = data[0]
    assert isinstance(f, GameFeatures)
    assert isinstance(h, int) and isinstance(a, int)


def test_grid_search_not_worse_than_default_on_train():
    """그리드에 기본 조합이 포함되므로 train 점수는 기본보다 나쁘지 않다."""
    data = build_dataset(n_games=800, seed=2)
    res = grid_search(data)
    assert res.best_score <= res.default_score + 1e-9
    assert res.n_combos == 27   # 3×3×3


def test_time_split_disjoint_and_ordered():
    data = build_dataset(n_games=100, seed=3)
    train, test = time_split(data, train_frac=0.7)
    assert len(train) == 70 and len(test) == 30
    # 시간순 유지: test 의 첫 경기는 원본의 71번째.
    assert test[0] is data[70]
    assert train[-1] is data[69]


def test_walk_forward_no_overfit():
    """충분한 데이터(4000경기)에서 튜닝이 test 를 악화시키지 않아야 한다(과적합 없음).

    합성 진짜 계수가 기본값 부근이라, 데이터가 적으면 그리드서치가 train 노이즈에
    과적합해 test 가 소폭 나빠질 수 있다 → 신뢰할 만한 튜닝엔 표본이 필요함을 보여준다.
    """
    res = walk_forward(n_games=4000, seed=42, train_frac=0.7)
    assert res.tuned_test["log_loss"] <= res.default_test["log_loss"] + 1e-3


def test_evaluate_returns_metrics():
    data = build_dataset(n_games=200, seed=4)
    m = evaluate(data, ModelParams.from_config())
    assert "log_loss" in m and "brier" in m
    assert 0 < m["log_loss"] < 1.0
