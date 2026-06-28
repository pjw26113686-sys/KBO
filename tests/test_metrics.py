"""백테스트 지표 및 시뮬레이션 재현성 테스트."""

import numpy as np

from kbo.backtest.metrics import (
    brier_score,
    calibration_table,
    interval_coverage,
    log_loss,
    roi_simulation,
)
from kbo.sim.run import run_simulation


def test_brier_perfect():
    assert brier_score([1.0, 0.0], [1, 0]) == 0.0


def test_brier_worst():
    assert brier_score([0.0, 1.0], [1, 0]) == 1.0


def test_log_loss_better_for_confident_correct():
    confident = log_loss([0.9, 0.1], [1, 0])
    unsure = log_loss([0.6, 0.4], [1, 0])
    assert confident < unsure


def test_calibration_table_perfect_calibration():
    rng = np.random.default_rng(0)
    probs = rng.uniform(0, 1, 20000)
    outcomes = (rng.uniform(0, 1, 20000) < probs).astype(int)
    table = calibration_table(probs, outcomes, bins=10)
    for b in table:
        assert abs(b.mean_pred - b.mean_actual) < 0.03


def test_interval_coverage_basic():
    actuals = [5, 8, 12]
    intervals = [(4, 6), (10, 14), (10, 14)]   # 1번째만 포함, 2·3번째 중 3번째만
    # 5∈[4,6] yes; 8∈[10,14] no; 12∈[10,14] yes → 2/3
    assert abs(interval_coverage(actuals, intervals) - 2 / 3) < 1e-9


def test_simulation_reproducible():
    """같은 seed → 같은 결과(재현성)."""
    a = run_simulation(n_games=50, seed=7)
    b = run_simulation(n_games=50, seed=7)
    assert [r.home_score for r in a.records] == [r.home_score for r in b.records]
    assert [r.away_score for r in a.records] == [r.away_score for r in b.records]


def test_simulation_calibration_reasonable():
    """충분한 표본에서 Brier 가 합리적 범위(<0.25)."""
    sim = run_simulation(n_games=2000, seed=42)
    probs, occ = [], []
    for rec in sim.records:
        for o in rec.outcomes:
            probs.append(o.model_prob)
            occ.append(1.0 if o.occurred else 0.0)
    assert brier_score(probs, occ) < 0.25


def test_ev_filter_beats_unfiltered():
    """+EV 후보 ROI 가 전체 베팅 baseline 보다 높다(엣지 존재 검증)."""
    sim = run_simulation(n_games=3000, seed=42)
    roi = roi_simulation(sim)
    assert roi["candidates"].n_bets > 0
    assert roi["candidates"].roi > roi["all"].roi


def test_db_persistence():
    """SQLite 적재가 동작하는지(스키마 검증)."""
    import tempfile, os, sqlite3
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "sim.sqlite")
        run_simulation(n_games=20, seed=1, db_path=path)
        conn = sqlite3.connect(path)
        n_games = conn.execute("SELECT COUNT(*) FROM games").fetchone()[0]
        n_results = conn.execute("SELECT COUNT(*) FROM results").fetchone()[0]
        n_preds = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        conn.close()
        assert n_games == 20
        assert n_results == 20
        assert n_preds == 20 * 6   # 경기당 6개 outcome
