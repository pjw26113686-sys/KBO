"""과대분산(음이항) 시뮬레이션 동작 테스트.

핵심 주장: 실제 득점이 과대분산일 때 Poisson 모델은 예측구간이 좁아 과소커버하고,
음이항 모델은 커버리지를 회복한다(설계서 §4 Step2).
"""

from kbo.backtest.metrics import interval_coverage
from kbo.sim.run import run_simulation


def _total_coverage(sim) -> float:
    return interval_coverage(
        [r.total for r in sim.records],
        [r.interval_total for r in sim.records],
    )


def test_poisson_undercovers_when_truth_overdispersed():
    """실제 과대분산 + Poisson 모델 → 커버리지가 목표(0.80)보다 확연히 낮다."""
    sim = run_simulation(n_games=2000, seed=42, true_dispersion=8.0)
    assert _total_coverage(sim) < 0.78


def test_nbinom_restores_coverage():
    """실제 과대분산 + 음이항 모델(일치) → 커버리지 회복(Poisson 대비 개선)."""
    pois = run_simulation(n_games=2000, seed=42, true_dispersion=8.0)
    nb = run_simulation(n_games=2000, seed=42,
                        true_dispersion=8.0, model_dispersion=8.0)
    assert _total_coverage(nb) > _total_coverage(pois) + 0.05


def test_defaults_unchanged_are_poisson():
    """기본(dispersion 미지정)은 Poisson 과 동일 — 1차 회귀 없음."""
    a = run_simulation(n_games=300, seed=1)
    b = run_simulation(n_games=300, seed=1, true_dispersion=None,
                       model_dispersion=None)
    assert [r.total for r in a.records] == [r.total for r in b.records]
