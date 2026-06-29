"""핵심 가설 검증 (시뮬레이션이 의미를 갖는 이유).

이 테스트들이 통과하면, 동일 인터페이스에 실데이터를 끼웠을 때 백테스트 결과를
신뢰할 근거가 된다.
"""
import numpy as np
import pytest

from kbo.backtest import metrics
from kbo.sim import simulate


def _calib_arrays(res):
    return (
        np.array([c.model_prob for c in res.calib]),
        np.array([c.outcome for c in res.calib]),
    )


def test_perfect_model_is_well_calibrated():
    # σ_model=0 → 모델이 진실 λ 를 그대로 사용 → 거의 완벽 보정 → 게이트 통과.
    res = simulate.run(n_games=4000, sigma_model=0.0, sigma_market=0.15, seed=1)
    gate = metrics.gate_check(res.calib)
    assert gate.passed, gate.reasons
    assert gate.ece < 0.05


def test_model_more_accurate_than_market_is_profitable():
    # (a) 모델이 시장보다 정확(σ_model<σ_market, 효율적 시장) → EV 베팅 ROI 양수.
    res = simulate.run(
        n_games=8000, sigma_model=0.0, sigma_market=0.05, margin=0.05, seed=7
    )
    r = metrics.roi(res.bets)
    assert r.n_bets > 50
    assert r.roi > 0, f"기대: ROI>0, 실제 {r.roi:.4f}"


def test_model_worse_than_market_loses():
    # (b) σ_model > σ_market → 마진을 못 이겨 ROI 가 음수여야 한다.
    res = simulate.run(
        n_games=8000, sigma_model=0.35, sigma_market=0.05, margin=0.05, seed=11
    )
    r = metrics.roi(res.bets)
    assert r.n_bets > 50
    assert r.roi < 0, f"기대: ROI<0, 실제 {r.roi:.4f}"


def test_sweep_monotone_trend():
    # 엣지 민감도: σ_model 이 커질수록 ROI 는 대체로 하락해야 한다.
    sweep = metrics.edge_sensitivity_sweep(
        sigma_models=[0.0, 0.1, 0.2, 0.3, 0.4],
        sigma_market=0.20,
        n_games=4000,
        margin=0.05,
        threshold=0.05,
        seed=3,
    )
    roi_low = sweep[0].roi    # σ_model=0 (최강 엣지)
    roi_high = sweep[-1].roi  # σ_model=0.4 (열위)
    assert roi_low > roi_high
