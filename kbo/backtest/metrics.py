"""백테스트 지표 (설계서 6장).

판단 기준 3종:
  (1) 캘리브레이션 + 정확도: log-loss, Brier, 캘리브레이션 표
  (2) EV 베팅 ROI: 플랫 베팅 누적 수익률 + 뱅크롤 곡선
  (3) 시장 대비 엣지 민감도: σ_model 스윕 → ROI/log-loss
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..sim import simulate
from ..sim.simulate import BetRow, CalibRow


# ----------------------------- (1) 정확도/캘리브레이션 -----------------------------

def log_loss(probs: np.ndarray, outcomes: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(probs, eps, 1 - eps)
    y = outcomes.astype(float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
    return float(np.mean((probs - outcomes.astype(float)) ** 2))


@dataclass
class CalibrationBin:
    lo: float
    hi: float
    count: int
    mean_pred: float
    mean_actual: float


def calibration_table(
    probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10
) -> list[CalibrationBin]:
    """예측확률 bin 별 (평균 예측, 실제 적중률). 잘 보정되면 둘이 비슷해야 한다."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[CalibrationBin] = []
    idx = np.clip(np.digitize(probs, edges) - 1, 0, n_bins - 1)
    for b in range(n_bins):
        mask = idx == b
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        bins.append(
            CalibrationBin(
                lo=float(edges[b]),
                hi=float(edges[b + 1]),
                count=cnt,
                mean_pred=float(probs[mask].mean()),
                mean_actual=float(outcomes[mask].astype(float).mean()),
            )
        )
    return bins


def calibration_error(bins: list[CalibrationBin]) -> float:
    """ECE (Expected Calibration Error): bin별 |예측-실제| 의 가중 평균."""
    total = sum(b.count for b in bins)
    if total == 0:
        return float("nan")
    return sum(b.count * abs(b.mean_pred - b.mean_actual) for b in bins) / total


# ----------------------------- (2) ROI -----------------------------

@dataclass
class RoiSummary:
    n_bets: int
    total_profit: float
    roi: float                 # 플랫 베팅 1단위 기준 평균 순손익
    bankroll: np.ndarray       # 누적 뱅크롤 곡선(시작 0)
    win_rate: float


def roi(bets: list[BetRow]) -> RoiSummary:
    if not bets:
        return RoiSummary(0, 0.0, 0.0, np.array([0.0]), 0.0)
    profits = np.array([b.profit for b in bets])
    bankroll = np.concatenate([[0.0], np.cumsum(profits)])
    wins = int(np.sum(profits > 0))
    return RoiSummary(
        n_bets=len(bets),
        total_profit=float(profits.sum()),
        roi=float(profits.mean()),
        bankroll=bankroll,
        win_rate=wins / len(bets),
    )


def roi_by_market(bets: list[BetRow]) -> dict[str, RoiSummary]:
    out: dict[str, RoiSummary] = {}
    for mkt in sorted({b.market for b in bets}):
        out[mkt] = roi([b for b in bets if b.market == mkt])
    return out


# ----------------------------- 게이트 판정 (설계서 6장) -----------------------------

@dataclass
class GateResult:
    log_loss: float
    brier: float
    ece: float
    passed: bool
    reasons: list[str]


def gate_check(
    calib: list[CalibRow],
    *,
    max_log_loss: float = 0.68,   # 동전던지기(0.693)보다 나아야 함
    max_brier: float = 0.25,      # 0.25 = 항상 0.5 예측 수준
    max_ece: float = 0.05,
) -> GateResult:
    """모델이 실전 신호로 쓸 자격이 있는지 판정. 하나라도 미달이면 NO-GO."""
    probs = np.array([c.model_prob for c in calib])
    outs = np.array([c.outcome for c in calib])
    ll = log_loss(probs, outs)
    bs = brier_score(probs, outs)
    ece = calibration_error(calibration_table(probs, outs))

    reasons: list[str] = []
    if ll > max_log_loss:
        reasons.append(f"log-loss {ll:.4f} > {max_log_loss}")
    if bs > max_brier:
        reasons.append(f"brier {bs:.4f} > {max_brier}")
    if ece > max_ece:
        reasons.append(f"ECE {ece:.4f} > {max_ece}")
    return GateResult(ll, bs, ece, passed=not reasons, reasons=reasons)


# ----------------------------- (3) 엣지 민감도 스윕 -----------------------------

@dataclass
class SweepRow:
    sigma_model: float
    n_bets: int
    roi: float
    log_loss: float
    ece: float


def edge_sensitivity_sweep(
    sigma_models: list[float],
    *,
    sigma_market: float,
    n_games: int,
    margin: float,
    threshold: float,
    seed: int,
) -> list[SweepRow]:
    """σ_model 을 바꿔가며 (모델 정확도 ↔ 시장 정확도) ROI/log-loss 변화를 본다.

    σ_model < σ_market 영역에서 ROI>0, 그 위에서 ROI<0 으로 갈리는 것이 핵심 검증.
    """
    rows: list[SweepRow] = []
    for sm in sigma_models:
        r = simulate.run(
            n_games=n_games,
            sigma_model=sm,
            sigma_market=sigma_market,
            margin=margin,
            threshold=threshold,
            seed=seed,
        )
        probs = np.array([c.model_prob for c in r.calib])
        outs = np.array([c.outcome for c in r.calib])
        rs = roi(r.bets)
        rows.append(
            SweepRow(
                sigma_model=sm,
                n_bets=rs.n_bets,
                roi=rs.roi,
                log_loss=log_loss(probs, outs),
                ece=calibration_error(calibration_table(probs, outs)),
            )
        )
    return rows
