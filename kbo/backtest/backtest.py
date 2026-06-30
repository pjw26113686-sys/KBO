"""백테스트 프로토콜 + GO/NO-GO 판정 (설계서 §6, 구현 단계 5).

설계서 §6:
  1. 시간순 train/test 분할(미래 누수 금지).
  2. 각 경기 모델 확률 → 실제 결과와 비교.
  3. 지표: log-loss, Brier, 캘리브레이션.
  4. (배당 확보 시) 과거 배당으로 ROI 시뮬 — 플랫 베팅 기준.
  5. 통과 기준 미달이면 모델로 실전 신호 보내지 않음.

이 베이스라인 모델은 결과(outcome)로 *학습*하지 않으므로 정보 누수 위험은 작지만,
파라미터 튜닝 시에는 반드시 train 구간만 써야 하므로 시간 분할 헬퍼를 함께 둔다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import metrics


def time_split(n: int, test_frac: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    """시간순 정렬된 n개 인덱스를 (train, test) 로 분할. 뒤쪽이 test(미래)."""
    cut = int(round(n * (1.0 - test_frac)))
    idx = np.arange(n)
    return idx[:cut], idx[cut:]


@dataclass(frozen=True)
class BacktestResult:
    """모델 vs 시장 베이스라인 비교 + 통과 여부."""

    n: int
    model_logloss: float
    market_logloss: float
    model_brier: float
    market_brier: float
    calib_error: float
    passed: bool
    reasons: list[str] = field(default_factory=list)

    def summary(self) -> str:
        verdict = "GO ✅ (모델이 시장보다 정확 — 신호 활성)" if self.passed else "NO-GO ⛔ (실전 신호 비활성)"
        lines = [
            f"  표본 경기-마켓 수 : {self.n}",
            f"  log-loss   모델 {self.model_logloss:.4f}  vs  시장 {self.market_logloss:.4f}",
            f"  Brier      모델 {self.model_brier:.4f}  vs  시장 {self.market_brier:.4f}",
            f"  캘리브레이션 오차 : {self.calib_error:.4f}",
            f"  판정       : {verdict}",
        ]
        if self.reasons:
            lines.append("  사유       : " + "; ".join(self.reasons))
        return "\n".join(lines)


def run_backtest(
    model_probs: np.ndarray,
    market_probs: np.ndarray,
    outcomes: np.ndarray,
    calib_threshold: float = 0.05,
) -> BacktestResult:
    """모델 확률 vs 시장(디-빅) 확률을 실제 결과와 비교해 GO/NO-GO 를 판정한다.

    통과 기준(설계서 §5/§6 신뢰도 게이트):
      (1) 모델 log-loss < 시장 베이스라인 log-loss  (모델이 더 정확)
      (2) 캘리브레이션 오차 < calib_threshold
    둘 다 만족해야 passed=True.
    """
    model_probs = np.asarray(model_probs, dtype=float)
    market_probs = np.asarray(market_probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)

    m_ll = metrics.log_loss(model_probs, outcomes)
    mk_ll = metrics.log_loss(market_probs, outcomes)
    m_br = metrics.brier_score(model_probs, outcomes)
    mk_br = metrics.brier_score(market_probs, outcomes)
    calib = metrics.calibration_error(model_probs, outcomes)

    reasons: list[str] = []
    if not (m_ll < mk_ll):
        reasons.append(f"모델 log-loss({m_ll:.4f})가 시장({mk_ll:.4f})보다 낮지 않음")
    if not (calib < calib_threshold):
        reasons.append(f"캘리브레이션 오차({calib:.4f}) ≥ 기준({calib_threshold})")

    return BacktestResult(
        n=int(model_probs.size),
        model_logloss=m_ll,
        market_logloss=mk_ll,
        model_brier=m_br,
        market_brier=mk_br,
        calib_error=calib,
        passed=len(reasons) == 0,
        reasons=reasons,
    )


@dataclass(frozen=True)
class BettingResult:
    """플랫 베팅 ROI 시뮬 결과 (설계서 §6.4)."""

    n_bets: int
    n_wins: int
    hit_rate: float
    total_profit: float    # 단위 베팅 기준 누적 손익
    roi: float             # total_profit / n_bets

    def summary(self) -> str:
        if self.n_bets == 0:
            return "  베팅 후보 없음 (EV 임계값 통과 0건)"
        return (
            f"  베팅 수 {self.n_bets} | 적중 {self.n_wins} ({self.hit_rate:.1%}) | "
            f"누적손익 {self.total_profit:+.2f}u | ROI {self.roi:+.2%}"
        )


def flat_betting_roi(decimal_odds: np.ndarray, won: np.ndarray) -> BettingResult:
    """1단위 플랫 베팅 ROI. 적중 시 +(odds−1), 실패 시 −1."""
    decimal_odds = np.asarray(decimal_odds, dtype=float)
    won = np.asarray(won, dtype=bool)
    n = int(decimal_odds.size)
    if n == 0:
        return BettingResult(0, 0, 0.0, 0.0, 0.0)
    profit = np.where(won, decimal_odds - 1.0, -1.0).sum()
    n_wins = int(won.sum())
    return BettingResult(
        n_bets=n,
        n_wins=n_wins,
        hit_rate=n_wins / n,
        total_profit=float(profit),
        roi=float(profit / n),
    )
