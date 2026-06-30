"""EV 계산 엔진 (설계서 §5).

인과 순서(혼동 주의 — 반대로 생각하기 쉬움):
    모델 확률  →  공정 배당(fair odds) = 1 / model_prob
              →  실제 시장 배당과 비교
              →  EV = model_prob × 실제배당 − 1
              →  EV > 0 이면 그 쪽이 밸류 베팅

EV 는 입력이 아니라 *결과*다. 시장과 크게 어긋날수록 "내 모델이 틀렸을" 확률이
높다(설계서 §5 경고). 따라서 백테스트/캘리브레이션 통과 전에는 이 후보를 실전
신호로 쓰지 않는다(신뢰도 게이트는 backtest 모듈에서 강제).
"""

from __future__ import annotations

from dataclasses import dataclass

from .. import config


def fair_odds(model_prob: float) -> float:
    """공정 배당 = 1 / model_prob. (마진 없는 이론 배당)"""
    if model_prob <= 0:
        return float("inf")
    return 1.0 / model_prob


def ev_of_bet(model_prob: float, decimal_odds: float) -> float:
    """단위 베팅당 기대수익률 EV = model_prob × decimal_odds − 1 (설계서 §5)."""
    return model_prob * decimal_odds - 1.0


@dataclass(frozen=True)
class Candidate:
    """+EV 후보 한 건."""

    game_id: int
    market: str          # WIN / OU / HDC
    line: float | None
    model_prob: float
    decimal_odds: float
    ev: float


def filter_candidates(
    rows: list[dict],
    threshold: float = config.DEFAULT_EV_THRESHOLD,
) -> list[Candidate]:
    """EV > threshold 인 마켓만 후보로 추린다 (설계서 §5).

    rows: 각 dict 는 game_id, market, line, model_prob, decimal_odds 를 가진다.
    """
    out: list[Candidate] = []
    for r in rows:
        ev = ev_of_bet(r["model_prob"], r["decimal_odds"])
        if ev > threshold:
            out.append(
                Candidate(
                    game_id=r["game_id"],
                    market=r["market"],
                    line=r.get("line"),
                    model_prob=r["model_prob"],
                    decimal_odds=r["decimal_odds"],
                    ev=ev,
                )
            )
    return out
