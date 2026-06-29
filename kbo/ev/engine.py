"""EV 엔진 (설계서 5장).

인과 순서(반드시 이 방향):
  모델 확률 → 공정배당(1/p) → 실제 시장배당과 비교 → EV = p*odds - 1 → EV>0 이면 밸류.

EV 는 입력이 아니라 결과다. 실전(수동 배당 입력)과 시뮬레이션이 동일 함수를 쓴다.
"""
from __future__ import annotations

from dataclasses import dataclass


def fair_odds(model_prob: float) -> float:
    """모델 확률 → 공정 배당(decimal)."""
    if not 0.0 < model_prob <= 1.0:
        raise ValueError(f"model_prob must be in (0, 1]: {model_prob}")
    return 1.0 / model_prob


def ev(model_prob: float, decimal_odds: float) -> float:
    """기대값. 양수면 밸류 베팅. EV = p*odds - 1 (스테이크 1 기준 순기대수익)."""
    return model_prob * decimal_odds - 1.0


@dataclass
class Candidate:
    game_id: int
    market: str         # WIN / OU / HDC
    selection: str      # HOME / AWAY / OVER / UNDER ...
    line: float | None
    model_prob: float
    decimal_odds: float
    ev: float
    fair_odds: float


def evaluate(
    game_id: int,
    market: str,
    selection: str,
    model_prob: float,
    decimal_odds: float,
    line: float | None = None,
) -> Candidate:
    """단일 마켓 선택지의 EV 평가 결과를 만든다."""
    return Candidate(
        game_id=game_id,
        market=market,
        selection=selection,
        line=line,
        model_prob=model_prob,
        decimal_odds=decimal_odds,
        ev=ev(model_prob, decimal_odds),
        fair_odds=fair_odds(model_prob),
    )


def filter_candidates(candidates: list[Candidate], threshold: float = 0.05) -> list[Candidate]:
    """EV > threshold 인 후보만 남긴다 (설계서 5장: 기본 0.05)."""
    return [c for c in candidates if c.ev > threshold]
