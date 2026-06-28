"""EV 엔진 (설계서 §5).

인과 순서 (혼동 주의):
    모델 확률  →  공정배당(fair odds) = 1 / model_prob
              →  실제 시장 배당과 비교
              →  EV = model_prob × 실제배당 − 1
              →  EV > threshold 이면 그 쪽이 밸류 베팅

EV는 입력이 아니라 결과다. 시장과 크게 어긋날수록 "내 모델이 틀림"일 확률이 높으므로,
캘리브레이션 백테스트 통과 전까지는 EV_ENGINE_ENABLED=False 로 두어 실전 신호를 막는다.
"""

from __future__ import annotations

from dataclasses import dataclass

from .. import config


@dataclass
class BetCandidate:
    game_id: int
    market: str          # WIN / OU / HDC
    model_prob: float
    decimal_odds: float
    ev: float
    line: float | None = None


def fair_odds(model_prob: float) -> float:
    """모델 확률에 대응하는 공정 배당. 1 / p."""
    if model_prob <= 0:
        return float("inf")
    return 1.0 / model_prob


def ev(model_prob: float, decimal_odds: float) -> float:
    """기대값. (model_prob × decimal_odds) − 1.

    +이면 밸류(시장배당이 공정배당보다 후함), −이면 제외.
    """
    return model_prob * decimal_odds - 1.0


def find_candidates(
    game_id: int,
    market_probs: dict[str, float],
    market_odds: dict[str, float],
    lines: dict[str, float | None] | None = None,
    threshold: float | None = None,
    enabled: bool | None = None,
) -> list[BetCandidate]:
    """한 경기의 마켓별 EV를 계산해 threshold 초과 후보만 반환한다.

    market_probs / market_odds: {"WIN": .., "OU": .., "HDC": ..} 형태.
    enabled=False(또는 config.EV_ENGINE_ENABLED=False)면 신뢰도 게이트로 빈 리스트.
    """
    threshold = config.EV_THRESHOLD if threshold is None else threshold
    enabled = config.EV_ENGINE_ENABLED if enabled is None else enabled
    lines = lines or {}

    if not enabled:
        return []  # 신뢰도 게이트: 백테스트 미통과 시 실전 신호 차단.

    candidates: list[BetCandidate] = []
    for market, prob in market_probs.items():
        odds = market_odds.get(market)
        if odds is None:
            continue
        e = ev(prob, odds)
        if e > threshold:
            candidates.append(
                BetCandidate(
                    game_id=game_id,
                    market=market,
                    model_prob=prob,
                    decimal_odds=odds,
                    ev=e,
                    line=lines.get(market),
                )
            )
    return candidates
