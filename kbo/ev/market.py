"""시뮬레이션용 북메이커 (실전 아님).

설계서 §5: 실전 배당은 1단계에서 *수동 입력*하고 자동 스크래핑은 보류한다.
하지만 시스템을 N경기 시뮬레이션으로 판단하려면(사용자 요청) 과거 배당을 대신할
'합성 시장 배당'이 필요하다. 이 모듈이 그 역할을 한다.

핵심 모델링:
  - 북메이커는 어떤 확률 `market_prob` 로 한쪽을 평가하고, 양쪽 implied 확률 합이
    1+margin(overround) 이 되도록 배당을 깐다(비례 vig 방식).
  - 시장이 *모든* 정보를 정확히 반영하지는 않는다. 특히 설계서 §2가 지목한
    "불펜 연투 + 일정 피로"를 시장이 부분적으로만 반영하면, 그 부분을 제대로 본
    모델에 엣지가 생긴다. '시장이 무엇을 보는가'는 runner 의 시나리오에서 결정하고,
    이 모듈은 받은 확률을 배당으로 변환 + 노이즈만 담당한다.
"""

from __future__ import annotations

import numpy as np

from .. import config


def decimal_odds(market_prob: float, margin: float = config.DEFAULT_MARGIN) -> float:
    """한쪽 확률 → 마진 포함 decimal 배당.

    implied_prob = market_prob × (1 + margin) → odds = 1 / implied_prob.
    두 사이드 implied 합 = (p + (1−p))(1+margin) = 1+margin (overround).
    """
    market_prob = float(np.clip(market_prob, 1e-6, 1 - 1e-6))
    implied = market_prob * (1.0 + margin)
    return 1.0 / implied


def make_odds(
    perceived_prob: float,
    margin: float = config.DEFAULT_MARGIN,
    rng: np.random.Generator | None = None,
    noise_sd: float = 0.0,
) -> float:
    """북메이커가 인지한 확률(+선택적 노이즈)에 마진을 입혀 배당을 만든다.

    noise_sd>0 이면 인지 확률에 가우시안 노이즈를 더해(시장도 완벽하지 않음을 표현)
    배당을 흔든다.
    """
    p = perceived_prob
    if noise_sd > 0 and rng is not None:
        p = float(np.clip(p + rng.normal(0.0, noise_sd), 1e-6, 1 - 1e-6))
    return decimal_odds(p, margin)
