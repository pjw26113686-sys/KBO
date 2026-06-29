"""데이터 소스 추상화.

GameRecord 는 모델이 λ 를 계산하는 데 필요한 '정제된 피처' + (백테스트용) 실제 결과 +
(있으면) 시장 배당을 담는다. 합성/실데이터/DB 어느 소스든 이 구조로 반환한다.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ..model.poisson import TeamMatchupFeatures


@dataclass
class GameRecord:
    game_id: int
    home_features: TeamMatchupFeatures
    away_features: TeamMatchupFeatures
    # 백테스트/튜닝용 실제 결과 (예정 경기는 None)
    home_score: int | None = None
    away_score: int | None = None
    # 메타데이터(리포트 표시용)
    date: str | None = None
    start_time: str | None = None
    home_name: str | None = None
    away_name: str | None = None
    stadium: str | None = None
    # 시장 정보(있을 때만)
    ou_line: float | None = None
    handicap: float | None = None
    odds: dict[str, float] = field(default_factory=dict)  # selection -> decimal odds

    @property
    def is_final(self) -> bool:
        return self.home_score is not None and self.away_score is not None


class DataSource(ABC):
    """경기 레코드를 공급하는 소스. generator 를 이 인터페이스 뒤로 숨긴다."""

    @abstractmethod
    def games(self) -> list[GameRecord]:
        ...
