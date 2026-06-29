"""포아송 모델의 튜닝 가능한 파라미터 (설계서 4장: '과거 시즌으로 회귀/그리드서치 튜닝').

기본값(모든 가중치=1.0)은 리팩터 이전 동작과 동일하다. tuning 모듈이 과거 데이터로
이 값들을 최적화한다.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class ModelParams:
    """λ 곱셈 구조의 계수.

    λ = league_avg
        × offense^w_offense
        × suppression^w_suppression
        × park^w_park
        × bullpen^w_bullpen
        × schedule^w_schedule
        × (home_advantage if is_home)

    각 보정 피처는 1.0 중심의 배수이므로 지수 가중치(w_*)가 1.0이면 원래 곱과 동일하고,
    0이면 그 피처를 무시, 1보다 크면 영향을 증폭한다.
    """

    league_avg: float = 4.6
    home_advantage: float = 1.03
    w_offense: float = 1.0
    w_suppression: float = 1.0
    w_park: float = 1.0
    w_bullpen: float = 1.0
    w_schedule: float = 1.0

    # 최적화 시 벡터 <-> 객체 변환에 쓰는 필드 순서
    _ORDER = (
        "league_avg",
        "home_advantage",
        "w_offense",
        "w_suppression",
        "w_park",
        "w_bullpen",
        "w_schedule",
    )

    def to_vector(self) -> list[float]:
        d = asdict(self)
        return [float(d[k]) for k in self._ORDER]

    @classmethod
    def from_vector(cls, vec) -> "ModelParams":
        return cls(**{k: float(v) for k, v in zip(cls._ORDER, vec)})

    @classmethod
    def bounds(cls) -> list[tuple[float, float]]:
        """L-BFGS-B 용 경계. 발산/비현실 값 방지."""
        return [
            (2.0, 8.0),     # league_avg
            (0.90, 1.20),   # home_advantage
            (0.0, 3.0),     # w_offense
            (0.0, 3.0),     # w_suppression
            (0.0, 3.0),     # w_park
            (0.0, 3.0),     # w_bullpen
            (0.0, 3.0),     # w_schedule
        ]


DEFAULT_PARAMS = ModelParams()
