"""합성 '진짜' 세계 생성 (ground truth).

실제 KBO 데이터(크롤러)는 봇 차단·법적 회색지대라 이 환경에서 검증할 수 없다.
대신 우리가 *진짜 팀 강도/구장/투수*를 직접 만들고, 그로부터 경기를 시뮬레이션해
모델이 그 구조를 (노이즈 낀 관측만으로) 회수하는지 판단한다.

여기서 만든 진짜 파라미터(`true_params`)와 팀 latent 값이 정답지 역할을 한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..model.params import ModelParams

# 10개 KBO 구단 (현행 리그 구성).
KBO_TEAMS = [
    "LG", "KT", "SSG", "NC", "두산", "KIA", "롯데", "삼성", "한화", "키움",
]


@dataclass(frozen=True)
class Team:
    """진짜 팀 강도(정답지)."""

    team_id: int
    name: str
    offense_z: float          # latent 타선 강도
    park_factor_run: float    # 홈 구장 득점 환경
    rotation_fip_z: list[float] = field(default_factory=list)  # 선발 로테이션의 진짜 FIP-z


def true_world_params() -> ModelParams:
    """시뮬 세계의 '진짜' 보정계수.

    설계서 §2가 지목한 대로 불펜/일정 피로가 실제로 득점에 의미 있게 작용하도록
    기본 ModelParams 보다 피로 탄력성을 약간 키운다 — 시장이 이 신호를 놓치면
    엣지가 생긴다는 가설을 시뮬로 검증하기 위함.
    """
    return ModelParams(
        starter_elasticity=0.55,
        offense_elasticity=0.35,
        bullpen_fatigue_elasticity=0.28,
        schedule_fatigue_elasticity=0.16,
    )


def build_world(rng: np.random.Generator, n_teams: int = 10) -> list[Team]:
    """진짜 팀 집합을 만든다. 시드 고정 시 재현 가능."""
    names = KBO_TEAMS[:n_teams] if n_teams <= len(KBO_TEAMS) else [f"T{i}" for i in range(n_teams)]
    teams: list[Team] = []
    for i, name in enumerate(names):
        teams.append(
            Team(
                team_id=i,
                name=name,
                offense_z=float(rng.normal(0.0, 1.0)),
                park_factor_run=float(np.clip(rng.normal(1.0, 0.05), 0.85, 1.15)),
                # 5인 로테이션, 각 선발의 진짜 FIP-z (낮을수록 좋은 투수).
                rotation_fip_z=[float(rng.normal(0.0, 1.0)) for _ in range(5)],
            )
        )
    return teams
