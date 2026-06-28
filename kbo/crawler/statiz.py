"""Statiz 크롤러 — 인터페이스 스텁 (설계서 §2).

⚠️ 주의: Statiz(statiz.sporki.com)는 무단 사용 시 법적 조치를 명시하고 있다.
   실제 구현은 반드시 다음을 지켜 개인 분석·저빈도 용도로만:
     - robots.txt 준수
     - 요청 간 2~5초 rate limit
     - 식별 가능한 User-Agent 설정
     - 캐싱으로 중복 요청 최소화

1차(시뮬레이션 코어)에서는 호출되지 않으며, 함수 시그니처만 둔다.
2차에서 GameFeatures 를 채울 실데이터를 반환하도록 구현한다.
"""

from __future__ import annotations

REQUEST_DELAY_SEC = 3.0   # rate limit (2~5초 권장)
USER_AGENT = "kbo-personal-analysis/0.1 (individual research)"


def fetch_pitcher_stats(player_id: int) -> dict:
    """선발투수 세이버 지표(FIP/xFIP, 구종 구사율, 구장·상대 통산)를 반환.

    TODO(2차): robots.txt 확인 → rate-limit 적용 → 파싱.
    """
    raise NotImplementedError("Statiz 크롤러는 2차 구현 예정입니다.")


def fetch_team_batting(team_id: int) -> dict:
    """팀 타선 지표(wOBA, 최근 14일 OPS, 좌우 스플릿)를 반환."""
    raise NotImplementedError("Statiz 크롤러는 2차 구현 예정입니다.")
