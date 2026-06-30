"""Statiz(statiz.sporki.com) 크롤러 — 인터페이스 stub (설계서 §2).

세이버 지표(wOBA, FIP, xFIP, WAR, WPA), 구장별·상대팀별 성적, 좌우 스플릿 소스.
실제 구현 전까지 계약만 정의한다.
"""

from __future__ import annotations


def fetch_pitcher_sabermetrics(player_id: int) -> dict:
    """선발/불펜 세이버 지표를 반환(예정).

    반환 계약: {player_id, fip, xfip, war, splits_vs_LHB, splits_vs_RHB, ...}.
    구현 시 robots.txt 준수 + 요청 간 2~5초 지연 + 개인 용도 한정.
    """
    raise NotImplementedError("Statiz 크롤러는 아직 구현되지 않음 (계약만 정의).")


def fetch_batter_sabermetrics(player_id: int) -> dict:
    """타자 세이버 지표 반환(예정). 반환: {player_id, woba, ops, vs_LHP, vs_RHP, ...}."""
    raise NotImplementedError("Statiz 크롤러는 아직 구현되지 않음 (계약만 정의).")
