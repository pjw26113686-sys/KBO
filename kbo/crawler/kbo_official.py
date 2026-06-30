"""KBO 공식(koreabaseball.com) 크롤러 — 인터페이스 stub (설계서 §2).

일정, 경기 시작 시각, 선발 예고, 당일 라인업, 불펜 등판 기록 소스.
동적 페이지 → POST 또는 Selenium 필요. 여기서는 계약만 정의.
"""

from __future__ import annotations


def fetch_schedule(date: str) -> list[dict]:
    """해당 날짜 일정 반환(예정).

    반환 계약: [{game_id, date, start_time, home, away, stadium, status}, ...].
    status 는 발송 직전 재확인 대상(설계서 §7: 우천 취소/시간 변경 대응).
    """
    raise NotImplementedError("KBO 공식 크롤러는 아직 구현되지 않음 (계약만 정의).")


def fetch_lineup(game_id: int) -> dict:
    """당일 라인업·선발 예고 반환(예정). 반환: {game_id, home_sp, away_sp, lineups...}."""
    raise NotImplementedError("KBO 공식 크롤러는 아직 구현되지 않음 (계약만 정의).")
