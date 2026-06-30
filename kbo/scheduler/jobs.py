"""잡 스케줄링 골격 (설계서 §7) — stub.

매일 오전 일정 크롤링 → 경기별 'start_time − 30분'에 개별 잡 등록(APScheduler 등).
실제 스케줄러 연동은 미구현. 시각 계산 등 순수 로직만 제공.
"""

from __future__ import annotations

from datetime import datetime, timedelta

LEAD_MINUTES = 30  # 경기 시작 N분 전 발송(설계서 §7).


def fire_time(start_time: datetime, lead_minutes: int = LEAD_MINUTES) -> datetime:
    """경기 시작 시각 → 발송 예정 시각(시작 lead_minutes 분 전)."""
    return start_time - timedelta(minutes=lead_minutes)


def register_pregame_jobs(games: list[dict]) -> None:
    """각 경기에 대해 발송 잡을 등록(예정 — stub).

    실제 구현: APScheduler 에 fire_time() 시점 잡 추가, 발송 직전 status 재확인 후
    분석→format_pick→send_webhook.
    """
    raise NotImplementedError("잡 등록은 아직 구현되지 않음 (fire_time 은 사용 가능).")
