"""경기별 잡 스케줄링 (설계서 §7) — APScheduler.

매 경기 "start_time − 30분"에 분석 잡을 등록한다.
잡 실행 시점에 status 를 재확인(우천취소/시간변경 대응)한 뒤 분석·발송한다.

신뢰도 게이트: 백테스트 통과 전까지 dry_run=True(드라이런)를 기본으로 둔다.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Callable

from .. import config
from .discord import send

# 잡 실행 시 건너뛸 경기 상태(우천취소/연기 등).
SKIP_STATUSES = {"postponed", "canceled", "cancelled", "suspended"}


def compute_run_time(start: datetime, offset_min: int = 30) -> datetime:
    """경기 시작 시각에서 offset_min 분 전 실행 시각을 계산한다."""
    return start - timedelta(minutes=offset_min)


def _default_status_check(game: dict) -> str:
    """기본 status 확인 훅. 실제로는 발송 직전 KBO 일정 재크롤로 교체."""
    return game.get("status", "scheduled")


def run_game_job(
    game: dict,
    threshold: float | None = None,
    webhook_url: str | None = None,
    dry_run: bool = True,
    status_check: Callable[[dict], str] | None = None,
) -> dict:
    """한 경기 분석 잡. status 재확인 → 분석 → (드라이런) 발송.

    반환: {"sent": bool, "status": str, "result": analyze_game 결과 or None}
    우천취소/연기 등이면 발송하지 않고 skip.
    """
    from ..ev.manual import analyze_game  # 순환 임포트 회피용 지연 임포트.

    status_check = status_check or _default_status_check
    status = status_check(game)
    if status in SKIP_STATUSES:
        print(f"[skip] game {game.get('game_id')} status={status} → 발송 안 함")
        return {"sent": False, "status": status, "result": None}

    res = analyze_game(game, threshold=threshold)
    sent = send(res["message"], webhook_url=webhook_url, dry_run=dry_run)
    return {"sent": sent, "status": status, "result": res}


def schedule_games(
    scheduler,
    games: list[dict],
    offset_min: int = 30,
    threshold: float | None = None,
    webhook_url: str | None = None,
    dry_run: bool = True,
    status_check: Callable[[dict], str] | None = None,
) -> list:
    """각 경기의 start_datetime(ISO) − offset_min 에 run_game_job 을 등록한다.

    scheduler: APScheduler Scheduler 인스턴스(BackgroundScheduler 등).
    반환: 등록된 Job 리스트.
    """
    threshold = config.EV_THRESHOLD if threshold is None else threshold
    jobs = []
    for g in games:
        start = datetime.fromisoformat(g["start_datetime"])
        run_at = compute_run_time(start, offset_min)
        job = scheduler.add_job(
            run_game_job,
            trigger="date",
            run_date=run_at,
            args=[g],
            kwargs={
                "threshold": threshold,
                "webhook_url": webhook_url,
                "dry_run": dry_run,
                "status_check": status_check,
            },
            id=f"game-{g.get('game_id')}",
            replace_existing=True,
        )
        jobs.append(job)
    return jobs
