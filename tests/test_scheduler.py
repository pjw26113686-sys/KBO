"""스케줄러 골격 테스트 (실시간 대기 없이 함수 직접 호출)."""

from datetime import datetime

from apscheduler.schedulers.background import BackgroundScheduler

from kbo.scheduler.jobs import compute_run_time, run_game_job, schedule_games


def _game(status: str = "scheduled", start: str = "2030-01-01T18:30:00") -> dict:
    return {
        "game_id": 1,
        "home": "HOME", "away": "AWAY",
        "start_datetime": start,
        "status": status,
        "features": {
            "home_woba": 0.330, "away_woba": 0.330,
            "home_sp_fip": 4.20, "away_sp_fip": 4.20,
        },
        "odds": {"OU_OVER": 2.2},
    }


def test_compute_run_time_30min_before():
    start = datetime(2026, 6, 28, 18, 30)
    assert compute_run_time(start, 30) == datetime(2026, 6, 28, 18, 0)


def test_run_game_job_dry_run_sends():
    res = run_game_job(_game(), dry_run=True)
    assert res["sent"] is True
    assert res["result"] is not None


def test_run_game_job_skips_postponed():
    res = run_game_job(_game(status="postponed"), dry_run=True)
    assert res["sent"] is False
    assert res["result"] is None


def test_run_game_job_custom_status_check():
    """status_check 훅이 우천을 반환하면 스킵."""
    res = run_game_job(_game(), dry_run=True,
                       status_check=lambda g: "canceled")
    assert res["sent"] is False


def test_schedule_games_registers_jobs_at_offset():
    sched = BackgroundScheduler()
    jobs = schedule_games(sched, [_game()], offset_min=30, dry_run=True)
    assert len(jobs) == 1
    # 18:30 시작 − 30분 = 18:00 에 실행 예정.
    assert jobs[0].trigger.run_date.hour == 18
    assert jobs[0].trigger.run_date.minute == 0
