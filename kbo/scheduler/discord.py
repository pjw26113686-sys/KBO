"""디스코드 발송 — 메시지 포맷 + 웹훅 스텁 (설계서 §7).

발송 자체는 2차에서 APScheduler(경기 시작 30분 전 잡)와 함께 구현한다.
신뢰도 게이트: 백테스트 통과 전까지 실전 발송 금지(설계서 §5/§8).
"""

from __future__ import annotations


def format_message(
    home: str,
    away: str,
    start_time: str,
    stadium: str,
    home_sp: str,
    away_sp: str,
    p_win: float,
    p_over: float,
    ou_line: float,
    p_handicap: float,
    handicap: float,
    candidates: list,
    note: str = "",
) -> str:
    """설계서 §7 예시 포맷의 디스코드 메시지 문자열을 만든다."""
    lines = [
        f"⚾ [{home} vs {away}] {start_time} {stadium}",
        f"선발: {home_sp} vs {away_sp}",
        f"모델: {home} 승 {p_win:.0%} | 오버 {ou_line} {p_over:.0%} "
        f"| 핸디 {handicap} {p_handicap:.0%}",
    ]
    if candidates:
        cand_str = ", ".join(
            f"[{c.market} @{c.decimal_odds:.2f} → EV {c.ev:+.3f}]" for c in candidates
        )
        lines.append(f"EV>0 후보: {cand_str} ✅")
    else:
        lines.append("EV>0 후보: 없음")
    if note:
        lines.append(f"주의: {note}")
    return "\n".join(lines)


def send_webhook(webhook_url: str, message: str) -> None:
    """디스코드 웹훅 발송 — 2차 구현 예정.

    TODO(2차): 발송 직전 경기 status(우천취소/시간변경) 재확인 후 POST.
    """
    raise NotImplementedError("디스코드 웹훅 발송은 2차 구현 예정입니다.")
