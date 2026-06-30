"""디스코드 발송 (설계서 §7).

메시지 *포맷터* 는 실제로 구현한다(순수 문자열 생성 → 테스트 가능). 실제 웹훅
전송(HTTP POST)만 stub 으로 둔다(외부 의존·동작 검증 불가).
"""

from __future__ import annotations

from ..ev.engine import Candidate


def format_pick(
    home: str,
    away: str,
    start_time: str,
    stadium: str,
    home_sp: str,
    away_sp: str,
    model_probs: dict[str, float],
    candidates: list[Candidate],
    warning: str | None = None,
) -> str:
    """설계서 §7 예시 포맷의 디스코드 메시지 문자열을 만든다."""
    lines = [
        f"⚾ [{home} vs {away}] {start_time} {stadium}",
        f"선발: {home_sp} vs {away_sp}",
        (
            f"모델: {home} 승 {model_probs.get('WIN', 0):.0%} | "
            f"오버 {model_probs.get('OU', 0):.0%} | "
            f"핸디 {model_probs.get('HDC', 0):.0%}"
        ),
    ]
    if candidates:
        picks = " / ".join(
            f"[{c.market}{'' if c.line is None else f' {c.line}'} @{c.decimal_odds:.2f} "
            f"→ EV {c.ev:+.3f}]"
            for c in candidates
        )
        lines.append(f"EV>0 후보: {picks} ✅")
    else:
        lines.append("EV>0 후보: 없음")
    if warning:
        lines.append(f"주의: {warning}")
    return "\n".join(lines)


def send_webhook(webhook_url: str, content: str) -> None:
    """디스코드 웹훅으로 전송(예정 — stub).

    실제 구현: requests.post(webhook_url, json={"content": content}). 발송 직전
    경기 status 재확인 후 호출할 것(설계서 §7).
    """
    raise NotImplementedError("디스코드 전송은 아직 구현되지 않음 (포맷터는 format_pick 참고).")
