"""수동 배당 입력 → EV 후보 + 디스코드 포맷 메시지 (설계서 5·7장).

1단계는 발송 직전 '수동 배당 입력'(숫자 몇 개 붙여넣기). 자동 스크래핑은 보류.
⚠️ 백테스트 게이트 통과 전에는 메시지에 '참고용/비활성' 배너를 강제한다.
"""
from __future__ import annotations

from ..model.params import ModelParams
from ..model.poisson import TeamMatchupFeatures, estimate_lambda, market_probabilities
from . import engine

# 선택지 코드 → 한글 라벨
SEL_LABEL = {
    "HOME": "홈승",
    "AWAY": "원정승",
    "OVER": "오버",
    "UNDER": "언더",
    "HDC_HOME": "핸디홈",
    "HDC_AWAY": "핸디원정",
}


def _features(d: dict, is_home: bool) -> TeamMatchupFeatures:
    return TeamMatchupFeatures(
        offense_strength=d.get("offense_strength", 1.0),
        opp_pitcher_suppression=d.get("opp_pitcher_suppression", 1.0),
        park_factor=d.get("park_factor", 1.0),
        bullpen_fatigue=d.get("bullpen_fatigue", 1.0),
        schedule_fatigue=d.get("schedule_fatigue", 1.0),
        is_home=is_home,
    )


def selection_probs(
    game: dict, params: ModelParams | None = None
) -> dict[str, float]:
    """게임 dict 에서 선택지별 모델 확률을 만든다.

    game 에 'probs' 가 있으면 그대로 사용(검증된 외부 확률 주입용).
    없으면 'features'(home/away) 로 모델이 직접 계산한다.
    """
    if "probs" in game:
        return dict(game["probs"])

    feats = game["features"]
    fh = _features(feats["home"], True)
    fa = _features(feats["away"], False)
    lh, la = estimate_lambda(fh, fa, params)
    ou_line = game.get("ou_line", 8.5)
    handicap = game.get("handicap", -1.5)
    mp = market_probabilities(lh, la, ou_line=ou_line, handicap=handicap)

    p_home = mp.p_home_win / (mp.p_home_win + mp.p_away_win)
    p_over = mp.p_over / (mp.p_over + mp.p_under)
    return {
        "HOME": p_home,
        "AWAY": 1 - p_home,
        "OVER": p_over,
        "UNDER": 1 - p_over,
        "HDC_HOME": mp.p_home_cover,
        "HDC_AWAY": 1 - mp.p_home_cover,
    }


def _market_of(selection: str) -> str:
    if selection in ("HOME", "AWAY"):
        return "WIN"
    if selection in ("OVER", "UNDER"):
        return "OU"
    return "HDC"


def evaluate_game(
    game: dict, threshold: float = 0.05, params: ModelParams | None = None
) -> list[engine.Candidate]:
    """배당이 입력된 선택지에 대해 EV 후보(EV>threshold)를 반환."""
    probs = selection_probs(game, params)
    odds = game.get("odds", {})
    cands: list[engine.Candidate] = []
    for sel, dec in odds.items():
        if sel not in probs:
            continue
        p = probs[sel]
        if not 0.0 < p < 1.0:
            continue
        line = game.get("ou_line") if _market_of(sel) == "OU" else (
            game.get("handicap") if _market_of(sel) == "HDC" else None
        )
        cands.append(engine.evaluate(
            game.get("game_id", 0), _market_of(sel), sel, p, dec, line=line
        ))
    return engine.filter_candidates(cands, threshold)


def build_message(
    games: list[dict],
    threshold: float = 0.05,
    gate_passed: bool = False,
    params: ModelParams | None = None,
) -> str:
    """경기 목록 → 디스코드 포맷 메시지 문자열."""
    lines: list[str] = []
    if not gate_passed:
        lines.append("⚠️ 백테스트 게이트 미통과 — 이 EV는 '참고용'이며 실전 신호로 쓰지 말 것.")
        lines.append("")

    for g in games:
        probs = selection_probs(g, params)
        ou_line = g.get("ou_line", 8.5)
        handicap = g.get("handicap", -1.5)
        head = f"⚾ [{g.get('home','HOME')} vs {g.get('away','AWAY')}]"
        meta = " ".join(x for x in (g.get("start_time"), g.get("stadium")) if x)
        lines.append(f"{head} {meta}".rstrip())
        lines.append(
            f"모델: 홈승 {probs['HOME']*100:.0f}% | "
            f"오버 {ou_line} {probs['OVER']*100:.0f}% | "
            f"핸디 {handicap} {probs['HDC_HOME']*100:.0f}%"
        )
        cands = evaluate_game(g, threshold, params)
        if cands:
            for c in cands:
                lines.append(
                    f"EV 후보: [{SEL_LABEL.get(c.selection, c.selection)} "
                    f"@{c.decimal_odds:.2f} → EV {c.ev*100:+.1f}%] ✅"
                )
        else:
            lines.append("EV 후보: 없음 (임계 미달)")
        if g.get("note"):
            lines.append(f"주의: {g['note']}")
        lines.append("")

    return "\n".join(lines).rstrip()
