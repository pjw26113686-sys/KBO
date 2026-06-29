"""N경기 end-to-end 시뮬레이션 (설계서 8단계 5번 백테스트의 합성 버전).

흐름: generator(진실) → 모델 관측 λ → 시장 관측 λ → 시장 배당 →
      모델 확률로 EV 계산 → EV>threshold 베팅 → 실제 스코어로 정산.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..ev import engine
from ..model.poisson import market_probabilities
from . import generator, market


@dataclass
class CalibRow:
    market: str
    model_prob: float   # 해당 마켓 '기준 선택지'의 모델 확률
    outcome: int        # 그 선택지가 실제로 적중했으면 1, 아니면 0 (push 제외)


@dataclass
class BetRow:
    game_id: int
    market: str
    selection: str
    model_prob: float
    decimal_odds: float
    ev: float
    profit: float       # 스테이크 1 기준 순손익 (push=0)


@dataclass
class SimResult:
    n_games: int
    sigma_model: float
    sigma_market: float
    margin: float
    threshold: float
    calib: list[CalibRow] = field(default_factory=list)
    bets: list[BetRow] = field(default_factory=list)


def _settle_win(home: int, away: int, selection: str) -> str:
    if home == away:
        return "push"
    home_won = home > away
    if selection == "HOME":
        return "win" if home_won else "lose"
    return "win" if not home_won else "lose"  # AWAY


def _settle_ou(home: int, away: int, selection: str, line: float) -> str:
    total = home + away
    if total == line:
        return "push"
    over = total > line
    if selection == "OVER":
        return "win" if over else "lose"
    return "win" if not over else "lose"  # UNDER


def _settle_hdc(home: int, away: int, selection: str, handicap: float) -> str:
    diff = home - away
    if diff == handicap:
        return "push"
    cover = diff > handicap
    if selection == "HDC_HOME":
        return "win" if cover else "lose"
    return "win" if not cover else "lose"  # HDC_AWAY


def _profit(outcome: str, decimal_odds: float) -> float:
    if outcome == "push":
        return 0.0
    if outcome == "win":
        return decimal_odds - 1.0
    return -1.0


def run(
    n_games: int = 2000,
    n_teams: int = 10,
    sigma_model: float = 0.07,
    sigma_market: float = 0.05,
    margin: float = 0.05,
    threshold: float = 0.05,
    handicap: float = -1.5,
    seed: int = 42,
    markets: tuple[str, ...] = ("WIN", "OU", "HDC"),
) -> SimResult:
    """시뮬레이션 실행.

    모델/시장이 각각 진실 λ 를 독립적으로 노이즈 관측한다(σ_model / σ_market).
    핵심: 모델의 노이즈가 커질수록 ROI 가 하락하며, 모델이 시장보다 '충분히' 부정확해지면
    마진+승자의저주를 못 이겨 ROI<0 으로 떨어진다(크로스오버는 보통 σ_model >> σ_market).
    주의: 합성 시장은 독립 추정기라 실제 효율적 시장보다 '이기기 쉽다' → 절대 ROI 는 과대평가.
    """
    teams = generator.make_league(n_teams, seed=seed)
    games = generator.generate_games(n_games, teams, seed=seed + 1)

    rng_model = np.random.default_rng(seed + 100)
    rng_market = np.random.default_rng(seed + 200)

    res = SimResult(n_games, sigma_model, sigma_market, margin, threshold)

    for g in games:
        # 모델/시장이 각각 독립적으로 진실 λ 를 관측
        lh_m = generator.observe_lambda(g.lam_home_true, sigma_model, rng_model)
        la_m = generator.observe_lambda(g.lam_away_true, sigma_model, rng_model)
        lh_k = generator.observe_lambda(g.lam_home_true, sigma_market, rng_market)
        la_k = generator.observe_lambda(g.lam_away_true, sigma_market, rng_market)

        # 시장이 선과 배당을 정함
        mk = market.make_market(lh_k, la_k, margin=margin, handicap=handicap)

        # 모델은 시장이 내건 동일 선에 대해 확률을 산출
        mp = market_probabilities(lh_m, la_m, ou_line=mk.ou_line, handicap=handicap)

        # 마켓별 모델 확률(양면)
        p_home = mp.p_home_win / (mp.p_home_win + mp.p_away_win)
        p_over = mp.p_over / (mp.p_over + mp.p_under)
        p_cover = mp.p_home_cover

        selections: list[tuple[str, str, float]] = []
        if "WIN" in markets:
            selections += [("WIN", "HOME", p_home), ("WIN", "AWAY", 1 - p_home)]
        if "OU" in markets:
            selections += [("OU", "OVER", p_over), ("OU", "UNDER", 1 - p_over)]
        if "HDC" in markets:
            selections += [("HDC", "HDC_HOME", p_cover), ("HDC", "HDC_AWAY", 1 - p_cover)]

        # --- 캘리브레이션: 각 마켓의 '기준 선택지' 하나만 (push 제외) ---
        if "WIN" in markets and g.home_score != g.away_score:
            res.calib.append(CalibRow("WIN", p_home, int(g.home_score > g.away_score)))
        if "OU" in markets and (g.home_score + g.away_score) != mk.ou_line:
            res.calib.append(
                CalibRow("OU", p_over, int((g.home_score + g.away_score) > mk.ou_line))
            )
        if "HDC" in markets and (g.home_score - g.away_score) != handicap:
            res.calib.append(
                CalibRow("HDC", p_cover, int((g.home_score - g.away_score) > handicap))
            )

        # --- EV 후보 선정 및 정산 ---
        for mkt, sel, prob in selections:
            if prob <= 0 or prob >= 1:
                continue
            odds = mk.odds[sel]
            cand = engine.evaluate(g.game_id, mkt, sel, prob, odds, line=mk.ou_line)
            if cand.ev <= threshold:
                continue
            if mkt == "WIN":
                outcome = _settle_win(g.home_score, g.away_score, sel)
            elif mkt == "OU":
                outcome = _settle_ou(g.home_score, g.away_score, sel, mk.ou_line)
            else:
                outcome = _settle_hdc(g.home_score, g.away_score, sel, handicap)
            res.bets.append(
                BetRow(g.game_id, mkt, sel, prob, odds, cand.ev, _profit(outcome, odds))
            )

    return res
