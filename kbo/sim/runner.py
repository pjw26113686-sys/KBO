"""End-to-end 시뮬레이션 (사용자 핵심 요청).

흐름(설계서 모듈 3~6 통합):
  1. 진짜 세계 생성 → N경기 몬테카를로 시뮬 (정답지)
  2. 모델이 *노이즈 낀 관측*으로 3종 마켓 확률 산출
  3. 합성 북메이커가 마진 포함 배당 생성 (시나리오별로 시장이 보는 정보가 다름)
  4. EV 엔진으로 +EV 후보 필터
  5. 백테스트: 캘리브레이션/Brier/log-loss + 플랫베팅 ROI → GO/NO-GO
  6. 리포트(콘솔 + CSV + 캘리브레이션 PNG)

시나리오(설계서 §5 경고 (a)/(b) 실증):
  edge  : 시장이 불펜/일정 피로를 *놓침* → 그걸 본 모델에 진짜 엣지 → ROI(+)
  fair  : 시장과 모델이 같은 정보를 독립적 노이즈로 봄 → 진짜 엣지 없음 → 후보는
          잡음, ROI ≈ −마진 (NO-GO 게이트가 막아야 함)
  worse : 시장이 모델보다 정확 → 모델이 틀림 → ROI 큰 음수
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import config
from ..backtest import backtest as bt
from ..backtest import metrics
from ..ev import engine, market
from ..model.params import ModelParams
from ..model.poisson import (
    Matchup,
    TeamInputs,
    estimate_lambda,
    markets_from_lambda,
)
from . import simulate, world

HANDICAP = -1.5  # 홈 -1.5 핸디캡(흔한 라인)

# 시장은 *효율적*이다: 진짜 확률로 배당을 깐다(설계서 전제). 우리의 엣지는 오직
# 시장이 놓치는 정보(불펜/일정 피로)에서만 나온다. 그래서 시나리오 차이는
# (1) 시장이 피로를 보는가, (2) 모델 관측 노이즈가 얼마나 큰가 두 축이다.
# market_fatigue_capture: 시장이 피로 신호를 반영하는 비율(0=완전히 놓침, 1=완전 반영).
# 시장은 (놓친 피로를 제외하면) 진짜 확률을 아는 '효율적 벤치마크'다. 따라서 모델이
# 시장을 이기려면 피로에서 얻는 정보 이득이 자신의 추정 노이즈를 넘어서야 한다.
SCENARIOS: dict[str, dict] = {
    # 시장이 피로의 30%만 반영(70% 놓침) + 모델은 작은 노이즈로 거의 정확 →
    # 정보 이득 > 노이즈 손실 → 모델이 시장을 이김 → ROI(+)·GO
    "edge": dict(model_noise=0.10, market_fatigue_capture=0.3),
    # 시장이 피로를 완전 반영(=진짜 확률) + 모델은 약간 노이즈 →
    # 효율적 시장은 못 이김 → ROI≈−마진·NO-GO
    "fair": dict(model_noise=0.20, market_fatigue_capture=1.0),
    # 시장 정확 + 모델이 매우 노이즈(틀림) → ROI≈−마진(변동성↑)·NO-GO
    "worse": dict(model_noise=0.55, market_fatigue_capture=1.0),
}


def _observe(true_in: TeamInputs, noise: float, rng: np.random.Generator) -> TeamInputs:
    """모델의 노이즈 낀 관측. 모델은 모든 변수(피로 포함)를 보지만 추정 오차가 있다."""
    return TeamInputs(
        starter_fip_z=float(true_in.starter_fip_z + rng.normal(0.0, noise)),
        offense_woba_z=float(true_in.offense_woba_z + rng.normal(0.0, noise)),
        bullpen_fatigue_z=float(true_in.bullpen_fatigue_z + rng.normal(0.0, noise)),
        schedule_fatigue_z=float(true_in.schedule_fatigue_z + rng.normal(0.0, noise)),
    )


def _market_view(true_in: TeamInputs, fatigue_capture: float) -> TeamInputs:
    """시장의 인식. 노이즈 없이 진짜 값을 보되, 피로 신호는 capture 비율만큼만 반영."""
    return TeamInputs(
        starter_fip_z=true_in.starter_fip_z,
        offense_woba_z=true_in.offense_woba_z,
        bullpen_fatigue_z=true_in.bullpen_fatigue_z * fatigue_capture,
        schedule_fatigue_z=true_in.schedule_fatigue_z * fatigue_capture,
    )


@dataclass
class ScenarioReport:
    scenario: str
    backtest: bt.BacktestResult
    betting: bt.BettingResult
    calib: list[metrics.CalibrationBin]
    per_game: list[dict]  # CSV 용 행


def simulate_games(n_games: int, seed: int) -> tuple[list[world.Team], list[simulate.GameTruth], ModelParams]:
    """진짜 세계 + N경기를 한 번 생성(시나리오 비교 시 동일 경기 재사용)."""
    rng = np.random.default_rng(seed)
    true_params = world.true_world_params()
    teams = world.build_world(rng)
    games = simulate.simulate_season(teams, n_games, rng, true_params)
    return teams, games, true_params


# 실전에서 실제로 베팅 가능한 배당대(롱샷 제외). 시뮬 ROI 가 극단 롱샷 분산에
# 휘둘리지 않도록 1차/주요 마켓 수준으로 제한한다.
ODDS_MIN = 1.3
ODDS_MAX = 4.0


def evaluate_scenario(
    games: list[simulate.GameTruth],
    true_params: ModelParams,
    scenario: str,
    margin: float = config.DEFAULT_MARGIN,
    threshold: float = config.DEFAULT_EV_THRESHOLD,
    obs_seed: int = 0,
    odds_min: float = ODDS_MIN,
    odds_max: float = ODDS_MAX,
) -> ScenarioReport:
    """한 시나리오에 대해 모델/시장 관측 → 확률 → EV → 백테스트/ROI."""
    cfg = SCENARIOS[scenario]
    rng = np.random.default_rng(obs_seed)

    # 백테스트용(마켓별 대표 사이드) 누적
    model_probs: list[float] = []
    market_probs: list[float] = []
    outcomes: list[float] = []

    # 베팅(EV 후보) 누적
    bet_odds: list[float] = []
    bet_won: list[bool] = []

    per_game: list[dict] = []

    for g in games:
        # --- 관측 ---
        # 모델: 노이즈 낀 관측(피로 포함). 시장: 진짜 값(피로는 시나리오에 따라).
        model_mu = Matchup(
            _observe(g.home_true, cfg["model_noise"], rng),
            _observe(g.away_true, cfg["model_noise"], rng),
            g.park_factor_run,
        )
        mkt_mu = Matchup(
            _market_view(g.home_true, cfg["market_fatigue_capture"]),
            _market_view(g.away_true, cfg["market_fatigue_capture"]),
            g.park_factor_run,
        )

        lam_h_m, lam_a_m = estimate_lambda(model_mu, true_params)
        lam_h_k, lam_a_k = estimate_lambda(mkt_mu, true_params)

        # 시장이 OU 라인을 자기 기대총득점 근처(반점)로 건다.
        ou_line = float(np.floor(lam_h_k + lam_a_k) + 0.5)

        mp = markets_from_lambda(lam_h_m, lam_a_m, ou_line, HANDICAP, true_params)
        kp = markets_from_lambda(lam_h_k, lam_a_k, ou_line, HANDICAP, true_params)

        # --- 실제 결과 ---
        total = g.home_score + g.away_score
        diff = g.home_score - g.away_score
        out = {
            "WIN": None if diff == 0 else (1.0 if diff > 0 else 0.0),  # 무승부는 머니라인서 제외
            "OU": 1.0 if total > ou_line else 0.0,
            "HDC": 1.0 if diff > HANDICAP else 0.0,
        }

        for mkt in ("WIN", "OU", "HDC"):
            y = out[mkt]
            if y is None:
                continue
            # 백테스트: 대표(yes) 사이드 — 모델 vs 시장(디-빅) 확률.
            model_probs.append(mp[mkt])
            market_probs.append(kp[mkt])
            outcomes.append(y)

            # EV 후보: yes/no 양 사이드 모두 검토.
            for side, p_model, p_mkt, won in (
                ("yes", mp[mkt], kp[mkt], y == 1.0),
                ("no", 1 - mp[mkt], 1 - kp[mkt], y == 0.0),
            ):
                odds = market.decimal_odds(p_mkt, margin)
                if not (odds_min <= odds <= odds_max):
                    continue  # 롱샷/초강팀 제외(실전 베팅 가능 배당대만)
                ev = engine.ev_of_bet(p_model, odds)
                if ev > threshold:
                    bet_odds.append(odds)
                    bet_won.append(won)

        per_game.append(
            {
                "game_id": g.game_id,
                "home_id": g.home_id,
                "away_id": g.away_id,
                "lam_home": round(g.lam_home, 3),
                "lam_away": round(g.lam_away, 3),
                "home_score": g.home_score,
                "away_score": g.away_score,
                "ou_line": ou_line,
                "model_win": round(mp["WIN"], 4),
                "market_win": round(kp["WIN"], 4),
                "model_over": round(mp["OU"], 4),
                "market_over": round(kp["OU"], 4),
                "model_hdc": round(mp["HDC"], 4),
                "market_hdc": round(kp["HDC"], 4),
            }
        )

    backtest_result = bt.run_backtest(
        np.array(model_probs), np.array(market_probs), np.array(outcomes)
    )
    betting_result = bt.flat_betting_roi(np.array(bet_odds), np.array(bet_won))
    calib = metrics.calibration_bins(np.array(model_probs), np.array(outcomes))

    return ScenarioReport(
        scenario=scenario,
        backtest=backtest_result,
        betting=betting_result,
        calib=calib,
        per_game=per_game,
    )


# ---------------------------------------------------------------------------
# 리포트 출력
# ---------------------------------------------------------------------------
def _print_calibration(calib: list[metrics.CalibrationBin]) -> None:
    print("  캘리브레이션 (예측 → 실제):")
    for b in calib:
        if b.count == 0:
            continue
        bar = "█" * int(round(b.mean_obs * 20))
        print(
            f"    [{b.lo:.1f}-{b.hi:.1f}] n={b.count:5d}  예측 {b.mean_pred:.3f}  "
            f"실제 {b.mean_obs:.3f}  {bar}"
        )


def print_report(report: ScenarioReport) -> None:
    print(f"\n=== 시나리오: {report.scenario} ===")
    print(report.backtest.summary())
    print(report.betting.summary())
    _print_calibration(report.calib)


def save_csv(report: ScenarioReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not report.per_game:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(report.per_game[0].keys()))
        writer.writeheader()
        writer.writerows(report.per_game)


def save_calibration_png(report: ScenarioReport, path: Path) -> bool:
    """캘리브레이션 곡선 PNG 저장. matplotlib 없으면 False 반환(텍스트로 폴백)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    pred = [b.mean_pred for b in report.calib if b.count > 0]
    obs = [b.mean_obs for b in report.calib if b.count > 0]
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
    ax.plot(pred, obs, "o-", label="model")
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed frequency")
    ax.set_title(f"Calibration — scenario: {report.scenario}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return True


def run(
    n_games: int = 2000,
    scenario: str = "all",
    margin: float = config.DEFAULT_MARGIN,
    threshold: float = config.DEFAULT_EV_THRESHOLD,
    seed: int = 42,
    report_dir: str | None = None,
) -> list[ScenarioReport]:
    """시뮬레이션 실행 진입점. scenario='all' 이면 3개 시나리오 비교."""
    teams, games, true_params = simulate_games(n_games, seed)
    scenarios = list(SCENARIOS) if scenario == "all" else [scenario]

    print(f"⚾ KBO EV 시뮬레이션 | 경기수={n_games} 마진={margin:.1%} "
          f"EV임계값={threshold:.0%} seed={seed}")
    print(f"   진짜 팀: {', '.join(t.name for t in teams)}")

    reports: list[ScenarioReport] = []
    for sc in scenarios:
        rep = evaluate_scenario(games, true_params, sc, margin, threshold, obs_seed=seed + 1)
        print_report(rep)
        reports.append(rep)
        if report_dir:
            d = Path(report_dir)
            save_csv(rep, d / f"predictions_{sc}.csv")
            ok = save_calibration_png(rep, d / f"calibration_{sc}.png")
            if not ok:
                print("  (matplotlib 없음 → PNG 생략, 위 텍스트 캘리브레이션 참고)")

    if len(reports) > 1:
        print("\n=== 시나리오 비교 요약 ===")
        print(f"  {'scenario':8} {'GO?':>5} {'model_LL':>9} {'mkt_LL':>8} {'bets':>6} {'ROI':>9}")
        for r in reports:
            go = "GO" if r.backtest.passed else "NO-GO"
            print(
                f"  {r.scenario:8} {go:>5} {r.backtest.model_logloss:9.4f} "
                f"{r.backtest.market_logloss:8.4f} {r.betting.n_bets:6d} {r.betting.roi:+8.2%}"
            )
        print("\n  해석: edge 만 ROI(+)·GO 여야 하고, fair/worse 는 ROI(−)·NO-GO 여야")
        print("        설계서 §5 경고(시장과 어긋남=대개 내 모델이 틀림)가 실증된다.")

    return reports
