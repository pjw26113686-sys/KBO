"""백테스트 검증 지표 (설계서 §6).

세 가지 질문에 답한다:
  1) 캘리브레이션 — 모델이 p% 라 한 사건이 실제로 ~p% 일어나는가? (Brier/log-loss/표)
  2) 예측구간 커버리지 — 실제 결과가 모델의 예측 구간 안에 들어오는가?
  3) EV 베팅 ROI — +EV 후보만 플랫 베팅하면 장기 수익이 나는가?
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..sim.run import SimResult

_EPS = 1e-12


# ---------------------------------------------------------------------------
# 1) 캘리브레이션
# ---------------------------------------------------------------------------
def log_loss(probs, outcomes) -> float:
    p = np.clip(np.asarray(probs, dtype=float), _EPS, 1 - _EPS)
    y = np.asarray(outcomes, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier_score(probs, outcomes) -> float:
    p = np.asarray(probs, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    return float(np.mean((p - y) ** 2))


@dataclass
class CalibrationBin:
    lo: float
    hi: float
    count: int
    mean_pred: float
    mean_actual: float


def calibration_table(probs, outcomes, bins: int = 10) -> list[CalibrationBin]:
    """예측확률을 bins 구간으로 나눠 구간별 평균예측 vs 실제발생률을 낸다."""
    p = np.asarray(probs, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    table: list[CalibrationBin] = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi) if i < bins - 1 else (p >= lo) & (p <= hi)
        n = int(mask.sum())
        if n == 0:
            continue
        table.append(CalibrationBin(
            lo=lo, hi=hi, count=n,
            mean_pred=float(p[mask].mean()),
            mean_actual=float(y[mask].mean()),
        ))
    return table


# ---------------------------------------------------------------------------
# 2) 예측구간 커버리지
# ---------------------------------------------------------------------------
def interval_coverage(actuals, intervals) -> float:
    """실제값이 [lo, hi] (양끝 포함) 안에 든 비율."""
    actuals = np.asarray(actuals)
    inside = [lo <= a <= hi for a, (lo, hi) in zip(actuals, intervals)]
    return float(np.mean(inside)) if inside else float("nan")


# ---------------------------------------------------------------------------
# 3) ROI 시뮬레이션
# ---------------------------------------------------------------------------
@dataclass
class RoiResult:
    n_bets: int
    staked: float
    profit: float
    roi: float            # profit / staked
    return_std: float     # 베팅당 수익 표준편차
    hit_rate: float


def _roi_over(bet_outcomes, stake: float) -> RoiResult:
    """bet_outcomes: (odds, occurred) 리스트. 플랫 베팅 ROI."""
    if not bet_outcomes:
        return RoiResult(0, 0.0, 0.0, float("nan"), float("nan"), float("nan"))
    returns = np.array([(odds - 1.0) if occ else -1.0
                        for odds, occ in bet_outcomes]) * stake
    staked = stake * len(bet_outcomes)
    hits = np.mean([1.0 if occ else 0.0 for _, occ in bet_outcomes])
    return RoiResult(
        n_bets=len(bet_outcomes),
        staked=float(staked),
        profit=float(returns.sum()),
        roi=float(returns.sum() / staked),
        return_std=float(returns.std()),
        hit_rate=float(hits),
    )


def roi_simulation(sim: SimResult, stake: float = 1.0) -> dict:
    """+EV 후보 플랫 베팅 ROI 와, 비교용 '전체 베팅' ROI 를 함께 반환한다."""
    candidate_bets = []
    all_bets = []
    per_market: dict[str, list] = {}
    for rec in sim.records:
        for o in rec.outcomes:
            all_bets.append((o.odds, o.occurred))
            if o.is_candidate:
                candidate_bets.append((o.odds, o.occurred))
                base = o.market.split("_")[0]   # WIN / OU / HDC
                per_market.setdefault(base, []).append((o.odds, o.occurred))

    return {
        "candidates": _roi_over(candidate_bets, stake),
        "all": _roi_over(all_bets, stake),
        "per_market": {m: _roi_over(b, stake) for m, b in per_market.items()},
    }


# ---------------------------------------------------------------------------
# 리포트
# ---------------------------------------------------------------------------
def _collect_outcomes(sim: SimResult):
    """모든 outcome 의 (model_prob, occurred) 를 모은다 (캘리브레이션용)."""
    probs, occ = [], []
    for rec in sim.records:
        for o in rec.outcomes:
            probs.append(o.model_prob)
            occ.append(1.0 if o.occurred else 0.0)
    return np.array(probs), np.array(occ)


def print_report(sim: SimResult, bins: int = 10) -> None:
    """검증 리포트를 콘솔에 출력한다."""
    probs, occ = _collect_outcomes(sim)

    print("=" * 64)
    print(f" KBO 시뮬레이션 검증 리포트  (N={sim.n_games}, seed={sim.seed})")
    print(f" OU line={sim.ou_line}  Handicap={sim.handicap}")
    print("=" * 64)

    # 1) 캘리브레이션 ------------------------------------------------------
    print("\n[1] 캘리브레이션 (모든 마켓 outcome 통합)")
    print(f"    Brier score : {brier_score(probs, occ):.4f}   (낮을수록 좋음)")
    print(f"    Log-loss    : {log_loss(probs, occ):.4f}   (낮을수록 좋음)")
    print(f"    표본 outcome 수: {len(probs):,}")
    print("\n    예측확률구간   표본수   평균예측   실제발생   오차")
    print("    " + "-" * 52)
    for b in calibration_table(probs, occ, bins):
        err = b.mean_actual - b.mean_pred
        flag = "OK" if abs(err) < 0.03 else "!!"
        print(f"    [{b.lo:.1f},{b.hi:.1f})    {b.count:6d}    "
              f"{b.mean_pred:6.3f}    {b.mean_actual:6.3f}   {err:+.3f} {flag}")

    # 2) 예측구간 커버리지 ------------------------------------------------
    level = sim.records[0].interval_level if sim.records else 0.0
    tot_cov = interval_coverage(
        [r.total for r in sim.records],
        [r.interval_total for r in sim.records],
    )
    diff_cov = interval_coverage(
        [r.diff for r in sim.records],
        [r.interval_diff for r in sim.records],
    )
    print(f"\n[2] 예측구간 커버리지 (목표 ≈ {level:.2f}, 정수 분포라 다소 보수적=과대 정상)")
    print(f"    총득점  : 실제가 예측구간 안에 든 비율 = {tot_cov:.3f}")
    print(f"    점수차  : 실제가 예측구간 안에 든 비율 = {diff_cov:.3f}")

    # 3) ROI --------------------------------------------------------------
    roi = roi_simulation(sim)
    c, a = roi["candidates"], roi["all"]
    print("\n[3] EV 베팅 ROI  (플랫 베팅, 단위 스테이크)")
    print(f"    [전체 베팅 baseline] 베팅수={a.n_bets:,}  "
          f"ROI={a.roi:+.3%}  (vig 때문에 음수가 정상)")
    if c.n_bets:
        print(f"    [+EV 후보만]        베팅수={c.n_bets:,}  "
              f"적중률={c.hit_rate:.3f}  ROI={c.roi:+.3%}  "
              f"수익={c.profit:+.1f}  σ={c.return_std:.2f}")
        for m, r in roi["per_market"].items():
            print(f"        - {m:4s}: 베팅수={r.n_bets:5d}  ROI={r.roi:+.3%}")
    else:
        print("    [+EV 후보만] 후보 없음 "
              "(EV_ENGINE_ENABLED=False 또는 threshold 초과 없음)")

    # GO / NO-GO 요약 -----------------------------------------------------
    print("\n" + "=" * 64)
    calib_ok = brier_score(probs, occ) < 0.25
    # 정수 분포의 예측구간은 과대커버(보수적)가 정상 — 위험한 것은 과소커버다.
    # 따라서 'level 보다 크게 모자라지 않으면' PASS 로 본다.
    cov_ok = (tot_cov >= level - 0.04) and (diff_cov >= level - 0.04)
    roi_ok = c.n_bets > 0 and c.roi > 0
    print(f" GO/NO-GO  캘리브레이션:{'PASS' if calib_ok else 'FAIL'}  "
          f"커버리지:{'PASS' if cov_ok else 'FAIL'}  "
          f"EV ROI:{'PASS' if roi_ok else 'FAIL'}")
    print("=" * 64)
