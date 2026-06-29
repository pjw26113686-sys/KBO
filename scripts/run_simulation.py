"""N경기 합성 시뮬레이션 실행 → 판단용 리포트 출력 (사용자 핵심 산출물).

사용 예:
    python -m scripts.run_simulation --n 2000 --seed 42
    python -m scripts.run_simulation --n 2000 --sigma-model 0.20 --sigma-market 0.10  # (b) 시나리오
    python -m scripts.run_simulation --n 2000 --plot   # matplotlib 곡선 PNG 저장
"""
from __future__ import annotations

import argparse

import numpy as np

from kbo.backtest import metrics
from kbo.sim import simulate


def _fmt_pct(x: float) -> str:
    return f"{x * 100:+.2f}%"


def build_report(args) -> tuple[str, simulate.SimResult]:
    res = simulate.run(
        n_games=args.n,
        n_teams=args.teams,
        sigma_model=args.sigma_model,
        sigma_market=args.sigma_market,
        margin=args.margin,
        threshold=args.threshold,
        seed=args.seed,
    )

    probs = np.array([c.model_prob for c in res.calib])
    outs = np.array([c.outcome for c in res.calib])

    lines: list[str] = []
    lines.append("=" * 64)
    lines.append("KBO 합성 시뮬레이션 리포트")
    lines.append("=" * 64)
    lines.append(
        f"경기수={args.n}  σ_model={args.sigma_model}  σ_market={args.sigma_market}  "
        f"마진={args.margin}  EV임계={args.threshold}  seed={args.seed}"
    )
    lines.append(
        "주의: 합성 시장은 독립 노이즈 추정기 → 실제 효율적 시장보다 이기기 쉽다."
    )
    lines.append(
        "      절대 ROI 수치는 과대평가됨. '상대적/질적' 결론(엣지 방향·크로스오버)만 신뢰할 것."
    )

    # --- (1) 캘리브레이션 + 정확도 ---
    gate = metrics.gate_check(res.calib)
    lines.append("")
    lines.append("[1] 캘리브레이션 + 정확도")
    lines.append(f"  log-loss = {gate.log_loss:.4f}   (동전던지기 0.6931)")
    lines.append(f"  Brier    = {gate.brier:.4f}   (무정보 0.25)")
    lines.append(f"  ECE      = {gate.ece:.4f}")
    lines.append(f"  게이트 판정: {'PASS ✅ (GO)' if gate.passed else 'FAIL ❌ (NO-GO)'}")
    if gate.reasons:
        lines.append(f"    사유: {', '.join(gate.reasons)}")

    bins = metrics.calibration_table(probs, outs, n_bins=10)
    lines.append("  캘리브레이션 표 (예측구간 | 표본 | 평균예측 | 실제적중):")
    for b in bins:
        lines.append(
            f"    {b.lo:.1f}-{b.hi:.1f} | {b.count:5d} | {b.mean_pred:.3f} | {b.mean_actual:.3f}"
        )

    # --- (2) EV 베팅 ROI ---
    lines.append("")
    lines.append("[2] EV 베팅 ROI (플랫 베팅, 스테이크 1단위)")
    overall = metrics.roi(res.bets)
    lines.append(
        f"  전체: 베팅={overall.n_bets}  적중률={_fmt_pct(overall.win_rate)}  "
        f"총손익={overall.total_profit:+.2f}  ROI={_fmt_pct(overall.roi)}"
    )
    for mkt, r in metrics.roi_by_market(res.bets).items():
        lines.append(
            f"    {mkt:4s}: 베팅={r.n_bets:5d}  적중률={_fmt_pct(r.win_rate)}  "
            f"ROI={_fmt_pct(r.roi)}"
        )

    # --- (3) 엣지 민감도 스윕 ---
    lines.append("")
    lines.append("[3] 시장 대비 엣지 민감도 (σ_model 스윕, σ_market 고정)")
    lines.append(
        f"  σ_market={args.sigma_market} 고정. σ_model 을 키우며 ROI 가 어디서 0을 뚫고 음수로 가는지 본다."
    )
    grid = [0.0, 0.03, 0.06, 0.09, 0.12, 0.16, 0.20, 0.25, 0.30]
    sweep = metrics.edge_sensitivity_sweep(
        sigma_models=grid,
        sigma_market=args.sigma_market,
        n_games=args.n,
        margin=args.margin,
        threshold=args.threshold,
        seed=args.seed,
    )
    lines.append("  σ_model | 베팅수 | ROI     | log-loss | ECE")
    prev = None
    crossover = None
    for s in sweep:
        flag = ""
        if prev is not None and prev.roi > 0 >= s.roi:
            flag = " <- ROI 부호 전환(크로스오버)"
            crossover = (prev.sigma_model, s.sigma_model)
        lines.append(
            f"   {s.sigma_model:5.3f}  | {s.n_bets:5d}  | {_fmt_pct(s.roi):>7s} | "
            f"{s.log_loss:.4f}  | {s.ece:.4f}{flag}"
        )
        prev = s

    lines.append("")
    if crossover:
        lines.append(
            f"해석: σ_model 이 약 {crossover[0]:.2f}~{crossover[1]:.2f} 부근에서 ROI 가 +→- 로 전환된다."
        )
    else:
        lines.append("해석: 이 구간에서는 ROI 부호 전환이 관측되지 않음(시장이 더 비효율적일수록 전환점이 높아짐).")
    lines.append("      → '모델이 충분히 정확할 때만 +EV가 실현된다'는 핵심 가설이 재현된 것.")
    lines.append("      실제 시장은 이보다 훨씬 효율적(σ_market 작음)이므로 전환점이 낮고 엣지는 더 희박하다.")
    lines.append("=" * 64)

    return "\n".join(lines), res


def maybe_plot(res: simulate.SimResult, args, sweep_for_plot=None) -> list[str]:
    """matplotlib 있으면 캘리브레이션/뱅크롤 PNG 저장. 경로 리스트 반환."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    import os
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    paths: list[str] = []

    probs = np.array([c.model_prob for c in res.calib])
    outs = np.array([c.outcome for c in res.calib])
    bins = metrics.calibration_table(probs, outs, n_bins=10)

    # 캘리브레이션 곡선
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", label="perfect")
    ax.plot([b.mean_pred for b in bins], [b.mean_actual for b in bins], "o-", label="model")
    ax.set_xlabel("mean predicted probability"); ax.set_ylabel("observed frequency")
    ax.set_title(f"Calibration (sigma_model={args.sigma_model}, sigma_market={args.sigma_market})")
    ax.legend()
    p1 = os.path.join(outdir, "calibration.png")
    fig.tight_layout(); fig.savefig(p1, dpi=120); plt.close(fig)
    paths.append(p1)

    # 뱅크롤 곡선
    r = metrics.roi(res.bets)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(r.bankroll)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("bet number"); ax.set_ylabel("cumulative P/L (units)")
    ax.set_title(f"Bankroll  ROI={r.roi*100:+.2f}%  n={r.n_bets}")
    p2 = os.path.join(outdir, "bankroll.png")
    fig.tight_layout(); fig.savefig(p2, dpi=120); plt.close(fig)
    paths.append(p2)
    return paths


def main():
    ap = argparse.ArgumentParser(description="KBO 합성 시뮬레이션 (판단용 리포트)")
    ap.add_argument("--n", type=int, default=2000, help="시뮬레이션 경기 수")
    ap.add_argument("--teams", type=int, default=10)
    ap.add_argument("--sigma-model", type=float, default=0.07, help="모델 λ 추정 노이즈")
    ap.add_argument("--sigma-market", type=float, default=0.05, help="시장 λ 추정 노이즈(작을수록 효율적 시장)")
    ap.add_argument("--margin", type=float, default=0.05, help="북메이커 마진(overround)")
    ap.add_argument("--threshold", type=float, default=0.05, help="EV 후보 임계값")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--plot", action="store_true", help="PNG 곡선 저장")
    ap.add_argument("--outdir", default="output", help="PNG 저장 폴더")
    args = ap.parse_args()

    report, res = build_report(args)
    print(report)
    if args.plot:
        paths = maybe_plot(res, args)
        if paths:
            print("\n저장된 그림:", ", ".join(paths))
        else:
            print("\n(matplotlib 미설치 → 그림 생략)")


if __name__ == "__main__":
    main()
