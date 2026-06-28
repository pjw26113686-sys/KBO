"""워크포워드 백테스트 (설계서 §6 — 미래누수 금지).

데이터를 시간순(인덱스 순)으로 train/test 분할한다(셔플 금지).
train 에서 그리드서치로 계수를 튜닝하고, test 에서 기본 계수 대비 개선 여부를 본다.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..model.poisson import ModelParams
from ..model.tuning import LabeledGame, build_dataset, evaluate, grid_search


@dataclass
class WalkForwardResult:
    n_train: int
    n_test: int
    default_test: dict          # {"log_loss":.., "brier":..}
    tuned_test: dict
    tuned_params: ModelParams
    train_default: float
    train_tuned: float


def time_split(data: list[LabeledGame],
               train_frac: float = 0.7) -> tuple[list[LabeledGame], list[LabeledGame]]:
    """앞 train_frac 은 train, 뒤는 test. 시간순 유지(셔플 없음)."""
    cut = int(len(data) * train_frac)
    return data[:cut], data[cut:]


def walk_forward(n_games: int = 4000, seed: int = 42,
                 train_frac: float = 0.7,
                 grid=None) -> WalkForwardResult:
    """전체 파이프라인: 데이터 생성 → 시간순 분할 → train 튜닝 → test 평가."""
    data = build_dataset(n_games=n_games, seed=seed)
    train, test = time_split(data, train_frac)

    tune = grid_search(train, grid)
    default = ModelParams.from_config()

    return WalkForwardResult(
        n_train=len(train),
        n_test=len(test),
        default_test=evaluate(test, default),
        tuned_test=evaluate(test, tune.best_params),
        tuned_params=tune.best_params,
        train_default=tune.default_score,
        train_tuned=tune.best_score,
    )


def print_walk_forward_report(res: WalkForwardResult) -> None:
    print("=" * 64)
    print(" 워크포워드 백테스트 리포트 (시간순 분할, 미래누수 금지)")
    print("=" * 64)
    print(f" train={res.n_train:,}경기   test={res.n_test:,}경기")
    print("\n[train] log-loss")
    print(f"    기본 계수 : {res.train_default:.4f}")
    print(f"    튜닝 계수 : {res.train_tuned:.4f}")
    print("\n[test] (관건: test 에서도 튜닝이 기본보다 낫거나 동등한가)")
    d, t = res.default_test, res.tuned_test
    print(f"    기본 계수 : log-loss={d['log_loss']:.4f}  Brier={d['brier']:.4f}")
    print(f"    튜닝 계수 : log-loss={t['log_loss']:.4f}  Brier={t['brier']:.4f}")
    improve = d["log_loss"] - t["log_loss"]
    print(f"    test 개선폭(log-loss): {improve:+.4f}  "
          f"({'개선' if improve > 0 else '악화/동등'})")
    print("\n[튜닝된 계수]")
    p = res.tuned_params
    print(f"    sp_fip_sensitivity      = {p.sp_fip_sensitivity}")
    print(f"    lineup_woba_sensitivity = {p.lineup_woba_sensitivity}")
    print(f"    home_advantage          = {p.home_advantage}")
    print("=" * 64)
    gate = "PASS" if improve >= -1e-4 else "FAIL"
    print(f" GO/NO-GO  워크포워드(과적합 아님): {gate}")
    print(" 주: 합성 데이터의 진짜 계수가 기본값 부근이라 개선폭은 작은 게 정상.")
    print("     관건은 'test 에서 악화되지 않음'(과적합 없음)이다.")
    print("=" * 64)
