"""포아송 모델 파라미터 튜닝 데모 (설계서 4·5단계).

합성 데이터를 '진짜 규칙(true_params)'으로 생성하되, 모델은 다른 기본값에서 출발한다.
시간순 train/test 분할(미래 누수 금지) 후 train 으로 MLE 튜닝 → test 에서 개선 확인.

사용:
    python -m scripts.tune_model --n 4000 --seed 42
"""
from __future__ import annotations

import argparse

from kbo.data import SyntheticDataSource
from kbo.model import tuning
from kbo.model.params import DEFAULT_PARAMS, ModelParams


def main():
    ap = argparse.ArgumentParser(description="포아송 파라미터 튜닝 데모")
    ap.add_argument("--n", type=int, default=4000, help="총 경기 수")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--feature-noise", type=float, default=0.0,
                    help="모델이 보는 피처의 관측 노이즈(>0 이면 가중치 감쇠 관찰)")
    args = ap.parse_args()

    # '진짜' 규칙: 기본값과 다르게 설정(홈 어드밴티지·리그평균·일부 가중치).
    true_params = ModelParams(
        league_avg=4.9,
        home_advantage=1.08,
        w_offense=1.0,
        w_suppression=1.0,
        w_park=1.0,
        w_bullpen=1.0,
        w_schedule=1.0,
    )

    src = SyntheticDataSource(
        n_games=args.n, seed=args.seed,
        true_params=true_params, feature_noise=args.feature_noise,
    )
    records = src.games()
    cut = int(len(records) * args.train_frac)
    train, test = records[:cut], records[cut:]  # 시간순(생성 순) 분할

    # 모델은 기본 파라미터에서 출발
    init = DEFAULT_PARAMS
    res = tuning.tune(train, init=init)

    ll0, bs0, n = tuning.win_eval(test, init)
    ll1, bs1, _ = tuning.win_eval(test, res.params)

    print("=" * 64)
    print("포아송 파라미터 튜닝 리포트")
    print("=" * 64)
    print(f"경기수={args.n} (train={len(train)}, test={len(test)})  "
          f"feature_noise={args.feature_noise}  seed={args.seed}")
    print(f"최적화 성공={res.success}  반복={res.n_iter}")
    print()
    print("[Train] 포아송 음의 로그우도 (낮을수록 적합 좋음)")
    print(f"  before={res.nll_before:.4f}  ->  after={res.nll_after:.4f}  "
          f"(개선 {res.nll_before - res.nll_after:+.4f})")
    print()
    print(f"[Test] 승패 마켓 홀드아웃 (표본 {n})")
    print(f"  log-loss : {ll0:.4f}  ->  {ll1:.4f}  (개선 {ll0 - ll1:+.4f})")
    print(f"  Brier    : {bs0:.4f}  ->  {bs1:.4f}  (개선 {bs0 - bs1:+.4f})")
    print()
    print("[파라미터 회복] true / before(기본) / after(튜닝)")
    for k in ModelParams._ORDER:
        tv = getattr(true_params, k)
        bv = getattr(init, k)
        av = getattr(res.params, k)
        print(f"  {k:16s}: true={tv:6.3f}  before={bv:6.3f}  after={av:6.3f}")
    print("=" * 64)
    print("해석: train NLL 과 test log-loss/Brier 가 함께 내려가고 after 파라미터가")
    print("      true 에 근접하면, 튜닝 머신러리가 올바르게 동작하는 것이다.")
    print("      참고: --feature-noise 를 키우면 가중치가 0쪽으로 줄어드는 감쇠(attenuation,")
    print("            errors-in-variables)가 나타난다 — 통계적으로 정상이며 일반화는 여전히 개선된다.")


if __name__ == "__main__":
    main()
