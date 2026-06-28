#!/usr/bin/env python3
"""파라미터 튜닝 + 워크포워드 백테스트 CLI.

사용:
    python scripts/tune.py --games 4000 --seed 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kbo.backtest.walk_forward import print_walk_forward_report, walk_forward


def main() -> None:
    parser = argparse.ArgumentParser(description="KBO 모델 파라미터 튜닝")
    parser.add_argument("--games", type=int, default=4000, help="데이터셋 경기 수")
    parser.add_argument("--seed", type=int, default=42, help="난수 시드")
    parser.add_argument("--train-frac", type=float, default=0.7,
                        help="train 비율(나머지 test)")
    args = parser.parse_args()

    res = walk_forward(n_games=args.games, seed=args.seed,
                       train_frac=args.train_frac)
    print_walk_forward_report(res)


if __name__ == "__main__":
    main()
