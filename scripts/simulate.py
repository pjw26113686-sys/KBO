#!/usr/bin/env python3
"""KBO 시뮬레이션 CLI 엔트리포인트.

사용:
    python scripts/simulate.py --games 1000 --seed 42
    python scripts/simulate.py --games 5000 --db sim.sqlite
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 패키지 임포트를 위해 프로젝트 루트를 path 에 추가.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kbo import config
from kbo.backtest.metrics import print_report
from kbo.sim.run import run_simulation


def main() -> None:
    parser = argparse.ArgumentParser(description="KBO EV 베팅 시뮬레이터")
    parser.add_argument("--games", type=int, default=1000, help="시뮬레이션 경기 수")
    parser.add_argument("--seed", type=int, default=42, help="난수 시드(재현성)")
    parser.add_argument("--ou-line", type=float, default=config.DEFAULT_OU_LINE,
                        help="오버언더 총득점 기준선")
    parser.add_argument("--handicap", type=float, default=config.DEFAULT_HANDICAP,
                        help="홈팀 핸디캡(런라인)")
    parser.add_argument("--threshold", type=float, default=config.EV_THRESHOLD,
                        help="EV 후보 threshold")
    parser.add_argument("--db", type=str, default=None,
                        help="결과 적재 SQLite 경로(미지정 시 적재 안 함)")
    parser.add_argument("--bins", type=int, default=10, help="캘리브레이션 구간 수")
    args = parser.parse_args()

    sim = run_simulation(
        n_games=args.games,
        seed=args.seed,
        ou_line=args.ou_line,
        handicap=args.handicap,
        db_path=args.db,
        threshold=args.threshold,
    )
    print_report(sim, bins=args.bins)

    if args.db:
        print(f"\n결과가 SQLite 에 적재되었습니다: {args.db}")


if __name__ == "__main__":
    main()
