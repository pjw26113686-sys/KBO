"""KBO EV 베팅 시스템 CLI (설계서 §8 구현 순서).

사용 예:
  python main.py initdb --db kbo.db
  python main.py simulate --games 2000 --scenario all --margin 0.05 \\
      --threshold 0.05 --seed 42 --report-dir out
"""

from __future__ import annotations

import argparse

from kbo import config
from kbo.sim import runner
from kbo.store import db as store_db


def cmd_initdb(args: argparse.Namespace) -> None:
    store_db.init_db(args.db)
    print(f"스키마 초기화 완료 → {args.db}")


def cmd_simulate(args: argparse.Namespace) -> None:
    runner.run(
        n_games=args.games,
        scenario=args.scenario,
        margin=args.margin,
        threshold=args.threshold,
        seed=args.seed,
        report_dir=args.report_dir,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="KBO 분석·스코어 예측·EV 베팅")
    sub = p.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("initdb", help="SQLite 스키마 초기화 (설계서 단계 1)")
    p_init.add_argument("--db", default="kbo.db", help="DB 파일 경로")
    p_init.set_defaults(func=cmd_initdb)

    p_sim = sub.add_parser("simulate", help="N경기 시뮬레이션으로 판단 (모듈 3~6)")
    p_sim.add_argument("--games", type=int, default=2000, help="시뮬 경기 수")
    p_sim.add_argument(
        "--scenario", default="all", choices=["all", "edge", "fair", "worse"],
        help="시장 정보 시나리오",
    )
    p_sim.add_argument("--margin", type=float, default=config.DEFAULT_MARGIN, help="북메이커 마진")
    p_sim.add_argument(
        "--threshold", type=float, default=config.DEFAULT_EV_THRESHOLD, help="EV 후보 임계값"
    )
    p_sim.add_argument("--seed", type=int, default=42, help="난수 시드(재현용)")
    p_sim.add_argument("--report-dir", default=None, help="CSV/PNG 저장 디렉터리")
    p_sim.set_defaults(func=cmd_simulate)
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
