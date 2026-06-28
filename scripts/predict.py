#!/usr/bin/env python3
"""수동 배당 입력 → EV 분석 → 디스코드 메시지(드라이런) CLI.

사용:
    python scripts/predict.py examples/games_sample.json
    python scripts/predict.py games.json --threshold 0.04
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kbo import config
from kbo.ev.manual import analyze_game, load_games
from kbo.scheduler.discord import send


def main() -> None:
    parser = argparse.ArgumentParser(description="수동 배당 EV 분석")
    parser.add_argument("input", help="게임/배당 JSON 경로")
    parser.add_argument("--threshold", type=float, default=config.EV_THRESHOLD,
                        help="EV 후보 threshold")
    parser.add_argument("--webhook", type=str, default=None,
                        help="디스코드 웹훅 URL(지정 시 실제 발송 시도)")
    args = parser.parse_args()

    games = load_games(args.input)
    total_candidates = 0
    for g in games:
        res = analyze_game(g, threshold=args.threshold)
        total_candidates += len(res["candidates"])
        send(res["message"], webhook_url=args.webhook,
             dry_run=args.webhook is None)
        print()

    print(f"총 {len(games)}경기 분석, EV>{args.threshold} 후보 "
          f"{total_candidates}건.")


if __name__ == "__main__":
    main()
