"""수동 배당 입력 EV 리포트 CLI (설계서 6단계).

경기/배당이 담긴 JSON 을 읽어 EV 후보 + 디스코드 포맷 메시지를 출력한다.
디스코드 '발송'은 하지 않는다(설계서: 검증 전 신호 발송 금지).

사용:
    python -m scripts.ev_report --input examples/games.json
    cat examples/games.json | python -m scripts.ev_report
    python -m scripts.ev_report --input examples/games.json --gate-passed   # 게이트 통과 가정
"""
from __future__ import annotations

import argparse
import json
import sys

from kbo.ev import report


def main():
    ap = argparse.ArgumentParser(description="수동 배당 입력 EV 리포트")
    ap.add_argument("--input", help="게임 JSON 파일 경로 (없으면 stdin)")
    ap.add_argument("--threshold", type=float, default=0.05, help="EV 후보 임계값")
    ap.add_argument("--gate-passed", action="store_true",
                    help="백테스트 게이트 통과를 가정(경고 배너 제거). 통과 전엔 쓰지 말 것.")
    args = ap.parse_args()

    raw = open(args.input, encoding="utf-8").read() if args.input else sys.stdin.read()
    data = json.loads(raw)
    games = data["games"] if isinstance(data, dict) else data

    msg = report.build_message(
        games, threshold=args.threshold, gate_passed=args.gate_passed
    )
    print(msg)


if __name__ == "__main__":
    main()
