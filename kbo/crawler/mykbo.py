"""MyKBO Stats 크롤러 — 스켈레톤 (설계서 §2).

MyKBO Stats(mykbostats.com)는 영어 기반 정적 페이지라 셋 중 파싱이 가장 쉽다.
당일 선발·순위·전날 결과가 매일 갱신된다.

⚠️ 중요: 이 코드는 **라이브 HTML 에 대고 검증되지 않았다.**
   개발 환경에서 외부 접속이 차단되어 실제 셀렉터를 확인할 수 없었다.
   → 아래 parse_* 함수의 CSS 셀렉터는 **로컬에서 실제 페이지 구조를 보고 완성**해야 한다.
   (브라우저 개발자도구로 DOM 확인 후 select() 인자 교체)

준수 사항(설계서 §2):
   - robots.txt 확인 후 허용 범위만
   - 요청 간 2~5초 rate limit
   - 식별 가능한 User-Agent
   - 개인 분석·저빈도 용도 한정
"""

from __future__ import annotations

import time
from dataclasses import dataclass

BASE_URL = "https://mykbostats.com"
REQUEST_DELAY_SEC = 3.0
USER_AGENT = "kbo-personal-analysis/0.1 (individual research)"


@dataclass
class ScheduledGame:
    """크롤 결과를 모델 입력(GameFeatures) 으로 연결하기 위한 중간 표현."""
    date: str
    start_time: str
    home: str
    away: str
    home_sp: str
    away_sp: str
    stadium: str


def _get(path: str) -> str:
    """rate-limit 을 지켜 HTML 을 가져온다. requests 는 함수 내 지연 임포트."""
    import requests  # 선택적 의존성: 크롤러 사용 시에만 필요.

    time.sleep(REQUEST_DELAY_SEC)
    resp = requests.get(
        f"{BASE_URL}{path}",
        headers={"User-Agent": USER_AGENT},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.text


def parse_schedule(html: str, date: str) -> list[ScheduledGame]:
    """일정 페이지 HTML → ScheduledGame 리스트.

    TODO(로컬): 실제 DOM 을 보고 셀렉터를 채운다. 예시 골격:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        for row in soup.select(".games .game"):      # ← 실제 클래스로 교체
            home = row.select_one(".home-team").get_text(strip=True)
            ...
    """
    raise NotImplementedError(
        "parse_schedule: 라이브 HTML 미검증. 로컬에서 셀렉터를 완성하세요."
    )


def fetch_today_schedule(date: str) -> list[ScheduledGame]:
    """당일 일정을 크롤링한다(로컬 전용). 외부망 차단 환경에서는 동작 불가."""
    html = _get(f"/schedule/{date}")   # ← 실제 경로로 교체 필요
    return parse_schedule(html, date)
