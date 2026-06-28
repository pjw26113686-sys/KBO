"""KBO 분석 · 스코어 예측 · EV 베팅 시스템.

1차 구현 범위: 합성 그라운드트루스 기반 시뮬레이션 코어.
  store/   - 스키마 + SQLite
  model/   - 포아송 결합 예측 모델
  ev/      - EV 엔진
  sim/     - N경기 시뮬레이터
  backtest/- 캘리브레이션 / 커버리지 / ROI 검증
  crawler/, scheduler/ - 인터페이스 스텁 (2차 구현 예정)
"""

__version__ = "0.1.0"
