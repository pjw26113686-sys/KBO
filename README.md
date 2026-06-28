# KBO 분석 · 스코어 예측 · EV 베팅 시스템

KBO 경기의 **승패 / 오버언더 / 핸디캡**을 포아송 결합 모델로 예측하고,
시장 배당과 비교해 **EV(+) 베팅 후보**를 가려내는 시스템.

> ⚠️ 전제: 스포츠 베팅에서 안정적 +EV는 매우 어렵다(북메이커 마진 4~7%, 시장 효율성).
> 이 시스템의 목표는 "이기는 보장"이 아니라 **체계적으로 검증 가능한 가설 생성기**다.
> **백테스트(캘리브레이션) 통과 전까지 실전 신호로 사용하지 않는다.**

---

## 1차 구현 범위: 검증 가능한 시뮬레이션 코어

실제 크롤링(법적 회색지대·봇차단) 이전에, **합성 그라운드트루스**로 N경기를 돌려
파이프라인 전체(모델 → EV → 백테스트)를 먼저 검증한다.

```
kbo/
  config.py             # 리그평균득점·보정계수·vig·EV threshold 등 상수
  store/                # SQLite 스키마(설계서 §3) + 적재 헬퍼
  model/poisson.py      # λ 추정 → 득점분포 → 3종 마켓 확률 + 예측구간 (ModelParams)
  model/tuning.py       # 보정계수 그리드서치 튜닝 [2차]
  ev/engine.py          # EV = model_prob × odds − 1, threshold 필터, 신뢰도 게이트
  ev/manual.py          # 수동 배당 입력 → EV 분석 [2차]
  sim/                  # 합성 팀/투수/피로 → 진짜 λ → 실제 스코어 + 시장배당
  backtest/metrics.py   # 캘리브레이션 / 예측구간 커버리지 / ROI
  backtest/walk_forward.py # 시간순 train/test 워크포워드 백테스트 [2차]
  scheduler/discord.py  # 디스코드 메시지 포맷 + 발송(드라이런)
  scheduler/jobs.py     # APScheduler 경기별 잡(시작 30분 전) [2차]
  crawler/              # Statiz/MyKBO 스켈레톤 (외부망 차단으로 미검증)
scripts/
  simulate.py           # 시뮬레이션 + 검증 리포트
  tune.py               # 파라미터 튜닝 + 워크포워드 [2차]
  predict.py            # 수동 배당 → EV → 디스코드 드라이런 [2차]
examples/games_sample.json  # predict.py 입력 예시
tests/                  # pytest (37개)
```

## 설치 & 실행

```bash
pip install -r requirements.txt

# 1차: N경기 시뮬레이션 후 검증 리포트
python scripts/simulate.py --games 5000 --seed 42
python scripts/simulate.py --games 1000 --db sim.sqlite   # SQLite 적재

# 2차: 파라미터 튜닝 + 워크포워드 백테스트(미래누수 금지)
python scripts/tune.py --games 4000 --seed 42

# 2차: 수동 배당 입력 → EV 후보 → 디스코드 메시지(드라이런)
python scripts/predict.py examples/games_sample.json

# 테스트
python -m pytest tests/ -q
```

## 시뮬레이터가 답하는 3가지 질문

1. **캘리브레이션** — 모델이 60% 라 한 경기가 실제로 ~60% 이기는가?
   (Brier score / log-loss / 구간별 예측-실제 표)
2. **예측구간 커버리지** — 실제 총득점·점수차가 모델의 예측 구간 안에 들어오는가?
   (정수 분포라 다소 보수적=과대커버가 정상. 위험한 것은 과소커버.)
3. **EV 베팅 ROI** — +EV 후보만 플랫 베팅하면 장기 수익이 나는가?

### 시뮬레이션 설계 (왜 신뢰할 수 있나)

- **진짜 파라미터**(팀 실력·선발 FIP·불펜/일정 피로)를 분포에서 샘플 → 진짜 λ →
  실제 스코어를 Poisson 추출. 진짜 λ와 모델은 **동일 함수형**을 공유한다.
- **모델이 보는 입력에는 추정오차(노이즈)를 주입** → 모델이 진실을 완벽히 못 본다(현실적).
- **시장(배당)은 불펜·일정 피로를 절반만 반영**(blind spot). 시장이 자주 놓치는
  이 지점이 모델의 진짜 엣지가 되도록 설계 → EV 베팅이 의미 있는지 검증 가능.

### 예시 출력 (N=5000, seed=42)

```
[1] 캘리브레이션 : Brier 0.23 / 구간별 예측-실제 오차 ~0.02 (PASS)
[2] 커버리지     : 총득점 0.84, 점수차 0.83 (목표 0.80, 보수적, PASS)
[3] EV 베팅 ROI  : 전체 베팅 -3.4% (vig 드래그) vs +EV 후보 +6.9% (PASS)
 GO/NO-GO  캘리브레이션:PASS  커버리지:PASS  EV ROI:PASS
```

> 핵심: **무작위로 다 베팅하면 마진(vig) 때문에 진다(-3.4%).**
> 시장의 blind spot 을 EV 필터로 골라낼 때만 양(+)의 ROI 가 나온다(+6.9%).
> 이것이 이 시스템이 검증하려는 가설이다.

## GO/NO-GO 분기 (설계서 §8)

5단계 백테스트 통과가 GO/NO-GO 분기점이다. 미통과 시 EV 엔진을 비활성
(`config.EV_ENGINE_ENABLED=False`)하여 실전 신호 발송을 막는다.

## 2차 구현 (완료): 모델 신뢰화 + 실전 워크플로우

- **파라미터 튜닝**(`model/tuning.py`) + **워크포워드 백테스트**(`backtest/walk_forward.py`):
  보정계수를 `ModelParams` 로 분리해 그리드서치로 튜닝. 시간순 train/test 분할로
  미래누수를 막고, **test 에서 악화되지 않음(과적합 없음)** 을 확인.
- **수동 배당 입력 → EV → 디스코드 드라이런**(`ev/manual.py`, `scripts/predict.py`):
  JSON 으로 배당을 붙여넣어 EV 후보 산출 + 설계서 §7 포맷 메시지 출력.
- **스케줄러 골격**(`scheduler/jobs.py`): APScheduler 로 경기 시작 30분 전 잡 등록,
  실행 시 status 재확인(우천취소/연기 스킵) 후 분석·발송.

> 신뢰도 게이트: 디스코드 발송은 기본 **드라이런**(stdout). 실제 웹훅 발송은
> 백테스트 통과 + 명시적 webhook URL 지정 시에만.

## 다음 단계 (3차)

- 실제 Statiz/KBO/MyKBO 크롤링 (외부망 허용 환경에서, robots.txt·rate-limit 준수).
  현재 `crawler/mykbo.py` 는 셀렉터 미검증 스켈레톤 — 라이브 HTML 보고 로컬 완성 필요.
- 과거 시즌 실데이터 적재 → 실데이터 기반 튜닝
- 실제 디스코드 웹훅 발송 활성화
- 날씨·음이항 분포·ML 고도화
