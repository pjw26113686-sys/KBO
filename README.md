# KBO 분석 · 스코어 예측 · EV 베팅 시스템

KBO 경기의 **승패 / 오버언더 / 핸디캡**을 포아송 모델로 예측하고, 시장 배당과 비교해
**EV(+) 후보**를 가려내는 파이프라인. 이 저장소의 1차 목표는 *돈 버는 봇*이 아니라
**"통제된 합성(몬테카를로) 환경에서 N경기를 돌려 모델→EV 파이프라인이 원리적으로 작동하는지
판단"** 하는 것이다. (설계서 8단계 중 1~6단계)

> ⚠️ 스포츠 베팅에서 안정적 +EV는 매우 어렵다(북메이커 마진 4~7% + 시장 효율성).
> 이 시스템은 "이기는 보장"이 아니라 **체계적으로 검증 가능한 가설 생성기**다.
> 백테스트 게이트를 통과하기 전에는 EV 수치를 실전 신호로 신뢰하지 않는다.

---

## 빠른 시작

```bash
pip install -r requirements.txt
pytest                                          # 25개 테스트
python -m scripts.run_simulation --n 3000 --seed 42 --plot   # 합성 백테스트 리포트
python -m scripts.tune_model --n 4000 --seed 42              # 파라미터 튜닝 데모
python -m scripts.ev_report --input examples/games.json      # 수동 배당 → EV 리포트
```

`--plot` 을 주면 `output/calibration.png`, `output/bankroll.png` 가 생성된다.

### 주요 옵션
| 옵션 | 기본값 | 의미 |
|---|---|---|
| `--n` | 2000 | 시뮬레이션 경기 수 |
| `--sigma-model` | 0.07 | 모델의 λ 추정 노이즈 (작을수록 정확) |
| `--sigma-market` | 0.05 | 시장의 λ 추정 노이즈 (작을수록 **효율적 시장**) |
| `--margin` | 0.05 | 북메이커 마진(overround) |
| `--threshold` | 0.05 | EV 후보 임계값 |
| `--seed` | 42 | 난수 시드 |

---

## 리포트 읽는 법 (판단 기준 3종)

1. **캘리브레이션 + 정확도** — log-loss, Brier, ECE, 캘리브레이션 표.
   "60%로 예측한 경기들이 실제로 ~60% 적중하는가"를 본다. `gate_check` 가 GO/NO-GO 판정.
2. **EV 베팅 ROI** — EV>threshold 후보만 플랫 베팅했을 때의 마켓별 누적 수익률.
3. **시장 대비 엣지 민감도** — `σ_model` 을 키우며 ROI 가 어디서 +→- 로 전환되는지(크로스오버).

### ⚠️ 절대 ROI 수치를 곧이곧대로 믿지 말 것
합성 시장은 진실 λ 의 **독립 노이즈 추정기**라, 실제 효율적 시장보다 **이기기 쉽다.**
따라서 절대 ROI(예: +7%)는 과대평가다. 신뢰할 것은 **상대적/질적 결론**이다:

- 모델이 정확할수록(σ_model↓) ROI↑, 캘리브레이션↑ — *방향성*.
- 모델이 충분히 부정확해지면 마진+승자의 저주를 못 이겨 **ROI<0** 로 전환 — *크로스오버의 존재*.
- 실제 시장은 σ_market 이 더 작아(효율적) 전환점이 낮고 엣지는 훨씬 희박하다.

이 패턴이 재현된다는 것이, 동일 인터페이스에 **실제 KBO 데이터**를 끼웠을 때 백테스트
결과를 신뢰할 근거가 된다.

---

## 아키텍처

```
kbo/
├── schema.py            # SQLite 스키마 (설계서 3장) + game_features(모델 입력 정제 결과)
├── store.py             # DB 연결/적재/조회 (시뮬·실데이터 공용 경로)
├── data/                # 데이터 어댑터 (generator 를 인터페이스 뒤로 숨김)
│   ├── base.py          #   GameRecord + DataSource ABC
│   ├── synthetic.py     #   SyntheticDataSource (합성)
│   └── sqlite_source.py #   SqliteDataSource + save_records (실데이터 토대)
├── model/
│   ├── params.py        # ModelParams (튜닝 가능한 λ 계수)
│   ├── poisson.py       # λ 추정 + 득점분포 + 3종 마켓 확률 + Skellam 승률 (설계서 4장)
│   └── tuning.py        # 포아송 MLE 파라미터 튜닝 + 홀드아웃 평가 (설계서 4·6장)
├── ev/
│   ├── engine.py        # 공정배당 → EV → 후보 필터 (설계서 5장, 실전·시뮬 공용)
│   └── report.py        # 수동 배당 입력 → EV 후보 → 디스코드 포맷 메시지 (설계서 7장)
├── sim/                 # 합성 리그/시장/N경기 end-to-end 몬테카를로
└── backtest/metrics.py  # log-loss/Brier/캘리브레이션/ROI/엣지 민감도 + 게이트 (설계서 6장)
scripts/
├── run_simulation.py    # 합성 백테스트 리포트 (+PNG)
├── tune_model.py        # 파라미터 튜닝 데모
└── ev_report.py         # 수동 배당 EV 리포트
examples/games.json      # ev_report 입력 예시
```

### 데이터 어댑터 (실데이터 교체 지점)
모델·튜닝·백테스트는 `generator` 가 아니라 `DataSource` 인터페이스에 의존한다.
2차에서 크롤러가 raw → 정제 후 `game_features` 테이블에 적재하면,
`SqliteDataSource` 가 동일 `GameRecord` 를 공급해 **파이프라인 변경 없이** 실데이터로 전환된다.

### 파라미터 튜닝
`ModelParams` (league_avg, home_advantage, 각 피처 지수 가중치)를 과거 데이터의 득점에 대해
**포아송 최대우도(MLE)** 로 적합한다(`scripts/tune_model.py`). 합성 데모에서는 진짜 규칙과
다른 기본값에서 출발해 튜너가 진짜 파라미터를 회복하는지 확인한다.
> 관측 노이즈(`--feature-noise`)를 키우면 가중치가 0쪽으로 줄어드는 **감쇠(attenuation,
> errors-in-variables)** 가 나타난다 — 통계적으로 정상이며 일반화 성능은 유지된다.

### 수동 배당 → EV 리포트
`examples/games.json` 처럼 경기 피처(또는 확률)와 배당을 JSON 으로 넣으면 EV 후보와
디스코드 포맷 메시지를 출력한다. **자동 스크래핑/발송은 하지 않는다.**
백테스트 게이트 통과 전에는 `--gate-passed` 없이는 항상 "참고용/비활성" 배너가 강제된다.
> 핸디캡은 홈 라인(홈 점수에 더함). 예: `-1.5` → 홈이 2점차 이상 이겨야 커버.

### 핵심 인과 순서 (EV는 입력이 아니라 결과)
```
모델 확률 → 공정배당(1/p) → 실제 시장배당과 비교 → EV = p×배당 − 1 → EV>0 이면 밸류
```

---

## 핵심 가설 (tests/test_simulate.py 로 검증)

진실 λ 를 우리가 아는 통제 환경이므로 다음을 증명/반증할 수 있다:

1. `σ_model=0`(모델=진실) → 캘리브레이션 거의 완벽 → 게이트 통과.
2. 모델이 시장보다 정확 → EV 베팅 ROI **양수**.
3. 모델이 시장보다 충분히 부정확 → 마진 못 이겨 ROI **음수** (거짓 +EV를 만들지 않음).

---

## 로드맵

설계서 8단계 순서를 따른다. **백테스트 게이트(5단계) 통과 전에는 실전 신호를 발송하지 않는다.**

- [x] 스키마 + SQLite (1단계)
- [x] 포아송 모델 + 합성 백테스트/게이트 (4·5·6단계)
- [x] **데이터 어댑터** (`DataSource`) — `generator` 를 인터페이스 뒤로 분리, 실데이터 교체 토대
- [x] **파라미터 튜닝** (포아송 MLE, `scripts/tune_model.py`)
- [x] **수동 배당 입력 + EV 리포트** (`scripts/ev_report.py`, 6단계)
- [ ] Statiz / KBO공식 / MyKBO 크롤러 — 정제 후 `game_features` 적재 (2단계).
      *개인 분석·저빈도·robots.txt 준수 한정, rate limit 2~5초.*
- [ ] 과거 배당 확보 시 실데이터 ROI 백테스트.
- [ ] 디스코드 웹훅 + APScheduler (경기 시작 30분 전 발송) — **게이트 통과 후에만** (7단계).
- [ ] 날씨·ML 고도화, 조합 베팅(상관관계 보정).

> 검증되지 않은 신호 발송 기능은 **의도적으로 만들지 않았다.**
