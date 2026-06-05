# 설계: E2 검지기 기반 관측 + 라즈베리파이 HIL

- **날짜**: 2026-06-06
- **대상 마일스톤**: 1학기 마지막 발표 (다음 주) — "마무리" 수준. 2학기 연속.
- **상태**: 설계 확정, 사용자 리뷰 대기

---

## 1. 배경 & 동기

이 프로젝트는 원래 CCTV 카메라로 차량을 인식해 신호제어로 연결하려 했으나,
실제 4지 교차로에 4방향 카메라가 모두 존재하지 않아 보류하고 **SUMO 시뮬레이션 전용**으로
범위를 좁혔다(`docs/project_charter.md`). 현재까지 다음을 완료했다:

- 평가 시나리오 5종 동결(low/medium/high/asymmetric/saturated)
- smart_signal(DQN) 1000ep 학습 → `results/smart_signal.pth`
- fixed / webster / smart_signal 3자 정량 비교 (throughput 포함, `results/evaluation.csv`)

**남은 한계**: 현재 RL의 관측(state)은 sumo_rl 기본값으로, **차로 전체의 모든 차량을
TraCI로 직접 읽는 "전지적 시점"**이다. 이는 현실 배포 불가능한 관측이다.

**이번 작업의 핵심 전환**: 카메라 대신 **현장에 실재하는 검지기(SUMO의 E2 =
laneAreaDetector)**가 측정하는 정보만으로 관측을 구성해 재학습하고, 그 정책을
**라즈베리파이(엣지 장비)에 이식해 실시간 추론·제어**한다.

---

## 2. 목표 & 발표 메시지

> "카메라 없이도, 현장 검지기(E2)가 측정한 정보만으로 신호를 학습·제어하고,
> 그 정책을 엣지 장비(라즈베리파이)에서 실시간 추론한다."

입증할 결과 2가지:
- **(a) 검지기 관측으로도 우수한 제어 성능** — E2 기반 모델이 fixed/webster 기준선을
  능가하며, 전지적 관측 모델(god-view)과 견줄 만함을 정량 비교로 제시.
- **(b) 엣지 배포 실증** — 학습 모델을 라즈베리파이에서 추론, HIL 라이브 데모.

### 방법론 프레이밍 (발표 논리)
- **smart_signal(RL)** = 내가 제안하는 방법.
- **fixed**(평범한 고정신호), **webster**(고정신호의 이론적 최선) = 이겨야 할 비교 기준선.
- 실험 = "내 방법이 기존 방식들을 능가함"을 입증 (셋 중 선택이 아님).

---

## 3. 범위 (이번 발표)

### 포함
- E2 검지기 추가 + E2 기반 커스텀 관측함수 (피처 범위 = **중간(①.5)**)
- E2 관측으로 재학습 → 새 모델 `results/smart_signal_e2.pth`
- 재평가: 5시나리오 × {fixed, webster, rl_global, rl_e2} → 4자 비교 차트
- HIL: numpy 엣지 추론서버 + demo.py HIL 모드, localhost 검증 → 실물 Pi 전환

### 제외 (2학기로 연기)
- E2 풀 피처셋(②: jamLength(m)+평균속도 등 전체) 및 그에 따른 구조 튜닝
- 다중 검지기(advance + stop-bar) 구성
- 실제 교차로 모델링

### E2 피처 범위 결정 — 중간(①.5)
차로별로 다음 4개 피처를 사용한다(E2가 "공짜로" 주는 검증된 유용 피처):
1. 대기열(jamLengthInVehicles)
2. 밀도/점유 차량수(vehicleNumber 기반 정규화 density)
3. 정지차량수(lastIntervalMeanHaltingNumber 또는 현재 halting)
4. occupancy(%)

state = phase one-hot(4) + min_green flag(1) + (차로수 × 4). 진입 차로 12개 가정 시 ≈ 53D.
**정확한 차원은 구현 시 환경 probe로 자동 확정**(smart_signal.py는 이미 probe_env로
STATE_SIZE 자동 감지). DQN 은닉층(128/128)은 유지, 입력층만 새 차원에 맞춘다.

---

## 4. 아키텍처 (독립 유닛 3개)

| 유닛 | 책임 | 입력 → 출력 | 의존 |
|---|---|---|---|
| **E2 센싱** | 진입 차로별 검지기 측정 | SUMO 상태 → 검지기 raw 값 | `network/e2.add.xml` |
| **E2 관측함수** | 검지기값 → state 벡터 | 검지기 raw → state(~53D) | sumo_rl ObservationFunction |
| **엣지 추론서버** | 정책 추론 | state(JSON/TCP) → action(int) | `.npz` 가중치, numpy |

각 유닛은 독립적으로 이해·테스트 가능하며 명확한 인터페이스(검지기 ID 규약, state 벡터
스키마, JSON 메시지 규약)로만 연결된다.

---

## 5. 컴포넌트 (신규/변경 파일)

| 파일 | 신규/변경 | 역할 |
|---|---|---|
| `network/e2.add.xml` | 신규 | 진입 차로마다 `laneAreaDetector` 1개, 정지선 앞 ~75m 커버 |
| `e2_observation.py` | 신규 | sumo_rl 커스텀 ObservationFunction. 검지기값 → state(①.5 피처) |
| `smart_signal.py` | 변경 | `--obs {global,e2}` 플래그. e2면 관측함수+add.xml 로드 → `smart_signal_e2.pth` |
| `evaluate.py` | 변경 | 모드 확장 `fixed/webster/rl_global/rl_e2`(모드별 관측·모델 분리) → 4자 비교 |
| `export_weights.py` | 신규 | `.pth` → `.npz`(W/b) 변환 (Pi에서 torch 없이 쓰기 위함) |
| `edge_server.py` | 신규 | Pi/로컬 실행. `.npz` 로드, 순수 numpy 추론, TCP로 state 수신→action 회신 |
| `demo.py` | 변경 | `--hil --host --port` 추가. 로컬 torch 대신 엣지서버에 state 전송→action 수신 |

---

## 6. 데이터 흐름 (HIL 데모)

```
SUMO(PC) → E2 검지기 → e2_observation(state ~53D) → demo.py
   → [JSON/TCP] → edge_server(Pi) → numpy DQN(matmul2+ReLU+argmax) → action(int)
   → [JSON/TCP] → demo.py → env.step(action) → SUMO 신호 적용 → (반복)
```

- 결정 주기 5초(sim), LAN 왕복 <10ms → 실시간 GUI 데모에 부담 없음.
- **localhost 추론서버로 먼저 완성 → 실물 Pi는 host IP만 교체**(코드 동일).
- 메시지 규약(JSON): 요청 `{"obs": [float, ...]}`, 응답 `{"action": int}`.

---

## 7. 학습 & 평가 정책

- **reward 불변**: 실제 대기시간 기반(학습 시 특권정보 사용 OK). **관측만** 센서 현실적.
  (sim-to-real 표준 관행: 학습은 특권정보로 보상, 추론은 센서 관측만.)
- **관측 일관성 (필수)**: 학습 때 본 피처 == 추론 때 받는 피처. 따라서 Pi에 올리는 모델은
  반드시 **E2로 학습한 `smart_signal_e2.pth`** 여야 한다(god-view 모델 금지).
- 재학습 1000ep ≈ 하룻밤(기존 로그상 ~10시간). 하이퍼파라미터 재사용.
- 재평가: 5시나리오 × {fixed, webster, rl_global, rl_e2} × 8ep → `results/evaluation_e2.csv`
  + 비교 차트. `rl_global`은 기존 god-view 모델(공정 대조군).
- **HIL은 데모 전용, 정량 평가는 PC 단독**: 배치 평가는 수천 스텝을 빠르게 도는데 스텝당
  소켓 왕복을 넣으면 평가가 비현실적으로 느려진다. 평가는 기존 로컬 추론으로 수행.

---

## 8. 테스트 / 검증 포인트

1. **E2 스모크**: 알려진 시나리오에서 검지기가 정상값(vehicleNumber/jamLength) 방출.
2. **차원 일치**: E2 관측벡터 차원 == 모델 state_size(probe 값).
3. **추론 패리티 (핵심)**: numpy 추론 결과 == torch 추론 결과 (동일 state → 동일 action).
   불일치 시 HIL 전체가 오작동하므로 반드시 통과.
4. **HIL 왕복**: localhost 엣지서버로 round-trip 통과 후 실물 Pi 전환.

---

## 9. 우선순위 (일주일)

1. E2 검지기(add.xml) + 관측함수 — 토대
2. E2 재학습 → `smart_signal_e2.pth` (하룻밤)
3. 재평가 4자 비교 차트 — "E2 ≈ god-view, 둘 다 baseline 압도" 결과 (**발표 가능 최소선**)
4. HIL: export_weights → edge_server → demo --hil, localhost→Pi
5. 발표자료(비교 차트 + HIL 라이브 데모 + 슬라이드)

> 3까지가 발표 가능한 최소선. 4(HIL)는 그 위의 임팩트 요소.

---

## 10. 리스크 & 대응

| 리스크 | 영향 | 대응 |
|---|---|---|
| E2 부분관측으로 수렴 저하(특히 saturated) | 학습 실패/성능↓ | reward/state 스케일 유지, 체크포인트, 학습곡선 대조 |
| numpy↔torch 추론 불일치 | HIL 오작동 | 추론 패리티 테스트(테스트 3) 필수 |
| Pi 네트워크/로지스틱스 | HIL 데모 차질 | localhost 우선 완성, IP 교체로 전환 |
| state 차원 증가(①.5)로 학습 시간↑ | 일정 압박 | 어차피 재학습 필요분, 은닉층 유지로 영향 최소화 |
| 일주일 빠듯 | 범위 초과 | 1→3이 핵심, 4는 그 위. HIL 난항 시 localhost 데모로 대체 |

---

## 11. 열린 질문 (구현 중 확정)

- 검지기 길이 75m가 이 네트워크 차로 길이에 적절한지(차로 길이 확인 후 조정).
- 발표 라이브 데모 고정 시나리오(asymmetric 추천 — smart_signal 우위가 가장 극적).
- E2 측정 주기(detector `period`/`freq`)를 결정 주기(delta_time=5s)에 맞출지.
