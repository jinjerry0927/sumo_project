# 실시간 HIL 대시보드 — 설계문서

> 작성 2026-06-07. 발표 라이브 데모용. `demo.py` 실행 중 SUMO GUI 옆에 띄워 "E2 검지기가 본 것 → SmartSignal 두뇌의 결정 → 신호 변화"를 매 스텝 실시간 시각화한다.

## 1. 목표 / 배경

발표에서 SUMO GUI만 틀면 차량 흐름과 신호색은 보이지만, **모델이 실제로 받는 입력(E2 검지기 관측)** 과 **두뇌의 결정 과정**은 보이지 않는다. 표로만 결과를 보여주기엔 설득력이 약하다. 따라서 SUMO GUI **옆에** 나란히 띄우는 별도 웹 대시보드를 만들어, SUMO가 못 보여주는 추상(센서값·결정·Q값)을 실시간으로 드러낸다.

핵심 메시지: *"god-view(전지적) 없이, 현장 설치형 E2 검지기 관측만으로, 라즈베리파이가 실시간 추론해 신호를 제어한다."*

## 2. 레이아웃 (확정)

"센서 우선" 하이브리드 — 큰 글씨·적은 숫자·한눈에:

- **상단 지표 스트립**: 현재 대기차량 / 평균 대기(▼ 줄면 초록) / 누적 통과 / 현재 현시(P0~P3 + 한글명).
- **좌측(메인): E2 차로별 대기열 막대 — 12차로 전부** (4방향 × 우회전/직진/좌회전, 방향별 그룹). 막대 = 검지 차량수, 회색=우회전/초록=직진/주황=좌회전. 우회전(_0)은 상시 비보호라 신호 영향은 적지만 차선별 대기 분포를 완전히 보여주기 위해 표시.
- **우측: SmartSignal 두뇌** — KEEP/SWITCH 크게(전환 시 강조) + Q값 막대 2개(유지/전환).
- **하단 범례** + "엣지: Raspberry Pi · numpy · 53D→128→128→2" 한 줄.

## 3. 아키텍처

```
demo.py 제어루프 ──매 스텝 state push──▶ Dashboard(메모리 최신상태) ◀──200ms 폴링── 브라우저(dashboard.html)
   │  obs/action/phase/wait/throughput                                         (DOM 막대·숫자·결정 갱신)
   └─(--hil) obs ─▶ Pi edge_server ─▶ {action, q} ─▶ demo
```

- **임베디드 단일 명령**: `python demo.py --hil --dashboard --scenario asymmetric` 한 줄로 SUMO + 대시보드 서버 동시 기동 후 브라우저 자동 오픈. 발표자는 명령 하나만.
- **새 의존성 0**: 파이썬 stdlib `http.server`(ThreadingHTTPServer) + 백그라운드 스레드. Flask 등 불필요.
- **표시값은 traci 원시 읽기**: 막대 숫자는 53D obs 역정규화가 아니라 `traci.lanearea` getter로 **진짜 차량수**(예: 17대)를 직접 읽는다(`monitor_e2.py`와 동일 getter). `--dashboard`는 E2 검지기 로드(`-a network/e2.add.xml`)를 강제하여 obs 모드와 무관하게 동작. 관측 정규화와 화면 표시를 분리.

## 4. 구성요소 (격리된 단위)

| 단위 | 책임 | 인터페이스 | 의존 |
|---|---|---|---|
| `dashboard.html` | 추천 레이아웃 페이지. JS가 200ms마다 `/state` 폴링 → DOM 갱신 | 정적 파일 | 없음 |
| `dashboard.py` | 경량 HTTP 서버 + 최신 상태 보관 | `Dashboard.start(port)->url` / `.update(state:dict)` / `.stop()`. `GET /`→html, `GET /state`→json | stdlib |
| `demo.py`(수정) | `--dashboard` 플래그, 서버 기동, 매 스텝 state 조립·push, 차로 라벨/현시 매핑 | — | dashboard.py |
| `edge_server.py`(수정) | 추론 결과를 `{action, q:[유지,전환]}`로 확장 반환(하위호환) | TCP JSON line | numpy |

## 5. 데이터 계약

### 5.1 state JSON (demo → dashboard → 브라우저)

```json
{
  "step": 24, "sim_time": 120.0,
  "phase": 0, "phase_name": "남북직진", "time_in_phase": 23.0,
  "decision": "keep",            // "keep" | "switch" | "fixed"(고정/Webster)
  "q": [2.1, 0.4],                // [유지, 전환]; rl/hil 일 때만, 아니면 null
  "metrics": {"waiting_vehicles": 42, "avg_wait": 11.3, "throughput": 318},
  "lanes": [                       // 12개, 방향별 그룹(북/남/동/서 × 우회전/직진/좌회전)
    {"label": "북 우회전", "group": "우회전", "count": 2, "cap": 20, "is_green": true},
    {"label": "북 직진", "group": "직진", "count": 17, "cap": 20, "is_green": true},
    {"label": "북 좌회전", "group": "좌회전", "count": 5, "cap": 20, "is_green": false},
    ...
  ],
  "mode": "rl", "edge": "127.0.0.1:9999"   // 헤더 표기용
}
```

### 5.2 차로 매핑 (확정)

차로 ID 규약 `<방향>2C_<차로>`, 검지기 `e2_<laneID>`. 차로 인덱스 `_0`=우회전 `_1`=직진 `_2`=좌회전. 방향 `N`=북 `S`=남 `E`=동 `W`=서.

표시 12차로(방향별 그룹, 각 방향 우회전`_0`→직진`_1`→좌회전`_2` 순): 북 `N2C_0/1/2` · 남 `S2C_0/1/2` · 동 `E2C_0/1/2` · 서 `W2C_0/1/2`. `cap = lanearea 길이 / 7.5`(≈20). 우회전은 회색으로 구분.

### 5.3 현시 매핑 (확정)

P0 남북직진 · P1 남북좌회전 · P2 동서직진 · P3 동서좌회전. `is_green`은 현재 phase로 결정: P0→남·북 직진, P1→남·북 좌회전, P2→동·서 직진, P3→동·서 좌회전.

### 5.4 엣지 프로토콜 확장 (하위호환)

- `edge_server`: 두 층 ReLU 후 logits = `W4@h+b4` 계산 → `{"action": int(argmax), "q": logits.tolist()}` 반환. (기존 `infer`는 argmax만 → logits 반환 함수 분리 후 argmax 래핑.)
- `EdgeClient.act(obs)`: 응답에서 `q`가 있으면 `self.last_q`에 저장 후 action 반환(없으면 종전대로). demo는 `edge.last_q`를 대시보드 Q값으로 사용.
- 로컬 rl 모드: 기존 `policy_net` logits를 `q.tolist()`로 그대로 사용.

## 6. 지표 산출

- `waiting_vehicles`: 전체 진입차로 12개 `getLastStepHaltingNumber` 합(교차로 전체 정지 차량 수). 막대(12차로)별 `count`와 동일 출처.
- `avg_wait`: `info["system_mean_waiting_time"]`(없으면 total/대수)로 라이브 표시.
- `throughput`: 매 시뮬스텝 `traci.simulation.getArrivedNumber` 누적(evaluate.py와 동일 정의). demo 루프에 누적 변수 추가.

## 7. 테스트

- `tests/test_dashboard.py`(신규): `Dashboard` 기동 → 샘플 state `update` → `GET /state` JSON 일치, `GET /` 가 html 반환 확인 후 `stop`. (스모크, 직접 실행)
- `tests/test_inference_parity.py`(확장): numpy logits ≈ torch logits(`allclose`) + argmax 동일(기존 100/100 유지).
- 수동 리허설: `python demo.py --dashboard --mode rl --scenario asymmetric` → 브라우저 막대/결정/지표가 SUMO와 동기되어 실시간 갱신되는지 육안 확인.

## 8. 범위 밖 (YAGNI)

결정 로그 피드, 시계열 차트, 다중 클라이언트, 인증, SSE(폴링으로 충분). 발표 후 안 쓰는 기능은 만들지 않는다.

## 9. 파일 변경 요약

- 신규: `dashboard.py`, `dashboard.html`, `tests/test_dashboard.py`
- 수정: `demo.py`(--dashboard), `edge_server.py`(q 반환), `tests/test_inference_parity.py`(q 검증)
- `.gitignore`: `.superpowers/` 추가(브레인스토밍 산출물 비추적)
