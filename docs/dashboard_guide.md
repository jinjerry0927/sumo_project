# 실시간 HIL 대시보드 — 노트북 점검 가이드

> SUMO GUI 옆에 띄워 "E2 검지기가 본 것 → SmartSignal 두뇌의 결정 → 신호 변화"를 실시간으로 보여주는 웹 대시보드. 발표 라이브 데모용. 설계=`docs/superpowers/specs/2026-06-07-hil-dashboard-design.md`.

## 0. 준비

- 이 저장소를 노트북에 `git pull` 로 최신화.
- SUMO 설치 + `SUMO_HOME` 환경변수(보통 `C:/Program Files (x86)/Eclipse/Sumo`).
- 파이썬 패키지: `sumo_rl`, `torch`(로컬 모드만), `numpy`. **대시보드 자체는 추가 의존성 없음**(파이썬 stdlib).
- 모델 파일 존재 확인: `results/smart_signal_e2.npz`(HIL용), `results/smart_signal.pth`(로컬 god-view용).

## 1. 가장 간단한 점검 — 로컬 모드 (Pi 불필요)

```bash
python demo.py --dashboard --mode rl --scenario asymmetric --duration 600
```

- SUMO GUI 창 + 기본 브라우저(`http://127.0.0.1:8000`)가 자동으로 열린다.
- SUMO 창에서 **▶ 재생**(또는 상단 Delay 슬라이더 50~100ms)을 누르면, 브라우저 대시보드의 막대·숫자·KEEP/SWITCH가 차량 흐름에 맞춰 실시간 갱신된다.
- 이 모드는 god-view(29D) 모델이 신호를 제어하고, 대시보드는 E2 검지기 원시 차량수를 표시한다(두 경로 분리).

**확인 포인트**
- 상단: 현재 대기차량 / 평균 대기(s) / 누적 통과 / 현재 현시(P0~P3)
- 좌측: 12차로 막대(회색=우회전, 초록=직진, 주황=좌회전) + 정지 차량수, 현재 녹색 차로는 라벨이 초록
- 우측: KEEP/SWITCH + Q값 막대 2개(유지/전환)

## 2. 발표 본 구성 — HIL 모드 (엣지 추론 + Q값)

터미널 2개를 쓴다.

```bash
# 터미널 A — 엣지(추론) 서버. 노트북에서 직접 띄워도 되고, 라즈베리파이에서 띄워도 됨.
python edge_server.py --weights results/smart_signal_e2.npz

# 터미널 B — SUMO + 대시보드
python demo.py --hil --dashboard --scenario asymmetric --duration 600
```

- 터미널 B에 `[hil] 엣지서버 연결: 127.0.0.1:9999` 와 `[dashboard] http://127.0.0.1:8000` 가 뜬다.
- 두뇌 패널의 Q값 막대가 **엣지서버가 계산한 값**으로 채워지고, 하단 범례가 `엣지: Pi 127.0.0.1:9999 …` 로 표기된다.
- 실물 라즈베리파이를 쓸 땐 B를 `--host <PI_IP>` 로 바꾼다(가이드: `docs/raspberry_pi_hil_guide.md`).

## 3. 옵션 / 자주 쓰는 인자

| 인자 | 뜻 | 기본 |
|---|---|---|
| `--scenario` | low/medium/high/asymmetric/saturated | (필수 권장) |
| `--duration` | 시뮬 길이(초) | 3600 |
| `--dash_port` | 대시보드 포트 | 8000 |
| `--mode` | fixed/webster/rl (대시보드는 rl/hil에서 Q값 표시) | rl |
| `--seed` | 트래픽 seed | 1000 |

> 다른 시나리오로 대조하려면 `--scenario saturated` 처럼 바꿔 실행. 같은 seed면 같은 트래픽.

## 4. 문제 해결

- **브라우저가 안 열림** → 터미널의 `[dashboard] http://127.0.0.1:8000` 주소를 수동으로 연다.
- **포트 충돌(8000 사용 중)** → `--dash_port 8050` 등으로 변경.
- **막대가 안 움직임** → SUMO 창에서 ▶ 재생을 눌렀는지 확인(처음엔 일시정지 상태일 수 있음).
- **HIL에서 연결 실패** → 터미널 A의 엣지서버가 먼저 떠 있는지, `--host`/`--port` 가 일치하는지 확인.
- **`Missing yellow phase` 경고** → 정상(우회전 상시통행 신호 구조 때문, 무해).

## 5. 자동 검증(코드 점검용, GUI 불필요)

```bash
python tests/test_dashboard.py          # [PASS] dashboard /state 왕복 OK
python tests/test_inference_parity.py   # [PASS] numpy == torch - action + logits (100/100)
```
