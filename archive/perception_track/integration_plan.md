# perception 모듈 → 졸업작품 전체 통합 설계 + 발표 outline

## 1. 큰 그림 — 졸업작품 핵심 컨셉

> **SUMO에서 학습한 DQN 신호 제어 정책을, 실시간 한국 교통 CCTV 입력으로 직접 검증하는 시스템.**

8차 발표까지 식별된 핵심 갭은 **"머리(DQN)와 눈(YOLO) 단절"**이었음.
이번 사이클에서 *눈* 쪽을 완성했고, 다음 사이클에서 *눈→머리* bridge를 채우면 졸업작품 컨셉이 닫힘.

## 2. 시스템 아키텍처

```
                  ┌──────────── 학습 단계 (오프라인) ────────────┐
                  │                                              │
                  │   SUMO 시뮬레이션 ──> DQN 학습 ──> dqn_final.pth
                  │   (network_v1/v2)                            │
                  └──────────────────────────────────────────────┘
                                       │
                                       │ 학습된 정책
                                       ▼
┌─────────── 실시간 추론 (발표 데모) ──────────────────────────────┐
│                                                                   │
│  국가교통정보센터 CCTV  ─┐                                         │
│  (모화사거리 등)         │                                         │
│                          ▼                                         │
│             ItsApiStreamSource (토큰 25s 자동갱신)                 │
│                          │  frame                                  │
│                          ▼                                         │
│             VehicleDetector (YOLOv8s, imgsz=1280)                  │
│                          │  boxes                                  │
│                          ▼                                         │
│             LaneAggregator (8방향 ROI: N_in/N_out/...)            │
│                          │  {N_in:3, N_out:1, ...}                 │
│                          ▼                                         │
│        ┌─────── StateEncoder (다음 사이클) ────────┐               │
│        │  카운트 dict → SUMO state 11차원 벡터     │               │
│        └─────────────────┬──────────────────────────┘             │
│                          │  state                                  │
│                          ▼                                         │
│             DQN policy_net (dqn_final.pth)                         │
│                          │  action (0=유지, 1=phase 전환)          │
│                          ▼                                         │
│        ┌──── SignalSink (다음 사이클) ────┐                        │
│        │   화면 표시 / Pi GPIO LED / CSV  │                        │
│        └──────────────────────────────────┘                        │
└────────────────────────────────────────────────────────────────────┘
```

**현재 완성**: 입력~카운트 (`perception/`)
**다음 사이클**: `StateEncoder` + `SignalSink`

## 3. perception 모듈 — 무엇이 어떻게 만들어졌나

| 구성요소 | 역할 | 핵심 결정 |
|---|---|---|
| `StreamSource` | 입력 추상화 | 파일/HLS/RTSP 단일 인터페이스. 자동 재연결. |
| `ItsApiStreamSource` | 한국 ITS CCTV | 토큰 30초 만료 → 25초마다 OpenAPI 재호출하여 URL 갱신 |
| `YoutubeLiveSource` | 안전망 | yt-dlp로 라이브 m3u8 추출 (ITS 미커버 지역용) |
| `VehicleDetector` | 차량 탐지 | YOLOv8s 기본, `imgsz`/박스 필터 CLI 노출 |
| `LaneAggregator` | 차로별 분류 | 임의 폴리곤 dict (4방향 또는 8방향 진입/진출) |

**튜닝 학습**:
- 야간 CCTV 실패(8차) → 데모는 주간 + `imgsz=1280`로 작은 객체 강화
- 720x480 저해상도 → `min_w/min_h` 완화
- HLS 디코딩 지연 → `frame-skip`으로 detection만 듬성듬성, 화면은 부드럽게

## 4. 모화사거리 데모 시나리오 (발표 1막)

1. PowerShell에서 한 줄 실행:
   ```powershell
   $env:ITS_API_KEY = "..."
   .\scripts\demo_mohwa.ps1
   ```
2. 창이 뜨고 모화사거리 라이브 영상 + 차량 박스 + 8방향 카운트 표시
3. 시청자에게 보여주는 것:
   - "녹화 영상이 아닙니다. 25초마다 토큰을 갱신해 실시간 ITS 라이브를 받습니다"
   - 진입 차량(`N_in=3`)이 점차 누적되는 모습 → "이게 신호 제어의 입력 신호"

**발표 직전 안전망**:
- ITS 접속 장애 대비 → 동일 옵션으로 `intersection.mp4` (`scripts/demo_mohwa_offline.ps1`로 추가 예정)
- 8방향 ROI 좌표 보정 안 됐으면 → ROI 끄고(`-NoRoi`) 총 카운트만

## 5. 한계와 향후 (발표 2막 — 솔직히 인정)

| 항목 | 현재 한계 | 어떻게 해결할지 |
|---|---|---|
| 진입/진출 구분 | ROI 폴리곤 위치만으로 분류 (트래킹 방향 미사용) | YOLO `track_id` 시간 변화로 방향 벡터 계산 → ROI 미스 보정 |
| state encoder 부재 | `LaneAggregator` 출력이 DQN으로 안 흘러감 (Critical #1 잔여) | `perception/state_encoder.py` — count dict → 11차원 매핑 |
| 신호 출력 부재 | DQN action이 화면/하드웨어로 안 나감 | `SignalSink` — 화면 LED + (시간 되면) RasPi GPIO |
| 실제 신호 ≠ 시뮬 신호 | 모화사거리 실제 신호 주기와 SUMO 환경 신호 주기 무관 | "실제 신호와 권고 신호를 나란히 표시"로 발표 시 분리 |
| 야간/우천 | YOLO 정확도 급락 | 데이터 augmentation은 범위 외, 데모 주간만으로 회피 |

## 6. 발표 슬라이드 outline (5장 권장)

| # | 제목 | 핵심 한 문장 | 시각 자료 |
|---|---|---|---|
| 1 | 문제 정의 | "도시 교차로 신호를 강화학습으로 최적화하되, 시뮬에만 갇히지 않고 실제 카메라로 검증한다" | 도시 교통 정체 사진 + SUMO 화면 |
| 2 | 시스템 구성 | "SUMO에서 학습 → 한국 ITS CCTV로 추론" | 위 §2 아키텍처 다이어그램 |
| 3 | perception 모듈 | "ITS OpenAPI를 25초마다 갱신하며 HLS 라이브를 받아 YOLOv8 + ROI 기반 차로별 카운트" | 모화사거리 실시간 데모 (영상) |
| 4 | 학습 결과 | "1000ep 학습 후 평균 대기 시간 N% 개선" (실제 수치는 `evaluate.py` 출력으로) | reward 수렴 곡선 + fixed vs RL 비교표 |
| 5 | 한계와 향후 | "state encoder/신호 출력 bridge가 잔여 작업. 라즈베리파이 LED 통합 예정" | §5 표 + 향후 일정 |

**발표 팁**:
- 3번 슬라이드에서 라이브 데모. 네트워크 실패 대비해 백업 영상 미리 화면 녹화해두기.
- 4번에서 96.3% 같은 과장된 숫자는 피하고, baseline(고정 신호 30초)이 어떻게 정의됐는지 한 줄로 설명. 평가자가 "기준이 뭐냐"고 묻기 전에 먼저 노출.
- 5번을 자신감 있게: 미완성을 인정하되 "다음 액션이 명확함"을 보이면 평가자 신뢰 ↑.

## 7. 다음 사이클 (졸업작품 마무리용)

우선순위:
1. **`perception/state_encoder.py`** — 8방향 카운트 → SUMO state 11차원. `dqn_agent.py:STATE_SIZE=11` 호환.
2. **`scripts/realtime_inference.py`** — perception 결과로 `dqn_final.pth` 호출 → action 출력.
3. (선택) RasPi GPIO sink — 전형주 담당 영역.
4. baseline 강화(Webster 또는 SUMO 내장 actuated TLS) — 평가 공정성 보강.

이 4개가 끝나면 8차 발표의 Critical #1~#3이 모두 닫히고, 졸업 발표에서 "시뮬 학습 → 실세계 검증"의 한 사이클을 보여줄 수 있음.
