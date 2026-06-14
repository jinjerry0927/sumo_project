# 진행 요약 (2026-05-27)

## 목표

졸업작품 "스마트 교차로 신호 제어"의 *눈* 역할 — **한국 ITS 실시간 CCTV에서 차량을 인식하고 4/8방향 차로별 카운트를 산출하는 모듈** — 을 만들기.

## 무엇이 만들어졌나

### `perception/` 패키지 (신규)
한 프로세스에서 CCTV 수신 → YOLO 인식 → 차로 카운트가 메모리상에서 한 번에 흐름.

```
ItsApiStreamSource → VehicleDetector(YOLOv8s) → LaneAggregator(ROI) → 화면+JSONL
```

| 파일 | 역할 |
|---|---|
| `stream_source.py` | 파일/HLS/RTSP 공용 + `ItsApiStreamSource`(토큰 25s 자동갱신) + `YoutubeLiveSource` |
| `detector.py` | YOLOv8 차량 탐지 + 박스 필터 (`imgsz`, `min_w/h` 조정 가능) |
| `lane_aggregator.py` | ROI 폴리곤 기반 방향별 카운트 (4방향 또는 8방향 진입/진출) |
| `run_realtime.py` | CLI 진입점. `q`=종료, `s`=스크린샷 |
| `roi_config/mohwa_template.json` | 모화사거리 8방향 ROI 템플릿 |

### `scripts/` (보조 도구)
| 파일 | 역할 |
|---|---|
| `list_its_cameras.py` | bbox로 ITS 카메라 목록 표 출력 |
| `capture_frame.py` | 일반 URL/파일 첫 프레임 PNG 저장 |
| `demo_mohwa.ps1` | 모화사거리 데모 한 줄 실행 wrapper (튜닝 옵션 박힘) |
| `check_streams.py` | (구) 경주시 ITS 직접 m3u8 도달성 테스트 |

### `docs/`
- `integration_plan.md` — 시스템 통합 설계 + 발표 5장 outline
- `progress.md` — 이 파일

## 시도했지만 안 된 것

| 시도 | 결과 |
|---|---|
| 경주시 ITS 직접 m3u8 (`221.157.65.155:1935`) | 무응답. URL 무효화 추정 |
| ITS 국가센터 OpenAPI에서 경주 시내 도심 카메라 | 없음. 고속도로/국도 위주 |

## 결국 잘 된 것

- **국가교통정보센터(data.go.kr) ITS CCTV 인증키 + ItsApiStreamSource** → 정상 동작
- **경주 모화사거리(`[국도7] 경주 모화사거리`, 사거리)** 카메라 발견 → 데모 카메라로 확정
- 720x480 라이브 영상에서 차량 인식 OK (`imgsz=1280`, `min-w=12`, `frame-skip=3`)

## 사용법

```powershell
cd C:\Users\James\Documents\GitHub\smart_signal_traffic
$env:ITS_API_KEY = "발급받은_키"
.\scripts\demo_mohwa.ps1
# 영상에서 's'키로 스크린샷 → ROI 좌표 보정 → 다시 실행
```

## 한계 (발표 시 솔직히 인정)

1. 카운트 결과가 **DQN으로 연결되지 않음** (Critical #1 잔여)
2. 진입/진출 구분이 ROI 폴리곤 위치 기반 — 트래킹 방향 미사용
3. 야간/우천은 정확도 급락 → 데모는 주간으로 회피

## 다음 사이클 (졸업 발표 마무리용)

1. `perception/state_encoder.py` — 카운트 dict → SUMO state 11차원
2. `scripts/realtime_inference.py` — perception 결과로 `dqn_final.pth` 호출 → action 출력
3. (선택) RasPi GPIO sink, baseline 강화(Webster/actuated TLS)
