# perception/ — 실시간 CCTV 차량 인식 모듈

졸업작품의 "눈" 역할. 실시간 영상에서 차량을 탐지하고 **4방향(N/S/E/W) 차로별 카운트**를 산출한다.
이 카운트는 다음 단계에서 DQN 정책(state 11차원)의 차선 queue 부분과 매핑된다.

## 구조

```
[입력 소스] ──> StreamSource ──> VehicleDetector(YOLOv8s) ──> LaneAggregator(ROI) ──> {화면, JSONL}
```

| 파일 | 역할 |
|---|---|
| `stream_source.py` | 파일/HLS/RTSP 공용 입력, 자동 재연결. `ItsApiStreamSource`(토큰 25초마다 갱신), `YoutubeLiveSource`(yt-dlp 기반 라이브) 서브클래스 |
| `detector.py` | YOLOv8s 차량 탐지 + 박스 크기/종횡비 필터 |
| `lane_aggregator.py` | 박스 중심점이 4방향 ROI 폴리곤 중 어디에 속하는지 분류 |
| `run_realtime.py` | CLI 진입점 |
| `roi_config/*.json` | 카메라별 ROI 폴리곤 좌표 |

## CCTV 소스 결정 트리

- **1순위 — ITS 국가센터 OpenAPI (data.go.kr)**: 공식·무료·논문 인용 가능. 인증키 즉시 발급. 도시부(`type=its`)에서 신호 교차로 카메라 선택.
- **2순위 — YouTube 라이브**: ITS에 적당한 카메라 못 찾을 때. 한국 교통 라이브 채널 검색해 라이브 URL을 `--source youtube --url ...`로.
- **3순위 — 기존 mp4 파일**: 데모 안전망. `--source file --url intersection.mp4`.

> **참고**: 경주시 도심 교차로(금장교 등)는 ITS 국가센터 OpenAPI에서 응답하지 않음(고속도로/국도 위주). 또한 경주시 ITS 사이트의 직접 m3u8 엔드포인트(`221.157.65.155:1935`)는 무응답 확인됨. 데모용 카메라는 서울/부산 같은 ITS OpenAPI 잘 잡히는 도시로 선택.

## 빠른 시작

### 1) 기존 mp4로 동작 확인 (개발/회귀 테스트)

```powershell
python -m perception.run_realtime --source file --url intersection.mp4 --display
```

### 2) ROI를 적용한 4방향 카운트

먼저 한 프레임 떠두기:

```powershell
python scripts/capture_frame.py --url intersection.mp4 --out frame.png
```

`frame.png`를 그림판/GIMP 등으로 열어 4방향 영역의 좌표 4점(또는 그 이상)을 확인하고
`perception/roi_config/intersection_mp4.json`을 작성 (포맷은 `roi_config/example.json` 참고).

```powershell
python -m perception.run_realtime --source file --url intersection.mp4 `
    --roi perception/roi_config/intersection_mp4.json --display --save-jsonl logs/run.jsonl
```

### 3) ITS 국가센터 OpenAPI (실제 한국 CCTV) — 권장 1순위

1. [공공데이터포털](https://www.data.go.kr) 가입 → "국가교통정보센터 CCTV 정보" 검색 → 인증키 신청 (즉시 발급)
2. 환경변수에 키 저장:
   ```powershell
   $env:ITS_API_KEY = "발급받은_키"
   ```
3. **bbox 안의 카메라 목록 먼저 확인** (적당한 교차로 카메라 찾기):
   ```powershell
   # 서울 강남 도심 (예시)
   python scripts/list_its_cameras.py --bbox 127.02 127.08 37.49 37.53 --only-intersection

   # 부산 서면 도심 (예시)
   python scripts/list_its_cameras.py --bbox 129.04 129.07 35.15 35.18 --only-intersection
   ```
   출력에서 마음에 드는 카메라 이름을 골라서:
4. 그 카메라로 실시간 인식:
   ```powershell
   python -m perception.run_realtime --source its-api `
       --bbox 127.02 127.08 37.49 37.53 --cctv-name "강남대로" `
       --road-type its --display
   ```

> **참고 bbox**: 서울 도심 `126.95 127.05 37.55 37.60`, 강남 `127.02 127.08 37.49 37.53`, 부산 서면 `129.04 129.07 35.15 35.18`, 대구 동성로 `128.59 128.61 35.86 35.88`. 좁게 잡을수록 응답 카메라 수 적어짐.

### 4) YouTube 라이브 (안전망) — 2순위 fallback

`pip install yt-dlp` 후:

```powershell
python -m perception.run_realtime --source youtube `
    --url "https://www.youtube.com/watch?v=<LIVE_ID>" --display
```

"한국 교통 라이브"로 검색해 안정적인 라이브 채널 찾기. 30분마다 m3u8 토큰 자동 갱신.

### 5) 임의의 직접 URL (m3u8/rtsp 등)

```powershell
python -m perception.run_realtime --source url --url "http://.../playlist.m3u8" --display
```

## ROI JSON 포맷

```json
{
  "image_size": [1920, 1080],
  "polygons": {
    "N": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
    "S": [...],
    "E": [...],
    "W": [...]
  }
}
```

- `image_size`: 폴리곤 좌표가 기준으로 한 해상도. 입력 프레임이 다른 해상도면 자동 스케일.
- 폴리곤은 3점 이상이면 동작 (4각형이 일반적).
- 폴리곤이 겹치면 N → S → E → W 순으로 첫 매치 우선.
- **카메라 시점에 맞춰 사용할 방향만 정의해도 됨**. 예: 카메라가 한 방향 도로만 보면 `polygons`에 `N`만 넣어도 동작 (나머지 방향 카운트는 0).

## JSONL 로그 포맷

```jsonl
{"ts": 1734940800.12, "frame": 0,  "total": 5, "N": 2, "S": 1, "E": 1, "W": 1}
{"ts": 1734940800.13, "frame": 1,  "total": 6, "N": 2, "S": 1, "E": 2, "W": 1}
```

pandas로 후처리:
```python
import pandas as pd
df = pd.read_json("logs/run.jsonl", lines=True)
df.set_index("ts")[["N","S","E","W"]].rolling(30).mean().plot()
```

## 트러블슈팅

| 증상 | 원인/해결 |
|---|---|
| `Stream timeout triggered after 30000ms` | OpenCV-FFmpeg 기본 타임아웃. 우리 코드는 10s로 단축돼 있으나 첫 read는 느릴 수 있음. 한 번 더 시도 |
| 한글 콘솔 깨짐 (Windows) | 모듈이 자동으로 stdout을 UTF-8로 reconfigure. 안 되면 `chcp 65001` 후 재실행 |
| `cctv_name` 매치 0건 | bbox를 넓혀보거나 `--cctv-name` 없이 첫 응답 사용 |
| 야간 정확도 저하 (8차 발표 실패 이력) | 모델을 `yolov8m.pt`로 상향 (`--model yolov8m.pt`). 또는 데모는 주간으로 |
| 토큰 만료로 영상 끊김 | ItsApiStreamSource가 25초마다 자동 갱신. 그래도 끊기면 `refresh_s` 더 짧게 |

## 다음 단계 (이번 작업 범위 외)

- `state_encoder.py`: `LaneAggregator` 출력 → SUMO state 11차원 벡터
- `realtime_detect.py`(v1)를 정식 deprecate하고 `archive/`로 이동
- DQN 정책 호출과 신호 제어까지 end-to-end 연결
