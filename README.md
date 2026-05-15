# 강화학습 기반 스마트 교차로 신호 제어 시스템

> **RL-based Adaptive Traffic Signal Control System**

## 📌 프로젝트 개요

고정된 신호 주기는 시간대별 교통량 변화에 대응하지 못해 불필요한 대기 시간을 만든다.
본 프로젝트는 **DQN(Deep Q-Network)** 강화학습 에이전트가 SUMO 시뮬레이션 환경에서
신호 타이밍을 스스로 학습하여 평균 대기 시간을 최소화하는 시스템을 구현한다.

추가로 **YOLOv10** 으로 실제 교통 카메라 영상에서 차량을 탐지하고,
**라즈베리파이 + LED** 미니 신호등으로 학습된 정책을 실물로 시연한다.

## 🎯 핵심 결과 (1000 에피소드 학습 완료)

30 episode 평가 기준, 고정 신호 vs DQN 신호:

| 메트릭 | Fixed Signal | RL (DQN 1000ep) | 개선 |
|---------|----------|----------------|--------|
| **Total Reward** | -185.00 ± 22.93 | **-7.04 ± 0.87** | **+96.2%** |
| **Avg Waiting Time** | 3905.79 ± 18.19 | **44.85 ± 0.75** | **+98.9%** |
| **Avg Queue Length** | 70.87 ± 0.38 | **15.01 ± 0.19** | **+78.8%** |
| **Max Queue Length** | 97.70 ± 0.74 | **24.53 ± 1.20** | **+74.9%** |

→ `results/evaluation_chart_1000ep.png`, `results/reward_convergence_1000.png`

## 🏗️ 시스템 아키텍처

```
┌──────────── PC (학습/시뮬레이션) ──────────────┐
│                                              │
│  SUMO 시뮬레이션 ── DQN 학습 ── 학습된 정책       │
│                                  │            │
│        ┌─────────────────────────┼──┐         │
│        ▼                         ▼  │         │
│   SUMO GUI (시각화)        현재 신호 phase     │
│                                  │            │
│  YOLO + 교통카메라 (별도 트랙: 차량 탐지)      │
│                                  │            │
└──────────────────────────────────┼────────────┘
                                   │ WiFi/Serial
                                   ▼
                  ┌── 라즈베리파이 + 미니 신호등 ──┐
                  │  GPIO → 4방향 LED 점등        │
                  └──────────────────────────────┘
```

## 📂 폴더 구조

```
sumo_project/
├── README.md
├── requirements.txt
├── .gitignore
│
├── network_v1/              # 1차 환경 (4-way × 1차선, 2-phase)
│   ├── intersection.nod.xml
│   ├── intersection.edg.xml
│   ├── intersection.net.xml
│   ├── intersection.rou.xml
│   └── intersection.sumocfg
│
├── network_v2/              # 2차 환경 (4-way × 3차선, 4-phase + 보호좌회전)
│   ├── intersection.nod.xml
│   ├── intersection.edg.xml
│   ├── intersection.con.xml
│   ├── intersection.net.xml
│   ├── intersection.rou.xml
│   └── intersection.sumocfg
│
├── network_geumjanggyo/     # 실제 교차로 OSM 모델 (참고용)
│
├── dqn_agent.py             # v1 환경 학습
├── dqn_agent_v2.py          # v2 환경 학습 (state/action 자동 감지)
├── evaluate.py              # Fixed vs RL 비교 (30 ep + 메트릭)
├── demo.py                  # SUMO GUI 데모
├── plot_results.py          # 학습 수렴 그래프
├── realtime_detect.py       # YOLO 차량 탐지
├── test_env.py              # SUMO 환경 동작 확인
│
├── results/                 # v1 학습 결과
│   ├── checkpoint_ep1000.pth
│   ├── dqn_final.pth
│   ├── training_log.csv
│   ├── evaluation_30ep.csv
│   ├── reward_convergence_1000.png
│   └── evaluation_chart_1000ep.png
│
└── archive/                 # 오래된 파일 보관
    ├── compare.py           # evaluate.py 이전 5ep 평가
    ├── reward_convergence_200ep.png
    └── old_checkpoints/
```

## ⚙️ 설치

### 1. SUMO 설치
[Eclipse SUMO 공식 사이트](https://eclipse.dev/sumo/)에서 1.20+ 버전 설치.
설치 후 `SUMO_HOME` 환경변수 자동 설정 (Windows 기본 경로:
`C:/Program Files (x86)/Eclipse/Sumo`).

### 2. Python 의존성
```powershell
pip install -r requirements.txt
```

## 🚀 실행 방법

### 환경 동작 확인
```powershell
python test_env.py
```

### v1 환경 학습 (4-way × 1차선)
```powershell
# 처음부터
python dqn_agent.py --episodes 1000

# 체크포인트에서 이어서
python dqn_agent.py --resume results/checkpoint_ep1000.pth --episodes 1500
```

### v2 환경 학습 (4-way × 3차선, 4-phase)
```powershell
python dqn_agent_v2.py --episodes 1000
```

### 평가 (Fixed vs RL 30 episode 비교)
```powershell
python evaluate.py
```

### SUMO GUI 데모
```powershell
# 학습된 RL 정책으로 시연
python demo.py --mode rl --duration 1800

# 고정 신호로 시연
python demo.py --mode fixed --duration 1800
```

### YOLO 차량 탐지 (영상 입력)
```powershell
python realtime_detect.py
```

## 🛠️ 기술 스택

| 영역 | 기술 |
|------|----|
| **시뮬레이션** | SUMO 1.20+, sumo-rl 1.4.5 |
| **강화학습** | PyTorch (Double DQN, Huber Loss, Gradient Clipping) |
| **차량 탐지** | YOLOv10 (ultralytics), OpenCV |
| **하드웨어** | Raspberry Pi 5 (예정), GPIO LED |
| **시각화** | matplotlib |

## 📈 DQN 하이퍼파라미터

| 항목 | 값 |
|------|-----|
| Hidden Layer | 128 × 128 (ReLU) |
| Optimizer | Adam (lr = 1e-3) |
| Loss | Huber (SmoothL1) |
| Gamma | 0.99 |
| Epsilon decay | 0.995 (1.0 → 0.05) |
| Batch size | 64 |
| Replay Buffer | 50,000 |
| Target Network update | 매 10 ep |
| Gradient clip | 10.0 |

## 🗺️ 진행 상황

- [x] 1~3차 발표: 주제·구성도·기술 스택 확정
- [x] 4차: YOLOv10 차량 탐지 PoC + 논문 조사
- [x] 5차: SUMO 환경 + DQN 파이프라인 구축
- [x] 6차: Double DQN 전환 + 200ep 학습
- [x] 7차: 405ep 학습 + Fixed vs RL 시연 영상
- [x] 8차: 실제 영상(금장교네거리) YOLO 탐지 시도 + 한계 분석
- [x] **1000ep 학습 완료 + 30ep 평가 + 4-panel 메트릭 비교**
- [ ] v2 환경(3차선 × 4-phase) 학습
- [ ] YOLO 영상 탐지 고도화 (주간 + 신호등 시점)
- [ ] 라즈베리파이 5 도착 → 미니 신호등 통신 + GPIO 제어
- [ ] 최종 시연 + 보고서 작성

## 📚 참고 문헌

- Wang, S. et al. *Traffic Signal Control via RL: A Review on Applications and Innovations*. MDPI 2025.
- *MD3DQN: End-to-End RL for Traffic Signal Control via Surveillance Video*. ICLR 2025.
- LucasAlegre, [sumo-rl](https://github.com/LucasAlegre/sumo-rl) — SUMO + Gymnasium 연동 라이브러리.

## 📝 라이선스

학술 목적 졸업작품. 코드 재사용 시 출처 표기 권장.
