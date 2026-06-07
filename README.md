# 강화학습 기반 스마트 교차로 신호 제어 시스템

> **RL-based Adaptive Traffic Signal Control System**

## 📌 프로젝트 개요

고정된 신호 주기는 시간대별 교통량 변화에 대응하지 못해 불필요한 대기 시간을 만든다.
본 프로젝트는 **DQN(Deep Q-Network)** 강화학습 에이전트가 SUMO 시뮬레이션 환경에서
**다양한 교통상황(한산·평시·피크·비대칭·과포화)** 에 대해 신호 타이밍을 스스로 학습하고,
**현실적 고정신호 기준선(Fixed-Avg / Webster 최적 고정주기)** 과 정량 비교하여 개선 효과를 입증한다.

- 1학기: **추상 4지 교차로 (3차선 × 4-phase 보호좌회전) 중심**, SUMO 시뮬레이션으로 완결
- 2학기: 실제 교차로 1곳을 SUMO로 모델링하여 확장

> **방향 재설정(2026-06-02)**: 실시간 CCTV 영상 인지(YOLO) 줄기는 4방향 카메라 부재 등
> 데이터 한계로 보류하고, SUMO 시뮬레이션 비교 검증으로 목표를 좁혔다.
> 관련 코드는 삭제하지 않고 `archive/perception_track/`에 보존한다.
> 전체 계획·부서 구조는 **[docs/project_charter.md](docs/project_charter.md)** 참조.

## 🎯 핵심 결과 (1000 에피소드 학습 완료)

30 episode 평가 기준, 고정 신호 vs DQN 신호:

| 메트릭 | Fixed Signal | RL (DQN 1000ep) | 개선 |
|---------|----------|----------------|--------|
| **Total Reward** | -185.00 ± 22.93 | **-7.04 ± 0.87** | **+96.2%** |
| **Avg Waiting Time** | 3905.79 ± 18.19 | **44.85 ± 0.75** | **+98.9%** |
| **Avg Queue Length** | 70.87 ± 0.38 | **15.01 ± 0.19** | **+78.8%** |
| **Max Queue Length** | 97.70 ± 0.74 | **24.53 ± 1.20** | **+74.9%** |

→ `results/evaluation_chart_1000ep.png`, `results/reward_convergence_1000.png`

## 🏗️ 시스템 아키텍처 (SUMO 전용)

```
┌──────────── 시뮬레이션 환경팀 (SimEnv) ───────────┐
│  SUMO 네트워크 + 교통 시나리오 5종 (랜덤/동결)      │
└───────────────┬───────────────────┬──────────────┘
                ▼                   ▼
   ┌─ 강화학습팀 (RL) ─┐   ┌─ 베이스라인·평가팀 (Eval) ─┐
   │  DQN 학습 → 정책   │   │  Fixed-Avg / Webster        │
   │  (.pth 체크포인트) │──▶│  시나리오별 통계 비교        │
   └──────────────────┘   └──────────────┬─────────────┘
                                         ▼
                          ┌─ 시각화·보고팀 (Viz) ─┐
                          │  비교 차트 / GUI 데모   │
                          └───────────────────────┘
```

## 📂 폴더 구조

```
sumo_project/
├── README.md
├── requirements.txt
├── .gitignore
│
├── network/                 # SmartSignal 환경 (4-way × 3차선, 4-phase + 보호좌회전)
│   ├── intersection.nod.xml / .edg.xml / .con.xml
│   ├── intersection.net.xml
│   ├── intersection.rou.xml   # 학습 시 매 ep 랜덤 생성됨
│   └── intersection.sumocfg
│
├── smart_signal.py          # ★ SmartSignal DQN 학습 (state/action 자동 감지)
├── evaluate.py              # Fixed vs SmartSignal 비교 (30 ep + 메트릭)
├── demo.py                  # SUMO GUI 데모 (rl / fixed)
├── plot_results.py          # 학습 수렴 그래프
├── test_env.py              # SUMO 환경 스모크 테스트
│
├── docs/
│   └── project_charter.md   # 재정립 계획 + 부서 구조 (필독)
│
├── results/                 # SmartSignal 학습 산출물 (현재 빈 상태 → 재학습)
│
└── archive/                 # 보관 (삭제 안 함)
    ├── perception_track/    # CCTV·YOLO·ITS 줄기 (보류)
    ├── v1/                  # 구 v1 환경·코드·결과 일체
    └── training_history/    # 구 v2 학습 로그 (발산 분석용)
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

### SmartSignal 학습 (처음부터)
```powershell
python smart_signal.py --episodes 1000

# 체크포인트에서 이어서
python smart_signal.py --resume results/checkpoint_ep100.pth --episodes 1000
```

### 평가 (Fixed vs SmartSignal 30 episode 비교)
```powershell
python evaluate.py
```

### SUMO GUI 데모
```powershell
# 학습된 SmartSignal 정책으로 시연
python demo.py --mode rl --duration 1800

# 고정 신호로 시연
python demo.py --mode fixed --duration 1800
```

### 라즈베리파이 HIL + 실시간 대시보드
E2 검지기 관측(53D) 모델을 엣지(라즈베리파이/로컬)에서 numpy로 추론하고, SUMO GUI 옆에 실시간 대시보드(12차로 대기차량 + 두뇌 KEEP/SWITCH + Q값)를 함께 띄운다.
```powershell
# 로컬 점검(Pi 불필요): SUMO + 대시보드만 확인
python demo.py --dashboard --mode rl --scenario asymmetric --duration 600

# HIL(엣지 추론) — 터미널 A에서 엣지서버, 터미널 B에서 데모
python edge_server.py --weights results/smart_signal_e2.npz          # 터미널 A
python demo.py --hil --dashboard --host 127.0.0.1 --scenario asymmetric   # 터미널 B
```
- 대시보드는 브라우저 `http://127.0.0.1:8000` 에 자동으로 열린다(추가 의존성 없음, 파이썬 stdlib).
- 실물 라즈베리파이 연동 절차 → `docs/raspberry_pi_hil_guide.md`, 대시보드 점검 가이드 → `docs/dashboard_guide.md`.

## 🛠️ 기술 스택

| 영역 | 기술 |
|------|----|
| **시뮬레이션** | SUMO 1.20+, sumo-rl 1.4.5 |
| **강화학습** | PyTorch (Double DQN, Huber Loss, Gradient Clipping) |
| **기준선** | Fixed-time, Webster 최적 고정주기, (선택) SUMO actuated TLS |
| **시각화** | matplotlib |
| ~~차량 탐지~~ | ~~YOLOv8/v10, OpenCV~~ → `archive/perception_track/` 보존 |

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
- [x] (구 v1) 1000ep 학습 + 30ep 평가 — `archive/v1/`로 보관
- [x] **방향 재설정 + CCTV 줄기 archive (2026-06-02)** → [docs/project_charter.md](docs/project_charter.md)
- [x] **M0**: v1 정리 + SmartSignal 개명 + 도구 재타겟 (스모크테스트 통과)
- [ ] **M1**: 시나리오 5종 동결 + 평가 시나리오 루프
- [ ] **M2**: SmartSignal 재학습 완주 + baseline (Fixed / Webster / actuated)
- [ ] **M3**: 시나리오별 RL vs baseline 비교 + 차트 + GUI 데모
- [ ] **M4**: 발표자료 + README·보고서 갱신
- [ ] (2학기) 실제 교차로 1곳 SUMO 모델링 확장

## 📚 참고 문헌

- Wang, S. et al. *Traffic Signal Control via RL: A Review on Applications and Innovations*. MDPI 2025.
- *MD3DQN: End-to-End RL for Traffic Signal Control via Surveillance Video*. ICLR 2025.
- LucasAlegre, [sumo-rl](https://github.com/LucasAlegre/sumo-rl) — SUMO + Gymnasium 연동 라이브러리.

## 📝 라이선스

학술 목적 졸업작품. 코드 재사용 시 출처 표기 권장.
