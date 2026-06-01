"""
교통 시나리오 정의·생성 (SmartSignal 환경 network/ 공용).

- 학습(smart_signal.py): 매 에피소드 random_demand() 로 랜덤 수요 → 도메인 랜덤화
- 평가(evaluate.py):     EVAL_SCENARIOS 5종을 고정 파일로 "동결" → 공정 비교

수요(demand)는 각 진입 방향(N/S/E/W)의 시간당 총 차량 수(veh/h)이며,
직진:우회전:좌회전 = 0.60:0.25:0.15 로 분배한다.

직접 실행하면 scenarios/eval/*.rou.xml 5종을 생성한다:
    python scenarios.py
"""
import os
import numpy as np

# 진입 방향별 (진출 방향, 회전종류) — network/ 의 edge 명명(N2C, C2S ...)과 일치
TURN_TABLE = {
    'N': [('S', 'straight'), ('W', 'right'), ('E', 'left')],
    'S': [('N', 'straight'), ('E', 'right'), ('W', 'left')],
    'E': [('W', 'straight'), ('N', 'right'), ('S', 'left')],
    'W': [('E', 'straight'), ('S', 'right'), ('N', 'left')],
}
TURN_RATIO = [0.60, 0.25, 0.15]   # 직진, 우회전, 좌회전

VTYPE = ('    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5"\n'
         '           maxSpeed="13.89" departLane="best"/>')

# ── 평가용 동결 시나리오 5종 (각 방향 veh/h) ──
EVAL_SCENARIOS = {
    'low':        {'N': 300,  'S': 300,  'E': 300,  'W': 300},   # 한산
    'medium':     {'N': 600,  'S': 600,  'E': 600,  'W': 600},   # 평시
    'high':       {'N': 1100, 'S': 1100, 'E': 1100, 'W': 1100},  # 피크
    'asymmetric': {'N': 1100, 'S': 1100, 'E': 300,  'W': 300},   # 출퇴근 편중(N-S 주축)
    'saturated':  {'N': 1500, 'S': 1500, 'E': 1500, 'W': 1500},  # 과포화(용량 초과)
}


def build_routes_xml(demand: dict) -> str:
    """방향별 수요 dict → SUMO .rou.xml 문자열."""
    routes_def = '\n'.join(
        f'    <route id="{d_in}_{d_out}" edges="{d_in}2C C2{d_out}"/>'
        for d_in, turns in TURN_TABLE.items() for d_out, _ in turns
    )
    flows = []
    for d_in, turns in TURN_TABLE.items():
        total = int(demand[d_in])
        for (d_out, _), ratio in zip(turns, TURN_RATIO):
            veh = max(1, int(total * ratio))
            flows.append(
                f'    <flow id="f_{d_in}{d_out}" type="car" route="{d_in}_{d_out}" '
                f'begin="0" end="3600" vehsPerHour="{veh}"/>'
            )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">\n'
        f'{VTYPE}\n\n'
        f'{routes_def}\n\n'
        f'{chr(10).join(flows)}\n'
        '</routes>\n'
    )


def write_routes(out_path: str, demand: dict) -> dict:
    """demand 로 .rou.xml 작성, demand 사본 반환(학습 로그용)."""
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(build_routes_xml(demand))
    return dict(demand)


def random_demand(t_min: int, t_max: int, rng=None) -> dict:
    """각 방향 수요를 [t_min, t_max]에서 랜덤 샘플링 (학습용 도메인 랜덤화)."""
    r = rng if rng is not None else np.random
    return {d: int(r.randint(t_min, t_max + 1)) for d in TURN_TABLE}


def freeze_eval_scenarios(out_dir: str = 'scenarios/eval') -> dict:
    """EVAL_SCENARIOS 5종을 정적 파일로 생성(동결). {name: path} 반환."""
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for name, demand in EVAL_SCENARIOS.items():
        p = os.path.join(out_dir, f'{name}.rou.xml')
        write_routes(p, demand)
        paths[name] = p
        print(f'[FROZEN] {name:11s} {demand}  ->  {p}')
    return paths


if __name__ == '__main__':
    print('평가용 시나리오 5종 동결 생성:')
    freeze_eval_scenarios()
    print('\n완료. evaluate.py 가 이 파일들을 순회하며 Fixed vs SmartSignal 을 비교합니다.')
