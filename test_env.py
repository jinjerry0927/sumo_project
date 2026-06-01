"""
SUMO 환경 스모크 테스트 — smart_signal 학습 전 network/ 가 정상 로드되고
state/action 차원이 기대대로인지 확인한다 (랜덤 정책으로 1 에피소드).
"""
import os
os.environ.setdefault('SUMO_HOME', 'C:/Program Files (x86)/Eclipse/Sumo')

import sumo_rl

env = sumo_rl.SumoEnvironment(
    net_file='network/intersection.net.xml',
    route_file='network/intersection.rou.xml',
    use_gui=True,
    num_seconds=3600,
    min_green=15,
    max_green=60,
    single_agent=True,
)

obs, info = env.reset()
print("환경 초기화 성공!")
print(f"State 크기: {obs.shape}  (smart_signal 기대값: 29)")
print(f"Green phase 수(action_space.n): {env.action_space.n}  (기대값: 4)")

done = False
step = 0
while not done:
    action = env.action_space.sample()  # 랜덤 행동 (DQN 붙이기 전 테스트용)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    step += 1
    if step % 100 == 0:
        print(f"Step {step} | reward: {reward:.3f}")

env.close()
print("시뮬레이션 완료!")
