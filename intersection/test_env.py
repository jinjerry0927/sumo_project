import os
os.environ['SUMO_HOME'] = 'C:/Program Files (x86)/Eclipse/Sumo'

import sumo_rl

env = sumo_rl.SumoEnvironment(
    net_file='intersection.net.xml',
    route_file='intersection.rou.xml',
    use_gui=True,
    num_seconds=3600,
    min_green=5,
    max_green=60,
    single_agent=True,
)

obs, info = env.reset()
print("환경 초기화 성공!")
print(f"State 크기: {obs.shape}")
print(f"Action 수: {env.action_space.n}")

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