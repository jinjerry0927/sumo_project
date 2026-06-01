"""
SUMO GUI 데모 — 학습된 SmartSignal 정책(rl) 또는 고정신호(fixed)를 시각적으로 시연.

smart_signal 과 동일한 제어 규약을 따른다:
- state: sumo-rl 29차원 (앞 num_green_phases = 현재 phase one-hot)
- DQN 출력: 0=Keep / 1=Next  → 사이클 순서대로 green phase 전환
- 안전 제약: min_green 15s, max_green 60s
"""
import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn


def set_sumo_home():
    if "SUMO_HOME" in os.environ:
        return
    for c in ["C:/Program Files (x86)/Eclipse/Sumo", "/usr/share/sumo", "/opt/homebrew/share/sumo"]:
        if c and os.path.isdir(c):
            os.environ["SUMO_HOME"] = c
            return
    print("[ERROR] SUMO_HOME을 찾을 수 없습니다.")
    sys.exit(1)


set_sumo_home()
import sumo_rl

parser = argparse.ArgumentParser()
parser.add_argument("--mode",       choices=["fixed", "rl"], default="rl",
                    help="fixed: 고정신호 시연 / rl: 학습된 SmartSignal 시연")
parser.add_argument("--model",      default="results/smart_signal.pth")
parser.add_argument("--net",        default="network/intersection.net.xml")
parser.add_argument("--route",      default="network/intersection.rou.xml")
parser.add_argument("--duration",   type=int, default=3600, help="시뮬레이션 시간(초)")
parser.add_argument("--delay",      type=float, default=0.0, help="스텝 간 딜레이(초)")
parser.add_argument("--fixed_hold", type=int, default=6,
                    help="고정신호: 각 green phase 유지 결정스텝 수")
args = parser.parse_args()

MIN_GREEN, MAX_GREEN = 15, 60
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_size),
        )

    def forward(self, x):
        return self.net(x)


# ── RL 모드: 체크포인트에서 차원 자동 인식 ──
policy_net = None
num_green_phases = 4
if args.mode == "rl":
    ckpt = torch.load(args.model, map_location=device)
    if isinstance(ckpt, dict) and "policy_net" in ckpt:
        state_size = ckpt.get("state_size", 29)
        action_size = ckpt.get("action_size", 2)
        num_green_phases = ckpt.get("num_green_phases", 4)
        weights = ckpt["policy_net"]
    else:  # 순수 state_dict
        weights = ckpt
        first_w = weights["net.0.weight"]
        last_w = weights[max(k for k in weights if k.endswith(".weight"))]
        state_size, action_size = first_w.shape[1], last_w.shape[0]
    policy_net = DQN(state_size, action_size).to(device)
    policy_net.load_state_dict(weights)
    policy_net.eval()
    print(f"[INFO] 모델 로드: {args.model} "
          f"(state={state_size}, action={action_size}, green_phases={num_green_phases})")

print(f"\n[INFO] 모드: {'SmartSignal(RL)' if args.mode == 'rl' else '고정신호'}")
print("[INFO] SUMO GUI가 열립니다. ▶ 버튼으로 시작하세요.\n")

env = sumo_rl.SumoEnvironment(
    net_file=args.net, route_file=args.route, use_gui=True,
    num_seconds=args.duration, min_green=MIN_GREEN, max_green=MAX_GREEN,
    single_agent=True,
)

obs, _ = env.reset()
total_reward = 0.0
done = False
step = 0

while not done:
    cur_phase = int(np.array(obs[:num_green_phases]).argmax())
    if args.mode == "fixed":
        env_action = (step // args.fixed_hold) % num_green_phases  # 사이클 고정
    else:
        with torch.no_grad():
            q = policy_net(torch.FloatTensor(np.array(obs, dtype=np.float32)).to(device))
            dqn_action = int(q.argmax().item())  # 0=keep, 1=next
        env_action = cur_phase if dqn_action == 0 else (cur_phase + 1) % num_green_phases

    obs, reward, terminated, truncated, _ = env.step(env_action)
    done = terminated or truncated
    total_reward += reward
    step += 1
    if args.delay > 0:
        time.sleep(args.delay)
    if step % 100 == 0:
        print(f"  Step {step:4d} | phase {cur_phase} | Reward {reward:7.3f} | Total {total_reward:8.1f}")

env.close()
print(f"\n{'='*45}")
print(f"  모드     : {'SmartSignal(RL)' if args.mode == 'rl' else '고정신호'}")
print(f"  총 스텝  : {step}")
print(f"  총 보상  : {total_reward:.2f}")
print(f"{'='*45}")
