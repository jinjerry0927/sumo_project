import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import sumo_rl

# ── SUMO_HOME 설정 ─────────────────────────────────────────
def set_sumo_home():
    if "SUMO_HOME" in os.environ:
        return
    candidates = [
        "C:/Program Files (x86)/Eclipse/Sumo",
        "/usr/share/sumo",
        "/opt/homebrew/share/sumo",
    ]
    for c in candidates:
        if c and os.path.isdir(c):
            os.environ["SUMO_HOME"] = c
            return
    print("[ERROR] SUMO_HOME을 찾을 수 없습니다.")
    sys.exit(1)

set_sumo_home()

# ── CLI 인자 ──────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--mode",       choices=["fixed", "rl"], default="rl",
                    help="fixed: 고정신호 시연 / rl: 학습된 모델 시연")
parser.add_argument("--model",      default="results/dqn_final.pth")
parser.add_argument("--net",        default="network_v1/intersection.net.xml")
parser.add_argument("--route",      default="network_v1/intersection.rou.xml")
parser.add_argument("--duration",   type=int, default=6000,
                    help="시뮬레이션 시간(초) - 기본 1800초(30분)")
parser.add_argument("--delay",      type=float, default=0.5,
                    help="스텝 간 딜레이(초) - 클수록 느려짐, 기본 0.05")
parser.add_argument("--fixed_cycle",type=int, default=30,
                    help="고정신호 교대 주기(스텝), 기본 30")
args = parser.parse_args()

# ── DQN 모델 ─────────────────────────────────────────────
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )
    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── RL 모드일 때만 모델 로드 ──────────────────────────────
policy_net = None
if args.mode == "rl":
    policy_net = DQN(11, 2).to(device)
    ckpt = torch.load(args.model, map_location=device)
    # checkpoint 형식 또는 state_dict 형식 둘 다 지원
    if "policy_net" in ckpt:
        policy_net.load_state_dict(ckpt["policy_net"])
    else:
        policy_net.load_state_dict(ckpt)
    policy_net.eval()
    print(f"[INFO] 모델 로드: {args.model}")

# ── 환경 생성 (GUI ON) ────────────────────────────────────
print(f"\n[INFO] 모드: {'RL 신호' if args.mode == 'rl' else '고정 신호'}")
print(f"[INFO] SUMO GUI가 열립니다. 하단 슬라이더로 속도 조절 가능합니다.\n")

env = sumo_rl.SumoEnvironment(
    net_file=args.net,
    route_file=args.route,
    use_gui=True,
    num_seconds=args.duration,
    min_green=5,
    max_green=60,
    single_agent=True,
)

obs, _ = env.reset()
total_reward = 0.0
done = False
step = 0

print("▶ 시뮬레이션 시작 (SUMO GUI에서 ▶ 버튼을 눌러주세요)\n")

while not done:
    # 행동 선택
    if args.mode == "fixed":
        action = (step // args.fixed_cycle) % 2  # 30스텝마다 교대
    else:
        with torch.no_grad():
            q = policy_net(torch.FloatTensor(obs).to(device))
            action = q.argmax().item()

    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    total_reward += reward
    step += 1

    # 속도 조절용 딜레이
    if args.delay > 0:
        time.sleep(args.delay)

    if step % 100 == 0:
        print(f"  Step {step:4d} | Reward: {reward:7.3f} | Total: {total_reward:8.1f} | Action: {action}")

env.close()
print(f"\n{'='*45}")
print(f"  모드     : {'RL 신호' if args.mode == 'rl' else '고정 신호'}")
print(f"  총 스텝  : {step}")
print(f"  총 보상  : {total_reward:.2f}")
print(f"{'='*45}")