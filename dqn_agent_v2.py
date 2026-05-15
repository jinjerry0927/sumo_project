"""
DQN Agent for v2 environment (4-way x 3-lane, 4-phase intersection).

기존 dqn_agent.py와 차이점:
- STATE_SIZE / ACTION_SIZE를 환경에서 자동 감지 (하드코딩 X)
- 기본 net/route 경로가 network_v2/ 가리킴
- 결과는 results_v2/ 에 저장 (기존 results/와 분리)
- 그 외 하이퍼파라미터/구조는 기존과 동일
"""

import os
import sys
import csv
import random
import argparse
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim


def set_sumo_home(path=None):
    if "SUMO_HOME" in os.environ:
        return os.environ["SUMO_HOME"]
    candidates = [
        path,
        "C:/Program Files (x86)/Eclipse/Sumo",
        "/usr/share/sumo",
        "/opt/homebrew/share/sumo",
    ]
    for c in candidates:
        if c and os.path.isdir(c):
            os.environ["SUMO_HOME"] = c
            print(f"[INFO] SUMO_HOME 설정: {c}")
            return c
    print("[ERROR] SUMO_HOME을 찾을 수 없습니다.")
    sys.exit(1)


parser = argparse.ArgumentParser()
parser.add_argument("--sumo_home",  default=None)
parser.add_argument("--net",        default="network_v2/intersection.net.xml")
parser.add_argument("--route",      default="network_v2/intersection.rou.xml")
parser.add_argument("--episodes",   type=int, default=1000)
parser.add_argument("--resume",     default=None)
parser.add_argument("--out_dir",    default="results_v2")
args = parser.parse_args()

set_sumo_home(args.sumo_home)
os.makedirs(args.out_dir, exist_ok=True)

import sumo_rl


# ── 하이퍼파라미터 (기존과 동일) ──
LR            = 1e-3
GAMMA         = 0.99
EPSILON_START = 1.0
EPSILON_MIN   = 0.05
EPSILON_DECAY = 0.995
BATCH_SIZE    = 64
TARGET_UPDATE = 10
BUFFER_SIZE   = 50000
CHECKPOINT_EVERY = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] 사용 디바이스: {device}")
if device.type == "cuda":
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")


# ── 환경 한 번 만들어서 state/action 크기 자동 감지 ──
print("[INFO] 환경 검증 중...")
probe_env = sumo_rl.SumoEnvironment(
    net_file=args.net,
    route_file=args.route,
    use_gui=False,
    num_seconds=3600,
    min_green=5,
    max_green=60,
    single_agent=True,
)
probe_obs, _ = probe_env.reset()
STATE_SIZE  = int(np.array(probe_obs).shape[0])
ACTION_SIZE = int(probe_env.action_space.n)
probe_env.close()
print(f"[INFO] STATE_SIZE = {STATE_SIZE}, ACTION_SIZE = {ACTION_SIZE}")


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, float(done)))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (np.array(s, dtype=np.float32),
                np.array(a, dtype=np.int64),
                np.array(r, dtype=np.float32),
                np.array(ns, dtype=np.float32),
                np.array(d, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


policy_net = DQN(STATE_SIZE, ACTION_SIZE).to(device)
target_net = DQN(STATE_SIZE, ACTION_SIZE).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.Adam(policy_net.parameters(), lr=LR)
buffer    = ReplayBuffer(BUFFER_SIZE)

start_episode = 1
epsilon       = EPSILON_START

if args.resume:
    ckpt = torch.load(args.resume, map_location=device)
    # 같은 환경(동일 STATE_SIZE/ACTION_SIZE)에서 저장된 체크포인트만 로드 가능
    policy_net.load_state_dict(ckpt["policy_net"])
    target_net.load_state_dict(ckpt["target_net"])
    optimizer.load_state_dict(ckpt["optimizer"])
    start_episode = ckpt["episode"] + 1
    epsilon       = ckpt["epsilon"]
    print(f"[INFO] 체크포인트 재개: episode {start_episode}, epsilon {epsilon:.3f}")


log_path = os.path.join(args.out_dir, "training_log.csv")
csv_file = open(log_path, "a", newline="")
csv_writer = csv.writer(csv_file)
if start_episode == 1:
    csv_writer.writerow(["episode", "total_reward", "epsilon", "buffer_size"])


def train_step():
    if len(buffer) < BATCH_SIZE:
        return
    s, a, r, ns, d = buffer.sample(BATCH_SIZE)

    s  = torch.from_numpy(s).to(device)
    a  = torch.from_numpy(a).to(device)
    r  = torch.from_numpy(r).to(device)
    ns = torch.from_numpy(ns).to(device)
    d  = torch.from_numpy(d).to(device)

    q_values = policy_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_actions = policy_net(ns).argmax(1)
        target_q     = r + GAMMA * target_net(ns).gather(1, next_actions.unsqueeze(1)).squeeze(1) * (1 - d)

    loss = nn.SmoothL1Loss()(q_values, target_q)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 10)
    optimizer.step()


print(f"\n학습 시작: {args.episodes} 에피소드 (episode {start_episode}부터)\n")

for episode in range(start_episode, args.episodes + 1):
    env = sumo_rl.SumoEnvironment(
        net_file=args.net,
        route_file=args.route,
        use_gui=False,
        num_seconds=3600,
        min_green=5,
        max_green=60,
        single_agent=True,
    )
    obs, _ = env.reset()
    total_reward = 0.0
    done = False

    while not done:
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q = policy_net(torch.from_numpy(np.array(obs, dtype=np.float32)).to(device))
                action = q.argmax().item()

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        buffer.push(obs, action, reward, next_obs, done)
        train_step()

        obs = next_obs
        total_reward += reward

    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
    env.close()

    print(f"Episode {episode:4d}/{args.episodes} | "
          f"Reward: {total_reward:8.1f} | "
          f"Epsilon: {epsilon:.4f} | "
          f"Buffer: {len(buffer)}")
    csv_writer.writerow([episode, round(total_reward, 2), round(epsilon, 4), len(buffer)])
    csv_file.flush()

    if episode % CHECKPOINT_EVERY == 0:
        ckpt_path = os.path.join(args.out_dir, f"checkpoint_ep{episode}.pth")
        torch.save({
            "episode":    episode,
            "epsilon":    epsilon,
            "policy_net": policy_net.state_dict(),
            "target_net": target_net.state_dict(),
            "optimizer":  optimizer.state_dict(),
            "state_size":  STATE_SIZE,
            "action_size": ACTION_SIZE,
        }, ckpt_path)
        print(f"  >> 체크포인트 저장: {ckpt_path}")


csv_file.close()
final_path = os.path.join(args.out_dir, "dqn_final.pth")
torch.save({
    "policy_net":  policy_net.state_dict(),
    "state_size":  STATE_SIZE,
    "action_size": ACTION_SIZE,
}, final_path)
print(f"\n학습 완료. 최종 모델: {final_path}")
print(f"학습 로그: {log_path}")
