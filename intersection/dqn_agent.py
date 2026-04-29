import os
os.environ['SUMO_HOME'] = 'C:/Program Files (x86)/Eclipse/Sumo'

import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import sumo_rl

# ── DQN 신경망 ──────────────────────────────────────────
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, x):
        return self.net(x)

# ── Replay Buffer ────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return np.array(s), np.array(a), np.array(r), np.array(ns), np.array(d)

    def __len__(self):
        return len(self.buffer)

# ── 하이퍼파라미터 ───────────────────────────────────────
STATE_SIZE   = 11
ACTION_SIZE  = 2
LR           = 1e-3
GAMMA        = 0.99
EPSILON      = 1.0
EPSILON_MIN  = 0.05
EPSILON_DECAY= 0.995
BATCH_SIZE   = 32
EPISODES     = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 디바이스: {device}")

policy_net = DQN(STATE_SIZE, ACTION_SIZE).to(device)
target_net = DQN(STATE_SIZE, ACTION_SIZE).to(device)
target_net.load_state_dict(policy_net.state_dict())
optimizer  = optim.Adam(policy_net.parameters(), lr=LR)
buffer     = ReplayBuffer()

# ── 학습 함수 ────────────────────────────────────────────
def train_step():
    if len(buffer) < BATCH_SIZE:
        return
    s, a, r, ns, d = buffer.sample(BATCH_SIZE)
    s  = torch.FloatTensor(s).to(device)
    a  = torch.LongTensor(a).to(device)
    r  = torch.FloatTensor(r).to(device)
    ns = torch.FloatTensor(ns).to(device)
    d  = torch.FloatTensor(d).to(device)

    q_values = policy_net(s).gather(1, a.unsqueeze(1)).squeeze()
    with torch.no_grad():
        target_q = r + GAMMA * target_net(ns).max(1)[0] * (1 - d)

    loss = nn.MSELoss()(q_values, target_q)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# ── 메인 학습 루프 ───────────────────────────────────────
epsilon = EPSILON

for episode in range(1, EPISODES + 1):
    env = sumo_rl.SumoEnvironment(
        net_file='intersection.net.xml',
        route_file='intersection.rou.xml',
        use_gui=False,          # 학습 중엔 GUI 끄면 빠름
        num_seconds=3600,
        min_green=5,
        max_green=60,
        single_agent=True,
    )
    obs, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        # Epsilon-greedy 행동 선택
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q = policy_net(torch.FloatTensor(obs).to(device))
                action = q.argmax().item()

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        buffer.push(obs, action, reward, next_obs, float(done))
        train_step()

        obs = next_obs
        total_reward += reward

    # Target network 업데이트 (매 에피소드)
    target_net.load_state_dict(policy_net.state_dict())
    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
    env.close()

    print(f"Episode {episode}/{EPISODES} | Total Reward: {total_reward:.1f} | Epsilon: {epsilon:.3f}")

# 모델 저장
torch.save(policy_net.state_dict(), "dqn_model.pth")
print("모델 저장 완료: dqn_model.pth")