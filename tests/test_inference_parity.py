"""numpy 엣지 추론이 torch 추론과 동일 action 을 내는지 검증(랜덤 100개). 직접 실행."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 루트의 edge_server import
import numpy as np, torch
from edge_server import load, infer

NPZ, PTH = "results/smart_signal_e2.npz", "results/smart_signal_e2.pth"

def torch_net():
    import torch.nn as nn
    ck = torch.load(PTH, map_location="cpu")
    w = ck["policy_net"] if "policy_net" in ck else ck
    ss = w["net.0.weight"].shape[1]; a_s = w["net.4.weight"].shape[0]
    net = nn.Sequential(nn.Linear(ss,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,a_s))
    sd = {k.replace("net.",""):v for k,v in w.items()}
    net.load_state_dict({f"{i}.{p}":sd[f"{i}.{p}"] for i in (0,2,4) for p in ("weight","bias")})
    net.eval(); return net, ss

def main():
    layers = load(NPZ); net, ss = torch_net()
    rng = np.random.default_rng(0)
    for _ in range(100):
        x = rng.random(ss).astype(np.float32)
        a_np = infer(layers, x)
        with torch.no_grad():
            a_t = int(net(torch.from_numpy(x)).argmax().item())
        assert a_np == a_t, f"불일치 np={a_np} torch={a_t}"
    print("[PASS] numpy == torch (100/100)")

if __name__ == "__main__":
    main()
