"""학습 모델(.pth)을 numpy(.npz)로 내보내 Pi에서 torch 없이 추론하게 한다."""
import argparse, torch, numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="results/smart_signal_e2.pth")
ap.add_argument("--out",   default="results/smart_signal_e2.npz")
a = ap.parse_args()
ckpt = torch.load(a.model, map_location="cpu")
w = ckpt["policy_net"] if isinstance(ckpt, dict) and "policy_net" in ckpt else ckpt
np.savez(a.out,
    W0=w["net.0.weight"].numpy(), b0=w["net.0.bias"].numpy(),
    W2=w["net.2.weight"].numpy(), b2=w["net.2.bias"].numpy(),
    W4=w["net.4.weight"].numpy(), b4=w["net.4.bias"].numpy())
print(f"[OK] {a.out}")
