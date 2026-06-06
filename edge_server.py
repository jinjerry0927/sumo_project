"""엣지(라즈베리파이) 추론 서버 — torch 없이 numpy 로 DQN 추론.
줄단위 JSON over TCP: {"obs":[...]} 수신 → {"action":int} 회신. demo.py --hil 와 짝."""
import argparse, json, socket
import numpy as np

def load(path):
    d = np.load(path)
    return (d["W0"], d["b0"]), (d["W2"], d["b2"]), (d["W4"], d["b4"])

def infer(layers, x):
    (W0,b0),(W2,b2),(W4,b4) = layers
    x = np.asarray(x, dtype=np.float32)
    h = np.maximum(0, W0 @ x + b0)
    h = np.maximum(0, W2 @ h + b2)
    return int(np.argmax(W4 @ h + b4))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="smart_signal_e2.npz")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=9999)
    a = ap.parse_args()
    layers = load(a.weights)
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((a.host, a.port)); srv.listen(1)
    print(f"[edge] listening on {a.host}:{a.port}  (weights={a.weights})")
    while True:
        conn, addr = srv.accept(); print("[edge] connected", addr)
        with conn, conn.makefile("rwb") as f:
            for line in f:
                try:
                    act = infer(layers, json.loads(line.decode())["obs"])
                    f.write((json.dumps({"action": act})+"\n").encode()); f.flush()
                except Exception as e:
                    f.write((json.dumps({"error": str(e)})+"\n").encode()); f.flush()

if __name__ == "__main__":
    main()
