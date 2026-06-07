# 실시간 HIL 대시보드 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `demo.py` 실행 중 SUMO GUI 옆에 띄우는 웹 대시보드로, E2 검지기가 본 차로별 대기차량(12차로)·SmartSignal 두뇌의 결정(KEEP/SWITCH + Q값)·핵심 지표를 매 스텝 실시간 표시한다.

**Architecture:** demo.py 제어루프가 매 스텝 상태 dict를 메모리에 push → 파이썬 stdlib HTTP 서버(`dashboard.py`, 백그라운드 스레드)가 보관 → 브라우저(`dashboard.html`)가 200ms 폴링으로 DOM 갱신. HIL 모드에선 Q값을 엣지서버(`edge_server.py`)가 확장 프로토콜 `{action, q}`로 함께 반환. 차로 차량수는 `traci.lanearea` 원시 getter로 직접 읽어 관측 정규화와 분리.

**Tech Stack:** Python(stdlib `http.server`/`threading`/`socket`/`json`), numpy(엣지 추론), traci(SUMO), PyTorch(로컬 추론·패리티 검증), 바닐라 JS(폴링·DOM).

설계문서: `docs/superpowers/specs/2026-06-07-hil-dashboard-design.md`

> **커밋 규약**: 이 저장소는 단계 마감을 `/wrap`이 담당한다. 아래 각 Task의 commit 스텝은 표준 형식상 포함하나, `/next` 실행 흐름에선 커밋을 보류하고 검증까지만 수행한 뒤 `/wrap`에서 일괄 커밋한다.

---

## Task 1: 엣지 추론 Q값(logits) 반환 + 패리티 확장

**Files:**
- Modify: `edge_server.py:10-15` (infer 분리), `edge_server.py:28-36` (서버 응답에 q 추가)
- Test: `tests/test_inference_parity.py` (logits 비교 추가)

- [x] **Step 1: 패리티 테스트를 logits 비교로 확장 (실패 유도)**

`tests/test_inference_parity.py` 전체를 아래로 교체:
```python
"""numpy 엣지 추론이 torch 와 동일 action + 동일 logits(q값) 을 내는지 검증(랜덤 100개). 직접 실행."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 루트의 edge_server import
import numpy as np, torch
from edge_server import load, forward, infer

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
        q_np = forward(layers, x)
        a_np = infer(layers, x)
        with torch.no_grad():
            q_t = net(torch.from_numpy(x)).numpy()
        a_t = int(np.argmax(q_t))
        assert a_np == a_t, f"action 불일치 np={a_np} torch={a_t}"
        assert np.allclose(q_np, q_t, atol=1e-4), f"logits 불일치 np={q_np} torch={q_t}"
    print("[PASS] numpy == torch — action + logits (100/100)")

if __name__ == "__main__":
    main()
```

- [x] **Step 2: 실행해서 실패 확인**

Run: `python tests/test_inference_parity.py`
Expected: `ImportError: cannot import name 'forward' from 'edge_server'` (아직 forward 없음)

- [x] **Step 3: edge_server에 forward 분리 + 서버 응답 확장**

`edge_server.py:10-15` 의 `infer` 함수를 아래로 교체:
```python
def forward(layers, x):
    (W0,b0),(W2,b2),(W4,b4) = layers
    x = np.asarray(x, dtype=np.float32)
    h = np.maximum(0, W0 @ x + b0)
    h = np.maximum(0, W2 @ h + b2)
    return W4 @ h + b4   # logits, shape (action_size,)

def infer(layers, x):
    return int(np.argmax(forward(layers, x)))
```

`edge_server.py:28-36` 의 서버 루프(`while True:` 블록)를 아래로 교체:
```python
    while True:
        conn, addr = srv.accept(); print("[edge] connected", addr)
        with conn, conn.makefile("rwb") as f:
            for line in f:
                try:
                    q = forward(layers, json.loads(line.decode())["obs"])
                    resp = {"action": int(np.argmax(q)), "q": [float(v) for v in q]}
                    f.write((json.dumps(resp)+"\n").encode()); f.flush()
                except Exception as e:
                    f.write((json.dumps({"error": str(e)})+"\n").encode()); f.flush()
```

- [x] **Step 4: 실행해서 통과 확인**

Run: `python tests/test_inference_parity.py`
Expected: `[PASS] numpy == torch — action + logits (100/100)`

- [x] **Step 5: 커밋**

```bash
git add edge_server.py tests/test_inference_parity.py
git commit -m "feat(hil): 엣지서버 q값(logits) 반환 + 패리티 logits 검증"
```

---

## Task 2: 대시보드 HTTP 서버 (`dashboard.py`)

**Files:**
- Create: `dashboard.py`
- Test: `tests/test_dashboard.py`

- [x] **Step 1: 스모크 테스트 작성 (실패 유도)**

`tests/test_dashboard.py` 생성:
```python
"""Dashboard 서버 스모크 — update 한 상태가 /state 로 그대로 반환되는지 검증. 직접 실행."""
import os, sys, json, urllib.request
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 루트의 dashboard import
from dashboard import Dashboard

def main():
    d = Dashboard()
    url = d.start(port=8765)
    try:
        sample = {"step": 3, "decision": "keep",
                  "lanes": [{"label": "북 직진", "group": "직진", "count": 17, "cap": 20, "is_green": True}]}
        d.update(sample)
        got = json.loads(urllib.request.urlopen(url + "/state", timeout=3).read().decode())
        assert got == sample, f"/state 불일치: {got}"
    finally:
        d.stop()
    print("[PASS] dashboard /state 왕복 OK")

if __name__ == "__main__":
    main()
```

- [x] **Step 2: 실행해서 실패 확인**

Run: `python tests/test_dashboard.py`
Expected: `ModuleNotFoundError: No module named 'dashboard'`

- [x] **Step 3: dashboard.py 작성**

`dashboard.py` 생성:
```python
"""실시간 HIL 대시보드 — 경량 HTTP 서버(파이썬 stdlib, 새 의존성 없음).
demo.py 제어루프가 매 스텝 update(state) 로 최신 상태를 넣고,
브라우저(dashboard.html)가 200ms 마다 GET /state 를 폴링해 DOM 을 갱신한다.
GET /        -> dashboard.html
GET /state   -> 최신 상태 JSON"""
import json, os, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_HTML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.html")


class Dashboard:
    def __init__(self):
        self._state = {}
        self._lock = threading.Lock()
        self._srv = None
        self._thread = None

    def update(self, state):
        with self._lock:
            self._state = state

    def _snapshot(self):
        with self._lock:
            return dict(self._state)

    def start(self, host="127.0.0.1", port=8000):
        dash = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):   # 콘솔 스팸 억제
                pass

            def _send(self, body, ctype):
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/state":
                    self._send(json.dumps(dash._snapshot()).encode("utf-8"),
                               "application/json; charset=utf-8")
                else:
                    try:
                        with open(_HTML, "rb") as fp:
                            body = fp.read()
                    except FileNotFoundError:
                        self.send_error(404, "dashboard.html not found")
                        return
                    self._send(body, "text/html; charset=utf-8")

        self._srv = ThreadingHTTPServer((host, port), Handler)
        self._thread = threading.Thread(target=self._srv.serve_forever, daemon=True)
        self._thread.start()
        return f"http://{host}:{port}"

    def stop(self):
        if self._srv is not None:
            self._srv.shutdown()
            self._srv.server_close()
            self._srv = None
```

- [x] **Step 4: 실행해서 통과 확인**

Run: `python tests/test_dashboard.py`
Expected: `[PASS] dashboard /state 왕복 OK`

- [x] **Step 5: 커밋**

```bash
git add dashboard.py tests/test_dashboard.py
git commit -m "feat(hil): 대시보드 경량 HTTP 서버 + /state 스모크 테스트"
```

---

## Task 3: 대시보드 페이지 (`dashboard.html`)

**Files:**
- Create: `dashboard.html`

- [x] **Step 1: dashboard.html 작성** (12차로 막대 + 두뇌 패널 + 상단지표, 200ms 폴링)

`dashboard.html` 생성:
```html
<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>실시간 HIL 대시보드</title>
<style>
  * { box-sizing: border-box; }
  body { margin:0; background:#070b10; color:#cdd6e0; font-family:system-ui,sans-serif; }
  .wrap { max-width:1100px; margin:0 auto; padding:16px; }
  h4 { margin:0 0 10px; font-size:12px; color:#7fd1c0; text-transform:uppercase; letter-spacing:1px; }
  .D { background:#0b0f14; border:1px solid #2a3441; border-radius:10px; padding:16px; }
  .strip { display:flex; justify-content:space-around; border-bottom:1px solid #2a3441; padding-bottom:12px; margin-bottom:14px; }
  .m { text-align:center; } .m .v { font-size:32px; font-weight:800; color:#fff; line-height:1; }
  .m .v.good { color:#2ecc71; } .m .d { font-size:11px; color:#8a97a8; margin-top:5px; }
  .body { display:flex; gap:18px; }
  .sens { flex:1; }
  .grp { font-size:10px; color:#5b6675; margin:8px 0 3px; text-transform:uppercase; letter-spacing:.5px; }
  .lane { display:flex; align-items:center; gap:8px; margin:3px 0; }
  .lane .nm { width:78px; font-size:12px; color:#aeb9c7; }
  .lane .tk { flex:1; height:14px; background:#161d28; border-radius:3px; overflow:hidden; }
  .lane .bar { height:100%; border-radius:3px; transition:width .25s ease; }
  .lane .n { width:26px; text-align:right; font-size:12px; font-weight:700; color:#fff; }
  .lane.green .nm { color:#2ecc71; }
  .grn { background:linear-gradient(90deg,#27ae60,#2ecc71); }
  .ora { background:linear-gradient(90deg,#e67e22,#f39c12); }
  .gry { background:#3a4654; }
  .brain { flex:0 0 210px; background:#0e141c; border-radius:8px; padding:14px; text-align:center; }
  .dec { font-size:36px; font-weight:900; letter-spacing:1px; margin:8px 0; }
  .keep { color:#5dade2; } .switch { color:#f39c12; text-shadow:0 0 14px rgba(243,156,18,.6); }
  .fixed { color:#8a97a8; font-size:22px; }
  .sub { font-size:11px; color:#8a97a8; }
  .qwrap { display:flex; gap:12px; justify-content:center; align-items:flex-end; height:70px; margin-top:16px; }
  .qbar { width:48px; border-radius:4px 4px 0 0; display:flex; flex-direction:column; justify-content:flex-end;
          align-items:center; color:#06121f; font-weight:700; font-size:11px; padding-bottom:2px; transition:height .25s; }
  .qsel { background:#5dade2; } .qoff { background:#34495e; color:#cdd6e0; }
  .qlbl { font-size:10px; color:#8a97a8; margin-top:5px; }
  .leg { display:flex; gap:16px; align-items:center; margin-top:12px; font-size:11px; color:#8a97a8; }
  .dot { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:4px; vertical-align:middle; }
  .off { opacity:.4; }
</style>
</head>
<body>
<div class="wrap">
  <div class="D">
    <div class="strip">
      <div class="m"><div class="v" id="m-wait">–</div><div class="d">현재 대기차량</div></div>
      <div class="m"><div class="v good" id="m-avg">–</div><div class="d">평균 대기(s)</div></div>
      <div class="m"><div class="v" id="m-thru">–</div><div class="d">누적 통과</div></div>
      <div class="m"><div class="v" id="m-phase">–</div><div class="d" id="m-phase-name">현재 현시</div></div>
    </div>
    <div class="body">
      <div class="sens">
        <h4>E2 검지기가 보는 것 — 차로별 대기차량(12차로)</h4>
        <div id="lanes"></div>
        <div class="sub" style="margin-top:10px">회색=우회전 · 초록=직진 · 주황=좌회전 · 숫자=정지 차량수 — <b>SUMO엔 안 보이는, 모델이 받는 입력</b></div>
      </div>
      <div class="brain">
        <h4 style="color:#5dade2">SmartSignal 두뇌</h4>
        <div class="dec keep" id="b-dec">–</div>
        <div class="sub" id="b-sub">대기 중</div>
        <div class="qwrap" id="qwrap" style="display:none">
          <div class="qbar qsel" id="q0">–</div>
          <div class="qbar qoff" id="q1">–</div>
        </div>
        <div class="qlbl" id="qlbl" style="display:none">Q값:  유지 / 전환</div>
      </div>
    </div>
    <div class="leg">
      <span><span class="dot" style="background:#2ecc71"></span>녹색현시</span>
      <span><span class="dot" style="background:#f1c40f"></span>황색</span>
      <span><span class="dot" style="background:#e74c3c"></span>적색</span>
      <span id="edge" style="margin-left:auto">엣지: Raspberry Pi · numpy · 53D→128→128→2</span>
    </div>
  </div>
</div>
<script>
const GROUP_CLASS = {"우회전":"gry", "직진":"grn", "좌회전":"ora"};
function render(s) {
  if (!s || !s.lanes) return;
  document.getElementById("m-wait").textContent = s.metrics.waiting_vehicles;
  document.getElementById("m-avg").textContent = s.metrics.avg_wait;
  document.getElementById("m-thru").textContent = s.metrics.throughput;
  document.getElementById("m-phase").textContent = "P" + s.phase;
  document.getElementById("m-phase-name").textContent = s.phase_name;
  // 차로 막대 (방향별 그룹 헤더는 4개마다… 여기선 3개마다 = 한 방향)
  const DIRS = ["북","남","동","서"];
  let html = "";
  s.lanes.forEach((ln, i) => {
    if (i % 3 === 0) html += `<div class="grp">${DIRS[i/3]}</div>`;
    const w = Math.min(100, Math.round(ln.count / ln.cap * 100));
    html += `<div class="lane ${ln.is_green ? "green" : ""}">
      <span class="nm">${ln.label}</span>
      <div class="tk"><div class="bar ${GROUP_CLASS[ln.group]}" style="width:${w}%"></div></div>
      <span class="n">${ln.count}</span></div>`;
  });
  document.getElementById("lanes").innerHTML = html;
  // 두뇌
  const dec = document.getElementById("b-dec");
  if (s.decision === "keep") { dec.textContent = "● KEEP"; dec.className = "dec keep"; }
  else if (s.decision === "switch") { dec.textContent = "▶ SWITCH"; dec.className = "dec switch"; }
  else { dec.textContent = "고정주기"; dec.className = "dec fixed"; }
  document.getElementById("b-sub").textContent = `현시 P${s.phase} · ${Math.round(s.time_in_phase)}s 경과`;
  const qw = document.getElementById("qwrap"), ql = document.getElementById("qlbl");
  if (s.q && s.q.length === 2) {
    qw.style.display = "flex"; ql.style.display = "block";
    const lo = Math.min(s.q[0], s.q[1]), hi = Math.max(s.q[0], s.q[1]), rng = (hi - lo) || 1e-6;
    const sel = s.q[0] >= s.q[1] ? 0 : 1;
    [0,1].forEach(k => {
      const el = document.getElementById("q"+k);
      el.style.height = (22 + 46 * (s.q[k] - lo) / rng) + "px";
      el.textContent = s.q[k].toFixed(1);
      el.className = "qbar " + (k === sel ? "qsel" : "qoff");
    });
  } else { qw.style.display = "none"; ql.style.display = "none"; }
  if (s.edge) document.getElementById("edge").textContent =
    `엣지: ${s.edge === "local" ? "로컬" : "Pi " + s.edge} · numpy · 53D→128→128→2`;
}
async function poll() {
  try { const r = await fetch("/state"); render(await r.json()); } catch (e) {}
}
setInterval(poll, 200); poll();
</script>
</body>
</html>
```

- [x] **Step 2: 단독 표시 확인** (서버만 띄워 페이지가 뜨는지)

Run: `python -c "import time; from dashboard import Dashboard; d=Dashboard(); print(d.start(port=8766)); d.update({'step':1,'phase':0,'phase_name':'남북직진','time_in_phase':10,'decision':'keep','q':[2.1,0.4],'metrics':{'waiting_vehicles':42,'avg_wait':11.3,'throughput':318},'lanes':[{'label':'북 우회전','group':'우회전','count':2,'cap':20,'is_green':True},{'label':'북 직진','group':'직진','count':17,'cap':20,'is_green':True},{'label':'북 좌회전','group':'좌회전','count':5,'cap':20,'is_green':False}]*4,'edge':'local'}); time.sleep(30)"`
Expected: `http://127.0.0.1:8766` 출력. 브라우저로 열면 상단지표·12막대·KEEP·Q막대가 보임. (확인 후 Ctrl+C)

- [x] **Step 3: 커밋**

```bash
git add dashboard.html
git commit -m "feat(hil): 대시보드 페이지(12차로 막대 + 두뇌 Q값, 200ms 폴링)"
```

---

## Task 4: demo.py 통합 — `--dashboard`

**Files:**
- Modify: `demo.py:70-73` (인자), `demo.py:111-127` (EdgeClient.last_q), `demo.py:158-163` (검지기 강제 로드), `demo.py:197-227` (대시보드 기동·상태 push·정리)

- [x] **Step 1: argparse 에 --dashboard 추가**

`demo.py:70-72` 의 `--hil`/`--host`/`--port` add_argument 3줄 **다음**(line 72 뒤)에 추가:
```python
parser.add_argument("--dashboard", action="store_true", help="실시간 대시보드(웹) 동시 기동")
parser.add_argument("--dash_port", type=int, default=8000, help="대시보드 포트")
```

- [x] **Step 2: EdgeClient 가 q값을 보관하도록 수정**

`demo.py:111-120` 의 `EdgeClient.__init__` 와 `act` 를 아래로 교체:
```python
class EdgeClient:
    def __init__(self, host, port):
        self.sock = socket.create_connection((host, port), timeout=5)
        self.f = self.sock.makefile("rwb")
        self.last_q = None
        print(f"[hil] 엣지서버 연결: {host}:{port}")

    def act(self, obs):
        self.f.write((json.dumps({"obs": list(map(float, obs))}) + "\n").encode())
        self.f.flush()
        resp = json.loads(self.f.readline().decode())
        self.last_q = resp.get("q")
        return int(resp["action"])
```

- [x] **Step 3: --dashboard 일 때 E2 검지기 로드 (관측 클래스는 불변)**

`demo.py:158-163` 의 ENV_KWARGS 구성 블록을 아래로 교체:
```python
ENV_KWARGS = {}
extra_cmd = "--no-step-log"   # VSCode 터미널의 'Step #...' 스팸 억제
if args.hil:
    from e2_observation import E2ObservationFunction
    ENV_KWARGS = dict(observation_class=E2ObservationFunction)
    extra_cmd = "-a network/e2.add.xml --no-step-log"
elif args.dashboard:
    # 대시보드는 차로별 차량수를 traci.lanearea 로 직접 읽으므로 검지기 로드만 필요.
    # 관측 클래스는 기본 유지 → 로컬 rl 모델(smart_signal.pth) 차원과 일치.
    extra_cmd = "-a network/e2.add.xml --no-step-log"
```

- [x] **Step 4: 대시보드 기동 + 차로/현시 매핑 (제어루프 진입 전)**

`demo.py:201` 의 `edge = EdgeClient(...) if args.hil else None` **다음 줄**에 추가:
```python

# ── 대시보드(웹) 기동 + 차로/현시 매핑 ──
dash = None
LANE_ORDER = LANE_LABELS = GROUP = PHASE_NAMES = PHASE_GREEN = cap_map = None
arrived = 0
dqn_action = None
last_qvals = None
if args.dashboard:
    import webbrowser
    from dashboard import Dashboard
    DIRS = [("N", "북"), ("S", "남"), ("E", "동"), ("W", "서")]
    SUB = [("0", "우회전"), ("1", "직진"), ("2", "좌회전")]
    LANE_ORDER = [f"{d}2C_{i}" for d, _ in DIRS for i, _ in SUB]
    LANE_LABELS = {f"{d}2C_{i}": f"{dk} {sk}" for d, dk in DIRS for i, sk in SUB}
    GROUP = {f"{d}2C_{i}": sk for d, _ in DIRS for i, sk in SUB}
    PHASE_NAMES = {0: "남북직진", 1: "남북좌회전", 2: "동서직진", 3: "동서좌회전"}
    PHASE_GREEN = {0: {"S2C_1", "N2C_1"}, 1: {"S2C_2", "N2C_2"},
                   2: {"E2C_1", "W2C_1"}, 3: {"E2C_2", "W2C_2"}}
    VEH_LEN = 7.5
    cap_map = {lane: max(env.sumo.lanearea.getLength("e2_" + lane) / VEH_LEN, 1.0)
               for lane in LANE_ORDER}
    dash = Dashboard()
    url = dash.start(port=args.dash_port)
    print(f"[dashboard] {url}  ← 브라우저에서 열기")
    try:
        webbrowser.open(url)
    except Exception:
        pass
```

- [x] **Step 5: rl 분기에서 q값 보관**

`demo.py:209-216` 의 `else:`(rl) 분기를 아래로 교체:
```python
    else:
        if edge is not None:
            dqn_action = edge.act(obs)          # 엣지서버(소켓) 추론
            last_qvals = edge.last_q
        else:
            with torch.no_grad():
                q = policy_net(torch.FloatTensor(np.array(obs, dtype=np.float32)).to(device))
                dqn_action = int(q.argmax().item())   # 0=keep, 1=next
                last_qvals = q.detach().cpu().numpy().tolist()
        env_action = cur_phase if dqn_action == 0 else (cur_phase + 1) % num_green_phases
```

- [x] **Step 6: 매 스텝 throughput 누적 + 대시보드 상태 push**

`demo.py:218` 의 `obs, reward, terminated, truncated, info = env.step(env_action)` **다음 줄**에 추가:
```python
    if args.dashboard:
        arrived += env.sumo.simulation.getArrivedNumber()
        la = env.sumo.lanearea
        halting = {lane: int(la.getLastStepHaltingNumber("e2_" + lane)) for lane in LANE_ORDER}
        disp_phase = int(np.array(obs[:num_green_phases]).argmax())
        decision = ("keep" if dqn_action == 0 else "switch") if args.mode == "rl" else "fixed"
        mean_wait = info.get("system_mean_waiting_time") if isinstance(info, dict) else None
        dash.update({
            "step": step,
            "phase": disp_phase, "phase_name": PHASE_NAMES[disp_phase],
            "time_in_phase": 0,
            "decision": decision,
            "q": last_qvals if args.mode == "rl" else None,
            "metrics": {
                "waiting_vehicles": sum(halting.values()),
                "avg_wait": round(float(mean_wait), 1) if mean_wait is not None else 0.0,
                "throughput": int(arrived),
            },
            "lanes": [{
                "label": LANE_LABELS[lane], "group": GROUP[lane],
                "count": halting[lane], "cap": int(cap_map[lane]),
                "is_green": (lane in PHASE_GREEN[disp_phase]) or lane.endswith("_0"),
            } for lane in LANE_ORDER],
            "mode": args.mode,
            "edge": f"{args.host}:{args.port}" if args.hil else "local",
        })
```

> 비고: `time_in_phase` 는 sumo_rl 가 외부로 노출하지 않아 0 고정(현시 P# 표기로 충분). 발표에 경과초가 꼭 필요하면 phase 변화 감지로 후속 추가 가능(YAGNI — 지금은 생략).

- [x] **Step 7: 루프 종료 후 대시보드 정리**

`demo.py:230-231` 의 `if edge:` / `edge.close()` **다음 줄**에 추가:
```python
if dash:
    dash.stop()
```

- [x] **Step 8: 구문 검사**

Run: `python -c "import py_compile; py_compile.compile('demo.py', doraise=True); print('OK')"`
Expected: `OK`

- [x] **Step 9: 커밋**

```bash
git add demo.py
git commit -m "feat(hil): demo --dashboard (실시간 웹 대시보드 연동)"
```

---

## Task 5: gitignore + 최종 통합 수동 검증

**Files:**
- Modify: `.gitignore`

- [x] **Step 1: 브레인스토밍 산출물 비추적**

`.gitignore` 맨 끝에 추가:
```
# 브레인스토밍 시각화 산출물(재생성 가능)
.superpowers/
```

- [x] **Step 2: 로컬 통합 검증 (대시보드 + 로컬 RL)** — 사람이 직접 수행

Run: `python demo.py --dashboard --mode rl --scenario asymmetric --duration 600`
Expected: SUMO GUI + 브라우저 자동 오픈. ▶ 재생 시 12차로 막대·상단지표·KEEP/SWITCH·Q값 막대가 차량 흐름에 맞춰 실시간 갱신. (god-view 29D 모델이 구동, 대시보드는 E2 원시값 표시.)

- [x] **Step 3: HIL 통합 검증 (대시보드 + 엣지서버)** — 사람이 직접 수행

터미널 A: `python edge_server.py --weights results/smart_signal_e2.npz`
터미널 B: `python demo.py --hil --dashboard --scenario asymmetric --duration 600`
Expected: `[hil] 엣지서버 연결` + `[dashboard] http://127.0.0.1:8000`. 브라우저에서 Pi/엣지가 낸 Q값 막대가 실시간 표시, 하단 범례가 `엣지: Pi 127.0.0.1:9999 …` 로 표기. (실물 Pi면 `--host <PI_IP>`.)

- [x] **Step 4: 커밋**

```bash
git add .gitignore
git commit -m "chore: .superpowers/ gitignore (브레인스토밍 산출물)"
```

---

## Self-Review (계획 검증)

- **스펙 커버리지**: §2 레이아웃=Task3(html) / §3 아키텍처=Task2(서버)+Task4(push) / §4 구성요소=Task1~4 / §5.1 state스키마=Task4 Step6 / §5.2 차로매핑=Task4 Step4 / §5.3 현시매핑=Task4 Step4(PHASE_GREEN, 우회전 상시green) / §5.4 엣지프로토콜=Task1 / §6 지표=Task4 Step6(halting합·mean_wait·arrived누적) / §7 테스트=Task1·2 + Task5 수동 / §8 YAGNI=계획 전반(time_in_phase 생략 등) / §9 파일=Task1~5. 누락 없음.
- **플레이스홀더**: 없음(모든 코드 스텝에 실제 코드 포함). `time_in_phase=0` 은 의도된 결정(비고 명시), placeholder 아님.
- **타입 일관성**: `forward`(edge_server, Task1) → parity(Task1)·서버(Task1) 동일 사용. state dict 키(`lanes/metrics/decision/q/phase/phase_name`)가 Task4 push 와 Task3 html 렌더(`render()`)에서 일치. 검지기 ID `e2_<lane>`(Task4 cap_map·halting) = make_e2_detectors 규약. `EdgeClient.last_q`(Task4 Step2) → 사용(Task4 Step5). LANE_ORDER 12개 = html `i%3` 그룹 가정과 일치(방향당 3차로).
- **리스크**: 로컬 rl + --dashboard 시 관측 차원(29D god-view) 보존(Task4 Step3 elif 로 observation_class 미변경) → 모델-관측 차원 일치. 대시보드 표시는 traci 원시값이라 obs 모드와 독립.
