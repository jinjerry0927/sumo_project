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
