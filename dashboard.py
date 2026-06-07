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
