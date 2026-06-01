"""경주시 ITS HLS 스트림 스모크 테스트.

각 카메라 URL에 대해 cv2.VideoCapture가 열리는지, 첫 프레임을 받을 수 있는지 확인.
학교망/집망 양쪽에서 한 번씩 돌려 네트워크 차단 여부를 사전 확인용.

사용:
    python scripts/check_streams.py
    python scripts/check_streams.py --show   # 첫 프레임을 창으로 띄움
    python scripts/check_streams.py --timeout 10  # open/read 타임아웃 (초)
"""
import argparse
import os
import sys
import time

# FFmpeg open/read 타임아웃 (us). cv2 import 전에 설정해야 적용됨.
_DEFAULT_TIMEOUT_US = "10000000"  # 10초
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS",
                      f"rw_timeout;{_DEFAULT_TIMEOUT_US}|stimeout;{_DEFAULT_TIMEOUT_US}")

# Windows cp949 콘솔에서도 깨지지 않도록 stdout을 UTF-8로
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import cv2

# ── 검증된 경주시 ITS 카메라 (조사 단계에서 확인) ──────────────
STREAM_BASE = "http://221.157.65.155:1935/live/live{n}.stream/playlist.m3u8"
CAMERAS = {
    3:  "터미널네거리",
    15: "선덕네거리",
    23: "코오롱삼거리",
}


def check_one(n: int, name: str, show: bool) -> bool:
    url = STREAM_BASE.format(n=n)
    print(f"\n[CAM {n:>2}] {name}")
    print(f"        URL: {url}")

    t0 = time.time()
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    open_ms = (time.time() - t0) * 1000

    if not cap.isOpened():
        print(f"        [NG] VideoCapture 실패 (open 시도 {open_ms:.0f}ms)")
        return False

    t1 = time.time()
    ok, frame = cap.read()
    read_ms = (time.time() - t1) * 1000

    if not ok or frame is None:
        print(f"        [NG] read 실패 (open {open_ms:.0f}ms / read {read_ms:.0f}ms)")
        cap.release()
        return False

    h, w = frame.shape[:2]
    print(f"        [OK] OK  {w}x{h}  (open {open_ms:.0f}ms / read {read_ms:.0f}ms)")

    if show:
        window = f"CAM {n} - {name}"
        cv2.imshow(window, cv2.resize(frame, (960, 540)))
        cv2.waitKey(1500)
        cv2.destroyWindow(window)

    cap.release()
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true", help="첫 프레임을 1.5초간 창에 표시")
    ap.add_argument("--timeout", type=int, default=10, help="open/read 타임아웃 초")
    args = ap.parse_args()

    # 인자로 받은 타임아웃으로 재설정
    us = args.timeout * 1_000_000
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"rw_timeout;{us}|stimeout;{us}"

    print("경주시 ITS HLS 스트림 스모크 테스트")
    print("=" * 50)
    results = {n: check_one(n, name, args.show) for n, name in CAMERAS.items()}

    print("\n" + "=" * 50)
    ok_count = sum(results.values())
    print(f"결과: {ok_count}/{len(results)} 정상")
    for n, ok in results.items():
        mark = "[OK]" if ok else "[NG]"
        print(f"  {mark} CAM {n:>2}  {CAMERAS[n]}")

    if ok_count == 0:
        print("\n[WARN] 한 개도 열리지 않음. 네트워크/방화벽/IP 변경 확인 필요.")
        print("  대안: ITS 국가센터 OpenAPI (perception/run_realtime.py --source its-api)")


if __name__ == "__main__":
    main()
