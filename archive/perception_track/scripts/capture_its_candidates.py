"""v2 사거리(4지, 3차로/방향) 후보 카메라 여러 개를 순차 캡처.

각 (이름, bbox) 쌍에 대해 ITS API로 fresh URL을 받아 즉시 한 프레임 저장.

사용:
    python scripts/capture_its_candidates.py --out-dir captures/
"""
from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rw_timeout;10000000|stimeout;10000000",
)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import cv2
import requests


ENDPOINT = "https://openapi.its.go.kr:9443/cctvInfo"

CANDIDATES = [
    # (slug,            cctv_name_substring,        bbox,                                   road_type)
    ("sabang",         "사방교차로",                 (129.20, 129.25, 35.92, 35.95),         "its"),
    ("yacheok",        "야척교차로",                 (129.14, 129.18, 35.84, 35.88),         "its"),
    ("hyeongok",       "현곡교차로",                 (129.15, 129.19, 35.87, 35.91),         "its"),
    ("moryang",        "모량교차로",                 (129.12, 129.16, 35.80, 35.84),         "its"),
    ("daegok",         "대곡교차로",                 (129.10, 129.14, 35.84, 35.88),         "its"),
    ("hyohyeon",       "효현교차로",                 (129.12, 129.16, 35.80, 35.83),         "its"),
]


def fetch_url(api_key: str, name_sub: str, bbox, road_type: str) -> str | None:
    minX, maxX, minY, maxY = bbox
    params = {
        "apiKey": api_key, "type": road_type, "cctvType": 2,
        "minX": minX, "maxX": maxX, "minY": minY, "maxY": maxY,
        "getType": "json",
    }
    r = requests.get(ENDPOINT, params=params, timeout=10)
    r.raise_for_status()
    items = r.json().get("response", {}).get("data", [])
    for it in items:
        if name_sub in it.get("cctvname", ""):
            return it.get("cctvurl", "")
    return None


def capture(url: str, out_path: str, frame_index: int = 30) -> tuple[bool, tuple[int, int] | None]:
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        return False, None
    frame = None
    for _ in range(frame_index + 1):
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            return False, None
    cap.release()
    cv2.imwrite(out_path, frame)
    h, w = frame.shape[:2]
    return True, (w, h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-key", default=os.environ.get("ITS_API_KEY"))
    ap.add_argument("--out-dir", default="captures")
    args = ap.parse_args()

    if not args.api_key:
        sys.exit("[ERROR] ITS_API_KEY 필요")

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"{'slug':<10}  {'name':<10}  {'url ok':<7}  {'capture':<7}  size")
    print("-" * 60)
    for slug, name_sub, bbox, rt in CANDIDATES:
        url = fetch_url(args.api_key, name_sub, bbox, rt)
        if not url:
            print(f"{slug:<10}  {name_sub:<10}  NO       -        -")
            continue
        out_path = os.path.join(args.out_dir, f"{slug}.png")
        ok, size = capture(url, out_path)
        size_str = f"{size[0]}x{size[1]}" if size else "-"
        print(f"{slug:<10}  {name_sub:<10}  YES      {'OK' if ok else 'FAIL':<7}  {size_str}")
        # avoid hammering — minor pause
        time.sleep(1.0)


if __name__ == "__main__":
    main()
