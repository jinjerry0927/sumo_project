"""특정 이름/도시의 ITS 카메라 캡처 + 서울 카메라 전체 이름 덤프.

사용:
    python scripts/capture_named.py --out-dir captures/sageori/
"""
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
    # (slug,           name_sub,         bbox,                                   rt)
    ("busan_jungang", "중앙사거리",       (129.20, 129.23, 35.24, 35.27),         "its"),
    ("ulju_gaegok",   "개곡사거리",       (129.26, 129.29, 35.50, 35.53),         "its"),
]


def fetch_url(api_key, name_sub, bbox, rt):
    minX, maxX, minY, maxY = bbox
    params = {"apiKey": api_key, "type": rt, "cctvType": 2,
              "minX": minX, "maxX": maxX, "minY": minY, "maxY": maxY, "getType": "json"}
    r = requests.get(ENDPOINT, params=params, timeout=10)
    r.raise_for_status()
    data = r.json().get("response", {}).get("data", []) or []
    if isinstance(data, dict): data = [data]
    for it in data:
        if isinstance(it, dict) and name_sub in it.get("cctvname", ""):
            return it.get("cctvurl", ""), it.get("cctvname", "")
    return None, None


def capture(url, out_path, frame_index=30):
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened(): return False, None
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


def dump_city_names(api_key, city, bbox, rt="its"):
    minX, maxX, minY, maxY = bbox
    params = {"apiKey": api_key, "type": rt, "cctvType": 2,
              "minX": minX, "maxX": maxX, "minY": minY, "maxY": maxY, "getType": "json"}
    r = requests.get(ENDPOINT, params=params, timeout=15)
    data = r.json().get("response", {}).get("data", []) or []
    if isinstance(data, dict): data = [data]
    names = sorted(set(it.get("cctvname", "") for it in data if isinstance(it, dict)))
    print(f"\n[{city}] {len(names)}개 카메라 이름:")
    for n in names: print(f"  {n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-key", default=os.environ.get("ITS_API_KEY"))
    ap.add_argument("--out-dir", default="captures/sageori")
    ap.add_argument("--dump-seoul", action="store_true",
                    help="서울 ITS 카메라 전체 이름 출력 (사거리 후보 수동 검색용)")
    args = ap.parse_args()
    if not args.api_key: sys.exit("ITS_API_KEY required")

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"{'slug':<16}  {'matched name':<30}  {'cap':<5}  size")
    print("-" * 75)
    for slug, name_sub, bbox, rt in CANDIDATES:
        url, full_name = fetch_url(args.api_key, name_sub, bbox, rt)
        if not url:
            print(f"{slug:<16}  {'(URL not found)':<30}  -      -")
            continue
        out = os.path.join(args.out_dir, f"{slug}.png")
        ok, size = capture(url, out)
        size_str = f"{size[0]}x{size[1]}" if size else "-"
        print(f"{slug:<16}  {full_name[:30]:<30}  {'OK' if ok else 'FAIL':<5}  {size_str}")
        time.sleep(1)

    if args.dump_seoul:
        dump_city_names(args.api_key, "서울", (126.80, 127.20, 37.45, 37.70))


if __name__ == "__main__":
    main()
