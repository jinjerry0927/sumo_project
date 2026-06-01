"""전국 ITS 카메라에서 '사거리' 이름 가진 4지 후보 sweep.

여러 도시 bbox를 순회하며 cctvname에 "사거리" 포함된 ITS HLS 카메라 모두 수집.
모화/야척처럼 ITS에 없는 경우/방향 한쪽만 보이는 경우를 피하고,
v2 모델 구조(4지×3차로/방향)에 맞는 후보를 찾기 위함.

사용:
    python scripts/sweep_sageori.py
"""
import argparse
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import requests


ENDPOINT = "https://openapi.its.go.kr:9443/cctvInfo"

CITIES = [
    # (name,     minX,    maxX,    minY,   maxY)
    ("서울",     126.80,  127.20,  37.45,  37.70),
    ("인천",     126.60,  126.85,  37.40,  37.55),
    ("수원",     126.95,  127.10,  37.23,  37.33),
    ("성남",     127.05,  127.20,  37.38,  37.50),
    ("부산",     128.95,  129.25,  35.05,  35.30),
    ("대구",     128.50,  128.75,  35.78,  35.95),
    ("대전",     127.30,  127.50,  36.28,  36.43),
    ("광주",     126.80,  127.00,  35.10,  35.25),
    ("울산",     129.20,  129.40,  35.50,  35.65),
    ("창원",     128.55,  128.75,  35.18,  35.30),
    ("천안",     127.10,  127.25,  36.78,  36.90),
    ("청주",     127.42,  127.55,  36.60,  36.72),
    ("전주",     127.05,  127.20,  35.78,  35.90),
    ("포항",     129.30,  129.42,  36.00,  36.10),
    ("경주광역", 129.10,  129.40,  35.70,  36.00),
]

KEYWORDS = ("사거리", "네거리")


def query(api_key: str, bbox, road_type: str):
    minX, maxX, minY, maxY = bbox
    params = {
        "apiKey": api_key, "type": road_type, "cctvType": 2,
        "minX": minX, "maxX": maxX, "minY": minY, "maxY": maxY,
        "getType": "json",
    }
    try:
        r = requests.get(ENDPOINT, params=params, timeout=15)
        r.raise_for_status()
        data = r.json().get("response", {}).get("data", []) or []
        # API returns dict if 1 item, list if many — normalize
        if isinstance(data, dict):
            data = [data]
        return [it for it in data if isinstance(it, dict)]
    except Exception as e:
        print(f"  [err] {e}", file=sys.stderr)
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-key", default=os.environ.get("ITS_API_KEY"))
    args = ap.parse_args()
    if not args.api_key:
        sys.exit("ITS_API_KEY required")

    all_hits = []
    for city, minX, maxX, minY, maxY in CITIES:
        bbox = (minX, maxX, minY, maxY)
        total_cams = 0
        sageori = []
        for rt in ("its", "ex"):
            items = query(args.api_key, bbox, rt)
            total_cams += len(items)
            for it in items:
                name = it.get("cctvname", "")
                if any(k in name for k in KEYWORDS):
                    sageori.append({
                        "city": city, "name": name,
                        "lon": float(it.get("coordx", 0)),
                        "lat": float(it.get("coordy", 0)),
                        "url": it.get("cctvurl", ""),
                        "road": rt,
                    })
        print(f"[{city:<10}] 전체 {total_cams:>3}개, 사거리/네거리 {len(sageori):>2}개")
        all_hits.extend(sageori)

    print(f"\n{'='*70}\n전국 사거리/네거리 ITS 카메라: {len(all_hits)}개\n{'='*70}")
    print(f"{'#':>3}  {'city':<8}  {'name':<32}  {'lon':>9}  {'lat':>9}  {'rt':<3}")
    print("-" * 90)
    for i, h in enumerate(all_hits):
        print(f"{i:>3}  {h['city']:<8}  {h['name'][:32]:<32}  "
              f"{h['lon']:>9.4f}  {h['lat']:>9.4f}  {h['road']:<3}")


if __name__ == "__main__":
    main()
