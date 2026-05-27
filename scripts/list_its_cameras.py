"""ITS 국가센터 OpenAPI에서 bbox 범위 안의 CCTV 목록을 출력.

먼저 인증키를 받자(즉시 발급):
    https://www.data.go.kr → "국가교통정보센터 CCTV 정보" 검색 → 활용신청

사용:
    $env:ITS_API_KEY = "발급키"
    python scripts/list_its_cameras.py --bbox 126.95 127.05 37.55 37.60 --road-type its

출력: 카메라 이름·좌표·스트림 URL을 표로. 교차로형 카메라(이름에 "사거리/네거리/교차로/IC" 등) 찾기 좋게.
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
INTERSECTION_KEYWORDS = ("사거리", "네거리", "교차로", "삼거리", "오거리", "로터리", "IC", "JC")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-key", default=os.environ.get("ITS_API_KEY"),
                    help="data.go.kr 인증키 (또는 ITS_API_KEY 환경변수)")
    ap.add_argument("--bbox", type=float, nargs=4, required=True,
                    metavar=("MINX", "MAXX", "MINY", "MAXY"),
                    help="경위도 범위. 좁게 잡을수록 카메라 수 적어짐")
    ap.add_argument("--road-type", default="its", choices=["its", "ex"],
                    help="its=국도/도시, ex=고속도로")
    ap.add_argument("--only-intersection", action="store_true",
                    help="이름에 사거리/네거리/교차로 등이 포함된 것만 표시")
    ap.add_argument("--limit", type=int, default=50)
    args = ap.parse_args()

    if not args.api_key:
        sys.exit("[ERROR] --api-key 또는 환경변수 ITS_API_KEY 필요")

    minX, maxX, minY, maxY = args.bbox
    params = {
        "apiKey":   args.api_key,
        "type":     args.road_type,
        "cctvType": 2,             # 2 = 동영상 HLS
        "minX": minX, "maxX": maxX, "minY": minY, "maxY": maxY,
        "getType": "json",
    }
    print(f"[GET] {ENDPOINT}")
    print(f"      bbox=({minX},{minY})~({maxX},{maxY})  road_type={args.road_type}")

    r = requests.get(ENDPOINT, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    items = data.get("response", {}).get("data", [])

    if args.only_intersection:
        items = [it for it in items
                 if any(k in it.get("cctvname", "") for k in INTERSECTION_KEYWORDS)]

    print(f"\n총 {len(items)}개 카메라\n")
    print(f"{'#':>3}  {'이름':<26}  {'lon':>9}  {'lat':>9}  url")
    print("-" * 110)
    for i, it in enumerate(items[:args.limit]):
        name = it.get("cctvname", "?")[:26]
        lon = float(it.get("coordx", 0))
        lat = float(it.get("coordy", 0))
        url = (it.get("cctvurl", "") or "")[:60]
        print(f"{i:>3}  {name:<26}  {lon:>9.4f}  {lat:>9.4f}  {url}")

    if len(items) > args.limit:
        print(f"\n... 외 {len(items) - args.limit}개 생략. --limit로 늘리세요.")

    if items:
        first = items[0]
        print("\n[예시] 첫 카메라로 바로 실행:")
        print(f'  python -m perception.run_realtime --source its-api `')
        print(f'      --bbox {minX} {maxX} {minY} {maxY} `')
        print(f'      --cctv-name "{first.get("cctvname","")}" --road-type {args.road_type} --display')


if __name__ == "__main__":
    main()
