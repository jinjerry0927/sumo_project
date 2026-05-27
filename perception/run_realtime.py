"""실시간 CCTV → YOLO → 4방향 카운트 CLI.

사용 예:
    # 로컬 mp4 (개발/회귀 테스트용)
    python -m perception.run_realtime --source file --url intersection.mp4 --display

    # 직접 HLS URL
    python -m perception.run_realtime --source url --url "http://.../playlist.m3u8" --display

    # ITS 국가센터 OpenAPI (data.go.kr 인증키 필요)
    python -m perception.run_realtime --source its-api \
        --api-key $env:ITS_API_KEY --bbox 129.20 129.25 35.83 35.87 \
        --cctv-name "금장교" --display --save-jsonl logs/run.jsonl

    # ROI 폴리곤 적용 (방향별 카운트)
    python -m perception.run_realtime --source file --url intersection.mp4 \
        --roi perception/roi_config/intersection_mp4.json --display
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# stdout을 UTF-8로 (Windows cp949 콘솔에서 한글/이모지 깨짐 방지)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import cv2

from .detector import VehicleDetector
from .lane_aggregator import LaneAggregator
from .stream_source import StreamSource, ItsApiStreamSource, YoutubeLiveSource


def build_source(args) -> StreamSource:
    if args.source in ("file", "url"):
        if not args.url:
            sys.exit("--url 필수 (file/url 모드)")
        return StreamSource(url=args.url, reconnect=(args.source == "url"))

    if args.source == "its-api":
        if not args.api_key:
            sys.exit("--api-key 필수 (또는 ITS_API_KEY 환경변수)")
        if not args.bbox or len(args.bbox) != 4:
            sys.exit("--bbox minX maxX minY maxY 필요")
        return ItsApiStreamSource(
            api_key=args.api_key,
            bbox=tuple(args.bbox),
            cctv_name=args.cctv_name or "",
            road_type=args.road_type,
        )

    if args.source == "youtube":
        if not args.url:
            sys.exit("--url 필수 (youtube 라이브 URL)")
        return YoutubeLiveSource(youtube_url=args.url)

    sys.exit(f"알 수 없는 source: {args.source}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["file", "url", "its-api", "youtube"], default="file")
    ap.add_argument("--url", default=None, help="file/url 모드의 입력 경로 또는 URL")
    ap.add_argument("--api-key", default=os.environ.get("ITS_API_KEY"),
                    help="data.go.kr ITS 인증키 (its-api 모드)")
    ap.add_argument("--bbox", type=float, nargs=4, metavar=("MINX", "MAXX", "MINY", "MAXY"),
                    help="its-api 검색 경위도 범위")
    ap.add_argument("--cctv-name", default=None, help="응답 cctvname에 포함될 부분문자열")
    ap.add_argument("--road-type", default="its", choices=["its", "ex"])
    ap.add_argument("--model", default="yolov8s.pt", help="YOLO 가중치 (자동 다운로드)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640,
                    help="YOLO 입력 해상도. 저해상도 영상(720x480 등)에서 작은 객체 탐지 시 1280 권장")
    ap.add_argument("--min-w", type=int, default=30, help="박스 최소 너비(px). 저해상도면 10~15")
    ap.add_argument("--min-h", type=int, default=20, help="박스 최소 높이(px). 저해상도면 8~12")
    ap.add_argument("--max-aspect", type=float, default=3.0, help="h/w 비율 상한 (사람/기둥 컷)")
    ap.add_argument("--roi", default=None, help="ROI JSON 경로 (방향별 카운트 활성화)")
    ap.add_argument("--display", action="store_true", help="화면 창에 결과 표시")
    ap.add_argument("--save-jsonl", default=None, help="카운트 로그 JSONL 저장 경로")
    ap.add_argument("--print-every", type=int, default=30, help="콘솔 카운트 출력 주기 (프레임)")
    ap.add_argument("--frame-skip", type=int, default=1,
                    help="N프레임에 1번만 YOLO 추론. 화면은 매 프레임 그려져 부드럽게 보임 (이전 박스 재사용). 30fps 영상이면 3 권장")
    ap.add_argument("--duration-sec", type=float, default=0,
                    help="N초 후 자동 종료 (0=무한 루프, 데모 5분이면 300)")
    args = ap.parse_args()

    # ── 구성 ─────────────────────────────────────────────────────
    src = build_source(args)
    if not src.open():
        sys.exit("스트림 open 실패")

    detector = VehicleDetector(
        model_path=args.model, conf=args.conf, imgsz=args.imgsz,
        min_w=args.min_w, min_h=args.min_h, max_aspect=args.max_aspect,
    )
    aggregator = (LaneAggregator.from_json(args.roi) if args.roi else None)

    jsonl_fp = None
    if args.save_jsonl:
        Path(args.save_jsonl).parent.mkdir(parents=True, exist_ok=True)
        jsonl_fp = open(args.save_jsonl, "a", encoding="utf-8")

    w, h = src.frame_size()
    fps = src.fps() or 30.0
    print(f"[run_realtime] source={args.source}  size={w}x{h}  fps={fps:.1f}", file=sys.stderr)
    print(f"[run_realtime] model={args.model}  imgsz={args.imgsz}  conf={args.conf}  "
          f"min_box={args.min_w}x{args.min_h}  roi={'on' if aggregator else 'off'}", file=sys.stderr)

    # ── 메인 루프 ────────────────────────────────────────────────
    idx = 0
    fps_t0 = time.time()
    fps_frames = 0
    boxes: list = []
    counts: dict = {"total": 0}
    skip = max(1, args.frame_skip)
    start_ts = time.time()
    try:
        while True:
            if args.duration_sec > 0 and (time.time() - start_ts) >= args.duration_sec:
                print(f"[run_realtime] --duration-sec {args.duration_sec} 도달, 종료", file=sys.stderr)
                break

            ok, frame = src.read()
            if not ok or frame is None:
                # 파일이면 EOF로 종료, 스트림이면 None은 재연결 사이클이므로 잠깐 쉬고 재시도
                if isinstance(src, StreamSource) and getattr(src, "_is_file", False):
                    print("[run_realtime] EOF", file=sys.stderr)
                    break
                time.sleep(0.1)
                continue

            # frame_skip마다만 YOLO 추론. 나머지 프레임은 이전 박스 재사용 → 화면 부드럽게
            did_detect = (idx % skip == 0)
            if did_detect:
                boxes = detector.detect(frame)
                counts = (aggregator.count(boxes, frame.shape[1::-1])
                          if aggregator else {"total": len(boxes)})

            # ── 시각화 ──
            if args.display:
                annotated = frame.copy()
                for b in boxes:
                    cv2.rectangle(annotated, (b.x1, b.y1), (b.x2, b.y2), (0, 255, 100), 2)
                    label = f"{b.conf:.2f}"
                    if b.track_id is not None:
                        label = f"#{b.track_id} {label}"
                    cv2.putText(annotated, label, (b.x1, max(b.y1 - 6, 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 100), 1)
                if aggregator:
                    aggregator.draw_overlay(annotated, counts)
                else:
                    cv2.putText(annotated, f"Vehicles: {len(boxes)}",
                                (18, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 100), 2)

                disp_h = 720
                disp_w = int(annotated.shape[1] * disp_h / max(annotated.shape[0], 1))
                cv2.imshow("perception", cv2.resize(annotated, (disp_w, disp_h)))
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # 's' 키 — 원본 해상도 프레임(박스 없음) + 어노테이션 둘 다 저장
                    Path("screenshots").mkdir(exist_ok=True)
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    raw_path = f"screenshots/raw_{ts}.png"
                    ann_path = f"screenshots/ann_{ts}.png"
                    cv2.imwrite(raw_path, frame)
                    cv2.imwrite(ann_path, annotated)
                    print(f"[saved] {raw_path}  {ann_path}", file=sys.stderr)

            # ── 로그 (detection이 실제 일어난 프레임만 기록) ──
            if jsonl_fp is not None and did_detect:
                record = {"ts": time.time(), "frame": idx, "total": len(boxes)}
                if aggregator:
                    record.update({d: counts.get(d, 0) for d in ("N", "S", "E", "W")})
                jsonl_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
                jsonl_fp.flush()

            if did_detect and (idx // skip) % args.print_every == 0:
                fps_frames += args.print_every
                elapsed = time.time() - fps_t0
                eff_fps = fps_frames / max(elapsed, 1e-6)
                msg = f"[f{idx:05d}] total={len(boxes)}  eff_fps={eff_fps:.1f}"
                if aggregator:
                    msg += "  " + " ".join(f"{d}={counts.get(d,0)}" for d in ("N","S","E","W"))
                print(msg, file=sys.stderr)

            idx += 1
    finally:
        src.release()
        if jsonl_fp is not None:
            jsonl_fp.close()
        if args.display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
