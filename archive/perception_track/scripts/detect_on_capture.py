"""캡처 PNG 1장에 YOLO 모델 돌려 차량 박스 시각화·저장.

모델 분기점(8차 발표 실패 지점) 검증용. 모델 교체하며 동일 프레임에 비교.

사용:
    python scripts/detect_on_capture.py --img captures/yacheok.png --model yolov8s.pt
    python scripts/detect_on_capture.py --img captures/yacheok.png --model yolov8m.pt
"""
from __future__ import annotations

import argparse
import os
import sys

# allow running from any cwd: prepend project root
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import cv2

from perception.detector import VehicleDetector, CLS_NAME


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True)
    ap.add_argument("--model", default="yolov8s.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--min-w", type=int, default=12)
    ap.add_argument("--min-h", type=int, default=8)
    ap.add_argument("--out", default=None,
                    help="시각화 PNG 출력 경로 (기본: <img>_<model>.png)")
    args = ap.parse_args()

    frame = cv2.imread(args.img)
    if frame is None:
        sys.exit(f"이미지 로드 실패: {args.img}")

    det = VehicleDetector(
        model_path=args.model, conf=args.conf,
        min_w=args.min_w, min_h=args.min_h, imgsz=args.imgsz,
    )
    boxes = det.detect(frame)

    print(f"\n[{args.model}] {len(boxes)}대 검출 (conf>={args.conf}, imgsz={args.imgsz}, min_wh={args.min_w}x{args.min_h})")
    by_cls = {}
    for b in boxes:
        by_cls[b.cls] = by_cls.get(b.cls, 0) + 1
    for cls, n in sorted(by_cls.items()):
        print(f"  {CLS_NAME.get(cls, cls):<6} {n}")

    # 시각화
    vis = frame.copy()
    for b in boxes:
        color = (0, 255, 0) if b.cls == 2 else (0, 165, 255)  # car=green, bus/truck=orange
        cv2.rectangle(vis, (b.x1, b.y1), (b.x2, b.y2), color, 2)
        label = f"{CLS_NAME.get(b.cls, '?')} {b.conf:.2f}"
        cv2.putText(vis, label, (b.x1, max(b.y1 - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    out_path = args.out or (
        os.path.splitext(args.img)[0]
        + f"_{os.path.splitext(os.path.basename(args.model))[0]}.png"
    )
    cv2.imwrite(out_path, vis)
    print(f"\n시각화 저장: {out_path}")


if __name__ == "__main__":
    main()
