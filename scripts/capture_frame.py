"""스트림/파일에서 한 프레임을 떠서 PNG로 저장.

ROI 폴리곤을 정의하려면 우선 카메라의 정지 이미지가 필요하다.
이 스크립트로 떠둔 PNG를 그림판/GIMP에서 열어 4방향 영역 좌표를 확인한 뒤
perception/roi_config/<id>.json 에 기록.

사용:
    python scripts/capture_frame.py --url intersection.mp4 --out frame.png
    python scripts/capture_frame.py --url "http://.../playlist.m3u8" --out frame.png
"""
import argparse
import os
import sys

# FFmpeg 타임아웃 사전 설정
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rw_timeout;10000000|stimeout;10000000",
)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="파일 경로 또는 스트림 URL")
    ap.add_argument("--out", default="frame.png")
    ap.add_argument("--frame-index", type=int, default=30,
                    help="N번째 프레임 (스트림은 시작 직후 키프레임 안정화 위해 30 권장)")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        sys.exit(f"open 실패: {args.url}")

    frame = None
    for i in range(args.frame_index + 1):
        ok, frame = cap.read()
        if not ok or frame is None:
            sys.exit(f"read 실패 (frame {i})")

    cap.release()
    cv2.imwrite(args.out, frame)
    h, w = frame.shape[:2]
    print(f"saved: {args.out}  ({w}x{h})")


if __name__ == "__main__":
    main()
