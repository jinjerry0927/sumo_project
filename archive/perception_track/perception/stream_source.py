"""스트림 입력 추상화.

세 가지 입력 소스를 동일 인터페이스로 다룬다:
    - 로컬 mp4 파일                       → StreamSource(path)
    - HLS/RTSP 등 직접 URL                → StreamSource(url)
    - ITS 국가센터 OpenAPI (30초 만료)    → ItsApiStreamSource(api_key, coord_box)

공통 동작:
    src = StreamSource(url)
    src.open()
    while True:
        ok, frame = src.read()        # ok=False면 내부적으로 자동 재연결 시도
        if frame is None: break
    src.release()
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

# cv2 import 전에 FFmpeg 타임아웃 옵션 적용 (HLS open/read hang 방지)
_DEFAULT_TIMEOUT_US = "10000000"  # 10s
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    f"rw_timeout;{_DEFAULT_TIMEOUT_US}|stimeout;{_DEFAULT_TIMEOUT_US}",
)

import cv2
import numpy as np


# ── 기본 스트림 소스 ─────────────────────────────────────────────
@dataclass
class StreamSource:
    """파일/HLS/RTSP 공용 입력 래퍼.

    - reconnect=True 이면 read 실패 시 reopen_backoff_s 후 재오픈 시도
    - 파일 입력의 경우 EOF에 도달하면 ok=False, frame=None 반환 (재연결 X)
    """
    url: str
    reconnect: bool = True
    reopen_backoff_s: float = 2.0
    max_consecutive_failures: int = 30   # 이 횟수 연속 실패 시 read()가 영구 None

    _cap: Optional[cv2.VideoCapture] = field(default=None, init=False, repr=False)
    _is_file: bool = field(default=False, init=False, repr=False)
    _fail_streak: int = field(default=0, init=False, repr=False)

    def open(self) -> bool:
        self._is_file = self._looks_like_file(self.url)
        self._cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        ok = self._cap.isOpened()
        if not ok:
            print(f"[StreamSource] open 실패: {self.url}", file=sys.stderr)
        return ok

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._cap is None and not self.open():
            return False, None
        assert self._cap is not None

        ok, frame = self._cap.read()
        if ok and frame is not None:
            self._fail_streak = 0
            return True, frame

        # 파일이면 EOF — 재연결하지 않음
        if self._is_file:
            return False, None

        # 스트림이면 재연결 시도
        self._fail_streak += 1
        if not self.reconnect or self._fail_streak > self.max_consecutive_failures:
            return False, None

        print(f"[StreamSource] read 실패 (#{self._fail_streak}) — {self.reopen_backoff_s}s 후 재연결",
              file=sys.stderr)
        time.sleep(self.reopen_backoff_s)
        self._refresh_url()
        self._cap.release()
        self._cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        return False, None  # 호출 측이 다음 루프에서 다시 read

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def fps(self) -> float:
        if self._cap is None:
            return 0.0
        f = self._cap.get(cv2.CAP_PROP_FPS) or 0.0
        return f if 0 < f < 200 else 0.0

    def frame_size(self) -> Tuple[int, int]:
        if self._cap is None:
            return (0, 0)
        return (int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # 서브클래스 후크: 만료된 토큰 URL 갱신용
    def _refresh_url(self) -> None:
        pass

    @staticmethod
    def _looks_like_file(s: str) -> bool:
        s = s.lower()
        if s.startswith(("http://", "https://", "rtsp://", "rtmp://")):
            return False
        return True


# ── ITS 국가센터 OpenAPI 소스 ────────────────────────────────────
@dataclass
class ItsApiStreamSource(StreamSource):
    """ITS 국가센터 OpenAPI(`https://openapi.its.go.kr:9443/cctvInfo`) 기반 HLS 소스.

    영상 URL의 토큰이 30초 후 만료되므로 주기적으로 재요청.

    필요 인자 (서브클래스 전용):
        api_key      : data.go.kr에서 발급받은 인증키
        bbox         : (minX, maxX, minY, maxY) 경위도. 좁게 잡을수록 카메라 1개로 좁혀짐
        cctv_name    : 응답 중 cctvname에 이 문자열이 포함된 첫 번째 카메라 선택
        road_type    : "ex"(고속) | "its"(국도/도시) — 경주 시내면 "its"
        refresh_s    : 토큰 갱신 주기 (기본 25초, 만료 30초보다 약간 짧게)
    """
    api_key: str = ""
    bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    cctv_name: str = ""
    road_type: str = "its"
    refresh_s: float = 25.0

    # 부모의 url은 처음엔 빈 값으로 두고 첫 _refresh_url에서 채움
    url: str = field(default="", init=False)
    _last_refresh_ts: float = field(default=0.0, init=False, repr=False)

    ENDPOINT = "https://openapi.its.go.kr:9443/cctvInfo"

    def open(self) -> bool:
        if not self.api_key:
            print("[ItsApiStreamSource] api_key 필요 (data.go.kr 발급)", file=sys.stderr)
            return False
        self._refresh_url()
        if not self.url:
            return False
        return super().open()

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        # 토큰 만료 임박 시 선제적으로 URL 갱신 + 재오픈
        if time.time() - self._last_refresh_ts > self.refresh_s:
            self._refresh_url()
            if self._cap is not None:
                self._cap.release()
                self._cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        return super().read()

    def _refresh_url(self) -> None:
        import requests  # 지연 import — requirements에 있지만 미사용자 환경 보호

        minX, maxX, minY, maxY = self.bbox
        params = {
            "apiKey":   self.api_key,
            "type":     self.road_type,
            "cctvType": 2,            # 2 = 동영상
            "minX": minX, "maxX": maxX, "minY": minY, "maxY": maxY,
            "getType": "json",
        }
        try:
            r = requests.get(self.ENDPOINT, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"[ItsApiStreamSource] API 호출 실패: {e}", file=sys.stderr)
            return

        items = (data.get("response", {})
                     .get("data", []))
        if not items:
            print("[ItsApiStreamSource] bbox 안에 카메라 0개", file=sys.stderr)
            return

        # cctv_name 부분일치 우선, 없으면 첫 번째
        target = None
        if self.cctv_name:
            for it in items:
                if self.cctv_name in it.get("cctvname", ""):
                    target = it
                    break
        if target is None:
            target = items[0]

        self.url = target.get("cctvurl", "")
        self._last_refresh_ts = time.time()
        print(f"[ItsApiStreamSource] URL 갱신 → {target.get('cctvname','?')}",
              file=sys.stderr)


# ── YouTube 라이브 스트림 소스 (안전망) ─────────────────────────
@dataclass
class YoutubeLiveSource(StreamSource):
    """YouTube 라이브 영상 URL을 yt-dlp로 m3u8 해석해 OpenCV로 열기.

    한국 교통 라이브 카메라(공식 채널 등) 데모용 fallback. 졸업 발표 안전망.

    필요:
        pip install yt-dlp

    사용:
        src = YoutubeLiveSource(youtube_url="https://www.youtube.com/watch?v=XXXX")
        src.open(); ...
    """
    youtube_url: str = ""
    refresh_s: float = 1800.0   # 30분마다 m3u8 토큰 갱신 시도

    url: str = field(default="", init=False)
    _last_refresh_ts: float = field(default=0.0, init=False, repr=False)

    def open(self) -> bool:
        if not self.youtube_url:
            print("[YoutubeLiveSource] youtube_url 필요", file=sys.stderr)
            return False
        self._refresh_url()
        if not self.url:
            return False
        return super().open()

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if time.time() - self._last_refresh_ts > self.refresh_s:
            self._refresh_url()
            if self._cap is not None and self.url:
                self._cap.release()
                self._cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        return super().read()

    def _refresh_url(self) -> None:
        try:
            import yt_dlp  # 지연 import
        except ImportError:
            print("[YoutubeLiveSource] yt-dlp 미설치: pip install yt-dlp", file=sys.stderr)
            return

        opts = {"quiet": True, "format": "best[protocol*=m3u8]/best",
                "noplaylist": True, "skip_download": True}
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(self.youtube_url, download=False)
        except Exception as e:
            print(f"[YoutubeLiveSource] yt-dlp 실패: {e}", file=sys.stderr)
            return

        stream_url = info.get("url", "") if info else ""
        if not stream_url:
            print("[YoutubeLiveSource] 스트림 URL 추출 실패", file=sys.stderr)
            return
        self.url = stream_url
        self._last_refresh_ts = time.time()
        print(f"[YoutubeLiveSource] URL 갱신 (제목: {info.get('title','?')[:40]})",
              file=sys.stderr)
