"""차량 탐지기.

기본 모델 = RT-DETR-L. 2026-05-27 야척교차로 야간 캡처에서 YOLOv8s/m/l 모두 0대 검출
실패, RT-DETR-L만 검출 성공해서 escalation 확정. ultralytics가 YOLO/RT-DETR 동일
인터페이스 제공해 model_path만 바꿔 교체.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from ultralytics import YOLO


# COCO 클래스: 2=car, 5=bus, 7=truck
VEHICLE_CLS = [2, 5, 7]
CLS_NAME = {2: "car", 5: "bus", 7: "truck"}


@dataclass
class Box:
    x1: int
    y1: int
    x2: int
    y2: int
    cls: int
    conf: float
    track_id: Optional[int] = None

    @property
    def center(self) -> tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    @property
    def w(self) -> int:
        return self.x2 - self.x1

    @property
    def h(self) -> int:
        return self.y2 - self.y1


# ── 차량 탐지기 ──────────────────────────────────────────────────
class VehicleDetector:
    def __init__(
        self,
        model_path: str = "rtdetr-l.pt",
        conf: float = 0.25,
        min_w: int = 30,
        min_h: int = 20,
        max_aspect: float = 3.0,
        imgsz: int = 640,
    ):
        self.model = YOLO(model_path)
        self.conf = conf
        self.min_w = min_w
        self.min_h = min_h
        self.max_aspect = max_aspect
        self.imgsz = imgsz   # YOLO 입력 해상도. 720x480 같은 저해상도 영상에서 작은 객체 탐지에 유효 (1280 권장)

    def detect(self, frame: np.ndarray) -> List[Box]:
        """프레임 1장에서 필터 통과한 차량 박스 리스트 반환."""
        results = self.model.track(
            frame, conf=self.conf, classes=VEHICLE_CLS,
            persist=True, verbose=False, imgsz=self.imgsz,
        )
        r0 = results[0]
        if r0.boxes is None or len(r0.boxes) == 0:
            return []

        xyxy = r0.boxes.xyxy.cpu().numpy()
        confs = r0.boxes.conf.cpu().numpy()
        clses = r0.boxes.cls.cpu().numpy().astype(int)
        ids = (r0.boxes.id.cpu().numpy().astype(int).tolist()
               if r0.boxes.id is not None else [None] * len(xyxy))

        out: List[Box] = []
        for (x1, y1, x2, y2), c, cls, tid in zip(xyxy, confs, clses, ids):
            x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
            if not self._is_valid(x1i, y1i, x2i, y2i):
                continue
            out.append(Box(x1i, y1i, x2i, y2i, int(cls), float(c), tid))
        return out

    def _is_valid(self, x1: int, y1: int, x2: int, y2: int) -> bool:
        w, h = x2 - x1, y2 - y1
        if w < self.min_w or h < self.min_h:
            return False
        if h / max(w, 1) > self.max_aspect:
            return False
        return True
