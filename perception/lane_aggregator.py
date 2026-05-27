"""4방향 ROI 폴리곤 기반 차로별 카운트.

ROI JSON 형식:
{
  "image_size": [1920, 1080],
  "polygons": {
    "N": [[x1,y1], [x2,y2], ...],
    "S": [...],
    "E": [...],
    "W": [...]
  }
}

좌표는 절대 픽셀(image_size 기준). 입력 프레임이 다른 해상도면 자동 스케일.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from .detector import Box

# 기본 4방향. 8방향(N_in/N_out/...) 등 임의 키도 JSON에 정의하면 그대로 사용됨.
DIRECTIONS = ("N", "S", "E", "W")


@dataclass
class LaneAggregator:
    polygons: Dict[str, np.ndarray]              # 방향 → (N,2) int32. 키는 임의 (N/S/E/W 또는 N_in/N_out 등)
    image_size: tuple[int, int] = (1920, 1080)   # 폴리곤 좌표 기준 해상도
    _scale_to: Optional[tuple[int, int]] = field(default=None, init=False, repr=False)
    _scaled: Optional[Dict[str, np.ndarray]] = field(default=None, init=False, repr=False)

    # ── 로딩 ─────────────────────────────────────────────────────
    @classmethod
    def from_json(cls, path: str | Path) -> "LaneAggregator":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        # JSON의 polygons 키 전체를 받아들임 (4방향이든 8방향이든)
        polys = {
            d: np.asarray(pts, dtype=np.int32)
            for d, pts in data["polygons"].items()
            if not d.startswith("_")   # _comment 같은 메타 키 무시
        }
        img_size = tuple(data.get("image_size", [1920, 1080]))
        return cls(polygons=polys, image_size=img_size)

    @property
    def directions(self) -> tuple[str, ...]:
        return tuple(self.polygons.keys())

    # ── 핵심 ─────────────────────────────────────────────────────
    def count(self, boxes: List[Box], frame_size: Optional[tuple[int, int]] = None) -> Dict[str, int]:
        """각 박스 중심점이 어느 방향 폴리곤에 속하는지 판정. 어디에도 안 속하면 무시(차로 밖)."""
        polys = self._polys_for(frame_size)
        counts = {d: 0 for d in polys.keys()}
        for b in boxes:
            cx, cy = b.center
            for d, poly in polys.items():
                if cv2.pointPolygonTest(poly, (float(cx), float(cy)), False) >= 0:
                    counts[d] += 1
                    break  # 폴리곤이 겹치면 첫 매치 우선
        return counts

    def snapshot(self, boxes: List[Box], frame_size: Optional[tuple[int, int]] = None) -> dict:
        """JSONL 로그 한 줄용 dict (ts + total + 방향별 카운트)."""
        c = self.count(boxes, frame_size)
        return {"ts": time.time(), "total": len(boxes), **c}

    # ── 시각화 ───────────────────────────────────────────────────
    def draw_overlay(self, frame: np.ndarray, counts: Dict[str, int]) -> None:
        """프레임에 폴리곤 윤곽과 방향 카운트를 직접 그림 (in-place)."""
        polys = self._polys_for(frame.shape[1::-1])  # (w, h)
        # 방향 prefix(N/S/E/W) 기반 색상. 진입(_in)은 진하게, 진출(_out)은 흐리게.
        base_colors = {
            "N": (255, 200, 100),
            "S": (100, 200, 255),
            "E": (200, 255, 100),
            "W": (255, 100, 200),
        }
        for d, poly in polys.items():
            prefix = d[0] if d and d[0] in base_colors else "N"
            color = base_colors.get(prefix, (200, 200, 200))
            # _out 폴리곤은 점선 느낌으로 thickness 1
            thickness = 1 if d.endswith("_out") else 2
            cv2.polylines(frame, [poly], isClosed=True, color=color, thickness=thickness)
            M = cv2.moments(poly)
            if M["m00"] > 0:
                tx, ty = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                cv2.putText(frame, f"{d}: {counts.get(d, 0)}",
                            (tx - 35, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # ── 내부: 해상도 스케일 캐시 ────────────────────────────────
    def _polys_for(self, frame_size: Optional[tuple[int, int]]) -> Dict[str, np.ndarray]:
        if frame_size is None or frame_size == self.image_size:
            return self.polygons
        if self._scale_to == frame_size and self._scaled is not None:
            return self._scaled

        sx = frame_size[0] / self.image_size[0]
        sy = frame_size[1] / self.image_size[1]
        scaled = {
            d: (poly * np.array([sx, sy], dtype=np.float32)).astype(np.int32)
            for d, poly in self.polygons.items()
        }
        self._scaled = scaled
        self._scale_to = frame_size
        return scaled
