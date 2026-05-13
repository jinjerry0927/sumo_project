import cv2
from ultralytics import YOLO

VIDEO_PATH  = r"C:\Users\James\Documents\GitHub\sumo_project\intersection.mp4"
MODEL_PATH  = "yolov8n.pt"
CONF        = 0.15
VEHICLE_CLS = [2, 5, 7]     # car, bus, truck

# 필터 기준
MIN_W      = 30
MIN_H      = 20
MAX_ASPECT = 3.0

# 교차로 ROI (y=300 이하만 탐지)
ROI_Y1 = 300


def is_valid(x1, y1, x2, y2):
    w, h = x2 - x1, y2 - y1
    cy = (y1 + y2) / 2
    if w < MIN_W or h < MIN_H:
        return False
    if h / max(w, 1) > MAX_ASPECT:
        return False
    if cy < ROI_Y1:
        return False
    return True


def main():
    model = YOLO(MODEL_PATH)
    cap   = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("ERROR: cannot open video")
        return

    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    delay = max(1, int(1000 / fps))   # 60fps → 16ms

    print(f"Resolution: {w}x{h}  FPS: {fps:.0f}  delay: {delay}ms")
    print("Press q to quit")

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results  = model.track(frame, conf=CONF, classes=VEHICLE_CLS,
                               persist=True, verbose=False)
        annotated = frame.copy()
        count     = 0

        if results[0].boxes is not None:
            for i in range(len(results[0].boxes)):
                x1, y1, x2, y2 = map(int, results[0].boxes.xyxy[i].cpu().numpy())
                if not is_valid(x1, y1, x2, y2):
                    continue

                conf_val = float(results[0].boxes.conf[i])
                track_id = ""
                if results[0].boxes.id is not None:
                    track_id = f"#{int(results[0].boxes.id[i])}"

                count += 1
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 100), 2)
                cv2.putText(annotated, f"car {track_id} {conf_val:.2f}",
                            (x1, max(y1 - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 100), 1)

        # ROI 경계선
        cv2.line(annotated, (0, ROI_Y1), (1920, ROI_Y1), (80, 80, 255), 1)
        cv2.putText(annotated, "ROI boundary", (8, ROI_Y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 255), 1)

        # 차량 수
        cv2.rectangle(annotated, (10, 10), (230, 55), (20, 20, 20), -1)
        cv2.putText(annotated, f"Vehicles: {count}",
                    (18, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 100), 2)

        if idx % 30 == 0:
            print(f"[Frame {idx:04d}] Vehicles: {count}")

        disp = cv2.resize(annotated, (1280, 720))
        cv2.imshow("Vehicle Detection", disp)

        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
        idx += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
