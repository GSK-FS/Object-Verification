import cv2
from model import detect_objects, get_model

VIDEO_PATH = "data/IMG_2565 2.MOV"   # local video
ALLOWED_CLASSES = ["car"]
FRAME_SKIP = 5                            # increase if video is large
CONF_THRESHOLD = 0.3

def main():
    print("[INFO] Loading model...")
    get_model()

    cap = cv2.VideoCapture(VIDEO_PATH)

    total_frames = 0
    car_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1
        if total_frames % FRAME_SKIP != 0:
            continue

        detections = detect_objects(frame, ALLOWED_CLASSES)

        car_found = False
        for det in detections:
            if det["confidence"] >= CONF_THRESHOLD:
                car_found = True
                x1, y1, x2, y2 = map(int, det["bbox"])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{det['label']} {det['confidence']}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        if car_found:
            car_frames += 1
            status = "CAR DETECTED"
            color = (0, 255, 0)
        else:
            status = "CAR LOST"
            color = (0, 0, 255)

        ratio = car_frames / max(1, total_frames)

        cv2.putText(frame, f"Frames: {total_frames}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Car Ratio: {ratio:.2f}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, status, (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("YOLO Car Walkaround Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("\n[RESULT]")
    print(f"Total Frames Processed: {total_frames}")
    print(f"Car Frames: {car_frames}")
    print(f"Presence Ratio: {car_frames / max(1, total_frames):.2f}")


if __name__ == "__main__":
    main()
