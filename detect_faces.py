
import cv2
import json
from pathlib import Path
from datetime import datetime
from insightface.app import FaceAnalysis

VIDEO_PATH = Path("videos/video_sample1.mp4")
OUTPUT_DIR = Path("logs/detected_faces")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading InsightFace model...")

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
app.prepare(ctx_id=0, det_size=(640, 640))

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

frame_number = 0
total_faces = 0
results = []

print("Processing video...")

while True:
    success, frame = cap.read()

    if not success:
        break

    frame_number += 1
    faces = app.get(frame)

    for face in faces:
        x1, y1, x2, y2 = map(int, face.bbox)
        height, width = frame.shape[:2]

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        crop = frame[y1:y2, x1:x2]

        total_faces += 1
        image_name = f"face_{total_faces:04d}.jpg"
        image_path = OUTPUT_DIR / image_name

        cv2.imwrite(str(image_path), crop)

        results.append({
            "frame": frame_number,
            "face_number": total_faces,
            "image": str(image_path),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "confidence": round(float(face.det_score), 4),
            "bbox": [x1, y1, x2, y2]
        })

    if frame_number % 30 == 0:
        print(f"Processed {frame_number} frames")

cap.release()

with open(OUTPUT_DIR / "detections.json", "w", encoding="utf-8") as file:
    json.dump(results, file, indent=2)

print("\nProcessing complete!")
print("Frames processed:", frame_number)
print("Face detections:", total_faces)
print("Output folder:", OUTPUT_DIR.resolve())
