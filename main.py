
import json
import logging
import sqlite3
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO
from insightface.app import FaceAnalysis


# ==================================================
# 1. CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parent

with open(ROOT / "config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

VIDEO_SOURCE = ROOT / config["video_source"]
YOLO_MODEL = ROOT / config["yolo_model"]
DB_PATH = ROOT / config["database_path"]
FACE_DIR = ROOT / config["face_images_dir"]
EVENTS_LOG = ROOT / config["events_log"]
OUTPUT_VIDEO = ROOT / "logs" / "annotated_output.mp4"

FRAME_SKIP = max(0, int(config["frame_skip"]))
CONFIDENCE = float(config["confidence_threshold"])
SIMILARITY_THRESHOLD = float(config["similarity_threshold"])
MAX_LOST_FRAMES = max(1, int(config["max_lost_frames"]))

for folder in [DB_PATH.parent, FACE_DIR, EVENTS_LOG.parent]:
    folder.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=str(EVENTS_LOG),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    encoding="utf-8"
)


# ==================================================
# 2. DATABASE
# ==================================================

db = sqlite3.connect(str(DB_PATH))

db.execute("""
CREATE TABLE IF NOT EXISTS visitors (
    face_id TEXT PRIMARY KEY,
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL,
    image_path TEXT NOT NULL,
    embedding BLOB NOT NULL
)
""")

db.execute("""
CREATE TABLE IF NOT EXISTS events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    face_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    frame_number INTEGER NOT NULL,
    image_path TEXT
)
""")

db.commit()


def log_event(face_id, event_type, frame_no, image_path):
    timestamp = datetime.now().isoformat(timespec="seconds")

    db.execute(
        """
        INSERT INTO events
        (face_id, event_type, timestamp, frame_number, image_path)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            face_id,
            event_type,
            timestamp,
            frame_no,
            str(image_path)
        )
    )
    db.commit()

    logging.info(
        "%s | %s | frame=%s | image=%s",
        event_type,
        face_id,
        frame_no,
        image_path
    )

    print(f"{event_type}: {face_id} | frame={frame_no}")


# ==================================================
# 3. LOAD REGISTERED VISITORS
# ==================================================

visitors = {}

rows = db.execute("""
    SELECT face_id, image_path, embedding, last_seen
    FROM visitors
""").fetchall()

for row in rows:
    visitors[row[0]] = {
        "image_path": row[1],
        "embedding": np.frombuffer(
            row[2], dtype=np.float32
        ).copy(),
        "last_seen_frame": -1,
        "last_seen_time": row[3],
        "active": False
    }

next_id = 1

for face_id in visitors:
    try:
        next_id = max(
            next_id,
            int(face_id.split("_")[-1]) + 1
        )
    except (ValueError, IndexError):
        pass


# ==================================================
# 4. INITIALIZE MODELS AND VIDEO
# ==================================================

if not VIDEO_SOURCE.exists():
    db.close()
    raise FileNotFoundError(f"Video not found: {VIDEO_SOURCE}")

if not YOLO_MODEL.exists():
    db.close()
    raise FileNotFoundError(f"YOLO model not found: {YOLO_MODEL}")

print("Loading YOLO model...")
yolo = YOLO(str(YOLO_MODEL))

print("Loading InsightFace...")
face_app = FaceAnalysis(
    name="buffalo_l",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
face_app.prepare(ctx_id=0, det_size=(640, 640))

cap = cv2.VideoCapture(str(VIDEO_SOURCE))

if not cap.isOpened():
    db.close()
    raise RuntimeError(f"Could not open video: {VIDEO_SOURCE}")

fps = cap.get(cv2.CAP_PROP_FPS)
if not fps or fps <= 0:
    fps = 25.0

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

OUTPUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)

writer = cv2.VideoWriter(
    str(OUTPUT_VIDEO),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

if not writer.isOpened():
    cap.release()
    db.close()
    raise RuntimeError(f"Could not create video: {OUTPUT_VIDEO}")

print("Starting face tracking...")
print("Input:", VIDEO_SOURCE)
print("Output:", OUTPUT_VIDEO)


# ==================================================
# 5. PROCESS VIDEO
# ==================================================

frame_no = 0
processed_frames = 0

try:
    while True:
        success, frame = cap.read()

        if not success:
            break

        frame_no += 1
        display_faces = []

        # Analyse every (FRAME_SKIP + 1) frames.
        should_process = (
            (frame_no - 1) % (FRAME_SKIP + 1) == 0
        )

        if should_process:
            processed_frames += 1
            seen_this_frame = set()

            result = yolo.predict(
                source=frame,
                conf=CONFIDENCE,
                verbose=False,
                device=0
            )[0]

            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(
                        int, box.xyxy[0].cpu().tolist()
                    )

                    x1 = max(0, min(width - 1, x1))
                    y1 = max(0, min(height - 1, y1))
                    x2 = max(0, min(width, x2))
                    y2 = max(0, min(height, y2))

                    if x2 <= x1 or y2 <= y1:
                        continue

                    # Expand crop so InsightFace can detect the face.
                    pad_x = int((x2 - x1) * 0.35)
                    pad_y = int((y2 - y1) * 0.35)

                    cx1 = max(0, x1 - pad_x)
                    cy1 = max(0, y1 - pad_y)
                    cx2 = min(width, x2 + pad_x)
                    cy2 = min(height, y2 + pad_y)

                    crop = frame[cy1:cy2, cx1:cx2]

                    if crop.size == 0:
                        continue

                    detected_faces = face_app.get(crop)

                    if not detected_faces:
                        # Keep a visible detection box even when
                        # InsightFace cannot generate an embedding.
                        display_faces.append(
                            (x1, y1, x2, y2, "Face detected")
                        )
                        continue

                    face = max(
                        detected_faces,
                        key=lambda item: float(item.det_score)
                    )

                    embedding = np.asarray(
                        face.normed_embedding,
                        dtype=np.float32
                    )

                    if embedding.size == 0:
                        continue

                    # Find closest known identity.
                    best_id = None
                    best_similarity = -1.0

                    for known_id, info in visitors.items():
                        known_embedding = info["embedding"]

                        if known_embedding.shape != embedding.shape:
                            continue

                        similarity = float(
                            np.dot(embedding, known_embedding)
                        )

                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_id = known_id

                    is_new = (
                        best_id is None
                        or best_similarity < SIMILARITY_THRESHOLD
                    )

                    now = datetime.now().isoformat(
                        timespec="seconds"
                    )

                    if is_new:
                        face_id = f"visitor_{next_id:04d}"
                        next_id += 1

                        image_path = FACE_DIR / f"{face_id}.jpg"

                        if not cv2.imwrite(str(image_path), crop):
                            print("Warning: could not save", image_path)
                            continue

                        db.execute(
                            """
                            INSERT INTO visitors
                            (face_id, first_seen, last_seen,
                             image_path, embedding)
                            VALUES (?, ?, ?, ?, ?)
                            """,
                            (
                                face_id,
                                now,
                                now,
                                str(image_path),
                                embedding.tobytes()
                            )
                        )
                        db.commit()

                        visitors[face_id] = {
                            "image_path": str(image_path),
                            "embedding": embedding.copy(),
                            "last_seen_frame": frame_no,
                            "last_seen_time": now,
                            "active": True
                        }

                        log_event(
                            face_id, "ENTRY", frame_no, image_path
                        )

                    else:
                        face_id = best_id

                        # Avoid assigning one identity twice in a frame.
                        if face_id in seen_this_frame:
                            display_faces.append(
                                (x1, y1, x2, y2, "Face detected")
                            )
                            continue

                        info = visitors[face_id]

                        was_missing = (
                            not info["active"]
                            or (
                                info["last_seen_frame"] >= 0
                                and frame_no - info["last_seen_frame"]
                                > MAX_LOST_FRAMES
                            )
                        )

                        if was_missing:
                            log_event(
                                face_id,
                                "ENTRY",
                                frame_no,
                                info["image_path"]
                            )

                        # Update identity embedding conservatively.
                        updated = (
                            0.8 * info["embedding"]
                            + 0.2 * embedding
                        )

                        norm = np.linalg.norm(updated)

                        if norm > 0:
                            updated = updated / norm

                        info["embedding"] = updated.astype(np.float32)
                        info["active"] = True
                        info["last_seen_frame"] = frame_no
                        info["last_seen_time"] = now

                        db.execute(
                            """
                            UPDATE visitors
                            SET last_seen = ?, embedding = ?
                            WHERE face_id = ?
                            """,
                            (
                                now,
                                info["embedding"].tobytes(),
                                face_id
                            )
                        )
                        db.commit()

                    seen_this_frame.add(face_id)

                    display_faces.append(
                        (x1, y1, x2, y2, face_id)
                    )

        # Check for visitor exits on every frame.
        for face_id, info in visitors.items():
            if (
                info["active"]
                and info["last_seen_frame"] >= 0
                and frame_no - info["last_seen_frame"] > MAX_LOST_FRAMES
            ):
                log_event(
                    face_id,
                    "EXIT",
                    frame_no,
                    info["image_path"]
                )
                info["active"] = False

        # Draw boxes on analysed frames.
        # Skipped frames retain the original video image.
        for x1, y1, x2, y2, face_id in display_faces:
            colour = (0, 255, 0)

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                colour,
                3
            )

            label = face_id
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2

            (text_w, text_h), baseline = cv2.getTextSize(
                label, font, font_scale, thickness
            )

            label_y = max(y1, text_h + 12)

            cv2.rectangle(
                frame,
                (x1, label_y - text_h - 10),
                (x1 + text_w + 12, label_y + baseline - 4),
                colour,
                -1
            )

            cv2.putText(
                frame,
                label,
                (x1 + 6, label_y - 5),
                font,
                font_scale,
                (0, 0, 0),
                thickness,
                cv2.LINE_AA
            )

        # Add a status label to the video.
        cv2.putText(
            frame,
            f"Frame: {frame_no} | Visitors registered: {len(visitors)}",
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

        writer.write(frame)

        if should_process and processed_frames % 10 == 0:
            print(
                f"Frames analysed: {processed_frames} | "
                f"Visitors registered: {len(visitors)}"
            )

finally:
    cap.release()
    writer.release()
    db.close()


# ==================================================
# 6. SUMMARY
# ==================================================

print("\nProcessing complete!")
print("Total video frames:", frame_no)
print("Frames analysed by YOLO:", processed_frames)
print("Registered visitor identities:", len(visitors))
print("Database:", DB_PATH)
print("Face images:", FACE_DIR)
print("Event log:", EVENTS_LOG)
print("Annotated video:", OUTPUT_VIDEO)
