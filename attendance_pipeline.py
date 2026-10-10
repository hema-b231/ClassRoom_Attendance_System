from __future__ import annotations

import csv
import json
import urllib.request
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
STUDENTS_DIR = DATA_DIR / "registered_students"
VIDEOS_DIR = ROOT / "test_videos"
MODELS_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
GALLERY_FILE = MODELS_DIR / "gallery.npz"

DETECTOR_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
RECOGNIZER_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
DETECTOR_FILE = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
RECOGNIZER_FILE = MODELS_DIR / "face_recognition_sface_2021dec.onnx"

# OpenCV's SFace tutorial uses about 0.363 as a general cosine-match threshold.
# 0.45 is intentionally stricter for this attendance prototype; calibrate it later.
COSINE_THRESHOLD = 0.45
STABLE_WINDOW = 7
STABLE_HITS = 5
MIN_FACE_WIDTH = 70


def download_models() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    for url, destination in ((DETECTOR_URL, DETECTOR_FILE), (RECOGNIZER_URL, RECOGNIZER_FILE)):
        if not destination.exists():
            print(f"Downloading {destination.name} ...")
            urllib.request.urlretrieve(url, destination)
    print("Models are ready.")


def make_models():
    download_models()
    detector = cv2.FaceDetectorYN.create(str(DETECTOR_FILE), "", (320, 320), 0.8, 0.3, 5000)
    recognizer = cv2.FaceRecognizerSF.create(str(RECOGNIZER_FILE), "")
    return detector, recognizer


def detect_faces(detector, image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    detector.setInputSize((width, height))
    _, faces = detector.detect(image)
    return np.empty((0, 15), dtype=np.float32) if faces is None else faces


def face_feature(recognizer, image: np.ndarray, face: np.ndarray) -> np.ndarray:
    aligned = recognizer.alignCrop(image, face)
    feature = recognizer.feature(aligned)
    return feature.astype(np.float32).reshape(1, -1)


def largest_face(faces: np.ndarray):
    if len(faces) == 0:
        return None
    return max(faces, key=lambda f: float(f[2] * f[3]))


def build_gallery() -> dict[str, np.ndarray]:
    detector, recognizer = make_models()
    labels, vectors = [], []
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    if not STUDENTS_DIR.exists():
        raise FileNotFoundError(f"Missing folder: {STUDENTS_DIR}")

    for student_dir in sorted(p for p in STUDENTS_DIR.iterdir() if p.is_dir()):
        student_vectors = []
        for image_path in sorted(student_dir.iterdir()):
            if image_path.suffix.lower() not in image_extensions:
                continue
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"SKIP unreadable image: {image_path}")
                continue
            face = largest_face(detect_faces(detector, image))
            if face is None:
                print(f"SKIP no face found: {image_path}")
                continue
            student_vectors.append(face_feature(recognizer, image, face))

        if not student_vectors:
            print(f"WARNING: no usable images for {student_dir.name}")
            continue
        labels.append(student_dir.name)
        vectors.append(np.mean(np.vstack(student_vectors), axis=0, keepdims=True))
        print(f"{student_dir.name}: {len(student_vectors)} usable images")

    if not vectors:
        raise RuntimeError("No usable enrollment images were found.")
    gallery = {"labels": np.array(labels), "features": np.vstack(vectors)}
    MODELS_DIR.mkdir(exist_ok=True)
    np.savez_compressed(GALLERY_FILE, **gallery)
    print(f"Saved gallery: {GALLERY_FILE}")
    return gallery


def load_gallery() -> dict[str, np.ndarray]:
    if not GALLERY_FILE.exists():
        return build_gallery()
    data = np.load(GALLERY_FILE, allow_pickle=False)
    return {"labels": data["labels"], "features": data["features"]}


def recognize(recognizer, feature: np.ndarray, gallery: dict[str, np.ndarray]):
    scores = [float(recognizer.match(feature, known, cv2.FaceRecognizerSF_FR_COSINE)) for known in gallery["features"]]
    if not scores:
        return None, 0.0
    best_index = int(np.argmax(scores))
    score = scores[best_index]
    return (str(gallery["labels"][best_index]), score) if score >= COSINE_THRESHOLD else (None, score)


def process_video(video_path: str | Path, output_path: str | Path | None = None) -> Path:
    detector, recognizer = make_models()
    gallery = load_gallery()
    video_path = Path(video_path)
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = Path(output_path) if output_path else OUTPUT_DIR / f"{video_path.stem}_processed.mp4"

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    histories = defaultdict(lambda: deque(maxlen=STABLE_WINDOW))
    present = {}
    frame_number = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame_number += 1
        faces = detect_faces(detector, frame)
        for index, face in enumerate(faces):
            x, y, w, h = [max(0, int(value)) for value in face[:4]]
            x2, y2 = min(width - 1, x + w), min(height - 1, y + h)
            if w < MIN_FACE_WIDTH:
                label, score = None, 0.0
            else:
                feature = face_feature(recognizer, frame, face)
                label, score = recognize(recognizer, feature, gallery)

            track_key = index
            histories[track_key].append(label)
            counts = {name: list(histories[track_key]).count(name) for name in set(histories[track_key]) if name}
            stable_label = max(counts, key=counts.get) if counts and max(counts.values()) >= STABLE_HITS else None
            colour = (0, 180, 0) if stable_label else (0, 0, 255)
            shown_label = f"{stable_label} PRESENT" if stable_label else ("Unknown" if label is None else f"Checking {label}")
            cv2.rectangle(frame, (x, y), (x2, y2), colour, 2)
            cv2.putText(frame, f"{shown_label} {score:.2f}", (x, max(25, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, colour, 2)
            if stable_label and stable_label not in present:
                present[stable_label] = {"frame": frame_number, "time_seconds": round(frame_number / fps, 2), "score": round(score, 4)}

        cv2.putText(frame, "Red=not confirmed | Green=attendance confirmed", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        writer.write(frame)

    capture.release()
    writer.release()
    timestamp = datetime.now().isoformat(timespec="seconds")
    attendance_file = OUTPUT_DIR / f"{video_path.stem}_attendance.csv"
    with attendance_file.open("w", newline="", encoding="utf-8") as file:
        writer_csv = csv.writer(file)
        writer_csv.writerow(["student_name", "status", "first_confirmed_seconds", "similarity", "processed_at"])
        for student_name in gallery["labels"]:
            student_name = str(student_name)
            entry = present.get(student_name)
            writer_csv.writerow([student_name, "Present" if entry else "Absent", entry.get("time_seconds", "") if entry else "", entry.get("score", "") if entry else "", timestamp])

    print(json.dumps({"video": str(video_path), "processed_video": str(output_path), "attendance_csv": str(attendance_file), "present": present}, indent=2))
    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build face gallery or process a recorded attendance video.")
    parser.add_argument("command", choices=["build-gallery", "process-video"])
    parser.add_argument("video", nargs="?", help="Video path, required for process-video")
    args = parser.parse_args()
    if args.command == "build-gallery":
        build_gallery()
    else:
        if not args.video:
            raise SystemExit("Provide a video path, for example test_videos/test1.mp4")
        process_video(args.video)
