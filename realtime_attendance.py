from __future__ import annotations

import argparse
import csv
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

import cv2

import attendance_pipeline as pipeline
import database


def parse_source(value: str):
    return int(value) if value.isdigit() else value


def run_realtime(
    source,
    output_csv: Path,
    show_window: bool = True,
    class_name: str = "Unassigned class",
    subject: str = "General",
    attendance_date: str | None = None,
    db_path: Path = database.DEFAULT_DB,
) -> dict[str, dict]:
    detector, recognizer = pipeline.make_models()
    gallery = pipeline.load_gallery()
    capture = cv2.VideoCapture(source)

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open source {source!r}. If using a webcam, try --source 1 or check Windows camera permissions."
        )

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    histories = defaultdict(lambda: deque(maxlen=pipeline.STABLE_WINDOW))
    present: dict[str, dict] = {}
    frame_number = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame_number += 1
        height, width = frame.shape[:2]
        faces = pipeline.detect_faces(detector, frame)

        for index, face in enumerate(faces):
            x, y, w, h = [max(0, int(value)) for value in face[:4]]
            x2, y2 = min(width - 1, x + w), min(height - 1, y + h)
            if w < pipeline.MIN_FACE_WIDTH:
                label, score = None, 0.0
            else:
                feature = pipeline.face_feature(recognizer, frame, face)
                label, score = pipeline.recognize(recognizer, feature, gallery)

            histories[index].append(label)
            counts = {
                name: list(histories[index]).count(name)
                for name in set(histories[index])
                if name
            }
            stable_label = (
                max(counts, key=counts.get)
                if counts and max(counts.values()) >= pipeline.STABLE_HITS
                else None
            )

            colour = (0, 180, 0) if stable_label else (0, 0, 255)
            if stable_label:
                text = f"{stable_label} - PRESENT"
                if stable_label not in present:
                    present[stable_label] = {
                        "first_confirmed_seconds": round(frame_number / fps, 2),
                        "similarity": round(score, 4),
                    }
            elif label:
                text = f"Checking {label} ({score:.2f})"
            else:
                text = f"Unknown ({score:.2f})"

            cv2.rectangle(frame, (x, y), (x2, y2), colour, 2)
            cv2.putText(
                frame,
                text,
                (x, max(25, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                colour,
                2,
                cv2.LINE_AA,
            )

        cv2.putText(
            frame,
            "q = stop attendance",
            (20, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if show_window:
            cv2.imshow("Real-time Attendance", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    capture.release()
    if show_window:
        cv2.destroyAllWindows()

    output_csv.parent.mkdir(exist_ok=True)
    timestamp = datetime.now().isoformat(timespec="seconds")
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "student_name",
            "status",
            "first_confirmed_seconds",
            "similarity",
            "processed_at",
        ])
        for student_name in gallery["labels"]:
            student_name = str(student_name)
            entry = present.get(student_name)
            writer.writerow([
                student_name,
                "Present" if entry else "Absent",
                entry["first_confirmed_seconds"] if entry else "",
                entry["similarity"] if entry else "",
                timestamp,
            ])

    session_id = database.save_attendance(
        gallery["labels"],
        present,
        str(source),
        class_name=class_name,
        subject=subject,
        attendance_date=attendance_date,
        db_path=db_path,
    )
    print(f"Saved database session: {session_id}")
    return present


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time or source-file attendance")
    parser.add_argument(
        "--source",
        default="0",
        help="Webcam index such as 0, or a video path such as test_videos/test1.mp4",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional CSV output path; defaults to outputs/realtime_attendance_TIMESTAMP.csv",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Process a file without opening a display window; useful for verification",
    )
    parser.add_argument("--class-name", default="Unassigned class", help="Class or timetable label")
    parser.add_argument("--subject", default="General", help="Subject name")
    parser.add_argument("--attendance-date", default=None, help="Attendance date in YYYY-MM-DD format")
    parser.add_argument("--db", default=str(database.DEFAULT_DB), help="SQLite database path")
    args = parser.parse_args()

    output = Path(args.output) if args.output else pipeline.OUTPUT_DIR / (
        "realtime_attendance_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv"
    )
    present = run_realtime(
        parse_source(args.source),
        output,
        show_window=not args.no_display,
        class_name=args.class_name,
        subject=args.subject,
        attendance_date=args.attendance_date,
        db_path=Path(args.db),
    )
    print(f"Saved attendance: {output}")
    print(f"Present: {', '.join(sorted(present)) if present else 'none'}")


if __name__ == "__main__":
    main()
