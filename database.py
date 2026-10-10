from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_DB = ROOT / "database" / "attendance.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS students (
    student_name TEXT PRIMARY KEY,
    active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS attendance_sessions (
    session_id INTEGER PRIMARY KEY AUTOINCREMENT,
    class_name TEXT NOT NULL,
    subject TEXT NOT NULL,
    source TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS attendance_records (
    record_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    student_name TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('Present', 'Absent')),
    first_confirmed_seconds REAL,
    similarity REAL,
    FOREIGN KEY(session_id) REFERENCES attendance_sessions(session_id),
    UNIQUE(session_id, student_name)
);

CREATE TABLE IF NOT EXISTS timetable_entries (
    entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
    day_name TEXT NOT NULL,
    period_index INTEGER NOT NULL,
    class_type TEXT NOT NULL CHECK(class_type IN ('Theory', 'Lab')),
    start_time TEXT NOT NULL,
    end_time TEXT NOT NULL,
    class_code TEXT NOT NULL DEFAULT '',
    subject TEXT NOT NULL DEFAULT '',
    room TEXT NOT NULL DEFAULT '',
    student_group TEXT NOT NULL DEFAULT '',
    UNIQUE(day_name, period_index, class_type)
);
"""


@contextmanager
def connect(db_path: str | Path = DEFAULT_DB):
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(SCHEMA)
        yield connection
        connection.commit()
    finally:
        connection.close()


def initialize_timetable(db_path: str | Path = DEFAULT_DB) -> None:
    theory_slots = [
        ("08:00", "08:50"), ("08:55", "09:45"), ("09:50", "10:40"),
        ("10:45", "11:35"), ("11:40", "12:30"), ("12:35", "13:25"),
        ("14:00", "14:50"), ("14:55", "15:45"), ("15:50", "16:40"),
        ("16:45", "17:35"), ("17:40", "18:30"), ("18:35", "19:25"),
    ]
    lab_slots = [
        ("08:00", "08:50"), ("08:50", "09:40"), ("09:50", "10:40"),
        ("10:40", "11:30"), ("11:40", "12:30"), ("12:30", "13:20"),
        ("14:00", "14:50"), ("14:50", "15:40"), ("15:50", "16:40"),
        ("16:40", "17:30"), ("17:40", "18:30"), ("18:30", "19:20"),
    ]
    with connect(db_path) as connection:
        for day in ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"):
            for class_type, slots in (("Theory", theory_slots), ("Lab", lab_slots)):
                for period_index, (start, end) in enumerate(slots, start=1):
                    connection.execute(
                        "INSERT OR IGNORE INTO timetable_entries(day_name, period_index, class_type, start_time, end_time) VALUES (?, ?, ?, ?, ?)",
                        (day, period_index, class_type, start, end),
                    )


def timetable_entries(day_name: str | None = None, db_path: str | Path = DEFAULT_DB):
    initialize_timetable(db_path)
    with connect(db_path) as connection:
        if day_name:
            return connection.execute(
                "SELECT entry_id, day_name, period_index, class_type, start_time, end_time, class_code, subject, room, student_group FROM timetable_entries WHERE day_name = ? ORDER BY period_index, class_type",
                (day_name,),
            ).fetchall()
        return connection.execute(
            "SELECT entry_id, day_name, period_index, class_type, start_time, end_time, class_code, subject, room, student_group FROM timetable_entries ORDER BY day_name, period_index, class_type"
        ).fetchall()


def save_timetable_entry(entry_id: int, class_code: str, subject: str, room: str, student_group: str, db_path: str | Path = DEFAULT_DB) -> None:
    with connect(db_path) as connection:
        connection.execute(
            "UPDATE timetable_entries SET class_code = ?, subject = ?, room = ?, student_group = ? WHERE entry_id = ?",
            (class_code.strip(), subject.strip(), room.strip(), student_group.strip(), entry_id),
        )


def clear_timetable_entries(db_path: str | Path = DEFAULT_DB) -> None:
    with connect(db_path) as connection:
        connection.execute("UPDATE timetable_entries SET class_code = '', subject = '', room = '', student_group = ''")


def sync_students(student_names, db_path: str | Path = DEFAULT_DB) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    with connect(db_path) as connection:
        for name in student_names:
            connection.execute(
                "INSERT INTO students(student_name, created_at) VALUES (?, ?) "
                "ON CONFLICT(student_name) DO UPDATE SET active = 1",
                (str(name), now),
            )


def save_attendance(
    student_names,
    present: dict[str, dict],
    source: str,
    class_name: str = "Unassigned class",
    subject: str = "General",
    attendance_date: str | None = None,
    db_path: str | Path = DEFAULT_DB,
) -> int:
    now = datetime.now()
    timestamp = f"{attendance_date}T{now.strftime('%H:%M:%S')}" if attendance_date else now.isoformat(timespec="seconds")
    names = [str(name) for name in student_names]
    with connect(db_path) as connection:
        sync_students(names, db_path)
        cursor = connection.execute(
            "INSERT INTO attendance_sessions(class_name, subject, source, started_at, ended_at) VALUES (?, ?, ?, ?, ?)",
            (class_name, subject, source, timestamp, timestamp),
        )
        session_id = int(cursor.lastrowid)
        for name in names:
            entry = present.get(name)
            connection.execute(
                "INSERT INTO attendance_records(session_id, student_name, status, first_confirmed_seconds, similarity) VALUES (?, ?, ?, ?, ?)",
                (
                    session_id,
                    name,
                    "Present" if entry else "Absent",
                    entry.get("first_confirmed_seconds") if entry else None,
                    entry.get("similarity") if entry else None,
                ),
            )
    return session_id


def clear_attendance_history(db_path: str | Path = DEFAULT_DB) -> None:
    with connect(db_path) as connection:
        connection.execute("DELETE FROM attendance_records")
        connection.execute("DELETE FROM attendance_sessions")


def student_list(db_path: str | Path = DEFAULT_DB):
    with connect(db_path) as connection:
        return connection.execute(
            "SELECT student_name, active, created_at FROM students ORDER BY student_name"
        ).fetchall()


def analytics_records(db_path: str | Path = DEFAULT_DB):
    with connect(db_path) as connection:
        return connection.execute(
            "SELECT s.session_id, date(s.started_at) AS date, s.class_name, s.subject, "
            "r.student_name, r.status, r.similarity "
            "FROM attendance_sessions s JOIN attendance_records r ON r.session_id = s.session_id "
            "ORDER BY s.started_at, r.student_name"
        ).fetchall()


def recent_sessions(limit: int = 20, db_path: str | Path = DEFAULT_DB):
    with connect(db_path) as connection:
        return connection.execute(
            "SELECT session_id, class_name, subject, source, started_at, ended_at FROM attendance_sessions ORDER BY session_id DESC LIMIT ?",
            (limit,),
        ).fetchall()


def session_records(session_id: int, db_path: str | Path = DEFAULT_DB):
    with connect(db_path) as connection:
        return connection.execute(
            "SELECT student_name, status, first_confirmed_seconds, similarity FROM attendance_records WHERE session_id = ? ORDER BY student_name",
            (session_id,),
        ).fetchall()
