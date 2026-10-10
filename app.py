from __future__ import annotations

import csv
import io
import re
import subprocess
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

from flask import Flask, flash, redirect, render_template, request, send_file, url_for

import database

ROOT = Path(__file__).resolve().parent
DB_PATH = database.DEFAULT_DB
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

app = Flask(__name__)
app.config["SECRET_KEY"] = "attendance-local-ui-change-this-before-network-deployment"


def active_student_count() -> int:
    with database.connect(DB_PATH) as connection:
        return int(connection.execute("SELECT COUNT(*) FROM students WHERE active = 1").fetchone()[0])


def scheduled_blocks() -> list[dict]:
    rows = [row for row in database.timetable_entries(db_path=DB_PATH) if row[6].strip()]
    rows.sort(key=lambda row: (DAYS.index(row[1]), row[3], row[2]))
    blocks: list[dict] = []
    for row in rows:
        identity = (row[1], row[3], row[6], row[7], row[8], row[9])
        if blocks and blocks[-1]["identity"] == identity and row[2] == blocks[-1]["last_period"] + 1:
            blocks[-1]["end_time"] = row[5]
            blocks[-1]["last_period"] = row[2]
        else:
            blocks.append({"identity": identity, "row": row, "start_time": row[4], "end_time": row[5], "last_period": row[2]})
    return blocks


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_") or "attendance"


@app.context_processor
def inject_navigation():
    return {"days": DAYS}


@app.route("/")
def dashboard():
    sessions = database.recent_sessions(6, DB_PATH)
    blocks = scheduled_blocks()
    today_name = DAYS[date.today().weekday()]
    today_classes = [block for block in blocks if block["row"][1] == today_name]
    return render_template("dashboard.html", page="dashboard", sessions=sessions, blocks=blocks, today_name=today_name, today_classes=today_classes, student_count=active_student_count())


@app.route("/timetable", methods=["GET", "POST"])
def timetable():
    if request.method == "POST":
        if request.form.get("action") == "clear":
            database.clear_timetable_entries(DB_PATH)
            flash("All timetable classes were cleared.", "success")
            return redirect(url_for("timetable"))
        entry_ids = [int(value) for value in request.form.getlist("entry_ids") if value.isdigit()]
        code = request.form.get("class_code", "").strip()
        subject = request.form.get("subject", "").strip()
        room = request.form.get("room", "").strip()
        group = request.form.get("student_group", "").strip()
        if not entry_ids:
            flash("Select at least one timetable period.", "error")
        elif not code:
            flash("A class code is required.", "error")
        else:
            for entry_id in entry_ids:
                database.save_timetable_entry(entry_id, code, subject, room, group, DB_PATH)
            flash(f"{code} was added to {len(entry_ids)} selected period(s).", "success")
        return redirect(url_for("timetable"))
    rows = database.timetable_entries(db_path=DB_PATH)
    lookup = {(row[1], row[2], row[3]): row for row in rows}
    theory_times = {row[2]: (row[4], row[5]) for row in rows if row[3] == "Theory"}
    return render_template("timetable.html", page="timetable", lookup=lookup, theory_times=theory_times, periods=range(1, 13))


@app.route("/attendance", methods=["GET", "POST"])
def take_attendance():
    blocks = scheduled_blocks()
    if request.method == "POST":
        selected = next((block for block in blocks if str(block["row"][0]) == request.form.get("class_entry_id", "")), None)
        attendance_date = request.form.get("attendance_date", "")
        source = request.form.get("source", "0")
        if not selected or not attendance_date:
            flash("Choose a scheduled class and an attendance date.", "error")
            return redirect(url_for("take_attendance"))
        row = selected["row"]
        output = ROOT / "outputs" / f"ui_{safe_filename(row[6])}_{row[1].lower()}.csv"
        command = [sys.executable, "realtime_attendance.py", "--source", source, "--class-name", row[6], "--subject", row[7] or "General", "--attendance-date", attendance_date, "--output", str(output), "--db", str(DB_PATH)]
        try:
            result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        except OSError as error:
            flash(f"Could not start the attendance camera: {error}", "error")
        else:
            flash("Attendance session saved successfully." if result.returncode == 0 else result.stderr.strip() or result.stdout.strip() or "Attendance could not start.", "success" if result.returncode == 0 else "error")
        return redirect(url_for("records"))
    return render_template("attendance.html", page="attendance", blocks=blocks, today=date.today().isoformat())


@app.route("/records")
def records():
    sessions = database.recent_sessions(100, DB_PATH)
    requested_id = request.args.get("session", type=int)
    session_id = requested_id or (sessions[0][0] if sessions else None)
    return render_template("records.html", page="records", sessions=sessions, session_id=session_id, records=database.session_records(session_id, DB_PATH) if session_id else [])


@app.route("/records/<int:session_id>/download")
def download_records(session_id: int):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Student name", "Status", "Confirmed seconds", "Similarity"])
    writer.writerows(database.session_records(session_id, DB_PATH))
    return send_file(io.BytesIO(output.getvalue().encode("utf-8")), as_attachment=True, download_name=f"attendance_session_{session_id}.csv", mimetype="text/csv")


@app.route("/analytics")
def analytics():
    rows = database.analytics_records(DB_PATH)
    subjects = sorted({row[3] for row in rows if row[3]})
    students = sorted({row[4] for row in rows if row[4]})
    dates = sorted({row[1] for row in rows if row[1]}, reverse=True)
    subject, student, selected_dates = request.args.get("subject", ""), request.args.get("student", ""), request.args.getlist("date")
    filtered = [row for row in rows if (not subject or row[3] == subject) and (not student or row[4] == student) and (not selected_dates or row[1] in selected_dates)]
    student_data: dict[str, dict] = defaultdict(lambda: {"classes": 0, "present": 0})
    class_data: dict[str, dict] = defaultdict(lambda: {"present": 0, "total": 0})
    trend_data: dict[str, dict] = defaultdict(lambda: {"present": 0, "total": 0})
    for row in filtered:
        student_data[row[4]]["classes"] += 1
        student_data[row[4]]["present"] += row[5] == "Present"
        class_key = f"{row[2]} — {row[3]}"
        class_data[class_key]["total"] += 1
        class_data[class_key]["present"] += row[5] == "Present"
        trend_data[row[1]]["total"] += 1
        trend_data[row[1]]["present"] += row[5] == "Present"
    student_summary = sorted(({"name": name, **value, "rate": round(value["present"] / value["classes"] * 100, 1)} for name, value in student_data.items()), key=lambda item: item["rate"])
    class_summary = sorted(({"name": name, **value, "rate": round(value["present"] / value["total"] * 100, 1)} for name, value in class_data.items()), key=lambda item: item["rate"], reverse=True)
    trend = [{"date": name, "rate": round(value["present"] / value["total"] * 100, 1)} for name, value in sorted(trend_data.items())]
    return render_template("analytics.html", page="analytics", subjects=subjects, students=students, dates=dates, selected_subject=subject, selected_student=student, selected_dates=selected_dates, total=len(filtered), present=sum(row[5] == "Present" for row in filtered), sessions=len({row[0] for row in filtered}), student_summary=student_summary, class_summary=class_summary, trend=trend)


@app.route("/students", methods=["GET", "POST"])
def students():
    if request.method == "POST":
        try:
            with (ROOT / "students.csv").open(newline="", encoding="utf-8") as file:
                names = [row["student_name"].strip() for row in csv.DictReader(file) if row.get("student_name", "").strip()]
            database.sync_students(names, DB_PATH)
            flash(f"{len(names)} student(s) synced from students.csv.", "success")
        except (FileNotFoundError, KeyError) as error:
            flash(f"Could not sync students: {error}", "error")
        return redirect(url_for("students"))
    return render_template("students.html", page="students", students=database.student_list(DB_PATH), csv_exists=(ROOT / "students.csv").exists())


if __name__ == "__main__":
    database.initialize_timetable(DB_PATH)
    app.run(host="127.0.0.1", port=5000, debug=True)
