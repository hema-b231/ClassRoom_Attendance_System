# Recorded-video face attendance prototype

This prototype builds a **face embedding gallery** from the images in `data/registered_students/`, then processes recorded MP4 videos from `test_videos/`. It draws red boxes until a match is stable across several frames; then the box becomes green and the person is marked present once.

This is enrollment/embedding generation, not training a neural network from scratch. It uses OpenCV YuNet for face detection and SFace for face recognition.

## 1. Open a terminal in this folder

```bash
cd /c/Users/akash/attendance_project
```

## 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `python -m pip` is unavailable, install Python from python.org and enable **Add Python to PATH**, then retry.

## 3. Check your folders

```text
data/registered_students/Akash/*.jpeg
data/registered_students/Ashmit/*.jpeg
test_videos/test1.mp4
```

Use one folder per consenting participant. The folder name becomes the student name written to the attendance CSV. Keep the names consistent and avoid duplicate folder names.

## 4. Build the gallery

The first run downloads two ONNX model files into `models/` and creates `models/gallery.npz`:

```bash
python attendance_pipeline.py build-gallery
```

This may take a few minutes the first time. It will report usable images and skip images where no face is found.

## 5. Process a recorded video

```bash
python attendance_pipeline.py process-video test_videos/test1.mp4
```

Results are written to `outputs/`:

- `test1_processed.mp4`: annotated video
- `test1_attendance.csv`: Present/Absent results

The realtime command also stores sessions in `database/attendance.db`. You can provide class metadata:

```bash
python realtime_attendance.py --source 0 --class-name A1-SWE1017-ETH --subject "Software Engineering"
```

The CSV remains available for Excel, while SQLite is used by the lecturer dashboard. Each run creates one attendance session and one record for every registered student.

## 6. Start the lecturer dashboard

The project uses a custom Flask interface instead of Streamlit. Start it with:

```bash
python app.py
```

Then open `http://127.0.0.1:5000` in your browser. The dashboard includes the timetable, live attendance setup, records with CSV export, analytics, and student syncing. The face-recognition camera window still opens separately; press `q` in that window to finish a session.

## 7. Use the notebook

Start Jupyter from this folder:

```bash
jupyter notebook
```

Open `face_attendance_training.ipynb` and run the cells from top to bottom. The notebook is a user-friendly wrapper around the same pipeline.

## Important notes

- `0.45` is a deliberately strict starting cosine-similarity threshold. It is not a universal value; calibrate it using new test images.
- A match must be stable in at least 5 of the last 7 observations before attendance is confirmed.
- This first version draws boxes around faces, not full bodies. Full-body red boxes require adding a person detector such as YOLO in a later phase.
- Do not use enrollment images as test images. Keep separate recordings for evaluation.
- Face data is biometric data. Obtain permission, restrict access, and delete it when the project requires.
