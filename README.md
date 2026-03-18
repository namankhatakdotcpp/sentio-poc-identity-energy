# 🚀 SentioMind — Face Recognition Video Intelligence System

> Transforming raw video into intelligent insights using real-time face detection, recognition, and tracking.

---

## 📌 Overview

**SentioMind** is an end-to-end AI pipeline that processes videos to:

* 🎯 Detect faces in each frame
* 🧠 Recognize known identities
* 🔁 Track individuals across frames
* 🎬 Generate annotated demo videos
* 📊 Produce structured analytics

Built with a production-first mindset, this system mimics real-world AI video intelligence platforms used in surveillance, analytics, and smart environments.

---

## ⚙️ Tech Stack

| Component        | Technology               |
| ---------------- | ------------------------ |
| Backend          | FastAPI                  |
| Face Detection   | `face_recognition` (HOG) |
| Face Recognition | dlib embeddings          |
| Video Processing | OpenCV                   |
| ML Utilities     | NumPy                    |
| Server           | Uvicorn                  |

---

## 🧠 System Architecture

```
Video Input
     ↓
Frame Extraction
     ↓
Face Detection (HOG)
     ↓
Face Encoding (128-d embeddings)
     ↓
Identity Matching (Euclidean Distance)
     ↓
Tracking + Aggregation
     ↓
Annotated Video + Reports
```

---

## 📁 Project Structure

```
sentio-poc-identity-energy/
│
├── api.py                  # FastAPI entrypoint
├── solution.py             # Main pipeline orchestration
├── requirements.txt
├── README.md
│
├── src/
│   ├── detector.py         # Face detection logic
│   ├── face_identity.py    # Face recognition & matching
│
├── known_faces/            # Reference images (input identities)
│   ├── naman.jpg
│   ├── dheeraj.jpg
│
└── outputs/
    ├── demo.mp4            # Annotated output video
    ├── report.json         # Structured results
```

---

## 🚀 Features

### 🎯 Face Detection

* Uses HOG-based detection for CPU efficiency
* Works reliably on real-world video input

### 🧠 Face Recognition

* Encodes faces into 128D embeddings
* Matches using similarity threshold
* Supports multiple known identities

### 🎬 Demo Video Rendering

* Bounding boxes (green/red)
* Identity labels with confidence
* FPS counter
* Clean UI overlays

### 📊 Analytics Output

* Number of people detected
* Frame-wise detections
* Identity aggregation

---

## 🧪 Local Setup

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/sentio-poc-identity-energy.git
cd sentio-poc-identity-energy
```

---

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Add Known Faces

Place images inside:

```
known_faces/
```

Example:

```
known_faces/naman.jpg
known_faces/dheeraj.jpg
```

---

### 5️⃣ Run Server

```bash
uvicorn api:app --reload
```

---

### 6️⃣ Open API Docs

```
http://127.0.0.1:8000/docs
```

---

### 7️⃣ Test with Video Upload

Use `/process-video/` endpoint to upload a video.

---

## 📈 Sample Output

### 🎬 Demo Video

* Annotated with bounding boxes
* Identity labels + confidence
* Smooth playback

### 📊 JSON Output

```json
{
  "persons_found": 2,
  "identities": ["naman", "dheeraj"],
  "frames_processed": 28
}
```

---

## ⚡ Performance

| Metric          | Value                       |
| --------------- | --------------------------- |
| Processing Time | ~3–5 sec per frame          |
| Detection Speed | CPU optimized               |
| Accuracy        | Depends on lighting & angle |

---

## ⚠️ Limitations

* Sensitive to lighting conditions
* HOG detector less robust than deep models
* Requires clear frontal faces

---

## 🔮 Future Improvements

* 🔥 Deep learning detector (RetinaFace / YOLO)
* ⚡ Real-time webcam support
* 🌐 Web UI dashboard
* 📊 Advanced analytics (heatmaps, timelines)
* ☁️ Cloud deployment (AWS / Render)

---

## 👨‍💻 Author

**Naman**
AI/ML Developer | System Builder

---

## ⭐ Why This Project Stands Out

* End-to-end ML system (not just a model)
* Real-world pipeline (video → insights)
* Production-style architecture
* Clean visualization output

---

## 📬 Contributing

Feel free to fork, improve, and submit PRs!

---

## 🏁 Final Note

> This project demonstrates the transition from "model building" to "system engineering" — the key skill for real-world AI roles.

---

🔥 *If you like this project, give it a star!*
