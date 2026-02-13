# AI-Based Face Detection Dashboard

This project implements the hackathon brief for an **accurate, real-time, and scalable AI-based face detection solution** with a **simple, user-friendly dashboard**.

The solution:

- Detects **all visible human faces** in video streams (webcam or uploaded video).
- Handles **multiple faces** and varying resolutions.
- Provides a **real-time dashboard** with:
  - Live video with **bounding boxes** around detected faces.
  - **Face count** per frame.
  - **Timestamps** for detections.
  - A **detection log table** for further analysis.

## 1. Tech Stack

- **Face Detection & Analysis**
  - [`ultralytics`](https://pypi.org/project/ultralytics/) YOLOv8 model (default).
  - Fallback to **OpenCV Haar Cascade** if YOLO is unavailable or disabled.
- **Dashboard**
  - **Streamlit** for a rapid, interactive ML dashboard.
- **Development Environment**
  - Works on local machines or cloud GPU environments (e.g. Google Colab / Kaggle).

## 2. Installation

Create and activate a Python 3.10+ virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

If you have trouble installing `torch` on Windows, use the official PyTorch installation command from the PyTorch website, then re-run:

```bash
pip install ultralytics
```

## 3. Running the Dashboard

From the project root:

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (usually `http://localhost:8501`) in your browser.

## 4. Features Mapped to Hackathon Objectives

- **Algorithm Development**
  - Real-time face detection using YOLOv8 (or OpenCV cascade).
  - Handles multiple faces and varying resolutions.
- **Dashboard Creation**
  - Live video preview with bounding boxes and labels.
  - Key metrics: **current face count**, **total frames processed**, **FPS estimate**.
  - Detection log with **timestamps**, **bounding box coordinates**, and **confidence scores**.
  - Option to **export detections as CSV**.

## 5. Project Structure

```text
app.py              # Streamlit dashboard entry point
detector.py         # Face detection logic (YOLO + Haar fallback)
utils.py            # Helper utilities for drawing, logging, and configuration
requirements.txt    # Python dependencies
README.md           # Project documentation (this file)
```

## 6. Extending the Solution

- **Face Recognition / Names**
  - Plug in libraries like `deepface`, `insightface`, or `face_recognition` to assign identities to faces.
  - Store known embeddings and match them at inference time.
- **Advanced Analytics**
  - Aggregate statistics for:
    - Average faces per frame.
    - Time periods with highest crowd density.
  - Integrate with external databases or REST APIs (FastAPI backend).

## 7. Deployment

### Option 1: Streamlit Cloud (Recommended - Free & Easy)

1. **Push your code to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account
   - Click "New app"
   - Select your repository and branch
   - Set main file path to: `app.py`
   - Click "Deploy"

   **Note:** Streamlit Cloud will automatically install dependencies from `requirements.txt`. The first deployment may take 5-10 minutes as it downloads YOLOv8 models.

### Option 2: Docker Deployment

1. **Build the Docker image**
   ```bash
   docker build -t face-detection-dashboard .
   ```

2. **Run the container**
   ```bash
   docker run -p 8501:8501 face-detection-dashboard
   ```

3. **Access the app**
   - Open `http://localhost:8501` in your browser

### Option 3: Deploy to Cloud Platforms

#### Heroku
```bash
# Install Heroku CLI, then:
heroku create your-app-name
git push heroku main
```

#### AWS/Azure/GCP
- Use the Dockerfile provided
- Deploy to container services (ECS, Azure Container Instances, Cloud Run)
- Ensure GPU support if using YOLO model for better performance

### Option 4: Local Server Deployment

For production use on your own server:

```bash
# Install dependencies
pip install -r requirements.txt

# Run with custom port and host
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

## 8. Submission Checklist

- **Source Code**
  - `app.py`, `detector.py`, `utils.py`, and this `README.md`.
- **Demo Dashboard**
  - Run `streamlit run app.py` and record a short demo video showing:
    - Live webcam detection.
    - Uploaded video detection with detection log.
- **Technical Report (2–4 pages)**
  - Methodology and model choice (YOLOv8 vs Haar).
  - Model architecture overview (high-level).
  - Performance analysis (speed, accuracy observations).
  - Dashboard features and UX design.

