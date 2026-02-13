from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import List, Literal

import cv2
import numpy as np
import streamlit as st

from detector import FaceDetector, draw_detections
from utils import DetectionRecord, append_detections_to_records, records_to_dataframe


def _init_session_state() -> None:
    """Initialize Streamlit session state variables."""
    defaults = {
        "running_webcam": False,
        "running_video": False,
        "records": [],  # type: ignore[var-annotated]
        "webcam_frame_index": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _build_sidebar() -> dict:
    st.sidebar.title("Controls")
    source = st.sidebar.radio(
        "Video Source",
        options=["Webcam", "Uploaded Video"],
        index=0,
    )

    model_type: Literal["yolo", "haar"] = st.sidebar.selectbox(
        "Detection Model",
        options=["yolo", "haar"],
        index=0,
        help="YOLOv8: Detects heads from all angles including bowed faces and back views. Haar: Detects frontal and profile faces.",
    )

    head_detection_mode = st.sidebar.checkbox(
        "Head Detection Mode (YOLO only)",
        value=True,
        help="Focus on head regions instead of full body. Better for exam scenarios with bowed faces.",
        disabled=False,
    )

    confidence = st.sidebar.slider(
        "Detection Confidence",
        min_value=0.1,
        max_value=0.9,
        value=0.3,
        step=0.05,
        help="Lower values detect more heads (including bowed/back views) but may include false positives.",
    )

    max_frames = st.sidebar.number_input(
        "Max Frames to Process (for uploaded video)",
        min_value=50,
        max_value=10_000,
        value=1_000,
        step=50,
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Tip: For best performance, run on a machine with a discrete GPU or in Colab."
    )

    return {
        "source": source,
        "model_type": model_type,
        "confidence": confidence,
        "max_frames": int(max_frames),
        "head_detection_mode": head_detection_mode if model_type == "yolo" else False,
    }


# Module-level cache for VideoCapture (can't be in session state)
_webcam_cap = None

def process_webcam_frame(detector: FaceDetector, frame_placeholder, table_placeholder, download_placeholder) -> None:
    """Process a single webcam frame. Called on each Streamlit rerun."""
    global _webcam_cap
    
    if _webcam_cap is None:
        _webcam_cap = cv2.VideoCapture(0)
        if not _webcam_cap.isOpened():
            st.error("Could not open webcam. Please check your camera permissions.")
            st.session_state["running_webcam"] = False
            _webcam_cap = None
            return
        st.session_state["webcam_frame_index"] = 0
    
    frame_index = st.session_state.get("webcam_frame_index", 0)
    records: List[DetectionRecord] = st.session_state["records"]
    
    ret, frame = _webcam_cap.read()
    if not ret:
        st.warning("No frame received from webcam.")
        st.session_state["running_webcam"] = False
        _webcam_cap.release()
        _webcam_cap = None
        return
    
    frame_index += 1
    st.session_state["webcam_frame_index"] = frame_index
    
    detections = detector.detect(frame)
    append_detections_to_records(records, detections, frame_index)

    annotated = draw_detections(frame, detections)
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    frame_placeholder.image(
        annotated_rgb,
        channels="RGB",
        caption=f"Webcam Stream - Frame {frame_index}",
        use_container_width=True,
    )

    # Live update detection table in parallel
    df = records_to_dataframe(records)
    table_placeholder.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        height=400,
    )

    if not df.empty:
        csv_data = df.to_csv(index=False).encode("utf-8")
        download_placeholder.download_button(
            "Download Detections as CSV",
            data=csv_data,
            file_name="face_detections.csv",
            mime="text/csv",
            key=f"webcam_download_{frame_index}",  # Unique key for each frame
        )
    
    # Rerun to process next frame
    time.sleep(0.01)
    st.rerun()


def run_uploaded_video(detector: FaceDetector, uploaded_file, max_frames: int) -> None:
    """Process an already-uploaded video file and update the live table in parallel."""

    # Save to a temporary file for OpenCV
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    frame_placeholder = st.empty()
    table_placeholder = st.empty()
    download_placeholder = st.empty()

    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        st.error("Could not open the uploaded video.")
        return

    frame_index = 0
    records: List[DetectionRecord] = st.session_state["records"]
    start_time = time.time()

    try:
        while st.session_state.get("running_video", False) and frame_index < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            frame_index += 1
            detections = detector.detect(frame)
            append_detections_to_records(records, detections, frame_index)

            annotated = draw_detections(frame, detections)
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            frame_placeholder.image(
                annotated_rgb,
                channels="RGB",
                caption=f"Uploaded Video - Frame {frame_index}",
                use_container_width=True,
            )

            # Live update detection table in parallel
            df = records_to_dataframe(records)
            table_placeholder.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                height=400,
            )

            if not df.empty:
                csv_data = df.to_csv(index=False).encode("utf-8")
                download_placeholder.download_button(
                    "Download Detections as CSV",
                    data=csv_data,
                    file_name="face_detections.csv",
                    mime="text/csv",
                    key=f"video_download_{frame_index}",  # Unique key for each frame
                )

            # Allow Streamlit to process UI events
            time.sleep(0.005)
    finally:
        cap.release()
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass


def main() -> None:
    st.set_page_config(
        page_title="Real-Time Face Detection Dashboard",
        layout="wide",
        page_icon="👁️",
    )
    _init_session_state()

    st.title("Real-Time AI Head & Face Detection Dashboard")
    st.markdown(
        """
This dashboard implements **AI-powered head and face detection** optimized for exam scenarios:

- Real-time or near-real-time detection from **webcam** or **uploaded video**.
- **Enhanced detection** for bowed faces, side views, and back views (head detection).
- Visual **bounding boxes** around detected heads/faces.
- **Detection history table** with timestamps and bounding box coordinates.
- Works best with **YOLO model** in head detection mode for exam monitoring.
"""
    )

    config = _build_sidebar()
    detector = FaceDetector(
        model_type=config["model_type"],
        confidence=config["confidence"],
        head_detection_mode=config["head_detection_mode"],
    )

    st.markdown("### 1. Live Video Panel")

    # Create placeholders for webcam display (always create them in same location)
    webcam_frame_placeholder = st.empty()
    webcam_table_placeholder = st.empty()
    webcam_download_placeholder = st.empty()

    # Clean up webcam if switching to video source
    global _webcam_cap
    if config["source"] == "Uploaded Video" and st.session_state.get("running_webcam", False):
        st.session_state["running_webcam"] = False
        if _webcam_cap is not None:
            _webcam_cap.release()
            _webcam_cap = None
    
    controls_col, _ = st.columns([1, 2])
    with controls_col:
        if config["source"] == "Webcam":
            if st.button(
                "Start Webcam",
                type="primary",
                disabled=st.session_state.get("running_webcam", False),
            ):
                st.session_state["running_webcam"] = True
                st.session_state["records"] = []
                # Clean up old webcam if exists
                if _webcam_cap is not None:
                    _webcam_cap.release()
                    _webcam_cap = None
                st.session_state["webcam_frame_index"] = 0
                st.rerun()

            if st.button("Stop Webcam", disabled=not st.session_state["running_webcam"]):
                st.session_state["running_webcam"] = False
                # Clean up webcam
                if _webcam_cap is not None:
                    _webcam_cap.release()
                    _webcam_cap = None
                if "webcam_frame_index" in st.session_state:
                    del st.session_state["webcam_frame_index"]
                st.rerun()
        
        # Process webcam frames if running
        if config["source"] == "Webcam":
            if st.session_state.get("running_webcam", False):
                process_webcam_frame(detector, webcam_frame_placeholder, webcam_table_placeholder, webcam_download_placeholder)
            else:
                # Clear placeholders when not running
                webcam_frame_placeholder.empty()
                webcam_table_placeholder.empty()
                webcam_download_placeholder.empty()

        if config["source"] == "Uploaded Video":
            uploaded_file = st.file_uploader(
                "Upload a video file (MP4, AVI, MOV, etc.)",
                type=["mp4", "avi", "mov", "mkv"],
            )

            if st.button(
                "Process Uploaded Video",
                type="primary",
                disabled=st.session_state.get("running_video", False),
            ):
                if uploaded_file is None:
                    st.warning("Please upload a video file before processing.")
                else:
                    st.session_state["running_video"] = True
                    st.session_state["records"] = []
                    run_uploaded_video(detector, uploaded_file, config["max_frames"])
                    st.session_state["running_video"] = False

    st.markdown("---")
    st.markdown(
        "This prototype is ready for extension with **face recognition (names)** and "
        "**advanced analytics** for deployment in smart surveillance environments."
    )


if __name__ == "__main__":
    main()

