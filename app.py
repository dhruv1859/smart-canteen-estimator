"""
app.py  –  Streamlit dashboard for Smart Canteen Crowd & Queue Estimator
─────────────────────────────────────────────────────────────────────────────
Run:  streamlit run app.py
─────────────────────────────────────────────────────────────────────────────
Tabs
  1. 📸 Image Analysis   – upload a single image, see detections + heatmap
  2. 🎬 Video Analysis   – upload a short video, process frame-by-frame
  3. 📷 Webcam Live      – real-time feed from webcam (browser getUserMedia)
  4. 📊 Analytics        – occupancy charts from logged CSV data
"""

import os, time, io, tempfile
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Canteen Estimator",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS tweaks ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .metric-box {
      background:#1e1e2e; border-radius:12px; padding:16px;
      text-align:center; margin-bottom:8px;
  }
  .metric-title { font-size:0.85rem; color:#aaa; }
  .metric-value { font-size:2rem; font-weight:700; }
  .low      { color: #2ecc71; }
  .moderate { color: #f39c12; }
  .high     { color: #e74c3c; }
</style>
""", unsafe_allow_html=True)

# ── Lazy-load pipeline (cached so model loads only once) ─────────────────────
@st.cache_resource(show_spinner="Loading YOLOv8 model…")
def load_pipeline(model_name: str, conf: float, log_path: str):
    from src.pipeline import CanteenPipeline
    return CanteenPipeline(model_path=model_name, conf=conf,
                           log_path=log_path, enable_log=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image("assets/logo.png", use_container_width=True) if os.path.isfile("assets/logo.png") else None
st.sidebar.title("⚙️ Settings")

model_choice = st.sidebar.selectbox(
    "YOLO Model",
    ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
    index=0,
    help="Nano = fastest; Small / Medium = more accurate but slower",
)
conf_thresh = st.sidebar.slider(
    "Confidence threshold", 0.1, 0.9, 0.35, 0.05
)
show_heatmap = st.sidebar.toggle("Show heatmap overlay", value=True)
log_path     = st.sidebar.text_input("CSV log path", "logs/occupancy_log.csv")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Smart Canteen Crowd & Queue Estimator**  \n"
    "CV Workshop 24B15CS352 · 5th Sem · 2026-27  \n"
    "Dhruv Agrawal · Vrinda Wasan · Shivi Goel"
)

pipe = load_pipeline(model_choice, conf_thresh, log_path)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_img, tab_vid, tab_cam, tab_stats = st.tabs(
    ["📸 Image", "🎬 Video", "📷 Webcam Live", "📊 Analytics"]
)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 – Image Analysis
# ═══════════════════════════════════════════════════════════════════════════════
with tab_img:
    st.header("Single-Image Analysis")
    uploaded = st.file_uploader("Upload a canteen image",
                                type=["jpg", "jpeg", "png", "bmp", "webp"])
    if uploaded:
        pil_img = Image.open(uploaded).convert("RGB")
        frame   = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        result  = pipe.process_frame(frame, show_heatmap=show_heatmap)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Annotated Frame")
            st.image(cv2.cvtColor(result["annotated_frame"], cv2.COLOR_BGR2RGB),
                     use_container_width=True)
        with col2:
            st.subheader("Heatmap Snapshot")
            st.image(cv2.cvtColor(result["heatmap_snapshot"], cv2.COLOR_BGR2RGB),
                     use_container_width=True)

        # Metrics row
        total = result["total"]
        level = result["crowd_level"]
        level_css = level.lower()

        st.markdown("---")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("👤 Total People", total)
        m2.metric("🟢 Crowd Level", level)
        for col, (zone, cnt) in zip([m3, m4, m5], list(result["zone_counts"].items())[:3]):
            col.metric(f"📍 {zone}", cnt)

        st.markdown("#### Zone Breakdown")
        df_zones = pd.DataFrame(
            result["zone_counts"].items(), columns=["Zone", "Count"]
        )
        fig = px.bar(df_zones, x="Zone", y="Count",
                     color="Count", color_continuous_scale="Reds",
                     title="People per Zone")
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 – Video Analysis
# ═══════════════════════════════════════════════════════════════════════════════
with tab_vid:
    st.header("Video Analysis")
    vid_file = st.file_uploader(
        "Upload a video clip",
        type=["mp4", "avi", "mov", "mkv"],
        key="vid_upload",
    )
    max_frames = st.slider("Max frames to process", 30, 500, 150, 10)

    if vid_file and st.button("▶ Process Video"):
        pipe.reset_heatmap()

        # Save to temp file
        suffix = os.path.splitext(vid_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(vid_file.read())
            tmp_path = tmp.name

        frame_placeholder = st.empty()
        heat_placeholder  = st.empty()
        progress          = st.progress(0)
        status_txt        = st.empty()

        all_totals   = []
        all_times    = []
        frame_count  = 0

        cap = cv2.VideoCapture(tmp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or max_frames
        cap.release()

        for result in pipe.process_video(tmp_path, show_heatmap=show_heatmap):
            frame_count += 1
            all_totals.append(result["total"])
            all_times.append(frame_count)

            if frame_count % 5 == 0 or frame_count == 1:
                col_a, col_b = frame_placeholder.columns(2)
                col_a.image(
                    cv2.cvtColor(result["annotated_frame"], cv2.COLOR_BGR2RGB),
                    caption=f"Frame {frame_count} | Total: {result['total']} | {result['crowd_level']}",
                    use_container_width=True,
                )
                col_b.image(
                    cv2.cvtColor(result["heatmap_snapshot"], cv2.COLOR_BGR2RGB),
                    caption="Heatmap",
                    use_container_width=True,
                )

            progress.progress(min(frame_count / max(total_frames, 1), 1.0))
            status_txt.text(
                f"Frame {frame_count} | Persons: {result['total']} | Level: {result['crowd_level']}"
            )
            if frame_count >= max_frames:
                break

        os.unlink(tmp_path)
        st.success(f"Processed {frame_count} frames.")

        # Time-series chart
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=all_times, y=all_totals,
                                  mode="lines+markers", name="Total Persons",
                                  line=dict(color="#3498db")))
        fig2.update_layout(title="Person Count Over Frames",
                           xaxis_title="Frame", yaxis_title="Count")
        st.plotly_chart(fig2, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 – Webcam Live (Streamlit camera_input)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_cam:
    st.header("📷 Webcam Snapshot Analysis")
    st.info(
        "Click **Take Photo** to capture a frame from your webcam. "
        "The pipeline will analyse it immediately. "
        "Re-click to refresh."
    )
    cam_img = st.camera_input("Take Photo")

    if cam_img:
        pil_cam = Image.open(cam_img).convert("RGB")
        cam_frame = cv2.cvtColor(np.array(pil_cam), cv2.COLOR_RGB2BGR)
        cam_result = pipe.process_frame(cam_frame, show_heatmap=show_heatmap)

        c1, c2 = st.columns(2)
        with c1:
            st.image(cv2.cvtColor(cam_result["annotated_frame"], cv2.COLOR_BGR2RGB),
                     caption="Detected Persons", use_container_width=True)
        with c2:
            st.image(cv2.cvtColor(cam_result["heatmap_snapshot"], cv2.COLOR_BGR2RGB),
                     caption="Heatmap", use_container_width=True)

        st.metric("Total Persons", cam_result["total"])
        st.metric("Crowd Level",   cam_result["crowd_level"])

        df_cam = pd.DataFrame(
            cam_result["zone_counts"].items(), columns=["Zone", "Count"]
        )
        st.dataframe(df_cam, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 – Analytics
# ═══════════════════════════════════════════════════════════════════════════════
with tab_stats:
    st.header("📊 Occupancy Analytics")

    if os.path.isfile(log_path) and os.path.getsize(log_path) > 0:
        df = pd.read_csv(log_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["total"]     = pd.to_numeric(df["total"], errors="coerce").fillna(0).astype(int)
        df = df.dropna(subset=["timestamp"])

        zone_cols = [c for c in df.columns
                     if c not in ("timestamp", "total", "crowd_level")]
        for col in zone_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

        st.markdown(f"**{len(df)} log entries** from `{log_path}`")

        # ── Summary metrics ───────────────────────────────────────────────────
        c1, c2, c3 = st.columns(3)
        c1.metric("Peak Count",   int(df["total"].max()))
        c2.metric("Average Count", f"{df['total'].mean():.1f}")
        c3.metric("Sessions Logged", len(df))
        st.divider()

        # ── 1. Total persons over time (native st chart – always renders) ─────
        st.subheader("👤 Total Persons Over Time")
        st.line_chart(df.set_index("timestamp")["total"])

        # ── 2. Crowd level counts ─────────────────────────────────────────────
        st.subheader("🚦 Crowd Level Breakdown")
        level_counts = df["crowd_level"].value_counts().reset_index()
        level_counts.columns = ["Level", "Count"]
        st.bar_chart(level_counts.set_index("Level"))

        # ── 3. Zone counts over time ──────────────────────────────────────────
        if zone_cols:
            st.subheader("📍 Zone Occupancy Over Time")
            st.area_chart(df.set_index("timestamp")[zone_cols])

        # ── 4. Hourly average ─────────────────────────────────────────────────
        st.subheader("🕐 Average Occupancy by Hour of Day")
        df["hour"] = df["timestamp"].dt.hour.astype(int)
        hourly = df.groupby("hour")["total"].mean().rename("Avg Persons")
        st.bar_chart(hourly)

        # ── Raw data + download ───────────────────────────────────────────────
        st.divider()
        st.subheader("📋 Raw Log Data")
        st.dataframe(df.drop(columns=["hour"], errors="ignore").tail(100),
                     use_container_width=True)
        csv_bytes = df.drop(columns=["hour"], errors="ignore").to_csv(index=False).encode()
        st.download_button("⬇️ Download CSV", csv_bytes,
                           file_name="occupancy_log.csv", mime="text/csv")
    else:
        st.warning("No log data yet. Run an image or video analysis first.")

