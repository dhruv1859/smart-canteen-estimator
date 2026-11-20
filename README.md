# Smart Canteen Crowd & Queue Estimator

> **Course:** Computer Vision Workshop (24B15CS352) · 5th Semester · Session 2026-27  
> **Members:** Dhruv Agrawal (2401010006) · Vrinda Wasan (2401010013) · Shivi Goel (2401010058)  
> **Lab Instructors:** Prof. Shikha Mehta · Ms. Monika  

---

## 📌 Project Overview

A **real-time computer vision system** that analyses a canteen camera feed (image, video, or webcam) to:

| Feature | Detail |
|---------|--------|
| 👤 **Person Detection** | YOLOv8 (nano/small) pretrained on COCO |
| 📍 **Zone Counting** | Seating Area · Serving Counter · Billing Counter · Queue/Exit |
| 🌡️ **Crowd-Density Heatmap** | Gaussian accumulator with temporal decay |
| 📊 **Occupancy Classification** | Low / Moderate / High using configurable thresholds |
| 📈 **Time-Series Logging** | CSV log → Plotly charts in Streamlit |
| 🌐 **Web Dashboard** | Streamlit app with Image / Video / Webcam / Analytics tabs |

---

## 🗂️ Project Structure

```
smart-canteen-estimator/
│
├── app.py                  # Streamlit web app (4 tabs)
├── evaluate.py             # Offline precision/recall evaluation
├── requirements.txt        # Python dependencies
│
├── src/
│   ├── __init__.py
│   ├── config.py           # Central config (model, zones, thresholds)
│   ├── preprocessor.py     # Resize · CLAHE · Gaussian blur
│   ├── detector.py         # YOLOv8 wrapper + IoU evaluation
│   ├── zone_manager.py     # Polygon zones + crowd level
│   ├── heatmap.py          # Gaussian accumulator heatmap
│   ├── logger.py           # CSV occupancy logger
│   └── pipeline.py         # End-to-end pipeline (image / video / webcam)
│
├── tests/
│   ├── test_preprocessor.py
│   ├── test_zone_manager.py
│   ├── test_heatmap.py
│   └── test_detector.py
│
├── data/
│   └── sample_frames/      # Put test images here for evaluate.py
│
├── logs/
│   └── occupancy_log.csv   # Auto-generated during runtime
│
└── assets/
    └── logo.png            # (Optional) sidebar logo
```

---

## ⚡ Quick Start

### 1. Install dependencies

```bash
# Create and activate virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

pip install -r requirements.txt
```

### 2. Run the Streamlit dashboard

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

### 3. Run unit tests

```bash
pytest tests/ -v
```

### 4. Offline evaluation (optional)

```bash
python evaluate.py --gt data/gt_labels.csv --images data/sample_frames/
```

---

## 🔧 Configuration

Edit **`src/config.py`** to customise everything:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `YOLO_MODEL` | `yolov8n.pt` | Model weights (nano/small/medium) |
| `YOLO_CONF` | `0.35` | Detection confidence threshold |
| `DEFAULT_ZONES` | 4 zones | Polygon coords for your camera |
| `CROWD_THRESHOLDS` | Low≤5, Mod≤15, High>15 | Crowd classification |
| `HEATMAP_DECAY` | `0.98` | Frame-to-frame heatmap fade |
| `LOG_INTERVAL_S` | `5` | Seconds between CSV log entries |

### Customising Zones

Zones are polygons in pixel coordinates matching your 640×480 frame. Edit `DEFAULT_ZONES` in `config.py`:

```python
DEFAULT_ZONES = [
    {
        "name"  : "My Zone",
        "color" : (0, 200, 0),               # BGR
        "points": [(10,10),(300,10),(300,230),(10,230)],
    },
    ...
]
```

---

## 🧰 Technology Stack

| Tool | Purpose |
|------|---------|
| **YOLOv8** (Ultralytics) | Person detection (class 0, pretrained) |
| **OpenCV** | Frame I/O, preprocessing, drawing |
| **NumPy / Pandas** | Array ops, CSV handling |
| **Matplotlib / Plotly** | Charts in dashboard |
| **Streamlit** | Web UI |

---

## 📐 Objectives Checklist

- [x] Preprocess frames (resize, colour convert, noise reduction, contrast enhancement)
- [x] Detect persons using pretrained YOLOv8n (no training required)
- [x] Count persons per zone using `cv2.pointPolygonTest`
- [x] Classify crowd level (Low / Moderate / High)
- [x] Generate crowd-density heatmap with Gaussian accumulator
- [x] Time-based occupancy logging (CSV)
- [x] Streamlit dashboard (image · video · webcam · analytics tabs)
- [x] Offline precision / recall / F1 evaluation script
- [x] Unit test suite (preprocessor, detector, zones, heatmap)
- [ ] Fine-tune on labelled canteen images *(optional / time-permitting)*

---

## 📝 Notes

- The first run will auto-download `yolov8n.pt` (~6 MB) from Ultralytics.
- For GPU acceleration, install the CUDA version of PyTorch before `ultralytics`.
- Zone polygons are configured for a 640×480 frame; adjust for your camera resolution.

