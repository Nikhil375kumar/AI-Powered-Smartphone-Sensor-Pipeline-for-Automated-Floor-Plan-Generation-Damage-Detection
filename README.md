# 🏠 Indoor Mapping Pipeline

> **Automated indoor scanning pipeline — smartphone sensor data → dimensioned floor plan + damage report.**

Supports 3 input tiers (LiDAR, Video, Photo) and produces:
- Dimensioned per-room floor plan (walls, ceiling height, floor area, openings)
- Stitched multi-room whole-property plan
- Per-surface damage detection & concealed-damage flags
- Confidence intervals on every measurement
- JSON output conforming to a published schema + rendered PNG

---

## 📁 Project Structure

```
.
├── config/
│   ├── default.yaml          # All tunable parameters (no hardcoding in source)
│   └── output_schema.json    # Published JSON output schema
├── src/
│   ├── config.py             # Singleton config loader (YAML + .env)
│   ├── capture/
│   │   ├── data_loader.py    # Loads IMU, odometry, depth, RGB
│   │   └── validator.py      # Fail-fast input validation per tier
│   ├── reconstruction/
│   │   ├── point_cloud.py    # Depth + pose → global 3D point cloud
│   │   ├── plane_detection.py# RANSAC floor/ceiling/wall detection
│   │   ├── drift_correction.py # ICP drift refinement + ablation helper
│   │   └── monocular_depth.py# MiDaS wrapper for Video/Photo tiers
│   ├── floor_plan/
│   │   ├── wall_extractor.py # Occupancy-grid wall + opening detection
│   │   ├── stitcher.py       # Multi-room adjacency graph + polygon union
│   │   └── renderer.py       # Matplotlib floor plan renderer
│   ├── damage/
│   │   ├── detector.py       # YOLOv8 damage detection
│   │   └── rules.py          # Config-driven concealed damage rules
│   ├── output/
│   │   ├── schema.py         # Pydantic v2 output models
│   │   ├── confidence.py     # Tier-aware confidence intervals
│   │   └── exporter.py       # Assembles + writes JSON + metadata
│   └── pipeline/
│       ├── base.py           # Abstract pipeline template
│       ├── lidar_pipeline.py # Tier 3: LiDAR (depth + poses)
│       ├── video_pipeline.py # Tier 2: Video (ORB-VO + MiDaS)
│       └── photo_pipeline.py # Tier 1: Photo (SfM + Depth-Anything)
├── scripts/
│   └── run_pipeline.py       # ← ONE COMMAND TO RUN EVERYTHING
├── tests/
│   ├── test_geometry.py
│   ├── test_data_loader.py
│   └── test_confidence.py
├── .env.example              # Template for secrets/overrides
├── .gitignore
├── requirements.txt
└── pyproject.toml
```

---

## ⚡ Quick Start (Under 15 Minutes on a Clean Machine)

### 1. Clone & Setup

```bash
git clone <your-repo-url>
cd Assignment

# Create virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy the template and fill in any optional API keys
copy .env.example .env    # Windows
# cp .env.example .env   # Linux/Mac
```

### 3. Run the Pipeline

#### 🔵 Tier 3 — LiDAR (Sample Data, Highest Accuracy)
```bash
python scripts/run_pipeline.py \
  --scan-dir "Dataset/single_scan_floor_only/1a8384c3f6" \
  --tier lidar
```

#### 🟡 Tier 2 — Video
```bash
python scripts/run_pipeline.py \
  --scan-dir "Dataset/my_video_scan" \
  --tier video
```

#### 🟢 Tier 1 — Photo
```bash
python scripts/run_pipeline.py \
  --scan-dir "Dataset/my_house_photos" \
  --tier photo
```

#### Ablation (no drift correction)
```bash
python scripts/run_pipeline.py \
  --scan-dir "Dataset/single_scan_floor_only/1a8384c3f6" \
  --tier lidar \
  --no-drift-correction
```

### 4. Run Tests
```bash
pytest tests/ -v
```

---

## 📂 Output

After a successful run, outputs are written to:
```
outputs/<scan_id>/
  ├── output.json       ← Full pipeline output (validated against schema)
  └── floor_plan.png    ← Rendered dimensioned floor plan
```

---

## ⚙️ Configuration

All pipeline parameters live in [`config/default.yaml`](config/default.yaml).  
**Zero hardcoding in source code** — every threshold, path and tolerance is config-driven.

Environment variables can override any config value:
```bash
# Override RANSAC threshold at runtime
PLANE_DETECTION__DISTANCE_THRESHOLD=0.015 python scripts/run_pipeline.py ...

# Override output directory
OUTPUT_DIR=./results python scripts/run_pipeline.py ...
```

---

## 📐 Input Data Format (LiDAR Tier)

```
scan_dir/
  ├── imu.csv               # timestamp, a_x, a_y, a_z, alpha_x, alpha_y, alpha_z
  ├── odometry.csv          # timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy
  ├── camera_matrix.csv     # 3×3 intrinsic matrix
  ├── depth/
  │   ├── 000000.png        # 16-bit depth maps (raw value × 0.001 = metres)
  │   └── ...
  ├── confidence/           # (optional) per-pixel depth confidence (0/1/2)
  │   └── 000000.png
  └── rgb.mp4               # (optional) colour video for damage detection
```

---

## 🎯 Assessment Gate Reference

| Gate | Threshold | Where it's addressed |
|------|-----------|---------------------|
| Opening widths | ±2 cm on ≥85% | `wall_extractor.py::_find_gaps_in_wall` |
| Ceiling height | ±1.5 cm per room | `wall_extractor.py::_estimate_ceiling_height` |
| Repeatability | ≤1 cm or 0.5% per wall | Config `frame_stride`, `voxel_size` |
| Drift | ICP + ablation table | `drift_correction.py` |
| Photo-tier stitch | ±8% with calibrated intervals | `photo_pipeline.py` + `confidence.py` |
| Video-tier | ±3% | `video_pipeline.py` + `confidence.py` |

---

## 🤝 Capture Route (Route 2 — Stock Protocol)

**Recommended app:** `Record3D` (iOS, free tier sufficient)

1. Install `Record3D` from the App Store on iPhone 15 Pro or newer.
2. Open the app → tap **Record** → walk slowly through each room (1 m/s or slower).
3. Keep the phone pointed at walls, floor and ceiling in each room.
4. For multi-room captures, walk through doorways between rooms without pausing.
5. Export: tap Share → **Export ZIP** → transfer to your computer.
6. Unzip → point `--scan-dir` at the extracted folder.

**Device Matrix:**

| Tier | Device Requirement | Expected Wall Accuracy |
|------|-------------------|----------------------|
| LiDAR | iPhone 15 Pro / Pro Max | ±1% |
| Video | iPhone 15 or newer | ±3% |
| Photo | iPhone 15 or newer | ±8% |

---

## 📦 Model Weights

- **YOLOv8** (damage detection): downloaded automatically via `ultralytics` on first run.
- **MiDaS** (monocular depth): downloaded automatically via `torch.hub` on first run.
- Fine-tuned damage detector: place at `models/damage_detector.pt` (optional).

---

## 🧪 Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=term-missing
```

---

## 📄 License

MIT
