# 🏠 Applied AI Engineer — Case Study: Problem Analysis & Best Approach

---

## 🎯 What Is This Problem, In Plain English?

> **Build a pipeline that takes smartphone sensor data (photos / video / LiDAR) and outputs a fully dimensioned floor plan with damage detection — automatically.**

Think of it like building the backend engine of **Magicplan** or **Poly.cam**, but from scratch.

A user walks through a property with their iPhone. Your system:
1. Ingests the raw sensor data
2. Reconstructs the 3D geometry of all rooms
3. Detects damage on surfaces
4. Outputs a professional floor plan (JSON + rendered image) with measurements and confidence intervals

---

## 📋 The 5 Parts Broken Down

### Part 1 — Capture Pipeline (3 Tiers, All Mandatory)

You don't provide the capture app — you define HOW to capture:

| Tier | Input | Difficulty | Accuracy |
|------|-------|-----------|---------|
| **Photos** | 2–8 stills per room, no depth, no poses | Hardest | Widest intervals (±8%) |
| **Video** | Handheld walkthrough clip, iPhone 15+ | Medium | ±3% |
| **LiDAR** | Depth maps + poses + camera intrinsics | Easiest | Tightest intervals |

**Two Route Choices:**
- **Route 1**: Build your own iOS app (ARKit/RoomPlan/LiDAR SDK) → TestFlight
- **Route 2 (Recommended for this assignment)**: Use an existing app (like `RECORD 3D`, `Scandy Pro`, `StrayScanner`) + write a one-page capture protocol

---

### Part 2 — Output Contract (What You Must Produce Per Capture)

Every single capture must output:
- ✅ **Dimensioned per-room plan** (walls, ceiling height, floor area, openings like doors/windows)
- ✅ **Stitched multi-room plan** (all rooms connected correctly = a real floor plan)
- ✅ **Per-surface damage regions** (class + metric size, e.g., "crack, 0.3m²")
- ✅ **Concealed damage flags** (logical rules, e.g., "stain near pipe = possible leak behind wall")
- ✅ **Scope line items** (repair items keyed to surfaces)
- ✅ **Confidence intervals on every measurement**
- ✅ **One command to run the full pipeline**
- ✅ **JSON output** (to a published schema)
- ✅ **Rendered floor plan image**

---

### Part 3 — Head-to-Head Comparison

Compare YOUR output vs. a consumer app (Magicplan / Poly.cam free tier) on 2 rooms.
Must **beat or tie on ≥70% of shared dimensions**.

---

### Part 4 — The Fix Loop (25% of Score!)

1. Find your worst-performing gate
2. Explain WHY it fails (root cause)
3. Fix it, show before/after results
4. Submit regenerable runs + readable diff

> ⚠️ **No shipped fix = zero marks.** Analysis without implementation counts for nothing.

---

### Part 5 — Process Evidence

- **Commit regularly** as you work (multiple commits per day)
- A repo that appears fully formed in 1–2 commits scores **zero** on this part
- AI tools are allowed — but you defend your design decisions live

---

## 📊 Scoring Breakdown

| Weight | Component |
|--------|-----------|
| **30%** | Walk-in test: cold run on THEIR capture, scored live with laser |
| **25%** | Fix loop delta |
| **15%** | Benchmark accuracy (all 3 tiers) |
| **10%** | Compliance matrix |
| **10%** | Head-to-head vs Magicplan/Poly.cam |
| **5%** | Capture route quality |
| **5%** | Process evidence (commit history) |

---

## 🗂️ What the Sample Data Contains

```
Dataset/
├── single_scan_floor_only/
│   └── 1a8384c3f6/
│       ├── imu.csv           ← Accelerometer + Gyroscope (11,399 rows @ ~100Hz)
│       ├── odometry.csv      ← Camera pose + position + quaternion + focal length (5,252 rows)
│       ├── camera_matrix.csv ← 3x3 intrinsic matrix (fx=1598.79, fy=1598.79, cx=955.39, cy=717.76)
│       ├── depth/            ← Per-frame depth maps as PNG (1,200+ frames)
│       ├── confidence/       ← Per-pixel depth confidence maps
│       └── rgb.mp4           ← Color video (134 MB)
├── single_scan_with_ceiling/
└── single_room/
```

**Key insight from data:**
- `odometry.csv` columns: `timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy`
  - `(x,y,z)` = camera position in world coordinates
  - `(qx,qy,qz,qw)` = camera orientation as quaternion
  - `fx, fy, cx, cy` = per-frame camera intrinsics
- `imu.csv` columns: `timestamp, a_x, a_y, a_z, alpha_x, alpha_y, alpha_z`
  - Linear acceleration + angular velocity (for drift correction)
- Depth PNGs = LiDAR depth images (16-bit, values in mm)

---

## 🏗️ Best Approach Architecture

### Pipeline Overview

```
Input (Photos / Video / LiDAR)
        ↓
[Tier 1] Feature Matching + Monocular Depth (MiDaS/Depth-Anything)
[Tier 2] VO/SfM from Video (COLMAP or ORB-SLAM3)
[Tier 3] Fuse LiDAR depth + Odometry poses → Point Cloud
        ↓
Point Cloud Reconstruction
        ↓
Floor Plan Extraction (RANSAC plane detection → wall detection)
        ↓
Room Segmentation & Stitching
        ↓
Damage Detection (fine-tuned ViT / YOLO on damage dataset)
        ↓
JSON Output + Floor Plan Rendering
        ↓
Confidence Estimation
```

---

### Tier-by-Tier Strategy

#### 🔵 Tier 3 (LiDAR) — Start Here (what the sample data is!)

**You already have this data.** The sample data IS the LiDAR tier.

**Steps:**
1. Load `odometry.csv` → get camera pose per frame `(x, y, z, qx, qy, qz, qw)`
2. Load depth PNGs → convert to 3D points using camera matrix (backprojection)
3. Transform each frame's point cloud into world coordinates using the pose
4. Merge all frames → global 3D point cloud
5. Apply RANSAC to find floor plane → define coordinate system
6. Detect walls using vertical plane fitting (RANSAC on vertical planes)
7. Extract room boundary as 2D polygon
8. Detect openings (gaps in walls) = doors/windows

**Key libraries:** `open3d`, `numpy`, `scipy`, `opencv`

#### 🟡 Tier 2 (Video)

**Steps:**
1. Run COLMAP or ORB-SLAM3 on the video to get poses
2. Use those poses + MiDaS depth estimates to build pseudo-LiDAR
3. Continue same pipeline as Tier 3

**Key libraries:** `colmap`, `ORB-SLAM3`, `torch` (MiDaS)

#### 🟢 Tier 1 (Photos)

**Steps:**
1. Run image feature matching (SuperGlue/LightGlue) across room photos
2. Estimate camera poses via SfM
3. Use Depth-Anything v2 for monocular depth
4. Same pipeline with wider confidence intervals

**Key libraries:** `lightglue`, `depth-anything-v2`, `colmap`

---

### Floor Plan Extraction Algorithm

```python
# Pseudocode for wall extraction
1. Filter point cloud to floor-level band (z ≈ 0 to ceiling_height)
2. Project to 2D (XY plane = top-down view)
3. Create occupancy grid (2D histogram of point density)
4. Apply morphological operations → detect wall boundaries
5. Fit line segments to wall boundaries (Hough transform or RANSAC)
6. Find corner points → room polygon
7. Detect gaps in walls → openings (doors/windows)
8. Measure: wall lengths, room area, ceiling height, opening widths
```

---

### Damage Detection

- Fine-tune **YOLO v8** or **Segment Anything Model (SAM)** on damage categories:
  - Cracks, water stains, mold, peeling paint, structural damage
- Per-surface = map detected damage back to the 3D plane it belongs to
- Concealed damage = rule-based flags (e.g., stain near plumbing route)

---

### Multi-Room Stitching

- Use overlapping point clouds (ICP alignment) between adjacent rooms
- Alternatively, use loop closure from odometry (IMU helps here)
- Graph-based room adjacency → shared walls = connections

---

## ✅ What to Build for THIS Assessment (48h scope)

Given the sample data provided and the 48-hour deadline, here's the realistic priority order:

| Priority | Task | Why |
|----------|------|-----|
| **P0** | LiDAR pipeline (Tier 3) using sample data | Data is already there, highest accuracy |
| **P0** | Floor plan extraction + JSON output | Core deliverable |
| **P1** | Room polygon + measurements + confidence | Scored directly |
| **P1** | Basic damage detection | Required in output contract |
| **P2** | Video tier (Tier 2) | Need COLMAP/ORB-SLAM |
| **P3** | Photo tier (Tier 1) | Hardest, monocular depth |
| **P3** | Fix loop | Needs working pipeline first |

---

## 🔧 Recommended Tech Stack

```
Language:     Python 3.10+
3D Processing: open3d, numpy, scipy
Computer Vision: opencv-python, scikit-image
Deep Learning: torch, transformers (Depth-Anything, SAM)
Feature Matching: kornia, lightglue
Floor Plan: shapely (2D geometry), matplotlib (rendering)
SLAM/SfM: pycolmap, or hloc (hierarchical localization)
Output: json, Pillow (SVG/PNG floor plan rendering)
```

---

## ⚠️ Key Gates You Must Pass

| Gate | Threshold |
|------|-----------|
| Opening widths | ±2 cm on ≥85% of openings |
| Ceiling height | ±1.5 cm per room |
| Repeatability | Same room → same plan within 1 cm or 0.5% per wall |
| Drift correction | Must show ablation (with/without loop closure) |
| Photo-tier stitch | ±8% wall lengths with calibrated intervals |
| Video-tier | ±3% wall lengths |

---

## 📌 Summary

> This is a **3D indoor mapping + damage assessment system** built from smartphone sensor data.
> The sample data gives you **LiDAR depth maps + camera poses + IMU** for a real room scan.
> Your goal: convert that into a professional floor plan with measurements, damage annotations, and confidence intervals — automatically, from one command.

