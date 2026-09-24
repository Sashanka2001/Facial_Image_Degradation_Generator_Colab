# Facial Image Dataset Generation & Synthesized Degradation Engine

A modular Python pipeline (`Google_Col_pre_enhancement_facial_recoverability.ipynb`) designed to construct a standardized dataset of high-resolution facial images, apply AI-based background removal, synthesize controlled single and multi-factor image degradations, and compile a master metadata registry.

---

## Overview

This repository handles **Phase 1** of the pre-enhancement facial recoverability pipeline:

1. **Face Dataset Acquisition:** Ingests high-resolution Flickr-Faces-HQ (FFHQ) images directly into persistent Google Drive storage.
2. **AI Background Removal:** Employs `rembg` (U-2-Net model) to isolate facial regions and composite them on solid white backgrounds (`face_XXXX_bgr.png`).
3. **Controlled Degradation Synthesis:** Applies 7 categories of image degradation across mathematically controlled parameter ranges.
4. **Master Metadata Registry:** Indexing script parses generated assets to build `metadata.csv` linking original faces, cropped cutouts, degradation types, severity parameters, and storage paths.
5. **Exploratory Data Analysis:** Generates class distribution charts and sample visual comparison grids saved locally and backed up to Google Drive.

---

## Degradation Parameters Matrix

The engine generates **32 degraded variants per facial cutout** across 7 distinct categories:

| Category            | Degradation Type    | Parameter Name                   | Values / Tiers                                                                                | Variants per Face |
| ------------------- | ------------------- | -------------------------------- | --------------------------------------------------------------------------------------------- | ----------------- |
| **1. Sharpness**    | Gaussian Blur       | Kernel Sigma ($\sigma$)          | `1`, `2`, `3`, `5`, `7`                                                                       | 5                 |
| **2. Noise**        | Gaussian Noise      | Std Deviation ($\sigma_{noise}$) | `5`, `10`, `15`, `20`, `30`                                                                   | 5                 |
| **3. Luminance**    | Brightness Shift    | Scale Factor ($\alpha$)          | `0.6`, `0.8`, `1.2`, `1.4`                                                                    | 4                 |
| **4. Dynamics**     | Contrast Adjustment | Scale Factor ($\beta$)           | `0.6`, `0.8`, `1.2`, `1.5`                                                                    | 4                 |
| **5. Compression**  | JPEG Compression    | Quality Factor ($Q$)             | `95`, `80`, `60`, `40`, `20`                                                                  | 5                 |
| **6. Resolution**   | Low Resolution      | Dimension ($D \times D$)         | `256`, `128`, `64`                                                                            | 3                 |
| **7. Multi-Factor** | Combined Presets    | Compound Settings                | `Blur+Noise`, `Blur+JPEG`, `Noise+JPEG`, `LowRes+Blur`, `LowRes+Noise`, `Severe Multi-Factor` | 6                 |
| **TOTAL**           | **7 Categories**    | —                                | —                                                                                             | **32 Variants**   |

---

## Repository & Directory Architecture

```text
pre_enhancement_facial_recoverability/
│
├── dataset/
│   ├── original/                                           # FFHQ Original Faces (face_0001.png)
│   ├── cropped/                                            # Background-removed Cutouts (face_0001_bgr.png)
│   ├── metadata/
│   │   └── metadata.csv                                    # Synchronized Master Registry
│   └── degraded/
│       ├── blur/                                           # Gaussian Blur variants
│       ├── noise/                                          # Gaussian Noise variants
│       ├── brightness/                                     # Brightness Shift variants
│       ├── contrast/                                       # Contrast Adjustment variants
│       ├── jpeg/                                           # JPEG Compression variants
│       ├── low_resolution/                                 # Downsampled/Upscaled variants
│       └── combined/                                       # Multi-Factor Preset variants
│
└── graphs/
    ├── degradation_distribution.png                        # Dataset composition chart
    └── sample_degradations_grid.png                        # Visual comparison grid

```

---

## Installation & Dependencies

### Required Packages

```bash
pip install mediapipe pandas matplotlib scikit-image tqdm datasets kagglehub
pip install -U --prefer-binary "rembg[cpu]" onnxruntime pillow opencv-python-headless

```

---

## Execution Steps

### Step 1: Environment & Directory Setup

Mount Google Drive and initialize local Colab and Drive directory structures:

```python
import os
from google.colab import drive

drive.mount('/content/drive')

```

### Step 2: Download FFHQ Face Dataset

Ingest raw face images directly into Drive via `kagglehub`:

```python
# Batch download faces 1 to 1000 directly to Drive
download_ffhq_straight_to_drive(output_dir=DIR_ORIGINAL, start_index=1, end_index=1000)

```

### Step 3: AI Background Removal

Process original images using `rembg` with solid white background compositing:

```python
segment_faces_directly_to_drive(
    input_dir=DIR_ORIGINAL,
    output_dir=DIR_CROPPED,
    background_color=(255, 255, 255),
    start_index=1,
    end_index=1000
)

```

### Step 4: Generate Synthesized Degradations

Execute the 7 degradation generators across the cutout images:

```python
blur_records = generate_blur_degradations(start_index=1, end_index=1000)
noise_records = generate_noise_degradations(start_index=1, end_index=1000)
bright_records = generate_brightness_degradations(start_index=1, end_index=1000)
contrast_records = generate_contrast_degradations(start_index=1, end_index=1000)
jpeg_records = generate_jpeg_degradations(start_index=1, end_index=1000)
lowres_records = generate_lowres_degradations(start_index=1, end_index=1000)
combined_records = generate_combined_degradations(start_index=1, end_index=1000)

```

### Step 5: Rebuild Master Metadata Registry

Scan Google Drive output folders to generate `metadata.csv`:

```python
# Rebuilds and synchronizes metadata.csv across local and Drive paths
df_rebuilt.to_csv("/content/drive/MyDrive/pre_enhancement_facial_recoverability/dataset/metadata/metadata.csv", index=False)

```

### Step 6: Dataset Visualization & Quality Inspection

Generate class distribution counts and export visual sample grid comparisons for verification:

* `graphs/degradation_distribution.png`
* `graphs/sample_degradations_grid.png`

---

## Dataset Output Registry Schema (`metadata.csv`)

| Column Name          | Type         | Description                         | Example                                                     |
| -------------------- | ------------ | ----------------------------------- | ----------------------------------------------------------- |
| `original_filename`  | String       | Source FFHQ face reference          | `face_0800.png`                                             |
| `degraded_filename`  | String       | Synthesized degraded image filename | `face_0800_bgr_blur_s3.png`                                 |
| `degradation_type`   | String       | Formal degradation classification   | `Gaussian Blur`                                             |
| `severity_parameter` | String       | Parameter name applied              | `sigma`                                                     |
| `parameter_value`    | String/Float | Value or setting used               | `3.0`                                                       |
| `drive_path`         | String       | Absolute path on Google Drive       | `/content/drive/MyDrive/.../blur/face_0800_bgr_blur_s3.png` |
