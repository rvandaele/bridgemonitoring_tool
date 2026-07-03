# Bridge Camera Monitor

A system for monitoring bridge cameras: a deep learning script runs debris and water detection on images, and a Streamlit dashboard displays the results.

---

## Components

- **`prediction_bridges.py`** — runs a segmentation model over camera images and writes per-camera CSV files with pixel counts. Run once for debris detection and once for water detection, each with its own model weights.
- **`streamlit_app.py`** — a Streamlit dashboard for browsing camera images and reviewing automatically computed debris/water status.

---

## Setup

### Requirements

```bash
pip install streamlit pandas imageio torch torchvision segmentation-models-pytorch opencv-python numpy
```

---

## Running the Debris & Water Detection

The detection script runs a segmentation model over all camera folders and appends results to per-camera CSV files. It must be run **twice** — once for debris and once for water — using different model checkpoints each time. If left running, these scripts will continuously monitor the image repository for new images and update the CSV files.

**Debris detection:**

```bash
python prediction_bridges.py \\
  --checkpoint ./weights/debris_model.pt \\
  --input-dir ./images \\
  --csv-dir ./logs/debris \\
  --use-tiling
```

**Water detection:**

```bash
python prediction_bridges.py \\
  --checkpoint ./weights/water_model.pt \\
  --input-dir ./images \\
  --csv-dir ./logs/water \\
  --use-tiling
```

| Argument | Default | Description |
|---|---|---|
| `--checkpoint` | *(required)* | Path to the model weights file (`.pt`). Default should be model_weights/debris_model.pt or water_model.pt |
| `--input-dir` | *(required)* | Parent folder containing `reolink_*` subfolders. Default should be images |
| `--csv-dir` | *(required)* | Folder where per-camera CSV result files are written. Default should be debris_csv or water_csv |
| `--device` | `cuda` | `cuda` or `cpu` — falls back to CPU automatically if no GPU is available |
| `--threshold` | from checkpoint | Override the detection probability threshold (0–1) |
| `--use-tiling` | off | Enable tiled inference, recommended for high-resolution images |
| `--tile-size` | `256` | Tile size in pixels (used with `--use-tiling`) |
| `--stride` | `192` | Stride between tiles in pixels (used with `--use-tiling`) |
| `--poll-seconds` | `60` | Seconds to wait between scans of the camera folders |

The script runs continuously, polling for new images every 60 seconds and skipping any already recorded in the CSV. Each CSV row contains the image path, creation timestamp, and the number of pixels classified as debris or water.

> **Note:** Point `--csv-dir` to the same path configured as **Debris CSVs repository** in the dashboard sidebar.

---

## Downloading the models
---

The debris and water segmentation models can be downloaded on [Hugging Face](https://huggingface.co/rvandaele/bridgemonitoring)

---

## Running the Dashboard

```bash
streamlit run app.py
```

In the sidebar, set:
- **Camera repository path** — the folder containing one subfolder per camera (i.e. your `--output-dir` from above)
- **Debris CSVs repository** — the folder containing CSV files output by the debris detection algorithm

---

## Camera Folder Structure

```
images/
├── reolink_0/
├── reolink_1/
├── reolink_3/
...
```

Each subfolder corresponds to one camera.

---

## Masks

Per-camera mask images are placed in the masks directory and named `reolink_N.png` (e.g. `reolink_0.png`). White pixels in the mask are blacked out in saved images to hide personally identifiable areas. The privacy mask must be updated when the camera is moved.

---

## Authors

Developed by the University of Exeter:

- **Remy Vandaele**
- **Prakash Kripakaran**
- **Diego Panici**

---

## License

This project is licensed under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

You are free to share and adapt this work for any purpose, provided appropriate credit is given to the authors and the University of Exeter.
