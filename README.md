# Bridge Camera Monitor

A system for monitoring bridge cameras: a GMail script extracts images from incoming emails, a deep learning script runs debris and water detection on those images, and a Streamlit dashboard displays the results.

---

## Components

- **`gmail_extractor.py`** — polls a GMail inbox, extracts image attachments, applies privacy masks, and saves images into per-camera folders.
- **`prediction_bridges.py`** — runs a segmentation model over camera images and writes per-camera CSV files with pixel counts. Run once for debris detection and once for water detection, each with its own model weights.
- **`streamlit_app.py`** — a Streamlit dashboard for browsing camera images and reviewing automatically computed debris/water status.

---

## Setup

### Requirements

```bash
pip install streamlit pandas imageio google-api-python-client google-auth-oauthlib \
            torch torchvision segmentation-models-pytorch opencv-python numpy
```

---

## Gmail Credentials Setup

The Gmail script authenticates via OAuth 2.0. You will need to generate a `credentials.json` file from the Google Cloud Console using the Bridge Monitoring GMail address available in the handover document.

### 1. Create or open a Google Cloud project

Using the GMail account, go to [console.cloud.google.com](https://console.cloud.google.com) and create a new project, or select an existing one.

### 2. Enable the Gmail API

- In the sidebar, go to **APIs & Services → Library**
- Search for **Gmail API** and click **Enable**

### 3. Configure the OAuth consent screen

- Go to **APIs & Services → OAuth consent screen**
- Choose **External** (or **Internal** if using a Google Workspace account)
- Fill in the required fields (app name, support email)
- Under **Scopes**, add `https://www.googleapis.com/auth/gmail.modify`
- Add the Gmail account as a **test user** if staying in test mode

### 4. Create OAuth credentials

- Go to **APIs & Services → Credentials**
- Click **Create Credentials → OAuth client ID**
- Choose **Desktop app** as the application type
- Give it a name and click **Create**
- Download the JSON file and save it as `credentials.json` in the project root

### 5. First run & token

On first run, the script will open a browser window asking you to log in and grant access to the Gmail account. After authorising, a `token.pickle` file is saved locally — subsequent runs will use this automatically without re-prompting.

> **Note:** If an existing `token.pickle` is available from a previous setup, it can be handed over directly to skip the browser authorisation step.

> **Note:** If the OAuth consent screen is left in **test mode**, tokens expire every 7 days and require re-authorisation. To avoid this, either submit the app for Google verification, or switch to **Internal** if the account is part of a Google Workspace organisation.

---

## Running the Gmail Extractor

```bash
python gmail_extractor.py \
  --output-dir ./images \
  --credentials credentials.json \
  --mask-dir ./masks
```

| Argument | Default | Description |
|---|---|---|
| `--output-dir` | `./images` | Base directory where per-camera subfolders and images are stored |
| `--credentials` | `credentials.json` | Path to the Gmail OAuth credentials file |
| `--mask-dir` | `./masks` | Path to per-camera mask images used to hide personal data |

The script polls the inbox every 60 seconds, saves new images into the appropriate camera subfolder, applies privacy masks, and moves processed emails to trash.

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

Each subfolder corresponds to one camera and is populated automatically by the Gmail extractor.

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
