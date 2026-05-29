#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import os
import time

from datetime import datetime

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CAMERA_DIRS = [f"reolink_{i}" for i in range(9)]
POLL_SECONDS = 60


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continuous inference over 9 Reolink camera folders.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to best_model.pt")
    parser.add_argument("--input-dir", type=Path, required=True, help="Parent folder containing reolink_0 … reolink_8")
    parser.add_argument("--csv-dir", type=Path, required=True, help="Folder where per-camera CSV files are written")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--threshold", type=float, default=None, help="Override checkpoint threshold")
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=192)
    parser.add_argument("--use-tiling", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=POLL_SECONDS, help="Seconds to wait between scans")
    return parser.parse_args()


def list_images(folder: Path) -> List[Path]:
    return sorted(p for p in folder.glob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def load_existing_results(csv_path: Path) -> set:
    if not csv_path.exists():
        return set()
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        return {row["image_path"] for row in reader}


def append_result(csv_path: Path, image_path: Path, pixel_count: int) -> None:
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    ctime = datetime.fromtimestamp(os.path.getctime(image_path)).isoformat(timespec="seconds")
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "date", "pixel_count"])
        if write_header:
            writer.writeheader()
        writer.writerow({"image_path": str(image_path), "date": ctime, "pixel_count": pixel_count})


def get_model(arch: str, encoder: str, encoder_weights: str):
    arch_map = {
        "Unet": smp.Unet,
        "UnetPlusPlus": smp.UnetPlusPlus,
        "FPN": smp.FPN,
        "PAN": smp.PAN,
        "DeepLabV3": smp.DeepLabV3,
        "DeepLabV3Plus": smp.DeepLabV3Plus,
        "PSPNet": smp.PSPNet,
        "Linknet": smp.Linknet,
        "MAnet": smp.MAnet,
    }
    if arch not in arch_map:
        raise ValueError(f"Unsupported architecture '{arch}'. Supported: {list(arch_map)}")
    return arch_map[arch](encoder_name=encoder, encoder_weights=encoder_weights, in_channels=3, classes=1)


def preprocess_image(image_rgb: np.ndarray, encoder: str, encoder_weights: str) -> torch.Tensor:
    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weights)
    image = preprocessing_fn(image_rgb).astype(np.float32)
    image = np.transpose(image, (2, 0, 1))
    return torch.from_numpy(image).unsqueeze(0)


def pad_to_tile_size(image: np.ndarray, tile_size: int):
    h, w = image.shape[:2]
    pad_h = (tile_size - h % tile_size) % tile_size
    pad_w = (tile_size - w % tile_size) % tile_size
    if pad_h == 0 and pad_w == 0:
        return image, (0, 0)
    padded = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
    return padded, (pad_h, pad_w)


@torch.no_grad()
def predict_full_image(model, image_rgb, encoder, encoder_weights, device) -> np.ndarray:
    x = preprocess_image(image_rgb, encoder, encoder_weights).to(device)
    return torch.sigmoid(model(x))[0, 0].cpu().numpy()


@torch.no_grad()
def predict_tiled(model, image_rgb, encoder, encoder_weights, device, tile_size=256, stride=192) -> np.ndarray:
    if stride <= 0 or stride > tile_size:
        raise ValueError("stride must be in the range 1..tile_size")

    orig_h, orig_w = image_rgb.shape[:2]
    padded, _ = pad_to_tile_size(image_rgb, tile_size)
    h, w = padded.shape[:2]

    prob_sum = np.zeros((h, w), dtype=np.float32)
    count_sum = np.zeros((h, w), dtype=np.float32)

    ys = list(range(0, max(h - tile_size + 1, 1), stride))
    xs = list(range(0, max(w - tile_size + 1, 1), stride))
    if ys[-1] != h - tile_size:
        ys.append(h - tile_size)
    if xs[-1] != w - tile_size:
        xs.append(w - tile_size)

    for y in ys:
        for x in xs:
            tile = padded[y:y + tile_size, x:x + tile_size]
            inp = preprocess_image(tile, encoder, encoder_weights).to(device)
            probs = torch.sigmoid(model(inp))[0, 0].cpu().numpy()
            prob_sum[y:y + tile_size, x:x + tile_size] += probs
            count_sum[y:y + tile_size, x:x + tile_size] += 1.0

    return (prob_sum / np.maximum(count_sum, 1e-8))[:orig_h, :orig_w]


def process_camera(
    camera_dir: Path,
    csv_path: Path,
    model,
    encoder: str,
    encoder_weights: str,
    device: torch.device,
    threshold: float,
    use_tiling: bool,
    tile_size: int,
    stride: int,
) -> int:
    """Process all new images in one camera folder. Returns the number of newly processed images."""
    already_processed = load_existing_results(csv_path)
    images = list_images(camera_dir)
    new_images = [p for p in images if str(p) not in already_processed]

    for image_path in new_images:
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            print(f"  Skipping unreadable image: {image_path}")
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        if use_tiling:
            probs = predict_tiled(
                model=model, image_rgb=image_rgb, encoder=encoder,
                encoder_weights=encoder_weights, device=device,
                tile_size=tile_size, stride=stride,
            )
        else:
            probs = predict_full_image(
                model=model, image_rgb=image_rgb, encoder=encoder,
                encoder_weights=encoder_weights, device=device,
            )

        pixel_count = int((probs >= threshold).sum())
        append_result(csv_path, image_path, pixel_count)
        print(f"  {image_path.name}: {pixel_count} pixels")

    return len(new_images)


def main() -> None:
    args = parse_args()
    args.csv_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    arch = checkpoint["arch"]
    encoder = checkpoint["encoder"]
    encoder_weights = checkpoint["encoder_weights"]
    threshold = args.threshold if args.threshold is not None else checkpoint.get("threshold", 0.5)

    model = get_model(arch, encoder, encoder_weights)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"Device:    {device}")
    print(f"Model:     {arch} / {encoder}")
    print(f"Threshold: {threshold:.3f}")
    print(f"Tiling:    {args.use_tiling}")
    print(f"Watching:  {args.input_dir}")
    print(f"Poll:      every {args.poll_seconds}s")
    print()

    # Build the per-camera path pairs once
    cameras = [
        (args.input_dir / cam, args.csv_dir / f"{cam}.csv")
        for cam in CAMERA_DIRS
        if (args.input_dir / cam).is_dir()
    ]

    if not cameras:
        raise RuntimeError(f"No reolink_* subdirectories found under {args.input_dir}")

    print(f"Found {len(cameras)} camera folder(s): {[c.name for c, _ in cameras]}")

    while True:
        total_new = 0
        for camera_dir, csv_path in cameras:
            print(f"[{camera_dir.name}]")
            n = process_camera(
                camera_dir=camera_dir,
                csv_path=csv_path,
                model=model,
                encoder=encoder,
                encoder_weights=encoder_weights,
                device=device,
                threshold=threshold,
                use_tiling=args.use_tiling,
                tile_size=args.tile_size,
                stride=args.stride,
            )
            if n == 0:
                print("  No new images.")
            total_new += n

        print(f"\nScan complete — {total_new} new image(s) processed. Waiting {args.poll_seconds}s…\n")
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()