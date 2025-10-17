#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Realtime snippet: capture 3–5s from camera -> motion image -> classify with unified model.
Designed for Raspberry Pi 5. Uses CPU by default; will use CUDA if available.

Run:
  python run_unified_infer.py \
    --model_dir Models/Unified/Mixed/seed_100 \
    --camera_index 0 \
    --duration 3.5

Expected files in --model_dir:
  - model_state.pt
  - config.json   (should contain "classes" and image_size used at train)
"""

import os
import time
import json
import argparse
from pathlib import Path
import numpy as np
import cv2

import torch
from torch import nn
from torchvision import transforms
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

try:
    from picamera2 import Picamera2
    _PICAM2_OK = True
except Exception:
    _PICAM2_OK = False

# -----------------------------
# Args & defaults
# -----------------------------
def get_args():
    p = argparse.ArgumentParser(description="Unified motion-image inference (Pi 5).")
    p.add_argument("--model_dir", type=str, required=True,
                   help="Directory with model_state.pt and config.json")
    p.add_argument("--camera_index", type=int, default=0, help="OpenCV camera index, default 0")
    p.add_argument("--duration", type=float, default=5, help="Seconds to capture frames")
    p.add_argument("--sample_every", type=int, default=1, help="Temporal sampling (every Nth frame)")
    p.add_argument("--image_size", type=int, default=224, help="Model input size (should match training)")
    p.add_argument("--fall_thresh", type=float, default=0.70, help="Confidence to confirm FALL")
    p.add_argument("--gesture_thresh", type=float, default=0.80, help="Confidence to confirm gesture")
    p.add_argument("--thresh_percentile", type=float, default=90.0, help="Percentile on non-zero diffs")
    p.add_argument("--thresh_min", type=float, default=2.0, help="Floor for diff threshold [0..255]")
    p.add_argument("--save_debug", type=str, default="debug_motion.png", help="Where to save motion image PNG")
    p.add_argument("--width", type=int, default=640, help="Capture width hint")
    p.add_argument("--height", type=int, default=480, help="Capture height hint")
    p.add_argument("--fps", type=int, default=30, help="Capture FPS hint")
    p.add_argument("--use_picam2", action="store_true", help="Use Picamera2 for capture instead of OpenCV")
    p.add_argument("--preview", action="store_true", help="Show live preview during capture")
    return p.parse_args()


# -----------------------------
# Model definition (match training head)
# -----------------------------
def make_backbone_and_dim(image_size: int):
    """Create conv backbone and return (module, feature_channels)."""
    m = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)

    backbone = m.features  # conv feature extractor (no classifier)
    # Infer channel dim with a dummy forward at current IMAGE_SIZE
    with torch.no_grad():
        backbone.eval()
        dummy = torch.zeros(1, 3, image_size, image_size)
        feats = backbone(dummy)
        feat_dim = feats.shape[1]  # channels
    return backbone, feat_dim

class UnifiedNet(nn.Module):
    """
    Simple classifier:
      - conv backbone (MobileNetV3-Large)
      - global max pooling (keeps strong motion edges)
      - BN + Dropout
      - Linear -> ReLU -> Dropout -> Linear (to NUM_CLASSES)
    """
    def __init__(self,
                 num_classes: int,
                 image_size: int = 224,
                 freeze_backbone: bool = True,
                 head_hidden: int = 128,
                 dropout: float = 0.3):
        super().__init__()
        self.backbone, feat_dim = make_backbone_and_dim(image_size)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.pool = nn.AdaptiveMaxPool2d(1)
        self.bn   = nn.BatchNorm1d(head_hidden)
        self.drop = nn.Dropout(dropout)
        self.fc1  = nn.Linear(feat_dim, head_hidden)
        self.relu = nn.ReLU(inplace=True)
        self.fc2  = nn.Linear(head_hidden, num_classes)

    def forward(self, x):
        x = self.backbone(x)           # [B, C, H, W]
        x = self.pool(x).flatten(1)    # [B, C]
        x = self.fc1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.drop(x)
        logits = self.fc2(x)
        return logits


# -----------------------------
# Preprocessing: frames -> motion image
# -----------------------------
def pad_to_square(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    s = max(h, w)
    top = (s - h) // 2
    bottom = s - h - top
    left = (s - w) // 2
    right = s - w - left
    return cv2.copyMakeBorder(img, top, bottom, left, right,
                              cv2.BORDER_CONSTANT, value=0)

def frames_to_motion_image(frames_bgr, sample_every: int,
                           thresh_percentile: float, thresh_min: float,
                           out_size: int) -> np.ndarray:
    """Returns RGB uint8 motion image (H,W,3)."""
    # sample and grayscale
    sampled = []
    for i, frame in enumerate(frames_bgr):
        if i % sample_every == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            sampled.append(gray)
    if len(sampled) < 2:
        raise RuntimeError(f"Not enough sampled frames: {len(sampled)}")

    acc = np.zeros_like(sampled[0], dtype=np.float32)
    prev = sampled[0]
    for cur in sampled[1:]:
        diff = np.abs(cur - prev)
        nz = diff[diff > 0]
        if nz.size > 0:
            thr = max(np.percentile(nz, thresh_percentile), thresh_min)
        else:
            thr = thresh_min
        mask = diff >= thr
        acc[mask] += diff[mask]
        prev = cur

    # normalize to 0..255
    m = float(acc.max())
    if m > 0:
        acc = acc / m
    acc = (acc * 255.0).clip(0, 255).astype(np.uint8)

    # pad -> resize -> stack 3ch
    acc = pad_to_square(acc)
    acc = cv2.resize(acc, (out_size, out_size), interpolation=cv2.INTER_AREA)
    rgb = np.stack([acc, acc, acc], axis=2)  # HWC uint8
    return rgb


# -----------------------------
# Camera capture
# -----------------------------
def capture_frames(camera_index: int, duration_sec: float, width: int, height: int, fps: int, show_preview: bool):
    cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index, cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS,          fps)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    time.sleep(0.3)

    if show_preview:
        cv2.namedWindow("Preview", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Preview", width, height)

    frames, t0, misses = [], time.time(), 0
    while (time.time() - t0) < duration_sec:
        ok, frame = cap.read()
        if not ok:
            misses += 1
            if misses > 5:
                break
            time.sleep(0.02)
            continue

        frames.append(frame)
        if show_preview:
            view = draw_guides(frame)
            cv2.imshow("Preview", view)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break

    cap.release()
    if show_preview:
        cv2.destroyWindow("Preview")
    return frames

def draw_guides(img):
    h, w = img.shape[:2]
    out = img.copy()
    # crosshair
    cv2.line(out, (w//2, 0), (w//2, h), (255, 255, 255), 1)
    cv2.line(out, (0, h//2), (w, h//2), (255, 255, 255), 1)
    # safe box
    r = int(min(w, h) * 0.3)
    cx, cy = w//2, h//2
    cv2.rectangle(out, (cx - r, cy - r), (cx + r, cy + r), (255, 255, 255), 1)
    return out

def capture_frames_picam2(duration_sec: float, width: int, height: int, fps: int, show_preview: bool):
    from picamera2 import Picamera2
    picam = Picamera2()
    cfg = picam.create_preview_configuration(
        main={"size": (width, height), "format": "RGB888"}
    )
    picam.configure(cfg)
    picam.start()
    time.sleep(0.2)

    if show_preview:
        cv2.namedWindow("Preview", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Preview", width, height)

    frames = []
    t0 = time.time()
    while (time.time() - t0) < duration_sec:
        rgb = picam.capture_array()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        frames.append(bgr)

        if show_preview:
            view = draw_guides(bgr)
            cv2.imshow("Preview", view)
            # allow ESC or q to stop early
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break

    picam.stop()
    if show_preview:
        cv2.destroyWindow("Preview")
    return frames


# -----------------------------
# Main
# -----------------------------
def main():
    args = get_args()
    model_dir = Path(args.model_dir)
    state_path = model_dir / "model_state.pt"
    cfg_path = model_dir / "config.json"

    if not state_path.exists():
        raise FileNotFoundError(f"Missing model_state.pt at {state_path}")
    if not cfg_path.exists():
        print(f"Warning: {cfg_path} not found; will fall back to defaults")

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load config
    classes = ["fall", "light", "fan", "curtain", "screen", "none"]
    image_size = args.image_size
    dropout = 0.3
    if cfg_path.exists():
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
        if "classes" in cfg and isinstance(cfg["classes"], list):
            classes = cfg["classes"]
        if "image_size" in cfg:
            image_size = int(cfg["image_size"])
        if "dropout" in cfg:
            dropout = float(cfg["dropout"])

    num_classes = len(classes)
    # class_to_idx = {c: i for i, c in enumerate(classes)}
    gesture_set = {"light", "fan", "curtain", "screen"}

    # model
    model = UnifiedNet(num_classes=num_classes, image_size=image_size, dropout=dropout)
    state = torch.load(state_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval().to(device)

    # transforms (must match training normalization; resize handled in motion builder)
    to_tensor = transforms.Compose([
        transforms.ToTensor(),  # HWC uint8 -> CHW float [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # ------------- pipeline timing -------------
    t_all0 = time.time()

    # 1) capture
    t0 = time.time()
    
    if args.use_picam2:
        frames = capture_frames_picam2(args.duration, args.width, args.height, args.fps, args.preview)
    else:
        frames = capture_frames(args.camera_index, args.duration, args.width, args.height, args.fps, args.preview)
    
    t1 = time.time()

    if len(frames) < 2:
        raise RuntimeError("Camera capture returned too few frames.")

    # 2) preprocess -> motion image
    t2 = time.time()
    motion_rgb = frames_to_motion_image(
        frames, sample_every=args.sample_every,
        thresh_percentile=args.thresh_percentile,
        thresh_min=args.thresh_min,
        out_size=image_size
    )
    t3 = time.time()

    # save debug motion image
    debug_path = Path(args.save_debug)
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(debug_path), cv2.cvtColor(motion_rgb, cv2.COLOR_RGB2BGR))

    # 3) classify
    t4 = time.time()
    x = to_tensor(motion_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)                          # [1, num_classes]
        probs = torch.softmax(logits, dim=1)[0]    # [num_classes]
    t5 = time.time()

    # 4) decision thresholds
    top_idx = int(torch.argmax(probs).item())
    top_label = classes[top_idx]
    top_conf = float(probs[top_idx].item())

    # apply class-group thresholds
    # - fall: confirm if >= fall_thresh
    # - gestures: confirm if >= gesture_thresh
    # - else -> "none"
    conf_label = "none"
    if top_label == "fall":
        if top_conf >= args.fall_thresh:
            conf_label = "fall"
    elif top_label in gesture_set:
        if top_conf >= args.gesture_thresh:
            conf_label = top_label
    elif top_label == "none":
        conf_label = "none"  # already none

    t_all1 = time.time()

    # ------------- report -------------
    print("\n=== Unified inference result ===")
    print(f"Top-1 predicted label: {top_label}  | confidence: {top_conf:.3f}")
    print(f"Thresholded label:     {conf_label} (fall≥{args.fall_thresh:.2f}, gesture≥{args.gesture_thresh:.2f})")
    print(f"Saved motion image to: {debug_path}")

    # latency
    print("\n--- Latency (seconds) ---")
    print(f"Capture:       {(t1 - t0):.3f}")
    print(f"Preprocess:    {(t3 - t2):.3f}")
    print(f"Inference:     {(t5 - t4):.3f}")
    print(f"Preprocessing and Classification: {((t_all1 - t_all0) - (t1 - t0)):.3f}")
    print(f"Total pipeline:{(t_all1 - t_all0):.3f}")

    # optional: print full class table
    print("\nClass probabilities:")
    for i, c in enumerate(classes):
        print(f"  {c:8s}: {float(probs[i]):.3f}")


if __name__ == "__main__":
    main()
