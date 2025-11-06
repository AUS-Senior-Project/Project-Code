#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Realtime snippet: capture N seconds from camera -> motion image -> classify with unified model.
Raspberry Pi 5 friendly. Uses CPU by default; uses CUDA if available.

Example:
  python VideoProcessTest.py \
    --model_dir Models/Unified/Mixed/seed_100 \
    --use_picam2 --preview \
    --use_gpio --use_lcd 
"""

# =============================
# Imports
# =============================
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

from twilio.rest import Client
from datetime import datetime

# Picamera2 (optional)
try:
    from picamera2 import Picamera2
    _PICAM2_OK = True
except Exception:
    _PICAM2_OK = False

# GPIO (RPi.GPIO only)
try:
    import RPi.GPIO as GPIO
    _GPIO_OK = True
    print("Using RPi.GPIO")
except Exception:
    GPIO = None
    _GPIO_OK = False
    print("RPi.GPIO not available, GPIO disabled")

# LCD (PCF8574)
try:
    from RPLCD.i2c import CharLCD
    _LCD_OK = True
except Exception:
    _LCD_OK = False


# =============================
# Args
# =============================
def get_args():
    p = argparse.ArgumentParser(description="Unified motion-image inference (Pi 5).")

    # model / thresholds
    p.add_argument("--model_dir", type=str, default="Models/Unified/Mixed/seed_100", help="Directory with model_state.pt and config.json",)
    p.add_argument("--duration", type=float, default=5.0, help="Seconds to capture frames")
    p.add_argument("--sample_every", type=int, default=1, help="Temporal sampling (every Nth frame)")
    p.add_argument("--image_size", type=int, default=224, help="Model input size (should match training)")
    p.add_argument("--fall_thresh", type=float, default=0.50, help="Confidence to confirm FALL")
    p.add_argument("--gesture_thresh", type=float, default=0.60, help="Confidence to confirm gesture")
    p.add_argument("--thresh_percentile", type=float, default=80.0, help="Percentile on non-zero diffs")

    # debug image
    p.add_argument("--save_debug", type=str, default="debug_motion.jpg", help="Where to save motion image; use .jpg to avoid libpng warnings",)
    p.add_argument("--no_debug_image", action="store_true", help="Skip saving the motion image")

    # capture settings
    p.add_argument("--width", type=int, default=640, help="Capture width")
    p.add_argument("--height", type=int, default=480, help="Capture height")
    p.add_argument("--fps", type=int, default=30, help="Capture FPS")

    # capture backends and preview
    p.add_argument("--use_picam2", action="store_true", help="Use Picamera2 for capture")
    p.add_argument("--preview", action="store_true", help="Show live preview during capture")

    # Hardware flags
    p.add_argument("--use_gpio", action="store_true", help="Enable GPIO (LED/fan/curtain)")
    p.add_argument("--led_pin", type=int, default=17, help="BCM pin for LED (default 17, physical 11)")
    p.add_argument("--fan_pin", type=int, default=18, help="BCM pin for FAN (default 18, physical 12)")
    p.add_argument("--curtain_pin", type=int, default=22, help="BCM pin for curtain motor/relay (default 22)")

    p.add_argument("--use_lcd", action="store_true", help="Enable I2C LCD (PCF8574)")
    p.add_argument("--lcd_addr", type=lambda x: int(x, 0), default=0x27, help="LCD I2C addr (0x27 or 0x3F)")
    p.add_argument("--lcd_cols", type=int, default=16, help="LCD columns")
    p.add_argument("--lcd_rows", type=int, default=2, help="LCD rows")

    # debug: keep outputs as-is on exit
    p.add_argument("--no_cleanup", action="store_true", help="Skip hardware cleanup on exit (debug)")

    return p.parse_args()


# =============================
# Model
# =============================
def make_backbone_and_dim(image_size: int, model_name: str):
    if model_name == "efficientnet_v2_s":
        m = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
    else:
        raise ValueError(f"Only 'efficientnet_v2_s' supported. Unknown model name: {model_name}")

    backbone = m.features
    with torch.no_grad():
        backbone.eval()
        dummy = torch.zeros(1, 3, image_size, image_size)
        feats = backbone(dummy)
        feat_dim = feats.shape[1]
    return backbone, feat_dim


class UnifiedNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        image_size: int = 224,
        model_name: str = "efficientnet_v2_s",
        freeze_backbone: bool = True,
        head_hidden: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.backbone, feat_dim = make_backbone_and_dim(image_size, model_name)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.pool = nn.AdaptiveMaxPool2d(1)
        self.bn = nn.BatchNorm1d(head_hidden)
        self.drop = nn.Dropout(dropout)
        self.fc1 = nn.Linear(feat_dim, head_hidden)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(head_hidden, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x).flatten(1)
        x = self.fc1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.drop(x)
        logits = self.fc2(x)
        return logits


# =============================
# Hardware control
# =============================
class Hardware:
    """
    Device states:
      - LED: on/off
      - Fan: latched on/off
      - Curtain: logical on/off (toggled), but GPIO only pulses 5s each gesture
      - Screen (LCD): off at start; toggled by 'screen' gesture

    light   -> LED pulse + optional LCD message
    fan     -> TOGGLE fan state + LCD summary (if screen ON)
    curtain -> TOGGLE curtain_state, send 5s pulse, LCD summary (if screen ON)
    screen  -> TOGGLE LCD (screen) on/off
    fall    -> LED blink + fall notification + LCD alert (if screen ON)
    none    -> LCD 'None' (if screen ON), no hardware change
    """

    def __init__(
        self,
        use_gpio: bool,
        led_pin: int,
        fan_pin: int,
        curtain_pin: int,
        use_lcd: bool,
        lcd_addr: int,
        lcd_cols: int,
        lcd_rows: int,
    ):
        # GPIO basic config
        self.use_gpio = use_gpio and _GPIO_OK
        self.led_pin = led_pin
        self.fan_pin = fan_pin
        self.curtain_pin = curtain_pin

        # simple state variables
        self.led_state = False
        self.fan_state = False
        self.curtain_state = False   # logical state (open/closed)
        self.screen_state = False    # LCD on/off

        # LCD
        self.use_lcd = use_lcd and _LCD_OK
        self.lcd_addr = lcd_addr
        self.lcd_cols = lcd_cols
        self.lcd_rows = lcd_rows
        self._lcd = None

    def setup(self):
        # GPIO setup
        if self.use_gpio:
            try:
                GPIO.setmode(GPIO.BCM)
                GPIO.setwarnings(False)
                GPIO.setup(self.led_pin, GPIO.OUT, initial=GPIO.LOW)
                GPIO.setup(self.fan_pin, GPIO.OUT, initial=GPIO.LOW)
                GPIO.setup(self.curtain_pin, GPIO.OUT, initial=GPIO.LOW)
                print("[HW] RPi.GPIO initialized")
            except Exception as e:
                print(f"[HW] RPi.GPIO init failed: {e}")
                self.use_gpio = False

        # LCD: create object but keep it OFF initially
        if self.use_lcd:
            try:
                self._lcd = CharLCD(
                    i2c_expander="PCF8574",
                    address=self.lcd_addr,
                    port=1,
                    cols=self.lcd_cols,
                    rows=self.lcd_rows,
                    charmap="A02",
                    auto_linebreaks=True,
                )
                self.lcd_off()
                self.screen_state = False
                print("[HW] LCD initialized (screen OFF)")
            except Exception as e:
                print(f"[warn] LCD init failed: {e}")
                self._lcd = None
                self.use_lcd = False

    def cleanup(self):
        # turn off outputs
        try:
            self.led(False)
            self.fan(False)
            self.curtain_pin_output(False)
        except Exception:
            pass

        # GPIO cleanup
        if self.use_gpio and _GPIO_OK:
            try:
                GPIO.cleanup()
            except Exception:
                pass
        # leave LCD as-is

    # --- low-level GPIO ops ---
    def led(self, on: bool):
        self.led_state = bool(on)
        if not (self.use_gpio and _GPIO_OK):
            return
        GPIO.output(self.led_pin, GPIO.HIGH if on else GPIO.LOW)

    def fan(self, on: bool):
        self.fan_state = bool(on)
        if not (self.use_gpio and _GPIO_OK):
            return
        GPIO.output(self.fan_pin, GPIO.HIGH if on else GPIO.LOW)

    def curtain_pin_output(self, on: bool):
        """
        Physical pin control for curtain (relay/motor trigger).
        Does NOT touch curtain_state (logical open/closed).
        """
        if not (self.use_gpio and _GPIO_OK):
            return
        GPIO.output(self.curtain_pin, GPIO.HIGH if on else GPIO.LOW)

    # led stays latched
    def led_toggle(self):
        self.led(not self.led_state)
    
    # fan stays latched
    def fan_toggle(self):
        self.fan(not self.fan_state)

    # --- LCD helpers ---
    def lcd_on(self):
        if not self._lcd:
            return
        try:
            self._lcd.display_enabled = True
        except Exception:
            try:
                self._lcd.display_on()
            except Exception:
                pass
        try:
            self._lcd.backlight_enabled = True
        except Exception:
            try:
                self._lcd.backlight = True
            except Exception:
                pass
        self.screen_state = True

    def lcd_off(self):
        if not self._lcd:
            return
        try:
            self._lcd.display_enabled = False
        except Exception:
            try:
                self._lcd.display_off()
            except Exception:
                pass
        try:
            self._lcd.backlight_enabled = False
        except Exception:
            try:
                self._lcd.backlight = False
            except Exception:
                pass
        self.screen_state = False

    def lcd_print(self, line1: str, line2: str = ""):
        if not self._lcd or not self.screen_state:
            return
        try:
            self._lcd.clear()
            self._lcd.write_string(str(line1)[: self.lcd_cols])
            if self.lcd_rows > 1 and line2:
                self._lcd.cursor_pos = (1, 0)
                self._lcd.write_string(str(line2)[: self.lcd_cols])
        except Exception as e:
            print(f"[warn] LCD print failed: {e}")

    # --- Gesture mapping ---
    def apply_from_label(self, conf_label: str, probs_table: dict):
        """
        Map decision label to hardware and LCD actions.
        """

        print(
            f"[HW] Decision: {conf_label} "
            f"(LED={self.led_state}, fan={self.fan_state}, "
            f"curtain={self.curtain_state}, screen={self.screen_state})"
        )

        if conf_label == "light":
            # toggle LED like a light
            self.led_toggle()
            state = "ON" if self.led_state else "OFF"
            print(f"[HW] LED {state} (latched)")
            if self.screen_state:
                self.lcd_print("Light toggled", f"LED: {state}")
        elif conf_label == "fan":
            # toggle latched fan
            self.fan_toggle()
            state = "ON" if self.fan_state else "OFF"
            print(f"[HW] FAN {state} (latched)")
            if self.screen_state:
                self.lcd_print("Fan toggled", f"Fan: {state}")
        elif conf_label == "curtain":
            # toggle logical curtain state
            self.curtain_state = not self.curtain_state
            state = "ON" if self.curtain_state else "OFF"
            print(f"[HW] CURTAIN logical {state} (5s pulse)")
            if self.screen_state:
                self.lcd_print("Curtain toggled", f"Curtain: {state}")
            # pulse the physical pin for 5 seconds
            self.curtain_pin_output(True)
            time.sleep(5.0)
            self.curtain_pin_output(False)
            if self.screen_state:
                self.lcd_off()
                self.lcd_on()
        elif conf_label == "screen":
            # toggle screen on/off
            if self.screen_state:
                print("[HW] Screen OFF (screen gesture)")
                self.lcd_off()
            else:
                print("[HW] Screen ON (screen gesture)")
                self.lcd_on()
                self.lcd_print("Screen ON", "")
        elif conf_label == "fall":
            old_led = self.led_state # save old LED state
            # fall notification + LED blink + LCD alert
            fall_notif()
            if self.screen_state:
                self.lcd_print("ALERT: FALL", "")
            for _ in range(5):
                self.led(True)
                time.sleep(0.2)
                self.led(False)
                time.sleep(0.2)
            self.led(old_led)  # restore old LED state
        else:
            # none / low confidence
            print("[HW] No hardware action")
            if self.screen_state:
                self.lcd_print("Gesture: NONE", "")


# =============================
# Motion image
# =============================
def pad_to_square(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    s = max(h, w)
    top = (s - h) // 2
    bottom = s - h - top
    left = (s - w) // 2
    right = s - w - left
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)


def frames_to_motion_image(
    frames_bgr,
    sample_every: int,
    thresh_percentile: float,
    out_size: int,
) -> np.ndarray:
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
        thr = np.percentile(diff, thresh_percentile)
        mask = diff >= thr
        acc[mask] += diff[mask]
        prev = cur

    m = acc.max()
    if m > 0:
        acc = acc / m
    acc = (acc * 255.0).clip(0, 255).astype(np.uint8)

    acc = pad_to_square(acc)
    acc = cv2.resize(acc, (out_size, out_size), interpolation=cv2.INTER_AREA)
    rgb = np.stack([acc, acc, acc], axis=2)
    return rgb


# =============================
# Capture
# =============================
def draw_guides(img):
    h, w = img.shape[:2]
    out = img.copy()
    cv2.line(out, (w // 2, 0), (w // 2, h), (255, 255, 255), 1)
    cv2.line(out, (0, h // 2), (w, h // 2), (255, 255, 255), 1)
    r = int(min(w, h) * 0.3)
    cx, cy = w // 2, h // 2
    cv2.rectangle(out, (cx - r, cy - r), (cx + r, cy + r), (255, 255, 255), 1)
    return out


def capture_frames_picam2(picam, duration_sec: float, width: int, height: int, fps: int, show_preview: bool):
    if not _PICAM2_OK:
        raise RuntimeError("Picamera2 requested but not available")
    if picam is None:
        raise RuntimeError("Picamera2 instance is None (not initialized)")

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
            cv2.imshow("Preview", draw_guides(bgr))
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

    if show_preview:
        cv2.destroyAllWindows()
        
    return frames


# =============================
# Notification
# =============================
def fall_notif():
    # WARNING: these creds should normally be stored in env vars or config, not hardcoded.
    ACCOUNT_SID = "AC194a75f5144923182eb4cd82b1640dd8"
    AUTH_TOKEN = "8d2719aedd404f114b2a3767d74547e5"

    client = Client(ACCOUNT_SID, AUTH_TOKEN)
    timestamp = datetime.now().strftime("%B %d, %Y at %I:%M %p")

    recipients = [
        "whatsapp:+971508655525",  # Mohammad
        "whatsapp:+971525525562",  # Hamad
        "whatsapp:+971552590909",  # Saif
        "whatsapp:+971505092255",  # Obaid
    ]

    for number in recipients:
        message = client.messages.create(
            from_="whatsapp:+14155238886",
            body=f"🚨 Fall detected at {timestamp} - please check immediately.",
            to=number,
        )
        print(f"[Twilio] Message sent to {number}: {message.sid}")


# =============================
# Main
# =============================
def main():
    args = get_args()
    model_dir = Path(args.model_dir)
    state_path = model_dir / "model_state.pt"
    cfg_path = model_dir / "config.json"

    if not state_path.exists():
        raise FileNotFoundError(f"Missing model_state.pt at {state_path}")
    if not cfg_path.exists():
        print(f"Warning: {cfg_path} not found; will fall back to defaults")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # config
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
    gesture_set = {"light", "fan", "curtain", "screen"}

    # model
    model = UnifiedNet(num_classes=num_classes, image_size=image_size, dropout=dropout)
    state = torch.load(state_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval().to(device)

    # transforms
    to_tensor = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    # hardware
    hw = Hardware(
        use_gpio=args.use_gpio,
        led_pin=args.led_pin,
        fan_pin=args.fan_pin,
        curtain_pin=args.curtain_pin,
        use_lcd=args.use_lcd,
        lcd_addr=args.lcd_addr,
        lcd_cols=args.lcd_cols,
        lcd_rows=args.lcd_rows,
    )
    hw.setup()
    
    # Picamera2 init
    picam = None
    if args.use_picam2:
        if not _PICAM2_OK:
            raise RuntimeError("Picamera2 requested but not available on this system.")

        picam = Picamera2()
        cfg = picam.create_preview_configuration(
            main={"size": (args.width, args.height), "format": "RGB888"}
        )
        picam.configure(cfg)
        picam.start()
        time.sleep(0.2)
        print("[Cam] Picamera2 started")
    
    print("\n=== Unified inference pipeline ready ===")
    print("All devices start OFF, screen OFF. Use gestures to control them.\n")

    try:
        while True:
            input("\nPress Enter to start capture and inference (or Ctrl+C to exit)...")
            print("\n========= Starting unified inference pipeline =========\n")

            t_all0 = time.time()

            # capture
            t0 = time.time()
            if args.use_picam2:
                frames = capture_frames_picam2(
                    picam,
                    args.duration,
                    args.width,
                    args.height,
                    args.fps,
                    args.preview,
                )
            else:
                raise RuntimeError("No capture backend selected. Use --use_picam2.")
            t1 = time.time()

            if len(frames) < 2:
                raise RuntimeError("Camera capture returned too few frames.")

            # preprocess
            t2 = time.time()
            motion_rgb = frames_to_motion_image(
                frames,
                sample_every=args.sample_every,
                thresh_percentile=args.thresh_percentile,
                out_size=image_size,
            )
            t3 = time.time()

            # optional debug image
            debug_path = None
            if not args.no_debug_image:
                debug_path = Path(args.save_debug)
                debug_path.parent.mkdir(parents=True, exist_ok=True)
                bgr = cv2.cvtColor(motion_rgb, cv2.COLOR_RGB2BGR)
                if debug_path.suffix.lower() in [".jpg", ".jpeg"]:
                    cv2.imwrite(str(debug_path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                else:
                    cv2.imwrite(str(debug_path), bgr)

            # classify
            t4 = time.time()
            x = to_tensor(motion_rgb).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=1)[0]
            t5 = time.time()

            # decision
            top_idx = int(torch.argmax(probs).item())
            top_label = classes[top_idx]
            top_conf = float(probs[top_idx].item())

            conf_label = "none"
            if top_label == "fall":
                if top_conf >= args.fall_thresh:
                    conf_label = "fall"
            elif top_label in gesture_set:
                if top_conf >= args.gesture_thresh:
                    conf_label = top_label
            elif top_label == "none":
                conf_label = "none"

            # hardware actions
            probs_table = {classes[i]: float(probs[i]) for i in range(len(classes))}
            hw.apply_from_label(conf_label, probs_table)

            t_all1 = time.time()

            # report
            print("\n=== Unified inference result ===")
            print(f"Top-1 predicted label: {top_label}  | confidence: {top_conf:.3f}")
            print(
                f"Thresholded label:     {conf_label} "
                f"(fall>={args.fall_thresh:.2f}, gesture>={args.gesture_thresh:.2f})"
            )
            if debug_path is not None:
                print(f"Saved motion image to: {debug_path}")

            print("\n--- Latency (seconds) ---")
            print(f"Capture:                 {(t1 - t0):.3f}")
            print(f"Preprocess:              {(t3 - t2):.3f}")
            print(f"Inference:               {(t5 - t4):.3f}")
            print(f"Preprocess + Inference:  {(t5 - t2):.3f}")
            print(f"Total pipeline:          {(t_all1 - t_all0):.3f}")

            print("\nClass probabilities:")
            for i, c in enumerate(classes):
                print(f"  {c:8s}: {float(probs[i]):.3f}")

            print(
                "\nDevice states now:"
                f"  LED={hw.led_state}"
                f"  Fan={hw.fan_state}"
                f"  Curtain={hw.curtain_state}"
                f"  Screen={hw.screen_state}"
            )

    except KeyboardInterrupt:
        print("\n[Main] Ctrl+C received, exiting.")
    finally:
        # hw cleanup
        if not args.no_cleanup:
            hw.cleanup()
            print("[HW] Cleaned up GPIO and left LCD as-is.")
        else:
            print("[HW] Skipping cleanup (debug mode). States left as-is.")
        
        # camera cleanup
        if args.use_picam2:
            try:
                if picam is not None:
                    picam.stop()
                    picam.close()
                    print("[Cam] Picamera2 stopped and closed.")
            except Exception as e:
                print(f"[Cam] Error during camera cleanup: {e}")

if __name__ == "__main__":
    main()
