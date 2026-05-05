"""
Rollodor Vision Server — MobileNet-SSD Person Detection
========================================================
Testable RIGHT NOW with just a laptop webcam. No ESP32 needed.

Usage:
    # Install deps first:
    #   pip install opencv-python numpy
    #
    # Download the model files (one-time):
    #   python detector.py --download-model
    #
    # Run with laptop webcam:
    #   python detector.py --source webcam
    #
    # Run with ESP32-CAM stream (when you have hardware):
    #   python detector.py --source http://<ESP32_IP>/stream
    #
    # Run with a test video file:
    #   python detector.py --source path/to/video.mp4

"""

import cv2
import numpy as np
import argparse
import sys
import os
import urllib.request
import json
import time
from dataclasses import dataclass
from enum import Enum


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class SteerCommand(Enum):
    STOP = "stop"
    FORWARD = "forward"
    LEFT = "left"
    RIGHT = "right"
    SEARCH = "search"


@dataclass
class Detection:
    """A single person detection with bounding box and confidence."""
    x: int          # top-left x
    y: int          # top-left y
    w: int          # width
    h: int          # height
    confidence: float
    center_x: float  # normalized 0.0 (left) to 1.0 (right)
    area_ratio: float  # bbox area / frame area — proxy for distance


@dataclass
class NavigationResult:
    """The output of one vision processing cycle."""
    command: SteerCommand
    detections: list  # list of Detection
    target: Detection | None  # the person we're steering toward
    debug_frame: np.ndarray | None  # annotated frame for display


# ---------------------------------------------------------------------------
# Model downloader
# ---------------------------------------------------------------------------

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
# Canonical source: https://github.com/chuanqi305/MobileNet-SSD
# The prototxt is "deploy.prototxt" and the weights are "mobilenet_iter_73000.caffemodel"
# We rename them locally to MobileNetSSD_deploy.* for clarity.
PROTOTXT_URL = (
    "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/"
    "master/deploy.prototxt"
)
CAFFEMODEL_URL = (
    "https://github.com/chuanqi305/MobileNet-SSD/"
    "raw/master/mobilenet_iter_73000.caffemodel"
)

# MobileNet-SSD class labels — index 15 is "person"
CLASS_LABELS = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train",
    "tvmonitor"
]
PERSON_CLASS_ID = 15


def _download_progress(block_num, block_size, total_size):
    """Callback for urlretrieve to show download progress."""
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 // total_size)
        mb = downloaded / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        print(f"\r    {pct:3d}%  ({mb:.1f} / {total_mb:.1f} MB)", end="", flush=True)
    else:
        mb = downloaded / (1024 * 1024)
        print(f"\r    {mb:.1f} MB downloaded", end="", flush=True)


def download_model():
    """Download MobileNet-SSD model files if not present."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    prototxt_path = os.path.join(MODEL_DIR, "MobileNetSSD_deploy.prototxt")
    caffemodel_path = os.path.join(MODEL_DIR, "MobileNetSSD_deploy.caffemodel")

    for url, path, name, min_size in [
        (PROTOTXT_URL, prototxt_path, "prototxt", 1000),
        (CAFFEMODEL_URL, caffemodel_path, "caffemodel (~22 MB)", 20_000_000),
    ]:
        if os.path.exists(path) and os.path.getsize(path) > min_size:
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  [OK] {name} already exists ({size_mb:.1f} MB): {path}")
        else:
            print(f"  Downloading {name} from:\n    {url}")
            try:
                urllib.request.urlretrieve(url, path, _download_progress)
                print()  # newline after progress

                actual = os.path.getsize(path)
                if actual < min_size:
                    os.remove(path)
                    print(f"  [ERROR] Downloaded file too small ({actual} bytes).")
                    print(f"          GitHub may have rate-limited or redirected.")
                    print(f"          Try downloading manually:")
                    print(f"          1. Go to: https://github.com/chuanqi305/MobileNet-SSD")
                    print(f"          2. Download 'deploy.prototxt' → save as {prototxt_path}")
                    print(f"          3. Download 'mobilenet_iter_73000.caffemodel' → save as {caffemodel_path}")
                    return None, None

                print(f"  [OK] Saved to {path}")
            except Exception as e:
                print(f"\n  [ERROR] Download failed: {e}")
                print(f"          Manual download instructions:")
                print(f"          1. Go to: https://github.com/chuanqi305/MobileNet-SSD")
                print(f"          2. Download 'deploy.prototxt' → save as {prototxt_path}")
                print(f"          3. Download 'mobilenet_iter_73000.caffemodel' → save as {caffemodel_path}")
                return None, None

    return prototxt_path, caffemodel_path


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class PersonDetector:
    """
    Wraps MobileNet-SSD for person detection and outputs steering commands.

    The navigation logic is simple but effective:
    - Divide the frame into three vertical zones: LEFT | CENTER | RIGHT
    - If the largest detected person is in CENTER → FORWARD
    - If in LEFT → steer LEFT
    - If in RIGHT → steer RIGHT
    - If no person detected → SEARCH (rotate to scan)
    """

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        center_zone: tuple[float, float] = (0.35, 0.65),
        close_distance_ratio: float = 0.25,  # stop when person fills 25% of frame
    ):
        self.confidence_threshold = confidence_threshold
        self.center_zone = center_zone  # normalized x range considered "center"
        self.close_distance_ratio = close_distance_ratio
        self.net = None

    def load_model(self):
        """Load the neural network. Call once at startup."""
        prototxt = os.path.join(MODEL_DIR, "MobileNetSSD_deploy.prototxt")
        caffemodel = os.path.join(MODEL_DIR, "MobileNetSSD_deploy.caffemodel")

        if not os.path.exists(prototxt) or not os.path.exists(caffemodel):
            print("Model files not found. Downloading...")
            download_model()

        self.net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
        print("[OK] MobileNet-SSD model loaded.")

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run person detection on a single frame."""
        h, w = frame.shape[:2]
        frame_area = h * w

        # MobileNet-SSD expects 300x300 input
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            0.007843, (300, 300), 127.5
        )
        self.net.setInput(blob)
        raw_detections = self.net.forward()

        people = []
        for i in range(raw_detections.shape[2]):
            class_id = int(raw_detections[0, 0, i, 1])
            confidence = float(raw_detections[0, 0, i, 2])

            if class_id != PERSON_CLASS_ID:
                continue
            if confidence < self.confidence_threshold:
                continue

            # Scale bounding box back to frame dimensions
            box = raw_detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)

            bw = x2 - x1
            bh = y2 - y1
            center_x = (x1 + x2) / 2.0 / w  # normalize to 0..1

            people.append(Detection(
                x=x1, y=y1, w=bw, h=bh,
                confidence=confidence,
                center_x=center_x,
                area_ratio=(bw * bh) / frame_area,
            ))

        # Sort by area (closest person first)
        people.sort(key=lambda d: d.area_ratio, reverse=True)
        return people

    def navigate(self, frame: np.ndarray) -> NavigationResult:
        """Detect people and decide on a steering command."""
        detections = self.detect(frame)
        # Ensure sorted by area (closest first) regardless of detect() impl
        detections.sort(key=lambda d: d.area_ratio, reverse=True)
        debug = frame.copy()
        h, w = frame.shape[:2]

        # Draw zone lines on debug frame
        left_boundary = int(w * self.center_zone[0])
        right_boundary = int(w * self.center_zone[1])
        cv2.line(debug, (left_boundary, 0), (left_boundary, h), (50, 50, 50), 1)
        cv2.line(debug, (right_boundary, 0), (right_boundary, h), (50, 50, 50), 1)

        # Zone labels
        cv2.putText(debug, "LEFT", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        cv2.putText(debug, "CENTER", (left_boundary + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        cv2.putText(debug, "RIGHT", (right_boundary + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

        if not detections:
            cv2.putText(debug, "CMD: SEARCH", (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return NavigationResult(
                command=SteerCommand.SEARCH,
                detections=[], target=None, debug_frame=debug,
            )

        # Target the closest (largest) person
        target = detections[0]

        # Draw all detections
        for i, det in enumerate(detections):
            color = (0, 255, 0) if i == 0 else (200, 200, 0)
            cv2.rectangle(debug,
                          (det.x, det.y),
                          (det.x + det.w, det.y + det.h),
                          color, 2)
            label = f"person {det.confidence:.0%} | area {det.area_ratio:.1%}"
            cv2.putText(debug, label, (det.x, det.y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # Decide steering
        if target.area_ratio >= self.close_distance_ratio:
            command = SteerCommand.STOP  # close enough, arrived
        elif target.center_x < self.center_zone[0]:
            command = SteerCommand.LEFT
        elif target.center_x > self.center_zone[1]:
            command = SteerCommand.RIGHT
        else:
            command = SteerCommand.FORWARD

        # Draw command
        cmd_colors = {
            SteerCommand.FORWARD: (0, 255, 0),
            SteerCommand.LEFT: (255, 200, 0),
            SteerCommand.RIGHT: (255, 200, 0),
            SteerCommand.STOP: (0, 140, 255),
        }
        cv2.putText(debug, f"CMD: {command.value.upper()}",
                    (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    cmd_colors.get(command, (255, 255, 255)), 2)

        # Draw target center crosshair
        cx = int(target.center_x * w)
        cy = target.y + target.h // 2
        cv2.drawMarker(debug, (cx, cy), (0, 255, 0),
                       cv2.MARKER_CROSS, 20, 2)

        return NavigationResult(
            command=command,
            detections=detections,
            target=target,
            debug_frame=debug,
        )


# ---------------------------------------------------------------------------
# ESP32 command sender (stub for now, real HTTP later)
# ---------------------------------------------------------------------------

class CommandSender:
    """
    Sends steering commands to the ESP32.
    In test mode, just prints to console.
    When you have hardware, swap in HTTP requests.
    """

    def __init__(self, esp32_url: str | None = None):
        self.esp32_url = esp32_url
        self.last_command = None

    def send(self, command: SteerCommand):
        if command == self.last_command:
            return  # don't spam the same command

        self.last_command = command

        if self.esp32_url:
            # TODO: When you have hardware, uncomment:
            # import requests
            # requests.get(f"{self.esp32_url}/cmd?dir={command.value}", timeout=1)
            print(f"  >> ESP32 [{self.esp32_url}]: {command.value}")
        else:
            print(f"  >> [SIM] Command: {command.value}")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(source: str, esp32_url: str | None = None):
    """Main detection + navigation loop."""

    detector = PersonDetector()
    detector.load_model()
    sender = CommandSender(esp32_url)

    # Open video source
    if source == "webcam":
        cap = cv2.VideoCapture(0)
        print("[OK] Opened laptop webcam.")
    elif source.startswith("http"):
        cap = cv2.VideoCapture(source)
        print(f"[OK] Opened stream: {source}")
    else:
        cap = cv2.VideoCapture(source)
        print(f"[OK] Opened video file: {source}")

    if not cap.isOpened():
        print("[ERROR] Could not open video source.")
        sys.exit(1)

    print("\nRunning detection. Press 'q' to quit.\n")
    fps_time = time.time()
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if source != "webcam" and not source.startswith("http"):
                    print("End of video file.")
                    break
                continue

            result = detector.navigate(frame)
            sender.send(result.command)

            # FPS counter
            frame_count += 1
            elapsed = time.time() - fps_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_time = time.time()
                if result.debug_frame is not None:
                    cv2.putText(result.debug_frame, f"FPS: {fps:.1f}",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (100, 255, 100), 1)

            # Show debug window
            if result.debug_frame is not None:
                cv2.imshow("Rollodor Vision", result.debug_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\nShutdown.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rollodor Vision Server — Person Detection & Navigation"
    )
    parser.add_argument(
        "--source", default="webcam",
        help="Video source: 'webcam', an HTTP stream URL, or a video file path"
    )
    parser.add_argument(
        "--esp32-url", default=None,
        help="ESP32 base URL for sending commands (e.g. http://192.168.1.100)"
    )
    parser.add_argument(
        "--download-model", action="store_true",
        help="Download MobileNet-SSD model files and exit"
    )
    parser.add_argument(
        "--confidence", type=float, default=0.5,
        help="Minimum detection confidence (0.0 - 1.0)"
    )

    args = parser.parse_args()

    if args.download_model:
        print("Downloading MobileNet-SSD model files...")
        download_model()
        print("\nDone! You can now run: python detector.py --source webcam")
        sys.exit(0)

    run(source=args.source, esp32_url=args.esp32_url)