# Rollodor — Motorized Belmet Rollodor Ashtray Robot

A motorized ashtray that drives to you using computer vision, voice commands, and tank steering.

## Project Structure

```
rollodor/
├── vision_server/          # Python — runs on laptop/Pi
│   ├── detector.py         # MobileNet-SSD person detection + navigation
│   └── test_detector.py    # Unit tests (no hardware needed)
│
├── esp32_firmware/         # C++ — runs on ESP32-CAM
│   ├── platformio.ini      # PlatformIO configuration
│   └── src/
│       └── main.cpp        # Motor control, WiFi, HTTP endpoints
│
└── web_ui/
    └── index.html          # Voice command + manual control interface
```

## Quick Start (No Hardware Needed)

### 1. Run the vision tests
```bash
cd vision_server
python test_detector.py
```

### 2. Test with your laptop webcam
```bash
cd vision_server
pip install opencv-python numpy
python detector.py --download-model
python detector.py --source webcam
```
Stand in front of your webcam — you'll see bounding boxes and steering commands.

### 3. Open the control UI
Open `web_ui/index.html` in Chrome. It starts in simulation mode:
- Click the mic button and say "come here" or "stop"
- Use the arrow buttons
- Use keyboard: W/A/S/D or arrow keys
- All commands log to the terminal at the bottom

### 4. When you have hardware
1. Flash `esp32_firmware` via PlatformIO or Arduino IDE
2. Enter the ESP32's IP address in the web UI
3. Point `detector.py` at the ESP32 camera stream:
   ```bash
   python detector.py --source http://<ESP32_IP>/stream --esp32-url http://<ESP32_IP>
   ```

## Hardware Shopping List

| Part | Purpose | ~Cost |
|------|---------|-------|
| ESP32-CAM (Hosyond/AI-Thinker) | Brain + camera | $6 |
| FTDI USB-to-Serial adapter | Programming the ESP32 | $4 |
| 2x 360° continuous rotation servos | Tank steering | $12 |
| 2-3x HC-SR04 ultrasonic sensors | Collision avoidance | $3 |
| Battery pack (7.4V LiPo or 4xAA) | Power | $8 |
| Jumper wires + breadboard | Prototyping | $5 |

**Optional upgrade:** RPLidar A1 for room mapping (~$100)

## Architecture

```
┌──────────────────┐     WiFi      ┌──────────────────┐
│    ESP32-CAM     │ ◄──────────►  │  Vision Server   │
│                  │   commands     │  (Python/OpenCV)  │
│  - Camera stream │               │                  │
│  - Motor control │               │  - MobileNet-SSD │
│  - Sensor reads  │               │  - Navigation    │
│  - HTTP server   │               │  - Command sender│
└──────────────────┘               └──────────────────┘
        ▲                                   ▲
        │ HTTP                              │
        ▼                                   │
┌──────────────────┐                        │
│    Web UI        │   (voice commands      │
│  (Browser)       │    also go to ESP32    │
│                  │    or vision server)   │
│  - Voice input   │                        │
│  - Manual control│                        │
│  - Command log   │                        │
└──────────────────┘
```