# Virtual Steering Wheel

A computer vision-based virtual steering wheel using hand tracking with MediaPipe. Control racing games by holding your hands in front of a webcam like you're gripping a steering wheel.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Run the script:
   ```bash
   python VirtualSteeringWheel.py
   ```

2. Hold both hands in front of your webcam like gripping a steering wheel

3. Press **'q'** to quit

## Hand States & Controls

| Hand State | Action | Key |
|------------|--------|-----|
| **Open** | Gas forward | `W` |
| **Open + Left higher** | Gas + Turn left | `W` + `A` |
| **Open + Right higher** | Gas + Turn right | `W` + `D` |
| **Fist** | Brake (no gas) | No key |
| **Fist + Left higher** | Turn left | `A` |
| **Fist + Right higher** | Turn right | `D` |
| **Index up** | Brake | `S` |
| **Index + Left higher** | Brake + Turn left | `S` + `A` |
| **Index + Right higher** | Brake + Turn right | `S` + `D` |

## Steering Detection

- Detects angle between both hands using Pythagorean calculation
- Threshold: angle > 10° triggers steering
- Visual steering wheel guide shows circle + horizontal line connecting hand joints

## Requirements

- Webcam
- Python 3.8+
- OpenCV
- MediaPipe 0.10+
- NumPy
- PyAutoGUI
- `VirtualSteeringWheel.task` (MediaPipe hand landmarker model)

## Notes

- Uses MediaPipe Tasks API (not deprecated `solutions` module)
- Ensure good lighting for hand detection
- Keep both hands within the camera frame
- Hand detection: wrist landmark (index 0)
- Index finger detection: landmark 8 above PIP (6) and MCP (5)
- Fist detection: all fingertips below PIP joints
