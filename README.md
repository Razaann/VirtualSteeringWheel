# Virtual Steering Wheel

A computer vision-based virtual steering wheel using hand tracking. Control racing games by holding your hands in front of a webcam like you're gripping a steering wheel.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Run the script:
   ```bash
   python steering_wheel.py
   ```

2. Hold both hands in front of your webcam:
   - Left hand controls the left side
   - Right hand controls the right side
   - The angle between your hands determines steering direction

3. Press **'q'** to quit

## Controls

- **Straight** (-10° to 10°): No input
- **Slight Left** (10° to 30°): Left arrow key
- **Medium Left** (30° to 50°): Left arrow key
- **Hard Left** (>50°): Left arrow key
- **Slight Right** (-30° to -10°): Right arrow key
- **Medium Right** (-50° to -30°): Right arrow key
- **Hard Right** (<-50°): Right arrow key

## Requirements

- Webcam
- Python 3.7+
- OpenCV
- MediaPipe
- NumPy
- PyAutoGUI

## Notes

- Ensure good lighting for hand detection
- Keep your hands within the camera frame
- The program uses the palm center (landmark 9) for tracking
