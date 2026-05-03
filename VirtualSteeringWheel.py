import cv2
import mediapipe as mp
import math
import pyautogui
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
HandLandmarkerResult = vision.HandLandmarkerResult

pyautogui.FAILSAFE = False


FINGER_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


def draw_hand(frame, hand_landmarks):
    h, w = frame.shape[:2]
    
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
    
    for i, j in FINGER_CONNECTIONS:
        cv2.line(frame, pts[i], pts[j], (0, 255, 0), 2)
    
    for x, y in pts:
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        cv2.circle(frame, (x, y), 5, (0, 0, 0), 1)
    
    return pts[0]


def run():
    options = BaseOptions(model_asset_path='VirtualSteeringWheel.task')
    options = HandLandmarkerOptions(base_options=options, num_hands=2)
    detector = HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    last_direction = None
    current_key = None
    direction = "No hands"
    angle = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)
        
        wrists = []
        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                wrist = draw_hand(frame, hand_landmarks)
                wrists.append(wrist)
        
        direction = "No hands"
        angle = 0
        
        if len(wrists) == 2:
            cv2.line(frame, wrists[0], wrists[1], (255, 0, 255), 3)
            
            wrists.sort(key=lambda p: p[0])
            left_wrist = wrists[0]
            right_wrist = wrists[1]
            
            dx = right_wrist[0] - left_wrist[0]
            dy = left_wrist[1] - right_wrist[1]
            hypotenuse = math.sqrt(dx**2 + dy**2)
            
            if hypotenuse > 0:
                ratio = min(abs(dy) / hypotenuse, 1.0)
                angle = math.degrees(math.asin(ratio))
            
            if angle > 10:
                if dy > 0:
                    direction = "Left higher"
                    key = 'a'
                else:
                    direction = "Right higher"
                    key = 'd'
                
                if current_key != key:
                    if current_key:
                        pyautogui.keyUp(current_key)
                    pyautogui.keyDown(key)
                    current_key = key
                
                pyautogui.keyDown(key)
            else:
                direction = "Balanced"
                if current_key:
                    pyautogui.keyUp(current_key)
                    current_key = None
        
        cv2.putText(frame, f"Direction: {direction}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"Angle: {angle:.1f}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Hand Landmarks', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run()