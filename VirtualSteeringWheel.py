import cv2
import mediapipe as mp
import math
import time
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


def is_open_palm(hand_landmarks):
    fingertips = [4, 8, 12, 16, 20]
    finger_pips = [2, 6, 10, 14, 18]
    
    for tip, pip in zip(fingertips, finger_pips):
        if hand_landmarks[tip].y > hand_landmarks[pip].y:
            return False
    return True


def is_index_up(hand_landmarks):
    index_tip = hand_landmarks[8]
    index_pip = hand_landmarks[6]
    index_mcp = hand_landmarks[5]
    wrist = hand_landmarks[0]
    
    return index_tip.y < index_pip.y and index_tip.y < index_mcp.y and index_tip.y < wrist.y


def draw_hand(frame, hand_landmarks):
    h, w = frame.shape[:2]
    
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
    
    for i, j in FINGER_CONNECTIONS:
        cv2.line(frame, pts[i], pts[j], (0, 255, 0), 2)
    
    for x, y in pts:
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        cv2.circle(frame, (x, y), 5, (0, 0, 0), 1)
    
    return pts[0]


def draw_steering_guide(frame, co, direction):
    if len(co) == 2:
        xm, ym = (co[0][0] + co[1][0]) / 2, (co[0][1] + co[1][1]) / 2
        radius = 150
        try:
            m = (co[1][1] - co[0][1]) / (co[1][0] - co[0][0])
        except:
            return
        a = 1 + m ** 2
        b = -2 * xm - 2 * co[0][0] * (m ** 2) + 2 * m * co[0][1] - 2 * m * ym
        c = xm ** 2 + (m ** 2) * (co[0][0] ** 2) + co[0][1] ** 2 + ym ** 2 - 2 * co[0][1] * ym - 2 * co[0][1] * co[0][0] * m + 2 * m * ym * co[0][0] - 22500
        xa = (-b + (b ** 2 - 4 * a * c) ** 0.5) / (2 * a)
        xb = (-b - (b ** 2 - 4 * a * c) ** 0.5) / (2 * a)
        ya = m * (xa - co[0][0]) + co[0][1]
        yb = m * (xb - co[0][0]) + co[0][1]
        
        xap = xa
        xbp = xb
        yap = ya
        ybp = yb
        
        if m != 0:
            ap = 1 + ((-1 / m) ** 2)
            bp = -2 * xm - 2 * xm * ((-1 / m) ** 2) + 2 * (-1 / m) * ym - 2 * (-1 / m) * ym
            cp = xm ** 2 + ((-1 / m) ** 2) * (xm ** 2) + ym ** 2 + ym ** 2 - 2 * ym * ym - 2 * ym * xm * (-1 / m) + 2 * (-1 / m) * ym * xm - 22500
            try:
                xap = (-bp + (bp ** 2 - 4 * ap * cp) ** 0.5) / (2 * ap)
                xbp = (-bp - (bp ** 2 - 4 * ap * cp) ** 0.5) / (2 * ap)
                yap = (-1 / m) * (xap - xm) + ym
                ybp = (-1 / m) * (xbp - xm) + ym
            except:
                pass
        
        cv2.circle(img=frame, center=(int(xm), int(ym)), radius=radius, color=(195, 255, 62), thickness=15)
        
        cv2.line(frame, (int(xb), int(yb)), (int(xa), int(ya)), (195, 255, 62), 20)


def run():
    options = BaseOptions(model_asset_path='VirtualSteeringWheel.task')
    options = HandLandmarkerOptions(base_options=options, num_hands=2)
    detector = HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    current_key = None
    current_steer = None
    direction = "No hands"
    angle = 0
    prev_time = time.time()
    fps = 0
    fps_samples = []
    last_fps_print_time = time.time()
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)
        
        wrists = []
        hand_states = []
        
        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                wrist = draw_hand(frame, hand_landmarks)
                wrists.append(wrist)
                if is_open_palm(hand_landmarks):
                    hand_states.append("Open Palm")
                elif is_index_up(hand_landmarks):
                    hand_states.append("Index")
                else:
                    hand_states.append("Fist")
        
        direction = "No hands"
        angle = 0
        steer_key = None
        gas_key = None
        
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
            
            is_open_palm_state = "Open Palm" in hand_states
            is_index_state = "Index" in hand_states
            
            steer_key = None
            
            if is_index_state:
                gas_key = None
                if angle > 10:
                    if dy > 0:
                        direction = "Left"
                        steer_key = 'a'
                    else:
                        direction = "Right"
                        steer_key = 'd'
                else:
                    direction = ""
            elif is_open_palm_state:
                gas_key = 's'
                if angle > 10:
                    if dy > 0:
                        direction = "Backward + Left"
                        steer_key = 'a'
                    else:
                        direction = "Backward + Right"
                        steer_key = 'd'
                else:
                    direction = "Backward"
            else:
                gas_key = 'w'
                if angle > 10:
                    if dy > 0:
                        direction = "Forward + Left"
                        steer_key = 'a'
                    else:
                        direction = "Forward + Right"
                        steer_key = 'd'
                else:
                    direction = "Forward"
            
            if gas_key and current_key != gas_key:
                if current_key:
                    pyautogui.keyUp(current_key)
                pyautogui.keyDown(gas_key)
                current_key = gas_key
            
            if steer_key and current_steer != steer_key:
                if current_steer:
                    pyautogui.keyUp(current_steer)
                pyautogui.keyDown(steer_key)
                current_steer = steer_key
            
            if not steer_key and current_steer:
                pyautogui.keyUp(current_steer)
                current_steer = None
            
            if not gas_key and current_key:
                pyautogui.keyUp(current_key)
                current_key = None
        else:
            if current_key:
                pyautogui.keyUp(current_key)
                current_key = None
            if current_steer:
                pyautogui.keyUp(current_steer)
                current_steer = None
        
        cv2.putText(frame, f"Direction: {direction}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        cv2.putText(frame, f"Angle: {angle:.1f}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        if len(wrists) == 2:
            draw_steering_guide(frame, wrists, direction)
        
        cv2.imshow('Virtual Steering Wheel', frame)
        
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        fps_samples.append(fps)
        
        if current_time - last_fps_print_time >= 10:
            elapsed_seconds = int(last_fps_print_time - start_time)
            avg_fps = sum(fps_samples[-30:]) / min(len(fps_samples), 30)
            print(f"FPS on {elapsed_seconds}s Frame: {avg_fps:.2f} FPS")
            last_fps_print_time = current_time
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    if current_key:
        pyautogui.keyUp(current_key)
    if current_steer:
        pyautogui.keyUp(current_steer)
    
    total_duration = time.time() - start_time
    overall_avg_fps = sum(fps_samples) / len(fps_samples) if fps_samples else 0
    print(f"Duration: {total_duration:.1f} seconds")
    print(f"Total frames: {len(fps_samples)}")
    print(f"Overall Average FPS: {overall_avg_fps:.2f}")
    print(f"Min FPS: {min(fps_samples):.2f}" if fps_samples else "Min FPS: N/A")
    print(f"Max FPS: {max(fps_samples):.2f}" if fps_samples else "Max FPS: N/A")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run()
