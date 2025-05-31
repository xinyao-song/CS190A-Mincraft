import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import time
import os
from collections import deque
import subprocess

# Suppress AVFoundation warnings (macOS specific)
os.environ['OPENCV_AVFOUNDATION_SKIP_AUTH'] = '1'
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Hand gesture to key mapping
GESTURE_TO_KEY = {
    'fist': 'w',        # Closed fist for forward
    'palm_up': 's',     # Open palm facing up for backward
    'peace': 'a',       # Peace sign (index+middle) for left
    'thumbs_up': 'd',   # Thumbs up for right
    'point_up': 'space'  # Index finger pointing up for jump
}

# Motion detection parameters
MOTION_HISTORY_LENGTH = 10
motion_history = deque(maxlen=MOTION_HISTORY_LENGTH)
DEBOUNCE_TIME = 0.5
last_key_time = 0

def focus_minecraft():
    """Focus the Minecraft window using AppleScript"""
    try:
        # AppleScript to focus Minecraft
        script = '''
        tell application "System Events"
            set frontProcess to first process whose name contains "Minecraft"
            set frontmost of frontProcess to true
        end tell
        '''
        subprocess.run(['osascript', '-e', script])
        return True
    except:
        return False

def initialize_camera():
    """Initialize the MacBook camera with proper settings"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        # Try different camera indices if 0 doesn't work
        for i in range(1, 3):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                break
    
    if not cap.isOpened():
        raise RuntimeError("Could not open camera. Check permissions.")
    
    # Set optimal resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    return cap

def get_hand_landmarks(image):
    """Process image and return hand landmarks"""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    return results.multi_hand_landmarks

def is_finger_extended(tip, base, threshold=0.1):
    """Check if a finger is extended"""
    return tip.y < base.y - threshold

def detect_gesture(landmarks):
    """Detect hand gestures based on landmarks"""
    if not landmarks:
        return None
    
    landmarks = landmarks[0].landmark
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]
    
    # Get finger base positions
    thumb_base = landmarks[2]
    index_base = landmarks[5]
    middle_base = landmarks[9]
    ring_base = landmarks[13]
    pinky_base = landmarks[17]
    
    # Check finger extensions
    index_extended = is_finger_extended(index_tip, index_base)
    middle_extended = is_finger_extended(middle_tip, middle_base)
    ring_extended = is_finger_extended(ring_tip, ring_base)
    pinky_extended = is_finger_extended(pinky_tip, pinky_base)
    thumb_extended = is_finger_extended(thumb_tip, thumb_base)
    
    # Count extended fingers
    fingers_extended = sum([index_extended, middle_extended, ring_extended, pinky_extended])
    
    # Fist (W) - All fingers closed
    if (not index_extended and not middle_extended and 
        not ring_extended and not pinky_extended and not thumb_extended):
        return 'fist'
    
    # Palm Up (S) - All fingers extended, palm up
    if (index_extended and middle_extended and 
        ring_extended and pinky_extended and
        thumb_extended):  # Palm facing up
        return 'palm_up'
    
    # Peace Sign (A) - Only index and middle fingers up
    if (index_extended and middle_extended and 
        not ring_extended and not pinky_extended and not thumb_extended):
        return 'peace'
    
    # Thumbs Up (D) - Only thumb up
    if (thumb_extended and not index_extended and 
        not middle_extended and not ring_extended and 
        not pinky_extended):
        return 'thumbs_up'
    
    # Point Down (Space) - Only index finger pointing down
    if (not thumb_extended and index_extended and 
        not middle_extended and not ring_extended and 
        not pinky_extended):  # Index finger pointing down
        return 'point_up'
    
    return None

def send_key(key):
    """Send keyboard input with debouncing"""
    global last_key_time
    current_time = time.time()
    if current_time - last_key_time > DEBOUNCE_TIME:
        # Focus Minecraft window before sending key
        if focus_minecraft():
            pyautogui.press(key)
            last_key_time = current_time
            print(f"Pressed key: {key}")
        else:
            print("Minecraft window not found!")

def main():
    print("Starting Minecraft gesture control...")
    print("Make sure Minecraft is running and in focus!")
    print("Press 'q' to quit the gesture control")
    
    cap = initialize_camera()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera disconnected or frame read failed")
                time.sleep(1)
                cap.release()
                cap = initialize_camera()
                continue
            
            frame = cv2.flip(frame, 1)
            
            landmarks = get_hand_landmarks(frame)
            
            if landmarks:
                # Draw landmarks
                for hand_landmarks in landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                gesture = detect_gesture(landmarks)
                if gesture:
                    motion_history.append(gesture)
                    
                    if len(motion_history) == MOTION_HISTORY_LENGTH:
                        most_common = max(set(motion_history), key=motion_history.count)
                        if motion_history.count(most_common) > MOTION_HISTORY_LENGTH * 0.7:
                            key = GESTURE_TO_KEY.get(most_common)
                            if key:
                                send_key(key)
            
            # Add instruction text to the frame
            cv2.putText(frame, "Minecraft Controls:", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "W: Fist (all fingers closed)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "S: Open palm facing up", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "A: Peace sign (index+middle up)", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "D: Thumbs up", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Space: Index finger pointing up", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Minecraft Gesture Control', frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()