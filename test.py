import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # Changed to detect up to 2 hands
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Minecraft control keys
FORWARD_KEY = 'w'
BACKWARD_KEY = 's'
LEFT_KEY = 'a'
RIGHT_KEY = 'd'
JUMP_KEY = 'space'
MOUSE_LEFT = 'left'

def is_hand_open(landmarks):
    """Check if all fingers are extended (open hand)"""
    finger_tips = [8, 12, 16, 20]  # Index, middle, ring, pinky tips
    finger_mcps = [6, 10, 14, 18]  # Corresponding MCP joints
    
    for tip, mcp in zip(finger_tips, finger_mcps):
        if landmarks[tip].y > landmarks[mcp].y:  # If tip is below MCP, finger is not extended
            return False
    return True

def is_fist(landmarks):
    """Check if all fingers are closed (fist)"""
    finger_tips = [8, 12, 16, 20]  # Index, middle, ring, pinky tips
    finger_mcps = [6, 10, 14, 18]  # Corresponding MCP joints
    
    for tip, mcp in zip(finger_tips, finger_mcps):
        if landmarks[tip].y < landmarks[mcp].y:  # If tip is above MCP, finger is extended
            return False
    return True

def is_index_up(landmarks):
    """Check if only index finger is up"""
    # Index finger up, others down
    if landmarks[8].y < landmarks[6].y:  # Index finger up
        if landmarks[12].y > landmarks[10].y:  # Middle finger down
            if landmarks[16].y > landmarks[14].y:  # Ring finger down
                if landmarks[20].y > landmarks[18].y:  # Pinky down
                    return True
    return False

def is_peace_sign(landmarks):
    """Check if index and middle fingers are up (peace sign)"""
    # Index and middle fingers up, others down
    if landmarks[8].y < landmarks[6].y:  # Index finger up
        if landmarks[12].y < landmarks[10].y:  # Middle finger up
            if landmarks[16].y > landmarks[14].y:  # Ring finger down
                if landmarks[20].y > landmarks[18].y:  # Pinky down
                    return True
    return False

def is_three_fingers_up(landmarks):
    """Check if index, middle, and ring fingers are up"""
    # Index, middle, and ring fingers up, pinky down
    if landmarks[8].y < landmarks[6].y:  # Index finger up
        if landmarks[12].y < landmarks[10].y:  # Middle finger up
            if landmarks[16].y < landmarks[14].y:  # Ring finger up
                if landmarks[20].y > landmarks[18].y:  # Pinky down
                    return True
    return False

def is_shaka(landmarks):
    """Check if only thumb and pinky are extended (shaka/hang loose gesture)"""
    # Thumb extended (pointing to the right in the camera view)
    if landmarks[4].x > landmarks[3].x:  # Thumb tip is to the right of thumb MCP
        # Pinky extended
        if landmarks[20].y < landmarks[18].y:  # Pinky up
            # Other fingers down
            if landmarks[8].y > landmarks[6].y:  # Index finger down
                if landmarks[12].y > landmarks[10].y:  # Middle finger down
                    if landmarks[16].y > landmarks[14].y:  # Ring finger down
                        return True
    return False

def is_thumbs_down(landmarks):
    """Check if thumb is down and other fingers are closed"""
    # Thumb down (pointing to the left in the camera view)
    if landmarks[4].x < landmarks[3].x:  # Thumb tip is to the left of thumb MCP
        # Check if other fingers are closed
        if landmarks[8].y > landmarks[6].y:  # Index finger down
            if landmarks[12].y > landmarks[10].y:  # Middle finger down
                if landmarks[16].y > landmarks[14].y:  # Ring finger down
                    if landmarks[20].y > landmarks[18].y:  # Pinky down
                        return True
    return False

def process_hand_gesture(landmarks):
    """Process hand landmarks and determine the gesture for a single hand"""
    if not landmarks:
        return None
    
    # Check for each gesture in order of priority
    if is_thumbs_down(landmarks):
        return 'stop'
    elif is_hand_open(landmarks):
        return 'forward'
    elif is_fist(landmarks):
        return 'backward'
    elif is_index_up(landmarks):
        return 'left'
    elif is_peace_sign(landmarks):
        return 'right'
    elif is_three_fingers_up(landmarks):
        return 'mouse_left'
    elif is_shaka(landmarks):
        return 'jump'
    
    return 'stop'

def main():
    print("Starting Minecraft Gesture Control...")
    print("Press 'q' to quit")
    print("\nGesture Guide:")
    print("Open Hand: Move Forward")
    print("Fist: Move Backward")
    print("Index Finger Up: Move Left")
    print("Peace Sign: Move Right")
    print("Three Fingers Up: Left Mouse Click")
    print("Shaka (Thumb + Pinky): Jump")
    print("Thumbs Down: Stop All Actions")
    print("No Hand/Other Gestures: Stop")
    print("\nYou can use both hands simultaneously!")
    print("Example: Open hand + Shaka = Walk forward while jumping")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)
        
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and detect hands
        results = hands.process(rgb_frame)
        
        # Initialize active gestures set
        active_gestures = set()
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks on the frame
                mp_draw.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS
                )
                
                # Process the gesture for this hand
                gesture = process_hand_gesture(hand_landmarks.landmark)
                if gesture:
                    active_gestures.add(gesture)
        
        # Release all keys first
        pyautogui.keyUp(FORWARD_KEY)
        pyautogui.keyUp(BACKWARD_KEY)
        pyautogui.keyUp(LEFT_KEY)
        pyautogui.keyUp(RIGHT_KEY)
        pyautogui.keyUp(JUMP_KEY)
        pyautogui.mouseUp(button=MOUSE_LEFT)
        
        # If thumbs down is detected, stop all actions
        if 'stop' in active_gestures:
            active_gestures.clear()
        else:
            # Apply all active gestures
            for gesture in active_gestures:
                if gesture == 'forward':
                    pyautogui.keyDown(FORWARD_KEY)
                elif gesture == 'backward':
                    pyautogui.keyDown(BACKWARD_KEY)
                elif gesture == 'left':
                    pyautogui.keyDown(LEFT_KEY)
                elif gesture == 'right':
                    pyautogui.keyDown(RIGHT_KEY)
                elif gesture == 'jump':
                    pyautogui.keyDown(JUMP_KEY)
                elif gesture == 'mouse_left':
                    pyautogui.mouseDown(button=MOUSE_LEFT)
        
        # Display active gestures
        gesture_text = "Active Gestures: " + ", ".join(active_gestures) if active_gestures else "No active gestures"
        cv2.putText(
            frame,
            gesture_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        # Display the frame
        cv2.imshow('Minecraft Gesture Control', frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == "__main__":
    main() 