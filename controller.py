import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import numpy as np
from cnn_train import GESTURE_MAPPING
from pynput.keyboard import Key, Controller as KeyboardController
from pynput.mouse import Button, Controller as MouseController
import time
import threading

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Create reverse mapping for gesture names
REVERSE_GESTURE_MAPPING = {v: k for k, v in GESTURE_MAPPING.items()}

# Define gesture to action mapping
GESTURE_ACTIONS = {
    '01_palm': 'Moving Forward',
    '02_l': 'Moving Left',
    '03_fist': 'Moving Forward + Left Click',
    '04_fist_moved': 'Moving Forward + Left Click',
    '05_thumb': 'Moving Forward + Left Click',
    '06_index': 'Moving Right',
    '07_ok': 'Moving Forward + Jump',
    '08_palm_moved': 'Moving Forward',
    '09_c': 'Moving Forward + Left Click',
    '10_down': 'Moving Backward'
}

class MinecraftController:
    def __init__(self):
        self.keyboard = KeyboardController()
        self.mouse = MouseController()
        self.current_action = None
        self.last_action = None
        self.action_thread = None
        self.running = True
        self.pressed_keys = set()

    def press_key(self, key):
        if key not in self.pressed_keys:
            self.keyboard.press(key)
            self.pressed_keys.add(key)

    def release_key(self, key):
        if key in self.pressed_keys:
            self.keyboard.release(key)
            self.pressed_keys.remove(key)

    def release_all_keys(self):
        for key in list(self.pressed_keys):
            self.keyboard.release(key)
        self.pressed_keys.clear()

    def handle_action(self, action):
        if action == self.last_action:
            return

        # Stop previous action
        self.release_all_keys()
        if self.last_action and 'Left Click' in self.last_action:
            self.mouse.release(Button.left)

        # Start new action
        if action == 'Moving Forward':
            self.press_key('w')
        elif action == 'Moving Backward':
            self.press_key('s')
        elif action == 'Moving Left':
            self.press_key('a')
        elif action == 'Moving Right':
            self.press_key('d')
        elif action == 'Moving Forward + Jump':
            self.press_key('w')
            self.press_key(Key.space)
        elif action == 'Moving Forward + Left Click':
            self.press_key('w')
            self.mouse.press(Button.left)

        self.last_action = action

    def action_loop(self):
        while self.running:
            if self.current_action:
                self.handle_action(self.current_action)
            time.sleep(0.1)

    def cleanup(self):
        self.running = False
        self.release_all_keys()
        if self.last_action and 'Left Click' in self.last_action:
            self.mouse.release(Button.left)

# Define the model architecture to match the saved model
class HandGestureCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(HandGestureCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(63, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HandGestureCNN(num_classes=10).to(device)
model.load_state_dict(torch.load('best_hand_gesture_model.pth'))
model.eval()

def process_landmarks(landmarks):
    # Convert landmarks to the same format as training data
    landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
    landmarks_array = landmarks_array.flatten()
    return torch.FloatTensor(landmarks_array).unsqueeze(0)  # Add batch dimension

def main():
    # Initialize Minecraft controller
    mc_controller = MinecraftController()
    
    # Start action loop in a separate thread
    mc_controller.action_thread = threading.Thread(target=mc_controller.action_loop)
    mc_controller.action_thread.start()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Set window properties
    cv2.namedWindow('Minecraft Controller', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Minecraft Controller', 1280, 720)
    
    try:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Failed to read from webcam")
                continue
                
            # Flip the image horizontally for a later selfie-view display
            image = cv2.flip(image, 1)
            
            # Convert the BGR image to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process the image and detect hands
            results = hands.process(image_rgb)
            
            # Draw hand landmarks and predict gesture
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks
                    mp_drawing.draw_landmarks(
                        image, 
                        hand_landmarks, 
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                    )
                    
                    # Process landmarks and predict gesture
                    landmarks_tensor = process_landmarks(hand_landmarks)
                    landmarks_tensor = landmarks_tensor.to(device)
                    
                    with torch.no_grad():
                        outputs = model(landmarks_tensor)
                        _, predicted = torch.max(outputs.data, 1)
                        gesture_idx = predicted.item()
                        
                        # Get gesture name and corresponding action
                        gesture_name = REVERSE_GESTURE_MAPPING[gesture_idx]
                        action = GESTURE_ACTIONS[gesture_name]
                        
                        # Update Minecraft controller
                        mc_controller.current_action = action
                        
                        # Display action
                        cv2.putText(
                            image,
                            f"Action: {action}",
                            (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA
                        )
            else:
                # No hand detected, stop current action
                mc_controller.current_action = None
            
            # Display instructions
            cv2.putText(
                image,
                f"Press 'q' to quit",
                (50, image.shape[0] - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )
            
            # Show the image
            cv2.imshow('Minecraft Controller', image)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        mc_controller.cleanup()
        if mc_controller.action_thread:
            mc_controller.action_thread.join()

if __name__ == '__main__':
    main() 