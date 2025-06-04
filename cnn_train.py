import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import mediapipe as mp
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)
    
# Define gesture mapping
GESTURE_MAPPING = {
    '01_palm': 0,
    '02_l': 1,
    '03_fist': 2,
    '04_fist_moved': 3,
    '05_thumb': 4,
    '06_index': 5,
    '07_ok': 6,
    '08_palm_moved': 7,
    '09_c': 8,
    '10_down': 9
}

class HandGestureDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, augment=False):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.augment = augment

    def __len__(self):
        return len(self.image_paths)

    def augment_landmarks(self, landmarks_array):
        # Random rotation (small angle)
        angle = random.uniform(-10, 10)
        rotation_matrix = cv2.getRotationMatrix2D((0.5, 0.5), angle, 1.0)
        landmarks_reshaped = landmarks_array.reshape(-1, 3)
        landmarks_reshaped[:, :2] = cv2.transform(landmarks_reshaped[:, :2].reshape(-1, 1, 2), rotation_matrix).reshape(-1, 2)
        
        # Random scaling (small scale factor)
        scale = random.uniform(0.9, 1.1)
        landmarks_reshaped *= scale
        
        # Random noise
        noise = np.random.normal(0, 0.01, landmarks_reshaped.shape)
        landmarks_reshaped += noise
        
        return landmarks_reshaped.flatten()

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process image with MediaPipe
        results = hands.process(image)
        
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]
            # Convert landmarks to numpy array
            landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
            landmarks_array = landmarks_array.flatten()  # Flatten to 1D array
            
            # Apply augmentation if enabled
            if self.augment:
                landmarks_array = self.augment_landmarks(landmarks_array)
        else:
            # If no hand detected, use zeros
            landmarks_array = np.zeros(63)  # 21 landmarks * 3 coordinates
        
        if self.transform:
            landmarks_array = self.transform(landmarks_array)
            
        return torch.FloatTensor(landmarks_array), self.labels[idx]

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

def load_dataset(data_dir, is_training=True):
    image_paths = []
    labels = []
    
    # Determine which folders to process
    if is_training:
        folders = [f"{i:02d}" for i in range(9)]  # 00-08 for training
        print("Loading training data...")
    else:
        folders = ["09"]  # 09 for testing
        print("Loading test data...")
    
    for folder in tqdm(folders, desc="Processing folders"):
        folder_path = os.path.join(data_dir, folder)
        if not os.path.exists(folder_path):
            continue
            
        # Process each subfolder in the main folder
        for subfolder in os.listdir(folder_path):
            subfolder_path = os.path.join(folder_path, subfolder)
            if not os.path.isdir(subfolder_path):
                continue
                
            # Get the gesture label from the subfolder name using the mapping
            gesture_name = subfolder
            if gesture_name in GESTURE_MAPPING:
                gesture_label = GESTURE_MAPPING[gesture_name]
                
                # Process all images in the subfolder
                for image_file in os.listdir(subfolder_path):
                    if image_file.endswith(('.png', '.jpg', '.jpeg')):
                        image_paths.append(os.path.join(subfolder_path, image_file))
                        labels.append(gesture_label)
    
    print(f"Found {len(image_paths)} images")
    return image_paths, labels

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, device='cuda'):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_val_accuracy = 0.0
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    current_lr = optimizer.param_groups[0]['lr']
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * train_correct / train_total:.2f}%'
            })
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
        with torch.no_grad():
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                val_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100 * val_correct / val_total:.2f}%'
                })
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        val_accuracy = 100 * val_correct / val_total
        
        # Update learning rate
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        if new_lr != old_lr:
            print(f'\nLearning rate decreased from {old_lr:.6f} to {new_lr:.6f}')
        
        print(f'\nEpoch {epoch+1}/{num_epochs} Summary:')
        print(f'Training Loss: {train_loss:.4f}, Accuracy: {100 * train_correct / train_total:.2f}%')
        print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%')
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_hand_gesture_model.pth')
            print(f'New best model saved! (Validation Accuracy: {val_accuracy:.2f}%)')
    
    print(f'\nTraining completed!')
    print(f'Best validation accuracy: {best_val_accuracy:.2f}%')
    return train_losses, val_losses

def evaluate_model(model, test_loader, device='cuda'):
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    print("\nEvaluating on test data...")
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100 * correct / total
    print(f'\nTest Accuracy: {accuracy:.2f}%')
    
    # Print confusion matrix
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    # Create reverse mapping for labels
    reverse_mapping = {v: k for k, v in GESTURE_MAPPING.items()}
    labels = [reverse_mapping[i] for i in range(len(GESTURE_MAPPING))]
    
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load training and validation data
    data_dir = 'leapGestRecog'
    train_image_paths, train_labels = load_dataset(data_dir, is_training=True)
    
    # Split training data into train and validation sets (80/20)
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        train_image_paths, train_labels, test_size=0.2, random_state=42, stratify=train_labels
    )
    
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    
    # Create datasets with augmentation for training
    train_dataset = HandGestureDataset(X_train, y_train, augment=True)
    val_dataset = HandGestureDataset(X_val, y_val, augment=False)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    model = HandGestureCNN(num_classes=10).to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Train model
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        num_epochs=50, device=device
    )
    
    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('training_history.png')
    plt.close()
    
    # Load and evaluate on test data
    test_image_paths, test_labels = load_dataset(data_dir, is_training=False)
    test_dataset = HandGestureDataset(test_image_paths, test_labels, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Load the best model for testing
    model.load_state_dict(torch.load('best_hand_gesture_model.pth'))
    evaluate_model(model, test_loader, device)

if __name__ == '__main__':
    main()
