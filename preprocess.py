import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Define dataset path
DATASET_PATH = "dataset/FER2013/"

# Define emotion labels
emotion_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Function to load images
def load_images_from_folder(folder):
    images = []
    labels = []
    
    for label, emotion in enumerate(emotion_labels):
        emotion_folder = os.path.join(folder, emotion)
        
        for img_name in os.listdir(emotion_folder):
            img_path = os.path.join(emotion_folder, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read image in grayscale
            
            if img is not None:
                img = cv2.resize(img, (48, 48))  # Resize to 48x48
                images.append(img)
                labels.append(label)
    
    return np.array(images), np.array(labels)

# Load train and test images
X_train, y_train = load_images_from_folder(os.path.join(DATASET_PATH, "train"))
X_test, y_test = load_images_from_folder(os.path.join(DATASET_PATH, "test"))

# Normalize pixel values
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape for CNN input
X_train = X_train.reshape(-1, 48, 48, 1)
X_test = X_test.reshape(-1, 48, 48, 1)

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, num_classes=7)
y_test = to_categorical(y_test, num_classes=7)

print(f"Training data shape: {X_train.shape}, Training labels shape: {y_train.shape}")
print(f"Testing data shape: {X_test.shape}, Testing labels shape: {y_test.shape}")

# Save preprocessed data (optional)
np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)
np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)

# Display some images
plt.figure(figsize=(10, 5))
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(X_train[i].reshape(48, 48), cmap='gray')
    plt.title(f"Emotion: {emotion_labels[np.argmax(y_train[i])]}")
    plt.axis('off')
plt.show()
