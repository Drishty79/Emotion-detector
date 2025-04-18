import os
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

MODEL_PATH = "models/emotion_detection_model.h5"

#  Check if model already exists
if os.path.exists(MODEL_PATH):
    print("✅ Model already exists. Skipping training...")
else:
    print(" Training model...")

    # Load preprocessed data
    X_train = np.load("X_train.npy")
    y_train = np.load("y_train.npy")
    X_test = np.load("X_test.npy")
    y_test = np.load("y_test.npy")

    # Define CNN model
    model = Sequential([
        Conv2D(64, (3,3), activation='relu', input_shape=(48, 48, 1)),
        MaxPooling2D(2,2),

        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),

        Conv2D(256, (3,3), activation='relu'),
        MaxPooling2D(2,2),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(7, activation='softmax')  # 7 emotion classes
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    early_stop = EarlyStopping(monitor='val_loss', patience=5)
    history = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test), callbacks=[early_stop])

    # Save the model
    model.save(MODEL_PATH)

    print(f"🎉 Model trained and saved at {MODEL_PATH}")
