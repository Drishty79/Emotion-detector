import os
import train
import detect

MODEL_PATH = "models/emotion_model.h5"

def main():
    print("Choose an option:")
    print("1: Train Model (Only if not already trained)")
    print("2: Start Real-Time Detection")

    choice = input("Enter your choice: ")

    if choice == "1":
        train.train()  # Call the modified train function
    elif choice == "2":
        detected_emotion = detect.detect_emotions()
        from music_player import play_music
        play_music(detected_emotion)
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()
