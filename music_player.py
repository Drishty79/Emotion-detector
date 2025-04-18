import pygame
import os

def play_music(emotion):
    song_path = f"C:/Users/Drishty/OneDrive/Desktop/emotion detector/songs/{emotion}_song.mp3"
    
    if os.path.exists(song_path):
        print(f"🎶 Playing: {song_path}")
        
        pygame.mixer.init()
        pygame.mixer.music.load(song_path)
        pygame.mixer.music.play()

        # Wait for the song to finish playing
        while pygame.mixer.music.get_busy():
            continue  # Keeps the program running while music is playing
        
    else:
        print(f"❌ Song file not found at: {song_path}")
