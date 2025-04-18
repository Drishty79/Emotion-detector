import os

song_path = r"C:\Users\Drishty\OneDrive\Desktop\emotion detector\songs\happy_song.mp3.mp3"

if os.path.exists(song_path):
    print("✅ File exists!")
else:
    print("❌ File not found! Check path and OneDrive sync.")
