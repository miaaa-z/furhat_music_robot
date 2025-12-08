import librosa
import matplotlib.pyplot as plt


audio_path = '/Users/miaaa/Desktop/music robot/3_songs_downloads/back_to_black.wav'
y, sr = librosa.load(audio_path)

# beat tracking
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

# Convert the frame to a time point (in seconds)
beat_times = librosa.frames_to_time(beat_frames, sr=sr)

print(f" (BPM): {tempo}")
print(f"The number of the beats: {len(beat_times)} ")
print(f"The time points of the first 10 beats）: {beat_times[:10]}")

# visualization
plt.figure(figsize=(14, 5))
librosa.display.waveshow(y, sr=sr, alpha=0.6)
plt.vlines(beat_times, -1, 1, color='r', alpha=0.8, linestyle='--', label='Beats')
plt.title('Back to Black - Beat Detection')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.legend()
plt.tight_layout()
plt.show()
