import pandas as pd
import librosa
import numpy as np

# File paths
csv_path = 'annotations/APT.csv'
audio_path = 'train/apt.wav'

print("Processing song: APT")

# 1. Read CSV
df = pd.read_csv(csv_path)
print(f"Found {len(df)} segments\n")

# 2. Load audio
y, sr = librosa.load(audio_path, sr=None)
print(f"Audio: sampling rate {sr} Hz, duration {len(y) / sr:.2f}s\n")

# 3. Process each segment
for idx, row in df.iterrows():
    start_time = row['start_time']
    end_time = row['end_time']

    # Convert time to sample indices
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)

    # Extract audio segment
    segment = y[start_sample:end_sample]

    # Extract features
    # 1. Tempo
    tempo_result = librosa.beat.beat_track(y=segment, sr=sr)
    if isinstance(tempo_result, tuple):
        tempo = tempo_result[0]
    else:
        tempo = tempo_result
    if isinstance(tempo, np.ndarray):
        tempo = float(tempo[0])
    else:
        tempo = float(tempo)

    # 2. Energy (RMS)
    rms = librosa.feature.rms(y=segment)
    energy = float(np.mean(rms))

    # 3. Brightness (Spectral Centroid)
    spectral_centroids = librosa.feature.spectral_centroid(y=segment, sr=sr)
    brightness = float(np.mean(spectral_centroids))

    # Print the results
    print(f"{idx + 1:2d}  ({start_time:5.1f}s-{end_time:5.1f}s): "
          f"Tempo={tempo:6.2f} BPM,  Energy={energy:.4f},  Brightness={brightness:7.2f} Hz")
