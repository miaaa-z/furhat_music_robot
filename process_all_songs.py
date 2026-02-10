import pandas as pd
import librosa
import numpy as np
import os
from pathlib import Path

# Folder paths
annotations_folder = 'annotations'
audio_folder = 'train'
output_file = 'features.csv'

# Get all CSV files except metadata
csv_files = [f for f in os.listdir(annotations_folder)
             if f.endswith('.csv') and f != 'metadata.csv']

print(f"Found {len(csv_files)} annotation files")

# List to store all features
all_features = []

# Process each song
for csv_file in sorted(csv_files):
    # Get song name (remove .csv extension)
    song_name = csv_file[:-4]

    # Build audio file path
    audio_path = os.path.join(audio_folder, f"{song_name}.wav")

    # Check if audio file exists
    if not os.path.exists(audio_path):
        print(f" {song_name}: audio file not found")
        continue

    print(f"\nProcessing: {song_name}")

    # Read CSV annotations
    csv_path = os.path.join(annotations_folder, csv_file)
    df = pd.read_csv(csv_path)

    # Load audio
    y, sr = librosa.load(audio_path, sr=None)
    print(f"  Audio loaded: {sr} Hz, {len(y) / sr:.2f}s, {len(df)} segments")

    # Process each segment
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

        # Store features and labels
        feature_row = {
            'song_name': song_name,
            'start_time': start_time,
            'end_time': end_time,
            'duration': row['duration'],
            'tempo': tempo,
            'energy': energy,
            'brightness': brightness,
            'head_movement': row['head movement'],
            'facial_expression': row['facial expression'],
            'intensity': row['intensity']
        }

        all_features.append(feature_row)

    print(f"   Extracted features for {len(df)} segments")

# Create DataFrame from all features
features_df = pd.DataFrame(all_features)

# Save to CSV
features_df.to_csv(output_file, index=False)

print(f"All features saved to: {output_file}")
print(f"Total segments: {len(features_df)}")

# Show summary
print("\nDataset summary:")
print(f"  Total songs: {features_df['song_name'].nunique()}")
print(f"  Total segments: {len(features_df)}")
print(f"  Features: tempo, energy, brightness")
print(f"  Labels: head_movement, facial_expression, intensity")