import pandas as pd
import librosa
import numpy as np
import os


annotations_folder = 'annotations'
audio_folder = 'train'
output_file = 'features_v2.csv'

#  extract all features for one segment
# Returns a flat dict of ~66 numbers


def extract_features(segment, sr):
    features = {}

    #  1. MFCC (13 coefficients × mean + std = 26 values) 
    mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f'mfcc_{i+1}_mean'] = float(np.mean(mfcc[i]))
        features[f'mfcc_{i+1}_std'] = float(np.std(mfcc[i]))

    #  2. MFCC Delta / rate of change (13 × mean+std = 26) 
    mfcc_delta = librosa.feature.delta(mfcc)
    for i in range(13):
        features[f'mfcc_delta_{i+1}_mean'] = float(np.mean(mfcc_delta[i]))
        features[f'mfcc_delta_{i+1}_std'] = float(np.std(mfcc_delta[i]))

    #  3. MFCC Delta-Delta / acceleration (13 × mean+std = 26) 
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    for i in range(13):
        features[f'mfcc_delta2_{i+1}_mean'] = float(np.mean(mfcc_delta2[i]))
        features[f'mfcc_delta2_{i+1}_std'] = float(np.std(mfcc_delta2[i]))

    #  4. Chroma (12 pitch classes × mean+std = 24) 
    chroma = librosa.feature.chroma_stft(y=segment, sr=sr)
    for i in range(12):
        features[f'chroma_{i+1}_mean'] = float(np.mean(chroma[i]))
        features[f'chroma_{i+1}_std'] = float(np.std(chroma[i]))

    #  5. RMS Energy (mean + std = 2) 
    rms = librosa.feature.rms(y=segment)
    features['rms_mean'] = float(np.mean(rms))
    features['rms_std']  = float(np.std(rms))

    #  6. Spectral Centroid / Brightness (mean + std = 2) 
    centroid = librosa.feature.spectral_centroid(y=segment, sr=sr)
    features['spectral_centroid_mean'] = float(np.mean(centroid))
    features['spectral_centroid_std']  = float(np.std(centroid))

    #  7. Spectral Bandwidth (mean + std = 2) 
    bandwidth = librosa.feature.spectral_bandwidth(y=segment, sr=sr)
    features['spectral_bandwidth_mean'] = float(np.mean(bandwidth))
    features['spectral_bandwidth_std']  = float(np.std(bandwidth))

    #  8. Zero Crossing Rate (mean + std = 2) 
    zcr = librosa.feature.zero_crossing_rate(y=segment)
    features['zcr_mean'] = float(np.mean(zcr))
    features['zcr_std']  = float(np.std(zcr))

    #  9. Tempo (single value, will be kept for robot nod speed control)
    tempo_result = librosa.beat.beat_track(y=segment, sr=sr)
    tempo = tempo_result[0] if isinstance(tempo_result, tuple) else tempo_result
    tempo = float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo)
    features['tempo'] = tempo

    return features


csv_files = [f for f in os.listdir(annotations_folder)
             if f.endswith('.csv') and f != 'metadata.csv']

print(f"Found {len(csv_files)} annotation files")

all_rows = []

for csv_file in sorted(csv_files):
    song_name = csv_file[:-4]
    audio_path = os.path.join(audio_folder, f"{song_name}.wav")

    if not os.path.exists(audio_path):
        print(f"  SKIP {song_name}: audio file not found")
        continue

    print(f"\nProcessing: {song_name}")
    df = pd.read_csv(os.path.join(annotations_folder, csv_file))
    y, sr = librosa.load(audio_path, sr=None)
    print(f"  {sr} Hz | {len(y)/sr:.1f}s | {len(df)} segments")

    for idx, row in df.iterrows():
        start_sample = int(row['start_time'] * sr)
        end_sample = int(row['end_time'] * sr)
        segment = y[start_sample:end_sample]

        # Skip segments that are too short (< 0.5s)
        if len(segment) < sr * 0.5:
            print(f"    Skipping segment {idx}: too short ({len(segment)/sr:.2f}s)")
            continue

        feats = extract_features(segment, sr)

        meta = {
            'song_name':         song_name,
            'start_time':        row['start_time'],
            'end_time':          row['end_time'],
            'duration':          row['duration'],
            'head_movement':     row['head movement'],
            'facial_expression': row['facial expression'],
            'intensity':         row['intensity'],
        }

        all_rows.append({**meta, **feats})

    print(f"  Done: {len(df)} segments")


# Save
features_df = pd.DataFrame(all_rows)
features_df.to_csv(output_file, index=False)

n_feature_cols = len(features_df.columns) - 7
print(f"\nSaved to: {output_file}")
print(f"Total segments : {len(features_df)}")
print(f"Total songs    : {features_df['song_name'].nunique()}")
print(f"Feature columns: {n_feature_cols}")