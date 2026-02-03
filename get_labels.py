import librosa
import numpy as np
import jams
from collections import Counter


def read_emotion_labels(jams_path):
    """Read emotion_label annotations from JAMS file"""
    jam = jams.load(jams_path)
    labels = []

    for ann in jam.annotations:
        if ann.namespace == 'tag_open' and \
                ann.annotation_metadata.data_source == 'emotion_label':
            for obs in ann.data:
                labels.append((obs.time, obs.time + obs.duration, obs.value))

    return labels


def extract_features(audio_segment, sr):
    """Extract 3 features: tempo, brightness, energy"""
    # 1. Tempo (BPM)
    tempo, _ = librosa.beat.beat_track(y=audio_segment, sr=sr)
    tempo_val = tempo.item()

    # 2. Brightness (Spectral Centroid, Hz)
    brightness = float(np.mean(
        librosa.feature.spectral_centroid(y=audio_segment, sr=sr)
    ))

    # 3. Energy (RMS)
    energy = float(np.mean(
        librosa.feature.rms(y=audio_segment)
    ))

    return [tempo_val, brightness, energy]


def process_song(audio_path, jams_path, window_size=5.0, min_overlap=3.0):
    """Process one song: split windows, extract features, match labels"""
    print(f"\nProcessing: {audio_path}\n")

    # Step 1: Load audio
    y, sr = librosa.load(audio_path)
    duration = librosa.get_duration(y=y, sr=sr)
    print(f"Audio duration: {duration:.2f}s\n")

    # Step 2: Read annotations
    annotations = read_emotion_labels(jams_path)
    print(f"Number of annotations: {len(annotations)}\n")

    # Step 3: Split windows, extract features, match labels
    samples = []
    num_windows = int(duration // window_size)

    for i in range(num_windows):
        window_start = i * window_size
        window_end = window_start + window_size

        # Extract audio segment for this window
        start_sample = int(window_start * sr)
        end_sample = int(window_end * sr)
        audio_segment = y[start_sample:end_sample]

        # Skip if audio is too short (less than 60% of window size)
        if len(audio_segment) < sr * window_size * 0.6:
            continue

        # Extract features
        try:
            features = extract_features(audio_segment, sr)
        except:
            continue

        # Find the best label for this window
        # Create a dictionary to record different seconds for each label
        label_total_time = {}

        # Create a dictionary to record first appearance order for each label
        label_order = {}

        # Iterate through all annotations
        for order, (ann_start, ann_end, label) in enumerate(annotations):
            # Calculate overlap time between window and annotation
            overlap_start = max(window_start, ann_start)
            overlap_end = min(window_end, ann_end)
            overlap_duration = max(0, overlap_end - overlap_start)

            # If there is overlap
            if overlap_duration > 0:
                # Accumulate overlap time for this label
                if label in label_total_time:
                    label_total_time[label] += overlap_duration
                else:
                    label_total_time[label] = overlap_duration

                # Record first appearance order (order is the second in the window the label appears)
                if label not in label_order:
                    label_order[label] = order

        # Select the label with longest accumulated time
        if label_total_time:
            best_label = None
            max_overlap = 0

            for label, total_time in label_total_time.items():
                # If this label has longer time
                if total_time > max_overlap:
                    max_overlap = total_time
                    best_label = label
                # If time is equal, choose the one that appeared first
                elif total_time == max_overlap:
                    if label_order[label] < label_order[best_label]:
                        best_label = label
        else:
            best_label = None
            max_overlap = 0

        # Save sample if overlap >= min_overlap
        if max_overlap >= min_overlap and best_label:
            samples.append((features, best_label))

            # Print detailed overlap info
            overlap_info = ", ".join([f"{label}:{time:.1f}s" for label, time in label_total_time.items()])
            print(f"[{i + 1:2d}] {window_start:5.1f}-{window_end:5.1f}s → {best_label:15s}   ({overlap_info})")
        else:
            # Skip this window if overlap is too short
            overlap_info = ", ".join([f"{label}:{time:.1f}s" for label, time in label_total_time.items()])
            print(f"[{i + 1:2d}] {window_start:5.1f}-{window_end:5.1f}s → SKIPPED ({overlap_info})")

    print(f"\nGenerated {len(samples)} training windows\n")
    return samples


if __name__ == "__main__":
    samples = process_song('/train/uptown_funk.wav',
                           '/Users/miaaa/Desktop/music robot/furhat_music_robot/jams/uptown_2.jams')

    # Display results
    print("Sample preview (first 10):\n")
    for i, (features, label) in enumerate(samples[:10]):
        print(f"Sample {i + 1}: {label}")
        print(f"  Tempo:      {features[0]:.2f} BPM")
        print(f"  Brightness: {features[1]:.2f} Hz")
        print(f"  Energy:     {features[2]:.4f}\n")

    # Count label distribution
    print("\nLabel distribution:")
    label_counts = Counter([label for _, label in samples])
    for label, count in sorted(label_counts.items()):
        print(f"{label:20s}: {count:3d} samples")
