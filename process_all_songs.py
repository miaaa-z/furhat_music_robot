import os
import pickle
from collections import Counter
from get_labels import process_song


def process_all_songs(audio_folder, jams_folder, output_file='dataset.pkl'):
    """
    Process all songs in the folders

    Args:
        audio_folder: folder containing .wav files
        jams_folder: folder containing .jams files
        output_file: output pickle file name

    Returns:
        all_samples: list of (features, label) for all songs
    """

    # Get all wav files
    wav_files = [f for f in os.listdir(audio_folder) if f.endswith('.wav')]
    wav_files.sort()  # Sort for consistent order

    print(f"Found {len(wav_files)} audio files\n")
    print("=" * 60)

    all_samples = []
    song_sample_counts = {}  # Track samples per song

    # Process each song
    for i, wav_file in enumerate(wav_files, 1):
        # Get base name (without extension)
        base_name = wav_file.replace('.wav', '')

        # Try to find matching JAMS file
        jams_file = None
        possible_jams = [
            f"{base_name}.jams",
            f"{base_name}_2.jams",
            f"{base_name}_1.jams",
        ]

        for possible in possible_jams:
            if os.path.exists(os.path.join(jams_folder, possible)):
                jams_file = possible
                break

        if jams_file is None:
            print(f"[{i}/{len(wav_files)}]  Skipping {wav_file}: no matching JAMS file found")
            print("\n")
            continue

        # Full paths
        audio_path = os.path.join(audio_folder, wav_file)
        jams_path = os.path.join(jams_folder, jams_file)

        print(f"\n[{i}/{len(wav_files)}] Processing: {wav_file}")
        print(f"           JAMS file: {jams_file}")
        print("-" * 60)

        # Process this song
        try:
            samples = process_song(audio_path, jams_path)
            all_samples.extend(samples)
            song_sample_counts[wav_file] = len(samples)
            print(f"✓ Added {len(samples)} samples from {wav_file}")
        except Exception as e:
            print(f"✗ Error processing {wav_file}: {e}")
            song_sample_counts[wav_file] = 0

        print("=" * 60)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nTotal songs processed: {len(song_sample_counts)}")
    print(f"Total samples generated: {len(all_samples)}\n")

    # Samples per song
    print("Samples per song:")
    for song, count in song_sample_counts.items():
        print(f"  {song:30s}: {count:3d} samples")

    # Label distribution
    print("\nLabel distribution:")
    label_counts = Counter([label for _, label in all_samples])
    for label, count in sorted(label_counts.items()):
        print(f"  {label:20s}: {count:4d} samples")

    # Save to pickle file
    print(f"\nSaving dataset to {output_file}...")
    with open(output_file, 'wb') as f:
        pickle.dump(all_samples, f)
    print(" Dataset saved successfully!")

    return all_samples


if __name__ == "__main__":
    audio_folder = 'train'
    jams_folder = 'jams'
    output_file = 'dataset.pkl'

    # Process all songs
    all_samples = process_all_songs(audio_folder, jams_folder, output_file)

    # Display first 3 samples
    print("\n")
    for i, (features, label) in enumerate(all_samples[:3], 1):
        print(f"\nSample {i}: {label}")
        print(f"  Tempo:      {features[0]:.2f} BPM")
        print(f"  Brightness: {features[1]:.2f} Hz")
        print(f"  Energy:     {features[2]:.4f}")
