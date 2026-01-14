import librosa
import numpy as np
import os

AUDIO_DIR = '/Users/miaaa/Desktop/music robot/3_songs_downloads'

SONGS = {
    'Uptown Funk': {
        'file': 'Uptown_Funk.wav',
    },
    'Someone Like You': {
        'file': 'Adele_Someone_Like_You.wav',
    },
    'Back to Black': {
        'file': 'back_to_black.wav',
    }
}


def analyze_song(audio_path):
    """Extract audio features from a song file."""
    y, sr = librosa.load(audio_path)
    duration = len(y) / sr

    # Global tempo
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    if hasattr(tempo, '__len__'):
        tempo = tempo[0]
    tempo = float(tempo)

    # Global energy (normalized to 0-100)
    rms = librosa.feature.rms(y=y)[0]
    energy_raw = float(np.mean(rms))
    energy = (energy_raw / np.max(rms)) * 100

    # Spectral centroid (brightness)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    avg_centroid = float(np.mean(centroid))

    # Zero crossing rate (complexity)
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    avg_zcr = float(np.mean(zcr))

    # Windowed analysis (5-second windows with 2-second hop)
    window_size = 5 * sr
    hop_size = 2 * sr

    tempos = []
    energies = []
    times = []

    for i in range(0, len(y) - window_size, hop_size):
        window = y[i:i + window_size]

        # Tempo per window
        try:
            t, _ = librosa.beat.beat_track(y=window, sr=sr)
            if hasattr(t, '__len__'):
                t = t[0]
            tempos.append(float(t))
        except:
            tempos.append(None)

        # Energy per window
        rms_window = librosa.feature.rms(y=window)[0]
        energies.append(float(np.mean(rms_window)))
        times.append(i / sr)

    # Tempo statistics
    valid_tempos = [t for t in tempos if t is not None]
    tempo_std = float(np.std(valid_tempos)) if valid_tempos else 0
    tempo_min = float(np.min(valid_tempos)) if valid_tempos else 0
    tempo_max = float(np.max(valid_tempos)) if valid_tempos else 0

    return {
        'duration': duration,
        'global_tempo': tempo,
        'global_energy': energy,
        'spectral_centroid': avg_centroid,
        'zero_crossing_rate': avg_zcr,
        'tempo_std': tempo_std,
        'tempo_range': (tempo_min, tempo_max),
        'energy_range': (min(energies), max(energies)) if energies else (0, 0),
        'windowed_tempos': tempos,
        'windowed_energies': energies,
        'window_times': times
    }


def main():
    print(f"\nAudio directory: {AUDIO_DIR}\n")

    results = {}

    # Analyze all songs
    for name, info in SONGS.items():
        audio_path = os.path.join(AUDIO_DIR, info['file'])

        if not os.path.exists(audio_path):
            print(f"File not found: {info['file']}")
            continue

        print(f"Analyzing: {name}")
        results[name] = analyze_song(audio_path)

    if not results:
        print("No songs analyzed successfully")
        return

    # Print detected features summary
    print("\nAUDIO FEATURES SUMMARY")
    print(f"{'Song':<20} {'BPM':<10} {'Energy':<10} {'Tempo Std':<12}")

    for name in SONGS.keys():
        if name not in results:
            continue

        res = results[name]
        print(f"{name:<20} {res['global_tempo']:<10.1f} {res['global_energy']:<10.1f} {res['tempo_std']:<12.1f}")

    # Detailed features
    print("\nDETAILED FEATURES")

    for name in SONGS.keys():
        if name not in results:
            continue

        res = results[name]
        print(f"\n{name}:")
        print(f"  Duration: {res['duration']:.2f}s")
        print(f"  Global tempo: {res['global_tempo']:.1f} BPM")
        print(f"  Global energy: {res['global_energy']:.1f}")
        print(f"  Tempo stability: {res['tempo_std']:.1f} (std dev, lower = more stable)")
        print(f"  Tempo range: {res['tempo_range'][0]:.1f} ~ {res['tempo_range'][1]:.1f} BPM")
        print(f"  Spectral centroid: {res['spectral_centroid']:.1f} Hz (brightness)")
        print(f"  Zero crossing rate: {res['zero_crossing_rate']:.4f} (complexity)")
        print(f"  Energy range: {res['energy_range'][0]:.4f} ~ {res['energy_range'][1]:.4f}")

    # Windowed data arrays
    print("\nWINDOWED DATA ARRAYS")

    for name in SONGS.keys():
        if name not in results:
            continue

        res = results[name]
        print(f"\n{name}:")
        print(f"  times = {[round(t, 1) for t in res['window_times']]}")
        print(f"  tempos = {[round(t, 1) if t is not None else None for t in res['windowed_tempos']]}")
        print(f"  energies = {[round(e, 4) for e in res['windowed_energies']]}")

    print()


if __name__ == "__main__":
    main()