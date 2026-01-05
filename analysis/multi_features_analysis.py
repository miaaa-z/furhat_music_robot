import librosa
import numpy as np
import os

AUDIO_DIR = '/Users/miaaa/Desktop/music robot/3_songs_downloads'

SONGS = {
    'Uptown Funk': {
        'file': 'Uptown_Funk.wav',
        'real_bpm': 116,
        'real_energy': 61,
        'real_danceability': 86,
    },
    'Someone Like You': {
        'file': 'Adele_Someone_Like_You.wav',
        'real_bpm': 135,
        'real_energy': 33,
        'real_danceability': 56,
    },
    'Back to Black': {
        'file': 'back_to_black.wav',
        'real_bpm': 124,
        'real_energy': 73,
        'real_danceability': 49,
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
        'energy_range': (min(energies), max(energies)) if energies else (0, 0)
    }


def calculate_liveliness_score(bpm, energy, danceability):
    """Calculate overall liveliness score: BPM×30% + Energy×30% + Danceability×40%"""
    return bpm * 0.3 + energy * 0.3 + danceability * 0.4


def get_motion_parameters(bpm, energy, danceability):
    """Generate robot motion parameters based on audio features."""
    beat_interval = 60 / bpm

    # Amplitude based on energy
    if energy > 60:
        amplitude = "Large (±30°)"
    elif energy > 40:
        amplitude = "Medium (±20°)"
    else:
        amplitude = "Small (±10°)"

    # Complexity based on danceability
    if danceability > 70:
        complexity = "Frequent changes"
    elif danceability > 50:
        complexity = "Moderate changes"
    else:
        complexity = "Simple repetition"

    # LED suggestion
    if danceability > 70:
        led = "Bright warm"
    elif energy > 60:
        led = "Bright"
    else:
        led = "Soft"

    return {
        'beat_interval': beat_interval,
        'amplitude': amplitude,
        'complexity': complexity,
        'led': led
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

    # Print BPM comparison
    print("\n" + "=" * 90)
    print("BPM ANALYSIS")
    print("=" * 90)
    print(f"{'Song':<20} {'Detected BPM':<15} {'Real BPM':<12} {'Error':<15} {'Stability (±)':<15}")
    print("-" * 90)

    for name, info in SONGS.items():
        if name not in results:
            continue

        res = results[name]
        real_bpm = info['real_bpm']
        detected_bpm = res['global_tempo']
        error = abs(detected_bpm - real_bpm)
        error_pct = (error / real_bpm) * 100

        print(f"{name:<20} {detected_bpm:<15.1f} {real_bpm:<12} "
              f"{error:.1f} ({error_pct:.1f}%){' ':<3} {res['tempo_std']:.1f} BPM")

    # Print energy comparison
    print(f"\n{'Song':<20} {'Detected Energy':<18} {'Real Energy':<15} {'Motion Guide':<30}")
    print("-" * 90)

    for name, info in SONGS.items():
        if name not in results:
            continue

        res = results[name]
        detected = res['global_energy']
        real = info['real_energy']

        if detected > 60:
            desc = "High → Large amplitude"
        elif detected > 40:
            desc = "Medium → Medium amplitude"
        else:
            desc = "Low → Small amplitude"

        print(f"{name:<20} {detected:<18.1f} {real:<15} {desc:<30}")

    # Liveliness scores
    print("\n" + "=" * 90)
    print("LIVELINESS SCORE (BPM×30% + Energy×30% + Danceability×40%)")
    print("=" * 90)

    scores = {}
    for name, info in SONGS.items():
        if name not in results:
            continue

        score = calculate_liveliness_score(
            info['real_bpm'],
            info['real_energy'],
            info['real_danceability']
        )
        scores[name] = score

        print(f"{name}: {score:.1f}")

    max_song = max(scores, key=scores.get)
    print(f"\nHighest score: {max_song} ({scores[max_song]:.1f})")

    # Robot motion parameters
    print("\n" + "=" * 90)
    print("ROBOT MOTION PARAMETERS")
    print("=" * 90)
    print(f"{'Song':<20} {'Beat Interval':<15} {'Amplitude':<18} {'Complexity':<20} {'LED':<15}")
    print("-" * 90)

    for name, info in SONGS.items():
        if name not in results:
            continue

        params = get_motion_parameters(
            info['real_bpm'],
            info['real_energy'],
            info['real_danceability']
        )

        print(f"{name:<20} {params['beat_interval']:.2f}s{' ' * 8} "
              f"{params['amplitude']:<18} {params['complexity']:<20} {params['led']:<15}")

    # Additional features
    print("\n" + "=" * 90)
    print("DETAILED FEATURES")
    print("=" * 90)

    for name, info in SONGS.items():
        if name not in results:
            continue

        res = results[name]
        print(f"\n{name}:")
        print(f"  Duration: {res['duration']:.2f}s")
        print(f"  Tempo stability: {res['tempo_std']:.1f} (std dev, lower = more stable)")
        print(f"  Tempo range: {res['tempo_range'][0]:.1f} ~ {res['tempo_range'][1]:.1f} BPM")
        print(f"  Spectral centroid: {res['spectral_centroid']:.1f} Hz (brightness)")
        print(f"  Zero crossing rate: {res['zero_crossing_rate']:.4f} (complexity)")

    print("\n" + "=" * 90 + "\n")


if __name__ == "__main__":
    main()