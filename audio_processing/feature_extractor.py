import librosa
import numpy as np

audio_path = '/Users/miaaa/Desktop/music robot/3_songs_downloads/Adele_Someone_Like_You.wav'
y, sr = librosa.load(audio_path)

# 1. TEMPO ANALYSIS
duration = librosa.get_duration(y=y, sr=sr)
window_size = 10.0

print(f"\n[TEMPO ANALYSIS - 5-second windows]")
print(f"   Total duration: {duration:.2f} seconds")
print(f"   Number of windows: {int(duration / window_size) + 1}")

tempo_values = []
for i, start_time in enumerate(np.arange(0, duration, window_size)):
    # Calculate end time for this window
    end_time = min(start_time + window_size, duration)

    # Convert time to sample indices
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)

    # Extract audio segment
    y_segment = y[start_sample:end_sample]

    # Calculate tempo for this segment
    tempo, _ = librosa.beat.beat_track(y=y_segment, sr=sr)

    # Extract scalar value properly
    tempo_value = tempo.item() if hasattr(tempo, 'item') else float(tempo)
    tempo_values.append(tempo_value)

    print(f"   Window {i + 1} ({start_time:.1f}s - {end_time:.1f}s): {tempo_value:.2f} BPM")

# Overall tempo statistics
mean_tempo = np.mean(tempo_values)
std_tempo = np.std(tempo_values)
print(f"\n   Average tempo across all windows: {mean_tempo:.2f} BPM")
print(f"   Standard deviation: {std_tempo:.2f} BPM")

# ===== 2. BEAT TRACKING (First 10 beats) =====
tempo_overall, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

# Convert beat frames to time points (in seconds)
beat_times = librosa.frames_to_time(beat_frames, sr=sr)

# Extract scalar tempo value properly
tempo_overall_value = tempo_overall.item() if hasattr(tempo_overall, 'item') else float(tempo_overall)

print(f"\n[BEAT DETECTION]")
print(f"   Overall tempo (BPM): {tempo_overall_value:.2f}")
print(f"   Total number of beats detected: {len(beat_times)}")
print(f"   Time points of first 10 beats (seconds):")
for i, beat_time in enumerate(beat_times[:10], 1):
    print(f"      Beat {i}: {beat_time:.3f}s")

# 3. ENERGY (RMS) - OVERALL
# Calculate Root Mean Square (RMS) energy for entire song
rms = librosa.feature.rms(y=y)[0]

# Calculate statistics
mean_energy = np.mean(rms)
min_energy = np.min(rms)
max_energy = np.max(rms)
median_energy = np.median(rms)
std_energy = np.std(rms)

print(f"\n[ENERGY (RMS) - OVERALL SONG]")
print(f"   Minimum: {min_energy:.6f}")
print(f"   Maximum: {max_energy:.6f}")
print(f"   Mean: {mean_energy:.6f}")
print(f"   Median: {median_energy:.6f}")
print(f"   Standard Deviation: {std_energy:.6f}")

# Calculate percentiles for categorization
p33 = np.percentile(rms, 33)
p67 = np.percentile(rms, 67)

print(f"\n[ENERGY PERCENTILES]")
print(f"   25th percentile: {np.percentile(rms, 25):.6f}")
print(f"   33rd percentile: {p33:.6f}")
print(f"   50th percentile (median): {median_energy:.6f}")
print(f"   67th percentile: {p67:.6f}")
print(f"   75th percentile: {np.percentile(rms, 75):.6f}")

# Suggested categorization thresholds
print(f"\n[SUGGESTED ENERGY THRESHOLDS]")
print(f"   Low energy:    0.000000 - {p33:.6f}")
print(f"   Medium energy: {p33:.6f} - {p67:.6f}")
print(f"   High energy:   {p67:.6f} - {max_energy:.6f}")

# ===== 4. ENERGY ANALYSIS (Every 5 seconds) - OPTIONAL =====
print(f"\n[ENERGY ANALYSIS - 5-second windows]")

energy_values = []
for i, start_time in enumerate(np.arange(0, duration, window_size)):
    # Calculate end time for this window
    end_time = min(start_time + window_size, duration)

    # Convert time to sample indices
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)

    # Extract audio segment
    y_segment = y[start_sample:end_sample]

    # Calculate RMS energy for this segment
    rms_segment = librosa.feature.rms(y=y_segment)[0]
    mean_energy_segment = np.mean(rms_segment)
    energy_values.append(mean_energy_segment)

    print(f"   Window {i + 1} ({start_time:.1f}s - {end_time:.1f}s): {mean_energy_segment:.6f}")

# Energy variation statistics
mean_energy_windowed = np.mean(energy_values)
std_energy_windowed = np.std(energy_values)
print(f"\n   Average energy across all windows: {mean_energy_windowed:.6f}")
print(f"   Standard deviation: {std_energy_windowed:.6f}")
print(f"   Energy variation range: {np.min(energy_values):.6f} - {np.max(energy_values):.6f}")

# ===== 5. ADDITIONAL INFO =====
print(f"\n[ADDITIONAL INFORMATION]")
print(f"   Audio duration: {duration:.2f} seconds")
print(f"   Sample rate: {sr} Hz")
print(f"   Number of samples: {len(y)}")
print(f"   (Note: Number of samples = sample_rate x duration)")
print(f"   ({sr} samples/sec x {duration:.2f} sec = {len(y)} samples)")
