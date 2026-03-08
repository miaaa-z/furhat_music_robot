import librosa
import numpy as np
import asyncio
import joblib
import random
from test_custom_behaviour import CustomBehaviorTester

AUDIO_PATH = "/Users/miaaa/Desktop/music robot/furhat_music_robot/train/beat_it.wav"
MODELS_PATH = "/Users/miaaa/Desktop/music robot/furhat_music_robot/train_classifier/models"
WINDOW_SIZE = 5.0

# ── Load models ───────────────────────────────────────────────────────────────
head_clf = joblib.load(f"{MODELS_PATH}/head_classifier.pkl")
head_scaler = joblib.load(f"{MODELS_PATH}/head_scaler.pkl")
facial_clf = joblib.load(f"{MODELS_PATH}/facial_classifier.pkl")
facial_scaler = joblib.load(f"{MODELS_PATH}/facial_scaler.pkl")
int_clf = joblib.load(f"{MODELS_PATH}/intensity_classifier.pkl")
int_scaler = joblib.load(f"{MODELS_PATH}/intensity_scaler.pkl")

# ── Gesture maps ──────────────────────────────────────────────────────────────
facial_gesture_map = {
    'big_smile':  ['BigSmile'],
    'smile':      ['Smile'],
    'surprise':   ['Surprise', 'Oh'],
    'frown':      ['ExpressSad', 'BrowFrown'],
    'thoughtful': ['Thoughtful'],
    'negative':   ['ExpressAnger', 'ExpressDisgust'],
    'neutral':    []
}

intensity_to_level = {'Low': 0.4, 'Medium': 0.7, 'High': 1.0}


# ── Feature extraction (must match training order) ────────────────────────────
def extract_features(y_segment, sr):
    mfcc = librosa.feature.mfcc(y=y_segment, sr=sr, n_mfcc=13)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    chroma = librosa.feature.chroma_stft(y=y_segment, sr=sr)
    rms = librosa.feature.rms(y=y_segment)
    sc = librosa.feature.spectral_centroid(y=y_segment, sr=sr)
    sb = librosa.feature.spectral_bandwidth(y=y_segment, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y=y_segment)
    tempo_arr, _ = librosa.beat.beat_track(y=y_segment, sr=sr)
    tempo = float(tempo_arr.item())

    feats = []
    for mat in [mfcc, mfcc_delta, mfcc_delta2, chroma]:
        feats.extend(mat.mean(axis=1).tolist())
        feats.extend(mat.std(axis=1).tolist())
    for mat in [rms, sc, sb, zcr]:
        feats.append(float(mat.mean()))
        feats.append(float(mat.std()))
    feats.append(tempo)

    return np.array(feats).reshape(1, -1), tempo


# ── Beat-synced nod (strategy_B style) ───────────────────────────────────────
async def beat_synced_nod(furhat, beat_times, start_time, window_end):
    for i, beat_time in enumerate(beat_times):
        if beat_time >= window_end:
            break
        elapsed = asyncio.get_event_loop().time() - start_time
        wait = beat_time - elapsed
        if wait > 0:
            await asyncio.sleep(wait)
        if i % 2 == 0:
            await furhat.request_face_headpose(yaw=0, pitch=10, roll=0, relative=True)
        else:
            await furhat.request_face_headpose(yaw=0, pitch=0, roll=0, relative=False)
    await furhat.request_face_headpose(yaw=0, pitch=0, roll=0, relative=False)

async def continuous_sway(furhat, start_time, window_end, intensity=1.0):
    while True:
        elapsed = asyncio.get_event_loop().time() - start_time
        if elapsed >= window_end:
            break
        await furhat.request_gesture_start("Roll", intensity=intensity, duration=1.0)
        await asyncio.sleep(1.5)

async def continuous_shake(furhat, start_time, window_end, intensity=1.0):
    while True:
        elapsed = asyncio.get_event_loop().time() - start_time
        if elapsed >= window_end:
            break
        await furhat.request_gesture_start("Shake", intensity=intensity, duration=1.0)
        await asyncio.sleep(1.0)

# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    custom = CustomBehaviorTester()
    await custom.setup()
    furhat = custom.furhat

    # Load audio
    y, sr = librosa.load(AUDIO_PATH)
    duration = librosa.get_duration(y=y, sr=sr)

    # Get all beat times upfront
    tempo_arr, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    all_beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    print(f"Song duration: {duration:.1f}s | Tempo: {float(tempo_arr):.0f} BPM")
    print(f"Beats detected: {len(all_beat_times)}\n")

    # Reset to center
    await furhat.request_face_headpose(yaw=0, pitch=0, roll=0, relative=False)

    # Play music
    music_url = "https://drive.google.com/uc?export=download&id=1lBeTwrHgxXo982SBbhQBcF_zHU84oYEs"
    await furhat.request_speak_audio(url=music_url, wait=False, abort=False, lipsync=False)

    start_time = asyncio.get_event_loop().time()
    window_start = 0.0

    while window_start + WINDOW_SIZE <= duration:
        window_end = window_start + WINDOW_SIZE

        # Extract features for this 5s window
        s = int(window_start * sr)
        e = int(window_end * sr)
        y_seg = y[s:e]
        feats, tempo = extract_features(y_seg, sr)

        # Predict all three
        head_pred = head_clf.predict(head_scaler.transform(feats))[0]
        facial_pred = facial_clf.predict(facial_scaler.transform(feats))[0]
        int_pred = int_clf.predict(int_scaler.transform(feats))[0]
        intensity_val = intensity_to_level.get(int_pred, 0.7)

        print(f"[{window_start:.0f}s-{window_end:.0f}s] "
              f"head={head_pred}, facial={facial_pred}, intensity={int_pred}")

        # ── Facial expression ──
        options = facial_gesture_map.get(facial_pred, [])
        if options:
            gesture = random.choice(options)
            await furhat.request_gesture_start(gesture, intensity=intensity_val, duration=WINDOW_SIZE)

        # ── Head movement ──
        # ── Head movement ──
        if head_pred == 'nod':
            beats_in_window = all_beat_times[
                (all_beat_times >= window_start) & (all_beat_times < window_end)
                ]
            await beat_synced_nod(furhat, beats_in_window, start_time, window_end)
        elif head_pred == 'shake':
            await continuous_shake(furhat, start_time, window_end, intensity=intensity_val)
        elif head_pred == 'sway':
            await continuous_sway(furhat, start_time, window_end, intensity=intensity_val)

        # Wait until this window ends before moving to next
        elapsed = asyncio.get_event_loop().time() - start_time
        wait = window_end - elapsed
        if wait > 0:
            await asyncio.sleep(wait)

        window_start = window_end

    await custom.cleanup()
    print("\nFinished!")


if __name__ == "__main__":
    asyncio.run(main())
