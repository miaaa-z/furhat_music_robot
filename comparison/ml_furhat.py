import librosa
import numpy as np
import asyncio
import joblib
from furhat_realtime_api import AsyncFurhatClient
from test_custom_behaviour import CustomBehaviorTester
from center_down_2 import strategy_A as do_nod_strategy_a


# AUDIO_PATH = "/Users/miaaa/Desktop/music robot/test/fortnight.wav"
AUDIO_PATH = "/Users/miaaa/Desktop/music robot/test/zoo.wav"
# AUDIO_PATH = "/Users/miaaa/Desktop/music robot/furhat_music_robot/train/beat_it.wav"
MODELS_DIR = "/Users/miaaa/Desktop/music robot/furhat_music_robot/models"
WINDOW_SIZE = 2.0


print("Loading models...")
head_rf = joblib.load(f"{MODELS_DIR}/head_classifier.pkl")
head_scaler = joblib.load(f"{MODELS_DIR}/head_scaler.pkl")
head_feats = joblib.load(f"{MODELS_DIR}/head_feature_cols.pkl")

facial_hmms = joblib.load(f"{MODELS_DIR}/facial_hmm_models.pkl")
facial_scaler = joblib.load(f"{MODELS_DIR}/facial_hmm_scaler.pkl")
facial_feats = joblib.load(f"{MODELS_DIR}/facial_hmm_feature_cols.pkl")
print("Models loaded!\n")


def extract_features(segment, sr):
    features = {}

    mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f'mfcc_{i+1}_mean'] = float(np.mean(mfcc[i]))
        features[f'mfcc_{i+1}_std'] = float(np.std(mfcc[i]))

    mfcc_delta = librosa.feature.delta(mfcc)
    for i in range(13):
        features[f'mfcc_delta_{i+1}_mean'] = float(np.mean(mfcc_delta[i]))
        features[f'mfcc_delta_{i+1}_std'] = float(np.std(mfcc_delta[i]))

    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    for i in range(13):
        features[f'mfcc_delta2_{i+1}_mean'] = float(np.mean(mfcc_delta2[i]))
        features[f'mfcc_delta2_{i+1}_std'] = float(np.std(mfcc_delta2[i]))

    chroma = librosa.feature.chroma_stft(y=segment, sr=sr)
    for i in range(12):
        features[f'chroma_{i+1}_mean'] = float(np.mean(chroma[i]))
        features[f'chroma_{i+1}_std'] = float(np.std(chroma[i]))

    rms = librosa.feature.rms(y=segment)
    features['rms_mean'] = float(np.mean(rms))
    features['rms_std'] = float(np.std(rms))

    centroid = librosa.feature.spectral_centroid(y=segment, sr=sr)
    features['spectral_centroid_mean'] = float(np.mean(centroid))
    features['spectral_centroid_std'] = float(np.std(centroid))

    bandwidth = librosa.feature.spectral_bandwidth(y=segment, sr=sr)
    features['spectral_bandwidth_mean'] = float(np.mean(bandwidth))
    features['spectral_bandwidth_std'] = float(np.std(bandwidth))

    zcr = librosa.feature.zero_crossing_rate(y=segment)
    features['zcr_mean'] = float(np.mean(zcr))
    features['zcr_std'] = float(np.std(zcr))

    tempo_result = librosa.beat.beat_track(y=segment, sr=sr)
    tempo = tempo_result[0] if isinstance(tempo_result, tuple) else tempo_result
    tempo = float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo)
    features['tempo'] = tempo

    return features


def predict_head(feature_dict):
    x = np.array([feature_dict[c] for c in head_feats]).reshape(1, -1)
    x_scaled = head_scaler.transform(x)
    return head_rf.predict(x_scaled)[0]  # 'nod', 'none', 'shake', 'sway'


def predict_facial(feature_dict):
    x = np.array([feature_dict[c] for c in facial_feats]).reshape(1, -1)
    x_scaled = facial_scaler.transform(x)
    best_cls, best_score = None, -np.inf
    for cls, model in facial_hmms.items():
        try:
            score = model.score(x_scaled)
            if score > best_score:
                best_score, best_cls = score, cls
        except:
            pass
    return best_cls  # 'smile', 'big_smile', 'frown', 'neutral'


async def apply_facial_expression(furhat, expression, intensity=1.0, duration=1.0):
    gesture_map = {
        'big_smile': 'BigSmile',
        'smile': 'Smile',
        'frown': 'BrowFrown',
        'neutral': None
    }
    gesture = gesture_map.get(expression)
    if gesture is None:
        await furhat.request_face_reset()
    else:
        await furhat.request_gesture_start(name=gesture, intensity=intensity, duration=duration, wait=False)


async def main():
    furhat = AsyncFurhatClient("127.0.0.1")
    custom = CustomBehaviorTester()

    await furhat.connect()
    await custom.setup()

    # Analyze the song
    print("Analyzing audio...")
    y, sr = librosa.load(AUDIO_PATH)
    duration = librosa.get_duration(y=y, sr=sr)

    tempo_arr, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    global_tempo = float(tempo_arr[0]) if isinstance(tempo_arr, np.ndarray) else float(tempo_arr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    print(f"Song: {duration:.0f}s | Tempo: {global_tempo:.0f} BPM | Windows: {int(duration/WINDOW_SIZE)}\n")

    # Pre-analyze windows
    print("Pre-analyzing windows...")
    windows_data = []
    for start in np.arange(0, duration, WINDOW_SIZE):
        start_sample = int(start * sr)
        end_sample = int((start + WINDOW_SIZE) * sr)
        y_window = y[start_sample:end_sample]

        feats = extract_features(y_window, sr)
        head_pred = predict_head(feats)
        facial_pred = predict_facial(feats)

        window_beats = beat_times[(beat_times >= start) & (beat_times < start + WINDOW_SIZE)]

        rms_val = feats['rms_mean']
        intensity = 1.0 if rms_val < 0.10 else 1.3 if rms_val < 0.20 else 1.5

        windows_data.append({
            'start':     start,
            'head':      head_pred,
            'facial':    facial_pred,
            'beats':     window_beats,
            'intensity': intensity,
        })

    print(f"\nAnalysis done! {len(windows_data)} windows ready.\n")

    # Play the music
    # fortnight
    # music_url = "https://drive.google.com/uc?export=download&id=1Vi9Nu_9GLnwk-SKvgzcWAKwxPjHgDXGX"
    # zoo
    music_url = "https://drive.google.com/uc?export=download&id=1GDjkRJKUhaMbDrWWpoVr_TaAYFmZMAs3"
    # beat it
    # music_url = "https://drive.google.com/uc?export=download&id=1lBeTwrHgxXo982SBbhQBcF_zHU84oYEs"
    try:
        await furhat.request_speak_audio(url=music_url, wait=False, abort=False, lipsync=False)
    except Exception as e:
        print(f"Audio error: {e}")
        await furhat.disconnect()
        await custom.cleanup()
        return

    start_time = asyncio.get_event_loop().time()
    current_facial = None
    current_head_task = None

    for i, window in enumerate(windows_data):
        elapsed = asyncio.get_event_loop().time() - start_time
        wait_time = window['start'] - elapsed
        if wait_time > 0:
            await asyncio.sleep(wait_time)

        head = window['head']
        facial = window['facial']
        beats = window['beats']
        intensity = window['intensity']

        actual = asyncio.get_event_loop().time() - start_time
        print(f"{actual:5.1f}s | head:{head:6s} | facial:{facial}")

        # Cancel or wait for previous head task, then always reset head to center
        if current_head_task is not None:
            if not current_head_task.done():
                current_head_task.cancel()
                try:
                    await current_head_task
                except asyncio.CancelledError:
                    pass
            # reset regardless of whether task was cancelled or finished naturally
            await furhat.request_face_headpose(yaw=0, pitch=0, roll=0, relative=False)
            current_head_task = None

        # Facial expression (only update when expression changes)
        if facial != current_facial:
            expr_duration = WINDOW_SIZE
            for j in range(i + 1, len(windows_data)):
                if windows_data[j]['facial'] == facial:
                    expr_duration += WINDOW_SIZE
                else:
                    break
            asyncio.create_task(apply_facial_expression(furhat, facial, intensity, expr_duration))
            current_facial = facial

        # Head movement
        if head == 'nod':
            current_head_task = asyncio.create_task(
                do_nod_strategy_a(furhat, beats, global_tempo, start_time)
            )
        elif head == 'sway':
            times = max(1, int(WINDOW_SIZE / 1.5))
            current_head_task = asyncio.create_task(
                custom.head_sway(times=times, intensity=intensity)
            )
        elif head == 'shake':
            times = max(1, int(WINDOW_SIZE / 1.0))
            current_head_task = asyncio.create_task(
                custom.head_shake_fast(times=times, intensity=intensity)
            )
        # head == 'none': current_head_task stays None

    remaining = duration - (asyncio.get_event_loop().time() - start_time)
    if remaining > 0:
        await asyncio.sleep(remaining)

    await furhat.request_face_reset()
    await furhat.disconnect()
    await custom.cleanup()
    print("Finished!")


if __name__ == "__main__":
    asyncio.run(main())
