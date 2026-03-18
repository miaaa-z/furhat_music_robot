import librosa
import numpy as np
import asyncio
from furhat_realtime_api import AsyncFurhatClient
from test_custom_behaviour import CustomBehaviorTester
from center_down_2 import strategy_A as do_nod_strategy_a

AUDIO_PATH = "/Users/miaaa/Desktop/music robot/test/zoo.wav"
# AUDIO_PATH = "/Users/miaaa/Desktop/music robot/test/fortnight.wav"
# AUDIO_PATH = "/Users/miaaa/Desktop/music robot/furhat_music_robot/train/beat_it.wav"
WINDOW_SIZE = 2.0


def categorize_tempo(bpm: float) -> str:
    if bpm < 110:
        return 'slow'
    elif bpm < 130:
        return 'moderate'
    else:
        return 'fast'

def categorize_energy(energy: float) -> str:
    if energy < 0.10:
        return 'low'
    elif energy < 0.20:
        return 'medium'
    else:
        return 'high'


def categorize_brightness(hz: float) -> str:
    if hz < 1800:
        return 'dark'
    elif hz < 2300:
        return 'neutral'
    else:
        return 'bright'


HEAD_RULES = {
    # (tempo, energy) → head label
    ('slow',     'low'):    'none',
    ('slow',     'medium'): 'sway',
    ('slow',     'high'):   'nod',
    ('moderate', 'low'):    'sway',
    ('moderate', 'medium'): 'sway',
    ('moderate', 'high'):   'nod',
    ('fast',     'low'):    'sway',
    ('fast',     'medium'): 'shake',
    ('fast',     'high'):   'shake',
}

FACIAL_RULES = {
    # (brightness, energy) → facial label
    ('bright',  'low'):    'smile',
    ('bright',  'medium'): 'smile',
    ('bright',  'high'):   'big_smile',
    ('neutral', 'low'):    'neutral',
    ('neutral', 'medium'): 'smile',
    ('neutral', 'high'):   'smile',
    ('dark',    'low'):    'neutral',
    ('dark',    'medium'): 'neutral',
    ('dark',    'high'):   'frown',
}


def predict_rule_based(tempo_bpm: float, energy: float, brightness_hz: float):
    t = categorize_tempo(tempo_bpm)
    e = categorize_energy(energy)
    b = categorize_brightness(brightness_hz)

    head = HEAD_RULES.get((t, e), 'sway')
    facial = FACIAL_RULES.get((b, e), 'neutral')
    return head, facial


async def apply_facial_expression(furhat, expression: str,
                                  intensity: float = 1.0,
                                  duration: float = 1.0):
    gesture_map = {
        'big_smile': 'BigSmile',
        'smile':     'Smile',
        'frown':     'BrowFrown',
        'neutral':   None,
    }
    gesture = gesture_map.get(expression)
    if gesture is None:
        await furhat.request_face_reset()
    else:
        await furhat.request_gesture_start(
            name=gesture, intensity=intensity,
            duration=duration, wait=False
        )


async def main():
    furhat = AsyncFurhatClient("127.0.0.1")
    custom = CustomBehaviorTester()

    await furhat.connect()
    await custom.setup()

    # Audio analysis
    print("Analyzing audio...")
    y, sr = librosa.load(AUDIO_PATH)
    duration = librosa.get_duration(y=y, sr=sr)

    # Global brightness (spectral centroid over full song)
    brightness_hz = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))

    # Beat times for beat-synced nodding
    tempo_arr, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    global_tempo = float(tempo_arr[0]) if isinstance(tempo_arr, np.ndarray) else float(tempo_arr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    print(f"Song: {duration:.0f}s | Tempo: {global_tempo:.0f} BPM | "
          f"Brightness: {categorize_brightness(brightness_hz)} | "
          f"Windows: {int(duration / WINDOW_SIZE)}\n")

    print("Pre-analyzing windows.")

    nod_task = None
    windows_data = []

    for start in np.arange(0, duration, WINDOW_SIZE):
        start_sample = int(start * sr)
        end_sample = int((start + WINDOW_SIZE) * sr)
        y_win = y[start_sample:end_sample]
        if len(y_win) == 0:
            break

        # Per-window tempo & energy
        tempo_result = librosa.beat.beat_track(y=y_win, sr=sr)
        tempo_val = tempo_result[0]
        tempo_bpm = float(tempo_val[0]) if isinstance(tempo_val, np.ndarray) else float(tempo_val)

        energy = float(np.mean(librosa.feature.rms(y=y_win)))

        head_pred, facial_pred = predict_rule_based(tempo_bpm, energy, brightness_hz)

        # Beats in this window (for nod-on-beat)
        window_beats = beat_times[(beat_times >= start) & (beat_times < start + WINDOW_SIZE)]

        # Intensity from energy (same scale as ML)
        intensity = 1.0 if energy < 0.10 else 1.3 if energy < 0.20 else 1.5

        windows_data.append({
            'start':     start,
            'tempo':     tempo_bpm,
            'energy':    energy,
            'head':      head_pred,
            'facial':    facial_pred,
            'beats':     window_beats,
            'intensity': intensity,
        })

    print(f"Analysis done! {len(windows_data)} windows ready.\n")

    # Play music
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
        t_cat = categorize_tempo(window['tempo'])
        e_cat = categorize_energy(window['energy'])
        print(f"{actual:5.1f}s | tempo:{t_cat:8s} energy:{e_cat:6s} | "
              f"head:{head:6s} | facial:{facial}")

        # Facial expression
        # Only update when expression changes; hold for identical windows
        if facial != current_facial:
            expr_duration = WINDOW_SIZE
            for j in range(i + 1, len(windows_data)):
                if windows_data[j]['facial'] == facial:
                    expr_duration += WINDOW_SIZE
                else:
                    break
            asyncio.create_task(
                apply_facial_expression(furhat, facial, intensity, expr_duration)
            )
            current_facial = facial

        # Head movement
        if head == 'nod':
            if nod_task is not None and not nod_task.done():
                nod_task.cancel()  # cancel the last one (last one is not finished)
            nod_task = asyncio.create_task(
                do_nod_strategy_a(furhat, beats, global_tempo, start_time)
            )

        elif head == 'sway':
            times = max(1, int(WINDOW_SIZE / 1.5))
            asyncio.create_task(custom.head_sway(times=times, intensity=intensity))

        elif head == 'shake':
            times = max(1, int(WINDOW_SIZE / 1.0))
            asyncio.create_task(custom.head_shake_fast(times=times, intensity=intensity))

        # head == 'none'

    remaining = duration - (asyncio.get_event_loop().time() - start_time)
    if remaining > 0:
        await asyncio.sleep(remaining)

    await furhat.request_face_reset()
    await furhat.disconnect()
    await custom.cleanup()
    print("Finished!")


if __name__ == "__main__":
    asyncio.run(main())

