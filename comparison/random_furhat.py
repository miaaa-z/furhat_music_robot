import librosa
import numpy as np
import asyncio
import random
from furhat_realtime_api import AsyncFurhatClient
from test_custom_behaviour import CustomBehaviorTester


AUDIO_PATH = "/Users/miaaa/Desktop/music robot/furhat_music_robot/train/cardigan.wav"
# AUDIO_PATH = "/Users/miaaa/Desktop/music robot/furhat_music_robot/train/closer.wav"
# AUDIO_PATH = "/Users/miaaa/Desktop/music robot/furhat_music_robot/train/highway_to_hell.wav"
WINDOW_SIZE = 2.0


FACIAL_GESTURES = [
    "BigSmile", "Smile", "Wink", "Surprise", "Oh", "Thoughtful",
    "ExpressAnger", "ExpressDisgust", "ExpressFear", "ExpressSad",
    "BrowRaise", "BrowFrown",
]

HEAD_CHOICES = ["nod", "shake", "roll", "sway", "none"]


# strategy_A from center_down_2.py
# 4/4: down → middle → up → middle, one per measure
async def do_nod_strategy_a(furhat, beat_times, tempo, start_time):
    if len(beat_times) == 0:
        return

    beat_duration = 60.0 / tempo
    # find measure starts (every 4th beat index, but filtered to this window)
    #  the first beat as a downbeat
    measure_starts = beat_times[::4]

    await furhat.request_face_headpose(yaw=0, pitch=0, roll=0, relative=False)

    for measure_start in measure_starts:
        timings = [
            measure_start,
            measure_start + beat_duration,
            measure_start + beat_duration * 2,
            measure_start + beat_duration * 3,
        ]
        movements = [
            (0, 10, 0),   # beat 1: down
            (0, -10, 0),  # beat 2: back to middle
            (0, -10, 0),  # beat 3: up
            (0, 10, 0),   # beat 4: back to middle
        ]
        for t, (yaw, pitch, roll) in zip(timings, movements):
            elapsed = asyncio.get_event_loop().time() - start_time
            wait = t - elapsed
            if wait > 0:
                await asyncio.sleep(wait)
            await furhat.request_face_headpose(yaw=yaw, pitch=pitch, roll=roll, relative=True)


async def main():
    furhat = AsyncFurhatClient("127.0.0.1")
    custom = CustomBehaviorTester()

    await furhat.connect()
    await custom.setup()

    # Audio (only need beats for nod timing)
    print("Analyzing audio")
    y, sr = librosa.load(AUDIO_PATH)
    duration = librosa.get_duration(y=y, sr=sr)

    tempo_arr, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(tempo_arr[0]) if isinstance(tempo_arr, np.ndarray) else float(tempo_arr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    print(f"Song: {duration:.0f}s | Tempo: {tempo:.0f} BPM | "
          f"Windows: {int(duration / WINDOW_SIZE)}\n")

    # Pre-sample random actions for every window
    print("Sampling random actions")
    windows_data = []
    for start in np.arange(0, duration, WINDOW_SIZE):
        window_beats = beat_times[(beat_times >= start) & (beat_times < start + WINDOW_SIZE)]
        windows_data.append({
            'start':  start,
            'head':   random.choice(HEAD_CHOICES),
            'facial': random.choice(FACIAL_GESTURES),
            'beats':  window_beats,
        })
    print(f"Got {len(windows_data)} windows.\n")

    # Play music
    # cardigan
    music_url = "https://drive.google.com/uc?export=download&id=1qalHZhDBCWYTo8pTI38jsTGExivpcPtE"
    # closer
    # music_url = "https://drive.google.com/uc?export=download&id=1cFggXL3JWij6Sa1p1IzEv-X9eKwLOean"
    # highway to hell
    # music_url = "https://drive.google.com/uc?export=download&id=1Szo3Q5jWBJ3P9jHhWskVAq7lkuICl7ox"

    try:
        await furhat.request_speak_audio(url=music_url, wait=False, abort=False, lipsync=False)
    except Exception as e:
        print(f"Audio error: {e}")
        await furhat.disconnect()
        await custom.cleanup()
        return

    start_time = asyncio.get_event_loop().time()

    for window in windows_data:
        elapsed = asyncio.get_event_loop().time() - start_time
        wait_time = window['start'] - elapsed
        if wait_time > 0:
            await asyncio.sleep(wait_time)

        head = window['head']
        facial = window['facial']
        beats = window['beats']

        actual = asyncio.get_event_loop().time() - start_time
        print(f"{actual:5.1f}s | head:{head:6s} | facial:{facial}")

        # Facial
        await furhat.request_gesture_start(
            name=facial, intensity=1.0, duration=WINDOW_SIZE, wait=False
        )

        # Head movement
        if head == 'nod':
            asyncio.create_task(do_nod_strategy_a(furhat, beats, tempo, start_time))

        elif head == 'sway':
            times = max(1, int(WINDOW_SIZE / 1.5))
            asyncio.create_task(custom.head_sway(times=times, intensity=1.0))

        elif head == 'shake':
            # use built-in Shake gesture
            await furhat.request_gesture_start(
                name='Shake', intensity=1.0, duration=WINDOW_SIZE, wait=False
            )

        elif head == 'roll':
            await furhat.request_gesture_start(
                name='Roll', intensity=1.0, duration=WINDOW_SIZE, wait=False
            )

        # head == 'none': stay still

    remaining = duration - (asyncio.get_event_loop().time() - start_time)
    if remaining > 0:
        await asyncio.sleep(remaining)

    await furhat.request_face_reset()
    await furhat.disconnect()
    await custom.cleanup()
    print("Finished!")


if __name__ == "__main__":
    asyncio.run(main())
