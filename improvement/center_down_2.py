import librosa
import numpy as np
import asyncio
from furhat_realtime_api import AsyncFurhatClient

AUDIO_PATH = "/Users/miaaa/Desktop/music robot/furhat_music_robot/train/faded.wav"
MAX_DURATION = 60   # test the first 60 s

async def strategy_A(furhat, beat_times, tempo, start_time):
    """
    down → middle → up → middle（4/4）
    """
    beat_duration = 60.0 / tempo
    measure_starts = beat_times[::4]

    # go to center first
    await furhat.request_face_headpose(yaw=0, pitch=0, roll=0, relative=False)

    for measure_start in measure_starts:
        timings = [
            measure_start,
            measure_start + beat_duration,
            measure_start + beat_duration * 2,
            measure_start + beat_duration * 3
        ]
        movements = [
            (0, 10, 0),  # down
            (0, -10, 0),  # back to middle
            (0, -10, 0),  # up
            (0, 10, 0)  # back to middle
        ]

        for time_point, (yaw, pitch, roll) in zip(timings, movements):
            elapsed = asyncio.get_event_loop().time() - start_time
            wait = time_point - elapsed
            if wait > 0:
                await asyncio.sleep(wait)

            await furhat.request_face_headpose(
                yaw=yaw,
                pitch=pitch,
                roll=roll,
                relative=True
            )
            print(f"{time_point:.2f}s: pitch {pitch:+3.0f}° (relative)")


async def strategy_B(furhat, beat_times, tempo, start_time):
    """
   go down at the downbeat (4/4)
    """
    # go to the center first
    await furhat.request_face_headpose(yaw=0, pitch=0, roll=0, relative=False)

    last_was_down = False

    for i, beat_time in enumerate(beat_times):
        elapsed = asyncio.get_event_loop().time() - start_time
        wait = beat_time - elapsed
        if wait > 0:
            await asyncio.sleep(wait)

        is_downbeat = (i % 4 == 0)

        if is_downbeat is True:
            # lower head
            await furhat.request_face_headpose(yaw=0, pitch=10, roll=0, relative=True)
            last_was_down = True
            print(f"{beat_time:.2f}s: ▼ down")

        elif last_was_down is True:
            # go back to center
            await furhat.request_face_headpose(yaw=0, pitch=0, roll=0, relative=True)
            last_was_down = False
            print(f"{beat_time:.2f}s: • up to center")
        else:

            print(f"{beat_time:.2f}s: • (stay)")


async def main():
    STRATEGY = "A"

    furhat = AsyncFurhatClient("127.0.0.1")
    await furhat.connect()

    # Face the center first
    await furhat.request_face_headpose(yaw=0, pitch=0, roll=0, relative=False)

    # analyse the song
    y, sr = librosa.load(AUDIO_PATH)
    duration = librosa.get_duration(y=y, sr=sr)

    # get tempo and beat times
    tempo_array, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    tempo = tempo_array.item()

    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    # limit the testing time to be 60s
    if MAX_DURATION:
        beat_times = beat_times[beat_times < MAX_DURATION]
        duration = min(duration, MAX_DURATION)

    print(f"Song: {duration:.0f}s | Tempo: {tempo:.0f} BPM")
    print(f"Beats detected: {len(beat_times)}")
    print(f"Beat duration: {60 / tempo:.2f}s per beat")
    print(f"Testing Strategy {STRATEGY}\n")

    # play the song
    music_url = "https://drive.google.com/uc?export=download&id=18dhwLGYrc_pLfngKxb0eNEbkGl6xTLa0"
    await furhat.request_speak_audio(
        url=music_url,
        wait=False,
        abort=False,
        lipsync=False
    )

    start_time = asyncio.get_event_loop().time()

    if STRATEGY == "A":
        await strategy_A(furhat, beat_times, tempo, start_time)
    else:
        await strategy_B(furhat, beat_times, tempo, start_time)

    remaining = duration - (asyncio.get_event_loop().time() - start_time)
    if remaining > 0:
        await asyncio.sleep(remaining)

    await furhat.request_face_headpose(
        yaw=0.0,
        pitch=0.0,
        roll=0.0,
        relative=False
    )

    await furhat.disconnect()
    print("\nTest finished!")


if __name__ == "__main__":
    asyncio.run(main())