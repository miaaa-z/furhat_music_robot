import librosa
import numpy as np
import asyncio
from furhat_realtime_api import AsyncFurhatClient
from test_custom_behaviour import CustomBehaviorTester
from behavior_definitions import categorize_tempo, categorize_energy, categorize_brightness, get_behavior

AUDIO_PATH = "/Users/miaaa/Desktop/music robot/furhat_music_robot/train/beat_it.wav"
WINDOW_SIZE = 5.0


async def main():
    furhat = AsyncFurhatClient("127.0.0.1")
    custom = CustomBehaviorTester()

    await furhat.connect()
    await custom.setup()

    # Load audio and analyze
    y, sr = librosa.load(AUDIO_PATH)
    duration = librosa.get_duration(y=y, sr=sr)
    brightness = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    brightness_cat = categorize_brightness(brightness)

    print(f"Song: {duration:.0f}s | Window: {WINDOW_SIZE}s | Brightness: {brightness_cat}\n")
    print("Analyzing windows...")

    windows_data = []
    for start in np.arange(0, duration, WINDOW_SIZE):
        start_sample = int(start * sr)
        end_sample = int((start + WINDOW_SIZE) * sr)
        y_window = y[start_sample:end_sample]
        if len(y_window) == 0:
            break

        tempo, _ = librosa.beat.beat_track(y=y_window, sr=sr)
        energy = float(np.mean(librosa.feature.rms(y=y_window)))

        tempo_cat = categorize_tempo(float(tempo))
        energy_cat = categorize_energy(energy)

        # choose 2 gestures randomly from BEHAVIOR_MAP
        gestures, led_color = get_behavior(tempo_cat, energy_cat, brightness_cat)

        # different intensity based on energy
        intensity = 1.0 if energy < 0.10 else 1.3 if energy < 0.20 else 1.5

        windows_data.append({
            'start': start,
            'tempo': float(tempo),
            'energy': energy,
            'gestures': gestures,
            'led_color': led_color,
            'intensity': intensity
        })

    print(f"Analysis complete! {len(windows_data)} windows ready.\n")

    # play music
    music_url = "https://drive.google.com/uc?export=download&id=1lBeTwrHgxXo982SBbhQBcF_zHU84oYEs"

    try:
        await furhat.request_speak_audio(
            url=music_url,
            wait=False,
            abort=False,
            lipsync=False
        )
        await asyncio.sleep(0.3)
    except Exception as e:
        print(f"Audio error: {e}")
        await furhat.disconnect()
        await custom.cleanup()
        return

    start_time = asyncio.get_event_loop().time()  # start time of the music starts
    custom_gestures = ['head_sway', 'head_nod_fast', 'head_shake_fast']
    eyes_closed = False

    # every window
    for window in windows_data:
        elapsed = asyncio.get_event_loop().time() - start_time  # from the beginning of a song to now
        wait_time = window['start'] - elapsed
        if wait_time > 0:
            await asyncio.sleep(wait_time)

        # LED
        await furhat.request_led_set(color=window['led_color'])

        gestures = window['gestures']
        intensity = window['intensity']

        # Eyes (if it closes eyes it'll not open)
        if eyes_closed and 'CloseEyes' not in gestures:
            await furhat.request_gesture_start('OpenEyes', intensity=1.0, duration=0.5, wait=True)
            eyes_closed = False

        if 'CloseEyes' in gestures:
            eyes_closed = True

        # execute 2 gestures at the same time
        for gesture_name in gestures:
            if gesture_name in custom_gestures:
                # through CustomBehaviorTester
                times = int(WINDOW_SIZE / 1.5) if gesture_name == 'head_sway' else int(WINDOW_SIZE / 1.0)
                method = getattr(custom, gesture_name)
                asyncio.create_task(method(times=times, intensity=intensity))
            else:
                # builtin
                await furhat.request_gesture_start(
                    name=gesture_name,
                    intensity=intensity,
                    duration=WINDOW_SIZE,
                    wait=False  # don't wait
                )

        actual_time = asyncio.get_event_loop().time() - start_time
        print(
            f"{actual_time:5.1f}s | T:{window['tempo']:3.0f} E:{window['energy']:.2f} I:{intensity:.1f} | {gestures[0]} + {gestures[1]} | LED:{window['led_color']}")

    # eyes should open after closing
    if eyes_closed:
        await furhat.request_gesture_start('OpenEyes', intensity=1.0, duration=0.5, wait=True)

    # wait for the song to stop
    remaining = duration - (asyncio.get_event_loop().time() - start_time)
    if remaining > 0:
        await asyncio.sleep(remaining)

    await furhat.disconnect()
    await custom.cleanup()
    print("Finished!")


if __name__ == "__main__":
    asyncio.run(main())