"""
back to black
(BPM): [123.046875]
477 beats
The time points of the first 10 beats: [0.37151927 0.85913832 1.34675737 1.83437642 2.32199546 2.78639456
 3.29723356 3.78485261 4.27247166 4.7600907 ]"""
import asyncio
from test_built_in_behaviour import BuiltinGestureTester
from test_custom_behaviour import CustomBehaviorTester


class BackToBlackPerformance:
    """Furhat performance synchronized with 'Back to Black'"""

    def __init__(self, music_url):
        self.music_url = music_url
        self.builtin = BuiltinGestureTester()
        self.custom = CustomBehaviorTester()
        self.active_tasks = []

        # Schedule: (start_sec, end_sec, "builtin" or "custom" or "speak", method_name, params_dict)
        # gesture_duration: how long each single gesture lasts (default = full duration)
        # Note: head_sway needs 'times' parameter - each sway cycle takes 3 seconds
        self.schedule = [
            # 0-15: swaying head to the beat + smile, normal (15/3 = 5 times)
            (0, 15, "custom", "head_sway", {"times": 5}),
            (0, 15, "builtin", "Smile", {"intensity": 1.0}),

            # 15-20: surprised-but-confused face
            (15, 20, "builtin", "Surprise", {"intensity": 1.0}),
            (15, 20, "builtin", "BrowRaise", {"intensity": 0.8}),

            # 20-27: smile
            (20, 27, "builtin", "Smile", {"intensity": 1.0}),

            # 27-30: frown
            (27, 30, "custom", "custom_slight_frown", {"intensity": 1.0}),

            # 30-45: swaying + smile and frown (15/3 = 5 times)
            (30, 45, "custom", "head_sway", {"times": 5}),
            (30, 45, "builtin", "Smile", {"intensity": 0.7}),
            (30, 45, "custom", "custom_slight_frown", {"intensity": 0.5}),

            # 45-67 (1:07): swaying + slightly frown (22/3 = 7 times)
            (45, 67, "custom", "head_sway", {"times": 7}),
            (45, 67, "custom", "custom_slight_frown", {"intensity": 0.7}),

            # 1:07-1:16 (67-76): nodding head + big smile
            (67, 76, "builtin", "Nod", {"intensity": 1.0, "repetitions": 9, "gesture_duration": 1.0}),
            (67, 76, "builtin", "BigSmile", {"intensity": 1.0}),

            # 1:16-1:22 (76-82): swaying + smile (6/3 = 2 times)
            (76, 82, "custom", "head_sway", {"times": 2}),
            (76, 82, "builtin", "Smile", {"intensity": 1.0}),

            # 1:22-1:27 (82-87): confused face
            (82, 87, "builtin", "BrowRaise", {"intensity": 1.0}),
            (82, 87, "custom", "custom_narrow", {"intensity": 0.6}),

            # 1:27-1:37 (87-97): nodding head to the beat + slightly frown, gentle
            (87, 97, "builtin", "Nod", {"intensity": 0.7, "repetitions": 10, "gesture_duration": 1.0}),
            (87, 97, "custom", "custom_slight_frown", {"intensity": 0.7}),

            # 1:37-1:54 (97-114): thoughtful face
            (97, 114, "builtin", "Thoughtful", {"intensity": 1.0}),

            # 1:54-1:57 (114-117): swaying head to the beat + thoughtful face, gentle (3/3 = 1 time)
            (114, 117, "custom", "head_sway", {"times": 1}),
            (114, 117, "builtin", "Thoughtful", {"intensity": 1.0}),

            # 1:57-2:04 (117-124): nodding head to the beat + thoughtful face, gentle
            (117, 124, "builtin", "Nod", {"intensity": 0.7, "repetitions": 7, "gesture_duration": 1.0}),
            (117, 124, "builtin", "Thoughtful", {"intensity": 1.0}),

            # 2:04-2:11 (124-131): nodding head to the beat + smile, normal
            (124, 131, "builtin", "Nod", {"intensity": 1.0, "repetitions": 7, "gesture_duration": 1.0}),
            (124, 131, "builtin", "Smile", {"intensity": 1.0}),

            # 2:11-2:15 (131-135): nodding head to the beat + thoughtful face, gentle
            (131, 135, "builtin", "Nod", {"intensity": 0.7, "repetitions": 4, "gesture_duration": 1.0}),
            (131, 135, "builtin", "Thoughtful", {"intensity": 1.0}),

            # 2:15-2:20 (135-140): curling lips
            (135, 140, "builtin", "ExpressDisgust", {"intensity": 1.0}),

            # 2:20-2:45 (140-165): swaying + curling lips, very gentle (25/3 = 8 times)
            (140, 165, "custom", "head_sway", {"times": 8}),
            (140, 165, "builtin", "ExpressDisgust", {"intensity": 0.8}),

            # 2:45-4:00 (165-240): swaying head, very gentle (75/3 = 25 times)
            (165, 240, "custom", "head_sway", {"times": 25}),
        ]

    async def setup(self):
        """Initialize Furhat connection for both builtin and custom behaviors
        Must be called before perform()"""
        await self.builtin.setup()
        await self.custom.setup()

    async def repeat_builtin_gesture(self, method_name, intensity, duration, repetitions, gesture_duration):
        """Repeat a builtin gesture continuously within a time window"""
        end_time = asyncio.get_event_loop().time() + duration

        while asyncio.get_event_loop().time() < end_time:
            # Wait for gesture to complete before starting next one
            await self.builtin.test_single_gesture(method_name, intensity, gesture_duration)

            # Check if there's enough time for another full gesture
            remaining = end_time - asyncio.get_event_loop().time()
            if remaining < gesture_duration * 0.5:  # Not enough time for another full one
                break

    async def perform(self):
        """Start synchronized performance"""
        print("Starting Back to Black performance...")

        # Start playing audio (non-blocking)
        try:
            await self.builtin.furhat.request_speak_audio(
                url=self.music_url,
                wait=False,
                abort=False,
                lipsync=False
            )
            await asyncio.sleep(0.3)
        except Exception as e:
            print(f"Audio playback error: {e}")
            return

        # Execute gesture sequence
        start = asyncio.get_event_loop().time()

        for start_sec, end_sec, source, method_name, params in self.schedule:
            # Wait until scheduled time
            now = asyncio.get_event_loop().time() - start
            wait = start_sec - now
            if wait > 0:
                await asyncio.sleep(wait)

            # Execute gesture
            duration = end_sec - start_sec

            try:
                if source == "builtin":
                    intensity = params.get("intensity", 1.0)
                    repetitions = params.get("repetitions", 1)
                    gesture_duration = params.get("gesture_duration", duration)  # Default: hold for full duration

                    task = asyncio.create_task(
                        self.repeat_builtin_gesture(method_name, intensity, duration, repetitions, gesture_duration)
                    )
                    self.active_tasks.append(task)

                else:  # custom
                    method = getattr(self.custom, method_name, None)
                    if method:
                        task = asyncio.create_task(method(**params))
                        self.active_tasks.append(task)
            except Exception as e:
                print(f"Gesture error at {start_sec}s: {e}")

        # Wait for music to finish (Back to Black is about 4:00 = 240 seconds)
        remaining = 240 - (asyncio.get_event_loop().time() - start)
        if remaining > 0:
            await asyncio.sleep(remaining)

        print("Performance complete!")

    async def cleanup(self):
        """Clean up connections"""
        for task in self.active_tasks:
            task.cancel()
        await self.builtin.cleanup()
        await self.custom.cleanup()


async def main():
    # Google Drive link converted to direct download URL
    music_url = "https://drive.google.com/uc?export=download&id=1qitxkr6iESomykRkODFBtZGcfWfRMRjl"

    perf = BackToBlackPerformance(music_url)

    try:
        await perf.setup()
        await perf.perform()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await perf.cleanup()


if __name__ == "__main__":
    asyncio.run(main())