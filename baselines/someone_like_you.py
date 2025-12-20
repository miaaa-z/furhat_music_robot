"""
someone like you
(BPM): [135.99917763]
The number of the beats: 608
The time points of the first 10 beats）: [0.81269841 1.25387755 1.69505669 2.13623583 2.57741497 3.04181406
 3.4829932  3.92417234 4.36535147 4.82975057]"""
import asyncio
from test_built_in_behaviour import BuiltinGestureTester
from test_custom_behaviour import CustomBehaviorTester


class SomeoneLikeYouPerformance:
    """Furhat performance synchronized with 'Someone Like You'"""

    def print_current_behavior(self, start_sec, end_sec, source, method_name, params):
        """Print current behavior information for video recording"""
        print("\n" + "=" * 60)
        print(f"Time: {start_sec:.1f}s - {end_sec:.1f}s")
        print(f"Type: {source.upper()}")
        print(f"Action: {method_name}")
        print(f"Parameters: {params}")
        print("=" * 60 + "\n")

    def __init__(self, music_url):
        self.music_url = music_url
        self.builtin = BuiltinGestureTester()
        self.custom = CustomBehaviorTester()
        self.active_tasks = []

        # Schedule: (start_sec, end_sec, "builtin" or "custom", method_name, params_dict)
        # gesture_duration: how long each single gesture lasts (default = full duration)
        self.schedule = [
            (0, 13, "builtin", "Roll", {"intensity": 0.8, "repetitions": 5, "gesture_duration": 2.0}),
            (13, 17, "builtin", "Surprise", {"intensity": 1.0}),  # hold for full duration
            (17, 31.5, "builtin", "Thoughtful", {"intensity": 0.4}),
            (31.5, 34, "builtin", "Surprise", {"intensity": 1.0}),
            (34, 40, "builtin", "Thoughtful", {"intensity": 1.0}),
            (40, 41.5, "custom", "custom_raise_one_brow", {"intensity": 1.0, "side": "left"}),
            (41.5, 50, "custom", "custom_slight_frown", {"intensity": 1.0}),
            (50, 52, "builtin", "Surprise", {"intensity": 1.0}),
            (52, 73, "builtin", "Roll", {"intensity": 0.6, "repetitions": 21, "gesture_duration": 0.5}),  # head swaying
            (73, 75, "builtin", "Blink", {"intensity": 1.0}),
            (75, 84, "builtin", "Smile", {"intensity": 1.0}),
            (84, 86, "builtin", "Oh", {"intensity": 1.0}),
            (86, 88, "builtin", "BrowRaise", {"intensity": 1.0}),
            (88, 93, "custom", "test_head_positions", {}),
            (93, 99, "builtin", "Smile", {"intensity": 1.0}),
            (99, 108, "custom", "custom_raise_one_brow", {"intensity": 1.0, "side": "left"}),
            (108, 118, "custom", "custom_slight_frown", {"intensity": 1.0}),
            (118, 120, "builtin", "BrowRaise", {"intensity": 1.0}),
            (120, 125, "custom", "custom_slight_frown", {"intensity": 1.0}),
            (125, 127, "custom", "test_head_positions", {}),
            (127, 131, "custom", "custom_narrow", {"intensity": 0.7}),
            (131, 134, "builtin", "BrowRaise", {"intensity": 1.0}),
            (134, 139, "builtin", "Roll", {"intensity": 0.6, "repetitions": 5, "gesture_duration": 0.5}),
            (139, 150, "custom", "custom_slight_frown", {"intensity": 1.0}),
            (150, 154, "builtin", "Surprise", {"intensity": 1.0}),
            (154, 169, "builtin", "BigSmile", {"intensity": 1.0}),
            (169, 180, "builtin", "Smile", {"intensity": 0.7}),
            (180, 184, "builtin", "Smile", {"intensity": 1.0}),
            (184, 194, "custom", "custom_slight_frown", {"intensity": 0.7}),
            (194, 204, "builtin", "Thoughtful", {"intensity": 1.0}),
            (204, 207, "builtin", "Surprise", {"intensity": 1.0}),
            (207, 217, "builtin", "Thoughtful", {"intensity": 1.0}),
            (217, 223, "builtin", "CloseEyes", {"intensity": 1.0}),
            (223, 231, "builtin", "Smile", {"intensity": 1.0}),
            (231, 238, "builtin", "BigSmile", {"intensity": 1.0}),
            (238, 267, "builtin", "Smile", {"intensity": 1.0}),
            (267, 285, "builtin", "Smile", {"intensity": 1.0}),
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
        print("\nStarting performance: Someone Like You\n")

        # Start playing audio
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

            # Print current behavior (this will show in terminal during recording)
            self.print_current_behavior(start_sec, end_sec, source, method_name, params)

            # Execute gesture
            duration = end_sec - start_sec

            try:
                if source == "builtin":
                    intensity = params.get("intensity", 1.0)
                    repetitions = params.get("repetitions", 1)
                    gesture_duration = params.get("gesture_duration", duration)

                    task = asyncio.create_task(
                        self.repeat_builtin_gesture(method_name, intensity, duration, repetitions, gesture_duration)
                    )
                    self.active_tasks.append(task)

                else:  # custom
                    method = getattr(self.custom, method_name, None)
                    if method:
                        await asyncio.create_task(method(**params))
            except Exception as e:
                print(f"Gesture error: {e}")

        # Wait for music to finish
        remaining = 285 - (asyncio.get_event_loop().time() - start)
        if remaining > 0:
            await asyncio.sleep(remaining)

        print("\nPerformance complete\n")

    async def cleanup(self):
        """Clean up connections"""
        # Cancel all active tasks
        for task in self.active_tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Try to cleanup connections with error handling
        try:
            await self.builtin.cleanup()
        except RuntimeError as e:
            print(f"Builtin cleanup note: {e}")
        except Exception as e:
            print(f"Builtin cleanup error: {e}")

        try:
            await self.custom.cleanup()
        except RuntimeError as e:
            print(f"Custom cleanup note: {e}")
        except Exception as e:
            print(f"Custom cleanup error: {e}")


async def main():
    music_url = "https://drive.google.com/uc?export=download&id=1y7o8Qaz46kL8Mn9fEFSNC4fJCN_IQoCY"

    perf = SomeoneLikeYouPerformance(music_url)

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