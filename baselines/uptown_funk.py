"""
Uptown Funk
(BPM): [117.45383523]
The number of the beats: 479
The time points of the first 10 beats）: [0.83591837 1.36997732 1.88081633 2.41487528 2.92571429 3.45977324
 3.9938322  4.52789116 5.0155102  5.54956916]"""
import asyncio
from test_built_in_behaviour import BuiltinGestureTester
from test_custom_behaviour import CustomBehaviorTester


class UptownFunkPerformance:
    """Furhat performance synchronized with 'Uptown Funk'"""

    def print_current_behavior(self, start_sec, end_sec, source, method_name, params):
        """Print current behavior information for video recording"""
        print(f"[{start_sec:.1f}s-{end_sec:.1f}s] {source.upper()}: {method_name} {params}")

    def __init__(self, music_url):
        self.music_url = music_url
        self.builtin = BuiltinGestureTester()
        self.custom = CustomBehaviorTester()
        self.active_tasks = []

        # Schedule: (start_sec, end_sec, "builtin" or "custom", method_name, params_dict)
        # gesture_duration: how long each single gesture lasts (default = full duration)
        self.schedule = [
            # 0-16: nodding to the beat, normal
            (0, 16, "builtin", "Nod", {"intensity": 1.0, "repetitions": 16, "gesture_duration": 1.0}),

            # 16-20: nodding + smile + eyes closed, strong
            (16, 20, "builtin", "Nod", {"intensity": 1.2, "repetitions": 4, "gesture_duration": 1.0}),
            (16, 20, "builtin", "Smile", {"intensity": 1.0}),
            (16, 20, "builtin", "CloseEyes", {"intensity": 1.0}),

            # 20-24: swaying head + big smile, strong
            (20, 24, "builtin", "Roll", {"intensity": 1.2, "repetitions": 4, "gesture_duration": 1.0}),
            (20, 24, "builtin", "BigSmile", {"intensity": 1.0}),

            # 24-34: nodding to the beat, strong
            (24, 34, "builtin", "Nod", {"intensity": 1.2, "repetitions": 8, "gesture_duration": 1.0}),

            # 34-41: craning and looking from side to side + biting lips
            (34, 41, "builtin", "Roll", {"intensity": 1.0, "repetitions": 7, "gesture_duration": 1.0}),
            (34, 41, "custom", "custom_slight_frown", {"intensity": 0.8}),

            # 41-47.5: shaking T-shirt + pout (can't do, make the original face)
            (41, 47.5, "builtin", "Smile", {"intensity": 0.5}),

            # 47.5-49: swaying head + smile, normal
            (47.5, 49, "builtin", "Roll", {"intensity": 1.0, "repetitions": 2, "gesture_duration": 0.75}),
            (47.5, 49, "builtin", "Smile", {"intensity": 1.0}),

            # 49-1:31 (91s): swaying head + smile + eyes closed, normal
            (49, 91, "builtin", "Roll", {"intensity": 1.0, "repetitions": 42, "gesture_duration": 1.0}),
            (49, 91, "builtin", "Smile", {"intensity": 1.0}),
            (49, 91, "builtin", "CloseEyes", {"intensity": 0.8}),

            # 1:31-1:34 (91-94s): scrunching up face
            (91, 94, "custom", "custom_narrow", {"intensity": 1.0}),
            (91, 94, "builtin", "BrowFurrow", {"intensity": 1.0}),

            # 1:34-1:47 (94-107s): swaying + smile + eyes closed
            (94, 107, "builtin", "Roll", {"intensity": 1.0, "repetitions": 13, "gesture_duration": 1.0}),
            (94, 107, "builtin", "Smile", {"intensity": 1.0}),
            (94, 107, "builtin", "CloseEyes", {"intensity": 0.8}),

            # 1:47-1:56 (107-116s): patting T-shirt + pout  (can't do, just smile)
            (107, 116, "builtin", "Smile", {"intensity": 0.6}),

            # 1:56-2:04 (116-124s): swaying shoulders + smile (can't do, just smile)
            (116, 124, "builtin", "Roll", {"intensity": 0.8, "repetitions": 8, "gesture_duration": 1.0}),
            (116, 124, "builtin", "Smile", {"intensity": 1.0}),

            # 2:04-2:47 (124-167s): shaking head + moving hands, very strong
            (124, 167, "builtin", "Roll", {"intensity": 1.5, "repetitions": 43, "gesture_duration": 1.0}),

            # 2:47-2:59 (167-179s): frown + question face
            (167, 179, "custom", "custom_slight_frown", {"intensity": 1.0}),
            (167, 179, "builtin", "BrowRaise", {"intensity": 1.0}),

            # 2:59-3:32 (179-212s): nodding + big smile, normal
            (179, 212, "builtin", "Nod", {"intensity": 1.0, "repetitions": 33, "gesture_duration": 1.0}),
            (179, 212, "builtin", "BigSmile", {"intensity": 1.0}),

            # 3:32-4:00 (212-240s): swaying + big smile + eyes closed + lip-syncing
            (212, 240, "builtin", "Roll", {"intensity": 1.0, "repetitions": 28, "gesture_duration": 1.0}),
            (212, 240, "builtin", "BigSmile", {"intensity": 1.0}),
            (212, 240, "builtin", "CloseEyes", {"intensity": 0.7}),

            # 4:00-4:10 (240-250s): craning to the beat + biting lips, strong
            (240, 250, "builtin", "Roll", {"intensity": 1.2, "repetitions": 10, "gesture_duration": 1.0}),
            (240, 250, "custom", "custom_slight_frown", {"intensity": 0.8}),

            # 4:10-4:29 (250-269s): nodding + smile, strong
            (250, 269, "builtin", "Nod", {"intensity": 1.2, "repetitions": 19, "gesture_duration": 1.0}),
            (250, 269, "builtin", "Smile", {"intensity": 1.0}),
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
        print("\nStarting performance: Uptown Funk\n")

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

            # Print current behavior (this will show in terminal during recording)
            self.print_current_behavior(start_sec, end_sec, source, method_name, params)

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

        # Wait for music to finish (Uptown Funk is about 4:29 = 269 seconds)
        remaining = 269 - (asyncio.get_event_loop().time() - start)
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
    # Google Drive link converted to direct download URL
    music_url = "https://drive.google.com/uc?export=download&id=1mWV0AleW4pAPowUQJ7oSCu0zHOWsLd2I"

    perf = UptownFunkPerformance(music_url)

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
