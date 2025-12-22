import asyncio
from test_built_in_behaviour import BuiltinGestureTester
from test_custom_behaviour import CustomBehaviorTester


class BackToBlackPerformance:
    def __init__(self, music_url):
        self.music_url = music_url
        self.builtin = BuiltinGestureTester()
        self.custom = CustomBehaviorTester()
        self.active_tasks = []

        # Schedule: (start_sec, end_sec, "builtin" or "custom", method_name, params_dict)
        # intensity: normal=1.0, gentle=0.7, very gentle=0.4
        self.schedule = [
            # 0-15: sway + smile (normal, times = 15/1.5 = 10)
            (0, 15, "custom", "head_sway", {"times": 10}),
            (0, 15, "builtin", "Smile", {}),

            # 15-20: surprise face + confused face
            (15, 20, "builtin", "Surprise", {}),
            (15, 20, "builtin", "BrowRaise", {}),

            # 20-27: smile
            (20, 27, "builtin", "Smile", {}),

            # 27-30: frown
            (27, 30, "builtin", "BrowFrown", {}),

            # 30-45: smile + frown
            (30, 45, "builtin", "Smile", {}),
            (30, 45, "builtin", "BrowFrown", {}),

            # 45-67: frown (gentle)
            (45, 67, "builtin", "BrowFrown", {"intensity": 0.7}),

            # 67-76 (1:07-1:16): nod + big smile
            (67, 76, "custom", "head_nod_fast", {"times": 9}),
            (67, 76, "builtin", "BigSmile", {}),

            # 76-82 (1:16-1:22): smile
            (76, 82, "builtin", "Smile", {}),

            # 82-87 (1:22-1:27): confused face
            (82, 87, "custom", "custom_confused", {}),

            # 87-97 (1:27-1:37): nod + frown (gentle)
            (87, 97, "custom", "head_nod_fast", {"times": 10, "intensity": 0.7}),
            (87, 97, "builtin", "BrowFrown", {"intensity": 0.7}),

            # 97-114 (1:37-1:54): thoughtful face
            (97, 114, "builtin", "Thoughtful", {}),

            # 114-117 (1:54-1:57): sway + thoughtful face (gentle, times = 3/1.5 = 2)
            (114, 117, "custom", "head_sway", {"times": 2, "intensity": 0.7}),
            (114, 117, "builtin", "Thoughtful", {"intensity": 0.7}),

            # 117-124 (1:57-2:04): nod + thoughtful face (gentle)
            (117, 124, "custom", "head_nod_fast", {"times": 7, "intensity": 0.7}),
            (117, 124, "builtin", "Thoughtful", {"intensity": 0.7}),

            # 124-131 (2:04-2:11): nod + smile (normal)
            (124, 131, "custom", "head_nod_fast", {"times": 7}),
            (124, 131, "builtin", "Smile", {}),

            # 131-135 (2:11-2:15): nod + disgust face (gentle)
            (131, 135, "custom", "head_nod_fast", {"times": 4, "intensity": 0.7}),
            (131, 135, "builtin", "ExpressDisgust", {"intensity": 0.7}),

            # 135-140 (2:15-2:20): disgust face
            (135, 140, "builtin", "ExpressDisgust", {}),

            # 140-165 (2:20-2:45): thoughtful face (very gentle)
            (140, 165, "builtin", "Thoughtful", {"intensity": 0.4}),

            # 165-240 (2:45-4:00): sway (very gentle, times = 75/1.5 = 50)
            (165, 240, "custom", "head_sway", {"times": 50, "intensity": 0.4}),
        ]

    async def setup(self):
        await self.builtin.setup()
        await self.custom.setup()

    async def repeat_builtin_gesture(self, method_name, intensity, duration):
        """Repeat a builtin gesture continuously within a time window"""
        end_time = asyncio.get_event_loop().time() + duration
        gesture_duration = 2.0

        while asyncio.get_event_loop().time() < end_time:
            await self.builtin.test_single_gesture(method_name, intensity, gesture_duration)
            remaining = end_time - asyncio.get_event_loop().time()
            if remaining < gesture_duration * 0.5:
                break

    async def perform(self):
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
            print(f"Audio error: {e}")
            return

        start = asyncio.get_event_loop().time()
        current_head_task = None  # Track current head movement task

        for start_sec, end_sec, source, method_name, params in self.schedule:
            # Wait until scheduled time
            now = asyncio.get_event_loop().time() - start
            wait = start_sec - now
            if wait > 0:
                await asyncio.sleep(wait)

            duration = end_sec - start_sec

            try:
                if source == "builtin":
                    intensity = params.get("intensity", 1.0)
                    task = asyncio.create_task(
                        self.repeat_builtin_gesture(method_name, intensity, duration)
                    )
                    self.active_tasks.append(task)
                else:  # custom - head movements
                    # Cancel previous head movement
                    if current_head_task and not current_head_task.done():
                        current_head_task.cancel()
                        try:
                            await current_head_task
                        except asyncio.CancelledError:
                            pass

                    # Start new head movement
                    method = getattr(self.custom, method_name, None)
                    if method:
                        current_head_task = asyncio.create_task(method(**params))
                        self.active_tasks.append(current_head_task)
            except Exception as e:
                print(f"Error: {e}")

        # Wait for music to finish
        remaining = 240 - (asyncio.get_event_loop().time() - start)
        if remaining > 0:
            await asyncio.sleep(remaining)

    async def cleanup(self):
        for task in self.active_tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        try:
            await self.builtin.cleanup()
        except:
            pass

        try:
            await self.custom.cleanup()
        except:
            pass


async def main():
    music_url = "https://drive.google.com/uc?export=download&id=1qitxkr6iESomykRkODFBtZGcfWfRMRjl"
    perf = BackToBlackPerformance(music_url)

    try:
        await perf.setup()
        await perf.perform()
    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await perf.cleanup()


if __name__ == "__main__":
    asyncio.run(main())