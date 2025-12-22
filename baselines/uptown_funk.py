import asyncio
from test_built_in_behaviour import BuiltinGestureTester
from test_custom_behaviour import CustomBehaviorTester


class UptownFunkPerformance:
    """Furhat performance synchronized with 'Uptown Funk'"""

    def __init__(self, music_url):
        self.music_url = music_url
        self.builtin = BuiltinGestureTester()
        self.custom = CustomBehaviorTester()
        self.active_tasks = []

        # Schedule: (start_sec, end_sec, "builtin" or "custom", method_name, params_dict)
        # intensity: normal=1.0, strong=1.3, very strong=1.5
        self.schedule = [
            # 0-16: nod (normal)
            (0, 16, "custom", "head_nod_fast", {"times": 16}),

            # 16-20: nod + smile + close eyes (strong)
            (16, 20, "custom", "head_nod_fast", {"times": 4, "intensity": 1.3}),
            (16, 20, "builtin", "Smile", {"intensity": 1.3}),
            (16, 20, "builtin", "CloseEyes", {"intensity": 1.3}),

            # 20-24: sway + big smile (strong)
            (20, 24, "builtin", "OpenEyes", {}),
            # times of head_sway =  time / 1.5(sleep time written in custom.py)
            (20, 24, "custom", "head_sway", {"times": 2, "intensity": 1.3}),
            (20, 24, "builtin", "BigSmile", {"intensity": 1.3}),

            # 24-34: nod (strong)
            (24, 34, "custom", "head_nod_fast", {"times": 9, "intensity": 1.3}),
            # make it nod 9 times to avoid conflicting with shaking

            # 34-41: shake
            (34, 41, "custom", "head_shake_fast", {"times": 7}),

            # 41-47.5: smile
            (41, 47.5, "builtin", "Smile", {}),

            # 47.5-49: sway + smile (normal)
            (47.5, 49, "custom", "head_sway", {"times": 1}),
            (47.5, 49, "builtin", "Smile", {}),

            # 49-91: sway + smile + close eyes (normal)
            (49, 91, "custom", "head_sway", {"times": 28}),
            (49, 91, "builtin", "Smile", {}),
            (49, 91, "builtin", "CloseEyes", {}),

            # 91-94: angry face
            (91, 94, "builtin", "OpenEyes", {}),
            (91, 94, "builtin", "ExpressAnger", {}),

            # 94-107: smile + close eyes
            (94, 107, "builtin", "Smile", {}),
            (94, 107, "builtin", "CloseEyes", {}),

            # 107-116: smile
            (107, 116, "builtin", "OpenEyes", {}),
            (107, 116, "builtin", "Smile", {}),

            # 116-124: shake (very strong)
            (116, 124, "custom", "head_shake_fast", {"times": 8, "intensity": 1.5}),

            # 124-167: frown + confused face
            (124, 167, "builtin", "Thoughtful", {}),
            (124, 167, "builtin", "BrowRaise", {}),

            # 167-179: nod + big smile (normal)
            (167, 179, "custom", "head_nod_fast", {"times": 12}),
            (167, 179, "builtin", "BigSmile", {}),

            # 179-212: sway + big smile + close eyes
            (179, 212, "custom", "head_sway", {"times": 22}),
            (179, 212, "builtin", "BigSmile", {}),
            (179, 212, "builtin", "CloseEyes", {}),

            # 212-240: sway (strong)
            (212, 240, "builtin", "OpenEyes", {}),
            (212, 240, "custom", "head_sway", {"times": 18, "intensity": 1.3}),

            # 240-250: nod + smile (strong)
            (240, 250, "custom", "head_nod_fast", {"times": 10, "intensity": 1.3}),
            (240, 250, "builtin", "Smile", {"intensity": 1.3}),
        ]

    async def setup(self):
        await self.builtin.setup()
        await self.custom.setup()

    async def repeat_builtin_gesture(self, method_name, intensity, duration):
        """Repeat a builtin gesture continuously within a time window"""
        end_time = asyncio.get_event_loop().time() + duration
        gesture_duration = 1.0

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
                else:  # custom
                    method = getattr(self.custom, method_name, None)
                    if method:
                        task = asyncio.create_task(method(**params))
                        self.active_tasks.append(task)
            except Exception as e:
                print(f"Error: {e}")

        # Wait for music to finish
        remaining = 269 - (asyncio.get_event_loop().time() - start)
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
    music_url = "https://drive.google.com/uc?export=download&id=1mWV0AleW4pAPowUQJ7oSCu0zHOWsLd2I"
    perf = UptownFunkPerformance(music_url)

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