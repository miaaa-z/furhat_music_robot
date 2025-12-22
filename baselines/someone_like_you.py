"""
Someone Like You - Fixed version with intensity support
"""
import asyncio
from test_built_in_behaviour import BuiltinGestureTester
from test_custom_behaviour import CustomBehaviorTester


class SomeoneLikeYouPerformance:

    def __init__(self, music_url):
        self.music_url = music_url
        self.builtin = BuiltinGestureTester()
        self.custom = CustomBehaviorTester()
        self.active_tasks = []

        # Schedule: (start_sec, end_sec, "builtin" or "custom", method_name, params_dict)
        self.schedule = [
            # 0-13s: sway (times = 13/1.5 = 8)
            (0, 13, "custom", "head_sway", {"times": 8, "intensity": 0.5}),

            # 13-17s: surprise
            (13, 17, "builtin", "Surprise", {}),

            # 17-31.5s: sway + thoughtful (very gentle, times = 14.5/1.5 = 9)
            (17, 31.5, "custom", "head_sway", {"times": 9, "intensity": 0.4}),
            (17, 31.5, "builtin", "Thoughtful", {"intensity": 0.4}),

            # 31.5-34s: surprise
            (31.5, 34, "builtin", "Surprise", {}),

            # 34-40s: thoughtful
            (34, 40, "builtin", "Thoughtful", {}),

            # 40-41.5s: lift left eyebrow
            (40, 41.5, "custom", "raise_left_brow", {}),

            # 41.5-50s: frown
            (41.5, 50, "builtin", "BrowFrown", {}),

            # 50-52s: surprise
            (50, 52, "builtin", "Surprise", {}),

            # 52-73s: sway + thoughtful (very gentle, times = 21/1.5 = 14)
            (52, 73, "custom", "head_sway", {"times": 14, "intensity": 0.4}),
            (52, 73, "builtin", "Thoughtful", {"intensity": 0.4}),

            # 73-75s: blink
            (73, 75, "builtin", "Blink", {}),

            # 75-84s: smile
            (75, 84, "builtin", "Smile", {}),

            # 84-86s: "Oh" face
            (84, 86, "builtin", "Oh", {}),

            # 86-88s: brows lifting
            (86, 88, "builtin", "BrowRaise", {}),

            # 88-93s: look top right + thoughtful
            (88, 93, "custom", "head_positions", {"x": -0.5, "y": 0.5, "z": 1.0, "duration": 5}),
            (88, 93, "builtin", "Thoughtful", {}),

            # 93-99s: smile
            (93, 99, "custom", "head_positions", {"x": 0.0, "y": 0.0, "z": 1.0}),  # ok
            (93, 99, "builtin", "Smile", {}),

            # 99-108s: lift left eyebrow
            (99, 108, "custom", "raise_left_brow", {}),

            # 108-118s: thoughtful
            (108, 118, "builtin", "Thoughtful", {}),

            # 118-120s: brows lifting
            (118, 120, "builtin", "BrowRaise", {}),

            # 120-125s: frown
            (120, 125, "builtin", "BrowFrown", {}),

            # 125-127s: look up
            (125, 127, "custom", "head_positions", {"x": 0.0, "y": 0.5, "z": 1.0}),

            # 127-131s: look down + narrow eyes
            # problem: up->down: ok if no narrow
            # else, it up can't go back to center
            (127, 131, "custom", "head_positions", {"x": 0.0, "y": 0, "z": 1.0, "duration": 4}),
            # (127, 131, "custom", "custom_narrow", {"intensity": 0.7}),

            # 131-134s: brows lifting

            (131, 134, "builtin", "BrowRaise", {}),

            # 134-139s: sway (very gentle, times = 5/1.5 = 3)
            (134, 139, "custom", "head_sway", {"times": 3, "intensity": 0.4}),

            # 139-150s: frown
            (139, 150, "builtin", "BrowFrown", {}),

            # 150-154s: surprise
            (150, 154, "builtin", "Surprise", {}),

            # 154-169s: look top right + big smile
            (154, 169, "custom", "head_positions", {"x": -0.5, "y": 0.5, "z": 1.0, "duration": 10}),
            (154, 169, "builtin", "BigSmile", {}),

            # 169-180s: sway + smile (very gentle, times = 11/1.5 = 7)
            (169, 171, "custom", "head_positions", {"x": 0.0, "y": 0.0, "z": 1.0}),
            (169, 180, "custom", "head_sway", {"times": 7, "intensity": 0.4}),
            (169, 180, "builtin", "Smile", {"intensity": 0.4}),

            # 180-184s: smile
            (180, 184, "builtin", "Smile", {}),

            # 184-194s: sway + frown (gentle, times = 10/1.5 = 6)
            (184, 194, "custom", "head_sway", {"times": 6, "intensity": 0.6}),
            (184, 194, "builtin", "BrowFrown", {"intensity": 0.6}),

            # 194-204s: thoughtful
            (194, 204, "builtin", "Thoughtful", {}),

            # 204-207s: surprise
            (204, 207, "builtin", "Surprise", {}),

            # 207-217s: look top left + thoughtful
            (207, 217, "custom", "head_positions", {"x": 0.5, "y": 0.5, "z": 1.0, "duration": 10}),
            (207, 217, "builtin", "Thoughtful", {}),

            # 217-223s: close eyes
            (217, 223, "custom", "head_positions", {"x": 0.0, "y": 0.0, "z": 1.0}),
            (217, 223, "builtin", "CloseEyes", {}),

            # 223s: open eyes
            (223, 231, "builtin", "OpenEyes", {}),

            # 223-231s: smile
            (223, 231, "builtin", "Smile", {}),

            # 231-238s: big smile
            (231, 238, "builtin", "BigSmile", {}),

            # 238-267s: smile
            (238, 267, "builtin", "Smile", {}),

            # 267-285s: smile
            (267, 285, "builtin", "Smile", {}),
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

                    # OpenEyes only executes once
                    if method_name == "OpenEyes":
                        task = asyncio.create_task(
                            self.builtin.test_single_gesture(method_name, intensity, 0.1)
                        )
                    else:
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
        remaining = 285 - (asyncio.get_event_loop().time() - start)
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
    music_url = "https://drive.google.com/uc?export=download&id=1y7o8Qaz46kL8Mn9fEFSNC4fJCN_IQoCY"
    perf = SomeoneLikeYouPerformance(music_url)

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