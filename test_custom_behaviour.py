from furhat_realtime_api import AsyncFurhatClient
import asyncio


class CustomBehaviorTester:
    def __init__(self, ip_address="127.0.0.1"):
        self.furhat = AsyncFurhatClient(ip_address)

    async def setup(self):
        await self.furhat.connect()
        await self.furhat.request_voice_config(gender="female", language="en-US")

    async def custom_very_happy(self, intensity=1.0):
        """Custom: Very happy"""
        params = {
            "SMILE_OPEN": intensity * 0.8,
            "BROW_UP_LEFT": intensity,
            "BROW_UP_RIGHT": intensity,
            "SURPRISE": intensity * 0.6
        }
        await self.furhat.request_face_params(params)
        await asyncio.sleep(2.0)
        await self.furhat.request_face_reset()

    async def custom_confused(self, intensity=1.0):
        params = {
            "BROW_DOWN_LEFT": intensity * 0.8,
            "BROW_UP_RIGHT": intensity * 1.0,
            "EXPR_FEAR": intensity * 0.5
        }
        await self.furhat.request_face_params(params)
        await asyncio.sleep(2.0)
        await self.furhat.request_face_reset()

    async def custom_determined(self, intensity=1.0):
        params = {
            "BROW_DOWN_LEFT": intensity * 0.6,
            "BROW_DOWN_RIGHT": intensity * 0.6,
            "EXPR_ANGER": intensity * 0.1,
        }
        await self.furhat.request_face_params(params)
        await asyncio.sleep(2.0)
        await self.furhat.request_face_reset()

    async def custom_shy(self, intensity=1.0):
        params = {
            "SMILE_OPEN": intensity * 0.2,
            "BROW_DOWN_LEFT": intensity * 0.4,
            "BROW_DOWN_RIGHT": intensity * 0.4,
        }
        await self.furhat.request_face_params(params)
        await asyncio.sleep(2.0)
        await self.furhat.request_face_reset()

    async def custom_narrow(self, intensity=1.0):
        params = {
            "EYE_SQUINT_LEFT": intensity,
            "EYE_SQUINT_RIGHT": intensity,
            "BROW_DOWN_LEFT": intensity * 0.3,
            "BROW_DOWN_RIGHT": intensity * 0.3,
        }
        await self.furhat.request_face_params(params)
        await asyncio.sleep(2.0)
        await self.furhat.request_face_reset()

    async def raise_left_brow(self, intensity=1.0):
        """Custom: Raise left eyebrow"""
        params = {
            "BROW_UP_LEFT": intensity,
            "BROW_UP_RIGHT": 0.0,
        }
        await self.furhat.request_face_params(params)
        await asyncio.sleep(2.0)
        await self.furhat.request_face_reset()

    async def raise_right_brow(self, intensity=1.0):
        params = {
            "BROW_UP_LEFT": 0.0,
            "BROW_UP_RIGHT": intensity,
        }
        await self.furhat.request_face_params(params)
        await asyncio.sleep(2.0)
        await self.furhat.request_face_reset()

    async def eye_look_direction(self, x=0.0, y=0.0, duration=2):
        """x: -1.0 (left) to 1.0 (right)
        y: -1.0 (down) to 1.0 (up)
        """
        params = {}

        # Left-right
        if x > 0:  # Look right
            params["EYE_LOOK_OUT_RIGHT"] = abs(x)  # out: right eye, in: left eye
            params["EYE_LOOK_IN_LEFT"] = abs(x)
        elif x < 0:  # Look left
            params["EYE_LOOK_OUT_LEFT"] = abs(x)
            params["EYE_LOOK_IN_RIGHT"] = abs(x)

        # Up-down
        if y > 0:  # Look up
            params["EYE_LOOK_UP_LEFT"] = abs(y)
            params["EYE_LOOK_UP_RIGHT"] = abs(y)
        elif y < 0:  # Look down
            params["EYE_LOOK_DOWN_LEFT"] = abs(y)
            params["EYE_LOOK_DOWN_RIGHT"] = abs(y)

        await self.furhat.request_face_params(params)
        await asyncio.sleep(duration)
        await self.furhat.request_face_reset()

    async def custom_relaxed(self, intensity=1.0):
        params = {
            "SMILE_OPEN": intensity * 0.2,
            "EYE_SQUINT_LEFT": 0.5,
            "EYE_SQUINT_RIGHT": 0.5,
            "BROW_UP_LEFT": intensity * 0.1,
            "BROW_UP_RIGHT": intensity * 0.1,
        }
        await self.furhat.request_face_params(params)
        await asyncio.sleep(2.0)
        await self.furhat.request_face_reset()

    # ============ Head Control ============

    async def head_positions(self, x=0.0, y=0.0, z=1.0, duration=2):
        await self.furhat.request_attend_location(x=x, y=y, z=z)
        await asyncio.sleep(duration)
        # go back to center
        await self.furhat.request_attend_location(x=0.0, y=0.0, z=1.0)

    async def head_tilts(self, yaw=0, pitch=0, roll=0, duration=2):
        # +: yaw:left, pitch:down, roll:left roll
        await self.furhat.request_face_headpose(yaw=yaw, pitch=pitch, roll=roll, relative=True)
        await asyncio.sleep(duration)
        # go back to center
        await self.furhat.request_face_headpose(yaw=0, pitch=0, roll=0, relative=False)

    async def head_nod_fast(self, times, intensity=1.0):
        for i in range(times):
            await self.furhat.request_gesture_start("Nod", intensity=intensity, duration=1)
            await asyncio.sleep(1.0)

    async def head_sway(self, times, intensity=1.0):
        for i in range(times):
            await self.furhat.request_gesture_start("Roll", intensity=intensity, duration=1)
            await asyncio.sleep(1.5)
            # duration=2,sleep=3 can make it non-stop, if it is i and 2 will pause
            # duration=1, sleep=1.5 can make it non-stop as well

    async def head_shake_fast(self, times, intensity=1):
        for i in range(times):
            await self.furhat.request_gesture_start("Shake", intensity=intensity, duration=1.0)
            await asyncio.sleep(1.0)
            # in builtin functions, sleep time is 2 when duration is 1

    async def cleanup(self):
        """Cleanup"""
        await self.furhat.request_face_reset()
        await self.furhat.request_attend_location(x=0.0, y=0.0, z=1.0)
        await self.furhat.disconnect()


async def main():
    tester = CustomBehaviorTester()
    await tester.setup()

    # await tester.furhat.request_speak_text("Testing very happy expression", wait=True)
    # await tester.custom_very_happy()

    # await tester.furhat.request_speak_text("Testing confused expression", wait=True)
    # await tester.custom_confused()

    # await tester.furhat.request_speak_text("Testing determined expression", wait=True)
    # await tester.custom_determined()

    # await tester.furhat.request_speak_text("shy", wait=True)
    # await tester.custom_shy()

    # await tester.furhat.request_speak_text("narrow eyes", wait=True)
    # await tester.custom_narrow()

    # await tester.furhat.request_speak_text("raise left eyebrow", wait=True)
    # await tester.raise_left_brow()

    # await tester.furhat.request_speak_text("raise right eyebrow", wait=True)
    # await tester.raise_right_brow()

    # await tester.furhat.request_speak_text("look to the left", wait=True)
    # await tester.eye_look_direction(x=-1, y=0)
    #
    # await tester.furhat.request_speak_text("look to the right", wait=True)
    # await tester.eye_look_direction(x=1, y=0)
    #
    # await tester.furhat.request_speak_text("look up", wait=True)
    # await tester.eye_look_direction(x=0, y=1)
    #
    # await tester.furhat.request_speak_text("look down", wait=True)
    # await tester.eye_look_direction(x=0, y=-1)
    #
    # await tester.furhat.request_speak_text("look to the upper right", wait=True)
    # await tester.eye_look_direction(x=0.5, y=0.5)

    # await tester.furhat.request_speak_text("relaxed face", wait=True)
    # await tester.custom_relaxed()

    #  ===== Head movements =====
    await tester.furhat.request_speak_text("head positions", wait=True)
    await tester.head_positions(0,-0.5,1)  # look down
    #
    # await tester.furhat.request_speak_text("head tilts", wait=True)
    # await tester.head_tilts(50, 0, 0)
    #
    # await tester.furhat.request_speak_text("nod", wait=True)
    # await tester.head_nod_fast(times=5)

    # await tester.furhat.request_speak_text("head sway", wait=True)
    # await tester.head_sway(times=5)

    # await tester.furhat.request_speak_text("shake", wait=True)
    # await tester.head_shake_fast(times=5)

    await tester.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
