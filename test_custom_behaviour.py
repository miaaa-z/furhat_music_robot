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

    async def custom_raise_one_brow(self, intensity=1.0, side="left"):
        """Custom: Raise one eyebrow """
        if side == "left":
            params = {
                "BROW_UP_LEFT": intensity,
                "BROW_UP_RIGHT": 0.0,
            }
        else:  # right
            params = {
                "BROW_UP_LEFT": 0.0,
                "BROW_UP_RIGHT": intensity,
            }
        await self.furhat.request_face_params(params)
        await asyncio.sleep(2.0)
        await self.furhat.request_face_reset()

    # ============ Head Control ============

    async def test_head_positions(self):
        """Test head positions"""
        positions = [
            ("Center", (0.0, 0.0, 1.0)),
            ("Left", (1.0, 0.0, 1.0)),
            ("Right", (-1.0, 0.0, 1.0)),
            ("Up", (0.0, 1.0, 1.0)),
            ("Down", (0.0, -1.0, 1.0)),
        ]

        for name, (x, y, z) in positions:
            await self.furhat.request_attend_location(x=x, y=y, z=z)
            await asyncio.sleep(2)

        await self.furhat.request_attend_location(x=0.0, y=0.0, z=1.0)

    async def test_head_tilts(self):
        """Test head tilts"""
        tilts = [
            {"yaw": 20, "pitch": 0, "roll": 0},    # Left
            {"yaw": -20, "pitch": 0, "roll": 0},   # Right
            {"yaw": 0, "pitch": 15, "roll": 0},    # Down
            {"yaw": 0, "pitch": -15, "roll": 0},   # Up
            {"yaw": 0, "pitch": 0, "roll": 15},    # Tilt right
            {"yaw": 0, "pitch": 0, "roll": -15},   # Tilt left
        ]

        for pose in tilts:
            await self.furhat.request_face_headpose(**pose, relative=True)
            await asyncio.sleep(1.5)

        await self.furhat.request_face_headpose(yaw=0, pitch=0, roll=0, relative=False)

    async def head_nod_fast(self, times):
        for i in range(times):
            await self.furhat.request_gesture_start("Nod", intensity=0.8, duration=0.3)
            await asyncio.sleep(0.4)

    async def head_sway(self, times):
        for i in range(times):
            await self.furhat.request_gesture_start("Roll", intensity=0.7, duration=2)
            await asyncio.sleep(3)
            # duration is 1 second shorter than sleep time can make it always sway left to right

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

    # await tester.furhat.request_speak_text(" raise left eyebrow", wait=True)
    # await tester.custom_raise_one_brow(intensity=1.0, side="left")
    #
    # await tester.furhat.request_speak_text("raise right eyebrow", wait=True)
    # await tester.custom_raise_one_brow(intensity=1.0, side="right")

    #  ===== Head movements =====
    # await tester.furhat.request_speak_text("Testing head positions", wait=True)
    # await tester.test_head_positions()
    #
    # await tester.furhat.request_speak_text("Testing head tilts", wait=True)
    # await tester.test_head_tilts()
    #
    # await tester.furhat.request_speak_text("Testing vertical head shake", wait=True)
    # await tester.head_nod_fast(times=5)
    #
    # # await tester.furhat.request_speak_text("Testing horizontal head shake", wait=True)
    # await tester.furhat.request_speak_text("Testing swaying head", wait=True)
    # await tester.head_sway(times=4)

    await tester.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
