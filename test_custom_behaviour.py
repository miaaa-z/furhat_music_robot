from furhat_realtime_api import AsyncFurhatClient
import asyncio


class CustomBehaviorTester:
    """Test custom behaviors (facial parameters, head control, compound actions)"""
    def __init__(self, ip_address="127.0.0.1"):
        self.furhat = AsyncFurhatClient(ip_address)

    async def setup(self):
        """Connect and configure Furhat"""
        await self.furhat.connect()
        await self.furhat.request_voice_config(gender="female", language="en-US")


    async def custom_happy_intense(self, intensity=1.0):
        """Custom: Very happy"""
        params = {
            "SMILE_OPEN": intensity * 0.8,
            "BROW_UP_LEFT": intensity,
            "BROW_UP_RIGHT": intensity,
            "SURPRISE": intensity * 0.5
        }
        await self.furhat.request_face_params(params)
        await asyncio.sleep(2.0)
        await self.furhat.request_face_reset()

    async def custom_confused(self, intensity=0.7):
        """Custom: Confused"""
        params = {
            "BROW_DOWN_LEFT": intensity * 0.8,
            "BROW_UP_RIGHT": intensity * 0.9,
            "EXPR_FEAR": intensity * 0.5
        }
        await self.furhat.request_face_params(params)
        await asyncio.sleep(2.0)
        await self.furhat.request_face_reset()

    async def custom_excited(self, intensity=1.0):
        """Custom: Excited"""
        params = {
            "OH": intensity,
            "BROW_UP_LEFT": intensity,
            "BROW_UP_RIGHT": intensity,
            "SURPRISE": intensity * 0.7,
            "SMILE_OPEN": intensity * 0.2
        }
        await self.furhat.request_face_params(params)
        await asyncio.sleep(2.0)
        await self.furhat.request_face_reset()

    async def custom_determined(self, intensity=0.8):
        """Custom: Determined"""
        params = {
            "BROW_DOWN_LEFT": intensity * 0.6,
            "BROW_DOWN_RIGHT": intensity * 0.6,
            "EXPR_ANGER": intensity * 0.1
        }
        await self.furhat.request_face_params(params)
        await asyncio.sleep(2.0)
        await self.furhat.request_face_reset()

    async def custom_shy(self, intensity=1.0):
        """Custom: Shy"""
        params = {
            "SMILE_OPEN": intensity * 0.2,
            "BROW_DOWN_LEFT": intensity * 0.3,
            "BROW_DOWN_RIGHT": intensity * 0.3,
        }
        await self.furhat.request_face_params(params)
        await asyncio.sleep(2.0)
        await self.furhat.request_face_reset()

    async def custom_slight_frown(self, intensity=0.5):
        """Custom: Slightly frown"""
        params = {
            "BROW_DOWN_LEFT": intensity * 0.5,
            "BROW_DOWN_RIGHT": intensity * 0.5,
            "BROW_IN_LEFT": intensity * 0.5,
            "BROW_IN_RIGHT": intensity * 0.5,
        }
        await self.furhat.request_face_params(params)
        await asyncio.sleep(2.0)
        await self.furhat.request_face_reset()

    async def custom_pout(self, intensity=1.0):
        """Custom: Pout lips """
        params = {
            "MOUTH_POUT": intensity,  # seems that it's not the built-in parameter
            "BROW_DOWN_LEFT": intensity * 0.3,
            "BROW_DOWN_RIGHT": intensity * 0.3,
        }
        await self.furhat.request_face_params(params)
        await asyncio.sleep(2.0)
        await self.furhat.request_face_reset()

    async def custom_scrunch_face(self, intensity=1.0):
        """Custom: Scrunch face """
        params = {
            "NOSE_WRINKLE": intensity,  # seems that it's not the built-in parameter
            "BROW_DOWN_LEFT": intensity * 0.9,
            "BROW_DOWN_RIGHT": intensity * 0.9,
            "BROW_IN_LEFT": intensity,
            "BROW_IN_RIGHT": intensity,
            "EXPR_DISGUST": intensity * 0.2,
        }
        await self.furhat.request_face_params(params)
        await asyncio.sleep(2.0)
        await self.furhat.request_face_reset()

    async def custom_bite_lip(self, intensity=0.7):
        """Custom: Bite lower lip"""
        params = {
            "LIP_BITE": intensity,      # seems that it's not the built-in parameter
            "BROW_UP_LEFT": intensity * 0.5,
            "BROW_UP_RIGHT": intensity * 0.3,
        }
        await self.furhat.request_face_params(params)
        await asyncio.sleep(2.0)
        await self.furhat.request_face_reset()

    async def custom_narrow(self, intensity=0.7):
        """Custom: narrow eyes """
        params = {
            "EYE_SQUINT_LEFT": intensity,
            "EYE_SQUINT_RIGHT": intensity,
            "BROW_DOWN_LEFT": intensity * 0.3,
            "BROW_DOWN_RIGHT": intensity * 0.3,
        }
        await self.furhat.request_face_params(params)
        await asyncio.sleep(2.0)
        await self.furhat.request_face_reset()

    async def custom_raise_one_brow(self, intensity=0.8, side="left"):
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

    async def head_shake_fast_vertical(self, repetitions=3, speed=0.3):
        """Head shake fast up-down """
        for i in range(repetitions):
            # Nod down
            await self.furhat.request_face_headpose(yaw=0, pitch=-15, roll=0, relative=True)
            await asyncio.sleep(speed)
            # Nod up
            await self.furhat.request_face_headpose(yaw=0, pitch=15, roll=0, relative=True)
            await asyncio.sleep(speed)
        # Return to center
        await self.furhat.request_face_headpose(yaw=0, pitch=0, roll=0, relative=True)

    async def head_nod_fast(self, times=4):
        for i in range(times):
            await self.furhat.request_gesture_start("Nod", intensity=1.0, duration=0.3)
            await asyncio.sleep(0.4)

    async def head_shake_fast_horizontal(self, times=3):
        # shaking head fast
        for i in range(times):
            await self.furhat.request_gesture_start("Roll", intensity=1.0, duration=0.3)
            await asyncio.sleep(0.4)


    # ============ Shoulder Movements (Simulated with gestures) ============
    # Note: Furhat doesn't have shoulders, so we simulate with head/body language

    async def shoulder_shrug_both(self, intensity=0.8):
        """Simulate shoulder shrug """
        # Raise head slightly and tilt it
        await self.furhat.request_face_headpose(yaw=0, pitch=-10, roll=0, relative=True)
        await self.furhat.request_face_params({
            "BROW_UP_LEFT": intensity,
            "BROW_UP_RIGHT": intensity,
            "SMILE_OPEN": 0.3
        })
        await asyncio.sleep(1.0)
        # Return
        await self.furhat.request_face_headpose(yaw=0, pitch=0, roll=0, relative=True)
        await self.furhat.request_face_reset()

    async def shoulder_shrug_one(self, side="left", intensity=0.8):
        """Simulate one shoulder shrug """
        if side == "left":
            # Tilt head to right (when left shoulder goes up)
            await self.furhat.request_face_headpose(yaw=0, pitch=0, roll=-10, relative=True)
            await self.furhat.request_face_params({
                "BROW_UP_LEFT": intensity * 0.5,
            })
        else:  # right
            # Tilt head to left (when right shoulder goes up)
            await self.furhat.request_face_headpose(yaw=0, pitch=0, roll=10, relative=True)
            await self.furhat.request_face_params({
                "BROW_UP_RIGHT": intensity * 0.5,
            })
        await asyncio.sleep(1.0)
        # Return
        await self.furhat.request_face_headpose(yaw=0, pitch=0, roll=0, relative=False)
        await self.furhat.request_face_reset()

    async def cleanup(self):
        """Cleanup"""
        await self.furhat.request_face_reset()
        await self.furhat.request_attend_location(x=0.0, y=0.0, z=1.0)
        await self.furhat.disconnect()


async def main():
    tester = CustomBehaviorTester()
    await tester.setup()

    # await tester.furhat.request_speak_text("Testing happy intense expression", wait=True)
    # await tester.custom_happy_intense(intensity=1.0)

    # await tester.furhat.request_speak_text("Testing confused expression", wait=True)
    # await tester.custom_confused(intensity=1.0)

    # await tester.furhat.request_speak_text("Testing excited expression", wait=True)
    # await tester.custom_excited(intensity=1.0)
    #
    # await tester.furhat.request_speak_text("Testing determined expression", wait=True)
    # await tester.custom_determined(intensity=1.0)
    #
    # await tester.furhat.request_speak_text("Testing shy expression", wait=True)
    # await tester.custom_shy(intensity= 0.8)

    # await tester.furhat.request_speak_text("Testing slight frown", wait=True)
    # await tester.custom_slight_frown(intensity=1.0)
    #
    # await tester.furhat.request_speak_text("Testing pout", wait=True)
    # await tester.custom_pout(intensity=1.0)

    # await tester.furhat.request_speak_text("Testing scrunch face", wait=True)
    # await tester.custom_scrunch_face(intensity=1.0)
    #
    # await tester.furhat.request_speak_text("Testing bite lip", wait=True)
    # await tester.custom_bite_lip(intensity=1.0)
    #
    # await tester.furhat.request_speak_text("Testing narrow eyes", wait=True)
    # await tester.custom_narrow(intensity=0.8)

    # await tester.furhat.request_speak_text("Testing raise left eyebrow", wait=True)
    # await tester.custom_raise_one_brow(intensity=1.0, side="left")
    #
    # await tester.furhat.request_speak_text("Testing raise right eyebrow", wait=True)
    # await tester.custom_raise_one_brow(intensity=1.0, side="right")
    #
    # # ===== Head movements =====
    # await tester.furhat.request_speak_text("Testing head positions", wait=True)
    # await tester.test_head_positions()
    #
    # await tester.furhat.request_speak_text("Testing head tilts", wait=True)
    # await tester.test_head_tilts()
    #
    # await tester.furhat.request_speak_text("Testing vertical head shake", wait=True)
    # await tester.head_nod_fast(times=5)
    #
    # await tester.furhat.request_speak_text("Testing horizontal head shake", wait=True)
    # await tester.head_shake_fast_horizontal(times=5)

    # # ===== Shoulder movements =====
    # await tester.furhat.request_speak_text("Testing both shoulders shrug", wait=True)
    # await tester.shoulder_shrug_both(intensity=0.8)
    #
    await tester.furhat.request_speak_text("Testing left shoulder shrug", wait=True)
    await tester.shoulder_shrug_one(side="left", intensity=0.8)

    await tester.furhat.request_speak_text("Testing right shoulder shrug", wait=True)
    await tester.shoulder_shrug_one(side="right", intensity=0.8)


    await tester.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
