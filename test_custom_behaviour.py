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

    # ============ Original Custom Expressions ============

    async def custom_happy_intense(self, intensity=1.0):
        """Custom: Very happy"""
        params = {
            "SMILE_OPEN": intensity,
            "BROW_UP_LEFT": intensity * 0.7,
            "BROW_UP_RIGHT": intensity * 0.7,
            "SURPRISE": intensity * 0.3
        }
        await self.furhat.request_face_params(params)
        await asyncio.sleep(2.0)
        await self.furhat.request_face_reset()

    async def custom_confused(self, intensity=0.7):
        """Custom: Confused"""
        params = {
            "BROW_DOWN_LEFT": intensity * 0.5,
            "BROW_UP_RIGHT": intensity * 0.6,
            "EXPR_FEAR": intensity * 0.2
        }
        await self.furhat.request_face_params(params)
        await asyncio.sleep(2.0)
        await self.furhat.request_face_reset()

    async def custom_excited(self, intensity=1.0):
        """Custom: Excited"""
        params = {
            "SMILE_OPEN": intensity,
            "BROW_UP_LEFT": intensity,
            "BROW_UP_RIGHT": intensity,
            "SURPRISE": intensity * 0.7
        }
        await self.furhat.request_face_params(params)
        await asyncio.sleep(2.0)
        await self.furhat.request_face_reset()

    async def custom_determined(self, intensity=0.8):
        """Custom: Determined"""
        params = {
            "BROW_DOWN_LEFT": intensity * 0.6,
            "BROW_DOWN_RIGHT": intensity * 0.6,
            "EXPR_ANGER": intensity * 0.3
        }
        await self.furhat.request_face_params(params)
        await asyncio.sleep(2.0)
        await self.furhat.request_face_reset()

    async def custom_shy(self, intensity=0.6):
        """Custom: Shy"""
        params = {
            "SMILE_OPEN": intensity * 0.5,
            "BROW_UP_LEFT": intensity * 0.3,
            "BROW_UP_RIGHT": intensity * 0.3,
        }
        await self.furhat.request_face_params(params)
        await self.furhat.request_attend_location(x=0.7, y=-0.3, z=1.0)
        await asyncio.sleep(2.0)
        await self.furhat.request_face_reset()
        await self.furhat.request_attend_location(x=0.0, y=0.0, z=1.0)

    # ============ New Custom Expressions ============

    async def custom_slight_frown(self, intensity=0.5):
        """Custom: Slightly frown"""
        params = {
            "BROW_DOWN_LEFT": intensity * 0.4,
            "BROW_DOWN_RIGHT": intensity * 0.4,
            "BROW_IN_LEFT": intensity * 0.3,
            "BROW_IN_RIGHT": intensity * 0.3,
        }
        await self.furhat.request_face_params(params)
        await asyncio.sleep(2.0)
        await self.furhat.request_face_reset()

    async def custom_pout(self, intensity=0.8):
        """Custom: Pout lips (撅嘴)"""
        params = {
            "MOUTH_POUT": intensity,
            "BROW_DOWN_LEFT": intensity * 0.3,
            "BROW_DOWN_RIGHT": intensity * 0.3,
        }
        await self.furhat.request_face_params(params)
        await asyncio.sleep(2.0)
        await self.furhat.request_face_reset()

    async def custom_scrunch_face(self, intensity=0.8):
        """Custom: Scrunch face (脸皱在一起)"""
        params = {
            "NOSE_WRINKLE": intensity,
            "BROW_DOWN_LEFT": intensity * 0.7,
            "BROW_DOWN_RIGHT": intensity * 0.7,
            "BROW_IN_LEFT": intensity * 0.6,
            "BROW_IN_RIGHT": intensity * 0.6,
            "EXPR_DISGUST": intensity * 0.5,
        }
        await self.furhat.request_face_params(params)
        await asyncio.sleep(2.0)
        await self.furhat.request_face_reset()

    async def custom_bite_lip(self, intensity=0.7):
        """Custom: Bite lower lip"""
        params = {
            "LIP_BITE": intensity,
            "BROW_UP_LEFT": intensity * 0.2,
            "BROW_UP_RIGHT": intensity * 0.2,
        }
        await self.furhat.request_face_params(params)
        await asyncio.sleep(2.0)
        await self.furhat.request_face_reset()

    async def custom_squint(self, intensity=0.7):
        """Custom: Squint eyes (眯眼)"""
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
        """Custom: Raise one eyebrow (单边挑眉)"""
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
            await asyncio.sleep(1.5)

        await self.furhat.request_attend_location(x=0.0, y=0.0, z=1.0)

    async def test_head_tilts(self):
        """Test head tilts"""
        tilts = [
            {"yaw": 20, "pitch": 0, "roll": 0},    # Left
            {"yaw": -20, "pitch": 0, "roll": 0},   # Right
            {"yaw": 0, "pitch": 15, "roll": 0},    # Up
            {"yaw": 0, "pitch": -15, "roll": 0},   # Down
            {"yaw": 0, "pitch": 0, "roll": 15},    # Tilt left
            {"yaw": 0, "pitch": 0, "roll": -15},   # Tilt right
        ]

        for pose in tilts:
            await self.furhat.request_face_headpose(**pose)
            await asyncio.sleep(1.5)

        await self.furhat.request_face_headpose(yaw=0, pitch=0, roll=0)

    async def head_shake_fast_vertical(self, repetitions=3, speed=0.3):
        """Head shake fast up-down (头上下快速甩)"""
        for i in range(repetitions):
            # Nod down
            await self.furhat.request_face_headpose(yaw=0, pitch=-15, roll=0)
            await asyncio.sleep(speed)
            # Nod up
            await self.furhat.request_face_headpose(yaw=0, pitch=15, roll=0)
            await asyncio.sleep(speed)
        # Return to center
        await self.furhat.request_face_headpose(yaw=0, pitch=0, roll=0)

    async def head_shake_fast_horizontal(self, repetitions=3, speed=0.3):
        """Head shake fast left-right (头左右快速甩)"""
        for i in range(repetitions):
            # Turn left
            await self.furhat.request_face_headpose(yaw=20, pitch=0, roll=0)
            await asyncio.sleep(speed)
            # Turn right
            await self.furhat.request_face_headpose(yaw=-20, pitch=0, roll=0)
            await asyncio.sleep(speed)
        # Return to center
        await self.furhat.request_face_headpose(yaw=0, pitch=0, roll=0)

    # ============ Shoulder Movements (Simulated with gestures) ============
    # Note: Furhat doesn't have shoulders, so we simulate with head/body language

    async def shoulder_shrug_both(self, intensity=0.8):
        """Simulate shoulder shrug (耸肩膀)"""
        # Raise head slightly and tilt it
        await self.furhat.request_face_headpose(yaw=0, pitch=5, roll=0)
        await self.furhat.request_face_params({
            "BROW_UP_LEFT": intensity * 0.5,
            "BROW_UP_RIGHT": intensity * 0.5,
        })
        await asyncio.sleep(1.0)
        # Return
        await self.furhat.request_face_headpose(yaw=0, pitch=0, roll=0)
        await self.furhat.request_face_reset()

    async def shoulder_shrug_one(self, side="left", intensity=0.8):
        """Simulate one shoulder shrug (单边耸肩)"""
        if side == "left":
            # Tilt head to right (when left shoulder goes up)
            await self.furhat.request_face_headpose(yaw=0, pitch=0, roll=-10)
            await self.furhat.request_face_params({
                "BROW_UP_LEFT": intensity * 0.5,
            })
        else:  # right
            # Tilt head to left (when right shoulder goes up)
            await self.furhat.request_face_headpose(yaw=0, pitch=0, roll=10)
            await self.furhat.request_face_params({
                "BROW_UP_RIGHT": intensity * 0.5,
            })
        await asyncio.sleep(1.0)
        # Return
        await self.furhat.request_face_headpose(yaw=0, pitch=0, roll=0)
        await self.furhat.request_face_reset()

    # ============ Compound Actions ============

    async def compound_happy_dance(self):
        """Compound: Happy dance"""
        await self.furhat.request_face_params({
            "SMILE_OPEN": 1.0,
            "BROW_UP_LEFT": 0.8,
            "BROW_UP_RIGHT": 0.8
        })

        await self.furhat.request_attend_location(x=1.0, y=0.0, z=1.0)
        await asyncio.sleep(0.5)

        await self.furhat.request_attend_location(x=-1.0, y=0.0, z=1.0)
        await asyncio.sleep(0.5)

        await self.furhat.request_attend_location(x=0.0, y=0.0, z=1.0)
        await self.furhat.request_gesture_start("Nod", intensity=0.8, duration=0.5)
        await asyncio.sleep(0.5)

        await self.furhat.request_face_reset()

    async def compound_sad_moment(self):
        """Compound: Sad moment"""
        await self.furhat.request_face_params({
            "EXPR_SAD": 0.8,
            "BROW_DOWN_LEFT": 0.5,
            "BROW_DOWN_RIGHT": 0.5
        })

        await self.furhat.request_attend_location(x=0.0, y=-1.0, z=1.0)
        await asyncio.sleep(2.0)

        await self.furhat.request_attend_location(x=0.0, y=0.0, z=1.0)
        await asyncio.sleep(1.0)

        await self.furhat.request_face_reset()

    async def compound_thinking(self):
        """Compound: Thinking"""
        await self.furhat.request_gesture_start("Thoughtful", duration=1.0)
        await self.furhat.request_attend_location(x=0.7, y=0.7, z=1.0)
        await asyncio.sleep(0.8)

        await self.furhat.request_face_params({
            "BROW_DOWN_LEFT": 0.5,
            "BROW_UP_RIGHT": 0.6,
        })
        await asyncio.sleep(1.0)

        await self.furhat.request_gesture_start("Oh", duration=1.0)
        await self.furhat.request_attend_location(x=0.0, y=0.0, z=1.0)
        await asyncio.sleep(0.5)

        await self.furhat.request_face_reset()

    # Template: Add your own
    async def custom_YOUR_EMOTION(self, intensity=0.8):
        """Template for your custom expression"""
        params = {
            # "SMILE_OPEN": intensity * 0.5,
            # "BROW_UP_LEFT": intensity * 0.3,
        }
        await self.furhat.request_face_params(params)
        await asyncio.sleep(2.0)
        await self.furhat.request_face_reset()

    async def compound_YOUR_ACTION(self):
        """Template for your compound action"""
        # Step 1: Set expression
        # await self.furhat.request_face_params({...})

        # Step 2: Move head
        # await self.furhat.request_attend_location(...)

        await self.furhat.request_face_reset()

    async def cleanup(self):
        """Cleanup"""
        await self.furhat.request_face_reset()
        await self.furhat.request_attend_location(x=0.0, y=0.0, z=1.0)
        await self.furhat.disconnect()


async def main():
    tester = CustomBehaviorTester()
    await tester.setup()

    # Choose one mode:

    # ===== Original expressions =====
    # await tester.custom_happy_intense(intensity=1.0)
    # await tester.custom_confused(intensity=0.7)
    # await tester.custom_excited(intensity=1.0)
    # await tester.custom_determined(intensity=0.8)
    # await tester.custom_shy(intensity=0.6)

    # ===== New expressions =====
    # await tester.custom_slight_frown(intensity=0.5)
    # await tester.custom_pout(intensity=0.8)
    # await tester.custom_scrunch_face(intensity=0.8)
    # await tester.custom_bite_lip(intensity=0.7)
    # await tester.custom_squint(intensity=0.7)
    # await tester.custom_raise_one_brow(intensity=0.8, side="left")  # or side="right"

    # ===== Head movements =====
    # await tester.test_head_positions()
    # await tester.test_head_tilts()
    # await tester.head_shake_fast_vertical(repetitions=3, speed=0.3)
    # await tester.head_shake_fast_horizontal(repetitions=3, speed=0.3)

    # ===== Shoulder movements (simulated) =====
    # await tester.shoulder_shrug_both(intensity=0.8)
    # await tester.shoulder_shrug_one(side="left", intensity=0.8)  # or side="right"

    # ===== Compound actions =====
    await tester.compound_happy_dance()
    # await tester.compound_sad_moment()
    # await tester.compound_thinking()

    await tester.cleanup()


if __name__ == "__main__":
    asyncio.run(main())