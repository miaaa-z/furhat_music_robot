from furhat_realtime_api import AsyncFurhatClient
import asyncio


class GestureTester:
    def __init__(self):
        self.furhat = AsyncFurhatClient("127.0.0.1")

    async def setup(self):
        await self.furhat.connect()
        print("Connected to Furhat\n")
        await self.furhat.request_voice_config(gender="female", language="en-US")
        print("Voice configured: female, English\n")

    async def test_predefined_gestures(self):
        """Test all predefined gestures from Furhat"""
        print("=== Testing Predefined Gestures ===\n")

        gestures = [
            "BigSmile",
            "Smile",
            "Wink",
            "Nod",  # Head movement gesture
            "Shake",  # Head movement gesture
            "Roll",  # Head movement gesture
            "BrowRaise",
            "BrowFrown",
            "Surprise",
            "Oh",
            "Thoughtful",
            "GazeAway",
            "Blink",
            "CloseEyes",
            "OpenEyes"
        ]

        for gesture in gestures:
            print(f"Testing gesture: {gesture}")
            try:
                # Announce gesture, then perform it
                await self.furhat.request_speak_text(f"{gesture}", wait=True)
                await asyncio.sleep(0.3)  # Brief pause after speaking

                # Perform gesture with higher intensity and longer duration
                await self.furhat.request_gesture_start(
                    gesture,
                    intensity=1.0,
                    duration=2.0,  # Longer duration for visibility
                    wait=True
                )
                await asyncio.sleep(0.5)
                print(f"Success\n")
            except Exception as e:
                print(f"Failed: {e}\n")

    async def test_face_params(self):
        """Test facial expression parameters"""
        print("\n=== Testing Facial Parameters ===\n")

        expressions = [
            ("Happy", {"SMILE_OPEN": 0.8, "BROW_UP_LEFT": 0.5, "BROW_UP_RIGHT": 0.5}),
            ("Sad", {"EXPR_SAD": 0.8, "BROW_DOWN_LEFT": 0.5, "BROW_DOWN_RIGHT": 0.5}),
            ("Surprised", {"SURPRISE": 1.0, "BROW_UP_LEFT": 1.0, "BROW_UP_RIGHT": 1.0}),
            ("Angry", {"EXPR_ANGER": 0.8, "BROW_DOWN_LEFT": 0.7, "BROW_IN_LEFT": 0.7}),
            ("Disgusted", {"EXPR_DISGUST": 0.8}),
            ("Fearful", {"EXPR_FEAR": 0.8}),
        ]

        for name, params in expressions:
            print(f"Expression: {name}")
            print(f"Parameters: {params}")

            # Announce expression, then apply it
            await self.furhat.request_speak_text(f"{name}", wait=True)
            await asyncio.sleep(0.3)

            await self.furhat.request_face_params(params)
            await asyncio.sleep(1.5)

            # Reset to neutral
            await self.furhat.request_face_reset()
            await asyncio.sleep(0.5)
            print(f"Done\n")

    async def test_attention_system(self):
        """Test attention system for head positioning"""
        print("\n=== Testing Attention System ===\n")

        # Attention points: (x, y, z) relative to robot in meters
        # x: horizontal (positive = robot's left, negative = right(robot's right))
        # y: vertical (positive = up, negative = down)
        # z: distance from robot (must be positive)
        locations = [
            ("Center", (0.0, 0.0, 1.0)),
            ("Left", (1.0, 0.0, 1.0)),
            ("Right", (-1.0, 0.0, 1.0)),
            ("Up", (0.0, 1.0, 1.0)),
            ("Down", (0.0, -1.0, 1.0)),
        ]

        for name, (x, y, z) in locations:
            print(f"Looking at: {name}")
            print(f"Position: x={x}, y={y}, z={z}")

            await self.furhat.request_speak_text(f"Looking {name}", wait=True)
            await asyncio.sleep(0.3)

            await self.furhat.request_attend_location(x=x, y=y, z=z)
            await asyncio.sleep(1.5)
            print(f"Done\n")

        # Return to center
        await self.furhat.request_attend_location(x=0.0, y=0.0, z=1.0)
        await asyncio.sleep(1)

    async def test_led_colors(self):
        """Test LED color changes"""
        print("\n=== Testing LED Colors ===\n")

        colors = [
            ("Red", "#FF0000"),
            ("Green", "#00FF00"),
            ("Blue", "#0000FF"),
            ("Yellow", "#FFFF00"),
            ("Purple", "#FF00FF"),
            ("Cyan", "#00FFFF"),
            ("White", "#FFFFFF"),
            ("Orange", "#FFA500"),
        ]

        for name, hex_code in colors:
            print(f"LED Color: {name} ({hex_code})")

            await self.furhat.request_speak_text(f"{name}", wait=True)
            await self.furhat.request_led_set(color=hex_code)
            await asyncio.sleep(1.5)
            print(f"Done\n")

    async def cleanup(self):
        """Reset robot to default state"""
        print("\n=== Cleanup ===")
        await self.furhat.request_face_reset()
        await self.furhat.request_led_set(color="#FFFFFF")  # White
        await self.furhat.request_attend_location(x=0.0, y=0.0, z=0.0)  # Center
        await self.furhat.disconnect()
        print("Test completed, disconnected\n")


async def main():
    tester = GestureTester()

    try:
        await tester.setup()

        # Select which tests to run (comment out if not needed)
        await tester.test_predefined_gestures()
        await tester.test_face_params()
        await tester.test_attention_system()
        await tester.test_led_colors()

    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()

    finally:
        await tester.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
