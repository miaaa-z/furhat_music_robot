from furhat_realtime_api import AsyncFurhatClient
import asyncio

class BuiltinGestureTester:
    """Test all Furhat builtin gestures"""

    def __init__(self, ip_address="127.0.0.1"):
        self.furhat = AsyncFurhatClient(ip_address)

        # Builtin gestures categorized
        self.builtin_gestures = {
            "basic_expressions": [
                "BigSmile",  # Raised brows + big smile
                "Smile",  # Raised brows + smile (less intense than BigSmile)
                "Wink",  # Left eye wink + head tilts slightly right
                "Surprise",  # Raised brows + mouth wide open
                "Oh",  # Raised brows + mouth in "oh" shape
                "Thoughtful",  # Frown brows + pout lips
            ],

            "emotion_expressions": [
                "ExpressAnger",  # Frown brows + show teeth (intense frown)
                "ExpressDisgust",  # Frown brows + mouth corner down/asymmetric (moderate frown)
                "ExpressFear",  # Brows shape like 八 + mouth slightly open
                "ExpressSad",  # Brows shape like 八 + mouth corners down/symmetric
            ],

            "head_movements": [
                "Nod",  # Nod up-down then return to center
                "Shake",  # Shake left-right then return to center
                "Roll",  # Sway left-right then return to center
            ],

            "brow_movements": [
                "BrowRaise",  # Raise both eyebrows
                "BrowFrown",  # Frown brows + slightly narrow eyes
            ],

            "eye_movements": [
                "GazeAway",  # Look down-left then return to center
                "Blink",  # Both eyes blink
                "CloseEyes",
                "OpenEyes",
            ],
        }

    async def setup(self):
        """Connect and configure Furhat"""
        await self.furhat.connect()
        await self.furhat.request_voice_config(gender="female", language="en-US")

    async def test_category(self, category_name, gestures, intensity=1.0, duration=2.0):
        """Test a category of gestures"""
        for gesture in gestures:
            await self.furhat.request_gesture_start(
                gesture,
                intensity=intensity,
                duration=duration,
                wait=True
            )
            await asyncio.sleep(0.5)

    async def test_single_gesture(self, gesture_name, intensity=1.0, duration=2.0):
        """Test a single gesture"""
        await self.furhat.request_gesture_start(
            gesture_name,
            intensity=intensity,
            duration=duration,
            wait=True
        )
        await asyncio.sleep(0.5)

    async def test_all_categories(self):
        """Test all categories of builtin gestures"""
        for category_name, gestures in self.builtin_gestures.items():
            await self.test_category(category_name, gestures)
            await asyncio.sleep(1)

    async def test_led_colors(self):
        """Test LED color changes"""
        colors = [
            "#FF0000",  # Red
            "#00FF00",  # Green
            "#0000FF",  # Blue
            "#FFFF00",  # Yellow
            "#FF00FF",  # Purple
            "#00FFFF",  # Cyan
            "#FFFFFF",  # White
            "#FFA500",  # Orange
        ]

        for hex_code in colors:
            await self.furhat.request_led_set(color=hex_code)
            await asyncio.sleep(1.5)

    async def cleanup(self):
        """Cleanup and disconnect"""
        await self.furhat.request_face_reset()
        await self.furhat.disconnect()

async def main():
    tester = BuiltinGestureTester()
    await tester.setup()

    # Test all gestures
    await tester.test_all_categories()

    # Test one category
    # await tester.test_category("emotion_expressions", tester.builtin_gestures["emotion_expressions"])

    # Test single gesture
    # await tester.test_single_gesture("ExpressAnger", intensity=0.9, duration=2.0)

    # Test LED colors
    await tester.test_led_colors()

    await tester.cleanup()

if __name__ == "__main__":
    asyncio.run(main())