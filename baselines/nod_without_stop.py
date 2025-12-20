from furhat_realtime_api import AsyncFurhatClient
import asyncio


class CustomBehaviorTester:
    def __init__(self, ip_address="127.0.0.1"):
        self.furhat = AsyncFurhatClient(ip_address)

    async def setup(self):
        await self.furhat.connect()
        await self.furhat.request_voice_config(gender="female", language="en-US")

    async def head_nod_fast(self, times):
        for i in range(times):
            await self.furhat.request_gesture_start("Nod", intensity=1.0, duration=1.0)
            await asyncio.sleep(1.0)
            # duration equals to sleep time, it will not stop after one full nodding

async def main():
    tester = CustomBehaviorTester()
    await tester.setup()

    await tester.furhat.request_speak_text("nod", wait=True)
    await tester.head_nod_fast(times=4)

if __name__ == "__main__":
    asyncio.run(main())
