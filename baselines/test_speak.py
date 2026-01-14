from furhat_realtime_api import AsyncFurhatClient
import asyncio
import logging

furhat = AsyncFurhatClient("127.0.0.1")

async def run_example():
    await furhat.connect()
    await furhat.request_voice_config(gender="female", language="en-US")
    await furhat.request_speak_text("wow.", wait=True)
    await furhat.disconnect()

asyncio.run(run_example())