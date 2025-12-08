# Test Furhat playing songs
from furhat_realtime_api import FurhatClient

furhat = FurhatClient("127.0.0.1")
furhat.connect()

# play the music
# furhat only accepts url format
# I downloaded the flac format then transformed it to wav then uploaded to Google drive and share with everyone
furhat.request_speak_audio(
    "https://drive.google.com/uc?export=download&id=1y7o8Qaz46kL8Mn9fEFSNC4fJCN_IQoCY",
    wait=True,
    abort=False,
    lipsync=False 
)

furhat.disconnect()
