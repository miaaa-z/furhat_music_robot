import essentia.standard as es
import numpy as np


def test_new_models():
    audio_path = "/Users/miaaa/Desktop/music robot/test/zoo.wav"
    model_dir = "/Users/miaaa/Desktop/music robot/furhat_music_robot/essentia_models/"

    # Load audio at 16kHz
    audio = es.MonoLoader(filename=audio_path, sampleRate=16000, resampleQuality=4)()
    print(f" Length: {len(audio) / 16000:.1f}s")

    # VGGish-based models
    print("\nTesting VGGish-based models...")

    # Extract VGGish embeddings
    print("\nExtracting VGGish embeddings...")
    try:
        vggish_model = es.TensorflowPredictVGGish(
            graphFilename=model_dir + "audioset-vggish-3.pb",
            output="model/vggish/embeddings"
        )
        vggish_embeddings = vggish_model(audio)
        print(f"VGGish embeddings shape: {vggish_embeddings.shape}")
    except Exception as e:
        print(f"VGGish extraction failed: {e}")
        return

    # Test Danceability (VGGish version)
    print("\nTesting Danceability (VGGish)...")
    try:
        model = es.TensorflowPredict2D(
            graphFilename=model_dir + "danceability-audioset-vggish-1.pb",
            output="model/Softmax"
        )
        predictions = model(vggish_embeddings)
        avg_predictions = np.mean(predictions, axis=0)

        print(f"Not Danceable: {avg_predictions[0]:.3f}, Danceable: {avg_predictions[1]:.3f}")
    except Exception as e:
        print(f"Error: {e}")

    # Test Mood Happy
    print("\nTesting Mood Happy...")
    try:
        model = es.TensorflowPredict2D(
            graphFilename=model_dir + "mood_happy-audioset-vggish-1.pb",
            output="model/Softmax"
        )
        predictions = model(vggish_embeddings)
        avg_predictions = np.mean(predictions, axis=0)

        print(f"Not Happy: {avg_predictions[0]:.3f}, Happy: {avg_predictions[1]:.3f}")
    except Exception as e:
        print(f"Error: {e}")

    # Extract EffNet embeddings
    print("\nExtracting EffNet embeddings...")
    try:
        effnet_model = es.TensorflowPredictEffnetDiscogs(
            graphFilename=model_dir + "discogs-effnet-bs64-1.pb",
            output="PartitionedCall:1"
        )
        effnet_embeddings = effnet_model(audio)
        print(f"EffNet embeddings shape: {effnet_embeddings.shape}")
    except Exception as e:
        print(f"EffNet extraction failed: {e}")
        return


if __name__ == "__main__":
    test_new_models()
