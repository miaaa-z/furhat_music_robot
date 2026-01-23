import essentia.standard as es
import numpy as np


def test_models():
    audio_path = "/Users/miaaa/Desktop/music robot/test/cruel_summer.wav"
    model_dir = "/Users/miaaa/Desktop/music robot/furhat_music_robot/essentia_models/"

    # Load audio at 16kHz sample rate
    audio = es.MonoLoader(filename=audio_path, sampleRate=16000, resampleQuality=4)()
    print(f"Length: {len(audio) / 16000:.1f}s")

    # Extract EffNet embeddings (shared across all models)
    print("\nExtracting EffNet embeddings...")
    try:
        embedding_model = es.TensorflowPredictEffnetDiscogs(
            graphFilename=model_dir + "discogs-effnet-bs64-1.pb",
            output="PartitionedCall:1"
        )
        embeddings = embedding_model(audio)
        print(f"Embeddings shape: {embeddings.shape}")
    except Exception as e:
        print(f"Embedding extraction failed: {e}")
        print("Please ensure discogs-effnet-bs64-1.pb is downloaded")
        return

    # Test Danceability model
    print("\nTesting Danceability model...")
    try:
        model = es.TensorflowPredict2D(
            graphFilename=model_dir + "danceability-discogs-effnet-1.pb",
            output="model/Softmax"
        )
        predictions = model(embeddings)
        avg_predictions = np.mean(predictions, axis=0)

        print(f"Not Danceable: {avg_predictions[0]:.3f}, Danceable: {avg_predictions[1]:.3f}")
        print(f"Result: {'High danceability!' if avg_predictions[1] > 0.5 else 'Low danceability'}")
    except Exception as e:
        print(f"Error: {e}")

    # Test Voice/Instrumental model
    print("\nTesting Voice/Instrumental model...")
    try:
        model = es.TensorflowPredict2D(
            graphFilename=model_dir + "voice_instrumental-discogs-effnet-1.pb",
            output="model/Softmax"
        )
        predictions = model(embeddings)
        avg_predictions = np.mean(predictions, axis=0)

        print(f"Instrumental: {avg_predictions[0]:.3f}, Voice: {avg_predictions[1]:.3f}")
        print(f"Result: {'Has vocals' if avg_predictions[1] > 0.5 else 'Instrumental only'}")
    except Exception as e:
        print(f"Error: {e}")

    # Test Acoustic model
    print("\nTesting Acoustic model...")
    try:
        model = es.TensorflowPredict2D(
            graphFilename=model_dir + "mood_acoustic-discogs-effnet-1.pb",
            output="model/Softmax"
        )
        predictions = model(embeddings)
        avg_predictions = np.mean(predictions, axis=0)

        print(f"Not Acoustic: {avg_predictions[0]:.3f}, Acoustic: {avg_predictions[1]:.3f}")
        print(f"Result: {'Acoustic music' if avg_predictions[1] > 0.5 else 'Electronic music'}")
    except Exception as e:
        print(f"Error: {e}")

    # Test Electronic model
    print("\nTesting Electronic model...")
    try:
        model = es.TensorflowPredict2D(
            graphFilename=model_dir + "mood_electronic-discogs-effnet-1.pb",
            output="model/Softmax"
        )
        predictions = model(embeddings)
        avg_predictions = np.mean(predictions, axis=0)

        print(f"Not Electronic: {avg_predictions[0]:.3f}, Electronic: {avg_predictions[1]:.3f}")
        print(f"Result: {'Electronic music' if avg_predictions[1] > 0.5 else 'Non-electronic music'}")
    except Exception as e:
        print(f"Model not found or error: {e}")

    print("\n" + "=" * 60)
    print("Testing complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_models()