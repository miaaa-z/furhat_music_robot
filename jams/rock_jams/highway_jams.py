import librosa
import jams
import numpy as np


def beat_track(infile, outfile):
    # Load the audio file
    y, sr = librosa.load(infile)

    # Compute the track duration
    track_duration = librosa.get_duration(y=y, sr=sr)

    # Extract tempo and beat estimates
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

    # Fix tempo format
    if isinstance(tempo, np.ndarray):
        tempo = float(tempo[0])
    else:
        tempo = float(tempo)

    # Convert beat frames to time
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    # Construct a new JAMS object and annotation records
    jam = jams.JAMS()

    # Store the track duration
    jam.file_metadata.duration = track_duration

    beat_a = jams.Annotation(namespace='beat')
    beat_a.annotation_metadata = jams.AnnotationMetadata(
        data_source='librosa beat tracker',
        annotator={'name': 'Mia', 'email': ''},
        version='1.0'
    )

    # Add beat timings to the annotation record
    for t in beat_times:
        beat_a.append(time=t, duration=0.0)

    # Store the new annotation in the jam
    jam.annotations.append(beat_a)

    # Add tempo estimation to the annotation
    tempo_a = jams.Annotation(namespace='tempo', time=0, duration=track_duration)
    tempo_a.annotation_metadata = jams.AnnotationMetadata(
        data_source='librosa tempo estimator',
        annotator={'name': 'Mia', 'email': ''},
        version='1.0'
    )

    tempo_a.append(time=0.0,
                   duration=track_duration,
                   value=tempo,
                   confidence=1.0)

    # Store the new annotation in the jam
    jam.annotations.append(tempo_a)

    # People's movement annotations
    # Head movement annotation (use tag_open namespace)
    head_movement_a = jams.Annotation(namespace='tag_open')
    head_movement_a.annotation_metadata = jams.AnnotationMetadata(
        data_source='head_movement',
        annotator={'name': 'Mia', 'email': ''},
        version='1.0'
    )

    # Facial expression annotation (use tag_open namespace)
    facial_expression_a = jams.Annotation(namespace='tag_open')
    facial_expression_a.annotation_metadata = jams.AnnotationMetadata(
        data_source='facial_expression',
        annotator={'name': 'Mia', 'email': ''},
        version='1.0'
    )

    # Intensity annotation (use tag_open namespace)
    intensity_a = jams.Annotation(namespace='tag_open')
    intensity_a.annotation_metadata = jams.AnnotationMetadata(
        data_source='intensity',
        annotator={'name': 'Mia', 'email': ''},
        version='1.0'
    )

    # **NEW: Label annotation (use tag_open namespace)**
    label_a = jams.Annotation(namespace='tag_open')
    label_a.annotation_metadata = jams.AnnotationMetadata(
        data_source='emotion_label',
        annotator={'name': 'Mia', 'email': ''},
        version='1.0'
    )

    # 0-10s
    facial_expression_a.append(time=0, duration=10, value='smile')
    label_a.append(time=0, duration=10, value='happy')

    # 10-20s
    head_movement_a.append(time=10, duration=10, value='nod')
    facial_expression_a.append(time=10, duration=10, value='smile')
    label_a.append(time=10, duration=10, value='happy')

    # 20-53s
    head_movement_a.append(time=20, duration=33, value='sway')
    facial_expression_a.append(time=20, duration=33, value='frown, smile')
    label_a.append(time=20, duration=33, value='energetic')

    # 53-60s (53s to 1:00)
    head_movement_a.append(time=53, duration=7, value='nod')
    facial_expression_a.append(time=53, duration=7, value='big smile')
    intensity_a.append(time=53, duration=7, value='strong')
    label_a.append(time=53, duration=7, value='very_happy')

    # 1:00-1:08 (60-68s)
    head_movement_a.append(time=60, duration=8, value='sway')
    facial_expression_a.append(time=60, duration=8, value='smile')
    label_a.append(time=60, duration=8, value='happy')

    # 1:11-1:47 (71-107s)
    head_movement_a.append(time=71, duration=36, value='sway, nod')
    facial_expression_a.append(time=71, duration=36, value='frown, smile')
    label_a.append(time=71, duration=36, value='energetic')

    # 1:47-2:05 (107-125s)
    head_movement_a.append(time=107, duration=18, value='shake')
    facial_expression_a.append(time=107, duration=18, value='frown, smile')
    intensity_a.append(time=107, duration=18, value='strong')
    label_a.append(time=107, duration=18, value='very_energetic')

    # 2:05-2:07 (125-127s)
    facial_expression_a.append(time=125, duration=2, value='oh face')
    label_a.append(time=125, duration=2, value='surprised')

    # 2:07-2:44 (127-164s)
    head_movement_a.append(time=127, duration=37, value='shake')
    facial_expression_a.append(time=127, duration=37, value='frown')
    label_a.append(time=127, duration=37, value='energetic')

    # 2:44-2:46 (164-166s)
    facial_expression_a.append(time=164, duration=2, value='surprise face')
    label_a.append(time=164, duration=2, value='surprised')

    # 2:46-3:06 (166-186s)
    head_movement_a.append(time=166, duration=20, value='shake')
    facial_expression_a.append(time=166, duration=20, value='close eyes')
    label_a.append(time=166, duration=20, value='energetic')

    # 3:06-3:22 (186-202s)
    head_movement_a.append(time=186, duration=16, value='nod')
    facial_expression_a.append(time=186, duration=16, value='frown')
    label_a.append(time=186, duration=16, value='energetic')

    # 3:22-3:28 (202-208s)
    facial_expression_a.append(time=202, duration=6, value='frown')
    label_a.append(time=202, duration=6, value='contemplative')

    # Append all annotations
    jam.annotations.append(head_movement_a)
    jam.annotations.append(facial_expression_a)
    jam.annotations.append(intensity_a)
    jam.annotations.append(label_a)

    # Save to disk
    jam.save(outfile)

    print("Finish")
    print(f"File saved: {outfile}")


if __name__ == '__main__':
    infile = '/Users/miaaa/Desktop/music robot/train/highway_to_hell.wav'
    outfile = '/Users/miaaa/Desktop/music robot/jams/highway_to_hell.jams'

    beat_track(infile, outfile)