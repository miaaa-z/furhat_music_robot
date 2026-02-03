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

    # 0-7s
    facial_expression_a.append(time=0, duration=7, value='frown')
    label_a.append(time=0, duration=7, value='contemplative')

    # 7-11s
    facial_expression_a.append(time=7, duration=4, value='smile')
    label_a.append(time=7, duration=4, value='happy')

    # 11-41s
    head_movement_a.append(time=11, duration=30, value='nod')
    facial_expression_a.append(time=11, duration=30, value='smile')
    intensity_a.append(time=11, duration=30, value='gentle')
    label_a.append(time=11, duration=30, value='happy')

    # 41-52s
    head_movement_a.append(time=41, duration=11, value='nod')
    label_a.append(time=41, duration=11, value='energetic')

    # 52-62s (52s to 1:02)
    head_movement_a.append(time=52, duration=10, value='sway')
    facial_expression_a.append(time=52, duration=10, value='smile')
    intensity_a.append(time=52, duration=10, value='strong')
    label_a.append(time=52, duration=10, value='happy')

    # 1:02-1:44 (62-104s)
    head_movement_a.append(time=62, duration=42, value='shake')
    intensity_a.append(time=62, duration=42, value='strong')
    label_a.append(time=62, duration=42, value='energetic')

    # 1:44-2:07 (104-127s)
    head_movement_a.append(time=104, duration=23, value='nod')
    facial_expression_a.append(time=104, duration=23, value='smile')
    intensity_a.append(time=104, duration=23, value='strong')
    label_a.append(time=104, duration=23, value='happy')

    # 2:07-2:30 (127-150s)
    head_movement_a.append(time=127, duration=23, value='sway')
    label_a.append(time=127, duration=23, value='happy')

    # 2:30-2:50 (150-170s)
    head_movement_a.append(time=150, duration=20, value='nod')
    label_a.append(time=150, duration=20, value='energetic')

    # 2:50-3:12 (170-192s)
    head_movement_a.append(time=170, duration=22, value='shake')
    intensity_a.append(time=170, duration=22, value='strong')
    label_a.append(time=170, duration=22, value='energetic')

    # 3:12-3:24 (192-204s)
    facial_expression_a.append(time=192, duration=12, value='thoughtful face')
    label_a.append(time=192, duration=12, value='thoughtful')

    # 3:24-3:36 (204-216s)
    head_movement_a.append(time=204, duration=12, value='nod')
    facial_expression_a.append(time=204, duration=12, value='frown, smile')
    intensity_a.append(time=204, duration=12, value='very strong')
    label_a.append(time=204, duration=12, value='very_energetic')

    # 3:36-3:47 (216-227s)
    head_movement_a.append(time=216, duration=11, value='sway')
    label_a.append(time=216, duration=11, value='happy')

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
    infile = '/Users/miaaa/Desktop/music robot/train/despacito.wav'
    outfile = '/Users/miaaa/Desktop/music robot/jams/despacito.jams'

    beat_track(infile, outfile)