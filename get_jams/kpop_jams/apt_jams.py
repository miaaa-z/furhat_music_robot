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

    # 0-8s
    label_a.append(time=0, duration=8, value='calm')

    # 8-12s
    head_movement_a.append(time=8, duration=4, value='nod')
    label_a.append(time=8, duration=4, value='energetic')

    # 12-18s
    head_movement_a.append(time=12, duration=6, value='shake')
    facial_expression_a.append(time=12, duration=6, value='smile')
    label_a.append(time=12, duration=6, value='energetic')

    # 18-25s
    head_movement_a.append(time=18, duration=7, value='nod')
    facial_expression_a.append(time=18, duration=7, value='frown')
    label_a.append(time=18, duration=7, value='energetic')

    # 25-31s
    head_movement_a.append(time=25, duration=6, value='shake')
    facial_expression_a.append(time=25, duration=6, value='smile')
    label_a.append(time=25, duration=6, value='energetic')

    # 31-45s
    head_movement_a.append(time=31, duration=14, value='nod')
    facial_expression_a.append(time=31, duration=14, value='big smile')
    label_a.append(time=31, duration=14, value='very_happy')

    # 45-57s
    head_movement_a.append(time=45, duration=12, value='shake')
    intensity_a.append(time=45, duration=12, value='strong')
    label_a.append(time=45, duration=12, value='energetic')

    # 57-63s (57s to 1:03)
    facial_expression_a.append(time=57, duration=6, value='surprise face, frown')
    label_a.append(time=57, duration=6, value='surprised')

    # 1:03-1:10 (63-70s)
    head_movement_a.append(time=63, duration=7, value='sway')
    facial_expression_a.append(time=63, duration=7, value='big smile')
    intensity_a.append(time=63, duration=7, value='strong')
    label_a.append(time=63, duration=7, value='very_happy')

    # 1:10-1:23 (70-83s)
    head_movement_a.append(time=70, duration=13, value='shake')
    label_a.append(time=70, duration=13, value='energetic')

    # 1:23-1:35 (83-95s)
    head_movement_a.append(time=83, duration=12, value='nod')
    facial_expression_a.append(time=83, duration=12, value='big smile')
    intensity_a.append(time=83, duration=12, value='very strong')
    label_a.append(time=83, duration=12, value='very_happy')

    # 1:35-1:40 (95-100s)
    facial_expression_a.append(time=95, duration=5, value='smile, frown')
    label_a.append(time=95, duration=5, value='happy')

    # 1:40-2:09 (100-129s)
    head_movement_a.append(time=100, duration=29, value='nod')
    facial_expression_a.append(time=100, duration=29, value='big smile')
    intensity_a.append(time=100, duration=29, value='very strong')
    label_a.append(time=100, duration=29, value='very_happy')

    # 2:09-2:14 (129-134s)
    facial_expression_a.append(time=129, duration=5, value='frown, smile')
    label_a.append(time=129, duration=5, value='happy')

    # 2:14-2:49 (134-169s)
    head_movement_a.append(time=134, duration=35, value='nod')
    facial_expression_a.append(time=134, duration=35, value='big smile')
    intensity_a.append(time=134, duration=35, value='very strong')
    label_a.append(time=134, duration=35, value='very_happy')

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
    infile = '/Users/miaaa/Desktop/music robot/train/apt.wav'
    outfile = '/Users/miaaa/Desktop/music robot/jams/apt.jams'

    beat_track(infile, outfile)