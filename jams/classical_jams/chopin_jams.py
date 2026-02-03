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

    # 0-24s
    facial_expression_a.append(time=0, duration=24, value='close eyes, smile')
    intensity_a.append(time=0, duration=24, value='very gentle')
    label_a.append(time=0, duration=24, value='enjoy')

    # 24-36s
    head_movement_a.append(time=24, duration=12, value='sway')
    facial_expression_a.append(time=24, duration=12, value='smile')
    intensity_a.append(time=24, duration=12, value='very gentle')
    label_a.append(time=24, duration=12, value='enjoy')

    # 36-48s
    head_movement_a.append(time=36, duration=12, value='sway')
    facial_expression_a.append(time=36, duration=12, value='frown')
    intensity_a.append(time=36, duration=12, value='very gentle')
    label_a.append(time=36, duration=12, value='contemplative')

    # 48-70s (48s to 1:10)
    head_movement_a.append(time=48, duration=22, value='sway')
    facial_expression_a.append(time=48, duration=22, value='close eyes, smile')
    intensity_a.append(time=48, duration=22, value='very gentle')
    label_a.append(time=48, duration=22, value='calm')

    # 1:10-1:20 (70-80s)
    head_movement_a.append(time=70, duration=10, value='sway')
    facial_expression_a.append(time=70, duration=10, value='smile')
    label_a.append(time=70, duration=10, value='happy')

    # 1:20-1:25 (80-85s)
    head_movement_a.append(time=80, duration=5, value='sway')
    facial_expression_a.append(time=80, duration=5, value='close eyes, smile')
    intensity_a.append(time=80, duration=5, value='very gentle')
    label_a.append(time=80, duration=5, value='calm')

    # 1:25-1:37 (85-97s)
    facial_expression_a.append(time=85, duration=12, value='oh face')
    label_a.append(time=85, duration=12, value='surprised')

    # 1:37-2:07 (97-127s)
    facial_expression_a.append(time=97, duration=30, value='smile')
    label_a.append(time=97, duration=30, value='calm')

    # 2:07-2:33 (127-153s)
    head_movement_a.append(time=127, duration=26, value='sway')
    facial_expression_a.append(time=127, duration=26, value='close eyes, smile')
    intensity_a.append(time=127, duration=26, value='very gentle')
    label_a.append(time=127, duration=26, value='calm')

    # 2:33-3:07 (153-187s)
    head_movement_a.append(time=153, duration=34, value='sway')
    facial_expression_a.append(time=153, duration=34, value='thoughtful face')
    intensity_a.append(time=153, duration=34, value='very gentle')
    label_a.append(time=153, duration=34, value='thoughtful')

    # 3:07-3:23 (187-203s)
    facial_expression_a.append(time=187, duration=16, value='oh face')
    label_a.append(time=187, duration=16, value='surprised')

    # 3:23-3:35 (203-215s)
    facial_expression_a.append(time=203, duration=12, value='surprise face')
    label_a.append(time=203, duration=12, value='surprised')

    # 3:35-3:57 (215-237s)
    facial_expression_a.append(time=215, duration=22, value='smile')
    label_a.append(time=215, duration=22, value='happy')

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
    infile = '/Users/miaaa/Desktop/music robot/train/chopin.wav'
    outfile = '/Users/miaaa/Desktop/music robot/jams/chopin.jams'

    beat_track(infile, outfile)