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
    label_a.append(time=0, duration=10, value='calm')

    # 10-16s
    facial_expression_a.append(time=10, duration=6, value='surprise face')
    label_a.append(time=10, duration=6, value='surprised')

    # 16-30s
    head_movement_a.append(time=16, duration=14, value='sway')
    intensity_a.append(time=16, duration=14, value='very gentle')
    label_a.append(time=16, duration=14, value='calm')

    # 30-43s
    head_movement_a.append(time=30, duration=13, value='nod')
    facial_expression_a.append(time=30, duration=13, value='frown, close eyes')
    label_a.append(time=30, duration=13, value='contemplative')

    # 43-54s
    head_movement_a.append(time=43, duration=11, value='shake')
    facial_expression_a.append(time=43, duration=11, value='frown')
    intensity_a.append(time=43, duration=11, value='gentle')
    label_a.append(time=43, duration=11, value='contemplative')

    # 54-76s (54s to 1:16)
    head_movement_a.append(time=54, duration=22, value='nod')
    facial_expression_a.append(time=54, duration=22, value='smile')
    intensity_a.append(time=54, duration=22, value='strog')
    label_a.append(time=54, duration=22, value='happy')

    # 1:16-1:36 (76-96s)
    head_movement_a.append(time=76, duration=20, value='shake')
    facial_expression_a.append(time=76, duration=20, value='close eyes')
    label_a.append(time=76, duration=20, value='energetic')

    # 1:36-1:47 (96-107s)
    facial_expression_a.append(time=96, duration=11, value='thoughtful face')
    label_a.append(time=96, duration=11, value='thoughtful')

    # 1:47-1:58 (107-118s)
    head_movement_a.append(time=107, duration=11, value='sway')
    facial_expression_a.append(time=107, duration=11, value='close eyes')
    label_a.append(time=107, duration=11, value='calm')

    # 1:58-2:05 (118-125s)
    facial_expression_a.append(time=118, duration=7, value='surprise face')
    label_a.append(time=118, duration=7, value='surprised')

    # 2:05-2:19 (125-139s)
    head_movement_a.append(time=125, duration=14, value='sway')
    intensity_a.append(time=125, duration=14, value='gentle')
    label_a.append(time=125, duration=14, value='calm')

    # 2:19-2:30 (139-150s)
    head_movement_a.append(time=139, duration=11, value='nod, sway')
    facial_expression_a.append(time=139, duration=11, value='frown')
    label_a.append(time=139, duration=11, value='contemplative')

    # 2:30-3:14 (150-194s)
    head_movement_a.append(time=150, duration=44, value='nod, sway')
    facial_expression_a.append(time=150, duration=44, value='frown, smile')
    intensity_a.append(time=150, duration=44, value='very strong')
    label_a.append(time=150, duration=44, value='very energetic')

    # 3:14-3:32 (194-212s)
    head_movement_a.append(time=194, duration=18, value='sway')
    intensity_a.append(time=194, duration=18, value='very gentle')
    label_a.append(time=194, duration=18, value='calm')

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
    infile = '/Users/miaaa/Desktop/music robot/train/faded.wav'
    outfile = '/Users/miaaa/Desktop/music robot/jams/faded.jams'

    beat_track(infile, outfile)