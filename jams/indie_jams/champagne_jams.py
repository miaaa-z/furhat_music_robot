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

    # 0-23s
    head_movement_a.append(time=0, duration=23, value='nod')
    intensity_a.append(time=0, duration=23, value='very gentle')
    label_a.append(time=0, duration=23, value='calm')

    # 23-44s
    head_movement_a.append(time=23, duration=21, value='nod')
    facial_expression_a.append(time=23, duration=21, value='thoughtful face')
    intensity_a.append(time=23, duration=21, value='very gentle')
    label_a.append(time=23, duration=21, value='thoughtful')

    # 44-66s (44s to 1:06)
    head_movement_a.append(time=44, duration=22, value='sway')
    facial_expression_a.append(time=44, duration=22, value='frown')
    intensity_a.append(time=44, duration=22, value='very gentle')
    label_a.append(time=44, duration=22, value='contemplative')

    # 1:06-1:28 (66-88s)
    head_movement_a.append(time=66, duration=22, value='nod')
    intensity_a.append(time=66, duration=22, value='very gentle')
    label_a.append(time=66, duration=22, value='calm')

    # 1:28-1:33 (88-93s)
    facial_expression_a.append(time=88, duration=5, value='frown, smile')
    label_a.append(time=88, duration=5, value='contemplative')

    # 1:33-2:02 (93-122s)
    head_movement_a.append(time=93, duration=29, value='nod')
    intensity_a.append(time=93, duration=29, value='very gentle')
    label_a.append(time=93, duration=29, value='calm')

    # 2:02-2:24 (122-144s)
    head_movement_a.append(time=122, duration=22, value='nod')
    facial_expression_a.append(time=122, duration=22, value='frown')
    intensity_a.append(time=122, duration=22, value='very gentle')
    label_a.append(time=122, duration=22, value='contemplative')

    # 2:24-3:10 (144-190s)
    head_movement_a.append(time=144, duration=46, value='sway')
    facial_expression_a.append(time=144, duration=46, value='sad face')
    intensity_a.append(time=144, duration=46, value='very gentle')
    label_a.append(time=144, duration=46, value='contemplative')

    # 3:10-3:44 (190-224s)
    head_movement_a.append(time=190, duration=34, value='nod')
    facial_expression_a.append(time=190, duration=34, value='frown')
    intensity_a.append(time=190, duration=34, value='very gentle')
    label_a.append(time=190, duration=34, value='contemplative')

    # 3:44-4:04 (224-244s)
    facial_expression_a.append(time=224, duration=20, value='frown')
    label_a.append(time=224, duration=20, value='contemplative')

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
    infile = '/Users/miaaa/Desktop/music robot/train/champagne_problems.wav'
    outfile = '/Users/miaaa/Desktop/music robot/jams/champagne_problems.jams'

    beat_track(infile, outfile)