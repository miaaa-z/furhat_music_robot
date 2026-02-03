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

    # 0-3s
    facial_expression_a.append(time=0, duration=3, value='smile')
    label_a.append(time=0, duration=3, value='calm')

    # 3-30s
    head_movement_a.append(time=3, duration=27, value='sway')
    facial_expression_a.append(time=3, duration=27, value='smile')
    intensity_a.append(time=3, duration=27, value='gentle')
    label_a.append(time=3, duration=27, value='happy')

    # 30-54s
    head_movement_a.append(time=30, duration=24, value='sway')
    facial_expression_a.append(time=30, duration=24, value='smile')
    label_a.append(time=30, duration=24, value='happy')

    # 54-67s (54s to 1:07)
    head_movement_a.append(time=54, duration=13, value='nod')
    label_a.append(time=54, duration=13, value='happy')

    # 1:07-1:42 (67-102s)
    head_movement_a.append(time=67, duration=35, value='sway')
    label_a.append(time=67, duration=35, value='happy')

    # 1:42-2:28 (102-148s)
    head_movement_a.append(time=102, duration=46, value='sway')
    intensity_a.append(time=102, duration=46, value='gentle')
    label_a.append(time=102, duration=46, value='calm')

    # 2:28-3:06 (148-186s)
    head_movement_a.append(time=148, duration=38, value='sway')
    facial_expression_a.append(time=148, duration=38, value='close eyes')
    intensity_a.append(time=148, duration=38, value='very gentle')
    label_a.append(time=148, duration=38, value='calm')

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
    infile = '/Users/miaaa/Desktop/music robot/train/take_me_home_country_roads.wav'
    outfile = '/Users/miaaa/Desktop/music robot/jams/take_me_home_country_roads.jams'

    beat_track(infile, outfile)