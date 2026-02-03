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

    # 0-18s
    head_movement_a.append(time=0, duration=18, value='nod')
    label_a.append(time=0, duration=18, value='energetic')

    # 18-44s
    head_movement_a.append(time=18, duration=26, value='nod')
    facial_expression_a.append(time=18, duration=26, value='frown')
    label_a.append(time=18, duration=26, value='energetic')

    # 44-54s
    head_movement_a.append(time=44, duration=10, value='sway')
    facial_expression_a.append(time=44, duration=10, value='smile')
    intensity_a.append(time=44, duration=10, value='gentle')
    label_a.append(time=44, duration=10, value='happy')

    # 54-68s (54s to 1:08)
    facial_expression_a.append(time=54, duration=14, value='surprise face')
    label_a.append(time=54, duration=14, value='surprised')

    # 1:08-1:24 (68-84s)
    head_movement_a.append(time=68, duration=16, value='nod')
    facial_expression_a.append(time=68, duration=16, value='smile')
    intensity_a.append(time=68, duration=16, value='gentle')
    label_a.append(time=68, duration=16, value='happy')

    # 1:24-2:05 (84-125s)
    head_movement_a.append(time=84, duration=41, value='nod')
    label_a.append(time=84, duration=41, value='energetic')

    # 2:05-2:15 (125-135s)
    head_movement_a.append(time=125, duration=10, value='sway')
    facial_expression_a.append(time=125, duration=10, value='big smile')
    intensity_a.append(time=125, duration=10, value='gentle')
    label_a.append(time=125, duration=10, value='very_happy')

    # 2:14-2:34 (134-154s)
    head_movement_a.append(time=134, duration=20, value='sway')
    facial_expression_a.append(time=134, duration=20, value='smile')
    intensity_a.append(time=134, duration=20, value='gentle')
    label_a.append(time=134, duration=20, value='happy')

    # 2:34-3:18 (154-198s)
    head_movement_a.append(time=154, duration=44, value='nod')
    label_a.append(time=154, duration=44, value='energetic')

    # 3:18-3:48 (198-228s)
    head_movement_a.append(time=198, duration=30, value='nod')
    facial_expression_a.append(time=198, duration=30, value='smile')
    label_a.append(time=198, duration=30, value='happy')

    # 3:48-3:52 (228-232s)
    facial_expression_a.append(time=228, duration=4, value='oh face')
    label_a.append(time=228, duration=4, value='surprised')

    # 3:52-4:00 (232-240s)
    facial_expression_a.append(time=232, duration=8, value='thpoughtful face')
    label_a.append(time=232, duration=8, value='thoughtful')

    # 4:00-4:12 (240-252s)
    head_movement_a.append(time=240, duration=12, value='sway')
    facial_expression_a.append(time=240, duration=12, value='big smile')
    label_a.append(time=240, duration=12, value='very_happy')

    # 4:12-4:36 (252-276s)
    head_movement_a.append(time=252, duration=24, value='sway')
    label_a.append(time=252, duration=24, value='happy')

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
    infile = '/Users/miaaa/Desktop/music robot/train/empire_state.wav'
    outfile = '/Users/miaaa/Desktop/music robot/jams/empire_state_of_mind.jams'

    beat_track(infile, outfile)