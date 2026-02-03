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
    facial_expression_a.append(time=0, duration=8, value='smile')
    label_a.append(time=0, duration=8, value='happy')

    # 8-18s
    head_movement_a.append(time=8, duration=10, value='nod')
    facial_expression_a.append(time=8, duration=10, value='smile')
    intensity_a.append(time=8, duration=10, value='gentle')
    label_a.append(time=8, duration=10, value='happy')

    # 18-28s
    head_movement_a.append(time=18, duration=10, value='sway')
    facial_expression_a.append(time=18, duration=10, value='close eyes')
    intensity_a.append(time=18, duration=10, value='gentle')
    label_a.append(time=18, duration=10, value='happy')

    # 28-36s
    head_movement_a.append(time=28, duration=8, value='shake')
    label_a.append(time=28, duration=8, value='energetic')

    # 36-55s
    head_movement_a.append(time=36, duration=19, value='nod')
    intensity_a.append(time=36, duration=19, value='strong')
    label_a.append(time=36, duration=19, value='energetic')

    # 55-79s (55s to 1:19)
    head_movement_a.append(time=55, duration=24, value='sway')
    facial_expression_a.append(time=55, duration=24, value='smile')
    intensity_a.append(time=55, duration=24, value='gentle')
    label_a.append(time=55, duration=24, value='happy')

    # 1:19-1:48 (79-108s)
    head_movement_a.append(time=79, duration=29, value='nod')
    facial_expression_a.append(time=79, duration=29, value='frown, smile')
    label_a.append(time=79, duration=29, value='happy')

    # 1:48-2:05 (108-125s)
    head_movement_a.append(time=108, duration=17, value='sway')
    label_a.append(time=108, duration=17, value='happy')

    # 2:05-2:10 (125-130s)
    facial_expression_a.append(time=125, duration=5, value='surprise face')
    label_a.append(time=125, duration=5, value='surprised')

    # 2:10-2:23 (130-143s)
    head_movement_a.append(time=130, duration=13, value='shake')
    facial_expression_a.append(time=130, duration=13, value='big smile')
    intensity_a.append(time=130, duration=13, value='strong')
    label_a.append(time=130, duration=13, value='very_happy')

    # 2:23-2:32 (143-152s)
    head_movement_a.append(time=143, duration=9, value='nod')
    intensity_a.append(time=143, duration=9, value='gentle')
    label_a.append(time=143, duration=9, value='calm')

    # 2:32-2:42 (152-162s)
    facial_expression_a.append(time=152, duration=10, value='surprise face')
    label_a.append(time=152, duration=10, value='surprised')

    # 2:42-3:00 (162-180s)
    head_movement_a.append(time=162, duration=18, value='nod')
    intensity_a.append(time=162, duration=18, value='strong')
    label_a.append(time=162, duration=18, value='energetic')

    # 3:00-3:23 (180-203s)
    head_movement_a.append(time=180, duration=23, value='sway')
    facial_expression_a.append(time=180, duration=23, value='close eyes')
    label_a.append(time=180, duration=23, value='happy')

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
    infile = '/Users/miaaa/Desktop/music robot/train/levitating.wav'
    outfile = '/Users/miaaa/Desktop/music robot/jams/levitating.jams'

    beat_track(infile, outfile)