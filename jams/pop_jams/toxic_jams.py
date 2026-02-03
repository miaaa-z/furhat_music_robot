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

    # 0-6s
    facial_expression_a.append(time=0, duration=6, value='smile')
    label_a.append(time=0, duration=6, value='happy')

    # 6-13s
    facial_expression_a.append(time=6, duration=7, value='frown')
    label_a.append(time=6, duration=7, value='contemplative')

    # 13-17s
    facial_expression_a.append(time=13, duration=4, value='big smile')
    label_a.append(time=13, duration=4, value='very_happy')

    # 17-36s
    head_movement_a.append(time=17, duration=19, value='sway')
    label_a.append(time=17, duration=19, value='happy')

    # 36-40s
    facial_expression_a.append(time=36, duration=4, value='surprise face')
    label_a.append(time=36, duration=4, value='surprised')

    # 40-54s
    head_movement_a.append(time=40, duration=14, value='sway')
    facial_expression_a.append(time=40, duration=14, value='smile, frown')
    label_a.append(time=40, duration=14, value='happy')

    # 54-59s
    head_movement_a.append(time=54, duration=5, value='shake')
    facial_expression_a.append(time=54, duration=5, value='big smile')
    label_a.append(time=54, duration=5, value='very_happy')

    # 59-72s (59s to 1:12)
    head_movement_a.append(time=59, duration=13, value='nod')
    facial_expression_a.append(time=59, duration=13, value='smile')
    label_a.append(time=59, duration=13, value='happy')

    # 1:12-1:18 (72-78s)
    facial_expression_a.append(time=72, duration=6, value='oh face')
    label_a.append(time=72, duration=6, value='surprised')

    # 1:18-1:38 (78-98s)
    head_movement_a.append(time=78, duration=20, value='nod')
    facial_expression_a.append(time=78, duration=20, value='smile')
    label_a.append(time=78, duration=20, value='happy')

    # 1:38-1:52 (98-112s)
    head_movement_a.append(time=98, duration=14, value='shake')
    label_a.append(time=98, duration=14, value='energetic')

    # 1:52-1:56 (112-116s)
    facial_expression_a.append(time=112, duration=4, value='oh face')
    label_a.append(time=112, duration=4, value='surprised')

    # 1:56-2:08 (116-128s)
    head_movement_a.append(time=116, duration=12, value='shake')
    facial_expression_a.append(time=116, duration=12, value='big smile')
    intensity_a.append(time=116, duration=12, value='gentle')
    label_a.append(time=116, duration=12, value='very_happy')

    # 2:08-2:21 (128-141s)
    head_movement_a.append(time=128, duration=13, value='nod')
    facial_expression_a.append(time=128, duration=13, value='frown')
    intensity_a.append(time=128, duration=13, value='gentle')
    label_a.append(time=128, duration=13, value='happy')

    # 2:21-2:40 (141-160s)
    facial_expression_a.append(time=141, duration=19, value='surprise face')
    label_a.append(time=141, duration=19, value='surprised')

    # 2:41-2:53 (161-173s)
    head_movement_a.append(time=161, duration=12, value='shake')
    facial_expression_a.append(time=161, duration=12, value='smile')
    label_a.append(time=161, duration=12, value='energetic')

    # 2:53-3:05 (173-185s)
    head_movement_a.append(time=173, duration=12, value='shake')
    facial_expression_a.append(time=173, duration=12, value='close eyes')
    label_a.append(time=173, duration=12, value='energetic')

    # 3:05-3:10 (185-190s)
    facial_expression_a.append(time=185, duration=5, value='oh face')
    label_a.append(time=185, duration=5, value='surprised')

    # 3:10-3:20 (190-200s)
    head_movement_a.append(time=190, duration=10, value='nod')
    intensity_a.append(time=190, duration=10, value='gentle')
    label_a.append(time=190, duration=10, value='calm')

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
    infile = '/Users/miaaa/Desktop/music robot/train/toxic.wav'
    outfile = '/Users/miaaa/Desktop/music robot/jams/toxic.jams'

    beat_track(infile, outfile)