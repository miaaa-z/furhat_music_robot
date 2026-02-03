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
    facial_expression_a.append(time=0, duration=7, value='smile')
    label_a.append(time=0, duration=7, value='calm')

    # 7-22s
    head_movement_a.append(time=7, duration=15, value='sway')
    facial_expression_a.append(time=7, duration=15, value='smile')
    intensity_a.append(time=7, duration=15, value='gentle')
    label_a.append(time=7, duration=15, value='happy')

    # 22-29s
    head_movement_a.append(time=22, duration=7, value='nod')
    facial_expression_a.append(time=22, duration=7, value='smile')
    intensity_a.append(time=22, duration=7, value='gentle')
    label_a.append(time=22, duration=7, value='happy')

    # 29-38s
    head_movement_a.append(time=29, duration=9, value='shake')
    facial_expression_a.append(time=29, duration=9, value='smile')
    intensity_a.append(time=29, duration=9, value='gentle')
    label_a.append(time=29, duration=9, value='happy')

    # 38-54s
    head_movement_a.append(time=38, duration=16, value='nod')
    intensity_a.append(time=38, duration=16, value='gentle')
    label_a.append(time=38, duration=16, value='calm')

    # 54-62s (54s to 1:02)
    head_movement_a.append(time=54, duration=8, value='shake')
    facial_expression_a.append(time=54, duration=8, value='frown')
    intensity_a.append(time=54, duration=8, value='gentle')
    label_a.append(time=54, duration=8, value='calm')

    # 1:02-1:11 (62-71s)
    head_movement_a.append(time=62, duration=9, value='nod')
    label_a.append(time=62, duration=9, value='happy')

    # 1:11-1:38 (71-98s)
    head_movement_a.append(time=71, duration=27, value='shake')
    facial_expression_a.append(time=71, duration=27, value='close eyes')
    label_a.append(time=71, duration=27, value='energetic')

    # 1:38-1:44 (98-104s)
    facial_expression_a.append(time=98, duration=6, value='frown')
    label_a.append(time=98, duration=6, value='contemplative')

    # 1:44-2:04 (104-124s)
    head_movement_a.append(time=104, duration=20, value='sway')
    facial_expression_a.append(time=104, duration=20, value='smile')
    intensity_a.append(time=104, duration=20, value='gentle')
    label_a.append(time=104, duration=20, value='happy')

    # 2:04-2:08 (124-128s)
    head_movement_a.append(time=124, duration=4, value='shake')
    facial_expression_a.append(time=124, duration=4, value='frown')
    label_a.append(time=124, duration=4, value='energetic')

    # 2:08-2:21 (128-141s)
    head_movement_a.append(time=128, duration=13, value='sway')
    facial_expression_a.append(time=128, duration=13, value='smile')
    label_a.append(time=128, duration=13, value='happy')

    # 2:21-2:27 (141-147s)
    facial_expression_a.append(time=141, duration=6, value='smile')
    label_a.append(time=141, duration=6, value='happy')

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
    infile = '/Users/miaaa/Desktop/music robot/train/fly_me_to_the_moon.wav'
    outfile = '/Users/miaaa/Desktop/music robot/jams/fly_me_to_the_moon.jams'

    beat_track(infile, outfile)