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

    # 0-5s
    head_movement_a.append(time=0, duration=5, value='nod')
    intensity_a.append(time=0, duration=5, value='strong')
    label_a.append(time=0, duration=5, value='very_energetic')

    # 5-9s
    facial_expression_a.append(time=5, duration=4, value='thoughtful face')
    label_a.append(time=5, duration=4, value='thoughtful')

    # 9-17s
    facial_expression_a.append(time=9, duration=8, value='smile')
    label_a.append(time=9, duration=8, value='happy')

    # 17-28s
    head_movement_a.append(time=17, duration=11, value='nod')
    facial_expression_a.append(time=17, duration=11, value='big smile')
    label_a.append(time=17, duration=11, value='very_happy')

    # 28-34s
    facial_expression_a.append(time=28, duration=6, value='surprise face')
    label_a.append(time=28, duration=6, value='surprised')

    # 34-40s
    facial_expression_a.append(time=34, duration=6, value='big smile')
    label_a.append(time=34, duration=6, value='very_happy')

    # 40-64s (40s to 1:04)
    head_movement_a.append(time=40, duration=24, value='sway')
    facial_expression_a.append(time=40, duration=24, value='smile')
    intensity_a.append(time=40, duration=24, value='strong')
    label_a.append(time=40, duration=24, value='very_happy')

    # 1:04-1:28 (64-88s)
    head_movement_a.append(time=64, duration=24, value='shake')
    facial_expression_a.append(time=64, duration=24, value='smile')
    label_a.append(time=64, duration=24, value='energetic')

    # 1:28-1:42 (88-102s)
    facial_expression_a.append(time=88, duration=14, value='close eyes, smile')
    label_a.append(time=88, duration=14, value='happy')

    # 1:42-2:06 (102-126s)
    head_movement_a.append(time=102, duration=24, value='nod')
    facial_expression_a.append(time=102, duration=24, value='smile')
    intensity_a.append(time=102, duration=24, value='strong')
    label_a.append(time=102, duration=24, value='very_happy')

    # 2:06-2:13 (126-133s)
    facial_expression_a.append(time=126, duration=7, value='oh face')
    label_a.append(time=126, duration=7, value='surprised')

    # 2:13-2:18 (133-138s)
    head_movement_a.append(time=133, duration=5, value='nod')
    label_a.append(time=133, duration=5, value='energetic')

    # 2:18-2:34 (138-154s)
    facial_expression_a.append(time=138, duration=16, value='surprise face')
    label_a.append(time=138, duration=16, value='surprised')

    # 2:34-2:42 (154-162s)
    head_movement_a.append(time=154, duration=8, value='nod')
    facial_expression_a.append(time=154, duration=8, value='big smile')
    label_a.append(time=154, duration=8, value='very_happy')

    # 2:42-2:51 (162-171s)
    head_movement_a.append(time=162, duration=9, value='nod')
    facial_expression_a.append(time=162, duration=9, value='surprise face')
    label_a.append(time=162, duration=9, value='surprised')

    # 2:51-3:11 (171-191s)
    head_movement_a.append(time=171, duration=20, value='shake')
    facial_expression_a.append(time=171, duration=20, value='smile')
    intensity_a.append(time=171, duration=20, value='very strong')
    label_a.append(time=171, duration=20, value='very_energetic')

    # 3:11-3:39 (191-219s)
    head_movement_a.append(time=191, duration=28, value='nod')
    intensity_a.append(time=191, duration=28, value='strong')
    label_a.append(time=191, duration=28, value='very_energetic')

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
    infile = '/Users/miaaa/Desktop/music robot/train/shake_it_off.wav'
    outfile = '/Users/miaaa/Desktop/music robot/jams/shake_it_off.jams'

    beat_track(infile, outfile)