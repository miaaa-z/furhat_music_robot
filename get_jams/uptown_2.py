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

    # 0-16s
    head_movement_a.append(time=0, duration=16, value='nod')
    label_a.append(time=0, duration=16, value='energetic')

    # 16-20s
    head_movement_a.append(time=16, duration=4, value='nod')
    facial_expression_a.append(time=16, duration=4, value='smile, close eyes')
    intensity_a.append(time=16, duration=4, value='strong')
    label_a.append(time=16, duration=4, value='very_happy')

    # 20-24s
    head_movement_a.append(time=20, duration=4, value='sway')
    facial_expression_a.append(time=20, duration=4, value='big smile')
    intensity_a.append(time=20, duration=4, value='strong')
    label_a.append(time=20, duration=4, value='very_happy')

    # 24-34s
    head_movement_a.append(time=24, duration=10, value='nod')
    intensity_a.append(time=24, duration=10, value='strong')
    label_a.append(time=24, duration=10, value='energetic')

    # 34-47s
    head_movement_a.append(time=34, duration=13, value='shake')
    label_a.append(time=34, duration=13, value='energetic')

    # 47-49s
    head_movement_a.append(time=47, duration=2, value='sway')
    facial_expression_a.append(time=47, duration=2, value='smile')
    label_a.append(time=47, duration=2, value='happy')

    # 49-91s (49s to 1:31)
    head_movement_a.append(time=49, duration=42, value='sway')
    facial_expression_a.append(time=49, duration=42, value='smile, close eyes')
    label_a.append(time=49, duration=42, value='happy')

    # 1:31-1:34 (91-94s)
    facial_expression_a.append(time=91, duration=3, value='angry face')
    label_a.append(time=91, duration=3, value='energetic')

    # 1:34-1:47 (94-107s)
    facial_expression_a.append(time=94, duration=13, value='smile, close eyes')
    label_a.append(time=94, duration=13, value='happy')

    # 1:47-1:56 (107-116s)
    head_movement_a.append(time=107, duration=9, value='shake')
    label_a.append(time=107, duration=9, value='energetic')

    # 1:56-2:04 (116-124s)
    facial_expression_a.append(time=116, duration=8, value='smile')
    label_a.append(time=116, duration=8, value='happy')

    # 2:04-2:47 (124-167s)
    head_movement_a.append(time=124, duration=43, value='shake')
    intensity_a.append(time=124, duration=43, value='very strong')
    label_a.append(time=124, duration=43, value='energetic')

    # 2:47-2:59 (167-179s)
    facial_expression_a.append(time=167, duration=12, value='frown, confused face')
    label_a.append(time=167, duration=12, value='contemplative')

    # 2:59-3:32 (179-212s)
    head_movement_a.append(time=179, duration=33, value='nod')
    facial_expression_a.append(time=179, duration=33, value='big smile')
    label_a.append(time=179, duration=33, value='very_happy')

    # 3:32-4:00 (212-240s)
    head_movement_a.append(time=212, duration=28, value='sway')
    facial_expression_a.append(time=212, duration=28, value='big smile, close eyes')
    label_a.append(time=212, duration=28, value='very_happy')

    # 4:00-4:10 (240-250s)
    head_movement_a.append(time=240, duration=10, value='shake')
    intensity_a.append(time=240, duration=10, value='strong')
    label_a.append(time=240, duration=10, value='energetic')

    # 4:10-4:29 (250-269s)
    head_movement_a.append(time=250, duration=19, value='nod')
    facial_expression_a.append(time=250, duration=19, value='smile')
    intensity_a.append(time=250, duration=19, value='strong')
    label_a.append(time=250, duration=19, value='very_happy')

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
    infile = '/Users/miaaa/Desktop/music robot/3_songs_downloads/uptown_funk.wav'
    outfile = '/Users/miaaa/Desktop/music robot/jams/uptown_2.jams'

    beat_track(infile, outfile)