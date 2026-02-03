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

    label_a = jams.Annotation(namespace='tag_open')
    label_a.annotation_metadata = jams.AnnotationMetadata(
        data_source='emotion_label',
        annotator={'name': 'Mia', 'email': ''},
        version='1.0'
    )

    # 0-13s
    head_movement_a.append(time=0, duration=13, value='sway')
    intensity_a.append(time=0, duration=13, value='very gentle')
    label_a.append(time=0, duration=13, value='calm')

    # 13-17s
    head_movement_a.append(time=13, duration=4, value='')
    facial_expression_a.append(time=13, duration=4, value='brows lifting, surprise face')
    label_a.append(time=13, duration=4, value='surprised')

    # 17-32s
    head_movement_a.append(time=17, duration=15, value='sway')
    facial_expression_a.append(time=17, duration=15, value='thoughtful face')
    intensity_a.append(time=17, duration=15, value='very gentle')
    label_a.append(time=17, duration=15, value='thoughtful')

    # 32-34s
    head_movement_a.append(time=32, duration=2, value='')
    facial_expression_a.append(time=32, duration=2, value='brows lifting, surprise face')
    label_a.append(time=32, duration=2, value='surprised')

    # 34-40s
    head_movement_a.append(time=34, duration=6, value='')
    facial_expression_a.append(time=34, duration=6, value='thoughtful face')
    label_a.append(time=34, duration=6, value='thoughtful')

    # 40-50s
    head_movement_a.append(time=40, duration=10, value='')
    facial_expression_a.append(time=40, duration=10, value='frown')
    label_a.append(time=40, duration=10, value='contemplative')

    # 50-52s
    head_movement_a.append(time=50, duration=2, value='')
    facial_expression_a.append(time=50, duration=2, value='brows lifting, surprise face')
    label_a.append(time=50, duration=2, value='surprised')

    # 52-73s (1:13)
    head_movement_a.append(time=52, duration=21, value='sway')
    facial_expression_a.append(time=52, duration=21, value='frown, thoughtful face')
    intensity_a.append(time=52, duration=21, value='very gentle')
    label_a.append(time=52, duration=21, value='contemplative')

    # 1:13-1:24
    head_movement_a.append(time=73, duration=11, value='')
    facial_expression_a.append(time=73, duration=11, value='smile')
    label_a.append(time=73, duration=11, value='happy')

    # 1:24-1:26 (84-86s)
    head_movement_a.append(time=84, duration=2, value='')
    facial_expression_a.append(time=84, duration=2, value='"Oh" face')
    label_a.append(time=84, duration=2, value='surprised')

    # 1:26-1:28 (86-88s)
    head_movement_a.append(time=86, duration=2, value='')
    facial_expression_a.append(time=86, duration=2, value='brows lifting')
    label_a.append(time=86, duration=2, value='surprised')

    # 1:28-1:33 (88-93s)
    head_movement_a.append(time=88, duration=5, value='Looking to the top right')
    facial_expression_a.append(time=88, duration=5, value='thoughtful face')
    label_a.append(time=88, duration=5, value='thoughtful')

    # 1:33-1:39 (93-99s)
    head_movement_a.append(time=93, duration=6, value='')
    facial_expression_a.append(time=93, duration=6, value='smile, frown')
    label_a.append(time=93, duration=6, value='mixed')

    # 1:39-1:48 (99-108s)
    head_movement_a.append(time=99, duration=9, value='')
    facial_expression_a.append(time=99, duration=9, value='lift left eyebrow, smile')
    label_a.append(time=99, duration=9, value='happy')

    # 1:48-1:58 (108-118s)
    head_movement_a.append(time=108, duration=10, value='')
    facial_expression_a.append(time=108, duration=10, value='frown, thoughtful face')
    label_a.append(time=108, duration=10, value='contemplative')

    # 1:58-2:00 (118-120s)
    head_movement_a.append(time=118, duration=2, value='')
    facial_expression_a.append(time=118, duration=2, value='brows lifting')
    label_a.append(time=118, duration=2, value='surprised')

    # 2:00-2:05 (120-125s)
    head_movement_a.append(time=120, duration=5, value='')
    facial_expression_a.append(time=120, duration=5, value='frown')
    label_a.append(time=120, duration=5, value='contemplative')

    # 2:05-2:07 (125-127s)
    head_movement_a.append(time=125, duration=2, value='look up')
    facial_expression_a.append(time=125, duration=2, value='brows lifting')
    label_a.append(time=125, duration=2, value='surprised')

    # 2:07-2:11 (127-131s)
    head_movement_a.append(time=127, duration=4, value='look down')
    facial_expression_a.append(time=127, duration=4, value='narrowing eyes, smile')
    label_a.append(time=127, duration=4, value='happy')

    # 2:11-2:14 (131-134s)
    head_movement_a.append(time=131, duration=3, value='look down')
    facial_expression_a.append(time=131, duration=3, value='brows lifting')
    label_a.append(time=131, duration=3, value='surprised')

    # 2:14-2:19 (134-139s)
    head_movement_a.append(time=134, duration=5, value='sway')
    intensity_a.append(time=134, duration=5, value='very gentle')
    label_a.append(time=134, duration=5, value='calm')

    # 2:19-2:30 (139-150s)
    head_movement_a.append(time=139, duration=11, value='look down')
    facial_expression_a.append(time=139, duration=11, value='frown')
    label_a.append(time=139, duration=11, value='contemplative')

    # 2:30-2:34 (150-154s)
    head_movement_a.append(time=150, duration=4, value='')
    facial_expression_a.append(time=150, duration=4, value='brows lifting, surprise face')
    label_a.append(time=150, duration=4, value='surprised')

    # 2:34-2:49 (154-169s)
    head_movement_a.append(time=154, duration=15, value='Looking to the top right')
    facial_expression_a.append(time=154, duration=15, value='big smile')
    label_a.append(time=154, duration=15, value='very_happy')

    # 2:49-3:00 (169-180s)
    head_movement_a.append(time=169, duration=11, value='look down, sway')
    facial_expression_a.append(time=169, duration=11, value='smile')
    intensity_a.append(time=169, duration=11, value='very gentle')
    label_a.append(time=169, duration=11, value='happy')

    # 3:00-3:04 (180-184s)
    head_movement_a.append(time=180, duration=4, value='')
    facial_expression_a.append(time=180, duration=4, value='brows lifting, smile')
    label_a.append(time=180, duration=4, value='surprised')

    # 3:04-3:14 (184-194s)
    head_movement_a.append(time=184, duration=10, value='looking to the right, sway')
    facial_expression_a.append(time=184, duration=10, value='frown')
    intensity_a.append(time=184, duration=10, value='gentle')
    label_a.append(time=184, duration=10, value='contemplative')

    # 3:14-3:24 (194-204s)
    head_movement_a.append(time=194, duration=10, value='look down')
    facial_expression_a.append(time=194, duration=10, value='thoughtful face')
    label_a.append(time=194, duration=10, value='thoughtful')

    # 3:24-3:27 (204-207s)
    head_movement_a.append(time=204, duration=3, value='')
    facial_expression_a.append(time=204, duration=3, value='brows lifting, surprise face')
    label_a.append(time=204, duration=3, value='surprised')

    # 3:27-3:37 (207-217s)
    head_movement_a.append(time=207, duration=10, value='Looking to the top right')
    facial_expression_a.append(time=207, duration=10, value='thoughtful face')
    label_a.append(time=207, duration=10, value='thoughtful')

    # 3:37-3:43 (217-223s)
    head_movement_a.append(time=217, duration=6, value='')
    facial_expression_a.append(time=217, duration=6, value='close eyes')
    label_a.append(time=217, duration=6, value='enjoy')

    # 3:43-3:51 (223-231s)
    head_movement_a.append(time=223, duration=8, value='')
    facial_expression_a.append(time=223, duration=8, value='smile, brows lifting')
    label_a.append(time=223, duration=8, value='happy')

    # 3:51-3:58 (231-238s)
    head_movement_a.append(time=231, duration=7, value='')
    facial_expression_a.append(time=231, duration=7, value='big smile')
    label_a.append(time=231, duration=7, value='very_happy')

    # 3:58-4:27 (238-267s)
    head_movement_a.append(time=238, duration=29, value='')
    facial_expression_a.append(time=238, duration=29, value='smile, frown, brows lifting')
    label_a.append(time=238, duration=29, value='contemplative')

    # 4:27-4:45 (267-285s)
    head_movement_a.append(time=267, duration=18, value='')
    facial_expression_a.append(time=267, duration=18, value='smile')
    label_a.append(time=267, duration=18, value='calm')

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
    infile = '/Users/miaaa/Desktop/music robot/3_songs_downloads/someone_like_you.wav'
    outfile = '/Users/miaaa/Desktop/music robot/jams/someone_2.jams'

    beat_track(infile, outfile)