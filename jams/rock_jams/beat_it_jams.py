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

    # 0-11s
    facial_expression_a.append(time=0, duration=11, value='frown')
    label_a.append(time=0, duration=11, value='contemplative')

    # 11-23s
    head_movement_a.append(time=11, duration=12, value='sway')
    label_a.append(time=11, duration=12, value='happy')

    # 23-41s
    head_movement_a.append(time=23, duration=18, value='nod')
    facial_expression_a.append(time=23, duration=18, value='smile')
    label_a.append(time=23, duration=18, value='happy')

    # 41-46s
    head_movement_a.append(time=41, duration=5, value='shake')
    facial_expression_a.append(time=41, duration=5, value='big smile')
    label_a.append(time=41, duration=5, value='very_happy')

    # 46-65s (46s to 1:05)
    head_movement_a.append(time=46, duration=19, value='nod')
    facial_expression_a.append(time=46, duration=19, value='smile')
    intensity_a.append(time=46, duration=19, value='strong')
    label_a.append(time=46, duration=19, value='very_happy')

    # 1:05-1:11 (65-71s)
    head_movement_a.append(time=65, duration=6, value='shake')
    facial_expression_a.append(time=65, duration=6, value='smile')
    label_a.append(time=65, duration=6, value='energetic')

    # 1:11-1:40 (71-100s)
    head_movement_a.append(time=71, duration=29, value='nod')
    label_a.append(time=71, duration=29, value='energetic')

    # 1:40-2:08 (100-128s)
    head_movement_a.append(time=100, duration=28, value='nod')
    facial_expression_a.append(time=100, duration=28, value='frown')
    intensity_a.append(time=100, duration=28, value='strong')
    label_a.append(time=100, duration=28, value='very_energetic')

    # 2:08-2:14 (128-134s)
    head_movement_a.append(time=128, duration=6, value='shake')
    intensity_a.append(time=128, duration=6, value='strong')
    label_a.append(time=128, duration=6, value='energetic')

    # 2:14-2:23 (134-143s)
    head_movement_a.append(time=134, duration=9, value='nod')
    facial_expression_a.append(time=134, duration=9, value='frown')
    label_a.append(time=134, duration=9, value='contemplative')

    # 2:23-2:30 (143-150s)
    facial_expression_a.append(time=143, duration=7, value='big smile')
    label_a.append(time=143, duration=7, value='very_happy')

    # 2:30-3:16 (150-196s)
    facial_expression_a.append(time=150, duration=46, value='thoughtful face')
    label_a.append(time=150, duration=46, value='thoughtful')

    # 3:16-3:22 (196-202s)
    head_movement_a.append(time=196, duration=6, value='nod')
    intensity_a.append(time=196, duration=6, value='very strong')
    label_a.append(time=196, duration=6, value='very_energetic')

    # 3:22-3:44 (202-224s)
    head_movement_a.append(time=202, duration=22, value='nod, shake')
    label_a.append(time=202, duration=22, value='energetic')

    # 3:44-3:50 (224-230s)
    head_movement_a.append(time=224, duration=6, value='nod')
    facial_expression_a.append(time=224, duration=6, value='big smile')
    intensity_a.append(time=224, duration=6, value='strong')
    label_a.append(time=224, duration=6, value='very_happy')

    # 3:50-4:11 (230-251s)
    head_movement_a.append(time=230, duration=21, value='nod, shake')
    facial_expression_a.append(time=230, duration=21, value='big smile')
    intensity_a.append(time=230, duration=21, value='strong')
    label_a.append(time=230, duration=21, value='very_happy')

    # 4:11-4:18 (251-258s)
    facial_expression_a.append(time=251, duration=7, value='thoughtful face')
    label_a.append(time=251, duration=7, value='thoughtful')

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
    infile = '/Users/miaaa/Desktop/music robot/train/beat_it.wav'
    outfile = '/Users/miaaa/Desktop/music robot/jams/beat_it.jams'

    beat_track(infile, outfile)