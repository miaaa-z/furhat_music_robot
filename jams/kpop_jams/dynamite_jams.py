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

    # 0-9s
    facial_expression_a.append(time=0, duration=9, value='thoughtful face')
    label_a.append(time=0, duration=9, value='thoughtful')

    # 9-25s
    facial_expression_a.append(time=9, duration=16, value='smile')
    label_a.append(time=9, duration=16, value='happy')

    # 25-42s
    head_movement_a.append(time=25, duration=17, value='sway')
    facial_expression_a.append(time=25, duration=17, value='smile')
    label_a.append(time=25, duration=17, value='happy')

    # 42-58s
    head_movement_a.append(time=42, duration=16, value='nod')
    facial_expression_a.append(time=42, duration=16, value='big smile')
    intensity_a.append(time=42, duration=16, value='strong')
    label_a.append(time=42, duration=16, value='very_happy')

    # 58-72s (58s to 1:12)
    head_movement_a.append(time=58, duration=14, value='nod')
    facial_expression_a.append(time=58, duration=14, value='smile')
    label_a.append(time=58, duration=14, value='happy')

    # 1:12-1:15 (72-75s)
    head_movement_a.append(time=72, duration=3, value='nod')
    facial_expression_a.append(time=72, duration=3, value='surprise face')
    label_a.append(time=72, duration=3, value='surprised')

    # 1:15-1:33 (75-93s)
    head_movement_a.append(time=75, duration=18, value='sway')
    facial_expression_a.append(time=75, duration=18, value='smile')
    label_a.append(time=75, duration=18, value='happy')

    # 1:33-1:48 (93-108s)
    head_movement_a.append(time=93, duration=15, value='nod')
    facial_expression_a.append(time=93, duration=15, value='frown, smile')
    label_a.append(time=93, duration=15, value='happy')

    # 1:48-2:05 (108-125s)
    head_movement_a.append(time=108, duration=17, value='shake')
    label_a.append(time=108, duration=17, value='energetic')

    # 2:05-2:10 (125-130s)
    head_movement_a.append(time=125, duration=5, value='nod')
    facial_expression_a.append(time=125, duration=5, value='surprise face')
    label_a.append(time=125, duration=5, value='surprised')

    # 2:10-2:23 (130-143s)
    head_movement_a.append(time=130, duration=13, value='nod')
    facial_expression_a.append(time=130, duration=13, value='smile')
    label_a.append(time=130, duration=13, value='happy')

    # 2:23-2:32 (143-152s)
    facial_expression_a.append(time=143, duration=9, value='smile, frown')
    label_a.append(time=143, duration=9, value='contemplative')

    # 2:32-2:38 (152-158s)
    head_movement_a.append(time=152, duration=6, value='sway')
    intensity_a.append(time=152, duration=6, value='gentle')
    label_a.append(time=152, duration=6, value='happy')

    # 2:38-3:05 (158-185s)
    head_movement_a.append(time=158, duration=27, value='nod')
    intensity_a.append(time=158, duration=27, value='strong')
    label_a.append(time=158, duration=27, value='energetic')

    # 3:05-3:14 (185-194s)
    head_movement_a.append(time=185, duration=9, value='sway')
    label_a.append(time=185, duration=9, value='happy')

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
    infile = '/Users/miaaa/Desktop/music robot/train/dynamite.wav'
    outfile = '/Users/miaaa/Desktop/music robot/jams/dynamite.jams'

    beat_track(infile, outfile)