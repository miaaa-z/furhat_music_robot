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
    facial_expression_a.append(time=0, duration=16, value='thoughtful face')
    label_a.append(time=0, duration=16, value='thoughtful')

    # 16-40s
    facial_expression_a.append(time=16, duration=24, value='smile, close eyes')
    label_a.append(time=16, duration=24, value='enjoy')

    # 40-48s
    facial_expression_a.append(time=40, duration=8, value='oh face')
    label_a.append(time=40, duration=8, value='surprised')

    # 48-63s (48s to 1:03)
    head_movement_a.append(time=48, duration=15, value='sway')
    facial_expression_a.append(time=48, duration=15, value='frown')
    intensity_a.append(time=48, duration=15, value='gentle')
    label_a.append(time=48, duration=15, value='contemplative')

    # 1:03-1:43 (63-103s)
    head_movement_a.append(time=63, duration=40, value='nod')
    facial_expression_a.append(time=63, duration=40, value='smile')
    intensity_a.append(time=63, duration=40, value='very gentle')
    label_a.append(time=63, duration=40, value='happy')

    # 1:43-2:05 (103-125s)
    facial_expression_a.append(time=103, duration=22, value='surprise face')
    label_a.append(time=103, duration=22, value='surprised')

    # 2:05-2:30 (125-150s)
    head_movement_a.append(time=125, duration=25, value='sway')
    intensity_a.append(time=125, duration=25, value='gentle')
    label_a.append(time=125, duration=25, value='calm')

    # 2:30-3:06 (150-186s)
    head_movement_a.append(time=150, duration=36, value='sway')
    facial_expression_a.append(time=150, duration=36, value='smile, close eyes')
    intensity_a.append(time=150, duration=36, value='very gentle')
    label_a.append(time=150, duration=36, value='enjoy')

    # 3:06-3:21 (186-201s)
    facial_expression_a.append(time=186, duration=15, value='surprise face')
    label_a.append(time=186, duration=15, value='surprised')

    # 3:21-3:47 (201-227s)
    head_movement_a.append(time=201, duration=26, value='sway')
    facial_expression_a.append(time=201, duration=26, value='frown')
    label_a.append(time=201, duration=26, value='contemplative')

    # 3:47-4:05 (227-245s)
    head_movement_a.append(time=227, duration=18, value='sway')
    facial_expression_a.append(time=227, duration=18, value='sad face')
    intensity_a.append(time=227, duration=18, value='very gentle')
    label_a.append(time=227, duration=18, value='contemplative')

    # 4:05-4:22 (245-262s)
    facial_expression_a.append(time=245, duration=17, value='oh face')
    label_a.append(time=245, duration=17, value='surprised')

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
    infile = '/Users/miaaa/Desktop/music robot/train/i_will_always_love_you.wav'
    outfile = '/Users/miaaa/Desktop/music robot/jams/i_will_always_love_you.jams'

    beat_track(infile, outfile)