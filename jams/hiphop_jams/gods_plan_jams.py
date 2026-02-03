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
    facial_expression_a.append(time=0, duration=11, value='smile')
    label_a.append(time=0, duration=11, value='happy')

    # 11-25s
    head_movement_a.append(time=11, duration=14, value='nod')
    intensity_a.append(time=11, duration=14, value='gentle')
    label_a.append(time=11, duration=14, value='calm')

    # 25-37s
    head_movement_a.append(time=25, duration=12, value='nod')
    label_a.append(time=25, duration=12, value='energetic')

    # 37-49s
    head_movement_a.append(time=37, duration=12, value='sway')
    intensity_a.append(time=37, duration=12, value='gentle')
    label_a.append(time=37, duration=12, value='calm')

    # 49-63s (49s to 1:03)
    head_movement_a.append(time=49, duration=14, value='nod')
    label_a.append(time=49, duration=14, value='energetic')

    # 1:03-1:26 (63-86s)
    head_movement_a.append(time=63, duration=23, value='nod')
    facial_expression_a.append(time=63, duration=23, value='frown')
    intensity_a.append(time=63, duration=23, value='gentle')
    label_a.append(time=63, duration=23, value='contemplative')

    # 1:26-1:33 (86-93s)
    facial_expression_a.append(time=86, duration=7, value='surprise face')
    label_a.append(time=86, duration=7, value='surprised')

    # 1:33-1:50 (93-110s)
    head_movement_a.append(time=93, duration=17, value='nod')
    facial_expression_a.append(time=93, duration=17, value='frown')
    intensity_a.append(time=93, duration=17, value='strong')
    label_a.append(time=93, duration=17, value='energetic')

    # 1:50-2:08 (110-128s)
    head_movement_a.append(time=110, duration=18, value='sway')
    label_a.append(time=110, duration=18, value='energetic')

    # 2:08-2:17 (128-137s)
    head_movement_a.append(time=128, duration=9, value='nod')
    label_a.append(time=128, duration=9, value='energetic')

    # 2:17-2:40 (137-160s)
    head_movement_a.append(time=137, duration=23, value='sway')
    facial_expression_a.append(time=137, duration=23, value='thoughtful face')
    intensity_a.append(time=137, duration=23, value='gentle')
    label_a.append(time=137, duration=23, value='thoughtful')

    # 2:40-3:18 (160-198s)
    head_movement_a.append(time=160, duration=38, value='nod')
    label_a.append(time=160, duration=38, value='energetic')

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
    infile = '/Users/miaaa/Desktop/music robot/train/gods_plan.wav'
    outfile = '/Users/miaaa/Desktop/music robot/jams/gods_plan.jams'

    beat_track(infile, outfile)