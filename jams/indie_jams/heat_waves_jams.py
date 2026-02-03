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
    facial_expression_a.append(time=0, duration=6, value='thoughtful face')
    label_a.append(time=0, duration=6, value='thoughtful')

    # 6-16s
    head_movement_a.append(time=6, duration=10, value='nod')
    facial_expression_a.append(time=6, duration=10, value='smile')
    intensity_a.append(time=6, duration=10, value='gentle')
    label_a.append(time=6, duration=10, value='happy')

    # 16-41s
    head_movement_a.append(time=16, duration=25, value='sway')
    label_a.append(time=16, duration=25, value='happy')

    # 41-47s
    head_movement_a.append(time=41, duration=6, value='nod')
    facial_expression_a.append(time=41, duration=6, value='surprise face')
    label_a.append(time=41, duration=6, value='surprised')

    # 47-64s (47s to 1:04)
    head_movement_a.append(time=47, duration=17, value='sway')
    label_a.append(time=47, duration=17, value='happy')

    # 1:04-1:28 (64-88s)
    head_movement_a.append(time=64, duration=24, value='nod')
    intensity_a.append(time=64, duration=24, value='very gentle')
    label_a.append(time=64, duration=24, value='calm')

    # 1:28-1:41 (88-101s)
    head_movement_a.append(time=88, duration=13, value='nod')
    facial_expression_a.append(time=88, duration=13, value='smile')
    label_a.append(time=88, duration=13, value='happy')

    # 1:41-1:58 (101-118s)
    head_movement_a.append(time=101, duration=17, value='sway')
    intensity_a.append(time=101, duration=17, value='gentle')
    label_a.append(time=101, duration=17, value='calm')

    # 1:58-2:03 (118-123s)
    label_a.append(time=118, duration=5, value='calm')

    # 2:03-2:27 (123-147s)
    head_movement_a.append(time=123, duration=24, value='sway')
    intensity_a.append(time=123, duration=24, value='strong')
    label_a.append(time=123, duration=24, value='energetic')

    # 2:27-2:40 (147-160s)
    facial_expression_a.append(time=147, duration=13, value='thoughtful face')
    label_a.append(time=147, duration=13, value='thoughtful')

    # 2:40-2:51 (160-171s)
    head_movement_a.append(time=160, duration=11, value='nod')
    intensity_a.append(time=160, duration=11, value='gentle')
    label_a.append(time=160, duration=11, value='calm')

    # 2:51-2:54 (171-174s)
    label_a.append(time=171, duration=3, value='calm')

    # 2:54-3:18 (174-198s)
    head_movement_a.append(time=174, duration=24, value='sway')
    intensity_a.append(time=174, duration=24, value='strong')
    label_a.append(time=174, duration=24, value='energetic')

    # 3:18-3:30 (198-210s)
    head_movement_a.append(time=198, duration=12, value='nod')
    label_a.append(time=198, duration=12, value='energetic')

    # 3:30-3:41 (210-221s)
    head_movement_a.append(time=210, duration=11, value='sway')
    intensity_a.append(time=210, duration=11, value='gentle')
    label_a.append(time=210, duration=11, value='happy')

    # 3:41-3:53 (221-233s)
    head_movement_a.append(time=221, duration=12, value='nod')
    label_a.append(time=221, duration=12, value='energetic')

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
    infile = '/Users/miaaa/Desktop/music robot/train/heat_waves.wav'
    outfile = '/Users/miaaa/Desktop/music robot/jams/heat_waves.jams'

    beat_track(infile, outfile)