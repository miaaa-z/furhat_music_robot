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

    # 0-15s
    head_movement_a.append(time=0, duration=15, value='nod')
    label_a.append(time=0, duration=15, value='energetic')

    # 15-28s
    head_movement_a.append(time=15, duration=13, value='sway')
    facial_expression_a.append(time=15, duration=13, value='smile')
    label_a.append(time=15, duration=13, value='happy')

    # 28-33s
    head_movement_a.append(time=28, duration=5, value='nod')
    facial_expression_a.append(time=28, duration=5, value='big smile')
    intensity_a.append(time=28, duration=5, value='strong')
    label_a.append(time=28, duration=5, value='very_happy')

    # 33-38s
    facial_expression_a.append(time=33, duration=5, value='oh face')
    label_a.append(time=33, duration=5, value='surprised')

    # 38-51s
    head_movement_a.append(time=38, duration=13, value='nod')
    facial_expression_a.append(time=38, duration=13, value='smile')
    label_a.append(time=38, duration=13, value='happy')

    # 51-58s
    head_movement_a.append(time=51, duration=7, value='nod')
    label_a.append(time=51, duration=7, value='energetic')

    # 58-70s (58s to 1:10)
    head_movement_a.append(time=58, duration=12, value='shake')
    facial_expression_a.append(time=58, duration=12, value='smile')
    label_a.append(time=58, duration=12, value='energetic')

    # 1:10-1:17 (70-77s)
    head_movement_a.append(time=70, duration=7, value='nod')
    label_a.append(time=70, duration=7, value='energetic')

    # 1:17-1:40 (77-100s)
    head_movement_a.append(time=77, duration=23, value='nod')
    facial_expression_a.append(time=77, duration=23, value='big smile')
    intensity_a.append(time=77, duration=23, value='strong')
    label_a.append(time=77, duration=23, value='very_happy')

    # 1:40-1:46 (100-106s)
    facial_expression_a.append(time=100, duration=6, value='surprise face')
    label_a.append(time=100, duration=6, value='surprised')

    # 1:46-2:01 (106-121s)
    facial_expression_a.append(time=106, duration=15, value='oh face')
    label_a.append(time=106, duration=15, value='surprised')

    # 2:01-2:05 (121-125s)
    facial_expression_a.append(time=121, duration=4, value='surprise face')
    label_a.append(time=121, duration=4, value='surprised')

    # 2:05-2:27 (125-147s)
    head_movement_a.append(time=125, duration=22, value='sway')
    intensity_a.append(time=125, duration=22, value='very strong')
    label_a.append(time=125, duration=22, value='very_energetic')

    # 2:27-2:50 (147-170s)
    head_movement_a.append(time=147, duration=23, value='nod')
    intensity_a.append(time=147, duration=23, value='very strong')
    label_a.append(time=147, duration=23, value='very_energetic')

    # 2:50-2:58 (170-178s)
    facial_expression_a.append(time=170, duration=8, value='big smile')
    label_a.append(time=170, duration=8, value='very_happy')

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
    infile = '/Users/miaaa/Desktop/music robot/train/cruel_summer.wav'
    outfile = '/Users/miaaa/Desktop/music robot/jams/cruel_summer.jams'

    beat_track(infile, outfile)