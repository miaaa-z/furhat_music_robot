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

    # 0-20s
    facial_expression_a.append(time=0, duration=20, value='smile')
    label_a.append(time=0, duration=20, value='happy')

    # 20-37s
    facial_expression_a.append(time=20, duration=17, value='thoughtful face, frown')
    label_a.append(time=20, duration=17, value='contemplative')

    # 37-54s
    facial_expression_a.append(time=37, duration=17, value='smile, frown')
    label_a.append(time=37, duration=17, value='contemplative')

    # 54-70s (54s to 1:10)
    facial_expression_a.append(time=54, duration=16, value='thoughtful face')
    label_a.append(time=54, duration=16, value='thoughtful')

    # 1:10-1:40 (70-100s)
    head_movement_a.append(time=70, duration=30, value='sway')
    facial_expression_a.append(time=70, duration=30, value='frown')
    intensity_a.append(time=70, duration=30, value='gentle')
    label_a.append(time=70, duration=30, value='contemplative')

    # 1:40-1:45 (100-105s)
    facial_expression_a.append(time=100, duration=5, value='frown')
    label_a.append(time=100, duration=5, value='contemplative')

    # 1:45-2:20 (105-140s)
    head_movement_a.append(time=105, duration=35, value='shake')
    facial_expression_a.append(time=105, duration=35, value='smile')
    intensity_a.append(time=105, duration=35, value='gentle')
    label_a.append(time=105, duration=35, value='happy')

    # 2:20-2:37 (140-157s)
    head_movement_a.append(time=140, duration=17, value='sway')
    facial_expression_a.append(time=140, duration=17, value='frown')
    intensity_a.append(time=140, duration=17, value='gentle')
    label_a.append(time=140, duration=17, value='contemplative')

    # 2:37-3:04 (157-184s)
    facial_expression_a.append(time=157, duration=27, value='smile, frown')
    label_a.append(time=157, duration=27, value='contemplative')

    # 3:04-3:15 (184-195s)
    facial_expression_a.append(time=184, duration=11, value='big smile')
    label_a.append(time=184, duration=11, value='very_happy')

    # 3:15-3:28 (195-208s)
    facial_expression_a.append(time=195, duration=13, value='frown')
    label_a.append(time=195, duration=13, value='contemplative')

    # 3:28-4:06 (208-246s)
    facial_expression_a.append(time=208, duration=38, value='smile, frown')
    label_a.append(time=208, duration=38, value='contemplative')

    # 4:06-4:38 (246-278s)
    head_movement_a.append(time=246, duration=32, value='shake')
    label_a.append(time=246, duration=32, value='energetic')

    # 4:38-4:55 (278-295s)
    facial_expression_a.append(time=278, duration=17, value='smile')
    label_a.append(time=278, duration=17, value='happy')

    # 4:55-5:10 (295-310s)
    head_movement_a.append(time=295, duration=15, value='sway')
    facial_expression_a.append(time=295, duration=15, value='smile')
    intensity_a.append(time=295, duration=15, value='gentle')
    label_a.append(time=295, duration=15, value='calm')

    # 5:10-5:30 (310-330s)
    facial_expression_a.append(time=310, duration=20, value='frown')
    label_a.append(time=310, duration=20, value='contemplative')

    # 5:30-5:54 (330-354s)
    facial_expression_a.append(time=330, duration=24, value='thoughtful face')
    label_a.append(time=330, duration=24, value='thoughtful')

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
    infile = '/Users/miaaa/Desktop/music robot/train/bohemian.wav'
    outfile = '/Users/miaaa/Desktop/music robot/jams/bohemian_rhapsody.jams'

    beat_track(infile, outfile)