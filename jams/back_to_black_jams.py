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
    beat_a.annotation_metadata = jams.AnnotationMetadata(data_source='librosa beat tracker')

    # Add beat timings to the annotation record
    for t in beat_times:
        beat_a.append(time=t, duration=0.0)

    # Store the new annotation in the jam
    jam.annotations.append(beat_a)

    # Add tempo estimation to the annotation
    tempo_a = jams.Annotation(namespace='tempo', time=0, duration=track_duration)
    tempo_a.annotation_metadata = jams.AnnotationMetadata(data_source='librosa tempo estimator')

    tempo_a.append(time=0.0,
                   duration=track_duration,
                   value=tempo,
                   confidence=1.0)

    # Store the new annotation in the jam
    jam.annotations.append(tempo_a)

    # People's movement annotations
    # Head movement annotation - use tag_open namespace
    head_movement_a = jams.Annotation(namespace='tag_open')
    head_movement_a.annotation_metadata = jams.AnnotationMetadata(
        data_source='head_movement'
    )

    # Facial expression annotation - use tag_open namespace
    facial_expression_a = jams.Annotation(namespace='tag_open')
    facial_expression_a.annotation_metadata = jams.AnnotationMetadata(
        data_source='facial_expression'
    )

    # Intensity annotation - use tag_open namespace
    intensity_a = jams.Annotation(namespace='tag_open')
    intensity_a.annotation_metadata = jams.AnnotationMetadata(
        data_source='intensity'
    )

    # Annotations based on the image for "Back to Black"

    # 0-15s
    head_movement_a.append(time=0, duration=15, value='sway')
    facial_expression_a.append(time=0, duration=15, value='smile')
    intensity_a.append(time=0, duration=15, value='normal')

    # 15-20s
    head_movement_a.append(time=15, duration=5, value='')
    facial_expression_a.append(time=15, duration=5, value='surprise face, confused face')

    # 20-27s
    head_movement_a.append(time=20, duration=7, value='')
    facial_expression_a.append(time=20, duration=7, value='smile')

    # 27-30s
    head_movement_a.append(time=27, duration=3, value='')
    facial_expression_a.append(time=27, duration=3, value='frown')

    # 30-45s
    head_movement_a.append(time=30, duration=15, value='')
    facial_expression_a.append(time=30, duration=15, value='smile, frown')

    # 45s-1:07 (45-67s)
    head_movement_a.append(time=45, duration=22, value='')
    facial_expression_a.append(time=45, duration=22, value='frown')
    intensity_a.append(time=45, duration=22, value='gentle')

    # 1:07-1:16 (67-76s)
    head_movement_a.append(time=67, duration=9, value='nod')
    facial_expression_a.append(time=67, duration=9, value='big smile')

    # 1:16-1:22 (76-82s)
    head_movement_a.append(time=76, duration=6, value='')
    facial_expression_a.append(time=76, duration=6, value='smile')

    # 1:22-1:27 (82-87s)
    head_movement_a.append(time=82, duration=5, value='')
    facial_expression_a.append(time=82, duration=5, value='confused face')

    # 1:27-1:37 (87-97s)
    head_movement_a.append(time=87, duration=10, value='nod')
    facial_expression_a.append(time=87, duration=10, value='frown')
    intensity_a.append(time=87, duration=10, value='gentle')

    # 1:37-1:54 (97-114s)
    head_movement_a.append(time=97, duration=17, value='')
    facial_expression_a.append(time=97, duration=17, value='thoughtful face')

    # 1:54-1:57 (114-117s)
    head_movement_a.append(time=114, duration=3, value='sway')
    facial_expression_a.append(time=114, duration=3, value='thoughtful face')
    intensity_a.append(time=114, duration=3, value='gentle')

    # 1:57-2:04 (117-124s)
    head_movement_a.append(time=117, duration=7, value='nod')
    facial_expression_a.append(time=117, duration=7, value='thoughtful face')
    intensity_a.append(time=117, duration=7, value='gentle')

    # 2:04-2:11 (124-131s)
    head_movement_a.append(time=124, duration=7, value='nod')
    facial_expression_a.append(time=124, duration=7, value='smile')
    intensity_a.append(time=124, duration=7, value='normal')

    # 2:11-2:15 (131-135s)
    head_movement_a.append(time=131, duration=4, value='nod')
    facial_expression_a.append(time=131, duration=4, value='disgust face')
    intensity_a.append(time=131, duration=4, value='gentle')

    # 2:15-2:20 (135-140s)
    head_movement_a.append(time=135, duration=5, value='')
    facial_expression_a.append(time=135, duration=5, value='disgust face')

    # 2:20-2:45 (140-165s)
    head_movement_a.append(time=140, duration=25, value='')
    facial_expression_a.append(time=140, duration=25, value='thoughtful face')
    intensity_a.append(time=140, duration=25, value='very gentle')

    # 2:45-4:00 (165-240s)
    head_movement_a.append(time=165, duration=75, value='sway')
    intensity_a.append(time=165, duration=75, value='very gentle')

    # Append all annotations
    jam.annotations.append(head_movement_a)
    jam.annotations.append(facial_expression_a)
    jam.annotations.append(intensity_a)

    # Save to disk
    jam.save(outfile)

    print("Finish")
    print(f"File saved: {outfile}")


if __name__ == '__main__':
    infile = '/Users/miaaa/Desktop/music robot/3_songs_downloads/back_to_black.wav'
    outfile = '/Users/miaaa/Desktop/music robot/3_songs_downloads/back_to_black.jams'

    beat_track(infile, outfile)