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

    # People's movement
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

    head_movement_a.append(time=0, duration=16, value='nod')
    intensity_a.append(time=0, duration=16, value='normal')

    head_movement_a.append(time=16, duration=4, value='nod')
    facial_expression_a.append(time=16, duration=4, value='smile, close eyes')
    intensity_a.append(time=16, duration=4, value='strong')

    head_movement_a.append(time=20, duration=4, value='sway')
    facial_expression_a.append(time=20, duration=4, value='big smile')
    intensity_a.append(time=20, duration=4, value='strong')

    head_movement_a.append(time=24, duration=8, value='nod')
    intensity_a.append(time=24, duration=8, value='strong')

    head_movement_a.append(time=34, duration=7, value='shake')

    head_movement_a.append(time=47.5, duration=1.5, value='sway')
    facial_expression_a.append(time=47.5, duration=1.5, value='smile')
    intensity_a.append(time=47.5, duration=1.5, value='normal')

    head_movement_a.append(time=49, duration=42, value='sway')
    facial_expression_a.append(time=49, duration=42, value='smile, close eyes')
    intensity_a.append(time=49, duration=42, value='normal')

    facial_expression_a.append(time=91, duration=3, value='angry face')

    facial_expression_a.append(time=94, duration=13, value='smile, close eyes')

    facial_expression_a.append(time=116, duration=8, value='smile')

    head_movement_a.append(time=124, duration=43, value='shake')
    intensity_a.append(time=124, duration=43, value='very strong')

    facial_expression_a.append(time=167, duration=12, value='frown, confused face')

    head_movement_a.append(time=179, duration=33, value='nod')
    facial_expression_a.append(time=179, duration=33, value='big smile')
    intensity_a.append(time=179, duration=33, value='normal')

    head_movement_a.append(time=212, duration=28, value='sway')
    facial_expression_a.append(time=212, duration=28, value='big smile, close eyes')

    head_movement_a.append(time=240, duration=10, value='sway')
    intensity_a.append(time=240, duration=10, value='strong')

    head_movement_a.append(time=250, duration=19, value='nod')
    facial_expression_a.append(time=250, duration=19, value='smile')
    intensity_a.append(time=250, duration=19, value='strong')

    jam.annotations.append(head_movement_a)
    jam.annotations.append(facial_expression_a)
    jam.annotations.append(intensity_a)

    # Save to disk
    jam.save(outfile)

    print("Finish")
    print(f"File saved: {outfile}")


if __name__ == '__main__':
    infile = '/Users/miaaa/Desktop/music robot/3_songs_downloads/uptown_funk.wav'
    outfile = '/Users/miaaa/Desktop/music robot/3_songs_downloads/uptown_funk.jams'

    beat_track(infile, outfile)