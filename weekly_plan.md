## Weekly Progress Report

### Behavior Analysis from Three Songs

 **18 unique behaviors** with **91 total instances**.

**Most Frequent Behaviors:**
- Swaying (19x), Brow Raise (11x), Looking/Smile/Nodding (10x each)
- Frown (8x), Thoughtful (7x)

**Low-Frequency Behaviors (≤2 instances):**
- Brow_raise_left, Shaking, Lifting_head, Blink, Oh_face, Narrow_eyes,
- Relaxed_face, Eyes_widen, Close_eyes, Craning

**Behavior Distribution by Category:**
- Head Movements: Nodding (10x), Swaying (19x), Shaking (2x)
- Facial Expressions: Smile (10x), Frown (8x), Thoughtful (7x), Surprise (4x)
- Brows: Brow_raise (11x)
- Eyes: Close_eyes, Eyes_widen, Narrow_eyes, Blink (1x each)

### Baseline Implementation

Implemented  baseline mapping system using three audio features:

- 1. **Tempo**: Calculated using `librosa.beat.beat_track()`,
- categorized as slow (<110 BPM), moderate (110-130 BPM), or fast (>130 BPM)
- 2. **Energy**: Computed using `librosa.feature.rms()` (Root Mean Square),
- categorized as low (<0.10), medium (0.10-0.20), or high (>0.20)
- 3. **Brightness**: Extracted using `librosa.feature.spectral_centroid()`,
- categorized as dark (<1800 Hz), neutral (1800-2300 Hz), or bright (>2300 Hz)

The baseline uses a predefined behavior map with 27 combinations (3×3×3) of these features,
each mapped to specific gestures and LED colors. Two gestures are randomly
selected  and executed simultaneously with LED.