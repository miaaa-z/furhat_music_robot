# Weekly Progress Report

## Overview
This week focused on evaluating pre-trained Essentia models for music feature extraction and planning improvements to the baseline behavior mapping system.

---

## Essentia Model Evaluation

### Models Downloaded and Tested
Downloaded 12 pre-trained models from Essentia's models

### Model Performance Analysis

#### Accurate Models
**Energy (Engagement) - Discogs EffNet:**
- Predictions align well with ground truth from TuneBat
- Reliable for integration into baseline system

**Voice/Instrumental Detection - Discogs EffNet:**
- Performs reasonably well
- Limited applicability to current behavior mapping needs

**Electronic Music Detection - Discogs EffNet:**
- Acceptable performance
- Limited applicability to current behavior mapping needs


### Benchmark Comparison

TuneBat and SongBPM results across 8 test songs:

| Song | Energy (TuneBat) | Danceability (TuneBat) | Happiness (TuneBat) | Tempo (BPM) |

| Not Like Us | 47 | 90 | 21 | 101 |
| Zoo | 80 | 86 | 90 | 129 |
| Love Story | 74 | 62 | 30 | 119 |
| Shake It Off | 79 | 65 | 94 | 160 |
| ...Ready For It? | 76 | 61 | 42 | 160 |
| Fortnight | 39 | 50 | 28 | 192 |
| Champagne Problems | 24 | 46 | 32 | 171 |
| Faded | 65 | 59 | 17 | 90 |

**Key Findings:**
- Essentia's engagement_regression (energy) model matches TuneBat energy values
- Danceability and happiness models from both VGGish and Discogs architectures show poor correlation with the websites


## Baseline System Improvements

### Current Baseline Implementation
The existing baseline uses three audio features extracted via Librosa:

1. **Tempo**: `librosa.beat.beat_track()` → Categorized as slow (<110 BPM), moderate (110-130 BPM), or fast (>130 BPM)
2. **Energy**: `librosa.feature.rms()` → Categorized as low (<0.10), medium (0.10-0.20), or high (>0.20)
3. **Brightness**: `librosa.feature.spectral_centroid()` → Categorized as dark (<1800 Hz), neutral (1800-2300 Hz), or bright (>2300 Hz)

Maps 27 combinations (3×3×3) to robot behaviors with randomized gesture pairs and LED colors.

### Planned Improvements

**Replace Librosa Energy with Essentia Energy:**
- Swap current RMS-based energy calculation with Essentia's `engagement_regression-discogs-effnet-1.pb` model
- Expected to provide more accurate energy detection aligned with human perception
- Maintain same categorical thresholds or recalibrate based on Essentia output range

**Retain Existing Features:**
- Keep Librosa-based tempo detection
- Keep spectral centroid for brightness
- Exclude danceability and happiness due to poor model performance

---

## Behavior Analysis Reference

### Annotated Behaviors from Three Songs
Identified **18 unique behaviors** with **91 total instances**:

**Most Frequent:**
- Swaying (19×), Brow Raise (11×), Looking/Smile/Nodding (10× each)
- Frown (8×), Thoughtful (7×)

**Low Frequency (≤2 instances):**
- Brow_raise_left, Shaking, Lifting_head, Blink, Oh_face, Narrow_eyes
- Relaxed_face, Eyes_widen, Close_eyes, Craning

**Distribution by Category:**
- Head Movements: Nodding (10×), Swaying (19×), Shaking (2×)
- Facial Expressions: Smile (10×), Frown (8×), Thoughtful (7×), Surprise (4×)
- Brows: Brow_raise (11×)
- Eyes: Close_eyes, Eyes_widen, Narrow_eyes, Blink (1× each)

---

## Next Steps

1. Integrate Essentia's energy model into baseline system
2. Test improved baseline on validation songs
3. Consider ML for danceability/happiness if needed for behavior mapping
4. Document performance comparison between Librosa-only and Essentia-enhanced baselines

