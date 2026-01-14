## **Weekly Progress Report**

### **Completed This Week**

#### 1. Behavior Annotation & Data Collection
- Completed detailed behavior annotations for three songs using YouTube "first time listening" reaction videos:
  - "Someone Like You" by Adele (65 instances, 16 unique behaviors)
  - "Uptown Funk" by Mark Ronson ft. Bruno Mars (29 instances, 10 unique behaviors)
  - "Back to Black" by Amy Winehouse (27 instances, 9 unique behaviors)
- **Total: 121 behavior instances across 22 unique behavior types**

---
#### 2. Behavior Vocabulary Analysis

**Actual Unique Vocabulary: 22 behaviors (21 implementable)**
(crane (people move heads forward/backward) can't be implemented)
**High-Frequency Behaviors (>= 10 instances)**
- smile (20×), sway (13×), frown (13×), brows lifting (12×), 
- thoughtful face (11×), nod (10×)

**Medium-Frequency Behaviors (3-9 instances)**
- surprise face (6×), big smile (6×), look down (5×), 
- close eyes (5×), looking to top right (3×), confused face (3×)

**Low-Frequency Behaviors (<3 instances)**
- lift left eyebrow (2×), crane (2×), shake (2×), disgust face (2×),
- look up (1×), looking to right (1×), blink (1×), "oh" face (1×),
- narrowing eyes (1×), angry face (1×)

---

#### 3. Used/Not Used Behaviors

**Built-in Furhat Gestures Used (13)**
- Smile, BigSmile, Surprise, Oh, Thoughtful
- BrowRaise, BrowFrown
- Nod, Shake, Roll
- CloseEyes, Blink
- ExpressDisgust, ExpressAnger

**Unused builtin gestures (5 gestures):**
- Wink, ExpressFear, ExpressSad, GazeAway, OpenEyes

**LED Colors:**
- 8 colors implemented: Red, Green, Blue, Yellow, Purple,
- Cyan, White, Orange


**Custom Behaviors Used in Annotations**
- head_sway() - continuous swaying (13×)
- head_nod_fast() - continuous nodding (10×)
- raise_left_brow() - left eyebrow raise (2×)
- custom_narrow() - narrowing eyes (1×)
- custom_confused() - confused expression (3×)
- head_positions() - directional looking (10× total)

**Custom Behaviors Implemented but Unused**
- custom_very_happy, custom_determined, custom_shy, custom_relaxed
- raise_right_brow, eye_look_direction
- Available for future annotations

---

#### 4. Behavior Cleaning

**Removed Infeasible Behaviors in CSV**
- Body movements: shoulder shrugging, hand gestures
- Complex facial expressions: pouting, lip biting, face scrunching
- Physical actions: lip-syncing, patting on T-shirts
- Crane/neck extension 
- Total removed: ~15 behavior types

**Standardized Mappings**
- "swaying head" / "head swaying" → sway
- "nodding" / "nodding to the beat" → nod
- "brows lifting" / "eyebrows lifting" → brows lifting (BrowRaise)
- "angry face" → ExpressAnger gesture
- "disgust face" → ExpressDisgust gesture
- Various looking directions → head_positions() with parameters


#### 6. Audio Feature Extraction
- Implemented Librosa-based feature extraction:
  - Tempo (BPM) using 5-second sliding windows
  - Energy (RMS)
  - Beat tracking
- Developed dynamic baseline (windowed features) systems

---

### **Current Challenges**

#### 1. Timing Synchronization
- Observed about 1 second delay between command execution and Furhat response
- Delay occurs in both programmatic control and web interface

#### 2. Tempo Estimation Accuracy
- Librosa shows inconsistencies:
  - Counter-intuitive results ("Uptown Funk" slower than "Someone Like You")
  - Large BPM fluctuations within songs using 3-second windows (90→120→130 BPM)
  - 
  - 5-second windows more stable but accuracy uncertain
- Possible causes: Eighth-note detection instead of quarter-notes, model limitations

#### 3. Data Imbalance
- Critical issue: 10 of 22 behaviors (45%) appear fewer than 3 times
- Insufficient training data for machine learning:
  - Cannot reliably learn patterns for rare behaviors
  - May require additional annotations or data augmentation
- Coverage varies by song:
  - "Someone Like You": best coverage (16 behaviors, 65 instances)
  - "Uptown Funk": moderate (10 behaviors, 29 instances)
  - "Back to Black": moderate (9 behaviors, 27 instances)

#### 4. Missing Furhat Parameters
- SDK lacks certain facial expression parameters: MOUTH_POUT, NOSE_WRINKLE, LIP_BITE
- Not documented in official Furhat documentation
- Workaround: Excluded from vocabulary

---

### **Next Steps**

#### 1. Code Organization (In Progress)
- Clean up and document code
- Upload to Git repository

#### 2. Tempo Analysis Experiment
- Query actual BPM from online databases for all three songs
- Generate tempo profile plots with 5-second windows
- Compare Librosa predictions against ground truth
- Visualize: detected tempo curve + actual BPM baseline
- Decide whether to use alternative tempo estimation models

#### 3. Delay Compensation
- Measure average Furhat response delay
- Implement early triggering mechanism (e.g., 0.5s advance)
- Test and validate improved synchronization

### **Summary Statistics**

```
Dataset:
- Songs: 3
- Total Instances: 121
- Unique Behaviors: 21 (21 implementable)

Frequency Distribution:
- High Frequency (≥10 instances): 6 behaviors (27%)
- Medium Frequency (3-9 instances): 6 behaviors (27%)
- Low Frequency (<3 instances): 10 behaviors (45%)

Implementation:
- Built-in Gestures(used): 13 
- Built-in Gestures(unused): 5
- Custom (used): 6
- Custom (unused): 6
```