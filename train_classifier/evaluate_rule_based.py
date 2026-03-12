import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.dummy import DummyClassifier

# ── Gesture → label mapping ──────────────────────────────────────────────────

HEAD_GESTURE_MAP = {
    'head_nod_fast':   'nod',
    'head_sway':       'sway',
    'head_shake_fast': 'shake',
}

FACIAL_GESTURE_MAP = {
    'BigSmile':   'big_smile',
    'Surprise':   'big_smile',
    'Smile':      'smile',
    'BrowRaise':  'smile',
    'ExpressSad': 'frown',
    'Thoughtful': 'frown',
    'CloseEyes':  'neutral',
    'GazeAway':   'neutral',
    'OpenEyes':   'neutral',
}

# ── BEHAVIOR_MAP ──────────────────────────────────────────────────────────────

BEHAVIOR_MAP = {
    ('slow', 'low', 'dark'):       ['Thoughtful', 'CloseEyes', 'head_sway', 'ExpressSad'],
    ('slow', 'low', 'neutral'):    ['Thoughtful', 'CloseEyes', 'head_sway', 'Smile', 'GazeAway'],
    ('slow', 'low', 'bright'):     ['Thoughtful', 'head_sway', 'Smile', 'OpenEyes'],
    ('slow', 'medium', 'dark'):    ['head_sway', 'Smile', 'Thoughtful', 'CloseEyes'],
    ('slow', 'medium', 'neutral'): ['head_sway', 'Smile', 'GazeAway'],
    ('slow', 'medium', 'bright'):  ['head_sway', 'Smile', 'BrowRaise', 'OpenEyes', 'BigSmile'],
    ('slow', 'high', 'dark'):      ['head_nod_fast', 'ExpressSad', 'BrowRaise', 'Thoughtful'],
    ('slow', 'high', 'neutral'):   ['head_nod_fast', 'BrowRaise', 'Surprise', 'Smile'],
    ('slow', 'high', 'bright'):    ['head_nod_fast', 'Surprise', 'BrowRaise', 'BigSmile', 'OpenEyes'],
    ('moderate', 'low', 'dark'):   ['head_sway', 'Thoughtful', 'GazeAway', 'CloseEyes'],
    ('moderate', 'low', 'neutral'): ['head_sway', 'Smile', 'GazeAway', 'head_nod_fast'],
    ('moderate', 'low', 'bright'): ['head_sway', 'Smile', 'BrowRaise', 'OpenEyes'],
    ('moderate', 'medium', 'dark'):    ['head_sway', 'Smile', 'Thoughtful', 'ExpressSad'],
    ('moderate', 'medium', 'neutral'): ['head_sway', 'Smile', 'BrowRaise', 'GazeAway'],
    ('moderate', 'medium', 'bright'):  ['head_sway', 'Smile', 'BrowRaise', 'Surprise', 'BigSmile'],
    ('moderate', 'high', 'dark'):      ['head_nod_fast', 'head_sway', 'Thoughtful', 'BrowRaise'],
    ('moderate', 'high', 'neutral'):   ['head_nod_fast', 'head_sway', 'Smile', 'Surprise', 'BrowRaise'],
    ('moderate', 'high', 'bright'):    ['head_nod_fast', 'Smile', 'Surprise', 'BrowRaise', 'BigSmile'],
    ('fast', 'low', 'dark'):    ['head_shake_fast', 'head_sway', 'Thoughtful', 'CloseEyes'],
    ('fast', 'low', 'neutral'): ['head_shake_fast', 'head_sway', 'Smile', 'GazeAway'],
    ('fast', 'low', 'bright'):  ['head_shake_fast', 'Smile', 'BrowRaise', 'OpenEyes'],
    ('fast', 'medium', 'dark'):    ['head_shake_fast', 'Smile', 'Thoughtful'],
    ('fast', 'medium', 'neutral'): ['head_shake_fast', 'Smile', 'BrowRaise', 'GazeAway'],
    ('fast', 'medium', 'bright'):  ['head_shake_fast', 'Smile', 'BrowRaise', 'Surprise', 'BigSmile'],
    ('fast', 'high', 'dark'):    ['head_shake_fast', 'head_nod_fast', 'BrowRaise', 'ExpressSad'],
    ('fast', 'high', 'neutral'): ['head_shake_fast', 'head_nod_fast', 'Surprise', 'BrowRaise'],
    ('fast', 'high', 'bright'):  ['head_shake_fast', 'head_nod_fast', 'Surprise', 'BrowRaise', 'BigSmile'],
}

# Categorisation functions
def categorize_tempo(bpm):
    return 'slow' if bpm < 110 else 'moderate' if bpm < 130 else 'fast'

def categorize_energy(e):
    return 'low' if e < 0.10 else 'medium' if e < 0.20 else 'high'

def categorize_brightness(hz):
    return 'dark' if hz < 1800 else 'neutral' if hz < 2300 else 'bright'


#  Normalise raw CSV labels → ML classes
HEAD_LABEL_NORM = {
    # nod family
    'nod':          'nod',
    'head_nod':     'nod',
    'head nod':     'nod',
    'nod_fast':     'nod',
    # sway family
    'sway':         'sway',
    'head_sway':    'sway',
    'head sway':    'sway',
    # shake family
    'shake':        'shake',
    'head_shake':   'shake',
    'head shake':   'shake',
    'shake_fast':   'shake',
    # none family
    'none':         'none',
    'still':        'none',
    '':             'none',
    'look up':      'none',
    'look down':    'none',
}

FACIAL_LABEL_NORM = {
    # big_smile family
    'big smile':           'big_smile',
    'big_smile':           'big_smile',
    'surprise':            'big_smile',
    'oh':                  'big_smile',
    'oh face':             'big_smile',
    'angry, surprise':     'big_smile',
    'browraise, surprise': 'big_smile',
    # smile family
    'smile':               'smile',
    'smile, frown':        'smile',
    'browraise':           'smile',
    # frown family
    'frown':               'frown',
    'frown, smile':        'frown',
    'thoughtful':          'frown',
    'thoughtful face':     'frown',
    'oh, frown':           'frown',
    'surprise, frown':     'frown',
    'disgust':             'frown',
    'sad face':            'frown',
    'angry face':          'frown',
    # neutral family
    'neutral':             'neutral',
    'close eyes':          'neutral',
}


def normalize_head(label):
    label = str(label).strip().lower()
    if label in HEAD_LABEL_NORM:
        return HEAD_LABEL_NORM[label]
    for cls in ['nod', 'sway', 'shake']:
        if cls in label:
            return cls
    print(f"  [WARN] Unknown head label: '{label}' → mapped to 'none'")
    return 'none'


def normalize_facial(label):
    label = str(label).strip().lower()
    if label in FACIAL_LABEL_NORM:
        return FACIAL_LABEL_NORM[label]
    for cls in ['big_smile', 'big smile', 'smile', 'frown', 'neutral']:
        if cls in label:
            return cls.replace(' ', '_')
    print(f"  [WARN] Unknown facial label: '{label}' → mapped to 'neutral'")
    return 'neutral'


# Predict labels from gesture list
def predict_head(gestures):
    for g in gestures:
        if g in HEAD_GESTURE_MAP:
            return HEAD_GESTURE_MAP[g]
    return 'none'


def predict_facial(gestures):
    for g in gestures:
        if g in FACIAL_GESTURE_MAP:
            return FACIAL_GESTURE_MAP[g]
    return 'neutral'


def rule_based_predict(row):
    tempo_cat = categorize_tempo(row['tempo'])
    energy_cat = categorize_energy(row['rms_mean'])
    brightness_cat = categorize_brightness(row['spectral_centroid_mean'])
    gestures = BEHAVIOR_MAP.get(
        (tempo_cat, energy_cat, brightness_cat),
        ['Smile', 'head_sway']
    )
    return predict_head(gestures), predict_facial(gestures)


# Load data
features = pd.read_csv('/Users/miaaa/Desktop/music robot/furhat_music_robot/features_v3.csv')
metadata = pd.read_csv('/Users/miaaa/Desktop/music robot/furhat_music_robot/annotations/metadata.csv')

features['head_movement'] = features['head_movement'].str.strip()
features['facial_expression'] = features['facial_expression'].str.strip()

# Normalise ground-truth labels to match ML 4-class scheme
features['head_movement'] = features['head_movement'].apply(normalize_head)
features['facial_expression'] = features['facial_expression'].apply(normalize_facial)

print("Normalised label distributions")
print("Head movement:")
print(features['head_movement'].value_counts().to_string(), "\n")
print("Facial expression:")
print(features['facial_expression'].value_counts().to_string(), "\n")

song_split = {row['title']: row['split'] for _, row in metadata.iterrows()}
features['split'] = features['song_name'].map(song_split)
test_data  = features[features['split'] == 'Test'].copy()
train_data = features[features['split'] == 'Train'].copy()

print(f"Test segments: {len(test_data)}\n")


#  Feature distributions
print("Feature distributions on TRAIN set:")
for col, name in [('tempo', 'Tempo'), ('rms_mean', 'Energy'), ('spectral_centroid_mean', 'Brightness')]:
    p33, p66 = train_data[col].quantile(0.33), train_data[col].quantile(0.66)
    print(f"  {name:12s}  min={train_data[col].min():.1f}  p33={p33:.1f}  p66={p66:.1f}  max={train_data[col].max():.1f}")
print()

# Run rule-based predictions
preds = test_data.apply(rule_based_predict, axis=1)
test_data['pred_head'] = [p[0] for p in preds]
test_data['pred_facial'] = [p[1] for p in preds]

#  Evaluate
def evaluate(y_true, y_pred, task_name):
    acc = accuracy_score(y_true, y_pred)
    print(f"  {task_name}")
    print(f"  Accuracy: {acc:.4f} ({acc*100:.1f}%)\n")
    print(classification_report(y_true, y_pred, zero_division=0))

    dummy = DummyClassifier(strategy='most_frequent')
    dummy.fit(y_true, y_true)
    dummy_acc = accuracy_score(y_true, dummy.predict(y_true))
    print(f"  Dummy baseline (most frequent): {dummy_acc:.4f} ({dummy_acc*100:.1f}%)")
    print(f"  Rule-based vs dummy:            {'+' if acc > dummy_acc else ''}{(acc - dummy_acc)*100:.1f}%\n")
    return acc


head_acc = evaluate(test_data['head_movement'],    test_data['pred_head'],   'HEAD MOVEMENT')
facial_acc = evaluate(test_data['facial_expression'], test_data['pred_facial'], 'FACIAL EXPRESSION')

#  Summary
print("  COMPARISON SUMMARY")
print(f"  {'System':<30} {'Head':>8} {'Facial':>10}")
print(f"  {'-'*48}")
print(f"  {'Rule-based (normalised)':<30} {head_acc*100:>7.1f}% {facial_acc*100:>9.1f}%")
print(f"  {'RF (all_111 / mfcc+chroma)':<30} {'53.8%':>8} {'34.6%':>10}")
print(f"  {'Dummy baseline':<30} {'~31%':>8} {'~20%':>10}")
