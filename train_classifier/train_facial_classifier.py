import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

print("Training Facial Expression Classifier (Random Forest with Merged Categories)\n")

# 1: Load data
features = pd.read_csv('/Users/miaaa/Desktop/music robot/furhat_music_robot/features_v2.csv')
metadata = pd.read_csv('/Users/miaaa/Desktop/music robot/furhat_music_robot/annotations/metadata.csv')

# Clean whitespace
features['facial_expression'] = features['facial_expression'].str.strip()

print(f"Total segments: {len(features)}")
print(f"Total songs:    {len(metadata)}")

# 2: Train/Val/Test split
song_split_dict = {row['title']: row['split'] for _, row in metadata.iterrows()}
features['split'] = features['song_name'].map(song_split_dict)

train_data = features[features['split'] == 'Train'].copy()
val_data = features[features['split'] == 'Validation'].copy()
test_data = features[features['split'] == 'Test'].copy()

print(f"Train:      {len(train_data)} segments")
print(f"Validation: {len(val_data)} segments")
print(f"Test:       {len(test_data)} segments")


# 3: merge 27 classes into 9 groups
def merge_facial_expression(expression):
    expression = expression.strip().lower()

    # Group 1: angry — any expression containing anger
    if 'angry' in expression:
        return 'angry'

    # Group 2: big smile — strong positive expressions
    elif expression in ['big smile', 'big smile, close eyes', 'browraise, smile']:
        return 'big_smile'

    # Group 3: surprise — raised brows, surprise, open mouth
    elif expression in ['browraise', 'browraise, surprise', 'surprise', 'oh', 'oh face']:
        return 'surprise'

    # Group 4: smile — mild positive expressions
    elif expression in ['smile', 'close eyes', 'close eyes, smile',
                        'narrow eyes, smile', 'smile, close eyes']:
        return 'smile'

    # Group 5: disgust
    elif 'disgust' in expression:
        return 'disgust'

    # Group 6: frown — any expression containing frown or sadness
    elif 'frown' in expression or 'sad' in expression:
        return 'frown'

    # Group 7: neutral
    elif expression == 'neutral':
        return 'neutral'

    # Group 8: thoughtful
    elif 'thoughtful' in expression or 'confused' in expression:
        return 'thoughtful'

    # Group 9: smile_frown — mixed/ambiguous expressions
    elif 'smile' in expression and 'frown' in expression:
        return 'smile_frown'

    # else: keep original if unmatched
    else:
        print(f"  [WARNING] Unmatched expression: '{expression}' → kept as-is")
        return expression


train_data['facial_merged'] = train_data['facial_expression'].apply(merge_facial_expression)
val_data['facial_merged'] = val_data['facial_expression'].apply(merge_facial_expression)
test_data['facial_merged'] = test_data['facial_expression'].apply(merge_facial_expression)

# Print merged distribution
print("\nMerged Facial Expression Distribution (Training set):")
unique, counts = np.unique(train_data['facial_merged'], return_counts=True)
total = len(train_data)
for label, count in zip(unique, counts):
    print(f"  {label:15s}: {count:3d} ({count/total*100:5.1f}%)")

# 4: Feature columns
label_cols = ['song_name', 'start_time', 'end_time', 'duration',
              'head_movement', 'facial_expression', 'intensity',
              'facial_merged', 'split']
feature_cols = [c for c in train_data.columns if c not in label_cols]

X_train = train_data[feature_cols].values
X_val = val_data[feature_cols].values
X_test = test_data[feature_cols].values

y_train = train_data['facial_merged'].values
y_val = val_data['facial_merged'].values
y_test = test_data['facial_merged'].values

print(f"\nFeature vector size: {len(feature_cols)} dimensions")

#  5: Normalise
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# 6: Dummy classifier
dummy = DummyClassifier(strategy='most_frequent', random_state=42)
dummy.fit(X_train, y_train)
dummy_val_acc = accuracy_score(y_val,  dummy.predict(X_val))
dummy_test_acc = accuracy_score(y_test, dummy.predict(X_test))
print(f"\nDummy Classifier (most_frequent) — Val: {dummy_val_acc:.4f}, Test: {dummy_test_acc:.4f}")

# 7: Train Random Forest
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    class_weight='balanced',   # compensate for remaining class imbalance
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
print("Training complete\n")

# Print top 10 most important features
print("Top 10 Feature Importances:")
importances = rf.feature_importances_
sorted_idx = np.argsort(importances)[::-1]
for i in range(min(10, len(feature_cols))):
    idx = sorted_idx[i]
    print(f"  {feature_cols[idx]:35s}: {importances[idx]:.4f}")

#  8: Evaluate
print("\nValidation Results")
y_val_pred = rf.predict(X_val)
val_acc = accuracy_score(y_val, y_val_pred)
print(f"Accuracy: {val_acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_val, y_val_pred, zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(y_val, y_val_pred))

print("\nTest Results")
y_test_pred = rf.predict(X_test)
test_acc = accuracy_score(y_test, y_test_pred)
print(f"Accuracy: {test_acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred, zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))

print("\nFinal Results:")
print(f"  Dummy Baseline  — Val: {dummy_val_acc:.4f}, Test: {dummy_test_acc:.4f}")
print(f"  Random Forest   — Val: {val_acc:.4f},  Test: {test_acc:.4f}")