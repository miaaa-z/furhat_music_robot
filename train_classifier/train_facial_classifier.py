import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

print("Training Facial Expression Classifier (Random Forest with Merged Categories) \n")

# Step 1-2: Load and prepare data
features = pd.read_csv('/Users/miaaa/Desktop/music robot/furhat_music_robot/features.csv')
metadata = pd.read_csv('/Users/miaaa/Desktop/music robot/furhat_music_robot/annotations/metadata.csv')

# Clean whitespace in facial_expression column
features['facial_expression'] = features['facial_expression'].str.strip()

print(f"Total segments: {len(features)}")
print(f"Total songs: {len(metadata)}")

# Create song name to split mapping
song_split_dict = {}
for index, row in metadata.iterrows():
    song_split_dict[row['title']] = row['split']

features['split'] = features['song_name'].map(song_split_dict)

# Step 3: Split into Train, Validation, Test
train_data = features[features['split'] == 'Train'].copy()
val_data = features[features['split'] == 'Validation'].copy()
test_data = features[features['split'] == 'Test'].copy()

print(f"Train:      {len(train_data)} segments")
print(f"Validation: {len(val_data)} segments")
print(f"Test:       {len(test_data)} segments")


# Merge facial expression categories
def merge_facial_expression(expression):
    """
    Merge 21 categories into 9 based on semantic similarity
    """
    expression = expression.strip().lower()

    # Group 1: angry series
    if 'angry' in expression:
        return 'angry'

    # Group 2: big smile series
    elif expression in ['big smile', 'big smile, close eyes', 'browraise, smile']:
        return 'big_smile'

    # Group 3: surprise series (browraise, browraise+surprise, surprise, oh)
    elif expression in ['browraise', 'browraise, surprise', 'surprise', 'oh']:
        return 'surprise'

    # Group 4: smile series (smile, close eyes, close eyes+smile)
    elif expression in ['smile', 'close eyes', 'close eyes, smile']:
        return 'smile'

    # Group 5: disgust
    elif expression == 'disgust':
        return 'disgust'

    # Group 6: frown series (frown, frown+confused, frown+thoughtful, smile+frown)
    elif 'frown' in expression:
        return 'frown'

    # Group 7: neutral
    elif expression == 'neutral':
        return 'neutral'

    # Group 8: sad
    elif expression == 'sad':
        return 'sad'

    # Group 9: thoughtful
    elif expression == 'thoughtful':
        return 'thoughtful'

    else:
        return expression


# Apply merging
train_data['facial_merged'] = train_data['facial_expression'].apply(merge_facial_expression)
val_data['facial_merged'] = val_data['facial_expression'].apply(merge_facial_expression)
test_data['facial_merged'] = test_data['facial_expression'].apply(merge_facial_expression)


# Print merged distribution
print("\nMerged Facial Expression Distribution (Training set):")
unique, counts = np.unique(train_data['facial_merged'], return_counts=True)
total = len(train_data)
for label, count in zip(unique, counts):
    print(f"  {label:15s}: {count:3d} ({count / total * 100:5.1f}%)")

# Step 4: Prepare features (X) and labels (y)
X_train = train_data[['tempo', 'energy', 'brightness']].values
X_val = val_data[['tempo', 'energy', 'brightness']].values
X_test = test_data[['tempo', 'energy', 'brightness']].values

y_train = train_data['facial_merged'].values
y_val = val_data['facial_merged'].values
y_test = test_data['facial_merged'].values

# Step 5: Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Step 6: Train Random Forest
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# Print feature importance
print("Feature Importance:")
feature_names = ['tempo', 'energy', 'brightness']
importances = rf.feature_importances_
for name, importance in zip(feature_names, importances):
    print(f"  {name}: {importance:.4f}")

# Step 7: Evaluate on validation set
print("Validation Results")
y_val_pred = rf.predict(X_val)
val_acc = accuracy_score(y_val, y_val_pred)
print(f"Accuracy: {val_acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_val, y_val_pred))
print("\nConfusion Matrix:")
cm_val = confusion_matrix(y_val, y_val_pred)
print(cm_val)

# Step 8: Evaluate on test set
print("Test Results")
y_test_pred = rf.predict(X_test)
test_acc = accuracy_score(y_test, y_test_pred)
print(f"Accuracy: {test_acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))
print("\nConfusion Matrix:")
cm_test = confusion_matrix(y_test, y_test_pred)
print(cm_test)


# Summary
print("Final Results:")
print(f"  Validation Accuracy: {val_acc:.4f}")
print(f"  Test Accuracy:       {test_acc:.4f}")
