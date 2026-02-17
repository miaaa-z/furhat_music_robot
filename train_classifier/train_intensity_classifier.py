import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

print("Training Intensity Classifier (Random Forest with Merged Classes)\n")

# Step 1-4: Load and prepare data
features = pd.read_csv('/Users/miaaa/Desktop/music robot/furhat_music_robot/features.csv')
metadata = pd.read_csv('/Users/miaaa/Desktop/music robot/furhat_music_robot/annotations/metadata.csv')

print(f"Total segments: {len(features)}")
print(f"Total songs: {len(metadata)}")

song_split_dict = {}
for index, row in metadata.iterrows():
    song_split_dict[row['title']] = row['split']

features['split'] = features['song_name'].map(song_split_dict)

train_data = features[features['split'] == 'Train']
val_data = features[features['split'] == 'Validation']
test_data = features[features['split'] == 'Test']

print(f"Train:      {len(train_data)} segments")
print(f"Validation: {len(val_data)} segments")
print(f"Test:       {len(test_data)} segments")


# Merge intensity classes function
def merge_intensity(intensity):
    """
    Merge 6 classes into 3:
    0-1 -> 0 (low intensity)
    2-3 -> 1 (medium intensity)
    4-5 -> 2 (high intensity)
    """
    if intensity <= 1:
        return 0
    elif intensity <= 3:
        return 1
    else:
        return 2


# Apply merging
train_data_copy = train_data.copy()
val_data_copy = val_data.copy()
test_data_copy = test_data.copy()

train_data_copy['intensity_merged'] = train_data_copy['intensity'].apply(merge_intensity)
val_data_copy['intensity_merged'] = val_data_copy['intensity'].apply(merge_intensity)
test_data_copy['intensity_merged'] = test_data_copy['intensity'].apply(merge_intensity)


# Print new class distribution
unique, counts = np.unique(train_data_copy['intensity_merged'], return_counts=True)
print("\nclass distribution (training set):")
for label, count in zip(unique, counts):
    print(f"  Class {label}: {count} samples ({count/len(train_data_copy)*100:.1f}%)")

X_train = train_data_copy[['tempo', 'energy', 'brightness']].values
X_val = val_data_copy[['tempo', 'energy', 'brightness']].values
X_test = test_data_copy[['tempo', 'energy', 'brightness']].values

y_train = train_data_copy['intensity_merged'].values
y_val = val_data_copy['intensity_merged'].values
y_test = test_data_copy['intensity_merged'].values


# Step 5: Normalize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Baseline: DummyClassifier
dummy = DummyClassifier(strategy='most_frequent', random_state=42)
dummy.fit(X_train, y_train)
dummy_val_acc = accuracy_score(y_val, dummy.predict(X_val))
dummy_test_acc = accuracy_score(y_test, dummy.predict(X_test))
print(f"Dummy Classifier (most_frequent) - Val Acc: {dummy_val_acc:.4f}, Test Acc: {dummy_test_acc:.4f}")


# Step 6: Train Random Forest
rf = RandomForestClassifier(
    n_estimators=200,          # Number of trees in the forest
    max_depth=None,              # Maximum depth of each tree
    min_samples_split=2,       # Minimum samples required to split a node
    min_samples_leaf=1,        # Minimum samples required at leaf node
    max_features='sqrt',
    class_weight='balanced',   # Handle class imbalance
    random_state=42,           # For reproducibility
    n_jobs=-1                  # Use all CPU cores
)
rf.fit(X_train, y_train)
print("Training complete\n")

# Print feature importance
print("Feature Importance:")
feature_names = ['tempo', 'energy', 'brightness']
importances = rf.feature_importances_
for name, importance in zip(feature_names, importances):
    print(f"  {name}: {importance:.4f}")

# Step 7-8: Evaluate
print("Validation Results")
y_val_pred = rf.predict(X_val)
val_acc = accuracy_score(y_val, y_val_pred)
print(f"Accuracy: {val_acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_val, y_val_pred,
                          target_names=['Low', 'Medium', 'High']))
print("\nConfusion Matrix:")
cm = confusion_matrix(y_val, y_val_pred)
print(cm)
print("\nConfusion Matrix Explanation:")
print("       Predicted: Low  Med  High")
print(f"Actual Low:  {cm[0]}")
print(f"Actual Med:  {cm[1]}")
print(f"Actual High: {cm[2]}")


print("Test Results")
y_test_pred = rf.predict(X_test)
test_acc = accuracy_score(y_test, y_test_pred)
print(f"Accuracy: {test_acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred,
                          target_names=['Low', 'Medium', 'High']))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))


print("Final Results:")
print(f"  Validation Accuracy: {val_acc:.4f}")
print(f"  Test Accuracy:       {test_acc:.4f}")
print("\n")

# Get probability predictions with song info
print("Sample Predictions (first 5 validation samples):")
val_probs = rf.predict_proba(X_val)[:5]
val_sample_info = val_data_copy.head(5)

for i, (true_label, pred_label, probs) in enumerate(zip(y_val[:5], y_val_pred[:5], val_probs)):
    song_name = val_sample_info.iloc[i]['song_name']
    start_time = val_sample_info.iloc[i]['start_time']
    end_time = val_sample_info.iloc[i]['end_time']

    print(f"\nSample {i + 1}: {song_name} ({start_time:.1f}s - {end_time:.1f}s)")
    print(f"  True: {true_label} ({['Low', 'Medium', 'High'][true_label]})")
    print(f"  Predicted: {pred_label} ({['Low', 'Medium', 'High'][pred_label]})")
    print(f"  Probabilities - Low: {probs[0]:.2f}, Medium: {probs[1]:.2f}, High: {probs[2]:.2f}")