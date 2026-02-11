import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

print("Training Intensity Classifier (SVM with Merged Classes) \n")

# Load and prepare data
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

# Print merge mapping
print("\nClass merge mapping:")
print("  Original classes 0,1 -> New class 0 (low intensity)")
print("  Original classes 2,3 -> New class 1 (medium intensity)")
print("  Original classes 4,5 -> New class 2 (high intensity)")

# Print new class distribution
unique, counts = np.unique(train_data_copy['intensity_merged'], return_counts=True)
print("\nNew class distribution (training set):")
for label, count in zip(unique, counts):
    print(f"  Class {label}: {count} samples ({count/len(train_data_copy)*100:.1f}%)")

X_train = train_data_copy[['tempo', 'energy', 'brightness']].values
X_val = val_data_copy[['tempo', 'energy', 'brightness']].values
X_test = test_data_copy[['tempo', 'energy', 'brightness']].values

y_train = train_data_copy['intensity_merged'].values
y_val = val_data_copy['intensity_merged'].values
y_test = test_data_copy['intensity_merged'].values

print(f"\nX_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")

# Step 5: Normalize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Step 6: Train SVM with class weights
svm = SVC(kernel='rbf', C=10, gamma='scale',
          class_weight='balanced')
svm.fit(X_train, y_train)
print("\nTraining complete\n")

# Step 7-8: Evaluate
print("Validation Results")
y_val_pred = svm.predict(X_val)
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

print("\n")
print("Test Results")
y_test_pred = svm.predict(X_test)
test_acc = accuracy_score(y_test, y_test_pred)
print(f"Accuracy: {test_acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred,
                          target_names=['Low', 'Medium', 'High']))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))

print("\n")
print("Final Results:")
print(f"  Validation Accuracy: {val_acc:.4f}")
print(f"  Test Accuracy:       {test_acc:.4f}")
